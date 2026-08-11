from abc import ABC, abstractmethod
from collections import deque
import numpy as np
from scipy.optimize import minimize, line_search


class Optimizer(ABC):
    """Base class for optimization algorithms."""

    @abstractmethod
    def minimize(self, scalar_loss, x0, callback=None):
        """Minimize scalar_loss starting from x0.

        Args:
            scalar_loss: function(x) -> float
            x0: initial point
            callback: optional function(step, x) called at each iteration

        Returns:
            optimized x
        """
        pass


class GradientDescent(Optimizer):
    """Simple gradient descent with fixed step size."""

    def __init__(self, lr: float = 0.2, nb_steps: int = 100):
        self.lr = lr
        self.nb_steps = nb_steps

    def minimize(self, scalar_loss, x0, callback=None):
        from loom import driver

        x = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        # jit ONCE and reuse across steps: without it each step re-traces + re-lowers the whole
        # graph (unbounded RSS growth, ~4x slower). The C++ `driver.call` survives the trace as an
        # FFI primitive; only the surrounding tensor algebra is fused by XLA. See driver.jit.
        grad = driver.jit(driver.grad(scalar_loss))

        for step in range(self.nb_steps):
            x = x - self.lr * grad(x)
            if callback is not None:
                callback(step, x)

        return x


def _two_phase_lbfgsb(fun, jac, x0_flat, shape_orig, *, max_iter: int, ftol: float,
                       min_iter: int, disp_tol: float | None, callback):
    """Coeur commun à `LBFGS` et `FusedLBFGS` : L-BFGS-B scipy en deux appels -- un premier à
    `maxiter=min_iter` avec `ftol=gtol=0` (aucun arrêt anticipé possible, voir `LBFGS.min_iter`),
    puis un second qui reprend `ftol` pour les pas restants, avec arrêt anticipé sur `disp_tol`
    (déplacement max d'un point) via `StopIteration` depuis le callback.

    `fun`/`jac` sont transmis TELS QUELS à `scipy.optimize.minimize` -- `jac` un callable
    (gradient séparé, `LBFGS`) ou `True` (`fun` renvoie `(valeur, gradient)`, `FusedLBFGS`) :
    seule cette paire diffère entre les deux optimiseurs, le reste (double appel, callbacks,
    critères d'arrêt) est repris ici pour ne pas diverger entre les deux implémentations.
    """
    step_counter = [0]  # mutable counter for callback, shared across the two minimize() calls
    prev_x = [x0_flat.copy()]

    def make_callback(check_disp: bool):
        def scipy_callback(x_flat):
            step = step_counter[0]
            if callback is not None:
                callback(step, x_flat.reshape(shape_orig))
            stop = False
            if check_disp and disp_tol is not None:
                disp = float(np.max(np.abs(x_flat - prev_x[0])))
                stop = disp < disp_tol
            prev_x[0] = x_flat.copy()
            step_counter[0] += 1
            if stop:
                raise StopIteration
        return scipy_callback

    x_flat = x0_flat
    if min_iter > 0:
        result = minimize(
            fun, x_flat, jac=jac, method='L-BFGS-B',
            callback=make_callback(check_disp=False),
            options={'maxiter': min_iter, 'ftol': 0.0, 'gtol': 0.0},
        )
        x_flat = result.x

    remaining = max(0, max_iter - step_counter[0])
    if remaining > 0:
        result = minimize(
            fun, x_flat, jac=jac, method='L-BFGS-B',
            callback=make_callback(check_disp=True),
            options={'maxiter': remaining, 'ftol': ftol},
        )
        x_flat = result.x

    return x_flat


class LBFGS(Optimizer):
    """L-BFGS optimization via scipy.

    `min_iter` : nombre de pas IMPOSÉ avant que scipy ait le droit de conclure à convergence.
    Sans ça, L-BFGS-B peut s'arrêter dès le pas 0/1 dès que `ftol`/`gtol` (défaut scipy) sont
    satisfaits -- observé sur le modèle DISQUES, où la perte est quasi stationnaire le long de
    directions pourtant loin d'être optimales (un disque peut glisser sans changer le résidu
    tant qu'il ne chevauche personne). Mis en oeuvre en DEUX appels `scipy.optimize.minimize`
    (voir `_two_phase_lbfgsb`) : un premier à `maxiter=min_iter` avec `ftol=gtol=0` (aucun arrêt
    anticipé possible), puis un second qui reprend le `ftol` demandé pour les pas restants.

    `disp_tol` : pendant ce second appel, arrêt anticipé (levée de `StopIteration` depuis le
    callback -- supporté nativement par `scipy.optimize.minimize` depuis la 1.11) dès que le
    déplacement max d'un point entre deux pas passe sous ce seuil (mêmes unités que les points).
    Combine bien avec `min_iter` : "au moins `min_iter` pas, puis tant que ça bouge encore".
    `None` (défaut) désactive ce critère -- seul `ftol` scipy arrête alors la phase 2.
    """

    def __init__(self, max_iter: int = 100, ftol: float = 1e-8,
                 min_iter: int = 0, disp_tol: float | None = None):
        self.max_iter = max_iter
        self.ftol = ftol
        self.min_iter = min_iter
        self.disp_tol = disp_tol

    def minimize(self, scalar_loss, x0, callback=None):
        from loom import driver

        x0 = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        shape_orig = x0.shape

        # Flatten to 1D for scipy
        x0_flat = x0.reshape(-1)
        # jit ONCE (compiled on the flat 1D signature scipy calls with) and reuse across iterations.
        loss_j = driver.jit(lambda xf: scalar_loss(xf.reshape(shape_orig)))
        grad_j = driver.jit(lambda xf: driver.grad(scalar_loss)(xf.reshape(shape_orig)).reshape(-1))

        x_flat = _two_phase_lbfgsb(
            loss_j, grad_j, x0_flat, shape_orig,
            max_iter=self.max_iter, ftol=self.ftol, min_iter=self.min_iter, disp_tol=self.disp_tol,
            callback=callback,
        )
        return x_flat.reshape(shape_orig)


class FusedLBFGS(Optimizer):
    """L-BFGS via scipy, comme `LBFGS`, mais pour un coût dont le gradient vient d'une évaluation
    FUSIONNÉE externe (`value_and_grad(points) -> (cost: float, grad: ndarray)`, ex.
    `dirac_sycl.diracs_cost_grad`) plutôt que de l'autodiff Jax -- tracer coût et gradient
    séparément (ce que `LBFGS` fait via `driver.jit`/`driver.grad`) jetterait cette fusion (le
    kernel SYCL calcule les deux en un seul passage, voir `dirac_sycl.py`).

    Même politique `min_iter`/`disp_tol` que `LBFGS` (voir sa docstring, machinerie commune dans
    `_two_phase_lbfgsb`) -- seule diffère la façon dont scipy obtient `(valeur, gradient)` :
    `jac=True`, `value_and_grad` répondant déjà au contrat scipy pour cette option.

    `minimize`'s `scalar_loss` n'est PAS ici une fonction scalaire jax-traçable comme pour les
    autres optimiseurs, mais directement `value_and_grad` -- voir `Reconstruction.run`, qui choisit
    l'un ou l'autre appel selon le type de l'optimiseur.
    """

    def __init__(self, max_iter: int = 100, ftol: float = 1e-8,
                 min_iter: int = 0, disp_tol: float | None = None):
        self.max_iter = max_iter
        self.ftol = ftol
        self.min_iter = min_iter
        self.disp_tol = disp_tol

    def minimize(self, value_and_grad, x0, callback=None):
        x0 = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        shape_orig = x0.shape
        x0_flat = x0.reshape(-1)

        def fun(x_flat):
            cost, grad = value_and_grad(x_flat.reshape(shape_orig))
            return float(cost), np.asarray(grad, dtype=np.float64).reshape(-1)

        x_flat = _two_phase_lbfgsb(
            fun, True, x0_flat, shape_orig,
            max_iter=self.max_iter, ftol=self.ftol, min_iter=self.min_iter, disp_tol=self.disp_tol,
            callback=callback,
        )
        return x_flat.reshape(shape_orig)


class GradientDescentLineSearch(Optimizer):
    """Gradient descent with backtracking line search."""

    def __init__(self, lr: float = 1.0, nb_steps: int = 100, c1: float = 1e-4, rho: float = 0.5):
        self.lr = lr
        self.nb_steps = nb_steps
        self.c1 = c1
        self.rho = rho

    def minimize(self, scalar_loss, x0, callback=None):
        from loom import driver

        x = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        grad = driver.jit(driver.grad(scalar_loss))
        loss_j = driver.jit(scalar_loss)

        for step in range(self.nb_steps):
            g = grad(x)
            loss_curr = loss_j(x)

            # Backtracking line search: find step size that decreases loss
            alpha = self.lr
            max_backtracks = 20
            for _ in range(max_backtracks):
                x_new = x - alpha * g
                loss_new = loss_j(x_new)
                if loss_new <= loss_curr - self.c1 * alpha * np.dot(g.reshape(-1), g.reshape(-1)):
                    break
                alpha *= self.rho
            else:
                x_new = x - alpha * g

            x = x_new
            if callback is not None:
                callback(step, x)

        return x


class Adam(Optimizer):
    """Adam optimizer with gradient clipping and adaptive learning rate."""

    def __init__(self, lr: float = 0.1, nb_steps: int = 200, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, grad_clip: float = None):
        self.lr = lr
        self.nb_steps = nb_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_clip = grad_clip

    def minimize(self, scalar_loss, x0, callback=None):
        from loom import driver

        x = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        grad = driver.jit(driver.grad(scalar_loss))

        m = np.zeros_like(x)  # first moment
        v = np.zeros_like(x)  # second moment

        for step in range(self.nb_steps):
            g = grad(x)

            # Optional gradient clipping
            if self.grad_clip is not None:
                grad_norm = np.linalg.norm(g)
                if grad_norm > self.grad_clip:
                    g = g * (self.grad_clip / grad_norm)

            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * g
            # Update biased second raw moment estimate
            v = self.beta2 * v + (1 - self.beta2) * (g ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - self.beta1 ** (step + 1))
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - self.beta2 ** (step + 1))

            # Update parameters
            x = x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if callback is not None:
                callback(step, x)

        return x


class SubspaceNewtonLBFGS(FusedLBFGS):
    """Prototype : au lieu de la mémoire de courbure interne à scipy (`LBFGS`), garde une pile
    explicite des `max_dirs - 1` derniers pas RÉELLEMENT pris (`x_k - x_{k-1}`, normalisés) plus
    le gradient courant, comme base d'un sous-espace, et résout à CHAQUE pas un Newton EXACT dans
    ce sous-espace (`m` <= `max_dirs` inconnues `a`, `m x m`) au lieu d'une simple direction de
    descente + line search 1D -- `dirac_sycl.subspace_hessian` fournit `(H, b)` en un second
    `driver.call` fusionné, formule fermée (voir sa docstring pour la dérivation).

    Empile les pas RÉELLEMENT pris, PAS les gradients successifs (première version testée) : près
    d'un optimum, des gradients successifs deviennent quasi-colinéaires (le zigzag classique de la
    descente de gradient), donc une base de gradients bruts finit par ne plus porter d'info
    nouvelle -- `H` devient mal conditionnée et le damping Tikhonov écrase le peu de signal utile
    avec le bruit. Les pas RÉELLEMENT pris jouent le rôle des `s_k` de L-BFGS (une direction
    différente à chaque itération QUAND ça converge bien) sans reconstituer tout son appareil de
    paires `(s_k, y_k)`. Le gradient courant reste TOUJOURS dans la base (frais, jamais historique)
    pour garantir au moins une direction de descente valide même quand l'historique est vide (1er
    pas) ou dégénéré.

    N'utilise PAS `_two_phase_lbfgsb`/scipy (pas de mémoire de courbure L-BFGS à faire tourner ici,
    la Hessienne restreinte est recalculée à chaque pas) : boucle Python maison, avec un
    backtracking Armijo comme garde-fou -- le pas Newton du sous-espace n'est optimal que pour le
    modèle quadratique LOCAL (assignation figée), pas pour la vraie perte (non convexe, non lisse
    aux changements d'assignation).
    """

    def __init__(self, sinogram, max_dirs: int = 5, max_iter: int = 60, ftol: float = 1e-10,
                 tikhonov: float = 1e-6, c1: float = 1e-4, rho: float = 0.5, max_backtracks: int = 20,
                 diag_callback=None):
        self.sinogram = sinogram
        self.max_dirs = max_dirs
        self.max_iter = max_iter
        self.ftol = ftol
        self.tikhonov = tikhonov
        self.c1 = c1
        self.rho = rho
        self.max_backtracks = max_backtracks
        #: optionnel, `diag_callback(step, x, cost, g, directions, m, H, b, a, step_dir,
        #: directional_deriv)` appelé APRÈS résolution du pas Newton du sous-espace mais AVANT le
        #: backtracking -- pour inspecter le modèle quadratique local (ex. le comparer à la vraie
        #: perte le long de `step_dir`, voir `experiments/lung_alveoli.run_subspace_alpha_profile`)
        #: sans dupliquer la logique de `minimize`. `directions` n'est passé QUE sur `[:m]` (les
        #: colonnes actives, pas le padding `MAX_DIRS`).
        self.diag_callback = diag_callback

    def minimize(self, value_and_grad, x0, callback=None):
        from . import dirac_sycl

        MAX_DIRS = dirac_sycl.MAX_DIRS
        if self.max_dirs > MAX_DIRS:
            raise ValueError(f"max_dirs={self.max_dirs} dépasse dirac_sycl.MAX_DIRS={MAX_DIRS}")

        x = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        shape = x.shape
        # pas RÉELLEMENT pris (normalisés), les plus anciens en tête -- le gradient courant (frais,
        # jamais historique) occupe le slot restant, voir la docstring de la classe.
        step_history = deque(maxlen=max(self.max_dirs - 1, 0))

        # pas de `callback(-1, x)` ici : `Reconstruction.run` l'a déjà appelé pour l'état initial
        # avant `optimizer.minimize` (voir sa docstring) -- même convention que `LBFGS`/
        # `GradientDescentLineSearch`, dont le premier callback est `step=0`.
        cost, g = value_and_grad(x)

        for step in range(self.max_iter):
            g_norm = np.linalg.norm(g)
            basis = ([g / g_norm] if g_norm > 0 else []) + list(step_history)

            m = len(basis)
            directions = np.zeros((MAX_DIRS,) + shape, dtype=x.dtype)
            for i, d in enumerate(basis):
                directions[i] = d

            H5, b5 = dirac_sycl.subspace_hessian(x, directions, self.sinogram)
            H = 0.5 * (H5[:m, :m] + H5[:m, :m].T)
            b = b5[:m]

            damping = self.tikhonov * (np.trace(H) / m if m > 0 else 1.0)
            try:
                a = -np.linalg.solve(H + damping * np.eye(m), b)
            except np.linalg.LinAlgError:
                a = -np.linalg.lstsq(H, b, rcond=None)[0]
            step_dir = sum(a[i] * directions[i] for i in range(m))

            # backtracking Armijo le long du pas Newton du sous-espace -- garde-fou : le modèle
            # quadratique local n'est exact qu'à assignation figée, pas globalement.
            directional_deriv = float(np.sum(g * step_dir))

            if self.diag_callback is not None:
                self.diag_callback(step, x, cost, g, directions[:m], m, H, b, a, step_dir,
                                    directional_deriv)

            t = 1.0
            x_new, cost_new = None, None
            for _ in range(self.max_backtracks):
                x_try = x + t * step_dir
                cost_try, g_try = value_and_grad(x_try)
                if cost_try <= cost + self.c1 * t * directional_deriv:
                    x_new, cost_new, g_new = x_try, cost_try, g_try
                    break
                t *= self.rho
            else:
                # secours : un pas de descente de gradient normalisé, garanti descendant pour t assez
                # petit -- ne devrait quasiment jamais se déclencher (H est PSD par construction).
                t = 1.0
                d = g / g_norm if g_norm > 0 else g
                for _ in range(self.max_backtracks):
                    x_try = x - t * d
                    cost_try, g_try = value_and_grad(x_try)
                    if cost_try <= cost - self.c1 * t * g_norm:
                        x_new, cost_new, g_new = x_try, cost_try, g_try
                        break
                    t *= self.rho
                else:
                    x_new, cost_new, g_new = x_try, cost_try, g_try

            # empile le pas RÉELLEMENT pris (pas le pas candidat `step_dir` -- le backtracking a pu
            # le raccourcir, voire basculer sur le secours gradient) pour la base de la prochaine
            # itération, voir la docstring de la classe.
            delta = x_new - x
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 0:
                step_history.append(delta / delta_norm)

            if callback is not None:
                callback(step, x_new)
            if abs(cost - cost_new) < self.ftol:
                x, cost = x_new, cost_new
                break
            x, cost, g = x_new, cost_new, g_new

        return x
