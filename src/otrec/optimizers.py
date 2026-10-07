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
        import loom

        x = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        # jit ONCE and reuse across steps: without it each step re-traces + re-lowers the whole
        # graph (unbounded RSS growth, ~4x slower). The C++ `loom.ffi_call` survives the trace as an
        # FFI primitive; only the surrounding tensor algebra is fused by XLA. See loom.jit.
        grad = loom.jit(loom.grad(scalar_loss))

        for step in range(self.nb_steps):
            x = x - self.lr * np.asarray(grad(x))
            if callback is not None:
                callback(step, x)

        return x


def _two_phase_lbfgsb(fun, jac, x0_flat, shape_orig, *, max_iter: int, ftol: float,
                       min_iter: int, disp_tol: float | None, callback):
    """Common core of `LBFGS` and `FusedLBFGS`: scipy L-BFGS-B in two calls -- a first one with
    `maxiter=min_iter` and `ftol=gtol=0` (see `LBFGS.min_iter`: this makes an early stop RARE,
    it does not forbid it),
    then a second one that takes `ftol` back for the remaining steps, with early stop on `disp_tol`
    (max displacement of a point) via `StopIteration` from the callback.

    `fun`/`jac` are passed AS IS to `scipy.optimize.minimize` -- `jac` a callable
    (separate gradient, `LBFGS`) or `True` (`fun` returns `(value, gradient)`, `FusedLBFGS`):
    only this pair differs between the two optimizers, the rest (double call, callbacks,
    stopping criteria) is shared here so as not to diverge between the two implementations.
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

    `min_iter`: number of steps IMPOSED before scipy is allowed to conclude convergence.
    Without it, L-BFGS-B may stop as early as step 0/1 as soon as `ftol`/`gtol` (scipy defaults) are
    satisfied -- observed on the DISKS model, where the loss is nearly stationary along
    directions that are nonetheless far from optimal (a disk can slide without changing the residual
    as long as it overlaps nobody). Implemented as TWO `scipy.optimize.minimize` calls
    (see `_two_phase_lbfgsb`): a first one with `maxiter=min_iter` and `ftol=gtol=0`, then a
    second one that takes the requested `ftol` back for the remaining steps.

    WARNING -- `min_iter` is a "best effort", not a guarantee: `ftol=0` does not forbid an
    early stop, it conditions it on an EXACTLY zero reduction (the `factr` test of
    L-BFGS-B), which a perfectly flat direction reaches. Observed on DISKS: 9 steps for
    `min_iter=10`, scipy returning `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`, and a
    call restarted from that point makes no step at all (a truly stationary point for the
    line search). Do not assert `nb_steps >= min_iter`: nothing can promise it.

    `disp_tol`: during this second call, early stop (raising `StopIteration` from the
    callback -- natively supported by `scipy.optimize.minimize` since 1.11) as soon as the
    max displacement of a point between two steps falls below this threshold (same units as the points).
    Combines well with `min_iter`: "at least `min_iter` steps, then as long as it still moves".
    `None` (default) disables this criterion -- only scipy's `ftol` then stops phase 2.
    """

    def __init__(self, max_iter: int = 100, ftol: float = 1e-8,
                 min_iter: int = 0, disp_tol: float | None = None):
        self.max_iter = max_iter
        self.ftol = ftol
        self.min_iter = min_iter
        self.disp_tol = disp_tol

    def minimize(self, scalar_loss, x0, callback=None):
        import loom

        x0 = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        shape_orig = x0.shape

        # Flatten to 1D for scipy
        x0_flat = x0.reshape(-1)
        # jit ONCE (compiled on the flat 1D signature scipy calls with) and reuse across iterations.
        loss_j = loom.jit(lambda xf: scalar_loss(xf.reshape(shape_orig)))
        loss_f = lambda xf: float(loss_j(xf))      # scipy wants a plain float (a torch scalar has a torch dtype)
        grad_j = loom.jit(lambda xf: loom.grad(scalar_loss)(xf.reshape(shape_orig)).reshape(-1))

        x_flat = _two_phase_lbfgsb(
            loss_f, grad_j, x0_flat, shape_orig,
            max_iter=self.max_iter, ftol=self.ftol, min_iter=self.min_iter, disp_tol=self.disp_tol,
            callback=callback,
        )
        return x_flat.reshape(shape_orig)


class FusedLBFGS(Optimizer):
    """L-BFGS via scipy, like `LBFGS`, but for a cost whose gradient comes from an external
    FUSED evaluation (`value_and_grad(points) -> (cost: float, grad: ndarray)`, e.g.
    `dirac_fused.diracs_cost_grad`) rather than from Jax autodiff -- tracing cost and gradient
    separately (what `LBFGS` does via `loom.jit`/`loom.grad`) would throw away this fusion (the
    fused kernel computes both in a single pass, see `dirac_fused.py`).

    Same `min_iter`/`disp_tol` policy as `LBFGS` (see its docstring, common machinery in
    `_two_phase_lbfgsb`) -- only the way scipy obtains `(value, gradient)` differs:
    `jac=True`, `value_and_grad` already meeting the scipy contract for this option.

    `minimize`'s `scalar_loss` is NOT here a jax-traceable scalar function as for the
    other optimizers, but directly `value_and_grad` -- see `Reconstruction.run`, which chooses
    one call or the other depending on the type of the optimizer.
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
        import loom

        x = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        grad = loom.jit(loom.grad(scalar_loss))
        loss_j = loom.jit(scalar_loss)

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
        import loom

        x = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        grad = loom.jit(loom.grad(scalar_loss))

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
    """Prototype: instead of scipy's internal curvature memory (`LBFGS`), keeps an
    explicit stack of the last `max_dirs - 1` steps ACTUALLY taken (`x_k - x_{k-1}`, normalized) plus
    the current gradient, as the basis of a subspace, and solves at EACH step an EXACT Newton in
    this subspace (`m` <= `max_dirs` unknowns `a`, `m x m`) instead of a simple descent
    direction + 1D line search -- `dirac_fused.subspace_hessian` provides `(H, b)` in a second fused
    `loom.ffi_call`, closed formula (see its docstring for the derivation).

    Stacks the steps ACTUALLY taken, NOT the successive gradients (first version tested): near
    an optimum, successive gradients become nearly collinear (the classic zigzag of
    gradient descent), so a basis of raw gradients ends up carrying no new
    info -- `H` becomes ill-conditioned and the Tikhonov damping crushes the little useful signal
    with noise. The steps ACTUALLY taken play the role of L-BFGS's `s_k` (a different direction
    at each iteration WHEN it converges well) without rebuilding its whole apparatus of
    `(s_k, y_k)` pairs. The current gradient ALWAYS stays in the basis (fresh, never historical)
    to guarantee at least one valid descent direction even when the history is empty (1st
    step) or degenerate.

    Does NOT use `_two_phase_lbfgsb`/scipy (no L-BFGS curvature memory to run here,
    the restricted Hessian is recomputed at each step): home-made Python loop, with an
    Armijo backtracking as a safeguard -- the subspace Newton step is only optimal for the
    LOCAL quadratic model (frozen assignment), not for the true loss (nonconvex, nonsmooth
    at assignment changes).
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
        #: optional, `diag_callback(step, x, cost, g, directions, m, H, b, a, step_dir,
        #: directional_deriv)` called AFTER solving the subspace Newton step but BEFORE the
        #: backtracking -- to inspect the local quadratic model (e.g. compare it to the true
        #: loss along `step_dir`, see `experiments/lung_alveoli.run_subspace_alpha_profile`)
        #: without duplicating the logic of `minimize`. `directions` is passed ONLY on `[:m]` (the
        #: active columns, not the `MAX_DIRS` padding).
        self.diag_callback = diag_callback

    def minimize(self, value_and_grad, x0, callback=None):
        from . import dirac_fused

        MAX_DIRS = dirac_fused.MAX_DIRS
        if self.max_dirs > MAX_DIRS:
            raise ValueError(f"max_dirs={self.max_dirs} exceeds dirac_fused.MAX_DIRS={MAX_DIRS}")

        x = x0.copy() if isinstance(x0, np.ndarray) else np.array(x0)
        shape = x.shape
        # steps ACTUALLY taken (normalized), oldest first -- the current gradient (fresh,
        # never historical) occupies the remaining slot, see the class docstring.
        step_history = deque(maxlen=max(self.max_dirs - 1, 0))

        # no `callback(-1, x)` here: `Reconstruction.run` already called it for the initial state
        # before `optimizer.minimize` (see its docstring) -- same convention as `LBFGS`/
        # `GradientDescentLineSearch`, whose first callback is `step=0`.
        cost, g = value_and_grad(x)

        for step in range(self.max_iter):
            g_norm = np.linalg.norm(g)
            basis = ([g / g_norm] if g_norm > 0 else []) + list(step_history)

            m = len(basis)
            directions = np.zeros((MAX_DIRS,) + shape, dtype=x.dtype)
            for i, d in enumerate(basis):
                directions[i] = d

            H5, b5 = dirac_fused.subspace_hessian(x, directions, self.sinogram)
            H = 0.5 * (H5[:m, :m] + H5[:m, :m].T)
            b = b5[:m]

            damping = self.tikhonov * (np.trace(H) / m if m > 0 else 1.0)
            try:
                a = -np.linalg.solve(H + damping * np.eye(m), b)
            except np.linalg.LinAlgError:
                a = -np.linalg.lstsq(H, b, rcond=None)[0]
            step_dir = sum(a[i] * directions[i] for i in range(m))

            # Armijo backtracking along the subspace Newton step -- safeguard: the local
            # quadratic model is only exact at frozen assignment, not globally.
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
                # fallback: a normalized gradient descent step, guaranteed descending for t small
                # enough -- should almost never trigger (H is PSD by construction).
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

            # stack the step ACTUALLY taken (not the candidate step `step_dir` -- the backtracking may have
            # shortened it, or even switched to the gradient fallback) for the basis of the next
            # iteration, see the class docstring.
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
