"""L-BFGS: two-loop-recursion direction + the same bracket/parabola step-size
selection as every other `LineSearch` here."""
import time

import numpy as np

from .base import LineSearch
from .recorder import Recorder
from .step_size import bracket_step, parabola_vertex


def _two_loop_direction(g, s_hist, y_hist):
    """Two-loop recursion: ``H_k^{-1}·g`` from the last ``len(s_hist)``
    curvature pairs (Nocedal & Wright). Returns a field shaped like ``g``
    — used as ``p - t·direction``, so on empty history (nothing accepted
    yet) this degrades exactly to plain gradient descent.
    """
    q = g.ravel().astype(np.float64).copy()
    m = len(s_hist)
    alphas = np.empty(m)
    rhos = np.empty(m)
    for i in range(m - 1, -1, -1):
        s, y = s_hist[i], y_hist[i]
        rhos[i] = 1.0 / np.dot(y, s)
        alphas[i] = rhos[i] * np.dot(s, q)
        q -= alphas[i] * y
    if m > 0:
        s_last, y_last = s_hist[-1], y_hist[-1]
        gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
    else:
        gamma = 1.0
    r = gamma * q
    for i in range(m):
        s, y = s_hist[i], y_hist[i]
        beta = rhos[i] * np.dot(y, r)
        r += s * (alphas[i] - beta)
    return r.reshape(g.shape).astype(np.float32)


class LBFGS(LineSearch):
    """L-BFGS: direction from the two-loop recursion over the last
    ``memory`` (position-delta, gradient-delta) pairs, combined with the SAME
    bracket+parabola step-size selection as every other `LineSearch` here —
    so runs differ only in DIRECTION, not in how far they step along it.

    Skips updating the curvature memory whenever ``s_k·y_k <= 0`` (the
    standard L-BFGS safeguard: that pair would make the implied Hessian
    approximation indefinite) — the step for that iteration still comes
    from the recursion over the EXISTING memory, only the update is
    dropped, same auto-restart spirit as `ConjugateGradient`'s Polak-Ribière+
    safeguard. Memory is capped at ``memory`` pairs (oldest dropped first).

    ``disp_tol`` — stop early once the max per-point per-iteration displacement
    (``max(|p_new - p|)``) drops below this (world units), on top of ``ftol``/
    ``plateau_frac``. Useful for a short refinement phase (e.g. a triangle-disks
    stage in an `optim.pipeline` pipeline) where the model's own length scale
    (the disk radius) gives a natural, resolution-independent tolerance -- e.g.
    ``disp_tol = 0.01 * radius`` stops once points move by less than 1% of the
    triangle size in a single step, instead of running the full ``max_iter``
    budget or relying on ``ftol`` (which doesn't know about the geometric
    scale). ``None`` (default): disabled.
    """
    algo = "lbfgs"

    def __init__(self, memory: int = 10, disp_tol: float | None = None):
        self.memory = memory
        self.disp_tol = disp_tol

    def run(self, points, cost_model, recorder=None, *, max_iter=60, ftol=1e-10,
           verbose=True, plateau_frac=None):
        recorder = recorder if recorder is not None else Recorder(enabled=False)

        p = np.asarray(points, dtype=np.float32).copy()
        cost, g = cost_model.cost_grad(p)
        g = np.asarray(g, dtype=np.float32)

        t_start = time.time()
        recorder.record_step(-1, p, cost, t_start, algo=self.algo, radius=cost_model.frame_radius)

        g_norms = np.linalg.norm(g, axis=-1)
        mean_n = float(np.mean(g_norms)) if len(g_norms) else 1.0
        a = 0.01 * cost_model.extent / mean_n if mean_n > 0 else 0.0

        s_hist: list = []   # position deltas p_{k+1} - p_k, flattened float64
        y_hist: list = []   # gradient deltas g_{k+1} - g_k, flattened float64
        disp_prev = None
        first_gain = None

        for step in range(max_iter):
            direction = _two_loop_direction(g, s_hist, y_hist)

            t_bracket = time.time()
            a, cost_half, cost_a, tries = bracket_step(cost_model.cost, p, cost, direction, a)
            t_bracket = (time.time() - t_bracket) * 1000
            d0 = -float(np.sum(g * direction))
            t = parabola_vertex(cost, cost_half, cost_a, a, d0=d0)

            p_new = p - t * direction
            disp = p_new - p
            disp_mag = float(np.max(np.abs(disp)))
            dir_cos = recorder.direction_cosine(disp, disp_prev)
            mean_disp = float(np.mean(np.linalg.norm(disp, axis=-1)))
            disp_prev = disp
            t_grad = time.time()
            cost_new, g_new = cost_model.cost_grad(p_new)
            t_grad = (time.time() - t_grad) * 1000
            g_new = np.asarray(g_new, dtype=np.float32)
            a = 2.0 * t

            s_k = disp.ravel().astype(np.float64)
            y_k = (g_new - g).ravel().astype(np.float64)
            sy = float(np.dot(s_k, y_k))
            restarted = sy <= 1e-12
            if not restarted:
                s_hist.append(s_k); y_hist.append(y_k)
                if len(s_hist) > self.memory:
                    s_hist.pop(0); y_hist.pop(0)

            recorder.record_step(step, p_new, cost_new, t_start, algo=self.algo,
                                 radius=cost_model.frame_radius,
                                 dir_cos=dir_cos, mean_disp=mean_disp)
            recorder.timings.append(dict(
                step=step, n_points=p.shape[0], algo=self.algo,
                t=round(float(t), 4), tries=tries, dir_cos=dir_cos,
                mean_disp=round(mean_disp, 6), mem=len(s_hist), restarted=restarted,
                bracket_ms=round(t_bracket, 1), grad_ms=round(t_grad, 1)))

            if verbose:
                cos_str = f", cos={dir_cos:+.3f}" if dir_cos is not None else ""
                print(f"  [lbfgs] step {step:3d}: loss={cost_new:.6f}, "
                      f"max disp={disp_mag:.4g}, mean disp={mean_disp:.4g}  "
                      f"(bracket tries={tries}, t={t:.4g}, mem={len(s_hist)}{cos_str})")

            if abs(cost - cost_new) < ftol:
                if verbose:
                    print(f"  [lbfgs] converged (step {step + 1}): |Δcost| < {ftol}")
                return p_new

            if self.disp_tol is not None and disp_mag < self.disp_tol:
                if verbose:
                    print(f"  [lbfgs] converged (step {step + 1}): max disp {disp_mag:.4g} "
                          f"< disp_tol={self.disp_tol:.4g}")
                return p_new

            gain = cost - cost_new
            if first_gain is None:
                first_gain = gain
            elif plateau_frac is not None and first_gain > 0 and gain < plateau_frac * first_gain:
                if verbose:
                    print(f"  [lbfgs] plateau (step {step + 1}): gain {gain:.3g} < "
                          f"{plateau_frac:.0%} of this stage's first gain ({first_gain:.3g})")
                return p_new

            p, cost, g = p_new, cost_new, g_new

        return p
