"""Reference/oracle line searches: exhaustive fine-grid search over the
gradient direction combined with the previous 1 or 2 steps' displacements.

Upper-bound how much a combined-direction method (`ConjugateGradient`,
`Quad2D`) could possibly gain, at the cost of O(n_grid^k) extra loss
evaluations per step. Meant to be run once and cached, not for production use.
"""
import time

import numpy as np

from .base import LineSearch
from .grid_scan import scan_grid2d, scan_grid3d
from .recorder import Recorder
from .step_size import bracket_step, parabola_vertex


class Grid2DOracle(LineSearch):
    """Reference/oracle line search: always takes the global minimum of the
    fine 2D grid over ``(t_g, t_prev)`` combining the gradient direction with
    the *previous* step's displacement (`grid_scan.scan_grid2d`) as the real
    step — not just for diagnostics. This upper-bounds how much a
    combined-direction line search (e.g. `ConjugateGradient`) could possibly
    gain, at the cost of ``O(n_grid_2d²)`` extra loss evaluations per step
    (plus a local refine).

    Meant to be run once and cached (see `viz.line_search_compare.save_run`),
    not for production use. The first step has no previous direction yet, so
    it falls back to a plain 1D bracket/parabola step (same as
    `GradientDescent`).
    """
    algo = "grid2d"

    def __init__(self, n_grid_2d: int = 15):
        self.n_grid_2d = n_grid_2d

    def run(self, points, cost_model, recorder=None, *, max_iter=60, ftol=1e-10,
           verbose=True, plateau_frac=None):
        recorder = recorder if recorder is not None else Recorder(enabled=False)

        p = np.asarray(points, dtype=np.float32).copy()
        cost, g = cost_model.cost_grad(p)
        g = np.asarray(g)

        t_start = time.time()
        recorder.record_step(-1, p, cost, t_start, algo=self.algo, radius=cost_model.frame_radius)

        g_norms = np.linalg.norm(g, axis=-1)
        mean_n = float(np.mean(g_norms)) if len(g_norms) else 1.0
        a = 0.01 * cost_model.extent / mean_n if mean_n > 0 else 0.0

        d_prev = None
        first_gain = None

        for step in range(max_iter):
            if d_prev is None:
                a, cost_half, cost_a, _tries = bracket_step(cost_model.cost, p, cost, g, a)
                t = parabola_vertex(cost, cost_half, cost_a, a, d0=-float(np.sum(g * g)))
                p_new = p - t * g
                diag = dict(tg=round(float(t), 4), tp=0.0)
            else:
                t_hi = max(2.0 * a, 1e-12)
                tg_best, tp_best, _, *_ = scan_grid2d(
                    cost_model.cost, p, g, d_prev, t_hi, self.n_grid_2d)
                p_new = p + tp_best * d_prev - tg_best * g
                a = 2.0 * tg_best if tg_best > 1e-9 else a
                diag = dict(tg=round(tg_best, 4), tp=round(tp_best, 4))

            disp = p_new - p
            dir_cos = recorder.direction_cosine(disp, d_prev)
            mean_disp = float(np.mean(np.linalg.norm(disp, axis=-1)))
            d_prev = disp
            t_grad = time.time()
            cost_new, g_new = cost_model.cost_grad(p_new)
            t_grad = (time.time() - t_grad) * 1000
            g_new = np.asarray(g_new, dtype=np.float32)

            recorder.record_step(step, p_new, cost_new, t_start, algo=self.algo,
                                 radius=cost_model.frame_radius,
                                 dir_cos=dir_cos, mean_disp=mean_disp)
            recorder.timings.append(dict(
                step=step, n_points=p.shape[0], algo=self.algo,
                dir_cos=dir_cos, mean_disp=round(mean_disp, 6),
                grad_ms=round(t_grad, 1), **diag))

            if verbose:
                disp_mag = float(np.max(np.abs(p_new - p)))
                cos_str = f", cos={dir_cos:+.3f}" if dir_cos is not None else ""
                print(f"  [grid2d] step {step:3d}: loss={cost_new:.6f}, "
                      f"max disp={disp_mag:.4g}, mean disp={mean_disp:.4g}  "
                      f"(t_g={diag['tg']:.4g}, t_prev={diag['tp']:.4g}{cos_str})")

            if abs(cost - cost_new) < ftol:
                if verbose:
                    print(f"  [grid2d] converged (step {step + 1}): |Δcost| < {ftol}")
                return p_new

            gain = cost - cost_new
            if first_gain is None:
                first_gain = gain
            elif plateau_frac is not None and first_gain > 0 and gain < plateau_frac * first_gain:
                if verbose:
                    print(f"  [grid2d] plateau (step {step + 1}): gain {gain:.3g} < "
                          f"{plateau_frac:.0%} of this stage's first gain ({first_gain:.3g})")
                return p_new

            p, cost, g = p_new, cost_new, g_new

        return p


class Grid3DOracle(LineSearch):
    """Reference/oracle line search, one level up from `Grid2DOracle`:
    combines the gradient direction with the *two* previous steps'
    displacements (`grid_scan.scan_grid3d`), always taking that 3D grid's
    global minimum as the real step. Upper-bounds `Grid2DOracle` the same way
    `Grid2DOracle` upper-bounds `ConjugateGradient` — if this curve doesn't
    improve meaningfully over grid2d's, a third direction isn't worth chasing.

    O(n_grid_3d³) loss evaluations per step — much slower than grid2d; run
    once and cache. Falls back to `GradientDescent`'s 1D step on the first
    iteration (no previous direction yet) and to `Grid2DOracle`'s 2D scan on
    the second (only one previous direction yet).
    """
    algo = "grid3d"

    def __init__(self, n_grid_3d: int = 7):
        self.n_grid_3d = n_grid_3d

    def run(self, points, cost_model, recorder=None, *, max_iter=60, ftol=1e-10,
           verbose=True, plateau_frac=None):
        recorder = recorder if recorder is not None else Recorder(enabled=False)

        p = np.asarray(points, dtype=np.float32).copy()
        cost, g = cost_model.cost_grad(p)
        g = np.asarray(g)

        t_start = time.time()
        recorder.record_step(-1, p, cost, t_start, algo=self.algo, radius=cost_model.frame_radius)

        g_norms = np.linalg.norm(g, axis=-1)
        mean_n = float(np.mean(g_norms)) if len(g_norms) else 1.0
        a = 0.01 * cost_model.extent / mean_n if mean_n > 0 else 0.0

        d_prev = None
        d_prev2 = None
        first_gain = None

        for step in range(max_iter):
            if d_prev is None:
                a, cost_half, cost_a, _tries = bracket_step(cost_model.cost, p, cost, g, a)
                t = parabola_vertex(cost, cost_half, cost_a, a, d0=-float(np.sum(g * g)))
                p_new = p - t * g
                diag = dict(tg=round(float(t), 4), tp=0.0, tp2=0.0)
            elif d_prev2 is None:
                t_hi = max(2.0 * a, 1e-12)
                tg_best, tp_best, _, *_ = scan_grid2d(
                    cost_model.cost, p, g, d_prev, t_hi, self.n_grid_3d)
                p_new = p + tp_best * d_prev - tg_best * g
                a = 2.0 * tg_best if tg_best > 1e-9 else a
                diag = dict(tg=round(tg_best, 4), tp=round(tp_best, 4), tp2=0.0)
            else:
                t_hi = max(2.0 * a, 1e-12)
                tg_best, tp_best, tp2_best, _ = scan_grid3d(
                    cost_model.cost, p, g, d_prev, d_prev2, t_hi, self.n_grid_3d)
                p_new = p + tp_best * d_prev + tp2_best * d_prev2 - tg_best * g
                a = 2.0 * tg_best if tg_best > 1e-9 else a
                diag = dict(tg=round(tg_best, 4), tp=round(tp_best, 4), tp2=round(tp2_best, 4))

            disp = p_new - p
            dir_cos = recorder.direction_cosine(disp, d_prev)
            mean_disp = float(np.mean(np.linalg.norm(disp, axis=-1)))
            d_prev2 = d_prev
            d_prev = disp
            t_grad = time.time()
            cost_new, g_new = cost_model.cost_grad(p_new)
            t_grad = (time.time() - t_grad) * 1000
            g_new = np.asarray(g_new, dtype=np.float32)

            recorder.record_step(step, p_new, cost_new, t_start, algo=self.algo,
                                 radius=cost_model.frame_radius,
                                 dir_cos=dir_cos, mean_disp=mean_disp)
            recorder.timings.append(dict(
                step=step, n_points=p.shape[0], algo=self.algo,
                dir_cos=dir_cos, mean_disp=round(mean_disp, 6),
                grad_ms=round(t_grad, 1), **diag))

            if verbose:
                disp_mag = float(np.max(np.abs(p_new - p)))
                cos_str = f", cos={dir_cos:+.3f}" if dir_cos is not None else ""
                print(f"  [grid3d] step {step:3d}: loss={cost_new:.6f}, "
                      f"max disp={disp_mag:.4g}, mean disp={mean_disp:.4g}  "
                      f"(t_g={diag['tg']:.4g}, t_prev={diag['tp']:.4g}, "
                      f"t_prev2={diag['tp2']:.4g}{cos_str})")

            if abs(cost - cost_new) < ftol:
                if verbose:
                    print(f"  [grid3d] converged (step {step + 1}): |Δcost| < {ftol}")
                return p_new

            gain = cost - cost_new
            if first_gain is None:
                first_gain = gain
            elif plateau_frac is not None and first_gain > 0 and gain < plateau_frac * first_gain:
                if verbose:
                    print(f"  [grid3d] plateau (step {step + 1}): gain {gain:.3g} < "
                          f"{plateau_frac:.0%} of this stage's first gain ({first_gain:.3g})")
                return p_new

            p, cost, g = p_new, cost_new, g_new

        return p
