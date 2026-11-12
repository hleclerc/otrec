"""Cheap 2D-quadratic line search: a 6-point paraboloid fit standing in for
`Grid2DOracle`'s exhaustive grid, cheap enough for production use.
"""
import time

import numpy as np

from .base import LineSearch
from .recorder import Recorder
from .step_size import bracket_step, parabola_vertex


def _quad2d_vertex(cost_fn, p, g, d_prev, cost0, hg, hp=0.5):
    """6-point cross+diagonal quadratic fit of cost(p + tp·d_prev - tg·g)
    around (tg, tp) = (0, 0); returns the analytic vertex ``(tg, tp)`` of
    the fitted paraboloid — a 2D generalisation of `step_size.parabola_vertex`'s
    3-point fit, using only 5 extra evaluations (vs. `grid_scan.scan_grid2d`'s
    O(n²) grid + Nelder-Mead refine). Falls back to a plain gradient half-step
    if the fitted Hessian isn't positive-definite (no genuine local minimum to
    jump to — e.g. still on a downhill slope).

    The linear terms ``bx``, ``by`` use the EXACT directional derivatives
    at (0, 0) instead of the finite-difference estimate ``(cxp-cxm)/2hg``
    — both probe directions are straight lines through ``p`` (``-g`` and
    ``d_prev``), so ``bx = -g·g`` and ``by = g·d_prev`` are free from the
    gradient already in hand. (A cheaper 3-probe variant, dropping
    ``cxm``/``cym`` and estimating ``axx``/``ayy`` one-sided against
    these exact linear terms, was tried and measured WORSE empirically —
    the one-sided curvature estimate's O(hg) truncation error outweighed
    the saved evaluations at this step scale — so ``cxm``/``cym`` stay,
    keeping the accurate central-difference curvature.)
    """
    def cost_2d(tg, tp):
        return cost_fn((p + tp * d_prev - tg * g).astype(np.float32))

    bx = -float(np.sum(g * g))
    by = float(np.sum(g * d_prev))

    cxp = cost_2d(hg, 0.0)
    cxm = cost_2d(-hg, 0.0)
    cyp = cost_2d(0.0, hp)
    cym = cost_2d(0.0, -hp)
    cd  = cost_2d(hg, hp)

    axx = (cxp - 2 * cost0 + cxm) / (hg * hg)
    ayy = (cyp - 2 * cost0 + cym) / (hp * hp)
    axy = (cd - cost0 - bx * hg - by * hp
          - 0.5 * axx * hg * hg - 0.5 * ayy * hp * hp) / (hg * hp)

    det = axx * ayy - axy * axy
    if axx > 0 and det > 0:
        tg, tp = np.linalg.solve(np.array([[axx, axy], [axy, ayy]]),
                                 -np.array([bx, by]))
        tg = float(np.clip(tg, 0.0, 4 * hg))
        tp = float(np.clip(tp, -2.0, 2.0))
    else:
        tg, tp = hg, 0.0

    return tg, tp


class Quad2D(LineSearch):
    """Cheap 2D-quadratic line search: same idea as `Grid2DOracle` (combine
    the gradient direction with the *previous* step's displacement) but
    instead of a full grid + Nelder-Mead refine, fits a paraboloid to 6
    points (a cross + one diagonal corner, using the gradient's exact linear
    terms, see `_quad2d_vertex`) and jumps to its analytic vertex — O(1)
    extra evaluations per step, cheap enough for production use (unlike
    `Grid2DOracle`/`Grid3DOracle`).

    Falls back to a plain 1D bracket/parabola step (same as
    `GradientDescent`) on the first iteration, when there is no previous
    direction yet.
    """
    algo = "quad2d"

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
                tg, tp = _quad2d_vertex(cost_model.cost, p, g, d_prev, cost, hg=a / 2)
                p_new = p + tp * d_prev - tg * g
                a = 2.0 * tg if tg > 1e-9 else a
                diag = dict(tg=round(tg, 4), tp=round(tp, 4))

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
                print(f"  [quad2d] step {step:3d}: loss={cost_new:.6f}, "
                      f"max disp={disp_mag:.4g}, mean disp={mean_disp:.4g}  "
                      f"(t_g={diag['tg']:.4g}, t_prev={diag['tp']:.4g}{cos_str})")

            if abs(cost - cost_new) < ftol:
                if verbose:
                    print(f"  [quad2d] converged (step {step + 1}): |Δcost| < {ftol}")
                return p_new

            gain = cost - cost_new
            if first_gain is None:
                first_gain = gain
            elif plateau_frac is not None and first_gain > 0 and gain < plateau_frac * first_gain:
                if verbose:
                    print(f"  [quad2d] plateau (step {step + 1}): gain {gain:.3g} < "
                          f"{plateau_frac:.0%} of this stage's first gain ({first_gain:.3g})")
                return p_new

            p, cost, g = p_new, cost_new, g_new

        return p


class GQuad2D(LineSearch):
    """Like `Quad2D`, but the 2D fit's *second* axis comes from a GRADIENT
    evaluated during the bracket phase, not from the previous accepted
    step's displacement.

    `Quad2D`/`Grid2DOracle`/`Grid3DOracle` all need a previous step before
    they have a second direction to combine with ``-g`` — their first
    iteration falls back to plain gradient descent. Here, the bracket phase
    already evaluates the LOSS at the far trial point ``p - a·g``; this
    method additionally evaluates the GRADIENT there (``g_a``, one extra
    `cost_grad` call — cheap, same cost class as `Quad2D`'s 5 extra probes).
    ``g_a`` differs from ``g`` by exactly the curvature the loss has along
    that line, so ``d2 = -a·g_a`` (what a gradient step of the SAME scale
    ``a`` would do FROM the endpoint) is a genuinely new direction, scaled to
    the same order of magnitude as the bracket's own displacement —
    available from the VERY FIRST step, no history needed.

    Feeds ``(g, d2)`` into the same 6-point paraboloid fit as `Quad2D`
    (`_quad2d_vertex`) and jumps to its analytic vertex.
    """
    algo = "gquad2d"

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

        disp_prev = None
        first_gain = None

        for step in range(max_iter):
            a, _cost_half, _cost_a, tries = bracket_step(cost_model.cost, p, cost, g, a)
            _, g_a = cost_model.cost_grad((p - a * g).astype(np.float32))
            g_a = np.asarray(g_a, dtype=np.float32)
            d2 = -a * g_a

            tg, tp = _quad2d_vertex(cost_model.cost, p, g, d2, cost, hg=a / 2)
            p_new = p + tp * d2 - tg * g
            a = 2.0 * tg if tg > 1e-9 else a
            diag = dict(tg=round(tg, 4), tp=round(tp, 4), tries=tries)

            disp = p_new - p
            dir_cos = recorder.direction_cosine(disp, disp_prev)
            mean_disp = float(np.mean(np.linalg.norm(disp, axis=-1)))
            disp_prev = disp
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
                print(f"  [gquad2d] step {step:3d}: loss={cost_new:.6f}, "
                      f"max disp={disp_mag:.4g}, mean disp={mean_disp:.4g}  "
                      f"(t_g={diag['tg']:.4g}, t_prev={diag['tp']:.4g}{cos_str})")

            if abs(cost - cost_new) < ftol:
                if verbose:
                    print(f"  [gquad2d] converged (step {step + 1}): |Δcost| < {ftol}")
                return p_new

            gain = cost - cost_new
            if first_gain is None:
                first_gain = gain
            elif plateau_frac is not None and first_gain > 0 and gain < plateau_frac * first_gain:
                if verbose:
                    print(f"  [gquad2d] plateau (step {step + 1}): gain {gain:.3g} < "
                          f"{plateau_frac:.0%} of this stage's first gain ({first_gain:.3g})")
                return p_new

            p, cost, g = p_new, cost_new, g_new

        return p
