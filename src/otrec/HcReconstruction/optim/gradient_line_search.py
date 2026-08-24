"""Plain gradient descent and nonlinear CG (Polak-Ribière+) — two thin classes
sharing one bracket/parabola loop via composition (`_run`), not inheritance.
"""
import time

import numpy as np

from ..cost.base import CostModel
from .base import LineSearch
from .instrumentation import instrument_step
from .recorder import Recorder
from .step_size import bracket_step, parabola_vertex


def _run(direction_fn, points, cost_model: CostModel, recorder: Recorder | None, *,
        algo: str, max_iter: int, ftol: float, verbose: bool,
        plateau_frac: float | None, instrument_iters: int):
    """Shared bracket/parabola line-search loop, parametrised by search direction.

    ``direction_fn(g, s_prev, g_prev) -> (s, diag)`` computes the search
    direction ``s`` (used as ``p - t·s``) for the current step from the current
    gradient ``g`` and the previous iteration's direction/gradient (``None`` on
    the first step). ``diag`` is a dict of extra per-step fields (e.g. PR's
    ``beta``) merged into the recorded timings row for that step.

    ``instrument_iters`` — for the first N steps, additionally: (1) do a fine
    1D scan of cost(p - t·s) to compare the true minimum along -s against the
    3-point parabola vertex actually used, and (2) do a 2D grid scan combining
    the search direction with the *previous* step's direction (the displacement
    just taken), to see whether a step off that axis would have done better.
    Results are appended to ``recorder.debug_scans`` (see `instrumentation.py`).

    ``plateau_frac`` — if set, stop as soon as one step's gain
    (``cost_before - cost_after``) drops below ``plateau_frac`` times the
    FIRST step's gain, instead of running to ``max_iter``/``ftol``. Meant
    for intermediate stages of `optim.pipeline.MultiscaleStage`: at a
    given resolution, further iterations mostly polish detail that the next
    (finer) stage will redo anyway, so it's wasted work — the same
    rationale as ``ftol``, just RELATIVE (a fixed absolute ``ftol``
    doesn't transfer across dirac counts, whose loss magnitude differs by
    orders of magnitude).
    """
    recorder = recorder if recorder is not None else Recorder(enabled=False)

    p = np.asarray(points, dtype=np.float32).copy()
    cost, g = cost_model.cost_grad(p)
    g = np.asarray(g)

    t_start = time.time()
    recorder.record_step(-1, p, cost, t_start, algo=algo, radius=cost_model.frame_radius)

    g_norms = np.linalg.norm(g, axis=-1)
    mean_n = float(np.mean(g_norms)) if len(g_norms) else 1.0
    a = 0.01 * cost_model.extent / mean_n if mean_n > 0 else 0.0

    recorder.debug_scans = []
    s_prev = None      # previous search direction, for the CG recursion
    g_prev = None       # previous gradient, for the CG recursion
    disp_prev = None    # previous step's displacement, for the 2D scan + colinearity
    first_gain = None    # this stage's step-0 gain, plateau_frac's reference

    for step in range(max_iter):
        s, diag = direction_fn(g, s_prev, g_prev)

        t_bracket = time.time()
        a, cost_half, cost_a, tries = bracket_step(cost_model.cost, p, cost, s, a)
        t_bracket = (time.time() - t_bracket) * 1000
        d0 = -float(np.sum(g * s))
        t = parabola_vertex(cost, cost_half, cost_a, a, d0=d0)

        if step < instrument_iters:
            recorder.debug_scans.append(
                instrument_step(cost_model, step, p, s, disp_prev, a, t, cost))

        p_new = p - t * s
        disp = p_new - p
        dir_cos = recorder.direction_cosine(disp, disp_prev)
        mean_disp = float(np.mean(np.linalg.norm(disp, axis=-1)))
        disp_prev = disp
        t_grad = time.time()
        cost_new, g_new = cost_model.cost_grad(p_new)
        t_grad = (time.time() - t_grad) * 1000
        g_new = np.asarray(g_new, dtype=np.float32)
        a = 2.0 * t

        recorder.record_step(step, p_new, cost_new, t_start, algo=algo,
                             radius=cost_model.frame_radius,
                             dir_cos=dir_cos, mean_disp=mean_disp)
        recorder.timings.append(dict(
            step=step, n_points=p.shape[0], algo=algo,
            t=round(float(t), 4), tries=tries, dir_cos=dir_cos,
            mean_disp=round(mean_disp, 6),
            bracket_ms=round(t_bracket, 1), grad_ms=round(t_grad, 1), **diag))

        if verbose:
            disp_mag = float(np.max(np.abs(p_new - p)))
            extra = "".join(f", {k}={v:.3g}" if isinstance(v, float) else f", {k}={v}"
                            for k, v in diag.items())
            cos_str = f", cos={dir_cos:+.3f}" if dir_cos is not None else ""
            print(f"  [{algo}] step {step:3d}: loss={cost_new:.6f}, "
                  f"max disp={disp_mag:.4g}, mean disp={mean_disp:.4g}  "
                  f"(bracket tries={tries}, t={t:.4g}{extra}{cos_str})")

        if abs(cost - cost_new) < ftol:
            if verbose:
                print(f"  [{algo}] converged (step {step + 1}): |Δcost| < {ftol}")
            return p_new

        gain = cost - cost_new
        if first_gain is None:
            first_gain = gain
        elif plateau_frac is not None and first_gain > 0 and gain < plateau_frac * first_gain:
            if verbose:
                print(f"  [{algo}] plateau (step {step + 1}): gain {gain:.3g} < "
                      f"{plateau_frac:.0%} of this stage's first gain ({first_gain:.3g})")
            return p_new

        s_prev, g_prev = s, g
        p, cost, g = p_new, cost_new, g_new

    return p


class GradientDescent(LineSearch):
    """Plain gradient descent with parabolic line search."""
    algo = "gd"

    def __init__(self, instrument_iters: int = 0):
        self.instrument_iters = instrument_iters

    def run(self, points, cost_model, recorder=None, *, max_iter=60, ftol=1e-10,
           verbose=True, plateau_frac=None):
        def direction(g, s_prev, g_prev):
            return g, {}
        return _run(direction, points, cost_model, recorder, algo=self.algo,
                   max_iter=max_iter, ftol=ftol, verbose=verbose,
                   plateau_frac=plateau_frac, instrument_iters=self.instrument_iters)


class ConjugateGradient(LineSearch):
    """Nonlinear conjugate gradient (Polak-Ribière+) with parabolic line search.

    Search direction ``s_k = g_k - beta_k · s_{k-1}`` (``s_0 = g_0``), so the
    step ``p - t·s_k`` moves along ``-g_k + beta_k·s_{k-1}`` — the classic CG
    combination of the current gradient and the *previous search direction*.
    ``beta_k = max(0, <g_k, g_k - g_{k-1}> / <g_{k-1}, g_{k-1}>)`` (Polak-Ribière+;
    the ``max(0, ·)`` auto-restarts to plain gradient descent whenever the PR
    ratio goes negative — the standard safeguard against divergence).
    """
    algo = "pr"

    def __init__(self, instrument_iters: int = 0):
        self.instrument_iters = instrument_iters

    def run(self, points, cost_model, recorder=None, *, max_iter=60, ftol=1e-10,
           verbose=True, plateau_frac=None):
        def direction(g, s_prev, g_prev):
            if s_prev is None:
                return g, dict(beta=0.0, restarted=True)
            denom = float(np.sum(g_prev * g_prev))
            beta = float(np.sum(g * (g - g_prev)) / denom) if denom > 0 else 0.0
            restarted = beta <= 0.0
            beta = max(0.0, beta)
            return g - beta * s_prev, dict(beta=round(beta, 4), restarted=restarted)
        return _run(direction, points, cost_model, recorder, algo=self.algo,
                   max_iter=max_iter, ftol=ftol, verbose=verbose,
                   plateau_frac=plateau_frac, instrument_iters=self.instrument_iters)
