"""Line-search debugging: how far the parabola vertex is from the true minimum.

Opt-in via `GradientDescent`/`ConjugateGradient`'s `instrument_iters` —
results accumulate in `Recorder.debug_scans` for `experiments.exp_hc`
(pipeline e.g. `gd(diracs; instrument_iters=4)`) to plot.
"""
import numpy as np

from ..cost.base import CostModel
from .grid_scan import scan_grid2d


def instrument_step(cost_model: CostModel, step, p, g, d_prev, a, t_parabola, cost0,
                    n_scan_1d: int = 41, n_grid_2d: int = 21) -> dict:
    """Record how far the parabola vertex is from the true minimum.

    ``ts`` spans ``[0, 2a]`` — the same support the 3-point parabola
    (0, a/2, a) was fit on doubled, since ``a`` itself was set to ``2·t``
    of the previous accepted step, so the true minimum can land past it.

    The 2D grid explores ``p + t_prev·d_prev - t_g·g``: t_prev=0 is exactly
    the pure-gradient line the parabola searches, so comparing the grid's
    global minimum to the best point on that t_prev=0 axis tells us whether
    mixing in the previous step's direction would have helped.
    """
    # local import: scipy is only needed for this opt-in debug path.
    from scipy.optimize import minimize_scalar

    cost_fn = cost_model.cost
    t_hi = max(2.0 * a, 1e-12)

    def cost_1d(tt):
        return cost_fn((p - tt * g).astype(np.float32))

    # coarse curve, for plotting only — NOT the ground truth (grid resolution
    # can miss a sharp minimum, see cost_true_1d vs. minimize_scalar below).
    ts = np.linspace(0.0, t_hi, n_scan_1d)
    costs_1d = np.array([cost_1d(tt) for tt in ts])

    res_1d = minimize_scalar(cost_1d, bounds=(0.0, t_hi), method="bounded",
                             options=dict(xatol=1e-6 * max(t_hi, 1.0)))

    record = dict(
        step=step, a=float(a), cost0=float(cost0), ts=ts, costs_1d=costs_1d,
        t_parabola=float(t_parabola),
        cost_parabola=float(cost_1d(t_parabola)),
        t_true_1d=float(res_1d.x),
        cost_true_1d=float(res_1d.fun),
    )

    if d_prev is not None:
        tg_best, tp_best, cost_best, tg_vals, tp_vals, cost_grid = \
            scan_grid2d(cost_fn, p, g, d_prev, t_hi, n_grid_2d)
        record.update(
            tg_vals=tg_vals, tp_vals=tp_vals, cost_grid=cost_grid,
            tg_best=tg_best, tp_best=tp_best, cost_grid_best=cost_best,
        )

    return record
