"""Fine-grid + local-refine scans over the gradient direction combined with
one or two previous displacements — the core primitive of `Grid2DOracle`/
`Grid3DOracle`, and reused as-is for `instrumentation.instrument_step`'s
diagnostics.
"""
import numpy as np


def scan_grid2d(cost_fn, p, g, d_prev, t_hi, n_grid_2d, tp_range=(-0.5, 1.5),
                max_expand: int = 4):
    """Fine grid + local refine of cost(p + t_prev·d_prev - t_g·g).

    If the refined optimum lands on a boundary of the search domain (``t_hi``
    or ``tp_range``), that boundary is grown (doubled / extended by its own
    width) and the grid+refine is redone — up to ``max_expand`` times — so a
    minimum that genuinely lies further out isn't silently clipped. ``t_g``'s
    lower bound stays fixed at 0 (stepping backward along the gradient isn't
    a useful direction to search).

    Returns ``(tg_best, tp_best, cost_best, tg_vals, tp_vals, cost_grid)`` —
    the grid/cost_grid returned are those of the LAST (possibly grown) pass.
    """
    # local import: scipy is only needed for this opt-in debug/reference path.
    from scipy.optimize import minimize

    def cost_2d(x):
        tg, tp = x
        return cost_fn((p + tp * d_prev - tg * g).astype(np.float32))

    tp_lo, tp_hi = tp_range
    for _expand in range(max_expand + 1):
        tg_vals = np.linspace(0.0, t_hi, n_grid_2d)
        tp_vals = np.linspace(tp_lo, tp_hi, n_grid_2d)
        cost_grid = np.empty((n_grid_2d, n_grid_2d), dtype=np.float64)
        for i, tp in enumerate(tp_vals):
            for j, tg in enumerate(tg_vals):
                cost_grid[i, j] = cost_2d((tg, tp))
        ij_best = np.unravel_index(int(np.argmin(cost_grid)), cost_grid.shape)

        # refine the grid's argmin with a local (bounded) search — the grid
        # is coarse (n_grid_2d points/axis) and can undershoot a sharp minimum.
        x0 = (tg_vals[ij_best[1]], tp_vals[ij_best[0]])
        res = minimize(cost_2d, x0=x0, method="Nelder-Mead",
                       bounds=[(0.0, t_hi), (tp_lo, tp_hi)],
                       options=dict(xatol=1e-6 * max(t_hi, 1.0), fatol=1e-10))
        tg_best, tp_best = float(res.x[0]), float(res.x[1])

        edge_g    = tg_best > t_hi * (1 - 1e-3)
        edge_p_hi = tp_best > tp_hi - (tp_hi - tp_lo) * 1e-3
        edge_p_lo = tp_best < tp_lo + (tp_hi - tp_lo) * 1e-3
        if not (edge_g or edge_p_hi or edge_p_lo):
            break
        if edge_g:
            t_hi *= 2.0
        if edge_p_hi:
            tp_hi += (tp_hi - tp_lo)
        if edge_p_lo:
            tp_lo -= (tp_hi - tp_lo)

    return tg_best, tp_best, float(res.fun), tg_vals, tp_vals, cost_grid


def scan_grid3d(cost_fn, p, g, d_prev, d_prev2, t_hi, n_grid_3d,
                tp_range=(-0.5, 1.5), tp2_range=(-0.5, 1.5), max_expand: int = 3):
    """Fine 3D grid + local refine of cost(p + t_prev·d_prev + t_prev2·d_prev2 - t_g·g).

    3-axis generalisation of `scan_grid2d`: combines the gradient direction
    with the *two* previous steps' displacements. Grows any axis whose
    refined optimum lands on its boundary, same rationale as `scan_grid2d`.
    O(n_grid_3d³) loss evaluations per call — meant for `Grid3DOracle`, an
    even more expensive reference/oracle than `Grid2DOracle`; run once and
    cache.

    Returns ``(tg_best, tp_best, tp2_best, cost_best)``.
    """
    # local import: scipy is only needed for this opt-in debug/reference path.
    from scipy.optimize import minimize

    def cost_3d(x):
        tg, tp, tp2 = x
        return cost_fn((p + tp * d_prev + tp2 * d_prev2 - tg * g).astype(np.float32))

    tp_lo, tp_hi = tp_range
    tp2_lo, tp2_hi = tp2_range

    for _expand in range(max_expand + 1):
        tg_vals  = np.linspace(0.0, t_hi, n_grid_3d)
        tp_vals  = np.linspace(tp_lo, tp_hi, n_grid_3d)
        tp2_vals = np.linspace(tp2_lo, tp2_hi, n_grid_3d)
        cost_grid = np.empty((n_grid_3d, n_grid_3d, n_grid_3d), dtype=np.float64)
        for k, tp2 in enumerate(tp2_vals):
            for i, tp in enumerate(tp_vals):
                for j, tg in enumerate(tg_vals):
                    cost_grid[k, i, j] = cost_3d((tg, tp, tp2))
        kij_best = np.unravel_index(int(np.argmin(cost_grid)), cost_grid.shape)

        x0 = (tg_vals[kij_best[2]], tp_vals[kij_best[1]], tp2_vals[kij_best[0]])
        res = minimize(cost_3d, x0=x0, method="Nelder-Mead",
                       bounds=[(0.0, t_hi), (tp_lo, tp_hi), (tp2_lo, tp2_hi)],
                       options=dict(xatol=1e-6 * max(t_hi, 1.0), fatol=1e-10))
        tg_best, tp_best, tp2_best = float(res.x[0]), float(res.x[1]), float(res.x[2])

        edge_g     = tg_best  > t_hi  * (1 - 1e-3)
        edge_p_hi  = tp_best  > tp_hi  - (tp_hi  - tp_lo)  * 1e-3
        edge_p_lo  = tp_best  < tp_lo  + (tp_hi  - tp_lo)  * 1e-3
        edge_p2_hi = tp2_best > tp2_hi - (tp2_hi - tp2_lo) * 1e-3
        edge_p2_lo = tp2_best < tp2_lo + (tp2_hi - tp2_lo) * 1e-3
        if not (edge_g or edge_p_hi or edge_p_lo or edge_p2_hi or edge_p2_lo):
            break
        if edge_g:
            t_hi *= 2.0
        if edge_p_hi:
            tp_hi += (tp_hi - tp_lo)
        if edge_p_lo:
            tp_lo -= (tp_hi - tp_lo)
        if edge_p2_hi:
            tp2_hi += (tp2_hi - tp2_lo)
        if edge_p2_lo:
            tp2_lo -= (tp2_hi - tp2_lo)

    return tg_best, tp_best, tp2_best, float(res.fun)
