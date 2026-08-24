"""Shared 1D step-size selection: bracket a descent step, then fit its vertex.

Every `LineSearch` reduces its step-size choice to these two free functions —
the one piece of numerics genuinely common to all of them. Both take a plain
`cost_fn(points) -> float` (usually a `CostModel.cost`), not a CostModel
itself, so they stay usable in isolation.
"""
import numpy as np


def bracket_step(cost_fn, p, cost0, g, a, max_tries: int = 30):
    """Find `a` s.t. a/2 is the lowest-cost of (0, a/2, a) along -g."""
    cost_half = cost_fn((p - (a / 2) * g).astype(np.float32))
    cost_a    = cost_fn((p - a * g).astype(np.float32))

    for tries in range(max_tries):
        if cost_half <= cost0 and cost_half <= cost_a:
            break
        if cost_a <= cost_half and cost_a <= cost0:
            a *= 2
            cost_half, cost_a = cost_a, cost_fn((p - a * g).astype(np.float32))
        else:
            a /= 2
            cost_a, cost_half = cost_half, cost_fn((p - (a / 2) * g).astype(np.float32))

    return a, cost_half, cost_a, tries


def parabola_vertex(cost0, cost_half, cost_a, a, d0: float | None = None):
    """Vertex of a polynomial fit through (0, cost0), (a/2, cost_half), (a, cost_a).

    ``d0`` — the analytic directional derivative ``d/dt cost(p - t·s)`` at
    ``t=0`` — is free at every call site (just ``-g·s`` from the gradient
    and direction already in hand before the bracket runs). When given,
    this upgrades the fit from a plain 3-point QUADRATIC to a Hermite
    CUBIC through ``(f(0), f'(0)=d0, f(a/2), f(a))`` — one more
    constraint than the quadratic, at zero extra evaluations. Falls back
    to the quadratic vertex (the previous behaviour) if the cubic has no
    interior minimum (degenerate leading coefficient, negative
    discriminant, or its critical point isn't a min / falls outside
    ``[0, a]``).
    """
    h = a / 2
    if d0 is not None and h > 0:
        dh = cost_half - cost0 - d0 * h
        da = cost_a - cost0 - d0 * a
        c3 = (da - 4 * dh) / (4 * h ** 3)
        c2 = dh / (h * h) - c3 * h
        disc = c2 * c2 - 3 * c3 * d0
        if c3 != 0 and disc >= 0:
            sq = disc ** 0.5
            for t in ((-c2 + sq) / (3 * c3), (-c2 - sq) / (3 * c3)):
                if 0.0 <= t <= a and 2 * c2 + 6 * c3 * t > 0:
                    return t

    denom = cost0 - 2 * cost_half + cost_a
    if denom <= 0:
        return h
    return min(max(h + h * (cost0 - cost_a) / (2 * denom), 0.0), a)
