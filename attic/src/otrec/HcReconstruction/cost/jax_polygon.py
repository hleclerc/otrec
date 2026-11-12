"""Projected mass profile of a rotated REGULAR n-gon — a genuine 2D polygon
(unlike `jax_disks.py`'s "triangle" `shape`, which is a radial density
PROFILE on an otherwise circularly-symmetric disk, see that module's
docstring). Each point is `[x, y, theta]`: a center plus an orientation that
is itself an optimized variable, not just position.
"""
import functools

import jax
import jax.numpy as jnp

from .jax_disks import _ot1d_disks_angle


def _polygon_mass_angle(points, n_sides: int, radius, nx, ny, pix_edges):
    """Projected MASS of a union of regular `n_sides`-gons (circumradius
    `radius`, orientation `points[:, 2]`), one per point, `[nb_pixels]`.

    Only the RELATIVE angle `beta = atan2(ny, nx) - theta` between the
    projection direction and a polygon's own orientation matters (rotating
    polygon and detector together changes nothing), so in the polygon's own
    local `(d, r)` frame — `d` = projection axis, `r` = perpendicular — the
    vertices are a standard regular n-gon at angles `gamma_k = 2*pi*k/n - beta`:
    `x_k = radius*cos(gamma_k)` (local offset along `d`), `y_k =
    radius*sin(gamma_k)` (local offset along `r`).

    `H(tau) = Area(polygon ∩ {local offset along d <= tau})` is built as a
    SUM OF PER-EDGE closed-form contributions (Green's theorem, `oint -y dx`
    around the clipped boundary). For edge `k` (`(x_k,y_k) -> (x_{k+1},y_{k+1})`,
    `dx = x_{k+1}-x_k`, `dy = y_{k+1}-y_k`), the portion of that edge with
    local-`d`-offset `<= tau` is parameter range `t in [0, t_clip]` when
    `dx > 0` (offset increasing along the edge) but `t in [t_clip, 1]` when
    `dx < 0` (offset DECREASING along the edge — the two cases are NOT
    symmetric: `t_clip = clip((tau-x_k)/dx, 0, 1)` lands on the correct side
    of the edge only when `dx > 0`); an EARLIER version of this used
    `[0, t_clip]` unconditionally and was wrong by exactly this — caught by
    a numeric cross-check against a brute-force half-plane clip, not by
    inspection. Contribution over range `[lo, hi]` is
    `-dx * (y_k*(hi-lo) + dy*(hi^2-lo^2)/2)`; `dx == 0` edges contribute 0
    regardless (safe division guard) — same `jnp.clip`-based regime-selection
    idiom as `jax_disks._triangle_mass_angle`'s `vr = jnp.clip(v, -radius, radius)`,
    no sorting needed.

    The two vertical "closing" segments of a conceptually-clipped polygon
    boundary contribute EXACTLY ZERO to `oint -y dx` (dx=0 along them), which
    is why summing clipped per-edge contributions alone gives the right
    answer for the clipped AREA, no explicit polygon-clipping algorithm
    needed — verified two ways: at `tau -> +inf` (`hi-lo=1` everywhere) this
    sum collapses to the standard shoelace-via-trapezoids formula
    `sum (x_k - x_{k+1})(y_k + y_{k+1})/2` for the FULL polygon area (CCW
    vertices, as generated here, give the expected positive area), and at
    `tau -> -inf` it is identically 0. Cross-checked numerically against a
    brute-force half-plane-clip rasterization (`n_sides` in {3,5}, several
    `(radius, theta, angle)`) and against Monte-Carlo rasterization in
    `tests/test_hc_polygon.py::hc_polygon_projection_matches_reference` —
    that test is the actual authority, this docstring is the derivation.

    `mass_tgt` is then `H[:, 1:] - H[:, :-1]` on `pix_edges`, same pattern as
    `_disk_mass_angle`/`_triangle_mass_angle`.
    """
    centers = points[:, :2]                                     # [npoly, 2]
    theta = points[:, 2]                                        # [npoly]
    phi = jnp.arctan2(ny, nx)                                    # scalar
    beta = phi - theta                                           # [npoly]
    k = jnp.arange(n_sides)                                      # [n_sides]
    gamma = 2.0 * jnp.pi * k[None, :] / n_sides - beta[:, None]   # [npoly, n_sides]
    x_v = radius * jnp.cos(gamma)                                # [npoly, n_sides] (local, along d)
    y_v = radius * jnp.sin(gamma)                                # [npoly, n_sides] (local, along r)

    x_next = jnp.roll(x_v, -1, axis=1)
    y_next = jnp.roll(y_v, -1, axis=1)
    dx = x_next - x_v                                            # [npoly, n_sides]
    dy = y_next - y_v                                            # [npoly, n_sides]

    s0 = centers[:, 0] * nx + centers[:, 1] * ny                 # [npoly]
    tau = pix_edges[None, :] - s0[:, None]                       # [npoly, nb_pixels+1]  (local offset)

    dx_b = dx[:, :, None]
    dy_b = dy[:, :, None]
    x_v_b = x_v[:, :, None]
    y_v_b = y_v[:, :, None]
    tau_b = tau[:, None, :]

    safe_dx = jnp.where(dx_b != 0, dx_b, 1.0)
    t_raw = (tau_b - x_v_b) / safe_dx
    t_clip = jnp.clip(t_raw, 0.0, 1.0)                           # [npoly, n_sides, nb_pixels+1]
    lo = jnp.where(dx_b > 0, 0.0, t_clip)
    hi = jnp.where(dx_b > 0, t_clip, 1.0)

    contribution = -dx_b * (y_v_b * (hi - lo) + dy_b * (hi * hi - lo * lo) / 2.0)
    H = jnp.sum(contribution, axis=1)                            # [npoly, nb_pixels+1]
    mass = H[:, 1:] - H[:, :-1]                                  # [npoly, nb_pixels]
    return jnp.sum(mass, axis=0)                                 # [nb_pixels]


def _polygon_cost_angle(points, radius, nx, ny, sino_row, bin_centers_src, pix_edges, n_sides: int):
    mass_tgt = _polygon_mass_angle(points, n_sides, radius, nx, ny, pix_edges)
    return _ot1d_disks_angle(sino_row, bin_centers_src, mass_tgt, pix_edges)


def polygon_ot_all_angles(points, n_sides: int, radius, normals, sino_vals, bin_centers_src, pix_edges):
    """Pure Jax OT for all angles, POLYGON variant. `points` `[npoly, 3]`
    (`x, y, theta`) is the only differentiated argument (see
    `cost.jax_cost.JaxPolygonCost`, which wraps this in
    `jax.value_and_grad(argnums=0)`) — the gradient w.r.t. `theta` falls out
    of ordinary autodiff through `_polygon_mass_angle`'s `gamma`, no separate
    derivation needed. `n_sides` is a static Python int, closed over via
    `functools.partial` before `vmap` as the LAST parameter (same slot
    `jax_disks.disk_ot_all_angles` binds `mass_fn` in) — `vmap`'s
    positional args fill the REMAINING signature slots in order, so a
    keyword-bound param must be the one position doesn't reach; binding it
    anywhere earlier collides with a positional arg meant for a different
    parameter."""
    cost_angle = functools.partial(_polygon_cost_angle, n_sides=n_sides)
    costs = jax.vmap(cost_angle, in_axes=(None, None, 0, 0, 0, None, None))(
        points, radius, normals[:, 0], normals[:, 1], sino_vals, bin_centers_src, pix_edges)
    return costs.sum()
