"""Pure Jax 1D OT cost for the DISKS model — one angle, then vmapped.

Mirrors `models.DiskModel`: the roles of source/target FLIP relative to
`jax_diracs.py`. The measured sinogram becomes the (fixed) SOURCE — weighted
diracs at the bin centers, one per detector bin, no shape of its own — and the
disk centers (the unknown) parametrise the TARGET, a piecewise-constant
density built by projecting the union of fixed-radius disks onto a
`nb_pixels`-wide grid (same chord-integral math as
`disks.DiskProjector._values_of`, ported to plain arrays). Because the
source's positions/weights are FIXED, no sort is needed here (bin centers are
trivially sorted by construction) — unlike `_ot1d_angle`, which sorts the
points because THEY are the (moving) source.
"""
import functools

import jax
import jax.numpy as jnp


def _disk_mass_angle(centers, radius, nx, ny, pix_edges):
    """Projected MASS (not density) of the union of `radius`-disks at one
    angle, per pixel of `pix_edges` (`[nb_pixels + 1]`). `[nb_pixels]`.

    Same chord-integral primitive as `disks.DiskProjector._values_of`
    (`G(t) = t*sqrt(r^2-t^2) + r^2*asin(t/r)`), with the same NaN-safe
    surrogate derivative (`H = stop_grad(G - chord*u) + stop_grad(chord)*u`)
    — see that method's docstring for why the plain autodiff of `G` is
    numerically unusable near a case/disk edge.
    """
    s0 = centers[:, 0] * nx + centers[:, 1] * ny                # [ndisks]
    u = pix_edges[None, :] - s0[:, None]                        # [ndisks, nb_pixels+1]
    t = jnp.clip(u, -radius, radius)
    sq = jnp.sqrt(jnp.maximum(radius * radius - t * t, 0.0))
    chord = 2.0 * sq
    G = t * sq + radius * radius * jnp.arcsin(jnp.clip(t / radius, -1.0, 1.0))
    H = jax.lax.stop_gradient(G - chord * u) + jax.lax.stop_gradient(chord) * u
    mass = H[:, 1:] - H[:, :-1]                                 # [ndisks, nb_pixels]
    return jnp.sum(mass, axis=0)                                # [nb_pixels]


def _triangle_mass_angle(centers, radius, nx, ny, pix_edges):
    """Projected MASS of a union of TRIANGULAR ("tent") profiles, one per
    disk centre, `[nb_pixels]` — the alternative `shape` to `_disk_mass_angle`
    (see `HcReconstruction.use_disks`). `tent(v) = max(0, 1-|v|/r)`: height 1
    at the peak, support half-width `r` (so area `r`, translation-invariant,
    same normalisation convention as `hc_ot_sycl.cpp`'s continuous kernel —
    the only shape it knows how to sweep in closed form, hence `shape` being
    selectable HERE too: it lets a jax run and a sycl run be compared on the
    literal same target shape).

    Purely polynomial in the clipped local coordinate `v` — no sqrt/asin, so
    (unlike `_disk_mass_angle`) no NaN-safe surrogate derivative is needed:
    `P` below is the exact antiderivative of `tent`, built piecewise from
    `v` clipped to `[-r, r]`, continuous and correctly saturating (`P=0` for
    `v<=-r`, `P=r` for `v>=r`) by construction.
    """
    s0 = centers[:, 0] * nx + centers[:, 1] * ny                # [ndisks]
    v = pix_edges[None, :] - s0[:, None]                        # [ndisks, nb_pixels+1]
    vr = jnp.clip(v, -radius, radius)
    left  = vr + vr * vr / (2 * radius) + radius / 2            # antiderivative, v in [-r, 0]
    right = radius / 2 + vr - vr * vr / (2 * radius)            # antiderivative, v in [0, r]
    P = jnp.where(v <= 0, left, right)
    mass = P[:, 1:] - P[:, :-1]                                 # [ndisks, nb_pixels]
    return jnp.sum(mass, axis=0)                                # [nb_pixels]


#: per-disk profile, selected by `HcReconstruction.use_disks`'s `shape`.
MASS_FNS = { "disk": _disk_mass_angle, "triangle": _triangle_mass_angle }


def _ot1d_disks_angle(sino_row, bin_centers_src, mass_tgt, pix_edges):
    """1D OT cost between the FIXED sinogram diracs (`sino_row` weights at
    `bin_centers_src`) and the disk-projected piecewise-constant density
    (`mass_tgt` over `pix_edges`, differentiable w.r.t. the disk centers
    through `mass_tgt`) — SOURCE/TARGET roles flipped relative to
    `_ot1d_angle` (see `models.DiskModel`'s docstring, "ils en ÉCHANGENT les
    rôles").

    A direct port of `sdot.distributions._pure_jax_cost1d.cost_1d_ot` — the
    closed-form pure-Jax path `OtPlan1d` itself uses for a dirac-source vs.
    `Image` pair — since `HcReconstruction` avoids importing loom/sdot (see
    the package docstring). Ported rather than hand re-derived: a from-scratch
    derivation (accumulating one GLOBAL target second-moment plus a per-
    dirac barycenter) agrees on well-behaved inputs but silently disagrees
    from this reference whenever the source carries a NEGATIVE weight (e.g.
    a phantom with a negative-density disk, so a `sino_row` entry < 0) —
    `cost_1d_ot`'s per-matched-window accumulation of the target's 2nd
    moment (`dM2`, not a global constant split off) is what `OtPlan1d`'s own
    C++ kernel agrees with in that case too (a known, matched quirk, see
    `cost_1d_ot`'s own docstring — not "more correct" OT theory, just the
    behaviour this reimplementation must match). Both sides are normalized
    to mass 1 internally.

    FLOAT32 FRAGILITY (found running a dirac+triangle multiscale pipeline via
    `experiments.exp_hc` on a real lung phantom): when `mass_tgt` has many THIN
    near-zero-mass gaps (e.g. late multiscale stages,
    where triangles pack close to their touching radius over an `nb_pixels`-fine grid), the
    `cumsum`/`searchsorted` below can lose enough precision in plain float32 to produce a
    NEGATIVE cost for some matched window — impossible in exact arithmetic (every window's
    contribution is `integral (x-p)^2 dnu(x) >= 0`). Confirmed float32-specific: the exact same
    inputs under `jax.config.update("jax_enable_x64", True)` give a sane, positive cost matching
    the (double-precision) sycl kernel. No mitigation lives HERE (would need call-site
    `jax.config`, out of scope for a pure function) — callers running the disks/triangle model
    at fine resolution should enable jax x64.
    """
    m = mass_tgt.shape[0]
    dw = pix_edges[1] - pix_edges[0]
    w_norm = sino_row / jnp.sum(sino_row)                       # [nb_bins], source mass fractions
    mass_norm = mass_tgt / jnp.maximum(jnp.sum(mass_tgt), 1e-30)  # [nb_pixels], target mass fractions

    C   = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass_norm)])
    M1e = jnp.concatenate([jnp.zeros(1), jnp.cumsum(
        mass_norm / dw * (pix_edges[1:] ** 2 - pix_edges[:-1] ** 2) / 2)])
    M2e = jnp.concatenate([jnp.zeros(1), jnp.cumsum(
        mass_norm / dw * (pix_edges[1:] ** 3 - pix_edges[:-1] ** 3) / 3)])

    order = jnp.argsort(bin_centers_src)
    p_sorted = bin_centers_src[order]
    w_sorted = w_norm[order]
    W = jnp.concatenate([jnp.zeros(1), jnp.cumsum(w_sorted)])

    c_idx = jnp.clip(jnp.searchsorted(C[1:], W, side="left"), 0, m - 1)
    y_c = mass_norm[c_idx] / dw
    e_c = pix_edges[c_idx]
    safe_y = jnp.where(y_c != 0, y_c, 1.0)
    t = jnp.where(y_c != 0, e_c + (W - C[c_idx]) / safe_y, e_c)

    M0_t = C[c_idx] + y_c * (t - e_c)
    M1_t = M1e[c_idx] + y_c * (t ** 2 - e_c ** 2) / 2
    M2_t = M2e[c_idx] + y_c * (t ** 3 - e_c ** 3) / 3

    dM0 = M0_t[1:] - M0_t[:-1]
    dM1 = M1_t[1:] - M1_t[:-1]
    dM2 = M2_t[1:] - M2_t[:-1]

    return jnp.sum(dM2 - 2 * p_sorted * dM1 + p_sorted ** 2 * dM0)


def _disk_cost_angle(centers, radius, nx, ny, sino_row, bin_centers_src, pix_edges, mass_fn):
    mass_tgt = mass_fn(centers, radius, nx, ny, pix_edges)
    return _ot1d_disks_angle(sino_row, bin_centers_src, mass_tgt, pix_edges)


def disk_ot_all_angles(centers, radius, normals, sino_vals, bin_centers_src, pix_edges, mass_fn):
    """Pure Jax OT for all angles, DISKS variant. `centers` `[ndisks, 2]` is
    the only differentiated argument (see `cost.backend`, which wraps this in
    `jax.value_and_grad(argnums=0)`). `mass_fn` (one of `MASS_FNS`'s values)
    is a plain Python callable, closed over by `functools.partial` before
    `vmap` rather than passed as a jax value."""
    cost_angle = functools.partial(_disk_cost_angle, mass_fn=mass_fn)
    costs = jax.vmap(cost_angle, in_axes=(None, None, 0, 0, 0, None, None))(
        centers, radius, normals[:, 0], normals[:, 1], sino_vals, bin_centers_src, pix_edges)
    return costs.sum()
