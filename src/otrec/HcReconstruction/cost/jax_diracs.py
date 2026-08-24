"""Pure Jax 1D OT cost+gradient for the DIRACS model — one angle, then vmapped.

Points ARE equal-mass diracs (the moving SOURCE); the sinogram is the fixed
piecewise-constant TARGET. See `jax_disks.py` for the model where these roles
flip.
"""
import jax
import jax.numpy as jnp


def _ot1d_angle(proj, bin_mass, bin_edges):
    """1D OT cost + gradient, fully parallel (no scan, no fori_loop).

    Uses the integrated quantile function M(q) = ∫₀^q Q(t) dt of the target,
    evaluated at dirac quantile boundaries via searchsorted + prefix sums.
    Barycenter i = (M((i+1)/n) − M(i/n)) / w  — all vectorised.
    """
    n = proj.shape[0]
    m = bin_mass.shape[0]
    w = 1.0 / n
    dw = bin_edges[1] - bin_edges[0]

    order = jnp.argsort(proj)
    s = proj[order]

    # cumulative target mass at bin boundaries
    cum     = jnp.cumsum(bin_mass)                           # [m]
    cum_sta = jnp.concatenate([jnp.zeros(1), cum[:-1]])      # [m]

    # prefix sum of M at bin boundaries:  M(cum[j]) = Σ_{k<j} bin_mass[k]·center[k]
    bin_center = bin_edges[:-1] + dw / 2
    prefix_M = jnp.concatenate([
        jnp.zeros(1), jnp.cumsum(bin_mass * bin_center)[:-1]])  # [m]

    # ---- M(q) = prefix_M[j] + bin_mass[j]·(edges[j]·f + dw·f²/2) ----
    #   where j = searchsorted(cum, q),  f = (q − cum_sta[j]) / bin_mass[j]
    def M_at(q):
        j = jnp.clip(jnp.searchsorted(cum, q, side='right'), 0, m - 1)
        bm = bin_mass[j]
        f = jnp.where(bm > 0, (q - cum_sta[j]) / bm, 0.0)
        return prefix_M[j] + bm * (bin_edges[j] * f + dw * f * f / 2)

    # dirac quantile boundaries
    i = jnp.arange(n)
    q0 = i * w
    q1 = q0 + w

    M0 = M_at(q0)
    M1 = M_at(q1)
    bary = (M1 - M0) / w                               # [n]

    grad_sorted = 2.0 * w * (s - bary)                 # [n]
    grad = grad_sorted[jnp.argsort(order)]

    # cost = W2²(μ, ν) = w·Σ s_i² − 2w·Σ s_i·bary_i + total_2nd_moment(ν)
    w2_nu = jnp.sum(bin_mass * bin_center ** 2) + dw * dw / 12
    cost = w * jnp.sum(s ** 2) - 2 * w * jnp.sum(s * bary) + w2_nu

    return cost, grad


@jax.jit
def ot_all_angles(points, normals, sino_vals, bin_edges):
    """Pure Jax OT for all angles.  vmapped over angles."""
    proj = points @ normals.T                            # [n, A]
    proj = proj.T                                        # [A, n]
    total = sino_vals.sum(axis=1, keepdims=True)         # [A, 1]
    bin_mass = sino_vals / total                         # [A, m]
    costs, grads_s = jax.vmap(_ot1d_angle, in_axes=(0, 0, None))(
        proj, bin_mass, bin_edges)
    return costs.sum(), grads_s.T @ normals              # [n, A] @ [A, 2]
