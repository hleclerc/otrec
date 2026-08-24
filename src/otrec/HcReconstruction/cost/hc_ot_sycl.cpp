/* hc_ot_sycl.cpp — Standalone SYCL fused OT kernel (cost + gradient, cost only).
   Pure SYCL — no loom/sdot headers.

   Optimised for Apple Silicon M4 (NEON 128-bit SIMD, ARMv8.5-A).

   Exports:
     double hc_ot_cost_grad(points, n, normals, nb_angles, sino_vals, nb_bins,
                            bin_edges, grad);
     double hc_ot_cost(points, n, normals, nb_angles, sino_vals, nb_bins,
                       bin_edges);
     double hc_ot_disks_cost_grad(centers, ndisks, normals, nb_angles,
                                  sino_vals, nb_bins, bin_centers, radius, grad);
     double hc_ot_disks_cost(centers, ndisks, normals, nb_angles,
                             sino_vals, nb_bins, bin_centers, radius);

   The `disks` pair is a CONTINUOUS counterpart of `models.DiskModel`/
   `disks.DiskProjector` (which discretises the disk projection onto a
   `nb_pixels` grid before running OT): here each disk's Radon profile — the
   parabolic chord `2*sqrt(r^2-u^2)` — is replaced by a TRIANGULAR ("tent")
   profile of the same support width `2r` (peak height 1 at the projected
   centre, 0 at `+-r`). A sum of possibly-overlapping tents is exactly
   piecewise-LINEAR, so its cumulative mass/first/second moment against the
   sinogram are closed-form polynomials — letting the whole projection be
   swept EXACTLY, with no pixel grid and no discretisation error, the same
   way the disk profile's true parabolic chord could not be (its quantile
   function has no closed form once several disks overlap). See
   `disks_cost_grad_impl`'s docstring for the algorithm and the gradient's
   envelope-theorem derivation.
*/

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>

using SI = std::int64_t;

namespace {

/* ---- sort key (projected position → sortable integer) ----------------- */

struct SortKey {
    float proj;
    SI    idx;
};

inline bool operator<(SortKey a, SortKey b) {
    // standard float comparison is fine — NaNs shouldn't appear in projections
    return a.proj < b.proj;
}

/* ---- disks: tent-sum breakpoints ---------------------------------------
   Each disk contributes 3 breakpoints (left edge, peak, right edge) to the
   sorted event list that reconstructs the combined piecewise-linear density
   `f(u) = sum_i tent_i(u)` — see `disks_cost_grad_impl`'s docstring.       */

struct DiskEvent {
    float   pos;      // breakpoint position along the detector axis
    float   dslope;   // slope CHANGE of f(u) at this breakpoint
    int32_t disk;      // which disk this breakpoint belongs to
    int32_t kind;      // 0 = left edge, 1 = peak, 2 = right edge
};

inline bool operator<(DiskEvent a, DiskEvent b) { return a.pos < b.pos; }

}  // anonymous namespace


extern "C" {

double hc_ot_cost_grad(
    const float *points,       // [n, 2]
    int n,
    const float *normals,      // [nb_angles, 2]
    int nb_angles,
    const float *sino_vals,    // [nb_angles, nb_bins]
    int nb_bins,
    const float *bin_edges,    // [nb_bins + 1]
    float *grad                // [n, 2] output
) {
    sycl::queue q(sycl::cpu_selector_v);

    const float dw = bin_edges[1] - bin_edges[0];
    const float inv_12 = 1.0f / 12.0f;

    // pre-compute normalised bin masses per angle
    float   *bin_mass   = sycl::malloc_shared<float>(nb_angles * nb_bins, q);
    SortKey *keys_buf   = sycl::malloc_device<SortKey>(nb_angles * n, q);
    float   *proj_buf   = sycl::malloc_device<float>(nb_angles * n, q);
    float   *cost_angle = sycl::malloc_shared<float>(nb_angles, q);
    float   *grad_dev   = sycl::malloc_device<float>(n * 2, q);

    // normalise bin masses
    {
        float *norms = sycl::malloc_shared<float>(nb_angles, q);
        q.parallel_for(nb_angles, [=](sycl::id<1> a) {
            float s = 0;
#pragma clang loop vectorize(enable) interleave(enable)
            for (int j = 0; j < nb_bins; j++)
                s += sino_vals[a * nb_bins + j];
            norms[a] = s;
        }).wait();
        q.parallel_for(sycl::range<1>(nb_angles), [=](sycl::id<1> a) {
            float inv = 1.0f / norms[a];
#pragma clang loop vectorize(enable)
            for (int j = 0; j < nb_bins; j++)
                bin_mass[a * nb_bins + j] = sino_vals[a * nb_bins + j] * inv;
        }).wait();
        sycl::free(norms, q);
    }

    q.memset(grad_dev, 0, n * 2 * sizeof(float));
    q.fill(cost_angle, 0.0f, nb_angles).wait();

    // main kernel: one work-item per angle
    q.parallel_for(sycl::range<1>(nb_angles), [=](sycl::id<1> idx) {
        int a = idx[0];
        float nx = normals[a * 2];
        float ny = normals[a * 2 + 1];

        float   *proj = proj_buf + a * n;
        SortKey *keys = keys_buf + a * n;

        // project + fill sort keys (NEON auto-vectorized)
#pragma clang loop vectorize(enable) interleave(enable)
        for (SI i = 0; i < n; i++) {
            proj[i]     = points[i * 2] * nx + points[i * 2 + 1] * ny;
            keys[i].proj = proj[i];
            keys[i].idx  = i;
        }

        // std::sort (Introsort on M4 — ~n log n comparisons, good cache)
        std::sort(keys, keys + n);

        // sweep
        float w = 1.0f / float(n);
        const float *mass_a = bin_mass + a * nb_bins;
        const float *edges  = bin_edges;

        int j = 0;
        float consumed = 0;
        float local_cost = 0;

        for (SI k = 0; k < n; k++) {
            SI    di = keys[k].idx;
            float p  = proj[di];
            float need = w;
            float moment = 0;

            while (need > 1e-15f && j < nb_bins) {
                float bm   = mass_a[j];
                float avail = bm - consumed;
                float take = need < avail ? need : avail;

                if (take > 0) {
                    float inv_bm = 1.0f / bm;
                    float dw_inv_bm = dw * inv_bm;
                    float slice_c = edges[j] + consumed * dw_inv_bm
                        + take * dw_inv_bm * 0.5f;
                    float slice_w = take * dw_inv_bm;
                    float diff    = slice_c - p;

                    moment += take * slice_c;
                    local_cost += take * (diff * diff
                                          + slice_w * slice_w * inv_12);

                    need     -= take;
                    consumed += take;
                }

                if (consumed >= bm - 1e-12f) {
                    j++;
                    consumed = 0;
                }
            }

            float bary = moment / w;
            float grad_s = 2.0f * w * (p - bary);

            auto atm_x = sycl::atomic_ref<
                float, sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>(grad_dev[di * 2]);
            auto atm_y = sycl::atomic_ref<
                float, sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>(grad_dev[di * 2 + 1]);

            atm_x.fetch_add(grad_s * nx);
            atm_y.fetch_add(grad_s * ny);
        }

        cost_angle[a] = local_cost;
    }).wait();

    q.memcpy(grad, grad_dev, n * 2 * sizeof(float)).wait();

    double total_cost = 0;
    for (int a = 0; a < nb_angles; a++)
        total_cost += double(cost_angle[a]);

    sycl::free(bin_mass, q);
    sycl::free(keys_buf, q);
    sycl::free(proj_buf, q);
    sycl::free(cost_angle, q);
    sycl::free(grad_dev, q);

    return total_cost;
}


double hc_ot_cost(
    const float *points, int n,
    const float *normals, int nb_angles,
    const float *sino_vals, int nb_bins,
    const float *bin_edges
) {
    sycl::queue q(sycl::cpu_selector_v);

    const float dw = bin_edges[1] - bin_edges[0];
    const float inv_12 = 1.0f / 12.0f;

    float   *bin_mass   = sycl::malloc_shared<float>(nb_angles * nb_bins, q);
    SortKey *keys_buf   = sycl::malloc_device<SortKey>(nb_angles * n, q);
    float   *proj_buf   = sycl::malloc_device<float>(nb_angles * n, q);
    float   *cost_angle = sycl::malloc_shared<float>(nb_angles, q);

    {
        float *norms = sycl::malloc_shared<float>(nb_angles, q);
        q.parallel_for(nb_angles, [=](sycl::id<1> a) {
            float s = 0;
#pragma clang loop vectorize(enable) interleave(enable)
            for (int j = 0; j < nb_bins; j++)
                s += sino_vals[a * nb_bins + j];
            norms[a] = s;
        }).wait();
        q.parallel_for(sycl::range<1>(nb_angles), [=](sycl::id<1> a) {
            float inv = 1.0f / norms[a];
#pragma clang loop vectorize(enable)
            for (int j = 0; j < nb_bins; j++)
                bin_mass[a * nb_bins + j] = sino_vals[a * nb_bins + j] * inv;
        }).wait();
        sycl::free(norms, q);
    }

    q.fill(cost_angle, 0.0f, nb_angles).wait();

    q.parallel_for(sycl::range<1>(nb_angles), [=](sycl::id<1> idx) {
        int a = idx[0];
        float nx = normals[a * 2];
        float ny = normals[a * 2 + 1];

        float   *proj = proj_buf + a * n;
        SortKey *keys = keys_buf + a * n;

#pragma clang loop vectorize(enable) interleave(enable)
        for (SI i = 0; i < n; i++) {
            proj[i]     = points[i * 2] * nx + points[i * 2 + 1] * ny;
            keys[i].proj = proj[i];
            keys[i].idx  = i;
        }

        std::sort(keys, keys + n);

        float w = 1.0f / float(n);
        const float *mass_a = bin_mass + a * nb_bins;
        const float *edges  = bin_edges;

        int j = 0;
        float consumed = 0;
        float local_cost = 0;

        for (SI k = 0; k < n; k++) {
            SI    di = keys[k].idx;
            float p  = proj[di];
            float need = w;

            while (need > 1e-15f && j < nb_bins) {
                float bm    = mass_a[j];
                float avail = bm - consumed;
                float take = need < avail ? need : avail;

                if (take > 0) {
                    float inv_bm = 1.0f / bm;
                    float dw_inv_bm = dw * inv_bm;
                    float slice_c = edges[j] + consumed * dw_inv_bm
                        + take * dw_inv_bm * 0.5f;
                    float slice_w = take * dw_inv_bm;
                    float diff    = slice_c - p;

                    local_cost += take * (diff * diff
                                          + slice_w * slice_w * inv_12);

                    need     -= take;
                    consumed += take;
                }

                if (consumed >= bm - 1e-12f) {
                    j++;
                    consumed = 0;
                }
            }
        }

        cost_angle[a] = local_cost;
    }).wait();

    double total_cost = 0;
    for (int a = 0; a < nb_angles; a++)
        total_cost += double(cost_angle[a]);

    sycl::free(bin_mass, q);
    sycl::free(keys_buf, q);
    sycl::free(proj_buf, q);
    sycl::free(cost_angle, q);

    return total_cost;
}

} // extern "C"


/* ---- disks: continuous fused OT (cost + gradient, cost only) -------------

   `disks_cost_grad_impl<WITH_GRAD>` — shared by `hc_ot_disks_cost_grad` and
   `hc_ot_disks_cost` (a compile-time flag instead of the hand-duplicated
   pair above: the disks sweep is involved enough that keeping ONE
   implementation was worth the `if constexpr` branches, unlike the shorter
   dirac sweep this file duplicates verbatim).

   ---- geometry, one angle ----

   Each disk `i` (all sharing the same fixed `radius`) projects to a tent
   `tent_i(u) = max(0, 1 - |u - s0_i| / r)`, `s0_i = centers[i] . normal_a` —
   support `[s0_i - r, s0_i + r]`, area exactly `r` (independent of `s0_i`,
   as long as the disk isn't clipped by the detector, the same assumption
   `Sinogram.add_disk`/`disks.DiskProjector` already make). The combined
   density `f(u) = sum_i tent_i(u)` is therefore piecewise-LINEAR with
   breakpoints at every `{s0_i-r, s0_i, s0_i+r}` (3 per disk): sorting all
   `3*ndisks` of them and walking through in order, applying each one's
   slope DELTA (`+1/r` at an edge, `-2/r` at a peak — a disk's tent rising
   then falling), reconstructs `f` exactly — the standard sweep for a sum of
   tents/triangles. Total target mass `M = ndisks * r` is therefore a
   CONSTANT, independent of the centers — no need to differentiate a
   normalisation factor.

   ---- cost, as an integral over the target ----

   For 1D OT with quadratic cost between two mass-1 measures, the optimal
   coupling is the MONOTONE (quantile-matching) one: point `u` of the target
   is matched to the source position `T(u)` at the same cumulative-mass
   level. So `cost = integral (u - T(u))^2 dnu(u) = (1/M) integral g(u) f(u)
   du`, `g(u) = (u - T(u))^2`. `T(u)` is piecewise CONSTANT (one value per
   matched source dirac, i.e. one measured-sinogram bin) — exactly the
   sweep `hc_ot_cost_grad` above already does, just with the SOURCE (sino
   bins) and TARGET (density) roles exchanged: here we walk the (sorted,
   piecewise-linear) TARGET breakpoints outward, consuming the (already
   sorted by construction — `bin_centers` is a regular grid) SOURCE's
   cumulative weight budget as we go. `g(u)*f(u)` integrates in closed form
   over each micro-piece (`f` affine, `g` quadratic — an elementary quartic
   polynomial, `INTEGRAL_QUARTIC` below), accumulated into a running
   cumulative integral `G`, recorded at every one of the `3*ndisks`
   breakpoints.

   ---- gradient, via the quantile-coupling formula ----

   `T` is piecewise constant, so it is tempting (and WRONG — an earlier
   version of this kernel did exactly this and silently produced a wrong
   gradient) to guess `d(cost)/df(u) = g(u)/M` by analogy with
   `hc_ot_cost_grad`'s `grad_s = 2w(s-bary)` and integrate `g(u)/M` against
   `d(tent_i)/d(s0_i)`. That reuses the FORWARD cost's `f`-weighted
   accumulator for a quantity that isn't `f`-weighted at all: `cost` is a
   functional of `nu`'s CDF (through where `T` switches), not a pointwise
   functional of the density `f(u)` itself, so there is no such `dCost/df`.

   The right route is the quantile representation `W2^2 = integral_0^1
   (Q_mu(t)-Q_nu(t))^2 dt`. Differentiating the TARGET's quantile function
   `Q_nu` w.r.t. `s0_i` at fixed quantile level `t` (inverse-function rule,
   `dQ_nu/dtheta|_t = -(dF/dtheta)(u) / f(u)`, `u=Q_nu(t)`), substituting
   back `t=F_nu(u)`, `dt=f(u)/M du`, and using that `d(tent_i)/d(s0_i)`
   integrates to a simple explicit ramp `G_i(u)` (0 outside `tent_i`'s
   support, `-1` at `u=s0_i`, linear in between) gives, after one
   integration by parts to swap `G_i` for its own antiderivative (so the
   `1/f(u)` cancels and a plain, num­erically well-behaved sweep remains):

       d(cost)/d(s0_i) = -(2/(M*r)) *
           ( [Shat(s0_i+r) - Shat(s0_i)] - [Shat(s0_i) - Shat(s0_i-r)] )

   where `Shat` is the DOUBLE running antiderivative of `h(u) = T(u) - u`
   (first `S(u) = integral h`, then `Shat(u) = integral S`) — piecewise
   cubic since `h` is affine on each matched micro-piece, accumulated by
   the SAME sweep as `cost_accum` above (just a different running quantity,
   NOT weighted by `f`). `s0_i-r`, `s0_i`, `s0_i+r` are exactly disk `i`'s
   own 3 breakpoints — their sorted index was recorded while sorting, so
   each disk's gradient is an O(1) lookup into `Shat`, no re-walk.
   `d(cost)/d(center_i) = d(cost)/d(s0_i) * normal_a`, atomically
   accumulated across angles like the dirac kernel's `grad`. Verified
   against finite differences (see the otrec test suite) — the closed form
   above was derived and checked numerically before being trusted here, the
   earlier `g(u)/M` guess having already burned that trust once.

   Negative sinogram weights (a phantom with a negative-density disk) can
   make the source's cumulative mass boundaries non-monotonic; `avail` is
   clamped to `>= 0` so the sweep still terminates with a well-defined
   (if not perfectly OT-theoretic) result instead of stalling or producing
   NaN — the JAX disks path documents the same accepted quirk, inherited
   from `sdot`'s own reference implementation, see `_ot1d_disks_angle`.   */

namespace {

inline double integral_quartic(double d, double f0, double s, double X) {
    // integral_0^X (d+x)^2 * (f0+s*x) dx, expanded and integrated term by term.
    return d*d*f0*X + 0.5*d*d*s*X*X + d*f0*X*X
         + (2.0/3.0)*d*s*X*X*X + (1.0/3.0)*f0*X*X*X + 0.25*s*X*X*X*X;
}

template <bool WITH_GRAD>
double disks_cost_grad_impl(
    const float *centers, int ndisks,
    const float *normals, int nb_angles,
    const float *sino_vals, int nb_bins,
    const float *bin_centers,
    float radius,
    float *grad
) {
    sycl::queue q(sycl::cpu_selector_v);

    const int ne = 3 * ndisks;                          // breakpoints per angle
    const double M = double(ndisks) * double(radius);   // constant total target mass

    float     *bin_mass  = sycl::malloc_shared<float>(nb_angles * nb_bins, q);
    DiskEvent *events    = sycl::malloc_device<DiskEvent>(nb_angles * ne, q);
    double    *Shat_at   = sycl::malloc_device<double>(nb_angles * ne, q);
    int32_t   *evt_left  = sycl::malloc_device<int32_t>(nb_angles * ndisks, q);
    int32_t   *evt_peak  = sycl::malloc_device<int32_t>(nb_angles * ndisks, q);
    int32_t   *evt_right = sycl::malloc_device<int32_t>(nb_angles * ndisks, q);
    float     *cost_angle = sycl::malloc_shared<float>(nb_angles, q);
    float     *grad_dev = nullptr;
    if constexpr (WITH_GRAD) {
        grad_dev = sycl::malloc_device<float>(ndisks * 2, q);
        q.memset(grad_dev, 0, ndisks * 2 * sizeof(float)).wait();
    }

    // normalise sino_vals -> bin_mass (source mass FRACTIONS per angle)
    {
        float *norms = sycl::malloc_shared<float>(nb_angles, q);
        q.parallel_for(nb_angles, [=](sycl::id<1> a) {
            float s = 0;
#pragma clang loop vectorize(enable) interleave(enable)
            for (int j = 0; j < nb_bins; j++)
                s += sino_vals[a * nb_bins + j];
            norms[a] = s;
        }).wait();
        q.parallel_for(sycl::range<1>(nb_angles), [=](sycl::id<1> a) {
            float inv = 1.0f / norms[a];
#pragma clang loop vectorize(enable)
            for (int j = 0; j < nb_bins; j++)
                bin_mass[a * nb_bins + j] = sino_vals[a * nb_bins + j] * inv;
        }).wait();
        sycl::free(norms, q);
    }

    q.fill(cost_angle, 0.0f, nb_angles).wait();

    q.parallel_for(sycl::range<1>(nb_angles), [=](sycl::id<1> idx) {
        int a = idx[0];
        float nx = normals[a * 2];
        float ny = normals[a * 2 + 1];

        DiskEvent *evts    = events + a * ne;
        double    *Shat    = Shat_at + a * ne;
        int32_t   *e_left  = evt_left + a * ndisks;
        int32_t   *e_peak  = evt_peak + a * ndisks;
        int32_t   *e_right = evt_right + a * ndisks;
        const float *mass_a = bin_mass + a * nb_bins;

        const float inv_r = 1.0f / radius;

        for (int i = 0; i < ndisks; i++) {
            float s0 = centers[i * 2] * nx + centers[i * 2 + 1] * ny;
            evts[3*i]     = DiskEvent{ s0 - radius,  inv_r,        i, 0 };
            evts[3*i + 1] = DiskEvent{ s0,          -2.0f * inv_r, i, 1 };
            evts[3*i + 2] = DiskEvent{ s0 + radius,  inv_r,        i, 2 };
        }

        std::sort(evts, evts + ne);

        for (int i = 0; i < ne; i++) {
            if (evts[i].kind == 0)      e_left[evts[i].disk]  = i;
            else if (evts[i].kind == 1) e_peak[evts[i].disk]  = i;
            else                        e_right[evts[i].disk] = i;
        }

        double slope      = 0.0;
        double f_val       = 0.0;   // f(u) just before the FIRST breakpoint is 0
        double cost_accum  = 0.0;
        double mass_accum  = 0.0;
        int    k_src       = 0;
        double Q_k         = double(mass_a[0]) * M;   // cumulative source mass boundary

        // S_run/Shat_run: running (single/double) antiderivative of the
        // residual h(u) = T(u) - u, for the GRADIENT (see docstring) --
        // unrelated to cost_accum (the forward cost's OWN, f-weighted,
        // accumulator).
        double S_run    = 0.0;
        double Shat_run = 0.0;

        Shat[0] = 0.0;
        slope += evts[0].dslope;   // active on [evts[0], evts[1])

        for (int i = 0; i + 1 < ne; i++) {
            double L    = double(evts[i + 1].pos) - double(evts[i].pos);
            double u_a  = double(evts[i].pos);
            double x0   = 0.0;
            double f_here = f_val;

            while (x0 < L - 1e-9) {
                double remaining = L - x0;
                double mass_full = f_here * remaining + 0.5 * slope * remaining * remaining;
                double avail = Q_k - mass_accum;
                if (avail < 0.0) avail = 0.0;

                double x_take;
                bool advance;
                if (mass_full <= avail || k_src == nb_bins - 1) {
                    x_take = remaining;
                    advance = false;
                } else if (sycl::fabs(slope) < 1e-12) {
                    x_take = f_here > 1e-20 ? avail / f_here : remaining;
                    advance = true;
                } else {
                    double disc = f_here * f_here + 2.0 * slope * avail;
                    if (disc < 0.0) disc = 0.0;
                    x_take = (-f_here + sycl::sqrt(disc)) / slope;
                    advance = true;
                }
                if (x_take < 0.0) x_take = 0.0;
                if (x_take > remaining) x_take = remaining;

                double d = (u_a + x0) - double(bin_centers[k_src]);
                cost_accum += integral_quartic(d, f_here, slope, x_take);

                // h(x) = T_k - (u_a+x0+x) = -d - x, local coord x in [0, x_take] --
                // integrate it (S) and its own running integral (Shat), in that order
                // (Shat's update uses S_run from BEFORE this piece).
                double delta_Shat = S_run * x_take - 0.5 * d * x_take * x_take
                                   - x_take * x_take * x_take / 6.0;
                double delta_S = -d * x_take - 0.5 * x_take * x_take;
                Shat_run += delta_Shat;
                S_run += delta_S;

                double piece_mass = f_here * x_take + 0.5 * slope * x_take * x_take;
                mass_accum += piece_mass;
                f_here += slope * x_take;
                x0 += x_take;

                if (advance && k_src < nb_bins - 1) {
                    k_src++;
                    Q_k += double(mass_a[k_src]) * M;
                }
            }

            f_val = f_here;
            Shat[i + 1] = Shat_run;
            slope += evts[i + 1].dslope;
        }

        cost_angle[a] = float(cost_accum / M);

        if constexpr (WITH_GRAD) {
            double coeff = -2.0 / (M * double(radius));
            for (int i = 0; i < ndisks; i++) {
                double right_diff = Shat[e_right[i]] - Shat[e_peak[i]];
                double left_diff  = Shat[e_peak[i]]  - Shat[e_left[i]];
                double dcost_ds0 = coeff * (right_diff - left_diff);

                auto atm_x = sycl::atomic_ref<
                    float, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(grad_dev[i * 2]);
                auto atm_y = sycl::atomic_ref<
                    float, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(grad_dev[i * 2 + 1]);
                atm_x.fetch_add(float(dcost_ds0 * nx));
                atm_y.fetch_add(float(dcost_ds0 * ny));
            }
        }
    }).wait();

    double total_cost = 0;
    for (int a = 0; a < nb_angles; a++)
        total_cost += double(cost_angle[a]);

    if constexpr (WITH_GRAD) {
        q.memcpy(grad, grad_dev, ndisks * 2 * sizeof(float)).wait();
        sycl::free(grad_dev, q);
    }

    sycl::free(bin_mass, q);
    sycl::free(events, q);
    sycl::free(Shat_at, q);
    sycl::free(evt_left, q);
    sycl::free(evt_peak, q);
    sycl::free(evt_right, q);
    sycl::free(cost_angle, q);

    return total_cost;
}

} // anonymous namespace


extern "C" {

double hc_ot_disks_cost_grad(
    const float *centers, int ndisks,
    const float *normals, int nb_angles,
    const float *sino_vals, int nb_bins,
    const float *bin_centers,
    float radius,
    float *grad
) {
    return disks_cost_grad_impl<true>(centers, ndisks, normals, nb_angles,
                                       sino_vals, nb_bins, bin_centers, radius, grad);
}

double hc_ot_disks_cost(
    const float *centers, int ndisks,
    const float *normals, int nb_angles,
    const float *sino_vals, int nb_bins,
    const float *bin_centers,
    float radius
) {
    return disks_cost_grad_impl<false>(centers, ndisks, normals, nb_angles,
                                        sino_vals, nb_bins, bin_centers, radius, nullptr);
}

} // extern "C"
