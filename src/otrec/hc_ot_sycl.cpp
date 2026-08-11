/* hc_ot_sycl.cpp — Standalone SYCL fused OT kernel (cost + gradient, cost only).
   Pure SYCL — no loom/sdot headers.

   Optimised for Apple Silicon M4 (NEON 128-bit SIMD, ARMv8.5-A).

   Exports:
     double hc_ot_cost_grad(points, n, normals, nb_angles, sino_vals, nb_bins,
                            bin_edges, grad);
     double hc_ot_cost(points, n, normals, nb_angles, sino_vals, nb_bins,
                       bin_edges);
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
