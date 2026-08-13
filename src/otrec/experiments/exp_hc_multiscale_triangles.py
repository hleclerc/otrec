"""Interleave the dirac multiscale with a few TRIANGLE-disks LBFGS iterations per stage.

`HcReconstruction.line_search_multiscale` (see its docstring) converges coarse -> fine in the
DIRAC model: converge at `--nb-diracs-init` points, split (x`--multiscale-factor`), reconverge,
repeat up to `--nb-diracs`. This experiment interleaves, via its `stage_callback` hook, a short
DISKS refinement (shape="triangle" -- the profile the SYCL kernel can sweep in closed form, see
`hc_ot_sycl.cpp`) after EVERY stage: `--triangle-max-iter` L-BFGS iterations from that stage's
dirac-converged points, at a radius that SHRINKS as the point count grows (`radius =
triangle_radius_factor * extent / sqrt(n)` -- the same "typical local spacing at n points"
heuristic `HcReconstruction.split`'s own jitter scale uses, so triangles start roughly touching
their neighbours and shrink to match as the cloud gets denser). The refinement FEEDS FORWARD:
its output becomes the base for the next dirac split (`stage_callback` returns the refined
points; `line_search_multiscale` folds its history into the same continuous timeline) -- so the
final result is a genuinely combined dirac+triangle multiscale, not a side-diagnostic.

Run for BOTH `--backend`s (jax, sycl) every time, unconditionally -- the whole point is to
compare them on the exact same pipeline (same triangle shape both sides, see
`HcReconstruction.use_disks`'s `shape` parameter). Results land in separate subdirectories,
`{results_dir}/jax/` and `{results_dir}/sycl/`, each with one JSON (loss/timings history, via
`viz.line_search_compare.save_run`), one comparison PNG, and one HTML animation.

NUMERICAL NOTE (found by actually running this): late multiscale stages pack many triangles at
close to their touching radius (`triangle_radius_factor` is exactly tuned for that), so the jax
disks path's `nb_pixels`-fine target ends up with many THIN near-zero-mass gaps between
neighbouring supports. In plain float32 (jax's default), `_ot1d_disks_angle`'s ~2000-element
`cumsum`/`searchsorted` (see `HcReconstruction._ot1d_disks_angle`) then loses enough precision to
occasionally produce a NEGATIVE cost -- caught here (the final jax loss came out at -1.3 on a real
lung phantom run before this fix), confirmed by re-running with float64 (the anomaly vanishes and
jax's loss matches sycl's to a few percent, as `hc_disks_jax_triangle_matches_sycl_triangle`
already checks at a coarser scale). The sycl kernel is unaffected (its sweep is double-precision
internally regardless of the ctypes float32 I/O) -- this is a jax-only, float32-only fragility.
Mitigated HERE (an experiment script, not a hot path) by enabling jax's x64 mode process-wide;
see [[HcReconstruction disks model]] for the write-up.

Run via:
    ./run experiment hc_multiscale_triangles
    ./run experiment hc_multiscale_triangles --nb-diracs=5000 --triangle-max-iter=15
"""
import time

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from otrec.HcReconstruction import HcReconstruction
from otrec.viz.line_search_compare import save_run, load_runs, plot_comparison
from loom.cli import experiment, Param


def _make_stage_callback(hc: HcReconstruction, extent: float, triangle_max_iter: int,
                         triangle_radius_factor: float, lbfgs_memory: int, verbose: bool):
    """Switch `hc` to the DISKS/triangle model, run a short L-BFGS refinement from that
    stage's dirac points, switch back to diracs, and return the refined cloud -- the
    `stage_callback` `line_search_multiscale` folds into its continuous timeline.
    """
    def stage_callback(stage, n, points):
        radius = triangle_radius_factor * extent / np.sqrt(n)
        hc.use_disks(radius=radius, shape="triangle")
        if verbose:
            print(f"  -- stage {stage} triangle refine: n={n}, radius={radius:.4f} --")
        refined = hc.line_search_lbfgs(points, max_iter=triangle_max_iter,
                                       memory=lbfgs_memory, verbose=verbose, ftol=0.0)
        hc.use_diracs()
        return refined
    return stage_callback


if p := experiment("hc_multiscale_triangles",
    results_dir = Param("tmp/hc_multiscale_triangles",
                        help = "dir accumulating {jax,sycl}/ subdirs, each one JSON + one HTML"),
    nb_alveoli = Param(50, help = "Number of alveoli (holes)"),
    nb_diracs = Param(2000, help = "Target (finest-scale) number of Dirac masses"),
    nb_diracs_init = Param(100, help = "Coarsest multiscale stage's dirac count"),
    multiscale_factor = Param(4, help = "Dirac count multiplied by this each multiscale stage"),
    plateau_frac = Param(0.01, help = "Non-final dirac stages stop once a step's gain drops "
                                    "below this fraction of that stage's first gain"),
    nb_angles = Param(600, help = "Nb angles"),
    method = Param("lbfgs", help = "dirac-stage line-search method"),
    max_iter = Param(20, help = "Max dirac line-search iterations per stage"),
    lbfgs_memory = Param(10, help = "L-BFGS curvature pairs kept (dirac stage AND triangle refine)"),
    triangle_max_iter = Param(10, help = "L-BFGS iterations of the triangle refinement, per stage"),
    triangle_radius_factor = Param(0.5, help = "triangle radius = this * extent / sqrt(n) -- "
                                             "shrinks every stage as n grows"),
    seed = Param(1, help = "initial point cloud seed"),
    print_steps = Param(False, help = "print every line-search step (dirac AND triangle)"),
):
    for backend in ("jax", "sycl"):
        print(f"\n{'='*20} backend={backend} {'='*20}")
        out_dir = f"{p.results_dir}/{backend}"

        hc, lobes, alveoli = HcReconstruction.make_lung_phantom(
            nb_alveoli = p.nb_alveoli, backend = backend, nb_angles = p.nb_angles,
            record = True)

        stage_callback = _make_stage_callback(
            hc, hc.extent, p.triangle_max_iter, p.triangle_radius_factor,
            p.lbfgs_memory, p.print_steps)

        t0 = time.time()
        hc.line_search_multiscale(
            method = p.method, nb_diracs_final = p.nb_diracs, nb_diracs_init = p.nb_diracs_init,
            factor = p.multiscale_factor, seed = p.seed, plateau_frac = p.plateau_frac,
            max_iter = p.max_iter, lbfgs_memory = p.lbfgs_memory,
            stage_callback = stage_callback)
        dt = time.time() - t0
        print(f"{backend}: {dt:.2f}s total, final loss={hc.loss_history[-1]['cost']:.6f}")

        meta = dict(method = p.method, backend = backend, nb_alveoli = p.nb_alveoli,
                   nb_diracs = p.nb_diracs, nb_angles = p.nb_angles, max_iter = p.max_iter,
                   triangle_max_iter = p.triangle_max_iter,
                   triangle_radius_factor = p.triangle_radius_factor, seed = p.seed)
        out_file = save_run(out_dir, "multiscale_triangles", meta, hc.loss_history, hc.timings)
        out_html = f"{out_dir}/multiscale_triangles.html"
        # `marker="triangle"` : the last stages are triangle-refined disk centres, not bare
        # diracs -- draw them as such (see `viz.points_html.export_positions_html`'s `marker`).
        hc.export_html(out_html, title = f"multiscale + triangles ({backend})", marker = "triangle")

        out_plot = f"{out_dir}/comparison.png"
        plot_comparison(load_runs(out_dir), out_plot)
        print(f"  saved -> {out_file}")
        print(f"  saved -> {out_html}")
        print(f"  saved -> {out_plot}")
