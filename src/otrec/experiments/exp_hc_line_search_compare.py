"""Accumulate & compare `HcReconstruction` line-search methods over separate runs.

Each run executes ONE method (--method=gd|pr|lbfgs|quad2d|gquad2d|grid2d|grid3d)
via `HcReconstruction.line_search_multiscale`: coarse -> fine, starting at
`--nb-diracs-init` (~100) points and repeatedly splitting (x`--multiscale-factor`)
up to the target `--nb-diracs`. Intermediate stages stop as soon as they
PLATEAU (`--plateau-frac`, relative to that stage's first iteration's gain —
see `HcReconstruction.line_search_multiscale`) instead of running to
`--max-iter`; only the final (target-resolution) stage runs to full
convergence. This is systematic on purpose: a uniform start at the target
count forces the optimizer to also solve the global arrangement, which is
where a plain gradient direction is at its worst (successive steps barely
change direction — see the exported animation, and the comparison plot's
"direction colinearity" panel, which exists specifically to quantify that).

Each run saves its behaviour under a NAME (--name, defaults to --method) into
`--results-dir` (one JSON + one HTML animation per name, both overwritten on
rerun — the point when you improve that method), then reloads every `*.json`
in that directory and redraws the comparison plot from all of them. So
methods accumulate across sessions: run "gd" today, "pr" tomorrow, and the
plot keeps both without re-running either.

"grid2d" is the fine-2D-grid reference/oracle (see `HcReconstruction.line_search_grid2d`)
— it always takes the true best combined (gradient, previous-direction) step, so
its curve is an upper bound on how much a combined-direction method like "pr" could
possibly gain. "grid3d" goes one further, adding the *second*-previous direction
as a third search axis — an upper bound on grid2d itself, i.e. is a third
direction worth chasing at all. Both are excluded from the loss-vs-time panel
(see `viz.line_search_compare._SLOW_METHODS`) since they're orders of magnitude
slower than any production method; run them once and let them sit in the
results directory.

"quad2d" is the cheap production-viable version of the same idea: instead of
grid2d's O(n²) grid + Nelder-Mead refine, it fits a paraboloid to 6 points
(cross + one diagonal corner) and jumps straight to its analytic vertex —
5 extra evaluations/step, same cost ballpark as "gd"/"pr". Its linear terms
use the gradient's EXACT directional derivatives (`bx = -g·g`, `by = g·d_prev`,
free — no extra evaluation) instead of a finite-difference estimate.
"gquad2d" fits the SAME 6-point paraboloid, but its second axis comes from a
gradient evaluated during the bracket phase (see
`HcReconstruction.line_search_gquad2d`) instead of the previous accepted step
— so, unlike "quad2d", it already has a real second direction on step 0.

"lbfgs" is L-BFGS (two-loop recursion over the last `--lbfgs-memory`
position/gradient-delta pairs) for the DIRECTION, using the same
bracket+parabola step-size selection as every other method here — so it
differs from the others only in direction choice, not in how far it steps
along it.

Every method's step-size selection also reuses the gradient it already has
in hand: `HcReconstruction._parabola_vertex`'s 3-point fit becomes a
zero-extra-cost Hermite CUBIC when given the analytic directional derivative
at t=0 (`-g·s`), and `_quad2d_vertex`'s 2D fit uses the analytic gradient
projections instead of finite-difference estimates for its linear terms —
same idea (a polynomial fit that uses derivative info already available for
free) applied everywhere a polynomial gets fit in this file.

Run via:
    ./run experiment hc_line_search_compare --method=gd
    ./run experiment hc_line_search_compare --method=pr
    ./run experiment hc_line_search_compare --method=lbfgs
    ./run experiment hc_line_search_compare --method=quad2d
    ./run experiment hc_line_search_compare --method=gquad2d
    ./run experiment hc_line_search_compare --method=grid2d --max-iter=15
    ./run experiment hc_line_search_compare --method=grid3d --max-iter=15 --n-grid-3d=7
    ./run experiment hc_line_search_compare --method=pr --name=pr_v2   # keep both
    ./run experiment hc_line_search_compare --method=gd,pr,lbfgs,quad2d,gquad2d  # sweep, see ./run's `,` cartesian product
"""
import time

from otrec.HcReconstruction import HcReconstruction
from otrec.viz.line_search_compare import save_run, load_runs, plot_comparison
from loom.cli import experiment, Param


if p := experiment( "hc_line_search_compare",
    method = Param( "gd", help = "gd | pr | lbfgs | quad2d | gquad2d | grid2d | grid3d",
                   choices = [ "gd", "pr", "lbfgs", "quad2d", "gquad2d", "grid2d", "grid3d" ] ),
    name = Param( "", help = "save name (defaults to --method); rerunning under the "
                            "same name overwrites its saved result" ),
    results_dir = Param( "tmp/hc_line_search_runs",
                        help = "dir accumulating one JSON + one HTML animation per named run" ),
    nb_alveoli = Param( 50, help = "Number of alveoli (holes)" ),
    nb_diracs = Param( 2000, help = "Target (finest-scale) number of Dirac masses" ),
    nb_diracs_init = Param( 100, help = "Coarsest multiscale stage's dirac count" ),
    multiscale_factor = Param( 4, help = "Dirac count multiplied by this each multiscale stage" ),
    plateau_frac = Param( 0.01, help = "Non-final stages stop once a step's gain drops below "
                                     "this fraction of that stage's first gain" ),
    nb_angles = Param( 600, help = "Nb angles" ),
    max_iter = Param( 20, help = "Max line-search iterations per stage" ),
    n_grid_2d = Param( 15, help = "grid2d only: grid resolution per axis" ),
    n_grid_3d = Param( 7, help = "grid3d only: grid resolution per axis (O(n^3) evals/step)" ),
    lbfgs_memory = Param( 10, help = "lbfgs only: number of curvature pairs kept" ),
    backend = Param( "sycl", help = "OT backend: jax | sycl" ),
    seed = Param( 1, help = "initial point cloud seed" ),
):
    name = p.name or p.method

    hc, lobes, alveoli = HcReconstruction.make_lung_phantom(
        nb_alveoli = p.nb_alveoli, backend = p.backend, nb_angles = p.nb_angles,
        record = True )

    t0 = time.time()
    hc.line_search_multiscale(
        method = p.method, nb_diracs_final = p.nb_diracs, nb_diracs_init = p.nb_diracs_init,
        factor = p.multiscale_factor, seed = p.seed, plateau_frac = p.plateau_frac,
        max_iter = p.max_iter, n_grid_2d = p.n_grid_2d, n_grid_3d = p.n_grid_3d,
        lbfgs_memory = p.lbfgs_memory )
    dt = time.time() - t0
    print(f"\n{name}: {dt:.2f}s total, final loss={hc.loss_history[-1]['cost']:.6f}")

    meta = dict( method = p.method, nb_alveoli = p.nb_alveoli, nb_diracs = p.nb_diracs,
               nb_angles = p.nb_angles, max_iter = p.max_iter, backend = p.backend,
               seed = p.seed )
    out_file = save_run( p.results_dir, name, meta, hc.loss_history, hc.timings )
    out_html = f"{p.results_dir}/{name}.html"
    hc.export_html( out_html, title = name )

    runs = load_runs( p.results_dir )
    out_plot = f"{p.results_dir}/comparison.png"
    plot_comparison( runs, out_plot )
    print(f"saved -> {out_file}")
    print(f"saved -> {out_html}")
    print(f"wrote {out_plot}  ({len(runs)} accumulated run(s): {', '.join(sorted(runs))})")
