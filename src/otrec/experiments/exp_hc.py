"""`HcReconstruction` demo/benchmark, parametrized by a PIPELINE expression —
replaces the old `exp_hc_lung.py` / `exp_hc_line_search_compare.py` /
`exp_hc_line_search_debug.py` / `exp_hc_multiscale_triangles.py`, each of
which hardcoded one particular combination of algorithm/model/multiscale
choices in its own file. See `HcReconstruction.optim.pipeline` for the DSL
grammar (`gd`/`pr`/`lbfgs`/`quad2d`/`gquad2d`/`grid2d`/`grid3d`, `diracs`/
`disk`/`triangle`/`polygon`, `multiscale(...)`, `+`) and why it uses `;`
where you'd normally write a Python `,`.

There is no `--marker` param: what gets DRAWN is derived automatically from
the pipeline's own model (`HcReconstruction.export_html`) — a `disk`/
`triangle` (radial-profile) stage draws a circle (that IS its real 2D
support, see `cost.jax_disks`'s docstring for why); a `polygon(n_sides=...)`
stage draws each point's true rotated n-gon from its optimized orientation
(see `cost.jax_polygon`). The procedure defines the shape, not a flag.

Each run saves its behaviour under a NAME (`--name`, defaults to a slug of
`--pipeline`) into `--results-dir` (one JSON + one HTML animation per name,
both overwritten on rerun), then reloads every `*.json` there and redraws
the comparison plot from all of them — so runs accumulate across sessions:
try one pipeline today, another tomorrow, and the plot keeps both. Comparing
BACKENDS is just two runs with different `--name`s (e.g.
`--backend=sycl --name=sycl` then `--backend=jax --name=jax`), the same way
comparing algorithms is.

If any `LineSearch` in the pipeline was built with `instrument_iters=N`
(`gd`/`pr` only — see `optim.gradient_line_search`), the recorder's
`debug_scans` are additionally rendered to `{results_dir}/{name}_debug_step*.png`
(1D scan of the true minimum vs. the parabola vertex actually taken, plus a
2D gradient x previous-direction grid).

Run via:
    ./run experiment hc --pipeline="lbfgs(diracs)"
    ./run experiment hc --pipeline="multiscale(gd(diracs); nb_points_init=100)" --name=gd
    ./run experiment hc --pipeline="multiscale(lbfgs(diracs); nb_points_init=100)" --name=lbfgs
    ./run experiment hc --pipeline="multiscale(grid2d(diracs); nb_points_init=100)" --max-iter=15 --name=grid2d
    ./run experiment hc --pipeline="gd(diracs; instrument_iters=4)" --max-iter=8 --name=debug
    ./run experiment hc --name=triangles --nb-diracs=5000 --pipeline="multiscale(gd(diracs) + lbfgs(triangle(radius_factor=0.5)); nb_points_init=100) + lbfgs(triangle(radius_factor=0.5))"
    ./run experiment hc --backend=jax --name=polygons --pipeline="multiscale(lbfgs(polygon(n_sides=3; radius_factor=0.5)); nb_points_init=100)"
"""
import re
import time

import jax
# The disks/triangle jax cost's ~2000-element cumsum/searchsorted loses enough
# float32 precision at late (many, near-touching) multiscale stages to
# occasionally produce a NEGATIVE cost -- see [[HcReconstruction disks model]].
# The sycl backend is unaffected (its sweep is double-precision internally);
# this is cheap insurance for jax-backend disks pipelines specifically.
jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from otrec.HcReconstruction import HcReconstruction
from otrec.viz.line_search_compare import save_run, load_runs, plot_comparison
from loom.cli import experiment, Param


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return slug[:80]


def _plot_debug_step(rec, out_dir, name):
    """One PNG per instrumented step -- see `optim.instrumentation.instrument_step`."""
    step = rec["step"]
    has_grid = "cost_grid" in rec

    fig, axes = plt.subplots(1, 2 if has_grid else 1, figsize=(11 if has_grid else 5.5, 4.2))
    ax0 = axes[0] if has_grid else axes

    ax0.plot(rec["ts"], rec["costs_1d"], "-", color="#3366cc", lw=1.5)
    ax0.axvline(rec["t_parabola"], color="crimson", ls="--", lw=1,
               label=f"parabola t={rec['t_parabola']:.4g}")
    ax0.axvline(rec["t_true_1d"], color="forestgreen", ls="--", lw=1,
               label=f"true argmin t={rec['t_true_1d']:.4g}")
    ax0.plot([rec["t_parabola"]], [rec["cost_parabola"]], "o", color="crimson")
    ax0.plot([rec["t_true_1d"]], [rec["cost_true_1d"]], "o", color="forestgreen")
    ax0.set_xlabel("t  (p - t*g)")
    ax0.set_ylabel("cost")
    ax0.set_title(f"step {step}: 1D scan along -g")
    ax0.legend(fontsize=8)

    if has_grid:
        ax1 = axes[1]
        tg_vals, tp_vals, grid = rec["tg_vals"], rec["tp_vals"], rec["cost_grid"]
        vmax = min(grid.max(), rec["cost0"], 3.0 * grid.min())
        im = ax1.imshow(grid, origin="lower", aspect="auto",
                        extent=[tg_vals[0], tg_vals[-1], tp_vals[0], tp_vals[-1]],
                        cmap="viridis", vmax=vmax)
        fig.colorbar(im, ax=ax1, label="cost (clipped)", extend="max")
        ax1.axhline(0.0, color="white", lw=0.8, ls=":", label="t_prev=0 (pure gradient axis)")
        ax1.plot([rec["t_parabola"]], [0.0], "o", color="crimson", label="parabola choice")
        ax1.plot([rec["tg_best"]], [rec["tp_best"]], "o", color="orange", label="2D grid min")
        ax1.set_xlabel("t_g  (gradient step)")
        ax1.set_ylabel("t_prev  (previous-direction mix)")
        ax1.set_title("2D grid: gradient x previous direction")
        ax1.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{name}_debug_step{step}.png", dpi=130)
    plt.close(fig)


if p := experiment("hc",
    pipeline = Param("multiscale(lbfgs(diracs); nb_points_init=100)",
                    help = "pipeline DSL expression -- see otrec.HcReconstruction.optim.pipeline "
                          "(use ';' where you'd write a Python ',')"),
    name = Param("", help = "save name (defaults to a slug of --pipeline); rerunning under "
                          "the same name overwrites its saved result"),
    results_dir = Param("tmp/hc_runs", help = "dir accumulating one JSON + one HTML per name"),
    nb_alveoli = Param(300, help = "Number of alveoli (holes)"),
    nb_diracs = Param(2000, help = "Target (finest-scale) number of Dirac masses"),
    nb_diracs_init = Param(100, help = "multiscale(...) nodes' default coarsest-stage count, "
                                     "unless a node sets its own nb_points_init"),
    multiscale_factor = Param(4, help = "multiscale(...) nodes' default point-count-per-stage "
                                       "growth factor, unless a node sets its own factor"),
    plateau_frac = Param(0.01, help = "multiscale(...) nodes' default early-stop threshold for "
                                     "non-final stages, unless a node sets its own plateau_frac"),
    nb_angles = Param(600, help = "Nb angles"),
    max_iter = Param(20, help = "default max line-search iterations per stage, unless a stage "
                               "sets its own max_iter"),
    backend = Param("sycl", help = "OT backend: jax | sycl"),
    seed = Param(1, help = "initial point cloud seed"),
    print_steps = Param(True, help = "print every line-search step (--print-steps to enable; "
                                     "'--verbose' collides with ./run's own global flag)"),
):
    name = p.name or _slugify(p.pipeline)

    hc, lobes, alveoli = HcReconstruction.make_lung_phantom(
        nb_alveoli = p.nb_alveoli, backend = p.backend, nb_angles = p.nb_angles, record = True)

    t0 = time.time()
    hc.run_pipeline(
        p.pipeline, nb_diracs = p.nb_diracs, seed = p.seed, max_iter = p.max_iter,
        verbose = p.print_steps, multiscale_plateau_frac = p.plateau_frac,
        nb_diracs_init = p.nb_diracs_init, factor = p.multiscale_factor)
    dt = time.time() - t0
    print(f"\n{name}: {dt:.2f}s total, final loss={hc.recorder.loss_history[-1]['cost']:.6f}")

    meta = dict(pipeline = p.pipeline, backend = p.backend, nb_alveoli = p.nb_alveoli,
               nb_diracs = p.nb_diracs, nb_angles = p.nb_angles, max_iter = p.max_iter,
               seed = p.seed)
    out_file = save_run(p.results_dir, name, meta, hc.recorder.loss_history, hc.recorder.timings)
    out_html = f"{p.results_dir}/{name}.html"
    hc.export_html(out_html, title = name)

    runs = load_runs(p.results_dir)
    out_plot = f"{p.results_dir}/comparison.png"
    plot_comparison(runs, out_plot)
    print(f"saved -> {out_file}")
    print(f"saved -> {out_html}")
    print(f"wrote {out_plot}  ({len(runs)} accumulated run(s): {', '.join(sorted(runs))})")

    if hc.recorder.debug_scans:
        for rec in hc.recorder.debug_scans:
            _plot_debug_step(rec, out_dir = p.results_dir, name = name)
        print(f"wrote {p.results_dir}/{name}_debug_step*.png")
