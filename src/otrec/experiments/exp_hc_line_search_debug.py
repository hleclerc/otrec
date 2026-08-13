"""Instrument `HcReconstruction.line_search` — how far is the parabola vertex
from the true 1D minimum, and would mixing in the previous step's direction help?

For the first `instrument_iters` steps this:
  1. does a fine 1D scan of cost(p - t·g) and compares its true argmin to the
     3-point-parabola vertex `line_search` actually picked;
  2. does a 2D grid scan over (t_g, t_prev), combining the gradient direction
     with the *previous* accepted step's direction, to see whether the global
     minimum of that 2D neighbourhood sits off the pure-gradient axis (t_prev=0)
     explored by the parabola.

Writes one PNG per instrumented step to tmp/hc_line_search_debug_step{N}.png.

Run via:
    ./run experiment hc_line_search_debug
    ./run experiment hc_line_search_debug --nb-diracs=2000 --instrument-iters=4
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from otrec.HcReconstruction import HcReconstruction
from loom.cli import experiment, Param


def _plot_step( rec, out_dir ):
    step = rec[ "step" ]
    has_grid = "cost_grid" in rec

    fig, axes = plt.subplots( 1, 2 if has_grid else 1, figsize = ( 11 if has_grid else 5.5, 4.2 ) )
    ax0 = axes[ 0 ] if has_grid else axes

    ax0.plot( rec[ "ts" ], rec[ "costs_1d" ], "-", color = "#3366cc", lw = 1.5 )
    ax0.axvline( rec[ "t_parabola" ], color = "crimson", ls = "--", lw = 1,
                label = f"parabola t={rec['t_parabola']:.4g}" )
    ax0.axvline( rec[ "t_true_1d" ], color = "forestgreen", ls = "--", lw = 1,
                label = f"true argmin t={rec['t_true_1d']:.4g}" )
    ax0.plot( [ rec[ "t_parabola" ] ], [ rec[ "cost_parabola" ] ], "o", color = "crimson" )
    ax0.plot( [ rec[ "t_true_1d" ] ], [ rec[ "cost_true_1d" ] ], "o", color = "forestgreen" )
    ax0.set_xlabel( "t  (p - t·g)" )
    ax0.set_ylabel( "cost" )
    ax0.set_title( f"step {step}: 1D scan along -g" )
    ax0.legend( fontsize = 8 )

    if has_grid:
        ax1 = axes[ 1 ]
        tg_vals, tp_vals, grid = rec[ "tg_vals" ], rec[ "tp_vals" ], rec[ "cost_grid" ]
        # clip the color scale close to the valley floor -- the full range (up to
        # cost0, or higher on the uphill side) washes out the near-minimum structure
        # that's actually of interest here.
        vmax = min( grid.max(), rec[ "cost0" ], 3.0 * grid.min() )
        im = ax1.imshow( grid, origin = "lower", aspect = "auto",
                         extent = [ tg_vals[ 0 ], tg_vals[ -1 ], tp_vals[ 0 ], tp_vals[ -1 ] ],
                         cmap = "viridis", vmax = vmax )
        fig.colorbar( im, ax = ax1, label = "cost (clipped)", extend = "max" )
        ax1.axhline( 0.0, color = "white", lw = 0.8, ls = ":",
                    label = "t_prev=0 (pure gradient axis)" )
        ax1.plot( [ rec[ "t_parabola" ] ], [ 0.0 ], "o", color = "crimson",
                 label = f"parabola choice" )
        ax1.plot( [ rec[ "tg_best" ] ], [ rec[ "tp_best" ] ], "o", color = "orange",
                 label = f"2D grid min" )
        ax1.set_xlabel( "t_g  (gradient step)" )
        ax1.set_ylabel( "t_prev  (previous-direction mix)" )
        ax1.set_title( "2D grid: gradient × previous direction" )
        ax1.legend( fontsize = 7, loc = "upper right" )

    fig.tight_layout()
    fig.savefig( f"{out_dir}/hc_line_search_debug_step{step}.png", dpi = 130 )
    plt.close( fig )


if p := experiment( "hc_line_search_debug",
    nb_alveoli = Param( 300, help = "Number of alveoli (holes)" ),
    nb_diracs = Param( 2000, help = "Number of Dirac masses" ),
    nb_angles = Param( 300, help = "Nb angles" ),
    max_iter = Param( 8, help = "Max line-search iterations" ),
    instrument_iters = Param( 4, help = "Nb of leading steps to instrument" ),
    backend = Param( "sycl", help = "OT backend: jax | sycl" ),
):

    hc, lobes, alveoli = HcReconstruction.make_lung_phantom(
        nb_alveoli = p.nb_alveoli, backend = p.backend, nb_angles = p.nb_angles,
        record = True )
    points = hc.random_points( p.nb_diracs, seed = 1 )

    hc.line_search( points, max_iter = p.max_iter,
                    instrument_iters = p.instrument_iters )

    print()
    for rec in hc.debug_scans:
        step = rec[ "step" ]
        dt = rec[ "t_true_1d" ] - rec[ "t_parabola" ]
        dc = rec[ "cost_parabola" ] - rec[ "cost_true_1d" ]
        rel = dc / rec[ "cost_true_1d" ] if rec[ "cost_true_1d" ] > 0 else 0.0
        print(f"step {step}: t_parabola={rec['t_parabola']:.4g}  "
              f"t_true_1d={rec['t_true_1d']:.4g}  (Δt={dt:+.4g})   "
              f"cost_parabola={rec['cost_parabola']:.6f}  "
              f"cost_true_1d={rec['cost_true_1d']:.6f}  "
              f"(gap={dc:+.3g}, {rel:+.2%})")
        if "cost_grid" in rec:
            dg = rec[ "cost_parabola" ] - rec[ "cost_grid_best" ]
            relg = dg / rec[ "cost_grid_best" ] if rec[ "cost_grid_best" ] > 0 else 0.0
            print(f"          2D grid best: t_g={rec['tg_best']:.4g}  "
                  f"t_prev={rec['tp_best']:.4g}  cost={rec['cost_grid_best']:.6f}  "
                  f"(gap vs parabola={dg:+.3g}, {relg:+.2%})")

    for rec in hc.debug_scans:
        _plot_step( rec, out_dir = "tmp" )
    print("\nwrote tmp/hc_line_search_debug_step*.png")
