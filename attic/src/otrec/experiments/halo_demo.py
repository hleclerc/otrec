"""Object wider than the detector: the leakage, its correction by the HALO, and what it yields.

The phantom is a ring with holes INSIDE the field of view (the voids we want to preserve) surrounded by
material OUTSIDE (which pollutes the sinogram). Without correction, the excess mass is redistributed
into the field and plugs the holes; `halo.alternate` removes it.

    python -m applications.reconstruction.experiments.halo_demo

Outputs in `tmp/`: the halo dashboard (`viz.halo_plot.plot_halo`, six views, including the
mesh colored by its density), the `M_in` sweep, and a comparison of the two clouds.

MEASURED on this phantom, and this is the result to remember: with the correct `M_in`, the halo recovers the
outer mass to within 0.8% (6.74 for 6.79) and the spread of `∫p_θ` falls from 6.0% to 2.4%.
With the default `M_in` (`min_θ ∫p_θ`, which overestimates it here by 50% -- no angle sees
the whole object), it only recovers 1.53. The inside/outside split dominates everything else.
"""
import matplotlib.pyplot as plt
import numpy as np

from ..halo import Halo, alternate, mass_profile, scan_interior_mass, void_fraction
from ..Sinogram import Sinogram
from ..viz.halo_plot import plot_halo, plot_interior_mass_scan


def make_phantom( extent = 4.0, nb_angles = 180, nb_bins = 600, seed = 0 ):
    """A disk with holes at the center, and three clusters outside the field. Returns `( sinogram, holes )`."""
    rng = np.random.default_rng( seed )
    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )

    sino.add_disk( center = [ 0.0, 0.0 ], radius = 1.5 )
    holes = [ ( 0.6, 0.6 ), ( -0.6, 0.6 ), ( 0.6, -0.6 ), ( -0.6, -0.6 ), ( 0.0, 0.0 ) ]
    for h in holes:
        sino.add_disk( center = list( h ), radius = 0.3, density = -1.0 )

    # the outer material: clusters of disks, so that it is EXTENDED (the real case) and not
    # pointlike -- a coarse mesh represents an isolated blob poorly, but a sheet very well.
    for cx, cy in [ ( 2.8, 0.6 ), ( -2.5, -1.2 ), ( 0.4, 2.9 ), ( -1.8, 2.2 ) ]:
        for _ in range( 12 ):
            off = rng.normal( scale = 0.45, size = 2 )
            sino.add_disk( center = [ cx + off[ 0 ], cy + off[ 1 ] ], radius = 0.3, density = 0.5 )
    return sino, holes


#: mass of the central holed disk -- the true `M_in`, known here since we build the phantom
TRUE_INTERIOR_MASS = np.pi * ( 1.5 ** 2 - 5 * 0.3 ** 2 )


def run( nb_points = 20_000, nb_outer = 3, outer_radius = 4.5, max_iter = 150,
         interior_mass = None, out_dashboard = "tmp/halo_dashboard.png",
         out_compare = "tmp/halo_points.png", out_scan = "tmp/halo_mass_scan.png" ):
    sino, holes = make_phantom()
    extent = sino.extent
    per_angle = mass_profile( sino )
    print( f"mass per angle: min={ per_angle.min():.4g} max={ per_angle.max():.4g} "
           f"(spread { per_angle.std() / per_angle.mean():.1%})" )

    def solve( rec ):
        return rec.multiscale( nb_points, nb_points_init = 500, max_iter = max_iter )

    common = dict( outer_radius = outer_radius, extent = extent, seed = 1, verbose = True )
    print( "\n-- without halo ------------------------------------------------" )
    plain, _ = alternate( sino, solve, nb_outer = 1, **common )

    # `M_in` is the decisive parameter (see `halo.scan_interior_mass`): we sweep it over the
    # uncorrected cloud, which suffices to give its shape, before launching the full alternation.
    scan = scan_interior_mass( Halo( sino, outer_radius = outer_radius ), plain.positions )
    plot_interior_mass_scan( scan, truth = TRUE_INTERIOR_MASS )
    plt.gcf().tight_layout(); plt.gcf().savefig( out_scan, dpi = 130 ); plt.close()
    print( f"figure saved: { out_scan }" )

    print( "\n-- with halo ------------------------------------------------" )
    fixed, halo = alternate( sino, solve, nb_outer = nb_outer, interior_mass = interior_mass,
                             **common )

    plot_halo( halo, points = fixed.positions, interior_mass = interior_mass, out = out_dashboard,
               title = f"halo on { halo.nb_cells } cells, { nb_outer } passes"
                       + ( "" if interior_mass is None else f", M_in imposed = { interior_mass:.3g}" ) )

    def in_holes( pts ):
        pts = np.asarray( pts )
        return float( np.any( [ ( ( pts - np.array( h ) ) ** 2 ).sum( 1 ) < 0.25 ** 2
                                for h in holes ], axis = 0 ).mean() )

    fig, axes = plt.subplots( 1, 2, figsize = ( 11, 5.6 ) )
    for ax, ( rec, name ) in zip( axes, [ ( plain, "without halo" ), ( fixed, "with halo" ) ] ):
        p = rec.positions
        ax.plot( p[ :, 0 ], p[ :, 1 ], ".", markersize = 0.7, color = "#222222" )
        for h in holes:
            ax.add_patch( plt.Circle( h, 0.3, fill = False, linewidth = 0.9, edgecolor = "#D55E00" ) )
        ax.set_xlim( -extent / 2, extent / 2 ); ax.set_ylim( -extent / 2, extent / 2 )
        ax.set_aspect( "equal" )
        ax.set_title( f"{ name } -- { in_holes( p ):.2%} of points in the holes, "
                      f"void { void_fraction( p, extent ):.1%}", fontsize = 9 )
    fig.tight_layout(); fig.savefig( out_compare, dpi = 130 )
    print( f"figure saved: { out_compare }" )
    return fixed, halo


if __name__ == "__main__":
    run()
