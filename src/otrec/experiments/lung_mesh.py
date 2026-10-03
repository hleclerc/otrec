"""Lung at `scale = 2`: the object is TWICE the detector size. Mesh first, Diracs second.

The phantom is that of `lung_alveoli.make_lung_phantom`, enlarged without touching the detector:
the shadow of the lobes overflows the visible window at every angle, the mass measured per angle is
no longer constant, and `SdotPlan1d` -- which normalizes both of its distributions -- redistributes the excess
INSIDE the field, plugging the alveoli we precisely wanted to preserve.

The chain, in order (see `mesh.py` for the full reasoning):

1. `GradedMesh.solve` reconstructs EVERYTHING on a graded mesh -- fine in the field of view,
   increasingly coarse outside. Convex, and linear by default (least squares + Laplacian,
   conjugate gradient); pass `tv = 3e-3` for the total-variation variant;
2. from this solution we read the inside/outside split of the mass, otherwise inaccessible;
3. `GradedMesh.corrected` removes the contribution of the OUTER cells from the sinogram;
4. the Diracs take over the interior, at full detector resolution, on this corrected sinogram.

No `disks` step here: it is expensive and not what we are trying to demonstrate.

    python -m applications.reconstruction.experiments.lung_mesh

Outputs in `tmp/`: the mesh solution (`viz.mesh_plot`) and the comparison of the Dirac clouds,
with and without correction, against the ground truth.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from ..halo import void_fraction
from ..mesh import GradedMesh
from ..optimizers import LBFGS
from ..Reconstruction import Reconstruction
from ..viz.mesh_plot import plot_mesh_solution
from ..viz.style import VERMILLION
from .lung_alveoli import make_lung_phantom, plot_phantom


def run( scale = 2.0, nb_alveoli = 1000, alveolus_radius = 0.7, nb_angles = 300, nb_bins = 1000,
         cell_size = None, smooth = 3e-2, tv = None, mesh_iter = None, nb_diracs = 20_000, max_iter = 40,
         out_mesh = "tmp/lung_mesh.png", out_points = "tmp/lung_mesh_points.png",
         plot_max_points = 200_000 ):
    print( f"generating phantom (scale={ scale }, { nb_alveoli } alveoli)..." )
    sino, lobes, alveoli = make_lung_phantom(
        nb_angles = nb_angles, nb_bins = nb_bins, nb_alveoli = nb_alveoli,
        alveolus_radius = alveolus_radius, scale = scale )
    bound = max( r + float( np.linalg.norm( c ) ) for c, r in lobes )
    fov = sino.extent / 2

    # analytical ground truth: lobes minus alveoli -- to judge the split found
    true_total = sum( np.pi * r * r for _, r in lobes ) - sum( np.pi * r * r for _, r in alveoli )
    inside = [ ( c, r ) for c, r in alveoli if float( np.linalg.norm( c ) ) + r <= fov ]
    true_inside = np.pi * fov * fov - sum( np.pi * r * r for _, r in inside )
    per_angle = np.asarray( sino.mass() )
    print( f"object of radius { bound:.1f} for a detector of half-width { fov:.1f} "
           f"-- mass per angle { per_angle.min():.1f}..{ per_angle.max():.1f} "
           f"(spread { per_angle.std() / per_angle.mean():.2%})" )
    print( f"ground truth: total mass { true_total:.1f}, of which { true_inside:.1f} in the field" )

    # -- 1. the mesh ----------------------------------------------------
    # the interior cell size follows the alveolus radius: finer would only cost more, since the
    # mesh is not meant to resolve the holes (that is the Diracs' job).
    cell = float( cell_size if cell_size is not None else alveolus_radius )
    t0 = time.time()
    mesh = GradedMesh( sino, outer_radius = bound * 1.03, cell_size = cell, nb_coarse_bins = 500 )
    print( f"\n{ mesh }" )
    mesh.solve( smooth = smooth, tv = tv, nb_iter = mesh_iter, verbose = True )
    print( f"mesh solved in { time.time() - t0:.1f}s: mass { mesh.mass():.1f} "
           f"(true { true_total:.1f}), of which { mesh.interior_mass():.1f} inside "
           f"(true { true_inside:.1f}, deviation { mesh.interior_mass() / true_inside - 1:+.1%})" )

    plot_mesh_solution( mesh, out = out_mesh,
                        title = f"lung scale={ scale } -- graded mesh, "
                                f"{ f'TV={ tv }' if tv else f'L2 smooth={ smooth }' }" )

    corrected = mesh.corrected()
    cor_mass = np.asarray( corrected.mass() )
    print( f"corrected sinogram: mass per angle { cor_mass.min():.1f}..{ cor_mass.max():.1f} "
           f"(spread { cor_mass.std() / cor_mass.mean():.2%})" )

    # -- 2. the Diracs, on the interior only --------------------------
    clouds = {}
    for name, data in ( ( "raw sinogram", sino ), ( "exterior removed", corrected ) ):
        print( f"\n-- Diracs ({ name }) --------------------------------------" )
        t0 = time.time()
        rec = Reconstruction( data, extent = sino.extent, seed = 1, verbose = True )
        rec.multiscale( nb_points_final = nb_diracs, nb_points_init = nb_diracs // 16, factor = 4,
                        optimizer_factory = lambda n: LBFGS( max_iter = max_iter, ftol = 1e-9 ),
                        noise_frac = 1e-2 )
        clouds[ name ] = rec.positions
        print( f"  { time.time() - t0:.1f}s, void { void_fraction( rec.positions, sino.extent ):.1%}" )

    # -- 3. the comparison -------------------------------------------------
    fig, axes = plt.subplots( 1, 3, figsize = ( 17, 6 ) )
    plot_phantom( lobes, alveoli, extent = sino.extent, ax = axes[ 0 ], bound = fov )
    axes[ 0 ].set_title( f"ground truth (cropped to the detector)", fontsize = 10 )
    for ax, ( name, pos ) in zip( axes[ 1: ], clouds.items() ):
        if len( pos ) > plot_max_points:
            pos = pos[ np.random.default_rng( 0 ).choice( len( pos ), plot_max_points, replace = False ) ]
        ax.plot( pos[ :, 0 ], pos[ :, 1 ], ".", markersize = 0.6, color = "black" )
        ax.add_patch( plt.Circle( ( 0, 0 ), fov, fill = False, linestyle = "--", linewidth = 1.0,
                                  edgecolor = VERMILLION ) )
        ax.set_title( f"{ nb_diracs } Diracs -- { name }\n"
                      f"void { void_fraction( clouds[ name ], sino.extent ):.1%}", fontsize = 10 )
    for ax in axes:
        ax.set_xlim( -fov, fov ); ax.set_ylim( -fov, fov ); ax.set_aspect( "equal" )
    fig.tight_layout(); fig.savefig( out_points, dpi = 150 )
    print( f"figure saved: { out_points }" )
    return mesh, clouds


if __name__ == "__main__":
    run()
