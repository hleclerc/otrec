"""Poumon à `scale = 2` : l'objet fait DEUX FOIS le détecteur. Maillage d'abord, diracs ensuite.

Le fantôme est celui de `lung_alveoli.make_lung_phantom`, agrandi sans toucher au détecteur :
l'ombre des lobes déborde la fenêtre visible à tous les angles, la masse mesurée par angle n'est
plus constante, et `OtPlan1d` -- qui normalise ses deux distributions -- redistribue l'excédent
DANS le champ, bouchant les alvéoles qu'on voulait justement préserver.

La chaîne, dans l'ordre (voir `mesh.py` pour le raisonnement complet) :

1. `GradedMesh.solve` reconstruit TOUT sur un maillage gradué -- fin dans le champ de vue,
   de plus en plus grossier dehors. Convexe, et linéaire au défaut (moindres carrés + laplacien,
   gradient conjugué) ; passer `tv = 3e-3` pour la variante à variation totale ;
2. on lit sur cette solution le partage intérieur/extérieur de la masse, inaccessible autrement ;
3. `GradedMesh.corrected` retire du sinogramme la contribution des cellules EXTÉRIEURES ;
4. les diracs reprennent l'intérieur, à pleine résolution détecteur, sur ce sinogramme corrigé.

Pas d'étape `disks` ici : elle coûte cher et n'est pas ce qu'on cherche à démontrer.

    python -m applications.reconstruction.experiments.lung_mesh

Sorties dans `tmp/` : la solution sur maillage (`viz.mesh_plot`) et la comparaison des nuages de
diracs, avec et sans correction, face à la vérité terrain.
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
    print( f"génération du fantôme (scale={ scale }, { nb_alveoli } alvéoles)..." )
    sino, lobes, alveoli = make_lung_phantom(
        nb_angles = nb_angles, nb_bins = nb_bins, nb_alveoli = nb_alveoli,
        alveolus_radius = alveolus_radius, scale = scale )
    bound = max( r + float( np.linalg.norm( c ) ) for c, r in lobes )
    fov = sino.extent / 2

    # vérité terrain analytique : lobes moins alvéoles -- de quoi juger le partage trouvé
    true_total = sum( np.pi * r * r for _, r in lobes ) - sum( np.pi * r * r for _, r in alveoli )
    inside = [ ( c, r ) for c, r in alveoli if float( np.linalg.norm( c ) ) + r <= fov ]
    true_inside = np.pi * fov * fov - sum( np.pi * r * r for _, r in inside )
    per_angle = np.asarray( sino.mass() )
    print( f"objet de rayon { bound:.1f} pour un détecteur de demi-largeur { fov:.1f} "
           f"-- masse par angle { per_angle.min():.1f}..{ per_angle.max():.1f} "
           f"(dispersion { per_angle.std() / per_angle.mean():.2%})" )
    print( f"vérité terrain : masse totale { true_total:.1f}, dont { true_inside:.1f} dans le champ" )

    # -- 1. le maillage ----------------------------------------------------
    # la maille intérieure suit le rayon des alvéoles : plus fine ne servirait qu'à payer, le
    # maillage n'ayant pas vocation à résoudre les trous (c'est le métier des diracs).
    cell = float( cell_size if cell_size is not None else alveolus_radius )
    t0 = time.time()
    mesh = GradedMesh( sino, outer_radius = bound * 1.03, cell_size = cell, nb_coarse_bins = 500 )
    print( f"\n{ mesh }" )
    mesh.solve( smooth = smooth, tv = tv, nb_iter = mesh_iter, verbose = True )
    print( f"maillage résolu en { time.time() - t0:.1f}s : masse { mesh.mass():.1f} "
           f"(vrai { true_total:.1f}), dont { mesh.interior_mass():.1f} dedans "
           f"(vrai { true_inside:.1f}, écart { mesh.interior_mass() / true_inside - 1:+.1%})" )

    plot_mesh_solution( mesh, out = out_mesh,
                        title = f"poumon scale={ scale } -- maillage gradué, "
                                f"{ f'TV={ tv }' if tv else f'L2 smooth={ smooth }' }" )

    corrected = mesh.corrected()
    cor_mass = np.asarray( corrected.mass() )
    print( f"sinogramme corrigé : masse par angle { cor_mass.min():.1f}..{ cor_mass.max():.1f} "
           f"(dispersion { cor_mass.std() / cor_mass.mean():.2%})" )

    # -- 2. les diracs, sur l'intérieur seulement --------------------------
    clouds = {}
    for name, data in ( ( "sinogramme brut", sino ), ( "extérieur retiré", corrected ) ):
        print( f"\n-- diracs ({ name }) --------------------------------------" )
        t0 = time.time()
        rec = Reconstruction( data, extent = sino.extent, seed = 1, verbose = True )
        rec.multiscale( nb_points_final = nb_diracs, nb_points_init = nb_diracs // 16, factor = 4,
                        optimizer_factory = lambda n: LBFGS( max_iter = max_iter, ftol = 1e-9 ),
                        noise_frac = 1e-2 )
        clouds[ name ] = rec.positions
        print( f"  { time.time() - t0:.1f}s, vide { void_fraction( rec.positions, sino.extent ):.1%}" )

    # -- 3. la comparaison -------------------------------------------------
    fig, axes = plt.subplots( 1, 3, figsize = ( 17, 6 ) )
    plot_phantom( lobes, alveoli, extent = sino.extent, ax = axes[ 0 ], bound = fov )
    axes[ 0 ].set_title( f"vérité terrain (recadrée sur le détecteur)", fontsize = 10 )
    for ax, ( name, pos ) in zip( axes[ 1: ], clouds.items() ):
        if len( pos ) > plot_max_points:
            pos = pos[ np.random.default_rng( 0 ).choice( len( pos ), plot_max_points, replace = False ) ]
        ax.plot( pos[ :, 0 ], pos[ :, 1 ], ".", markersize = 0.6, color = "black" )
        ax.add_patch( plt.Circle( ( 0, 0 ), fov, fill = False, linestyle = "--", linewidth = 1.0,
                                  edgecolor = VERMILLION ) )
        ax.set_title( f"{ nb_diracs } diracs -- { name }\n"
                      f"vide { void_fraction( clouds[ name ], sino.extent ):.1%}", fontsize = 10 )
    for ax in axes:
        ax.set_xlim( -fov, fov ); ax.set_ylim( -fov, fov ); ax.set_aspect( "equal" )
    fig.tight_layout(); fig.savefig( out_points, dpi = 150 )
    print( f"figure sauvée: { out_points }" )
    return mesh, clouds


if __name__ == "__main__":
    run()
