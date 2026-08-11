"""Cas de démonstration de la reconstruction par DISQUES (`models.DiskModel`).

Chaque cas est AUTONOME : il fabrique son sinogramme analytiquement (`Sinogram.add_disk`, donc
sans jamais discrétiser une image), part d'un tirage aléatoire de centres et optimise. Aucune
étape préalable n'est nécessaire -- mais `--diracs` en ajoute une (reconstruction en diracs puis
raffinement en disques sur le MÊME nuage), pour illustrer le chaînage qu'offre `Reconstruction`.

Sorties, dans `applications/reconstruction/benchmarks/results/` :
- `disks_<cas>.html` : le nuage de centres au fil de la convergence, rejouable (barre de temps),
  avec les RAYONS EXPORTÉS -- les disques sont dessinés à leur taille réelle, le slider n'étant
  plus qu'un facteur d'échelle (voir `viz/points_html.py`).
- `disks_<cas>_profils.png` (option `--profils`) : profils mesuré vs modèle à quelques angles,
  pour voir de près ce que la perte de Wasserstein a réellement ajusté.

Usage :
    python -m applications.reconstruction.experiments.disks_demo            # tous les cas
    python -m applications.reconstruction.experiments.disks_demo few ring
    python -m applications.reconstruction.experiments.disks_demo grid --profils
    python -m applications.reconstruction.experiments.disks_demo few --diracs
"""
import argparse
import os

import numpy as np

from ..Reconstruction import Reconstruction
from ..Sinogram import Sinogram


RESULTS = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ),
                        "benchmarks", "results" )


# ---- fantômes (vérité terrain : une liste de centres + un rayon commun) --------------------

def _few( seed = 0 ):
    return np.array( [ [ 0.8, 0.3 ], [ -0.9, 0.6 ], [ 0.1, -1.1 ], [ -0.5, -0.7 ], [ 1.3, -0.4 ] ] ), 0.4


def _ring( seed = 3, n = 60 ):
    # rng = np.random.default_rng( seed )
    # theta = 2 * np.pi * # rng.random( n )
    theta = np.linspace( 0, 2 * np.pi, n ) # rng.random( n )
    rad = 1.2 # + 0.25 * rng.random( n )
    return np.stack( [ rad * np.cos( theta ), rad * np.sin( theta ) ], axis = 1 ), 0.15


def _grid( seed = 5, nx = 9, ny = 9 ):
    """Grille hexagonale jittérée, tronquée en disque -- un cas dense où les projections de
    disques voisins se recouvrent largement à presque tous les angles."""
    r = 0.11
    spacing = 2.6 * r
    rng = np.random.default_rng( seed )
    pts = []
    for j in range( ny ):
        for i in range( nx ):
            x = ( i - ( nx - 1 ) / 2 + 0.5 * ( j % 2 ) ) * spacing
            y = ( j - ( ny - 1 ) / 2 ) * spacing * np.sqrt( 3 ) / 2
            if x * x + y * y < ( 0.5 * nx * spacing ) ** 2:
                pts.append( ( x, y ) )
    pts = np.array( pts ) + ( rng.random( ( len( pts ), 2 ) ) - 0.5 ) * 0.1 * spacing
    return pts, r


CASES = {
    # nom     -> ( fantôme, nb_angles, nb_bins, facteur de finesse de la grille image, étendue init )
    "few":  ( _few,  24, 128, 2, 3.0 ),
    "ring": ( _ring, 32, 192 * 2, 2, 3.0 ),
    "grid": ( _grid, 48, 256, 3, 2.0 ),
}


def run_case( name, extent = 6.0, max_iter = 500, seed = 7, profils = False, animate = True,
              diracs = False ):
    phantom, nb_angles, nb_bins, fineness, init_extent = CASES[ name ]
    truth, radius = phantom()

    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    for c in truth:
        sino.add_disk( center = list( c ), radius = radius )

    # grille image plus fine que le détecteur : un disque de rayon `radius` ne couvre que
    # `2*radius/dw` cases détecteur ; la raffiner rend la projection modélisée plus lisse sans
    # toucher aux données mesurées.
    rec = Reconstruction(
        sino, radius = radius, nb_pixels = fineness * nb_bins, extent = init_extent,
        max_iter = max_iter, ftol = 1e-14, record = animate, verbose = True,
    )
    rec.random_points( len( truth ), seed = seed )
    l0 = rec.loss()

    print( f"[{ name }] { len( truth ) } disques r={ radius }, { nb_angles } angles x { nb_bins } cases" )
    if diracs:
        # étage préalable en DIRACS : bon marché et insensible au tirage, il amène déjà le nuage
        # sur la masse ; l'étage disques n'a plus qu'à l'ajuster finement.
        rec.diracs( max_iter = min( 100, max_iter ) )
    rec.disks()
    l1 = rec.loss()
    floor = rec.floor()

    print( f"  perte { l0:.6f} -> { l1:.8f}  (plancher de quantification { floor:.8f})" )

    os.makedirs( RESULTS, exist_ok = True )
    out = os.path.join( RESULTS, f"disks_{ name }.html" )
    rec.export_html( out, extent = extent,           # rayon FIXE : exporté et dessiné tel quel
                     title = f"disques — { name } ({ len( truth ) } x r={ radius })", fps = 10 )

    if profils:
        _plot_profiles( sino, rec.disk_model().projector, rec.points,
                        os.path.join( RESULTS, f"disks_{ name }_profils.png" ) )

    return rec, truth, l0, l1, floor


def _plot_profiles( sino, proj, centers, out_path, nb_shown = 4 ):
    """Compare, à quelques angles, le profil MESURÉ (constant par morceaux sur le détecteur) et
    le profil MODÈLE (projection des disques reconstruits). Les deux sont normalisés à la masse 1
    par angle, comme le fait la perte -- c'est la seule chose que le transport optimal compare.
    """
    import matplotlib.pyplot as plt

    measured = np.asarray( sino.values )
    model = np.asarray( proj.values( centers ) )
    ks = np.linspace( 0, measured.shape[ 0 ] - 1, nb_shown, dtype = int )

    fig, axes = plt.subplots( nb_shown, 1, figsize = ( 8, 2.2 * nb_shown ), sharex = True )
    for ax, k in zip( np.atleast_1d( axes ), ks ):
        m = measured[ k ] / ( measured[ k ].sum() * sino.dw )
        g = model[ k ] / ( model[ k ].sum() * proj.dw )
        ax.step( sino.bin_centers, m, where = "mid", label = "mesuré", lw = 1.2 )
        ax.step( proj.pixel_centers, g, where = "mid", label = "modèle (disques)", lw = 1.0 )
        ax.set_ylabel( f"θ = { np.degrees( sino.angles[ k ] ):.0f}°" )
        ax.legend( loc = "upper right", fontsize = 8 )
    np.atleast_1d( axes )[ -1 ].set_xlabel( "coordonnée détecteur s" )
    fig.tight_layout()
    fig.savefig( out_path, dpi = 120 )
    plt.close( fig )
    print( f"  profils sauvés: { out_path }" )


def main():
    ap = argparse.ArgumentParser( description = __doc__ )
    ap.add_argument( "cases", nargs = "*", choices = list( CASES ),
                     help = "cas à lancer (défaut : tous)" )
    ap.add_argument( "--profils", action = "store_true", help = "exporter aussi la comparaison de profils (matplotlib)" )
    ap.add_argument( "--diracs", action = "store_true", help = "étage préalable en diracs, avant les disques" )
    ap.add_argument( "--max-iter", type = int, default = 500 )
    ap.add_argument( "--seed", type = int, default = 7 )
    args = ap.parse_args()

    for name in ( args.cases or list( CASES ) ):
        run_case( name, max_iter = args.max_iter, seed = args.seed, profils = args.profils,
                  diracs = args.diracs )
        print()


if __name__ == "__main__":
    main()
