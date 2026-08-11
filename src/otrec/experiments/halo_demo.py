"""Objet plus large que le détecteur : la fuite, sa correction par le HALO, et ce qu'elle rend.

Le fantôme est un anneau troué DANS le champ de vue (les vides qu'on veut préserver) entouré de
matière DEHORS (ce qui pollue le sinogramme). Sans correction, la masse en trop est redistribuée
dans le champ et bouche les trous ; `halo.alternate` la retire.

    python -m applications.reconstruction.experiments.halo_demo

Sorties dans `tmp/` : le tableau de bord du halo (`viz.halo_plot.plot_halo`, six vues, dont le
maillage colorié par sa densité), le balayage de `M_in`, et une comparaison des deux nuages.

MESURÉ sur ce fantôme, et c'est le résultat à retenir : à `M_in` correct, le halo retrouve la
masse extérieure à 0.8% près (6.74 pour 6.79) et la dispersion de `∫p_θ` tombe de 6.0% à 2.4%.
Avec le `M_in` par défaut (`min_θ ∫p_θ`, qui le surestime ici de 50% -- aucun angle ne voit
l'objet entier), il n'en récupère que 1.53. Le partage intérieur/extérieur domine tout le reste.
"""
import matplotlib.pyplot as plt
import numpy as np

from ..halo import Halo, alternate, mass_profile, scan_interior_mass, void_fraction
from ..Sinogram import Sinogram
from ..viz.halo_plot import plot_halo, plot_interior_mass_scan


def make_phantom( extent = 4.0, nb_angles = 180, nb_bins = 600, seed = 0 ):
    """Un disque troué au centre, et trois amas hors champ. Renvoie `( sinogram, trous )`."""
    rng = np.random.default_rng( seed )
    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )

    sino.add_disk( center = [ 0.0, 0.0 ], radius = 1.5 )
    holes = [ ( 0.6, 0.6 ), ( -0.6, 0.6 ), ( 0.6, -0.6 ), ( -0.6, -0.6 ), ( 0.0, 0.0 ) ]
    for h in holes:
        sino.add_disk( center = list( h ), radius = 0.3, density = -1.0 )

    # la matière extérieure : des amas de disques, pour qu'elle soit ÉTENDUE (le cas réel) et non
    # ponctuelle -- un maillage grossier représente mal un blob isolé, très bien une nappe.
    for cx, cy in [ ( 2.8, 0.6 ), ( -2.5, -1.2 ), ( 0.4, 2.9 ), ( -1.8, 2.2 ) ]:
        for _ in range( 12 ):
            off = rng.normal( scale = 0.45, size = 2 )
            sino.add_disk( center = [ cx + off[ 0 ], cy + off[ 1 ] ], radius = 0.3, density = 0.5 )
    return sino, holes


#: masse du disque troué central -- le vrai `M_in`, connu ici puisqu'on fabrique le fantôme
TRUE_INTERIOR_MASS = np.pi * ( 1.5 ** 2 - 5 * 0.3 ** 2 )


def run( nb_points = 20_000, nb_outer = 3, outer_radius = 4.5, max_iter = 150,
         interior_mass = None, out_dashboard = "tmp/halo_dashboard.png",
         out_compare = "tmp/halo_points.png", out_scan = "tmp/halo_mass_scan.png" ):
    sino, holes = make_phantom()
    extent = sino.extent
    per_angle = mass_profile( sino )
    print( f"masse par angle : min={ per_angle.min():.4g} max={ per_angle.max():.4g} "
           f"(dispersion { per_angle.std() / per_angle.mean():.1%})" )

    def solve( rec ):
        return rec.multiscale( nb_points, nb_points_init = 500, max_iter = max_iter )

    common = dict( outer_radius = outer_radius, extent = extent, seed = 1, verbose = True )
    print( "\n-- sans halo ------------------------------------------------" )
    plain, _ = alternate( sino, solve, nb_outer = 1, **common )

    # `M_in` est le paramètre décisif (voir `halo.scan_interior_mass`) : on le balaie sur le nuage
    # non corrigé, qui suffit à en donner la forme, avant de lancer l'alternance complète.
    scan = scan_interior_mass( Halo( sino, outer_radius = outer_radius ), plain.positions )
    plot_interior_mass_scan( scan, truth = TRUE_INTERIOR_MASS )
    plt.gcf().tight_layout(); plt.gcf().savefig( out_scan, dpi = 130 ); plt.close()
    print( f"figure sauvée: { out_scan }" )

    print( "\n-- avec halo ------------------------------------------------" )
    fixed, halo = alternate( sino, solve, nb_outer = nb_outer, interior_mass = interior_mass,
                             **common )

    plot_halo( halo, points = fixed.positions, interior_mass = interior_mass, out = out_dashboard,
               title = f"halo sur { halo.nb_cells } cellules, { nb_outer } passes"
                       + ( "" if interior_mass is None else f", M_in imposé = { interior_mass:.3g}" ) )

    def in_holes( pts ):
        pts = np.asarray( pts )
        return float( np.any( [ ( ( pts - np.array( h ) ) ** 2 ).sum( 1 ) < 0.25 ** 2
                                for h in holes ], axis = 0 ).mean() )

    fig, axes = plt.subplots( 1, 2, figsize = ( 11, 5.6 ) )
    for ax, ( rec, name ) in zip( axes, [ ( plain, "sans halo" ), ( fixed, "avec halo" ) ] ):
        p = rec.positions
        ax.plot( p[ :, 0 ], p[ :, 1 ], ".", markersize = 0.7, color = "#222222" )
        for h in holes:
            ax.add_patch( plt.Circle( h, 0.3, fill = False, linewidth = 0.9, edgecolor = "#D55E00" ) )
        ax.set_xlim( -extent / 2, extent / 2 ); ax.set_ylim( -extent / 2, extent / 2 )
        ax.set_aspect( "equal" )
        ax.set_title( f"{ name } -- { in_holes( p ):.2%} des points dans les trous, "
                      f"vide { void_fraction( p, extent ):.1%}", fontsize = 9 )
    fig.tight_layout(); fig.savefig( out_compare, dpi = 130 )
    print( f"figure sauvée: { out_compare }" )
    return fixed, halo


if __name__ == "__main__":
    run()
