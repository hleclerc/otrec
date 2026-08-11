"""Ce que la reconstruction sur MAILLAGE (`mesh.GradedMesh`) a trouvé.

`plot_mesh` dessine la solution telle qu'elle est : une cellule = un rectangle, sa densité = sa
couleur. Pas de rééchantillonnage sur une grille régulière, qui masquerait précisément ce qu'on
veut voir -- la graduation, et le fait que l'extérieur est représenté par très peu de mailles.

`plot_mesh_solution` y ajoute les trois vues qui disent si la solution tient : la masse par angle
avant/après retrait de l'extérieur (elle doit s'aplatir), le sinogramme mesuré, et la part que le
maillage attribue à l'extérieur.

Palette : voir `viz.style`.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle

from .style import BLUE, GREEN, GREY, SEQ, VERMILLION


def plot_mesh( mesh, ax = None, weights = None, vmax = None, show_grid = None,
               exterior_only = False, show_fov = True ):
    """Le maillage colorié par sa densité. Renvoie la `PatchCollection` (pour la barre de couleur).

    `show_grid` : trace le bord des cellules. Par défaut seulement en dessous de 4000 cellules --
    au-delà les traits couvrent la donnée au lieu de la structurer.
    `exterior_only` : n'affiche que ce qui sera RETIRÉ du sinogramme, ce que les diracs ne verront
    donc jamais.
    """
    ax = ax or plt.gca()
    w = np.asarray( mesh.weights if weights is None else weights, dtype = float )
    keep = ~mesh.interior if exterior_only else np.ones( mesh.nb_cells, dtype = bool )
    c, h, w = mesh.centers[ keep ], mesh.sizes[ keep ], w[ keep ]
    if show_grid is None:
        show_grid = len( c ) < 4000

    coll = PatchCollection(
        [ Rectangle( ( x - s / 2, y - s / 2 ), s, s ) for ( x, y ), s in zip( c, h ) ],
        cmap = SEQ, edgecolor = "white" if show_grid else "none",
        linewidth = 0.15 if show_grid else 0.0 )
    coll.set_array( w )
    coll.set_clim( 0.0, float( vmax if vmax is not None else max( w.max( initial = 0.0 ), 1e-30 ) ) )
    ax.add_collection( coll )

    if show_fov:
        ax.add_patch( Circle( ( 0, 0 ), mesh.inner_radius, fill = False, linestyle = "--",
                              linewidth = 1.0, edgecolor = VERMILLION ) )
    r = mesh.outer_radius * 1.02
    ax.set_xlim( -r, r ); ax.set_ylim( -r, r ); ax.set_aspect( "equal" )
    return coll


def plot_exterior_scale_scan( scan, ax = None, truth = None, extra = () ):
    """Le balayage de `mesh.scan_exterior_scale`, en LECTURE RELATIVE : chaque grandeur est
    normalisée par sa valeur maximale sur le balayage.

    Les grandeurs suivies n'ont ni la même unité ni le même ordre (une masse, une fraction de
    vide, un coût de transport) ; les mettre sur un seul axe brut n'aurait aucun sens, et deux
    axes des ordonnées inviteraient à comparer deux échelles arbitraires. Ce qu'on cherche à lire
    ici n'est de toute façon pas un niveau, c'est la présence -- ou l'absence -- d'un EXTREMUM.
    """
    ax = ax or plt.gca()
    a = scan[ "alphas" ]
    series = [ ( "vide du nuage", scan[ "void" ], BLUE ),
               ( "masse intérieure", scan[ "interior_mass" ], GREY ) ]
    series += [ ( name, np.asarray( values ), c )
                for ( name, values ), c in zip( extra, ( VERMILLION, GREEN ) ) ]
    for name, v, color in series:
        v = np.asarray( v, dtype = float )
        ax.plot( a, v / max( np.abs( v ).max(), 1e-30 ), color = color, linewidth = 1.6, label = name )
    if truth is not None:
        ax.axvline( truth, color = VERMILLION, linestyle = "--", linewidth = 1.0,
                    label = f"α optimal ({ truth:.2f})" )
    ax.set_xlabel( "α (facteur sur l'empreinte extérieure)", fontsize = 8 )
    ax.set_ylabel( "valeur / maximum du balayage", fontsize = 8 )
    ax.legend( fontsize = 8, frameon = False )
    ax.grid( alpha = 0.25, linewidth = 0.5 )
    ax.set_title( "balayage du facteur extérieur", fontsize = 9 )
    return ax


def plot_mesh_solution( mesh, sinogram = None, out = None, title = None ):
    """Les quatre vues de la solution sur maillage (voir la docstring du module)."""
    sino = sinogram if sinogram is not None else mesh.sinogram
    raw = np.asarray( sino.values, dtype = float )
    per_angle = raw.sum( axis = 1 ) * mesh.dw
    corrected = np.clip( raw - mesh.exterior_values(), 0.0, None ).sum( axis = 1 ) * mesh.dw
    deg = np.degrees( mesh.angles )
    vmax = float( mesh.weights.max( initial = 0.0 ) )

    fig, axes = plt.subplots( 2, 2, figsize = ( 12, 11 ) )

    coll = plot_mesh( mesh, ax = axes[ 0 ][ 0 ], vmax = vmax )
    fig.colorbar( coll, ax = axes[ 0 ][ 0 ], fraction = 0.046 )
    axes[ 0 ][ 0 ].set_title(
        f"solution sur { mesh.nb_cells } cellules -- masse { mesh.mass():.4g}\n"
        f"dont { mesh.interior_mass():.4g} dans le champ de vue (en tirets)", fontsize = 9 )

    # MÊME échelle de couleur que la vue complète : la comparaison visuelle n'a de sens qu'ainsi
    coll = plot_mesh( mesh, ax = axes[ 0 ][ 1 ], vmax = vmax, exterior_only = True )
    fig.colorbar( coll, ax = axes[ 0 ][ 1 ], fraction = 0.046 )
    axes[ 0 ][ 1 ].set_title(
        f"la part EXTÉRIEURE ({ int( ( ~mesh.interior ).sum() ) } cellules, "
        f"masse { mesh.mass() - mesh.interior_mass():.4g})\n"
        "-- c'est elle qu'on retire du sinogramme", fontsize = 9 )

    ax = axes[ 1 ][ 0 ]
    ax.plot( deg, per_angle, color = VERMILLION, linewidth = 1.6, label = "mesuré ∫p" )
    ax.plot( deg, corrected, color = BLUE, linewidth = 1.6, label = "corrigé ∫q" )
    ax.axhline( mesh.interior_mass(), color = GREY, linewidth = 1.0, linestyle = "--",
                label = "masse intérieure du maillage" )
    cv0 = per_angle.std() / per_angle.mean()
    cv1 = corrected.std() / max( corrected.mean(), 1e-30 )
    ax.set_title( f"masse par angle -- dispersion { cv0:.2%} → { cv1:.2%}", fontsize = 9 )
    ax.set_xlabel( "θ (deg)", fontsize = 8 ); ax.legend( fontsize = 8, frameon = False )
    ax.grid( alpha = 0.25, linewidth = 0.5 )

    ax = axes[ 1 ][ 1 ]
    im = ax.imshow( raw, aspect = "auto", origin = "lower", cmap = SEQ,
                    extent = [ mesh.s_min, mesh.s_min + mesh.nb_bins * mesh.dw, 0, 180 ] )
    fig.colorbar( im, ax = ax, fraction = 0.046 )
    ax.set_title( "sinogramme mesuré", fontsize = 9 )
    ax.set_xlabel( "s (détecteur)", fontsize = 8 ); ax.set_ylabel( "θ (deg)", fontsize = 8 )

    if title:
        fig.suptitle( title, fontsize = 11 )
    fig.tight_layout()
    if out:
        fig.savefig( out, dpi = 130 )
        print( f"figure sauvée: { out }" )
    return fig
