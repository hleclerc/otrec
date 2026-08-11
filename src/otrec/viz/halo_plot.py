"""Le tableau de bord du HALO : ce que l'ajustement a trouvé, et s'il faut le croire.

Six vues, dans l'ordre où on les lit :

1. le MAILLAGE avec sa densité par cellule, le nuage intérieur par-dessus -- la solution
   elle-même, telle qu'elle vit dans l'espace ;
2. la MASSE PAR ANGLE avant/après correction : `∫p_θ` doit passer d'une courbe variable à une
   droite. C'est le diagnostic le plus direct de la fuite (voir `halo.mass_profile`) ;
3. la masse VISIBLE du halo face à sa cible `∫p_θ − M_in` : l'ancrage dur de l'ajustement, donc
   le premier endroit où regarder si le résultat déçoit ;
4-6. le sinogramme MESURÉ, l'EMPREINTE trouvée, et le RÉSIDU final (mesuré − intérieur − halo).

C'est le résidu (6) qui dit si le maillage suffit : du bruit sans structure = le halo a fait son
travail ; des bandes cohérentes = il manque des degrés de liberté là où elles apparaissent.

Palette : voir `viz.style`.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge

from .style import BLUE as _BLUE, DIV as _DIV, GREEN as _GREEN, GREY as _GREY
from .style import SEQ as _SEQ, VERMILLION as _VERMILLION


def plot_halo_mesh( halo, ax = None, points = None, max_points = 20_000, show_grid = True ):
    """Le maillage log-polaire colorié par la densité ajustée, et le nuage intérieur par-dessus.

    Chaque cellule est un secteur d'anneau (`matplotlib.patches.Wedge`) ; c'est la densité qui est
    coloriée, pas la masse -- deux cellules de même couleur représentent donc des masses très
    différentes, les aires croissant fortement avec le rayon. Le cercle en tirets est le bord du
    champ de vue : rien du halo n'entre dedans, rien du nuage ne devrait en sortir.
    """
    ax = ax or plt.gca()
    w = np.asarray( halo.weights, dtype = float )
    vmax = float( w.max() ) if w.size and w.max() > 0 else 1.0

    wedges = [ Wedge( ( 0, 0 ), b, np.degrees( p0 ), np.degrees( p1 ), width = b - a )
               for a, b, p0, p1, _ in halo.cells ]
    coll = PatchCollection( wedges, cmap = _SEQ, edgecolor = "white" if show_grid else "none",
                            linewidth = 0.4 if show_grid else 0.0 )
    coll.set_array( w )
    coll.set_clim( 0.0, vmax )
    ax.add_collection( coll )

    if points is not None and len( points ) :
        pts = np.asarray( points, dtype = float )
        if len( pts ) > max_points:
            pts = pts[ np.random.default_rng( 0 ).choice( len( pts ), max_points, replace = False ) ]
        ax.plot( pts[ :, 0 ], pts[ :, 1 ], ".", markersize = 1.0, color = "#222222", alpha = 0.6 )

    ax.add_patch( Circle( ( 0, 0 ), halo.inner_radius, fill = False, linestyle = "--",
                          linewidth = 1.0, edgecolor = _VERMILLION ) )
    r = halo.outer_radius * 1.03
    ax.set_xlim( -r, r ); ax.set_ylim( -r, r ); ax.set_aspect( "equal" )
    ax.set_title( f"halo : densité sur { halo.nb_cells } cellules\n"
                  f"(masse { halo.mass():.3g}, champ de vue en tirets)", fontsize = 9 )
    return coll


def _sinogram_image( ax, values, halo, title, cmap = _SEQ, symmetric = False ):
    v = np.asarray( values, dtype = float )
    kw = dict( vmin = -np.abs( v ).max(), vmax = np.abs( v ).max() ) if symmetric else {}
    im = ax.imshow( v, aspect = "auto", origin = "lower", cmap = cmap, **kw,
                    extent = [ halo.s_min, halo.s_min + halo.nb_bins * halo.dw, 0, 180 ] )
    ax.set_title( title, fontsize = 9 )
    ax.set_xlabel( "s (détecteur)", fontsize = 8 ); ax.set_ylabel( "θ (deg)", fontsize = 8 )
    return im


def plot_interior_mass_scan( scan, ax = None, truth = None ):
    """Le balayage de `halo.scan_interior_mass` : masse récupérée par le halo, et dispersion de
    `∫q_θ` après correction, en fonction de `M_in`.

    Une seule grandeur en ordonnée (une masse) : la dispersion, sans commune mesure, n'apparaît
    que par la verticale de son minimum -- un second axe des ordonnées inviterait à comparer deux
    échelles arbitraires.
    """
    ax = ax or plt.gca()
    m = scan[ "masses" ]
    ax.plot( m, scan[ "halo_mass" ], color = _GREEN, linewidth = 1.6, label = "masse du halo" )
    ax.set_xlabel( "M_in (masse attribuée à l'intérieur)", fontsize = 8 )
    ax.set_ylabel( "masse du halo", fontsize = 8 )
    ax.grid( alpha = 0.25, linewidth = 0.5 )

    best = m[ int( np.argmin( scan[ "dispersion" ] ) ) ]
    ax.axvline( best, color = _BLUE, linestyle = ":", linewidth = 1.2,
                label = f"dispersion ∫q minimale ({ best:.3g})" )
    if truth is not None:
        ax.axvline( truth, color = _GREY, linestyle = "--", linewidth = 1.0, label = f"vrai M_in ({ truth:.3g})" )
    ax.legend( fontsize = 8, frameon = False )
    ax.set_title( "balayage de la masse intérieure", fontsize = 9 )
    return ax


def plot_halo( halo, points = None, interior_mass = None, radius = None, sinogram = None,
               out = None, title = None, max_points = 20_000 ):
    """Le tableau de bord complet (voir la docstring du module). Renvoie la figure.

    `points` : le nuage intérieur -- typiquement `rec.positions`. Sans lui, les panneaux qui en
    dépendent (le nuage superposé, le résidu final) sont simplement omis.
    `interior_mass` / `radius` : ce qui a été passé à `halo.alternate`, pour que le résidu affiché
    soit CELUI que l'ajustement a vu. `interior_mass` vaut par défaut `min_θ ∫p_θ`, comme là-bas.
    `out` : chemin d'écriture du PNG (optionnel).
    """
    # import tardif : `halo` tire `Reconstruction`, qui tire `viz.points_html` -- l'importer en
    # tête d'un module de `viz` marcherait aujourd'hui, mais boucle dès que `viz/__init__` expose
    # quoi que ce soit. Ce module n'a besoin de `halo` que pour tracer.
    from ..halo import interior_values, mass_profile

    sino = sinogram if sinogram is not None else halo.sinogram
    raw = np.asarray( sino.values, dtype = float )
    footprint = halo.values()
    per_angle = mass_profile( sino )
    m_in = float( per_angle.min() ) if interior_mass is None else float( interior_mass )
    deg = np.degrees( halo.angles )

    fig, axes = plt.subplots( 2, 3, figsize = ( 15, 9 ) )

    coll = plot_halo_mesh( halo, ax = axes[ 0 ][ 0 ], points = points, max_points = max_points )
    fig.colorbar( coll, ax = axes[ 0 ][ 0 ], fraction = 0.046 )

    ax = axes[ 0 ][ 1 ]
    corrected = np.clip( raw - footprint, 0.0, None ).sum( axis = 1 ) * halo.dw
    ax.plot( deg, per_angle, color = _VERMILLION, linewidth = 1.6, label = "mesuré ∫p" )
    ax.plot( deg, corrected, color = _BLUE, linewidth = 1.6, label = "corrigé ∫q" )
    ax.axhline( m_in, color = _GREY, linewidth = 1.0, linestyle = "--", label = "M_in visé" )
    cv0, cv1 = per_angle.std() / per_angle.mean(), corrected.std() / max( corrected.mean(), 1e-30 )
    ax.set_title( f"masse par angle -- dispersion { cv0:.1%} → { cv1:.1%}", fontsize = 9 )
    ax.set_xlabel( "θ (deg)", fontsize = 8 ); ax.legend( fontsize = 8, frameon = False )
    ax.grid( alpha = 0.25, linewidth = 0.5 )

    ax = axes[ 0 ][ 2 ]
    ax.plot( deg, np.maximum( per_angle - m_in, 0.0 ), color = _GREY, linewidth = 1.6,
             label = "cible ∫p − M_in" )
    ax.plot( deg, halo.visible_mass(), color = _GREEN, linewidth = 1.6, label = "halo visible" )
    ax.set_title( "masse du halo tombant dans le détecteur", fontsize = 9 )
    ax.set_xlabel( "θ (deg)", fontsize = 8 ); ax.legend( fontsize = 8, frameon = False )
    ax.grid( alpha = 0.25, linewidth = 0.5 )

    fig.colorbar( _sinogram_image( axes[ 1 ][ 0 ], raw, halo, "sinogramme mesuré" ),
                  ax = axes[ 1 ][ 0 ], fraction = 0.046 )
    fig.colorbar( _sinogram_image( axes[ 1 ][ 1 ], footprint, halo, "empreinte du halo" ),
                  ax = axes[ 1 ][ 1 ], fraction = 0.046 )

    ax = axes[ 1 ][ 2 ]
    if points is None:
        ax.set_axis_off()
        ax.text( 0.5, 0.5, "résidu : fournir `points`", ha = "center", va = "center", fontsize = 9 )
    else:
        inside = interior_values( sino, points, m_in, radius = radius, max_points = max_points )
        res = raw - inside - footprint
        rel = np.abs( res ).max() / max( raw.max(), 1e-30 )
        fig.colorbar( _sinogram_image( ax, res, halo, f"résidu mesuré − intérieur − halo "
                                       f"(max { rel:.1%} du signal)", cmap = _DIV, symmetric = True ),
                      ax = ax, fraction = 0.046 )

    if title:
        fig.suptitle( title, fontsize = 11 )
    fig.tight_layout()
    if out:
        fig.savefig( out, dpi = 130 )
        print( f"figure sauvée: { out }" )
    return fig
