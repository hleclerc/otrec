"""The HALO dashboard: what the fit found, and whether to believe it.

Six views, in the order they are read:

1. the MESH with its density per cell, the interior cloud on top -- the solution
   itself, as it lives in space;
2. the MASS PER ANGLE before/after correction: `∫p_θ` should go from a varying curve to a
   straight line. This is the most direct diagnostic of the leakage (see `halo.mass_profile`);
3. the VISIBLE mass of the halo against its target `∫p_θ − M_in`: the hard anchor of the fit, hence
   the first place to look if the result disappoints;
4-6. the MEASURED sinogram, the FOOTPRINT found, and the final RESIDUAL (measured − interior − halo).

It is the residual (6) that tells whether the mesh is sufficient: noise without structure = the halo did its
job; coherent bands = degrees of freedom are missing where they appear.

Palette: see `viz.style`.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge

from .style import BLUE as _BLUE, DIV as _DIV, GREEN as _GREEN, GREY as _GREY
from .style import SEQ as _SEQ, VERMILLION as _VERMILLION


def plot_halo_mesh( halo, ax = None, points = None, max_points = 20_000, show_grid = True ):
    """The log-polar mesh colored by the fitted density, with the interior cloud on top.

    Each cell is an annular sector (`matplotlib.patches.Wedge`); it is the density that is
    colored, not the mass -- two cells of the same color therefore represent very different
    masses, since areas grow strongly with radius. The dashed circle is the edge of the
    field of view: nothing of the halo enters it, nothing of the cloud should leave it.
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
    ax.set_title( f"halo: density on { halo.nb_cells } cells\n"
                  f"(mass { halo.mass():.3g}, field of view dashed)", fontsize = 9 )
    return coll


def _sinogram_image( ax, values, halo, title, cmap = _SEQ, symmetric = False ):
    v = np.asarray( values, dtype = float )
    kw = dict( vmin = -np.abs( v ).max(), vmax = np.abs( v ).max() ) if symmetric else {}
    im = ax.imshow( v, aspect = "auto", origin = "lower", cmap = cmap, **kw,
                    extent = [ halo.s_min, halo.s_min + halo.nb_bins * halo.dw, 0, 180 ] )
    ax.set_title( title, fontsize = 9 )
    ax.set_xlabel( "s (detector)", fontsize = 8 ); ax.set_ylabel( "θ (deg)", fontsize = 8 )
    return im


def plot_interior_mass_scan( scan, ax = None, truth = None ):
    """The sweep of `halo.scan_interior_mass`: mass recovered by the halo, and spread of
    `∫q_θ` after correction, as a function of `M_in`.

    A single quantity on the y axis (a mass): the spread, being incommensurable, only appears
    through the vertical line at its minimum -- a second y axis would invite comparing two
    arbitrary scales.
    """
    ax = ax or plt.gca()
    m = scan[ "masses" ]
    ax.plot( m, scan[ "halo_mass" ], color = _GREEN, linewidth = 1.6, label = "halo mass" )
    ax.set_xlabel( "M_in (mass attributed to the interior)", fontsize = 8 )
    ax.set_ylabel( "halo mass", fontsize = 8 )
    ax.grid( alpha = 0.25, linewidth = 0.5 )

    best = m[ int( np.argmin( scan[ "dispersion" ] ) ) ]
    ax.axvline( best, color = _BLUE, linestyle = ":", linewidth = 1.2,
                label = f"minimal ∫q spread ({ best:.3g})" )
    if truth is not None:
        ax.axvline( truth, color = _GREY, linestyle = "--", linewidth = 1.0, label = f"true M_in ({ truth:.3g})" )
    ax.legend( fontsize = 8, frameon = False )
    ax.set_title( "interior mass sweep", fontsize = 9 )
    return ax


def plot_halo( halo, points = None, interior_mass = None, radius = None, sinogram = None,
               out = None, title = None, max_points = 20_000 ):
    """The complete dashboard (see the module docstring). Returns the figure.

    `points`: the interior cloud -- typically `rec.positions`. Without it, the panels that
    depend on it (the superimposed cloud, the final residual) are simply omitted.
    `interior_mass` / `radius`: what was passed to `halo.alternate`, so that the displayed residual
    is THE ONE the fit saw. `interior_mass` defaults to `min_θ ∫p_θ`, as there.
    `out`: path to write the PNG (optional).
    """
    # late import: `halo` pulls in `Reconstruction`, which pulls in `viz.points_html` -- importing it at
    # the top of a `viz` module would work today, but loops as soon as `viz/__init__` exposes
    # anything. This module only needs `halo` for plotting.
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
    ax.plot( deg, per_angle, color = _VERMILLION, linewidth = 1.6, label = "measured ∫p" )
    ax.plot( deg, corrected, color = _BLUE, linewidth = 1.6, label = "corrected ∫q" )
    ax.axhline( m_in, color = _GREY, linewidth = 1.0, linestyle = "--", label = "target M_in" )
    cv0, cv1 = per_angle.std() / per_angle.mean(), corrected.std() / max( corrected.mean(), 1e-30 )
    ax.set_title( f"mass per angle -- spread { cv0:.1%} → { cv1:.1%}", fontsize = 9 )
    ax.set_xlabel( "θ (deg)", fontsize = 8 ); ax.legend( fontsize = 8, frameon = False )
    ax.grid( alpha = 0.25, linewidth = 0.5 )

    ax = axes[ 0 ][ 2 ]
    ax.plot( deg, np.maximum( per_angle - m_in, 0.0 ), color = _GREY, linewidth = 1.6,
             label = "target ∫p − M_in" )
    ax.plot( deg, halo.visible_mass(), color = _GREEN, linewidth = 1.6, label = "visible halo" )
    ax.set_title( "halo mass falling in the detector", fontsize = 9 )
    ax.set_xlabel( "θ (deg)", fontsize = 8 ); ax.legend( fontsize = 8, frameon = False )
    ax.grid( alpha = 0.25, linewidth = 0.5 )

    fig.colorbar( _sinogram_image( axes[ 1 ][ 0 ], raw, halo, "measured sinogram" ),
                  ax = axes[ 1 ][ 0 ], fraction = 0.046 )
    fig.colorbar( _sinogram_image( axes[ 1 ][ 1 ], footprint, halo, "halo footprint" ),
                  ax = axes[ 1 ][ 1 ], fraction = 0.046 )

    ax = axes[ 1 ][ 2 ]
    if points is None:
        ax.set_axis_off()
        ax.text( 0.5, 0.5, "residual: provide `points`", ha = "center", va = "center", fontsize = 9 )
    else:
        inside = interior_values( sino, points, m_in, radius = radius, max_points = max_points )
        res = raw - inside - footprint
        rel = np.abs( res ).max() / max( raw.max(), 1e-30 )
        fig.colorbar( _sinogram_image( ax, res, halo, f"residual measured − interior − halo "
                                       f"(max { rel:.1%} of the signal)", cmap = _DIV, symmetric = True ),
                      ax = ax, fraction = 0.046 )

    if title:
        fig.suptitle( title, fontsize = 11 )
    fig.tight_layout()
    if out:
        fig.savefig( out, dpi = 130 )
        print( f"figure saved: { out }" )
    return fig
