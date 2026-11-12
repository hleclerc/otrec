"""What the MESH reconstruction (`mesh.GradedMesh`) found.

`plot_mesh` draws the solution as it is: one cell = one rectangle, its density = its
color. No resampling onto a regular grid, which would hide precisely what we
want to see -- the grading, and the fact that the exterior is represented by very few cells.

`plot_mesh_solution` adds the three views that tell whether the solution holds: the mass per angle
before/after removing the exterior (it should flatten), the measured sinogram, and the share that the
mesh attributes to the exterior.

Palette: see `viz.style`.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle

from .style import BLUE, GREEN, GREY, SEQ, VERMILLION


def plot_mesh( mesh, ax = None, weights = None, vmax = None, show_grid = None,
               exterior_only = False, show_fov = True ):
    """The mesh colored by its density. Returns the `PatchCollection` (for the colorbar).

    `show_grid`: draws the cell edges. By default only below 4000 cells --
    beyond that the lines cover the data instead of structuring it.
    `exterior_only`: only displays what will be REMOVED from the sinogram, which the Diracs
    will therefore never see.
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
    """The sweep of `mesh.scan_exterior_scale`, as a RELATIVE READING: each quantity is
    normalized by its maximum value over the sweep.

    The tracked quantities have neither the same unit nor the same order (a mass, a void
    fraction, a transport cost); putting them on a single raw axis would make no sense, and two
    y axes would invite comparing two arbitrary scales. What we are trying to read
    here is in any case not a level, but the presence -- or absence -- of an EXTREMUM.
    """
    ax = ax or plt.gca()
    a = scan[ "alphas" ]
    series = [ ( "cloud void", scan[ "void" ], BLUE ),
               ( "interior mass", scan[ "interior_mass" ], GREY ) ]
    series += [ ( name, np.asarray( values ), c )
                for ( name, values ), c in zip( extra, ( VERMILLION, GREEN ) ) ]
    for name, v, color in series:
        v = np.asarray( v, dtype = float )
        ax.plot( a, v / max( np.abs( v ).max(), 1e-30 ), color = color, linewidth = 1.6, label = name )
    if truth is not None:
        ax.axvline( truth, color = VERMILLION, linestyle = "--", linewidth = 1.0,
                    label = f"optimal α ({ truth:.2f})" )
    ax.set_xlabel( "α (factor on the exterior footprint)", fontsize = 8 )
    ax.set_ylabel( "value / sweep maximum", fontsize = 8 )
    ax.legend( fontsize = 8, frameon = False )
    ax.grid( alpha = 0.25, linewidth = 0.5 )
    ax.set_title( "exterior factor sweep", fontsize = 9 )
    return ax


def plot_mesh_solution( mesh, sinogram = None, out = None, title = None ):
    """The four views of the mesh solution (see the module docstring)."""
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
        f"solution on { mesh.nb_cells } cells -- mass { mesh.mass():.4g}\n"
        f"of which { mesh.interior_mass():.4g} in the field of view (dashed)", fontsize = 9 )

    # SAME color scale as the full view: the visual comparison only makes sense that way
    coll = plot_mesh( mesh, ax = axes[ 0 ][ 1 ], vmax = vmax, exterior_only = True )
    fig.colorbar( coll, ax = axes[ 0 ][ 1 ], fraction = 0.046 )
    axes[ 0 ][ 1 ].set_title(
        f"the EXTERIOR part ({ int( ( ~mesh.interior ).sum() ) } cells, "
        f"mass { mesh.mass() - mesh.interior_mass():.4g})\n"
        "-- this is what is removed from the sinogram", fontsize = 9 )

    ax = axes[ 1 ][ 0 ]
    ax.plot( deg, per_angle, color = VERMILLION, linewidth = 1.6, label = "measured ∫p" )
    ax.plot( deg, corrected, color = BLUE, linewidth = 1.6, label = "corrected ∫q" )
    ax.axhline( mesh.interior_mass(), color = GREY, linewidth = 1.0, linestyle = "--",
                label = "interior mass of the mesh" )
    cv0 = per_angle.std() / per_angle.mean()
    cv1 = corrected.std() / max( corrected.mean(), 1e-30 )
    ax.set_title( f"mass per angle -- spread { cv0:.2%} → { cv1:.2%}", fontsize = 9 )
    ax.set_xlabel( "θ (deg)", fontsize = 8 ); ax.legend( fontsize = 8, frameon = False )
    ax.grid( alpha = 0.25, linewidth = 0.5 )

    ax = axes[ 1 ][ 1 ]
    im = ax.imshow( raw, aspect = "auto", origin = "lower", cmap = SEQ,
                    extent = [ mesh.s_min, mesh.s_min + mesh.nb_bins * mesh.dw, 0, 180 ] )
    fig.colorbar( im, ax = ax, fraction = 0.046 )
    ax.set_title( "measured sinogram", fontsize = 9 )
    ax.set_xlabel( "s (detector)", fontsize = 8 ); ax.set_ylabel( "θ (deg)", fontsize = 8 )

    if title:
        fig.suptitle( title, fontsize = 11 )
    fig.tight_layout()
    if out:
        fig.savefig( out, dpi = 130 )
        print( f"figure saved: { out }" )
    return fig
