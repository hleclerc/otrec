"""The pages of a 3D reconstruction run ( `experiments/exp_3d.py`'s `run.npz` ), with our own generators:

- `cloud.html`: the cloud, snapshot by snapshot ( `sdot.viz.Visualizer`, one frame per snapshot ), colored blue -> white
  -> red by the PULL of the views ( their disagreement: blue, one point of space explains all the barycenters; red, the
  views pull the dirac apart; by the snapshot's quantiles );
- `mass.html`: the same, colored by the MASS DERIVATIVE ( blue: the dirac would rather grow; white: content; red: shrink;
  each snapshot on its own scale );
the points are drawn at half their snapshot's spacing ( neighbours touch; bigger when there are fewer );
- `slice_pull.html`, `slice_mass.html`: the same two colorings on the diracs of a slab `|z - z0| < dz`, in ( x, y );
the first frame of each page is the object itself, gray: its surface when `viz.blend` wrote `surface.npz` next to the
run, else a sample of it;
- `convergence.html`, `newton.html`: the curves ( `sdot.viz.convergence` );
- `index.html`: the run's numbers and links to the pages.

    python -m otrec.viz.view3d runs/.../run.npz            # writes the pages next to it
"""
import json
import sys
from pathlib import Path

import numpy as np

#: the color classes of a scalar field, blue -> white -> red ( a pair per class: on a light page, on a dark one )
_BWR = [ ( "#2166ac", "#4f8fd6" ), ( "#4393c3", "#6fb0e0" ), ( "#92c5de", "#a8d4ea" ), ( "#9a9a9a", "#f2f2f2" ),
         ( "#f4a582", "#f6b294" ), ( "#d6604d", "#e8735f" ), ( "#b2182b", "#e0303f" ) ]


def _edges( v, diverging ):
    """the 6 cuts of the 7 color classes of a snapshot's values: symmetric around 0 ( diverging ), or on quantiles, the
    median white ( sequential )"""
    v = np.asarray( v, dtype = float )
    v = v[ np.isfinite( v ) ]
    if v.size == 0:                                      # ( a snapshot without this value: a stage that does not compute it )
        return np.array( [ -0.6, -0.3, -0.1, 0.1, 0.3, 0.6 ] ) if diverging else np.linspace( 0.0, 1.0, 6 )
    if diverging:
        m = np.quantile( np.abs( v ), 0.98 ) or 1.0
        return np.array( [ -0.6, -0.3, -0.1, 0.1, 0.3, 0.6 ] ) * m
    return np.quantile( v, [ 0.05, 0.2, 0.4, 0.6, 0.8, 0.95 ] )


def _cylinder_edges( c, nb = 48 ):
    """the wireframe of a cylinder ( x, y, z, r, h, d ): two circles and a few generators"""
    t = np.linspace( 0, 2 * np.pi, nb, endpoint = False )
    ring = np.stack( [ c[ 0 ] + c[ 3 ] * np.cos( t ), c[ 1 ] + c[ 3 ] * np.sin( t ) ], 1 )
    lo = np.concatenate( [ ring, np.full( ( nb, 1 ), c[ 2 ] - c[ 4 ] / 2 ) ], 1 )
    hi = np.concatenate( [ ring, np.full( ( nb, 1 ), c[ 2 ] + c[ 4 ] / 2 ) ], 1 )
    pos = np.concatenate( [ lo, hi ] )
    e = [ ( i, ( i + 1 ) % nb ) for i in range( nb ) ] + [ ( nb + i, nb + ( i + 1 ) % nb ) for i in range( nb ) ]
    e += [ ( i, nb + i ) for i in range( 0, nb, nb // 8 ) ]
    return pos, np.array( e )


def _frames( d, field = None ):
    """the snapshots, `( step, points, values of field )`; their sizes change from a stage to the next ( `f_sizes` )"""
    sizes = d[ "f_sizes" ] if "f_sizes" in d else np.full( len( d[ "f_step" ] ), d[ "f_points" ].shape[ 1 ] )
    flat = d[ "f_points" ].reshape( -1, 3 )
    vals = None if field is None else d[ field ].reshape( -1 )
    cut = np.concatenate( [ [ 0 ], np.cumsum( sizes ) ] )
    return [ ( int( d[ "f_step" ][ f ] ), flat[ cut[ f ]:cut[ f + 1 ] ], None if vals is None else vals[ cut[ f ]:cut[ f + 1 ] ] )
             for f in range( len( sizes ) ) ]


def _cloud_page( d, field, diverging, path, max_points, slab = None ):
    """a first frame with a sample of the object ( gray ), then one frame per snapshot, its points in 7 color classes of
    `field`; in 3D, or `slab = ( z0, dz )`: only that slab, in ( x, y )"""
    from sdot.viz.Visualizer import Visualizer
    v = Visualizer( title = path.stem, frame_axis = "pas", frame_value = -1, fps = 2.0 )
    rng = np.random.default_rng( 0 )

    def cut( x ):
        keep = np.ones( len( x ), bool ) if slab is None else np.abs( x[ :, 2 ] - slab[ 0 ] ) < slab[ 1 ]
        idx = np.flatnonzero( keep )
        if len( idx ) > max_points:
            idx = np.sort( rng.choice( idx, max_points, replace = False ) )
        return idx

    def frame_extras():
        if slab is None:
            for cyl in d[ "cylinders" ]:
                pos, e = _cylinder_edges( cyl )
                v.add_edges( pos, e, color = "#888888", opacity = 0.6 )

    # the points at half their snapshot's spacing, neighbours touch: fewer diracs, each carrying more mass, are bigger
    # ( the page's slider scales them all )
    from scipy.spatial import cKDTree

    def radius( x ):
        return 0.5 * float( np.median( cKDTree( x ).query( x, k = 2 )[ 0 ][ :, 1 ] ) )

    xl = _frames( d )[ -1 ][ 1 ]
    rad = radius( xl )
    surf = path.with_name( "surface.npz" )
    if slab is None and surf.exists():   # the real object as a surface ( `viz.blend`: the isosurface of its voxels )
        sf = np.load( surf )
        v.add_faces( sf[ "vertices" ], sf[ "triangles" ], color = ( "#b8b8b4", "#c8c8c4" ) )
    else:
        ts = d[ "truth_sample" ]
        i = cut( ts )
        v.add_points( ts[ i ] if slab is None else ts[ i, :2 ], radius = rad, color = ( "#555555", "#bbbbbb" ) )
    frame_extras()
    for step, x, val in _frames( d, field ):
        v.new_frame( step )
        rad = radius( x )
        i = cut( x )
        x, val = ( x[ i ] if slab is None else x[ i, :2 ] ), val[ i ]
        cls = np.searchsorted( _edges( val, diverging ), val )   # each snapshot on its own scale
        for c, col in enumerate( _BWR ):
            m = cls == c
            if m.any():
                v.add_points( x[ m ], radius = rad, color = col )
        frame_extras()
    v.write_html( str( path ) )


def write_view( npz, out_dir = None, title = "Reconstruction 3D", max_points = 200000, slab = None ):
    from sdot.viz.convergence import write_convergence_html
    npz = Path( npz )
    out = npz.parent if out_dir is None else Path( out_dir )
    d = dict( np.load( npz ) )
    if "cylinders" not in d:
        d[ "cylinders" ] = np.zeros( ( 0, 6 ) )
    frames = _frames( d )
    n, nb_f = len( frames[ -1 ][ 1 ] ), len( frames )

    _cloud_page( d, "f_disagreement", False, out / "cloud.html", max_points )
    _cloud_page( d, "f_mass_grad", True, out / "mass.html", max_points )

    # a slab through the middle, as thin as two pixels or enough to hold ~5 % of the points
    z = frames[ -1 ][ 1 ][ :, 2 ]
    dz = slab or max( 2 * float( d[ "detector" ][ 3 ] ), float( np.quantile( np.abs( z - np.median( z ) ), 0.025 ) ) )
    z0 = float( np.median( z ) )
    _cloud_page( d, "f_disagreement", False, out / "slice_pull.html", max_points, slab = ( z0, dz ) )
    _cloud_page( d, "f_mass_grad", True, out / "slice_mass.html", max_points, slab = ( z0, dz ) )

    H = { k[ 2: ]: d[ k ] for k in d if k.startswith( "h_" ) }
    steps = H[ "step" ]

    def series( *keys ):
        return { k: [ ( int( s ), float( y ) ) for s, y in zip( steps, H[ k ] ) if np.isfinite( y ) ] for k in keys if k in H }

    write_convergence_html( series( "cost", "disagreement_mean", "dist_mean", "coverage", "disp_mean", "mass_grad_std" ),
                            out / "convergence.html", title = title, xlabel = "pas", ylabel = "" )
    write_convergence_html( series( "nb_iter", "nb_diag", "time", "nb_points" ), out / "newton.html", title = f"{ title } — Newton par vue",
                            xlabel = "pas", ylabel = "", log_y = False )
    if any( k.startswith( "t_" ) for k in H ):
        write_convergence_html( series( *[ k for k in H if k.startswith( "t_" ) ] ), out / "times.html", title = f"{ title } — temps cumulé par phase ( s )",
                                xlabel = "pas", ylabel = "", log_y = False )

    last = { k: float( v[ -1 ] ) for k, v in H.items() if np.issubdtype( np.asarray( v ).dtype, np.number ) }
    rows = "".join( f"<tr><td>{ k }</td><td>{ v:.4g}</td></tr>" for k, v in last.items() )
    links = "".join( f'<li><a href="{ p }">{ p }</a> — { t }</li>' for p, t in [
        ( "cloud.html", "<b>le nuage en 3D</b>, coloré par le tiraillement des vues ( bleu : d’accord, rouge : tiraillé ) ; 1re image : l’objet" ),
        ( "mass.html", "le nuage en 3D, coloré par la dérivée en masse ( bleu : grossir, blanc : rien, rouge : maigrir )" ),
        ( "slice_pull.html", "une tranche en z, tiraillement" ), ( "slice_mass.html", "une tranche en z, dérivée en masse" ),
        ( "convergence.html", "les courbes" ), ( "newton.html", "le coût des solves, et le nombre de diracs" ) ]
        + ( [ ( "times.html", "le temps cumulé de chaque phase de la perte grille" ) ] if any( k.startswith( "t_" ) for k in H ) else [ ] ) )
    ( out / "index.html" ).write_text( f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>{ title }</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>:root{{--bg:#fff;--fg:#111;--mu:#666}}@media (prefers-color-scheme:dark){{:root{{--bg:#16171a;--fg:#eee;--mu:#999}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px}}
a{{color:#4a8be0}}td{{padding:2px 12px 2px 0;font-variant-numeric:tabular-nums}}td:first-child{{color:var(--mu)}}</style></head>
<body><h1 style="font-size:20px">{ title }</h1><p style="color:var(--mu)">{ n:,} diracs à la fin, { nb_f } instantanés</p><ul>{ links }</ul>
<h2 style="font-size:16px">Dernier pas</h2><table>{ rows }</table></body></html>""" )
    return out / "index.html"


if __name__ == "__main__":
    print( write_view( sys.argv[ 1 ] ) )
