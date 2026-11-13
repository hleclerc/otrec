"""The frames of a `Recorder` as one standalone HTML page, written with `sdot.Visualizer`: one frame per recorded iteration
( the play bar, the arrows ), the shapes drawn as discs of their own radius, over the truth when it is given; and, in the
panel of the curves, the loss ( above its floor ), the metrics of the recorder, the blur and the number of shapes, with a cursor at the
displayed frame ( a click in a chart goes to that iteration )."""
import numpy as np

CLOUD = ( "#1f5fa8", "#8ab4f8" )
TRUTH = ( "#c0392b", "#ff8a7a" )


def contours( image, extent, level = 0.5 ):
    """`( positions [ m, 2 ], edges [ m / 2, 2 ] )`: the lines where `image [ nx, ny ]` ( on the centred square of side
    `extent` ) crosses `level`, as segments ( marching squares ), for `truth_edges`"""
    v = np.asarray( image, dtype = float ) - level
    h = extent / v.shape[ 0 ]
    X, Y = np.meshgrid( *[ -extent / 2 + h * ( np.arange( n ) + 0.5 ) for n in v.shape ], indexing = "ij" )
    a, b, c, d = v[ :-1, :-1 ], v[ 1:, :-1 ], v[ 1:, 1: ], v[ :-1, 1: ]                     # the corners of each cell
    x0, y0 = X[ :-1, :-1 ], Y[ :-1, :-1 ]
    cut = lambda p, q: p / np.where( p == q, 1.0, p - q )                                 # where a side crosses, in [ 0, 1 ]
    sides = [ ( ( a < 0 ) != ( b < 0 ), x0 + h * cut( a, b ), y0 ), ( ( b < 0 ) != ( c < 0 ), x0 + h, y0 + h * cut( b, c ) ),
              ( ( d < 0 ) != ( c < 0 ), x0 + h * cut( d, c ), y0 + h ), ( ( a < 0 ) != ( d < 0 ), x0, y0 + h * cut( a, d ) ) ]
    two = sum( s[ 0 ].astype( int ) for s in sides ) == 2
    pos = [ ]
    for i in range( 4 ):
        for j in range( i + 1, 4 ):
            m = sides[ i ][ 0 ] & sides[ j ][ 0 ] & ( two | ( j == i + 1 ) & ( i != 1 ) )   # ( a saddle: two segments )
            pos.append( np.stack( [ sides[ i ][ 1 ][ m ], sides[ i ][ 2 ][ m ], sides[ j ][ 1 ][ m ], sides[ j ][ 2 ][ m ] ], 1 ).reshape( -1, 2 ) )
    pos = np.concatenate( pos )
    return pos, np.arange( len( pos ) ).reshape( -1, 2 )


def export_html( rec, out_path, title = None, truth_edges = None, truth_points = None, fps = 6.0 ):
    """`rec`: a `Reconstruction` with a recorder. `truth_edges`: `( positions, edges )`, the outline of the object ( 2D: see
    `contours` ); `truth_points`: `[ m, dim ]`, a sample of it ( 3D ). A metric whose name ends with "( log )" is drawn on
    a log scale."""
    from sdot.viz.Visualizer import Visualizer
    frames, dim = rec.recorder.frames, rec.dim
    its = [ fr[ "iteration" ] for fr in frames ]
    viz = Visualizer( title = title or f"otrec { dim }D", frame_axis = "iteration", frame_value = its[ 0 ], fps = fps, flat_points = dim == 2 )
    if truth_edges is not None:
        truth_edges = ( np.asarray( truth_edges[ 0 ], dtype = np.float32 ), np.asarray( truth_edges[ 1 ] ) )
    if truth_points is not None:
        truth_points = np.asarray( truth_points, dtype = np.float32 )
    for f, fr in enumerate( frames ):
        if f:
            viz.new_frame( fr[ "iteration" ] )
        viz.add_points( fr[ "points" ], radius = fr[ "radii" ], color = CLOUD, opacity = 0.6 if dim == 2 else 1.0 )
        if truth_points is not None:                      # ( the same arrays at each frame: stored once )
            viz.add_points( truth_points, color = TRUTH, opacity = 0.35 )
        if truth_edges is not None:
            viz.add_edges( truth_edges[ 0 ], truth_edges[ 1 ], color = TRUTH )
    # the loss ends on a floor it cannot go under ( `Loss1d.floor` ): on a log scale, only what is ABOVE it shows the convergence
    floor = rec.loss_fn.floor
    viz.add_curve( "loss above its floor" if floor else "loss", [ f - floor for _, f in rec.recorder.losses ], [ it for it, _ in rec.recorder.losses ], log_y = True )
    for name in frames[ 0 ][ "metrics" ]:
        log = name.endswith( "( log )" )
        viz.add_curve( name[ : -len( "( log )" ) ].strip() if log else name, [ fr[ "metrics" ][ name ] for fr in frames ], its, log_y = log )
    viz.add_curve( "blur", [ fr[ "sigma" ] for fr in frames ], its )
    viz.add_curve( "number of shapes", [ fr[ "nb_points" ] for fr in frames ], its )
    viz.add_curve( "median radius of the shapes", [ float( np.median( fr[ "radii" ] ) ) for fr in frames ], its )
    viz.write_html( str( out_path ) )
    return viz
