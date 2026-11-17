"""What a page of otrec draws besides the shapes ( the page itself: `Recorder.write_html` ): the outline of the object
( `contours` ), the colours of the cloud and of the truth."""
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

