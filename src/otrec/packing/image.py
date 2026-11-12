import numpy as np


def rasterize( centers, radii, dim, nb, extent, radius, height ):
    """`[ nb ] * dim`, float32: the solid fraction of each pixel of the centred square ( cube ) of side `extent`, for the
    disc of radius `radius` ( in 3D: the cylinder of axis z and height `height` ) MINUS the balls -- linear in the signed
    distance across a surface"""
    h = extent / nb
    g = -extent / 2 + h * ( np.arange( nb ) + 0.5 )
    grids = np.meshgrid( *[ g ] * dim, indexing = "ij", sparse = True )
    outside = np.hypot( grids[ 0 ], grids[ 1 ] ) - radius
    if dim == 3:
        outside = np.maximum( outside, np.abs( grids[ 2 ] ) - height / 2 )
    solid = np.clip( 0.5 - outside / h, 0.0, 1.0 ).astype( np.float32 )
    for c, r in zip( centers, radii ):                                 # each ball digs its window of pixels
        lo = np.clip( np.floor( ( c - r + extent / 2 ) / h ).astype( int ) - 1, 0, nb )
        hi = np.clip( np.ceil( ( c + r + extent / 2 ) / h ).astype( int ) + 1, 0, nb )
        window = tuple( slice( a, b ) for a, b in zip( lo, hi ) )
        d = np.sqrt( sum( ( grids[ a ][ tuple( window[ a ] if b == a else slice( None ) for b in range( dim ) ) ] - c[ a ] ) ** 2 for a in range( dim ) ) )
        solid[ window ] = np.minimum( solid[ window ], np.clip( 0.5 + ( d - r ) / h, 0.0, 1.0 ) )
    return solid
