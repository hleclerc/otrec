"""The cloud of shapes as arrays: where it starts, how it is refined."""
import numpy as np


def uniform_in_ball( rng, n, dim, radius ):
    d = rng.normal( size = ( n, dim ) )
    d /= np.linalg.norm( d, axis = 1, keepdims = True )
    return radius * d * rng.uniform( size = ( n, 1 ) ) ** ( 1 / dim )


def uniform_in_cylinder( rng, n, dim, radius, height = None ):
    """in the disk ( 2D ), or the cylinder of axis z and height `height`, centred ( 3D )"""
    rho, phi = radius * np.sqrt( rng.random( n ) ), 2 * np.pi * rng.random( n )
    cols = [ rho * np.cos( phi ), rho * np.sin( phi ) ]
    if dim == 3:
        cols.append( rng.uniform( -height / 2, height / 2, n ) )
    return np.stack( cols, 1 )


def split( rng, points, division, noise, n = None ):
    """each point becomes `division` points, `noise` around it; with `n`, only `n` of them are kept ( drawn at random )"""
    children = np.repeat( points, division, axis = 0 )
    children += 0.5 * noise * rng.normal( size = children.shape )
    if n is not None and n < len( children ):
        children = children[ np.sort( rng.choice( len( children ), n, replace = False ) ) ]
    return children


def spacing( points, nb = 3 ):
    """the mean distance from a point to its `nb` nearest neighbours. On a cloud a stage has settled in full matter,
    `spacing^2` is the area each point stands for, within 10 % ( 2D, measured on packings; the neighbours of a point of a
    border are along it and inside: the borders hardly count )"""
    from scipy.spatial import cKDTree
    return float( cKDTree( points ).query( points, min( nb, len( points ) - 1 ) + 1 )[ 0 ][ :, 1: ].mean() )


#: `first_peak^2` over the room of a point, 2D: 1.80 on a packing ( the cloud of a multiscale, against the area of the
#: matter ), 1.76 on a foam ( against the area of its filtered back projection )
PEAK_ROOM = 1.8


def first_peak( points, k = 6, nb = 60 ):
    """the distance between NEIGHBOURS: where the distances from each point to its `k` nearest ones are the most
    frequent ( a histogram of `nb` bins on a log scale, smoothed over 5 ), the first peak of the distribution of the
    distances between points. Coincident points are left out. On a cloud spread as the matter, the room of each point is
    `first_peak^2 / PEAK_ROOM` ( 2D )"""
    from scipy.spatial import cKDTree
    d = cKDTree( points ).query( points, min( k, len( points ) - 1 ) + 1 )[ 0 ][ :, 1: ].ravel()
    d = d[ d > 1e-12 * max( float( d.max() ), 1e-300 ) ]
    edges = np.geomspace( np.quantile( d, 0.001 ), np.quantile( d, 0.999 ), nb + 1 )
    h = np.convolve( np.histogram( d, edges )[ 0 ] / np.diff( np.log( edges ) ), np.ones( 5 ) / 5, mode = "same" )
    i = int( np.argmax( h ) )
    return float( np.sqrt( edges[ i ] * edges[ i + 1 ] ) )


def sizes( n_start, n_end, division ):
    """the numbers of points of the stages of a multiscale: `n_end` at the end, each one the next divided by `division`
    ( rounded up ), from the one nearest to `n_start`"""
    out = [ n_end ]
    while out[ 0 ] > n_start * division ** 0.5 and out[ 0 ] > 1:
        out.insert( 0, -( -out[ 0 ] // division ) )
    return out
