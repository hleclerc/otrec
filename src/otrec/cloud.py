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


def sizes( n_start, n_end, division ):
    """the numbers of points of the stages of a multiscale: `n_end` at the end, each one the next divided by `division`
    ( rounded up ), from the one nearest to `n_start`"""
    out = [ n_end ]
    while out[ 0 ] > n_start * division ** 0.5 and out[ 0 ] > 1:
        out.insert( 0, -( -out[ 0 ] // division ) )
    return out
