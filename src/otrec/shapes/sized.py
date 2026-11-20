"""SIZED balls ( what a cloud is once its shapes differ: its image, its density ): each one with a radius of its own, and a mass that FOLLOWS its size ( `radius^dim`: the same density at
the centre for all, the masses normalized to 1 in all ). World 2D. A cloud is ONE array `x [ n, 3 ]`, made to be
optimized as it stands ( no constraint, every column a length ):

    x[ :, :2 ]    the centre
    x[ :, 2 ]     SCALE r0 log( radius / r0 ): 0 is the radius `r0`
    x[ :, 3 ]     ( optional, `x [ n, 4 ]` ) DENSITY r0 log( density ): a density of its own, 0 is 1. The mass is then
                  `density radius^2`. `DENSITY = SCALE / 2`: a step of the density moves the mass as much as a step of
                  the radius ( the mass goes as the square of the radius ). Kept in `[ 1 / max_ratio^2, max_ratio^2 ]`

The radii are kept in `[ r0 / max_ratio, r0 max_ratio ]` ( what one shape may cover: its cost ) and the balls
inside the observable domain. The kernels are those of ellipses with a mass each ( `fused` ): a ball is a round one."""
import math

import numpy as np
import torch

from . import fused
from .balls import inside

SCALE = 10.0
DENSITY = SCALE / 2


def pack( centres, radii, r0, densities = None ):
    cols = [ centres, SCALE * r0 * torch.log( radii / r0 )[ :, None ] ]
    if densities is not None:
        cols.append( DENSITY * r0 * torch.log( densities )[ :, None ] )
    return torch.cat( cols, 1 )


def weights( x, model, column = 3 ):
    """`[ n ]`: the density of each shape, 1 without the column ( `x [ n, column ]` )"""
    if x.shape[ 1 ] <= column:
        return torch.ones_like( x[ :, 0 ] )
    m = model.max_ratio ** 2
    return torch.exp( x[ :, column ] / ( DENSITY * model.radius ) ).clamp( 1 / m, m )


def discs( x, model ):
    """`( centres, radii )`: the cloud these rows are"""
    r0 = model.radius
    r = ( r0 * torch.exp( x[ :, 2 ] / ( SCALE * r0 ) ) ).clamp( r0 / model.max_ratio, r0 * model.max_ratio )
    return inside( x[ :, :2 ], r, model.limit ), r


def bounded( x, model, column, scale, ratio ):
    """the column `column` ( `scale r0 log( value / r0 )` ) kept where `value / r0` is in `[ 1 / ratio, ratio ]`"""
    r0, lim = model.radius, scale * model.radius * math.log( ratio )
    return x[ :, column : column + 1 ].clamp( -lim, lim )


def project( x, model ):
    """the rows brought back inside ( `lbfgs( project = )` ): the same shapes. The radius and the density too, into
    their bounds: beyond, a row has no derivative towards them, and L-BFGS would push on a wall"""
    cols = [ discs( x, model )[ 0 ].detach(), bounded( x, model, 2, SCALE, model.max_ratio ) ]
    if x.shape[ 1 ] > 3:
        cols.append( bounded( x, model, 3, DENSITY, model.max_ratio ** 2 ) )
    return torch.cat( cols, 1 )


def shapes( x, model ):
    """`( centres [ n, 2 ], C [ n, 3 ], masses [ n ] )`: what the kernels take"""
    c, r = discs( x, model )
    r2 = r * r
    m = r2 * weights( x, model )
    return c, torch.stack( [ r2, r2, torch.zeros_like( r2 ) ], 1 ), m / m.sum()


def _window( model ):
    return ( model.radius / model.max_ratio, model.radius * model.max_ratio, model.k )


def deposit( centres, C, masses, model ):
    return fused.deposit_ellipses( centres, C, masses, *model.detector(), *_window( model ) )


def gather( centres, C, masses, model, d_cells ):
    return fused.gather_ellipses( centres, C, masses, *model.detector(), *_window( model ), d_cells )


def density( centres, radii, nb, extent, k, sub = 4, chunk = 2048, densities = None ):
    """`[ nb, nb ]`: the mass of the balls ( tensors; masses as `radius^2`, or `densities radius^2`, 1 in all ) in each
    pixel of the centred square of side `extent`, the density sampled on `sub^2` points per pixel -- what the balls say the
    object is"""
    with torch.no_grad():
        n, hp = nb * sub, extent / ( nb * sub )
        reach = int( math.ceil( float( radii.max() ) / hp ) ) + 1
        off = torch.arange( -reach, reach + 1, device = centres.device )
        w = torch.ones_like( radii ) if densities is None else densities
        amp = ( k + 0.5 ) / math.pi * hp * hp * w / ( w * radii * radii ).sum()           # ( mass / radius^2: the density )
        out = torch.zeros( n * n, dtype = centres.dtype, device = centres.device )
        for i in range( 0, len( centres ), chunk ):
            t = ( centres[ i:i + chunk ] + extent / 2 ) / hp - 0.5                       # in sample index units
            i0 = torch.round( t ).long()
            ia, ib = i0[ :, 0, None, None ] + off[ :, None ], i0[ :, 1, None, None ] + off[ None, : ]
            v = 1 - ( ( ( ia - t[ :, 0, None, None ] ) ** 2 + ( ib - t[ :, 1, None, None ] ) ** 2 ) * ( hp / radii[ i:i + chunk, None, None ] ) ** 2 )
            val = torch.where( v > 0, v.clamp( min = 0 ) ** ( k - 0.5 ), 0.0 ) * amp[ i:i + chunk, None, None ]
            ok = ( ia >= 0 ) & ( ia < n ) & ( ib >= 0 ) & ( ib < n )
            out.scatter_add_( 0, ( ia.clamp( 0, n - 1 ) * n + ib.clamp( 0, n - 1 ) ).expand_as( val ).reshape( -1 ), torch.where( ok, val, 0.0 ).reshape( -1 ) )
        return out.reshape( nb, sub, nb, sub ).sum( ( 1, 3 ) ).cpu().numpy()


def outlines( centres, radii, nb = 20 ):
    """`( positions [ n nb, 2 ], faces [ n, nb ] )`: discs ( arrays ) as polygons, for a page"""
    a = 2 * np.pi * np.arange( nb ) / nb
    pos = centres[ :, None, : ] + radii[ :, None, None ] * np.stack( [ np.cos( a ), np.sin( a ) ], 1 )
    return pos.reshape( -1, 2 ), np.arange( len( centres ) * nb ).reshape( -1, nb )
