"""ELLIPSES: each shape is the image of the unit shape ( see `shapes` ) by a symmetric matrix `exp( S )`, with a mass of its
own. Its projection along a view of axis `n` is the same polynomial, of half-width `w = | exp( S ) n |`.

A cloud of `n` ellipses of the plane is ONE array `x [ n, 6 ]`, made to be optimized as it stands ( no constraint, and
every column in the unit of a length, so that a step of the same size means as much on each ):

    x[ :, 0:2 ]    the centre
    x[ :, 2:5 ]    r0 ( S - log( r0 ) I ), as ( S_00, S_11, S_01 ): 0 is the BALL of radius `r0`; a change of `r0` is a factor
                   e on a half-axis. `exp( S )` is positive whatever `S`: nothing to keep in bounds
    x[ :, 5 ]      10 r0 log( mass ), up to a constant: the masses are normalized to 1 in all. ( The factor: with 1, the
                   masses take the steps and the shapes hardly move -- 400 iterations then leave the image error of a
                   packing at 0.13, against 0.085. )

The half-widths are kept in `[ r0 / max_ratio, r0 max_ratio ]` ( the window of cells a shape may reach is a static size )."""
import math

import numpy as np
import torch


def from_points( points ):
    """the balls of equal masses centred at `points [ n, 2 ]`, as ellipses"""
    return torch.cat( [ points, torch.zeros( len( points ), 4, dtype = points.dtype, device = points.device ) ], 1 )


def unpack( x, r0 ):
    """`( centres [ n, 2 ], S [ n, 2, 2 ], masses [ n ] )`"""
    q, diag = x[ :, 2:5 ] / r0, math.log( r0 )
    S = torch.stack( [ torch.stack( [ diag + q[ :, 0 ], q[ :, 2 ] ], 1 ), torch.stack( [ q[ :, 2 ], diag + q[ :, 1 ] ], 1 ) ], 1 )
    return x[ :, :2 ], S, torch.softmax( x[ :, 5 ] / ( 10 * r0 ), 0 )


def covariances( S ):
    """`[ n, 3 ]`: `C = exp( 2 S )` as `( c00, c11, c01 )` -- what the kernels take ( `shapes.fused` ); the half-width of an
    ellipse along a view of axis `n` is `sqrt( n.C n )`"""
    C = torch.linalg.matrix_exp( 2 * S )
    return torch.stack( [ C[ :, 0, 0 ], C[ :, 1, 1 ], C[ :, 0, 1 ] ], 1 )


def density( x, r0, max_ratio, nb, extent, k, sub = 4, chunk = 2048 ):
    """`[ nb, nb ]`: the mass of the cloud in each pixel of the centred square of side `extent` ( the density sampled on
    `sub^2` points per pixel ) -- what the ellipses say the object is"""
    with torch.no_grad():
        c, S, m = unpack( x, r0 )
        n, hp = nb * sub, extent / ( nb * sub )
        reach = int( math.ceil( r0 * max_ratio / hp ) ) + 1
        off = torch.arange( -reach, reach + 1, device = x.device )
        inv, amp = torch.linalg.matrix_exp( -S ), m * ( k + 0.5 ) / math.pi * torch.exp( -S[ :, 0, 0 ] - S[ :, 1, 1 ] ) * hp * hp
        out = torch.zeros( n * n, dtype = x.dtype, device = x.device )
        for i in range( 0, len( x ), chunk ):
            t = ( c[ i:i + chunk ] + extent / 2 ) / hp - 0.5                             # in sample index units
            i0 = torch.round( t ).long()
            ia, ib = i0[ :, 0, None, None ] + off[ :, None ], i0[ :, 1, None, None ] + off[ None, : ]
            d = torch.stack( [ ( ia - t[ :, 0, None, None ] ).expand( -1, -1, len( off ) ), ( ib - t[ :, 1, None, None ] ).expand( -1, len( off ), -1 ) ], -1 ) * hp
            u = torch.einsum( "nab,nijb->nija", inv[ i:i + chunk ], d )
            val = torch.clamp( 1 - ( u * u ).sum( -1 ), min = 0 ) ** ( k - 0.5 ) * amp[ i:i + chunk, None, None ]
            ok = ( ia >= 0 ) & ( ia < n ) & ( ib >= 0 ) & ( ib < n )
            out.scatter_add_( 0, ( ia.clamp( 0, n - 1 ) * n + ib.clamp( 0, n - 1 ) ).reshape( -1 ), torch.where( ok, val, 0.0 ).reshape( -1 ) )
        return out.reshape( nb, sub, nb, sub ).sum( ( 1, 3 ) ).cpu().numpy()


def outlines( x, r0, nb = 20 ):
    """`( positions [ n nb, 2 ], faces [ n, nb ] )`: the ellipses as polygons, for a page"""
    c, S, _ = unpack( torch.as_tensor( x ), r0 )
    a = 2 * math.pi * torch.arange( nb, dtype = c.dtype ) / nb
    pos = c[ :, None, : ] + torch.einsum( "nab,pb->npa", torch.linalg.matrix_exp( S ), torch.stack( [ torch.cos( a ), torch.sin( a ) ], 1 ) )
    return pos.reshape( -1, 2 ).numpy(), np.arange( len( c ) * nb ).reshape( -1, nb )
