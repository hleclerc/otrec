"""STAR POLYGONS, each of ONE density: a shape has a centre and `V` vertices around it, in the order of their angles: always
a polygon that turns once around its centre, whatever the parameters. Its projection is piecewise linear: nothing but
what is drawn. It has a MASS of its own ( 1 in all ), set at the start for all the polygons to have the same density.
The masses are free on purpose: with one density for all ( masses as the areas ), a polygon that grows takes its mass
from every other one, and a stage needs twice the iterations for a worse image -- and free, the densities stay within
1 % of each other.

A cloud of `n` polygons of the plane is ONE array `x [ n, 5 + 2 V ]`, made to be optimized as it stands ( no constraint,
every column in the unit of a length ):

    x[ :, 0:2 ]       the centre
    x[ :, 2 ]         SCALE r0 log( size / r0 ), as the radius of a sized ball ( `sized` )
    x[ :, 3 ]         r0 x the angle of the first vertex
    x[ :, 4:4+V ]     r0 d: the vertex `i` is at `size exp( d_i - mean( d ) )` from the centre
    x[ :, 4+V:4+2V ]  r0 g: the angle from a vertex to the next is `2 pi softmax( g )` ( 0: evenly spread )
    x[ :, 4+2V ]      SCALE r0 log( mass ), up to a constant

The distances are kept in `[ r0 / max_ratio, r0 max_ratio ]` ( the cells a shape may reach are a static size ) and the
polygons inside the observable domain ( see `balls.inside` ): `shapes( x, model )` is what the kernels take.

They start as the LAGUERRE CELLS of discs ( `from_discs` ): where two discs overlap, the side between their polygons is
their common chord, and where a disc touches nothing its border is an arc, cut in sides -- each cell then scaled around
its centre to the area its mass asks for."""
import math

import numpy as np
import torch

from . import fused
from .balls import inside

SCALE = 10.0                                                    # ( as in `sized` )


def cells( centres, radii, nb_vertices, nb_samples = 128, nb_neighbours = 16, chunk = 2048 ):
    """`( angles [ n, V ], distances [ n, V ] )`, on arrays: `V` vertices, seen from its centre, on the border of the cell of
    each disc -- its LAGUERRE cell ( what is nearer to it than to the discs it overlaps, for the power distance: the side
    between two cells is the common chord of their discs ) cut by the disc itself. The border is sampled along
    `nb_samples` directions, and the vertices are taken where it TURNS: one every `2 pi / V` of turning, so that a corner
    takes a vertex and an arc is cut in equal sides. With more corners than vertices, the sharpest ones are served."""
    from scipy.spatial import cKDTree
    n, V, M = len( centres ), nb_vertices, nb_samples
    dist, nbr = cKDTree( centres ).query( centres, min( nb_neighbours, n - 1 ) + 1 )
    dist, nbr = dist[ :, 1: ].reshape( n, -1 ), nbr[ :, 1: ].reshape( n, -1 )
    theta = 2 * np.pi * np.arange( M ) / M
    angles, rho = np.empty( ( n, V ) ), np.empty( ( n, V ) )
    for i in range( 0, n, chunk ):
        s = slice( i, i + chunk )
        r, D, rn = radii[ s, None ], np.maximum( dist[ s ], 1e-300 ), radii[ nbr[ s ] ]
        u = centres[ nbr[ s ] ] - centres[ s, None, : ]
        cut = np.maximum( ( D * D + r * r - rn * rn ) / ( 2 * D ), 0.05 * r )            # from the centre to the chord
        c = np.cos( theta[ None, None, : ] - np.arctan2( u[ :, :, 1 ], u[ :, :, 0 ] )[ :, :, None ] )
        with np.errstate( divide = "ignore", invalid = "ignore" ):
            reach = np.where( ( ( D < r + rn ) & ( D > 0 ) )[ :, :, None ] & ( c > 1e-9 ), cut[ :, :, None ] / c, np.inf )
        border = np.minimum( reach.min( 1 ), r )                                         # [ n, M ]: the distance along each direction
        p = border[ :, :, None ] * np.stack( [ np.cos( theta ), np.sin( theta ) ], 1 )
        a, b = p - np.roll( p, 1, 1 ), np.roll( p, -1, 1 ) - p
        turn = np.cumsum( np.abs( np.arctan2( a[ :, :, 0 ] * b[ :, :, 1 ] - a[ :, :, 1 ] * b[ :, :, 0 ], ( a * b ).sum( 2 ) ) ), 1 )
        level = ( ( np.arange( V ) + 0.5 ) / V - 1e-9 ) * turn[ :, -1: ]                # ( - 1e-9: a level reached to the rounding is reached )
        k = ( turn[ :, :, None ] < level[ :, None, : ] ).sum( 1 )                        # the first sample where the turning reaches a level
        k = np.minimum( np.maximum.accumulate( k - np.arange( V ), 1 ) + np.arange( V ), M - V + np.arange( V ) )   # ( all distinct )
        angles[ s ], rho[ s ] = theta[ k ], np.take_along_axis( border, k, 1 )
    return angles, rho


def pack( centres, angles, distances, masses, r0 ):
    """the rows of polygons given by their vertices seen from their centres ( tensors; the angles increasing, over less
    than a turn ) and their masses ( up to a factor )"""
    V, lr = angles.shape[ 1 ], torch.log( distances / r0 )
    gaps = torch.cat( [ angles[ :, 1: ] - angles[ :, :-1 ], 2 * math.pi + angles[ :, :1 ] - angles[ :, -1: ] ], 1 )
    return torch.cat( [ centres, SCALE * r0 * lr.mean( 1, keepdim = True ), r0 * angles[ :, :1 ], r0 * ( lr - lr.mean( 1, keepdim = True ) ),
                        r0 * torch.log( gaps * ( V / ( 2 * math.pi ) ) ), SCALE * r0 * torch.log( masses / masses.mean() )[ :, None ] ], 1 )


def from_discs( centres, radii, r0, nb_vertices, areas = None ):
    """the Laguerre cells of the discs ( tensors ), cut by them: see `cells`. `areas [ n ]`: each cell is scaled around
    its centre to that area. The masses are the areas: the same density for all"""
    T = lambda v: torch.as_tensor( v, dtype = centres.dtype, device = centres.device )
    a, d = cells( centres.cpu().numpy(), radii.cpu().numpy(), nb_vertices )
    if areas is not None:
        gaps = np.diff( np.concatenate( [ a, a[ :, :1 ] + 2 * np.pi ], 1 ), axis = 1 )
        d = d * np.sqrt( areas.cpu().numpy() / ( d * np.roll( d, -1, 1 ) * np.sin( gaps ) ).sum( 1 ) * 2 )[ :, None ]
    a, d = T( a ), T( d )
    gaps = torch.cat( [ a[ :, 1: ] - a[ :, :-1 ], 2 * math.pi + a[ :, :1 ] - a[ :, -1: ] ], 1 )
    return pack( centres, a, d, ( d * torch.roll( d, -1, 1 ) * torch.sin( gaps ) ).sum( 1 ) / 2, r0 )


def vertices( x, model ):
    """`( centres [ n, 2 ] inside the domain, vertices [ n, V, 2 ] )`"""
    r0, V = model.radius, ( x.shape[ 1 ] - 5 ) // 2
    d = ( x[ :, 4:4 + V ] - x[ :, 4:4 + V ].mean( 1, keepdim = True ) ) / r0
    rho = ( r0 * torch.exp( x[ :, 2:3 ] / ( SCALE * r0 ) + d ) ).clamp( r0 / model.max_ratio, r0 * model.max_ratio )
    gaps = 2 * math.pi * torch.softmax( x[ :, 4 + V:4 + 2 * V ] / r0, 1 )
    a = x[ :, 3:4 ] / r0 + torch.cumsum( gaps, 1 ) - gaps
    centres = inside( x[ :, :2 ], rho.max( 1 ).values, model.limit )
    return centres, centres[ :, None, : ] + rho[ :, :, None ] * torch.stack( [ torch.cos( a ), torch.sin( a ) ], 2 )


def areas( verts ):
    nxt = torch.roll( verts, -1, 1 )
    return ( verts[ :, :, 0 ] * nxt[ :, :, 1 ] - verts[ :, :, 1 ] * nxt[ :, :, 0 ] ).sum( 1 ) / 2


def shapes( x, model ):
    """`( vertices [ n, V, 2 ], densities [ n ] )`: a mass of 1 in all"""
    verts = vertices( x, model )[ 1 ]
    return verts, torch.softmax( x[ :, -1 ] / ( SCALE * model.radius ), 0 ) / areas( verts )


def discs( x, model ):
    """`( centres, radii )`: the balls of the model ( its `k` ) of the same centres and masses, at the mean density of the
    polygons at their centres"""
    centres, verts = vertices( x, model )
    A = areas( verts )
    return centres, torch.sqrt( torch.softmax( x[ :, -1 ] / ( SCALE * model.radius ), 0 ) * A.sum() / model.unit )


def _grid( model ):
    return ( *model.detector(), model.radius * model.max_ratio )


def deposit( verts, weights, model ):
    return fused.deposit_polygons( verts, weights, *_grid( model ) )


def gather( verts, weights, model, d_cells ):
    return fused.gather_polygons( verts, weights, *_grid( model ), d_cells )


def density( centres, verts, weights, nb, extent, sub = 4, chunk = 2048 ):
    """`[ nb, nb ]`: the mass of the polygons ( `vertices( x, model )` ), of densities `weights [ n ]` ( see `shapes` ), in
    each pixel of the centred square of side `extent` ( `sub^2` samples per pixel: in a polygon or not ) -- what the
    polygons say the object is"""
    with torch.no_grad():
        nv, n, hp = verts.shape[ 1 ], nb * sub, extent / ( nb * sub )
        w = hp * hp * weights
        reach = int( math.ceil( torch.linalg.vector_norm( verts - centres[ :, None, : ], dim = 2 ).max().item() / hp ) ) + 1
        off = torch.arange( -reach, reach + 1, device = verts.device )
        out = torch.zeros( n * n, dtype = verts.dtype, device = verts.device )
        for i in range( 0, len( verts ), chunk ):
            v = verts[ i:i + chunk ]
            i0 = torch.round( ( centres[ i:i + chunk ] + extent / 2 ) / hp - 0.5 ).long()
            ia, ib = i0[ :, 0, None, None ] + off[ :, None ], i0[ :, 1, None, None ] + off[ None, : ]
            px = ( ( ia.to( v.dtype ) + 0.5 ) * hp - extent / 2 ).expand( -1, -1, len( off ) )
            py = ( ( ib.to( v.dtype ) + 0.5 ) * hp - extent / 2 ).expand( -1, len( off ), -1 )
            odd = torch.zeros_like( px, dtype = torch.bool )                             # a ray towards x > 0 crosses the border an odd number of times
            for k in range( nv ):
                a, b = v[ :, k, :, None, None ], v[ :, ( k + 1 ) % nv, :, None, None ]
                odd ^= ( ( a[ :, 1 ] > py ) != ( b[ :, 1 ] > py ) ) & ( ( px - a[ :, 0 ] ) * ( b[ :, 1 ] - a[ :, 1 ] ).abs() < ( b[ :, 0 ] - a[ :, 0 ] ) * ( py - a[ :, 1 ] ) * torch.sign( b[ :, 1 ] - a[ :, 1 ] ) )
            ok = odd & ( ia >= 0 ) & ( ia < n ) & ( ib >= 0 ) & ( ib < n )
            out.scatter_add_( 0, ( ia.clamp( 0, n - 1 ) * n + ib.clamp( 0, n - 1 ) ).reshape( -1 ), ( ok.to( v.dtype ) * w[ i:i + chunk, None, None ] ).reshape( -1 ) )
        return out.reshape( nb, sub, nb, sub ).sum( ( 1, 3 ) ).cpu().numpy()


def outlines( centres, verts ):
    """`( positions [ n ( V + 1 ), 2 ], triangles [ n V, 3 ], edges [ n V, 2 ] )`: the polygons for a page, each as its
    triangles around its centre ( a fan from a vertex would overflow a polygon that is not convex; right as long as the
    polygon sees all its border from its centre ), and their borders"""
    pos = torch.cat( [ centres[ :, None, : ], verts ], 1 ).detach().cpu().numpy()
    n, V = verts.shape[ :2 ]
    o = ( V + 1 ) * np.arange( n )[ :, None ]
    i, j = o + 1 + np.arange( V ), o + 1 + ( np.arange( V ) + 1 ) % V
    return pos.reshape( -1, 2 ), np.stack( [ o + 0 * i, i, j ], 2 ).reshape( -1, 3 ), np.stack( [ i, j ], 2 ).reshape( -1, 2 )


def turns( centres, verts ):
    """`( gaps [ n, V ], reflex [ n, V ] )`: the angle from each vertex to the next, seen from the centre ( all under pi:
    the polygon cannot cross itself ), and whether the polygon turns the wrong way at each vertex ( not convex there )"""
    with torch.no_grad():
        r = verts - centres[ :, None, : ]
        n = torch.roll( r, -1, 1 )
        gaps = torch.atan2( r[ :, :, 0 ] * n[ :, :, 1 ] - r[ :, :, 1 ] * n[ :, :, 0 ], ( r * n ).sum( 2 ) ) % ( 2 * math.pi )
        a, b = verts - torch.roll( verts, 1, 1 ), torch.roll( verts, -1, 1 ) - verts
        cross = a[ :, :, 0 ] * b[ :, :, 1 ] - a[ :, :, 1 ] * b[ :, :, 0 ]
        return gaps, cross < -1e-9 * torch.linalg.vector_norm( a, dim = 2 ) * torch.linalg.vector_norm( b, dim = 2 )
