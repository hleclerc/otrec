"""A MESH of the matter ( world 2D ): triangles of ONE density, drawn from a cloud of balls. The balls overlap ( `k = 2` ):
their density is smooth, about that of the matter inside it and 0 in the holes, and the CONTOURS of the matter are where
it crosses a share of the typical one ( `contours` ). `from_balls` puts nodes along the contours and inside, a length
`edge` apart, and keeps the Delaunay triangles that lie in the matter.

`Mesh( triangles )` is a kind of shapes for `Loss1d`, the rows of `x [ m, 2 ]` being the NODES: of one density, the mesh
projects as its border does, and only the nodes of the border change the loss ( a side must stay shorter than `max_ratio` radii
of the model )."""
import math

import numpy as np
import torch

from . import fused, polygons, sized


def field( centres, radii, k, step, extent ):
    """`[ nb, nb ]`: the density of the balls ( tensors; a mass of 1 in all ) at the centres of the cells, of side about
    `step`, of the centred square of side `extent`"""
    nb = int( math.ceil( extent / step ) )
    return sized.density( centres, radii, nb, extent, k, sub = 2 ) * ( nb / extent ) ** 2


def contours( f, extent, level = 0.5 ):
    """`( loops, density )`: the closed lines ( each `[ m, 2 ]`, the first point not repeated ) where the field `f` ( see
    `field` ) crosses `level` x the typical density of the matter, and that density: the one half of the mass lies above"""
    from contourpy import contour_generator
    v = np.sort( f.ravel() )
    density = v[ np.searchsorted( np.cumsum( v ), 0.5 * v.sum() ) ]
    x = extent * ( ( np.arange( f.shape[ 0 ] ) + 0.5 ) / f.shape[ 0 ] - 0.5 )
    lines = contour_generator( x, x, f.T ).lines( level * density )
    return [ l[ :-1 ] for l in lines if len( l ) > 3 and np.allclose( l[ 0 ], l[ -1 ] ) ], density


def resample( loop, edge ):
    """the closed line `loop [ m, 2 ]` as points about `edge` apart along it ( at least 3 )"""
    p = np.concatenate( [ loop, loop[ :1 ] ] )
    s = np.concatenate( [ [ 0.0 ], np.cumsum( np.linalg.norm( np.diff( p, axis = 0 ), axis = 1 ) ) ] )
    t = s[ -1 ] * np.arange( max( 3, int( round( s[ -1 ] / edge ) ) ) ) / max( 3, int( round( s[ -1 ] / edge ) ) )
    return np.stack( [ np.interp( t, s, p[ :, 0 ] ), np.interp( t, s, p[ :, 1 ] ) ], 1 )


def from_balls( centres, radii, k, extent, edge, level = 0.5 ):
    """`( nodes [ m, 2 ], triangles [ t, 3 ], loops )` ( arrays ): a mesh of where the density of the balls ( tensors ) is
    over `level` x the typical one. `loops`: the contours as lists of nodes ( the first nodes of the mesh, in order
    along each contour; a contour shorter than 3 `edge` is left out ). Then nodes on a hexagonal grid inside, not nearer
    than 0.8 `edge` to those of the contours ( a side of a contour is then a side of the Delaunay triangulation, but
    across a wall thinner than `edge` ), and the triangles whose centre is in the matter, counter-clockwise."""
    from scipy.ndimage import map_coordinates
    from scipy.spatial import Delaunay, cKDTree
    f = field( centres, radii, k, edge / 2, extent )
    lines, density = contours( f, extent, level )
    lines = [ resample( l, edge ) for l in lines if np.linalg.norm( np.diff( np.concatenate( [ l, l[ :1 ] ] ), axis = 0 ), axis = 1 ).sum() >= 3 * edge ]
    ends = np.cumsum( [ 0 ] + [ len( l ) for l in lines ] )
    on = np.concatenate( lines )
    inside = lambda p: map_coordinates( f, ( p.T / extent + 0.5 ) * f.shape[ 0 ] - 0.5, order = 1 ) > level * density
    n = int( math.ceil( extent / edge ) )
    i, j = np.meshgrid( np.arange( -n, n + 1 ), np.arange( -n, n + 1 ), indexing = "ij" )
    grid = edge * np.stack( [ i + 0.5 * ( j % 2 ), j * math.sqrt( 0.75 ) ], 2 ).reshape( -1, 2 )
    grid = grid[ np.abs( grid ).max( 1 ) < extent / 2 ]
    grid = grid[ inside( grid ) ]
    nodes = np.concatenate( [ on, grid[ cKDTree( on ).query( grid )[ 0 ] > 0.8 * edge ] ] )
    tri = Delaunay( nodes ).simplices
    tri = tri[ inside( nodes[ tri ].mean( 1 ) ) ]
    a, b = nodes[ tri[ :, 1 ] ] - nodes[ tri[ :, 0 ] ], nodes[ tri[ :, 2 ] ] - nodes[ tri[ :, 0 ] ]
    tri[ a[ :, 0 ] * b[ :, 1 ] < a[ :, 1 ] * b[ :, 0 ] ] = tri[ a[ :, 0 ] * b[ :, 1 ] < a[ :, 1 ] * b[ :, 0 ] ][ :, ::-1 ]
    return nodes, tri, [ np.arange( ends[ l ], ends[ l + 1 ] ) for l in range( len( lines ) ) ]


def border( triangles ):
    """`[ e, 2 ]`: the sides of one triangle only ( array ), each as in its triangle: the matter on its left"""
    e = np.concatenate( [ triangles[ :, [ 0, 1 ] ], triangles[ :, [ 1, 2 ] ], triangles[ :, [ 2, 0 ] ] ] )
    m = int( e.max() ) + 1
    return e[ ~np.isin( e[ :, 0 ] * m + e[ :, 1 ], e[ :, 1 ] * m + e[ :, 0 ] ) ]


class Mesh:
    """the triangles `[ t, 3 ]` ( a tensor of nodes, counter-clockwise ) as shapes: `x [ m, 2 ]` are the nodes. Of one
    density, the mesh projects as its BORDER does: the kernels take its sides alone ( `fused`, open lines )"""

    def __init__( self, triangles ):
        self.triangles = triangles
        self.sides = torch.as_tensor( border( triangles.cpu().numpy() ), device = triangles.device )

    def shapes( self, x, model ):
        """`( sides [ e, 2, 2 ], densities [ e ] )`: one density, a mass of 1 in all"""
        v = x[ self.sides ]
        area = ( v[ :, 0, 0 ] * v[ :, 1, 1 ] - v[ :, 0, 1 ] * v[ :, 1, 0 ] ).sum() / 2
        return v, torch.ones_like( v[ :, 0, 0 ] ) / area

    def deposit( self, verts, weights, model ):
        return fused.deposit_polygons( verts, weights, *model.detector(), model.radius * model.max_ratio / 2, open = True )

    def gather( self, verts, weights, model, d_cells ):
        return fused.gather_polygons( verts, weights, *model.detector(), model.radius * model.max_ratio / 2, d_cells, open = True )

    def density( self, x, nb, extent ):
        """`[ nb, nb ]`: the mass of the mesh in each pixel ( see `polygons.density` )"""
        v = x[ self.triangles ]
        return polygons.density( v.mean( 1 ), v, torch.ones_like( v[ :, 0, 0 ] ) / polygons.areas( v ).sum(), nb, extent )
