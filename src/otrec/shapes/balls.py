"""BALLS of ONE radius ( `model.radius` ) and equal masses: a cloud is its centres `x [ n, dim ]`. What the kernels take
is `shapes( x, model )`: the centres, kept INSIDE the observable domain."""
import torch

from . import fused


def inside( centres, reach, limit ):
    """the centres, pulled back towards the origin where a shape that reaches `reach` ( a number, or `[ n ]` ) around its
    centre would leave the centred ball of radius `limit`: what every view sees"""
    room = ( limit - reach ) / torch.linalg.vector_norm( centres, dim = 1 ).clamp( min = 1e-300 )
    return centres * room.clamp( 0, 1 )[ :, None ]


def shapes( x, model ):
    """`( centres [ n, dim ], )`"""
    return ( inside( x, model.radius, model.limit ), )


def deposit( centres, model ):
    return fused.deposit( centres, torch.full_like( centres[ :, 0 ], model.radius ), *model.detector(), model.radius, model.k )


def gather( centres, model, d_cells ):
    return fused.gather( centres, torch.full_like( centres[ :, 0 ], model.radius ), *model.detector(), model.radius, model.k, d_cells )[ :1 ]


def discs( x, model ):
    """`( centres, radii )`: the cloud these rows are"""
    c = shapes( x, model )[ 0 ]
    return c, torch.full_like( c[ :, 0 ], model.radius )
