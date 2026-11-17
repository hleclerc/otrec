"""ELLIPSES: each ball of a cloud may stretch and turn. World 2D. An ellipse has the PROFILE of the balls
( `( 1 - |y|^2 )^( k - 1/2 )` with y = M^-1 ( x - centre ), k = 2 by default ): its projection is the same polynomial,
stretched in each view to the half-width `sqrt( n.C n )`. A cloud is ONE array `x [ n, 5 ]`, made to be optimized as it
stands ( no constraint, every column a length ):

    x[ :, :2 ]    the centre
    x[ :, 2 ]     SIZE r0 log( s / r0 ): s the radius of the disc of the same area ( 0 is the radius `r0` )
    x[ :, 3: ]    SHAPE r0 ( p, q ): M = s expm( S ), S = [ [ p, q ], [ q, -p ] ] -- a stretch of ratio exp( 2 |( p, q )| )
                  that keeps the area; ( 0, 0 ) is the disc. The covariance is C = M M^T = s^2 expm( 2 S )

The mass of an ellipse is its area `pi s^2` ( one density at the centre for all; 1 in all ). Size and shape are apart
because the loss is far stiffer along the first ( the mass of the shape moves with it ) than along the second ( an area
that only changes its form ): at the end of the ball stages, a change of the area costs ~80 times more than a move of the
centre of the same length, ~2000 times more than a stretch. With `( a, b, c )` mixing them, the scalar initial Hessian of
L-BFGS follows the area, and the centres hardly move in the first tens of iterations. `SIZE` brings the area to the
curvature of the centres. `SHAPE` is NOT the one that would do the same for the stretch ( ~0.25 ): the loss is flat, even
concave ( the shear at the disc ), along the shape, and a fast stretch runs to needles held by the bounds of the widths, at a
higher loss. 4 keeps the ellipses near round and ends lower than `( a, b, c )` ( measured on exp_2d::ellipses, small ).

The kernels are those of `fused`; s stays in `[ r0 / max_ratio, r0 max_ratio ]`, the half-widths too, and the ellipses
inside the observable domain."""
import torch

from . import sized
from .balls import inside

SIZE = sized.SCALE
SHAPE = 4.0


def from_discs( centres, radii, model ):
    """`x [ n, 5 ]`: the round ellipses of these discs ( tensors )"""
    r0 = model.radius
    z = torch.zeros_like( radii )
    return torch.cat( [ centres, torch.stack( [ SIZE * r0 * torch.log( radii / r0 ), z, z ], 1 ) ], 1 )


def _size_and_stretch( x, model ):
    """`( s [ n ], p [ n ], q [ n ], t^2 [ n ] )`"""
    r0 = model.radius
    s = ( r0 * torch.exp( x[ :, 2 ] / ( SIZE * r0 ) ) ).clamp( r0 / model.max_ratio, r0 * model.max_ratio )
    p, q = x[ :, 3 ] / ( SHAPE * r0 ), x[ :, 4 ] / ( SHAPE * r0 )
    return s, p, q, p * p + q * q


def _ch_sh( t2, f ):
    """`( cosh( f t ), sinh( f t ) / t )`, t = sqrt( t2 ), smooth through t = 0"""
    small = t2 < 1e-12
    t = torch.sqrt( torch.where( small, torch.ones_like( t2 ), t2 ) )
    ch = torch.where( small, 1 + f * f * t2 / 2, torch.cosh( f * t ) )
    sh = torch.where( small, f * ( 1 + f * f * t2 / 6 ), torch.sinh( f * t ) / t )
    return ch, sh


def shapes( x, model ):
    """`( centres [ n, 2 ], C [ n, 3 ] as ( c00, c11, c01 ), masses [ n ] )`: what the kernels take"""
    s, p, q, t2 = _size_and_stretch( x, model )
    ch, sh = _ch_sh( t2, 2.0 )
    s2 = s * s
    C = s2[ :, None ] * torch.stack( [ ch + sh * p, ch - sh * p, sh * q ], 1 )
    return inside( x[ :, :2 ], torch.sqrt( C[ :, 0 ] + C[ :, 1 ] ), model.limit ), C, s2 / s2.sum()


deposit, gather = sized.deposit, sized.gather              # ( the kernels of sized balls are those of ellipses )


def discs( x, model ):
    """`( centres, radii )`: the discs of the same centres and areas"""
    return shapes( x, model )[ 0 ], _size_and_stretch( x, model )[ 0 ]


def drawing( x, model ):
    """what a frame of a page keeps: `{ "matrices": [ n, 2, 2 ] }`, the M of each ellipse ( arrays )"""
    with torch.no_grad():
        s, p, q, t2 = _size_and_stretch( x, model )
        ch, sh = _ch_sh( t2, 1.0 )
        M = s[ :, None ] * torch.stack( [ ch + sh * p, sh * q, sh * q, ch - sh * p ], 1 )
        return dict( matrices = M.reshape( -1, 2, 2 ).cpu().numpy() )
