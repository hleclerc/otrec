"""The object the experiments and the tests reconstruct: a solid disc ( 2D ) or cylinder ( 3D ) full of round holes.

    ball_packing    the image of it
    gravity         the holes: balls of various sizes settled by gravity
    image           the solid fraction of each pixel"""
import numpy as np

from .gravity import settle
from .image import rasterize


def ball_packing( nb, dim = 2, extent = 2.0, radius = 0.8, height = 0.4, ball_radii = ( 0.04, 0.08 ), gap = 0.1, seed = 0 ):
    """`[ nb ] * dim`, float32, in [ 0, 1 ]: the image, on the centred square ( cube ) of side `extent`, of a disc of radius
    `radius` ( `dim = 3`: a cylinder of axis z and height `height` ) minus a PACKING OF BALLS -- slippery balls of radii
    drawn in `ball_radii`, settled by gravity in a box a little larger than the object ( so that balls stick out of it ),
    then shrunk by `gap` of their radius: the gutters between them are the material."""
    rng = np.random.default_rng( seed )
    half_width = radius + ball_radii[ 1 ]
    floor, top = ( -half_width, half_width ) if dim == 2 else ( -height / 2 - ball_radii[ 1 ], height / 2 + ball_radii[ 1 ] )
    # enough balls to fill the box well above the object: a dense packing holds 0.85 ( 2D ) or 0.64 ( 3D ) of the room
    room = ( 2 * half_width ) ** ( dim - 1 ) * ( top - floor + 2 * ball_radii[ 1 ] )
    unit = np.pi if dim == 2 else 4 * np.pi / 3
    radii = [ ]
    while unit * sum( r ** dim for r in radii ) < ( 0.9 if dim == 2 else 0.7 ) * room:
        radii.append( rng.uniform( *ball_radii ) )
    radii = np.array( radii )
    centers = settle( radii, dim, half_width, floor, seed )
    return rasterize( centers, ( 1 - gap ) * radii, dim, nb, extent, radius, height )
