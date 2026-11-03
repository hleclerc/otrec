"""The ACQUISITION GEOMETRY of a 3D reconstruction, in jax: where a point of space lands on the detector of each view.

A `Geometry` gives, for the view `k`:

- `project( x, k )`: the detector coordinates `( u, v )` of the points `x` ( `[ n, 3 ] -> [ n, 2 ]` );
- `jacobian( x, k )`: `d( u, v ) / dx` ( `[ n, 2, 3 ]` ), what the gradient and the lifted step use
  ( `reconstruction3d` );
- the detector: `u_min`, `v_min`, `du`, `dv`, `nb_u`, `nb_v` ( a plane, regular pixels, axis-aligned in ( u, v ) ).

Nothing downstream assumes the projection to be linear: a cone beam is another `Geometry` with the same
three entries ( its jacobian depends on the point ). `k` may be a traced integer, so that one compiled
function serves all the views.
"""
import jax.numpy as jnp
import numpy as np


class Geometry:
    """The interface ( see the module docstring ); `nb_views` and the detector fields are host numbers."""

    nb_views: int
    u_min: float
    v_min: float
    du: float
    dv: float
    nb_u: int
    nb_v: int

    def project( self, x, k ):
        raise NotImplementedError

    def jacobian( self, x, k ):
        raise NotImplementedError

    @property
    def detector( self ):
        """`( u_min, v_min, du, dv, nb_u, nb_v )`"""
        return self.u_min, self.v_min, self.du, self.dv, self.nb_u, self.nb_v


class ParallelGeometry( Geometry ):
    """A parallel beam: view `k` sees `( u, v ) = B_k x + c`, `B_k = bases[ k ]` ( `[ 2, 3 ]`, orthonormal rows ) and
    `c` the detector offset ( 0 by default ) -- a rotation around an axis, a tilt series or any set of orientations."""

    def __init__( self, bases, u_min, v_min, du, dv, nb_u, nb_v, offsets = None ):
        bases = np.asarray( bases, dtype = float )
        if bases.ndim != 3 or bases.shape[ 1: ] != ( 2, 3 ):
            raise ValueError( "bases must have shape [ nb_views, 2, 3 ]" )
        self.nb_views = len( bases )
        self.bases = jnp.asarray( bases )
        self.offsets = jnp.zeros( ( self.nb_views, 2 ) ) if offsets is None else jnp.asarray( offsets, dtype = float )
        self.u_min, self.v_min, self.du, self.dv = float( u_min ), float( v_min ), float( du ), float( dv )
        self.nb_u, self.nb_v = int( nb_u ), int( nb_v )

    def project( self, x, k ):
        b = self.bases[ k ].astype( x.dtype )
        return x @ b.T + self.offsets[ k ].astype( x.dtype )

    def jacobian( self, x, k ):
        return jnp.broadcast_to( self.bases[ k ].astype( x.dtype ), ( x.shape[ 0 ], 2, 3 ) )
