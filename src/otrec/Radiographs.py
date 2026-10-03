"""Generator of synthetic RADIOGRAPHS: the 3D counterpart of `Sinogram`.

A `Radiographs` represents the MEASURED data of a 3D reconstruction: for each angle, the
2D projection ( integral along the rays, parallel beam ) of a 3D object, sampled
on a discretized planar detector. Where the 2D sinogram stacks 1D PROFILES, here we stack 2D
IMAGES: `values[ num_angle, num_u, num_v ]`.

Usage: we start from zero and accumulate primitives whose projection is known analytically
( `add_sphere` ). `image( k )` / `batched_image()` present the radiographs as 2D
`Image`s, consumable as the target distribution of a `SdotPlanNd` ( the 2D semi-discrete transport ).

Geometric conventions -- a rotation around the `z` axis, like a tomograph:
- angles θ_k = k·π/nb_angles, regularly spread over [0, π);
- projection direction ( the ray ) d_θ = ( cos θ, sin θ, 0 );
- detector axes u_θ = ( −sin θ, cos θ, 0 ) and v = ( 0, 0, 1 ); the detector coordinates
  of a point p are ( u, v ) = ( p·u_θ, p·v ). At θ = 0, `u = y` and `v = z` -- and `Sinogram`, at
  θ = 0, measures `s = x`: the two conventions differ by a quarter turn, which has no
  consequence ( an angle is an angle ) but deserves to be stated;
- the detector covers [ center_u − extent_u/2, center_u + extent_u/2 ] × [ same in v ], divided
  into `nb_u × nb_v` pixels of sides `du`, `dv`.
"""
import numpy as np

from loom import ShapeVar, Axis, Aggregate, RealTensor
from sdot import Image


class Radiographs( Aggregate ):
    nb_angles : ShapeVar
    nb_u      : ShapeVar
    nb_v      : ShapeVar

    num_angle : Axis[ "nb_angles" ]
    num_u     : Axis[ "nb_u" ]
    num_v     : Axis[ "nb_v" ]

    values    : RealTensor[ "num_angle", "num_u", "num_v" ]

    #: the dimension of the space where the objects live ( what `Reconstruction` reads to draw
    #: points of the right size ) -- `Sinogram` is 2
    world_dim = 3

    def __init__( self, nb_angles: int, nb_u: int, nb_v: int, extent_u: float, extent_v: float | None = None,
                  detector_center = ( 0.0, 0.0 ), quadrature: int = 4 ) -> None:
        """`nb_u × nb_v` pixels over `extent_u × extent_v` ( `extent_v = extent_u` by default ).

        `quadrature`: the number of Gauss-Legendre points PER AXIS with which `add_sphere`
        integrates the projection over each pixel ( see its docstring ).
        """
        if nb_angles < 1 or nb_u < 1 or nb_v < 1:
            raise ValueError( "nb_angles, nb_u and nb_v must be >= 1" )
        extent_v = extent_u if extent_v is None else extent_v
        if extent_u <= 0 or extent_v <= 0:
            raise ValueError( "extent_u and extent_v must be > 0" )

        # detector / angular geometry: HOST data, constant throughout a
        # reconstruction ( see `Sinogram` for the same remark )
        self.extent_u, self.extent_v = float( extent_u ), float( extent_v )
        #: the side of the largest centered cube ALL of whose projections fit in the detector -- what
        #: `Reconstruction.random_points` draws in ( the diagonal of a face rotates in `u` )
        self.extent = min( self.extent_u / np.sqrt( 2 ), self.extent_v )
        self.detector_center = ( float( detector_center[ 0 ] ), float( detector_center[ 1 ] ) )
        self.nb_u_host, self.nb_v_host = int( nb_u ), int( nb_v )
        self.du = self.extent_u / nb_u
        self.dv = self.extent_v / nb_v
        self.u_min = self.detector_center[ 0 ] - self.extent_u / 2
        self.v_min = self.detector_center[ 1 ] - self.extent_v / 2
        self.quadrature = int( quadrature )

        angles = np.pi * np.arange( nb_angles ) / nb_angles                        # [ nb_angles ]
        self.angles = angles
        c, s = np.cos( angles ), np.sin( angles )
        z = np.zeros_like( angles )
        self.directions = np.stack( [ c, s, z ], axis = 1 )                      # [ nb_angles, 3 ]
        # the detector basis, `[ nb_angles, 2, 3 ]`: row 0 is `u_θ`, row 1 is `v`
        self.bases = np.stack( [ np.stack( [ -s, c, z ], axis = 1 ),
                                 np.stack( [ z, z, np.ones_like( angles ) ], axis = 1 ) ], axis = 1 )

        self.__base_init__(
            values = np.zeros( ( int( nb_angles ), int( nb_u ), int( nb_v ) ), dtype = float ),
        )

    # -- geometry ----------------------------------------------------------

    @property
    def u_edges( self ) -> np.ndarray:
        return self.u_min + self.du * np.arange( self.nb_u_host + 1 )

    @property
    def v_edges( self ) -> np.ndarray:
        return self.v_min + self.dv * np.arange( self.nb_v_host + 1 )

    @property
    def u_centers( self ) -> np.ndarray:
        return self.u_min + self.du * ( np.arange( self.nb_u_host ) + 0.5 )

    @property
    def v_centers( self ) -> np.ndarray:
        return self.v_min + self.dv * ( np.arange( self.nb_v_host ) + 0.5 )

    def project_points( self, points ) -> np.ndarray:
        """Detector coordinates `[ nb_angles, n, 2 ]` of the `points` ( `[ n, 3 ]` ), for each angle.

        HOST side, in numpy: the 3D reconstruction derives its cost through the envelope theorem
        ( `models.ProjectedDiracModel` ), not through autodiff, so the projection does not need to be
        traced. Its transpose is `unproject_grad`.
        """
        pts = np.asarray( points, dtype = float )
        if pts.ndim != 2 or pts.shape[ 1 ] != 3:
            raise ValueError( "points must have shape [ n, 3 ]" )
        return np.einsum( "kcd,nd->knc", self.bases, pts )

    def unproject_grad( self, grad_uv ) -> np.ndarray:
        """The adjoint of `project_points`: from cotangents `[ nb_angles, n, 2 ]` on the
        detector coordinates, the cotangent `[ n, 3 ]` on the points -- the sum over the angles
        of `g_k . base_k`."""
        return np.einsum( "knc,kcd->nd", np.asarray( grad_uv, dtype = float ), self.bases )

    def visual_hull_points( self, nb_points: int, seed: int = 0, extent: float | None = None,
                            threshold: float = 0.0 ) -> np.ndarray:
        """`nb_points` points drawn uniformly in the VISUAL HULL: the cube
        `[ -extent/2, extent/2 ]^3` ( `self.extent` by default ) reduced to the points ALL of whose
        projections fall on a pixel of value `> threshold` -- by rejection.

        The starting point a reconstruction wants: a dirac one of whose projections falls in the
        void has, at that angle, only a cell of almost zero measure, and the transport that must
        bring it to the shadow is as ill-conditioned as it is far ( see `SdotPlanNd`,
        the Newton of `SdotPlanNd` ). The visual hull contains the object, and it is already the object
        more or less when the angles are numerous enough.
        """
        rng = np.random.default_rng( seed )
        e = float( self.extent if extent is None else extent )
        vals = np.asarray( self.values )
        nb_angles = len( self.angles )
        pts, tried = [], 0
        while sum( len( p ) for p in pts ) < nb_points:
            batch = ( rng.random( ( max( 4 * nb_points, 1024 ), 3 ) ) - 0.5 ) * e
            uv = self.project_points( batch )
            iu = np.floor( ( uv[ ..., 0 ] - self.u_min ) / self.du ).astype( int )
            iv = np.floor( ( uv[ ..., 1 ] - self.v_min ) / self.dv ).astype( int )
            inside = ( iu >= 0 ) & ( iu < self.nb_u_host ) & ( iv >= 0 ) & ( iv < self.nb_v_host )
            v = np.where( inside, vals[ np.arange( nb_angles )[ :, None ], iu.clip( 0, self.nb_u_host - 1 ),
                                        iv.clip( 0, self.nb_v_host - 1 ) ], 0.0 )
            ok = np.all( v > threshold, axis = 0 )
            pts.append( batch[ ok ] )
            tried += len( batch )
            if tried > 200 * nb_points + 1e6:
                raise ValueError( "visual_hull_points: the visual hull is ( almost ) empty in this cube" )
        return np.concatenate( pts )[ :nb_points ]

    # -- accumulation ------------------------------------------------------

    def add_sphere( self, center, radius: float, density: float = 1.0 ) -> "Radiographs":
        """Adds the projection of a uniform ball.

        The integral along a ray passing at distance ρ from the axis of the ball is the chord
        `ρ_m · 2·√(r² − ρ²)` ( zero outside the projected disk ), where ρ² = (u − u0)² + (v − v0)² and
        ( u0, v0 ) the projected center. It is integrated over each pixel by a Gauss-Legendre
        quadrature ( `quadrature` points per axis -- exact everywhere except on the pixels that the
        edge of the disk crosses, where the chord is not polynomial ), and the stored value is the
        mean density over the pixel ( `mass = value · du · dv` ). The total mass per angle is
        therefore `4/3 π r³ ρ_m` up to quadrature errors, as long as the ball fits in the
        detector. This is not the exact integral of `Sinogram.add_disk`: in 2D the integral of the
        chord over a rectangle has no convenient closed form, and a transport-based reconstruction
        normalizes each angle to mass 1 anyway.

        Returns self to allow chaining.
        """
        center = np.asarray( center, dtype = float )
        if center.shape != ( 3, ):
            raise ValueError( "center must have shape [ 3 ]" )
        if radius <= 0:
            raise ValueError( "radius must be > 0" )

        r = float( radius )
        uv0 = self.project_points( center[ None, : ] )[ :, 0, : ]                  # [ nb_angles, 2 ]

        # the Gauss-Legendre nodes on [ 0, 1 ], and their weights ( sum 1 )
        x, w = np.polynomial.legendre.leggauss( self.quadrature )
        x, w = ( x + 1 ) / 2, w / 2
        us = self.u_edges[ :-1, None ] + self.du * x[ None, : ]                   # [ nb_u, q ]
        vs = self.v_edges[ :-1, None ] + self.dv * x[ None, : ]                   # [ nb_v, q ]

        contribution = np.zeros( ( len( self.angles ), self.nb_u_host, self.nb_v_host ) )
        for k in range( len( self.angles ) ):
            a2 = ( us - uv0[ k, 0 ] ) ** 2                                         # [ nb_u, q ]
            b2 = ( vs - uv0[ k, 1 ] ) ** 2                                         # [ nb_v, q ]
            chord = 2 * np.sqrt( np.maximum( r * r - a2[ :, :, None, None ] - b2[ None, None, :, : ], 0.0 ) )
            contribution[ k ] = np.einsum( "iajb,a,b->ij", chord, w, w )

        self.values = self.values + density * contribution
        return self

    def blurred( self, sigma: float ) -> "Radiographs":
        """The same radiographs BLURRED by a Gaussian of standard deviation `sigma` ( in world
        units ), each separately -- a new `Radiographs` of the same geometry.

        What the blur buys: no more ZEROS. A radiograph of balls is zero outside
        their shadows, and a dirac that projects there has neither a cell nor a gradient ( see `SdotPlanNd` );
        blurred at the domain scale ( `sigma ~ extent` ), it is a positive bump everywhere,
        and the transport is gentle. A reconstruction starts there and tightens the blur
        ( `Reconstruction.anneal_blur` ). The detector edge is extended by zero: what
        overflows is lost, which changes nothing for a target normalized per angle.
        """
        from scipy.ndimage import gaussian_filter
        out = Radiographs( nb_angles = len( self.angles ), nb_u = self.nb_u_host, nb_v = self.nb_v_host,
                           extent_u = self.extent_u, extent_v = self.extent_v,
                           detector_center = self.detector_center, quadrature = self.quadrature )
        vals = np.asarray( self.values )
        if sigma > 0:
            vals = np.stack( [ gaussian_filter( v, sigma = ( sigma / self.du, sigma / self.dv ), mode = "constant" )
                               for v in vals ] )
        out.values = vals
        return out

    # -- consumption -------------------------------------------------------

    def _image_kwargs( self ):
        return dict( origin = [ self.u_min, self.v_min ],
                     frame = [ [ self.du, 0.0 ], [ 0.0, self.dv ] ] )

    def image( self, k: int, background: float = 0.0 ) -> Image:
        """2D `Image` of the radiograph at angle k, in real detector coordinates.

        `background`: a density added EVERYWHERE ( as a fraction of the mean value of the angle )
        -- what a semi-discrete transport requires so that no Laguerre cell has zero
        measure ( a dirac projected outside the shadow of the object would otherwise have no gradient,
        see `SdotPlanNd` ). A radiograph of balls is zero outside their shadows.
        """
        vals = np.asarray( self.values )[ k ]
        if background:
            vals = vals + background * float( vals.mean() )
        return Image( values = vals, **self._image_kwargs() )

    def batched_image( self, background: float = 0.0 ) -> Image:
        """All the radiographs at once: an `Image` batched over `num_angle`."""
        vals = np.asarray( self.values )
        if background:
            vals = vals + background * vals.mean( axis = ( 1, 2 ), keepdims = True )
        return Image( values = vals, batch_axes = [ self.num_angle ], **self._image_kwargs() )

    def mass( self, k: int | None = None ):
        """Total mass ( ∫∫ radiograph du dv ) at angle k, or per angle if k is None."""
        m = np.asarray( self.values ).sum( axis = ( 1, 2 ) ) * self.du * self.dv
        return m if k is None else m[ k ]
