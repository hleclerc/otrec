"""Generator of synthetic 2D sinograms.

A `Sinogram` represents the MEASURED data of a reconstruction problem: the
1D projections (Radon transform) of a 2D object, sampled on a
discretized detector, for a set of angles.

Usage: we start from zero and accumulate primitives whose projection is
known analytically (`add_disk` for 2D). The values live in the tensor field
`values[ num_angle, num_bin ]`.

Geometric conventions:
- angles θ_k = k·π/nb_angles, regularly spread over [0, π);
- projection normal n_θ = (cos θ, sin θ); the detector coordinate of a point
  p is s = p·n_θ (projection along the direction perpendicular to n_θ);
- the detector covers [ center − extent/2, center + extent/2 ], divided into
  `nb_bins` cells of width dw = extent/nb_bins.

`values[ k ]` is the piecewise-constant function of the profile at angle k;
`image( k )` presents it as a 1D `Image` (in real detector coordinates),
directly consumable as the target distribution of an `SdotPlan1d`.
"""
import numpy as np

from loom import ShapeVar, Axis, Tensor, Aggregate, RealTensor
from sdot import Image


class Sinogram( Aggregate ):
    nb_angles : ShapeVar
    nb_bins   : ShapeVar

    num_angle : Axis[ "nb_angles" ]
    num_bin   : Axis[ "nb_bins" ]

    values    : RealTensor[ "num_angle", "num_bin" ]

    def __init__( self, nb_angles: int, nb_bins: int, extent: float, detector_center: float = 0.0 ) -> None:
        if nb_angles < 1 or nb_bins < 1:
            raise ValueError( "nb_angles and nb_bins must be >= 1" )
        if extent <= 0:
            raise ValueError( "extent must be > 0" )

        # detector / angular geometry (host attributes, not tensors: data generation
        # is not differentiated and is computed most simply in numpy)
        self.extent = float( extent )
        self.detector_center = float( detector_center )
        # the bin count, HOST side. Apparent duplicate of the `nb_bins` ShapeVar, but it is
        # available HERE, before `values` is set: `nb_bins.value` is resolved from the observed
        # sizes, so not before `__base_init__` below -- whereas `dw`/`s_min` need
        # it right away. The detector geometry is HOST data (like
        # `angles`/`normals`) and must remain so: it is constant throughout an optimization.
        # (`nb_bins.value` is now a `ShapeArray`, hence a host count too -- it is no
        # longer the traceable `Tensor` that made this duplicate mandatory.)
        self.nb_bins_host = int( nb_bins )
        self.dw = self.extent / nb_bins
        self.s_min = self.detector_center - self.extent / 2

        angles = np.pi * np.arange( nb_angles ) / nb_angles                        # [ nb_angles ]
        self.angles = angles
        self.normals = np.stack( [ np.cos( angles ), np.sin( angles ) ], axis = 1 )  # [ nb_angles, 2 ]
        # same geometry, Tensor side: serves the differentiable projection (`project_points`). The
        # normals carry a SHARED coordinate axis `_xy` (dim 2), over which the projection contracts
        # BY REFERENCE (no `@` that would assume an axis order). The angle axis is their own.
        self._xy = Axis( ShapeVar( 2 ) )
        self.normals_t = RealTensor[ Axis( ShapeVar( int( nb_angles ) ) ), self._xy ]( self.normals )

        # the values start at 0; `add_disk` accumulates them
        self.__base_init__(
            values = np.zeros( ( int( nb_angles ), int( nb_bins ) ), dtype = float ),
        )

    # -- geometry ----------------------------------------------------------

    @property
    def bin_edges( self ) -> np.ndarray:
        """Edges of the detector bins, [ nb_bins + 1 ]."""
        return self.s_min + self.dw * np.arange( self.nb_bins_host + 1 )

    @property
    def bin_centers( self ) -> np.ndarray:
        """Centers of the detector bins, [ nb_bins ]."""
        return self.s_min + self.dw * ( np.arange( self.nb_bins_host ) + 0.5 )

    def project_points( self, points ) -> Tensor:
        """Detector coordinates of the `points` (shape [ n, 2 ]) for each angle.

        Returns a `Tensor` [ nb_angles, n ]: s[ k, i ] = points[ i ] · n_θk. The projection is a
        CONTRACTION BY REFERENCE over the shared coordinate axis `_xy` (`normals.dot( points, over=_xy )`)
        -- no `@`, hence no assumption about axis order. Everything goes through `Tensor` algebra, so
        it is DIFFERENTIABLE and trace-compatible (which the reconstruction gradient requires).
        """
        val = points if isinstance( points, Tensor ) else RealTensor( points )
        if val.rank != 2 or val.shape[ 1 ] != 2:
            raise ValueError( "points must have shape [ n, 2 ]" )
        # the points share the normals' `_xy` axis; their "point" axis is their own
        pts = RealTensor[ Axis( ShapeVar() ), self._xy ]( val )
        return self.normals_t.dot( pts, over = self._xy )                          # [ nb_angles, n ]

    # -- accumulation ------------------------------------------------------

    def add_disk( self, center, radius: float, density: float = 1.0 ) -> "Sinogram":
        """Adds the projection of a uniform (2D) disk.

        The Radon profile of a disk of radius r and density ρ is the chord
        `ρ·2·√(r² − t²)` where t = s − s0 is the offset from the projected center
        s0 = center·n_θ. We INTEGRATE this profile over each bin to preserve
        mass: the stored value is the mean density over the bin
        (mass = value·dw), so that the total mass per angle is
        exactly ρ·π·r² as long as the disk fits in the detector.

        Returns self to allow chaining.
        """
        center = np.asarray( center, dtype = float )
        if center.shape != ( 2, ):
            raise ValueError( "center must have shape [ 2 ]" )
        if radius <= 0:
            raise ValueError( "radius must be > 0" )

        r = float( radius )
        s0 = self.normals @ center                                                 # [ nb_angles ]

        # offset of the bin edges from the projected center, clipped to [ −r, r ]
        edges = self.bin_edges[ None, : ] - s0[ :, None ]                          # [ nb_angles, nb_bins + 1 ]
        t = np.clip( edges, -r, r )

        # primitive of 2·√(r² − t²): G(t) = t·√(r² − t²) + r²·arcsin( t / r )
        G = t * np.sqrt( np.maximum( r * r - t * t, 0.0 ) ) + r * r * np.arcsin( t / r )
        contribution = density * ( G[ :, 1: ] - G[ :, :-1 ] ) / self.dw            # [ nb_angles, nb_bins ]

        self.values = self.values + contribution
        return self

    def blurred( self, sigma: float ) -> "Sinogram":
        """The same sinogram BLURRED by a Gaussian of standard deviation `sigma` ( world units ),
        profile by profile -- see `Radiographs.blurred` for what the blur buys."""
        from scipy.ndimage import gaussian_filter1d
        out = Sinogram( nb_angles = int( self.nb_angles.value ), nb_bins = self.nb_bins_host,
                        extent = self.extent, detector_center = self.detector_center )
        vals = np.asarray( self.values )
        if sigma > 0:
            vals = gaussian_filter1d( vals, sigma = sigma / self.dw, axis = 1, mode = "constant" )
        out.values = vals
        return out

    # -- consumption -------------------------------------------------------

    def image( self, k: int ) -> Image:
        """1D `Image` of the profile at angle k, in real detector coordinates.

        Directly usable as the target distribution of an `SdotPlan1d`.
        """
        return Image(
            values = self.values[ k ],
            origin = [ self.s_min ],
            frame = [ [ self.dw ] ],
        )

    def batched_image( self ) -> Image:
        """All the profiles at once: an `Image` BATCHED over `num_angle` (the batch axis already exists
        here -- it is that of `values`). `origin`/`frame`, identical at all angles, are
        SHARED (not batched). Serves as the target distribution of a batched `SdotPlan1d`, with no Python loop.
        """
        return Image(
            values = self.values,                       # [ num_angle, num_bin ]
            origin = [ self.s_min ],                    # shared (common detector geometry)
            frame = [ [ self.dw ] ],                    # shared
            batch_axes = [ self.num_angle ],
        )

    def mass( self, k: int | None = None ):
        """Total mass (∫ profile ds) at angle k, or per angle (rank 1) if k is None."""
        m = self.values.sum( axis = "num_bin" ) * self.dw
        return m if k is None else m[ k ]

    def debias_and_equalize_mass( self ) -> "Sinogram":
        """Corrects a sinogram whose object EXCEEDS the detector (radius > extent/2): the
        visible window then no longer descends to 0 (we only see a central piece of the
        object), and the measured mass VARIES with the angle (the width of the truncated shadow depends on the
        orientation). `SdotPlan1d` already restores src/dst mass equality via
        `normalized_version` (a multiplicative RESCALE per angle) -- correct for a true
        density distribution, but here it amounts to artificially inflating the visible part
        of a heavily truncated angle, distorting the reconstructed shape. An ADDITIVE offset is more
        faithful: we model the invisible part as a locally homogeneous background per angle
        (reasonable if the object is much larger than the window), which we remove before any
        reconstruction.

        Looks for a constant `c_k` per angle such that:
          - the resulting mass `mass_k - c_k * extent` is the SAME for all angles (M),
          - the GLOBAL minimum (all angles, all bins combined) of the corrected sinogram is
            exactly 0 -- not necessarily the minimum of EACH angle individually:
            the most truncated angle (the one that would allow the smallest mass by removing all
            its own background) sets M, the other angles remove a constant smaller than their
            own minimum to reach that same mass M.

        Derivation (see the class docstring for the notations `mass_k`/`m_k` = min of the raw
        profile of angle k): writing `min_k( m_k − c_k ) = 0` and `c_k = ( mass_k − M ) /
        extent`, we get `M = max_k( mass_k − m_k·extent )`.
        """
        vals = np.asarray( self.values )
        m = vals.min( axis = 1 )                                  # [ nb_angles ]
        mass = vals.sum( axis = 1 ) * self.dw                     # [ nb_angles ]
        M = float( np.max( mass - m * self.extent ) )
        c = ( mass - M ) / self.extent                            # [ nb_angles ], <= m everywhere

        out = Sinogram( nb_angles = self.nb_angles.value, nb_bins = self.nb_bins.value,
                         extent = self.extent, detector_center = self.detector_center )
        out.values = vals - c[ :, None ]
        return out
