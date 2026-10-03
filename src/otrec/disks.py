"""DIFFERENTIABLE projection (Radon transform) of a union of disks of FIXED radius.

This is the geometric building block of the `DiskModel` model (`models.py`), which a `Reconstruction`
(`Reconstruction.py`) decreases: the measured sinogram is seen there as weighted diracs,
and the MODEL it is compared to is the piecewise-constant image produced here.

What we gain by modeling disks rather than diracs: the loss becomes a SMOOTH function of the
unknowns. The 1D OT cost is linear in the target density; this density is here
the projected image, whose dependence on the centers is the chord `2*sqrt(r^2 - u^2)` INTEGRATED over
each pixel -- hence C^1 in the projected position (see `DiskProjector._values_of`). There is
no longer any non-smooth term inherited from the movement of diracs.

Geometric conventions: those of `Sinogram` (angles theta_k = k*pi/nb_angles, normal
n_k = (cos, sin), detector coordinate s = p.n_k). The grid of the model image covers the SAME
extent as the detector, but may be finer (`nb_pixels`) to correctly represent
the projection of a small-radius disk.

Everything is BATCHED over `num_angle`: a single `SdotPlan1d` handles the `nb_angles` transports, and the
path taken is the purely Jax one of `Image.try_update_sdotplan1d` (no C++ kernel).
"""
import numpy as np

from loom import Tensor, driver, RealTensor
from sdot import Image

from .Sinogram import Sinogram


class DiskProjector:
    """Differentiable projection of a union of disks of FIXED RADIUS, discretized as a
    piecewise-constant function per angle -- the batched `Image` consumed by `SdotPlan1d`.

    The geometry (angles, detector extent, image grid, radius) is fixed at construction;
    only the CENTERS vary from one call to the next -- they are the unknown of the optimization.

    `nb_pixels`: number of cells of the image grid, over the same extent as the detector (by
    default `sinogram.nb_bins`). Taking it larger refines the representation of the projection
    (useful when `radius` is small compared to the width of a detector bin) without touching the
    measured data, which remain sampled on `nb_bins`.
    """

    def __init__( self, sinogram: Sinogram, radius: float, nb_pixels: int | None = None,
                  max_chunk_elems: int = 1 << 24 ) -> None:
        if radius <= 0:
            raise ValueError( "radius must be > 0" )
        self.sinogram = sinogram
        self.radius = float( radius )
        # HOST-side counts, frozen here (outside any trace): `values` is called UNDER the optimizer's
        # `jit`, where re-reading a ShapeVar would give a tracer (cf. `Sinogram.nb_bins_host`).
        self.nb_angles = int( sinogram.nb_angles.value )
        self.nb_pixels = int( nb_pixels if nb_pixels is not None else sinogram.nb_bins_host )
        if self.nb_pixels < 1:
            raise ValueError( "nb_pixels must be >= 1" )

        self.dw = sinogram.extent / self.nb_pixels
        self.s_min = sinogram.s_min
        # edges of the image cells, [ nb_pixels + 1 ] -- backend side (graph constant)
        self.edges = driver.array( self.s_min + self.dw * np.arange( self.nb_pixels + 1 ) )

        # bound (in elements) of the intermediate tensor `[ nb_angles, nb_disks, nb_pixels + 1 ]`:
        # beyond it, `values` splits the loop over the DISKS into slices and accumulates -- the result
        # is only `[ nb_angles, nb_pixels ]`, only the intermediate is big.
        self.max_chunk_elems = int( max_chunk_elems )

        # the contribution of a slice, with its intermediates RECOMPUTED in the backward instead of
        # being kept on the tape (see `values`). Wrapped once here, not at each call:
        # `driver.checkpoint` builds a transformation object, better to do it only once.
        self._values_of_chunk = driver.checkpoint( self._values_of )

    # -- geometry ----------------------------------------------------------

    @property
    def pixel_centers( self ) -> np.ndarray:
        """Centers of the cells of the image grid, [ nb_pixels ]."""
        return self.s_min + self.dw * ( np.arange( self.nb_pixels ) + 0.5 )

    def _chunk_size( self, nb_disks: int ) -> int:
        per_disk = max( 1, self.nb_angles * ( self.nb_pixels + 1 ) )
        return max( 1, min( nb_disks, self.max_chunk_elems // per_disk ) )

    # -- projection --------------------------------------------------------

    def _values_of( self, centers, weights = None ):
        """Contribution (density, `[ nb_angles, nb_pixels ]`) of a packet of centers, as a raw
        backend array.

        `weights` (optional, `[ nb_disks ]`) weights the contribution of each disk. Used
        only to NEUTRALIZE the padding centers of an incomplete slice (weight 0, see
        `values`): their mass AND their gradient become exactly zero. No "far enough"
        position would do instead: whatever point is chosen, there exists an angle
        where its projection falls back into the detector.

        The Radon profile of a disk of radius r is the chord `c(u) = 2*sqrt(r^2 - u^2)` (zero
        outside the disk), where `u = s - s0` is the offset from the projected center `s0 = center.n_k`. We
        INTEGRATE this chord over each cell (mass exactly conserved, like `Sinogram.add_disk`)
        via its primitive `H(u) = G(clip(u, -r, r))` with
        `G(t) = t*sqrt(r^2 - t^2) + r^2*arcsin(t/r)`, then divide by the cell width to
        obtain a density.

        DERIVATIVE: `H` is C^1 and `H'(u) = c(u)` exactly -- but autodifferentiating the expression
        of `G` above is numerically unusable. `G'` shows up there as a sum of two
        terms in `1/sqrt(r^2 - t^2)` that cancel each other (catastrophic cancellation when a cell
        edge grazes the edge of the disk), and where `clip` saturates the derivative is `0 * inf = NaN`.
        We therefore supply the EXACT derivative by hand, through a first-order surrogate:

            H = stop_gradient( G - c*u ) + stop_gradient( c ) * u

        whose VALUE is `G` (the `c*u` cancel) and whose DERIVATIVE is `c(u)` -- the right one, in
        a single well-conditioned evaluation. No gradient goes through `sqrt`/`arcsin`/
        `clip` anymore, hence no more NaN nor possible cancellation.
        """
        r = self.radius
        s0 = self.sinogram.project_points( centers ).value          # [ nb_angles, nb_disks ]
        u = self.edges[ None, None, : ] - s0[ :, :, None ]           # [ nb_angles, nb_disks, nb_pixels + 1 ]

        t = driver.clip( u, -r, r )
        sq = driver.sqrt( driver.clip( r * r - t * t, 0.0, None ) )
        chord = 2.0 * sq
        G = t * sq + r * r * driver.arcsin( driver.clip( t / r, -1.0, 1.0 ) )
        H = driver.stop_gradient( G - chord * u ) + driver.stop_gradient( chord ) * u

        mass = H[ :, :, 1: ] - H[ :, :, :-1 ]                        # mass per ( angle, disk, pixel )
        if weights is not None:
            mass = mass * weights[ None, :, None ]
        return driver.sum( mass, axis = 1 ) / self.dw                # sum over the disks -> density

    def values( self, centers ) -> Tensor:
        """Projected density `[ num_angle, num_pixel ]`, differentiable with respect to `centers`
        (`[ nb_disks, 2 ]`, Tensor or array).

        The intermediate has size `nb_angles * nb_disks * ( nb_pixels + 1 )` -- much bigger
        than the result, `[ nb_angles, nb_pixels ]`. Beyond `max_chunk_elems` the sum over the
        disks is therefore split into SLICES accumulated one by one, which changes nothing in the
        result (up to the summation order) and bounds the memory peak to ONE slice -- provided
        it is done correctly, which requires the two mechanisms together:

        - `driver.fold` (a compiled loop, not an unrolled Python loop) makes execution
          SEQUENTIAL. Unrolled, the loop instead lets the compiler schedule all the
          slices at once -- measured: the peak stayed proportional to the number of disks, and the
          compile time too;
        - `driver.checkpoint` on `_values_of` makes the slice be RECOMPUTED in the backward instead of
          keeping its intermediates, which the loop would otherwise store once per iteration.

        Measured (600 angles x 2000 pixels, gradient): 5.4 GB for 512 disks before, 0.4 GB after,
        and now independent of the number of disks -- the ~5000 alveoli of
        `experiments/lung_alveoli.py` required more than 50 GB, they now fit within what
        `max_chunk_elems` prescribes. The price is one extra forward per slice.

        The last slice is completed with padding centers neutralized by a weight of 0
        -- `driver.fold` requires FIXED-size iterations.
        """
        pts = centers if isinstance( centers, Tensor ) else RealTensor( centers )
        if pts.rank != 2 or pts.shape[ 1 ] != 2:
            raise ValueError( "centers must have shape [ nb_disks, 2 ]" )

        nb_disks = int( pts.shape[ 0 ] )
        chunk = self._chunk_size( nb_disks )
        raw = pts.value

        if chunk >= nb_disks:
            acc = self._values_of_chunk( raw )                       # a single slice: no loop
        else:
            nb_chunks = -( -nb_disks // chunk )
            pad = nb_chunks * chunk - nb_disks
            # the padding centers are (0,0): their position does not matter
            # since they carry zero weight.
            padded = raw if pad == 0 else driver.pad( raw, ( ( 0, pad ), ( 0, 0 ) ) )
            weights = np.ones( nb_chunks * chunk )
            weights[ nb_disks: ] = 0.0

            xs = { "centers": padded.reshape( nb_chunks, chunk, 2 ),
                   "weights": driver.array( weights.reshape( nb_chunks, chunk ) ) }
            # the accumulator update stays OUTSIDE the checkpoint: inside, the accumulator itself
            # would become a residual stored at each iteration (see `driver.fold`).
            acc = driver.fold(
                lambda cum, x: cum + self._values_of_chunk( x[ "centers" ], x[ "weights" ] ),
                driver.zeros( ( self.nb_angles, self.nb_pixels ) ), xs )

        return Tensor.wrap( acc, [ self.sinogram.num_angle.name, "num_pixel" ] )

    def image( self, centers ) -> Image:
        """The projection as a 1D `Image` batched over `num_angle` -- directly consumable as the
        distribution of an `SdotPlan1d`.

        `current_mass` is provided explicitly (`sum of densities * cell width`, the very
        definition of the measure of a 1D piecewise-constant image) rather than left to
        `Image._update_current_mass`, which computes it through a `driver.call` (C++ kernel). The computation
        is trivial in `Tensor` algebra and avoiding it keeps the WHOLE loss in the Jax graph: no
        C++ compilation nor FFI round trip at each optimizer step.
        """
        values = self.values( centers )
        mass = values.sum( axis = "num_pixel" ) * self.dw            # [ num_angle ]
        return Image(
            values = values,
            origin = [ self.s_min ],                                 # shared (common geometry)
            frame = [ [ self.dw ] ],                                 # shared
            current_mass = mass,
            batch_axes = [ self.sinogram.num_angle ],
        )
