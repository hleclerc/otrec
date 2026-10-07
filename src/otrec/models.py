"""The reconstruction MODELS: what a point cloud REPRESENTS, and what it costs.

A `Model` answers a single question -- "what is the cost of this point cloud, against the
measured sinogram?" -- with a scalar differentiable with respect to the points. This is the
function that `Reconstruction` (`Reconstruction.py`) decreases; the model is the ONLY
place that knows how the points are compared to the data.

The two available models both rely on semi-discrete 1D optimal transport
(`SdotPlan1d`, batched over the angles), but they SWAP its roles:

- `DiracModel`: the points are the UNKNOWN seen as diracs of equal mass, projected on the
  fly onto each detector; the TARGET is the measured profile, a piecewise-constant function.
  Robust and cheap, but the loss is not differentiable everywhere (moving diracs).
- `DiskModel`: the points are the CENTERS of 2D disks of FIXED radius; it is the MEASURED
  sinogram that becomes a sum of weighted diracs (one per detector bin, at the center of the bin,
  with weight the pixel value), and the MODEL is the piecewise-constant image of the projection
  of the disks (`disks.DiskProjector`). The loss there is smooth in the unknowns.

They expose the same interface (`name`, `point_axis`, `cost`, `radii`, `floor`), so that
`Reconstruction` chains them without ever testing their type: the same cloud can be converged as
diracs then refined as disk centers.

In 3D ( `Radiographs`, one 2D image per angle ), `ProjectedDiracModel` plays the role of
`DiracModel` with a per-angle 2D semi-discrete transport ( `SdotPlanNd` ) -- and only offers
the FUSED cost + gradient evaluation ( see its docstring ).
"""
import warnings
from abc import ABC, abstractmethod

from loom import Tensor
from loom import RealTensor
import numpy as np

from sdot import Iterative, OtProblem, ProjectedSumOfDiracs, SdotPlan1d, SumOfDiracs, SumOfDiracs1d

from .Radiographs import Radiographs
from .Sinogram import Sinogram
from .dirac_fused import diracs_cost, diracs_cost_grad
from .disks import DiskProjector


class Model( ABC ):
    """Common interface of the models. A model is IMMUTABLE and carries no points: it is
    built once for a sinogram (+ its parameters) and evaluated on successive clouds.
    """

    #: readable name, for the traces and the `Reconstruction` history
    name = "model"

    #: name of the "point" axis of the position tensors -- purely documentary (it appears in
    #: shapes and error messages), but each model names its points the way it sees them.
    point_axis = "num_point"

    #: WORLD radius of the points when it is PART of the model (`None` = points with no extent of their own).
    #: `Reconstruction.export_html` passes it as is to `export_positions_html`, which then
    #: draws the disks at their true size instead of an arbitrary display size.
    radii = None

    def __init__( self, sinogram: Sinogram ) -> None:
        self.sinogram = sinogram

    @abstractmethod
    def cost( self, points ) -> Tensor:
        """Scalar cost (rank-0 Tensor) of the cloud `points` (`[ n, 2 ]`, Tensor or array),
        differentiable with respect to it."""

    @property
    def floor( self ) -> float:
        """INCOMPRESSIBLE cost: the value `cost` reaches at best, even at the ground truth.
        Serves as a reference to judge a reconstruction (0 when there is none)."""
        return 0.0

    def wrap( self, raw ) -> Tensor:
        """The backend array `raw` (`[ n, 2 ]`) presented as the model's Tensor of points."""
        return Tensor.wrap( raw, [ self.point_axis, "dim" ] )

    def __repr__( self ) -> str:
        return f"{ type( self ).__name__ }( { self.name } )"


class DiracModel( Model ):
    """The points are DIRACS of equal mass: the reconstructed density is their sum.

    The projection `s = point.n_k` is NOT materialized: `ProjectedSumOfDiracs` keeps the SHARED
    2D points (a single copy for all angles) and the normal PER ANGLE, the kernel computing
    the 1D position on the fly -- instead of an `[ nb_angles, n ]` tensor (80 GB at 1e7 diracs x 1000
    angles). Still differentiable: the backward atomically scatters the gradient of the projected
    position onto the shared 2D points.

    `with_barycenters`: passed as is to `SdotPlan1d`. When the ONLY gradient requested is that
    of the positions (the case here), storing the barycenters spares the backward from re-sorting + re-sweeping
    each angle, at the price of an `[ nb_angles, n ]` buffer -- to be enabled if this memory cost is
    acceptable (see the docstring of `SdotPlan1d.__init__`).
    """

    name = "diracs"
    point_axis = "num_dirac"

    def __init__( self, sinogram: Sinogram, with_barycenters: bool = False ) -> None:
        super().__init__( sinogram )
        self.with_barycenters = with_barycenters

    def cost( self, points ) -> Tensor:
        pts = points if isinstance( points, Tensor ) else RealTensor( points )
        src = ProjectedSumOfDiracs( points = pts, normal = self.sinogram.normals_t,
                                    batch_axes = [ self.sinogram.num_angle ] )
        dst = self.sinogram.batched_image()
        return SdotPlan1d( src, dst, with_barycenters = self.with_barycenters ).cost.sum()  # sum over the angles

    def value_and_grad( self, points ):
        """Fused `(cost, grad)` via the fused kernel (`dirac_fused.diracs_cost_grad`) -- same
        formula as `cost` + `jax.grad`, but computed IN A SINGLE pass (see its docstring), without
        going through Jax autodiff. `points`: `[ n, proj_dim ]`, Tensor or raw array -- no
        need to `wrap` (unlike `cost`). `with_barycenters` makes no sense here (no Jax
        bwd): ignored. Consumed by `optimizers.FusedLBFGS` (see
        `Reconstruction.diracs( backend = "fused" )`)."""
        return diracs_cost_grad( points, self.sinogram )

    def value( self, points ) -> float:
        """`cost` ALONE (Python float), SAME fused kernel as `value_and_grad` but WITHOUT
        the gradient computation (`dirac_fused.diracs_cost`) -- for the "cost only" evaluations of a
        line search, where the gradient would be thrown away anyway."""
        return diracs_cost( points, self.sinogram )


class DiskModel( Model ):
    """The points are the CENTERS of 2D disks of FIXED radius.

    The measured sinogram is here the discrete source (`sinogram_diracs`) and the projection of the
    disks (`disks.DiskProjector`) the continuous target. Each slice (angle) normalizes its two
    distributions to mass 1: neither the number of disks nor the radius need to be calibrated
    on the measured mass, only the SHAPE of the profile matters.

    `nb_pixels`: fineness of the model grid, over the same extent as the detector (by default
    that of the detector). Refining it better represents the projection of a small radius without touching
    the measured data -- it is a FREE choice, independent of `nb_bins`.

    `max_chunk_elems`: the size of the disk slices processed one by one, hence the MEMORY PEAK
    of the gradient -- bounded to one slice whatever the number of disks (see
    `DiskProjector.values`). Measured on the lung case (600 angles x 2000 pixels), the time only
    varies by ~20% between 3 and 223 disks per slice: the computation is limited by memory
    bandwidth, not by the number of slices -- so no point raising this bound to
    go faster, it only serves to choose the memory one is willing to consume.
    """

    name = "disks"
    point_axis = "num_disk"

    def __init__( self, sinogram: Sinogram, radius: float, nb_pixels: int | None = None,
                  max_chunk_elems: int = 1 << 24 ) -> None:
        super().__init__( sinogram )
        self.projector = DiskProjector( sinogram, radius = radius, nb_pixels = nb_pixels,
                                        max_chunk_elems = max_chunk_elems )
        self.radii = self.projector.radius       # common radius, exported as is to the visualization

    def cost( self, points ) -> Tensor:
        src = sinogram_diracs( self.sinogram )
        dst = self.projector.image( points )
        return SdotPlan1d( src, dst ).cost.sum()                        # sum over the angles

    @property
    def floor( self ) -> float:
        """QUANTIZATION floor: even at exact centers the loss is not 0, because the
        diracs condense each detector bin into its center -- which costs the variance of a
        uniform bin, `dw^2 / 12`, per angle."""
        return float( self.sinogram.nb_angles.value ) * self.sinogram.dw ** 2 / 12


class ProjectedDiracModel( Model ):
    """The points are 3D DIRACS of equal mass, confronted with RADIOGRAPHS
    ( `Radiographs` ): at each angle, their projections onto the detector are transported to
    the measured image by a 2D semi-discrete transport ( `SdotPlanNd`, one power diagram per
    angle ), and the cost is the sum over the angles of the `W_2^2`.

    The 3D counterpart of `DiracModel`, with a difference in nature: the 1D transport is EXACT ( a
    sort ), the 2D transport is a weight FIT ( `SdotPlanNd._fit`, iterative ). Hence:

    - the gradient with respect to the points does not go through autodiff but through the
      ENVELOPE theorem, at the fitted weights: `2 m_i ( p_i - b_i )` on the detector ( `b_i` the barycenter
      of the Laguerre cell, see `SdotPlanNd.cost_and_position_grad` ), then lifted to 3D by the
      transpose of the projection ( `Radiographs.unproject_grad` ). The model therefore only has a
      fused `value_and_grad` ( `FusedLBFGS` ) -- `cost` returns a float, not a `Tensor`;
    - each angle is solved by the damped Newton of `SdotPlanNd` ( all in C++ ), and the fitted
      weights are KEPT from one evaluation to the next ( `weights0` of the next `SdotPlanNd` ):
      points that move little require weights that move little, a few steps suffice. It is
      a cache, not a state -- the result does not depend on it, and it no longer has to be discarded when it
      empties a cell: the C++ itself picks the best of the three starts it knows ( the given
      weights, the Voronoi, the similarity that brings the cloud back into the detector ) and says which
      in `stats[ "start" ]`;
    - a projection is 2D, so the Newton step there is the one from the LIMITS ( `step = "auto"`
      -> `"limits"` ): the maximal relaxation coefficient of the cells that get squashed along
      the direction, computed EXACTLY instead of being searched by successive trials -- hence
      backtracks that have become rare ( 1.8 per fit on the case measured below, versus 515 when the
      problem is ill-posed );
    - the radiograph receives a BACKGROUND ( `background`, as a fraction of its mean value ), and it
      remains INDISPENSABLE: a projected ball is zero outside its shadow, and a cell that only
      sees zeros has NO weight that gives it its mass -- the problem has no solution,
      not just a bad starting point. The WIDTH CONTINUATION of `SdotPlanNd`
      ( `continuation = "auto"`: the wide convolved density first, tightened step by step,
      each restarting from the weights of the previous one ) softens the PATH, not the target: its last
      step is the density itself, zeros included. Measured ( 200 diracs, 6 angles, 64 x 64,
      start in the cube, `notes/2026-09-23-otrec-3d.md` ): without background, 36 fits out of 48 do not
      converge, 515 backtracks per call, and the loss GOES UP ( 2.6 -> 8.0 ); with a background of 1e-6, all
      converge, 22 diagrams per call, 98 % of the diracs end up in the balls;
    - what the continuation does change, however, is that a LOW background becomes usable.
      Same case, `continuation = "never"` versus `"auto"`, at background 1e-6: 25 fits out of 78 do not
      converge, 950 diagrams per call, 46 % of the diracs in the balls -- versus no
      failure, 22 diagrams and 98 %. At background 1e-3 it costs nothing and saves a third of the
      diagrams by eliminating the backtracks;
    - the STARTING point still matters, for what it costs: diracs drawn in the whole cube
      cause continuation steps to be redone at each angle and each evaluation, whereas
      `Reconstruction.hull_points` ( the visual hull ) starts right away near the object.
    """

    name = "diracs 3D"
    point_axis = "num_dirac"
    #: no traceable `cost`: only `value_and_grad` exists ( see the docstring )
    fused_only = True

    def __init__( self, radiographs: Radiographs, background: float = 1e-3, max_iter: int = 100,
                  mass_tol: float = 1e-4, kernel_dtype = None, continuation: str = "auto",
                  strict: bool = False ) -> None:
        """`background`, `max_iter`, `mass_tol`, `continuation`: see the class docstring and
        `SdotPlanNd`. The Newton is the KMT one, damped ( the UNdamped Newton was measured 5 to 50 times
        slower, blurred or not -- see `notes/2026-09-14-reconstruction-3d.md` -- and no longer
        exists ), the step coming from the LIMITS since a projection is 2D.

        `mass_tol` is RELATIVE to the mass of a dirac ( `1 / n` ) -- and bounded by what the kernel
        knows: in FP32, the area of a cell is only known to ~1e-5 relative, below that the
        Newton no longer finds a step that lowers the residual. Hence `kernel_dtype = None` by default,
        which lets `SdotPlanNd` choose FP64 -- FP32 is to be reserved for trials.

        `strict`: raise as soon as an angle does not converge, instead of counting it in
        `solver_stats` and returning the cost anyway ( an unfinished fit gives a wrong
        gradient by the envelope theorem, which assumes optimal weights )."""
        super().__init__( radiographs )
        self.radiographs = radiographs
        self.background = float( background )
        self.max_iter = int( max_iter )
        self.mass_tol = float( mass_tol )
        self.precision = { None: "auto", "FP64": "fp64", "FP32": "fp32" }.get( kernel_dtype, kernel_dtype )
        self.continuation = continuation
        self.strict = bool( strict )
        nb_angles = int( radiographs.nb_angles.value )
        self._images = [ radiographs.image( k, background = self.background ) for k in range( nb_angles ) ]
        # ONE PROBLEM PER ANGLE, kept from one evaluation to the next: it is the one that carries the weights of
        # the last solution, so the warm start no longer has to be copied here ( see
        # `OtProblem.solve` ). Built lazily -- the source only exists at the first
        # evaluation, and it changes size from one `Reconstruction.multiscale` stage to the next.
        self._problems = [ None ] * nb_angles
        # a target that keeps ZERO pixels cannot be transported ( see the class docstring:
        # measured, the fits do not converge there and the loss goes up ). We say so once, instead of
        # letting a multi-hour descent return a wrong cloud.
        if self.background <= 0 and float( np.asarray( radiographs.values ).min() ) <= 0:
            warnings.warn( "ProjectedDiracModel: the radiographs have ZERO pixels and no background "
                           "is added ( background = 0 ) -- the per-angle fits will not converge "
                           "( the continuation softens the path, not the target ). Give "
                           "`background > 0` ( 1e-6 is enough ).", stacklevel = 2 )
        #: what the fits have cost since the start, all angles combined ( see
        #: `solver_line` ) -- what we look at to know whether the cloud is still far off
        self.solver_stats = dict( nb_calls = 0, nb_iter = 0, nb_diag = 0, nb_backtracks = 0, nb_continuation_steps = 0,
                                  nb_voronoi = 0, nb_similarity = 0, nb_not_converged = 0 )

    def _plan( self, k, uv ):
        """the transport of angle `k`, fitted -- restarting from the weights of the last time"""
        # the NEWTON on the dual functional: a number of steps independent of the number of diracs,
        # and, from one evaluation to the next, only a few steps -- `OtProblem` keeps the weights of
        # its last solution and takes them as the start, including when the cloud has changed
        # SIZE ( a `Reconstruction.multiscale` stage: it then invalidates them by itself ). Weights that
        # empty a cell are not discarded: the C++ itself compares the starts
        # it knows and keeps the best ( `stats[ "start" ]` ).
        pb = self._problems[ k ]
        if pb is None:
            pb = self._problems[ k ] = OtProblem( SumOfDiracs( uv ), self._images[ k ] )
        else:
            pb.source = SumOfDiracs( uv )
        plan = pb.solve( Iterative( max_iter = self.max_iter, tol = self.mass_tol / len( uv ),
                                    precision = self.precision, continuation = self.continuation,
                                    on_failure = "ignore" ) )          # ( an approximate plan serves the descent; `solver_stats` counts them )
        self._account( k, plan )
        return plan

    def _account( self, k, plan ):
        """what the fit of angle `k` has cost, accumulated in `solver_stats`"""
        st = plan.stats
        s = self.solver_stats
        s[ "nb_calls" ] += 1
        for name in ( "nb_iter", "nb_diag", "nb_backtracks", "nb_continuation_steps" ):
            s[ name ] += st[ name ]
        if st[ "start" ] == "voronoi":
            s[ "nb_voronoi" ] += 1
        elif st[ "start" ] == "similarity":
            s[ "nb_similarity" ] += 1
        if not plan.converged:
            s[ "nb_not_converged" ] += 1
            if self.strict:
                raise RuntimeError( f"ProjectedDiracModel: angle { k } did not converge "
                                    f"( { st[ 'status' ] }, remainder { st[ 'residual' ]:.3e} ) -- the envelope "
                                    "gradient assumes optimal weights" )

    def solver_line( self ) -> str:
        """A line on what the fits have cost, AVERAGED per solved angle -- to see
        at a glance whether the cloud is still far off ( continuation steps, Voronoi starts )
        or already warm ( two or three Newton steps, no step )."""
        s = self.solver_stats
        nb = max( 1, s[ "nb_calls" ] )
        return ( f"{ s[ 'nb_calls' ] } fits: { s[ 'nb_iter' ] / nb:.1f} steps, "
                 f"{ s[ 'nb_diag' ] / nb:.1f} diagrams, { s[ 'nb_continuation_steps' ] / nb:.2f} continuation steps, "
                 f"{ s[ 'nb_backtracks' ] / nb:.2f} backtracks, starts { s[ 'nb_voronoi' ] } Voronoi / "
                 f"{ s[ 'nb_similarity' ] } similarity, { s[ 'nb_not_converged' ] } not converged" )

    def value_and_grad( self, points ):
        """`( cost, grad )`, `grad` of shape `[ n, 3 ]` -- see the class docstring."""
        pts = np.asarray( points, dtype = float ).reshape( -1, 3 )
        proj = self.radiographs.project_points( pts )                           # [ nb_angles, n, 2 ]
        cost, grad_uv = 0.0, np.zeros_like( proj )
        for k in range( len( proj ) ):
            c, g = self._plan( k, proj[ k ] ).cost_and_position_grad()
            cost += c
            grad_uv[ k ] = g
        return cost, self.radiographs.unproject_grad( grad_uv )

    def value( self, points ) -> float:
        pts = np.asarray( points, dtype = float ).reshape( -1, 3 )
        proj = self.radiographs.project_points( pts )
        return float( sum( self._plan( k, proj[ k ] ).cost for k in range( len( proj ) ) ) )

    def cost( self, points ):
        """The cost -- a FLOAT ( not differentiable by autodiff, see the docstring )."""
        return self.value( points )

    def wrap( self, raw ) -> Tensor:
        return Tensor.wrap( raw, [ self.point_axis, "dim" ] )


def sinogram_diracs( sinogram: Sinogram ) -> SumOfDiracs1d:
    """The MEASURED sinogram seen as a sum of weighted diracs, batched over the angles: one
    dirac per detector bin, placed at the CENTER of the bin and with weight the pixel value.

    The positions (bin centers) are the same at all angles: they are therefore SHARED
    (`[ nb_bins ]`, not `[ nb_angles, nb_bins ]`); only the weights are batched. This is what
    `raw_1d_diracs` then passes on to the pure-Jax path of `Image.try_update_sdotplan1d`, which
    never materializes more than one angle at a time.
    """
    return SumOfDiracs1d(
        positions = sinogram.bin_centers,
        weights = sinogram.values,
        batch_axes = [ sinogram.num_angle ],
    )
