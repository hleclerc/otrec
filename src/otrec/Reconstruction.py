"""SINGLE entry point for CT reconstruction: the `Reconstruction` class.

It carries the STATE (the measured sinogram, the current point cloud) and the DEFAULT
PARAMETERS (disk radius, model grid fineness, optimizer, seed, frame recording),
and exposes the algorithms as CHAINABLE methods: each one starts from the current cloud,
replaces it, and returns `self`.

    rec = Reconstruction( sino, radius = 0.15, record = True, verbose = True )
    rec.random_points( 60 ).diracs( max_iter = 100 ).disks( max_iter = 300 )
    rec.export_html( "out.html" )

This chaining is what lets us COMPOSE the two models of `models.py`: start with a
reconstruction in DIRACS (cheap, insensitive to the initial draw, but whose loss is not
differentiable everywhere) then take EXACTLY the same points over as DISK centers, whose
loss is smooth -- each step only having to refine what the previous one found.

No method assumes the order: `disks()` alone, `diracs()` alone, `multiscale()` then `disks()`
are all valid. The history (`history`) keeps one line per step, and `frames` the complete
trajectory when `record` is active -- enough to replay the whole sequence in a single HTML
page.
"""
import time

import numpy as np

from loom import Tensor, driver

from .Sinogram import Sinogram
from .models import DiracModel, DiskModel, Model, ProjectedDiracModel
from .optimizers import FusedLBFGS, LBFGS
from .viz.points_html import export_positions_html


class Reconstruction:
    """CT reconstruction of a point cloud from a sinogram, in chainable steps.

    `sinogram`: the data the points are confronted with. Constant in practice, but
    replaceable (`set_sinogram`) -- which is what the halo/interior alternation of
    `halo.alternate` needs, resubmitting a corrected sinogram at each pass.
    `points`: initial cloud `[ n, 2 ]` (Tensor or array) -- optional, `random_points` can
    draw it later.

    Default parameters, all overridable when calling each step:
    - `radius` / `nb_pixels` / `max_chunk_elems`: the DISKS model (`radius` is mandatory for
      `disks()`, but can be given either here or at the call; `max_chunk_elems` sets the
      gradient's memory peak, independently of the number of disks -- see `DiskProjector.values`);
    - `with_barycenters`: the DIRACS model (memory/speed trade-off of the backward);
    - `optimizer`, or failing that `max_iter`/`ftol` which define the L-BFGS used;
    - `extent`: the extent of the draws/of the split noise (by default that of the detector);
    - `seed`: the seed of EVERY random draw (incremented at each use, so that two
      successive draws are not identical);
    - `record` / `record_every`: capture of the positions along the steps, for the animation;
    - `verbose`: one summary line per step.
    """

    def __init__( self, sinogram: Sinogram, points = None, *,
                  radius: float | None = None, nb_pixels: int | None = None,
                  max_chunk_elems: int = 1 << 24, with_barycenters: bool = False,
                  optimizer = None, max_iter: int = 200, ftol: float = 1e-12,
                  min_iter: int = 0, disp_tol: float | None = None,
                  extent: float | None = None, seed: int = 0,
                  record: bool = False, record_every: int = 1, verbose: bool = False ) -> None:
        self.sinogram = sinogram
        #: the dimension of the point space: 2 for a `Sinogram`, 3 for `Radiographs`
        self.dim = int( getattr( sinogram, "world_dim", 2 ) )

        self.radius = radius
        self.nb_pixels = nb_pixels
        self.max_chunk_elems = max_chunk_elems
        self.with_barycenters = with_barycenters

        self.optimizer = optimizer
        self.max_iter = max_iter
        self.ftol = ftol
        #: see `LBFGS.min_iter`/`disp_tol` -- defaults applied to EACH stage (`diracs`/`disks`/
        #: `multiscale`) that does not provide its own `optimizer`, overridable at the call.
        self.min_iter = min_iter
        self.disp_tol = disp_tol

        self.extent = float( extent if extent is not None else sinogram.extent )
        self.seed = int( seed )
        self.record = record
        self.record_every = max( 1, int( record_every ) )
        self.verbose = verbose

        #: the current cloud, `[ n, 2 ]` -- the state that each step transforms
        self.points: Tensor | None = None
        #: captured trajectory (list of `[ n_t, 2 ]` arrays), if `record`
        self.frames: list[ np.ndarray ] = []
        #: one line per executed step: model, optimizer, losses, number of steps, duration
        self.history: list[ dict ] = []
        #: world radius of the points, inherited from the last model played (see `export_html`)
        self.radii = None
        #: the last model played (`run`) -- what an experiment queries afterwards when the
        #: model was built here and not by it (`anneal_blur`, `multiscale` without `model`):
        #: `ProjectedDiracModel.solver_line()`, for example.
        self.model: Model | None = None

        if points is not None:
            self.set_points( points )

    # -- state -------------------------------------------------------------

    @property
    def nb_points( self ) -> int:
        if self.points is None:
            return 0
        return int( self.points.shape[ 0 ] )

    @property
    def positions( self ) -> np.ndarray:
        """The current cloud as numpy `[ n, 2 ]` (host copy, for measurement/display)."""
        return np.asarray( self.points )

    def set_points( self, points ) -> "Reconstruction":
        """Replaces the current cloud. Accepts a Tensor, an `[ n, 2 ]` array, or another
        `Reconstruction` (its cloud is then taken over -- to restart from an existing result)."""
        if isinstance( points, Reconstruction ):
            points = points.points
        raw = points.raw if isinstance( points, Tensor ) else driver.array( np.asarray( points, dtype = float ) )
        if raw.ndim != 2 or raw.shape[ 1 ] != self.dim:
            raise ValueError( f"points must have shape [ n, { self.dim } ], got { tuple( raw.shape ) }" )
        self.points = Tensor.wrap( raw, [ "num_point", "dim" ] )
        return self

    def set_sinogram( self, sinogram: Sinogram ) -> "Reconstruction":
        """Replaces the data the points are confronted with, KEEPING the current cloud.

        The models are rebuilt at each step from `self.sinogram` (`dirac_model` /
        `disk_model`), nothing is cached here: the swap is therefore immediate. Used by
        the halo/interior alternation (`halo.alternate`), where each pass restarts from the previous cloud
        -- warm-started, that is -- on a better corrected sinogram.
        """
        self.sinogram = sinogram
        return self

    def _next_seed( self, seed: int | None ) -> int:
        """The seed to use: the one requested, or the default seed -- which we ADVANCE,
        so that two successive draws on the same object do not repeat."""
        if seed is not None:
            return int( seed )
        self.seed += 1
        return self.seed - 1

    def random_points( self, nb_points: int, extent: float | None = None,
                       seed: int | None = None ) -> "Reconstruction":
        """`nb_points` positions drawn uniformly in [ -extent/2, extent/2 ]^dim."""
        rng = np.random.default_rng( self._next_seed( seed ) )
        e = float( extent if extent is not None else self.extent )
        return self.set_points( ( rng.random( ( nb_points, self.dim ) ) - 0.5 ) * e )

    def hull_points( self, nb_points: int, extent: float | None = None, seed: int | None = None,
                     threshold: float = 0.0 ) -> "Reconstruction":
        """`nb_points` positions drawn in the VISUAL HULL of the data ( the points whose
        projections all fall on matter -- see `Radiographs.visual_hull_points` ),
        instead of the whole cube: the starting point that a 2D per-angle transport requires."""
        hull = getattr( self.sinogram, "visual_hull_points", None )
        if hull is None:
            raise TypeError( f"{ type( self.sinogram ).__name__ } cannot draw within its visual hull" )
        return self.set_points( hull( nb_points, seed = self._next_seed( seed ), extent = extent, threshold = threshold ) )

    def split( self, factor: int = 4, noise_frac: float = 0.05,
               seed: int | None = None ) -> "Reconstruction":
        """Replaces each point by `factor` children, superimposed on the parent then scattered by a
        uniform noise (per axis).

        `noise_frac`: noise amplitude as a fraction of the typical spacing at the CURRENT level
        (`extent / sqrt( n )`) -- the children must explore the IMMEDIATE neighborhood of the parent,
        otherwise each refinement would become a global rearrangement again (see `multiscale`).
        """
        if self.points is None:
            raise ValueError( "no point to split -- call `random_points` or `set_points` first" )
        rng = np.random.default_rng( self._next_seed( seed ) )
        noise_scale = noise_frac * self.extent / np.sqrt( max( 1, self.nb_points ) )
        tiled = np.repeat( self.positions, factor, axis = 0 )                     # [ n*factor, 2 ]
        return self.set_points( tiled + ( rng.random( tiled.shape ) - 0.5 ) * 2 * noise_scale )

    def subsample( self, nb_points: int, seed: int | None = None ) -> "Reconstruction":
        """Keeps `nb_points` points drawn without replacement (no effect if there are already fewer)."""
        if self.nb_points <= nb_points:
            return self
        idx = np.random.default_rng( self._next_seed( seed ) ).choice( self.nb_points, nb_points, replace = False )
        return self.set_points( self.positions[ idx ] )

    # -- models and optimizer ----------------------------------------------

    def dirac_model( self, with_barycenters: bool | None = None, **kwargs ) -> Model:
        """The diracs model of the data: `DiracModel` on a `Sinogram`, `ProjectedDiracModel`
        on `Radiographs` ( `kwargs` -> the latter: `background`, `max_iter`, ... )."""
        if self.dim == 3:
            return ProjectedDiracModel( self.sinogram, **kwargs )
        return DiracModel( self.sinogram,
                           with_barycenters = self.with_barycenters if with_barycenters is None else with_barycenters )

    def disk_model( self, radius: float | None = None, nb_pixels: int | None = None,
                    max_chunk_elems: int | None = None ) -> DiskModel:
        r = self.radius if radius is None else radius
        if r is None:
            raise ValueError( "the disks model requires a radius: `Reconstruction( ..., radius = ... )` "
                              "or `disks( radius = ... )`" )
        return DiskModel(
            self.sinogram, radius = r,
            nb_pixels = self.nb_pixels if nb_pixels is None else nb_pixels,
            max_chunk_elems = self.max_chunk_elems if max_chunk_elems is None else max_chunk_elems,
        )

    def default_model( self ) -> Model:
        """The implied model when none is specified (`loss()`, `multiscale()`): DISKS
        if a radius was set at construction, DIRACS otherwise."""
        return self.dirac_model() if self.radius is None else self.disk_model()

    def default_optimizer( self, max_iter: int | None = None, ftol: float | None = None,
                           min_iter: int | None = None, disp_tol: float | None = ...,
                           fused: bool = False ):
        """The optimizer explicitly provided at construction, or failing that an L-BFGS calibrated by
        `max_iter`/`ftol`/`min_iter`/`disp_tol` (see `LBFGS`, the most efficient measured on this
        problem elsewhere, cf. `benchmarks/optimizers`).

        `disp_tol` distinguishes `None` (intended, disables the criterion) from "not provided" (`...`, inherits
        the `Reconstruction` default) -- the three other parameters do not need this,
        `None` there already meaning "inherit the default".

        `fused`: builds a `FusedLBFGS` (see `Reconstruction.diracs( backend = "fused" )`) instead of
        an `LBFGS` -- then ignores `self.optimizer` (a generic optimizer provided at
        construction cannot consume a fused `value_and_grad`)."""
        if ( not fused and self.optimizer is not None and max_iter is None and ftol is None
             and min_iter is None and disp_tol is ... ):
            return self.optimizer
        cls = FusedLBFGS if fused else LBFGS
        return cls( max_iter = self.max_iter if max_iter is None else max_iter,
                   ftol = self.ftol if ftol is None else ftol,
                   min_iter = self.min_iter if min_iter is None else min_iter,
                   disp_tol = self.disp_tol if disp_tol is ... else disp_tol )

    # -- evaluation --------------------------------------------------------

    def loss( self, model: Model | None = None, points = None ) -> float:
        """Loss of the cloud (current, or `points` if provided) for `model` (or `default_model`)."""
        model = model or self.default_model()
        pts = self.points if points is None else points
        if pts is None:
            raise ValueError( "no point to evaluate" )
        return float( model.cost( pts ) )

    def floor( self, model: Model | None = None ) -> float:
        """The incompressible cost of the model -- the value to compare `loss()` against."""
        return ( model or self.default_model() ).floor

    # -- algorithms --------------------------------------------------------

    def run( self, model: Model, optimizer = None, max_iter: int | None = None,
             ftol: float | None = None, min_iter: int | None = None, disp_tol: float | None = ...,
             callback = None, label: str | None = None ) -> "Reconstruction":
        """ONE optimization step: descends `model.cost` from the current cloud, which it
        replaces with the result.

        `callback( step, points )` is called at each step, `step = -1` designating the initial state.
        Frame recording (if `record`) and the `history` line are handled here, hence
        common to all steps -- `diracs`/`disks`/`multiscale` only choose the model.

        `min_iter`/`disp_tol`: see `LBFGS` -- ignored if `optimizer` is explicitly provided.
        """
        if self.points is None:
            raise ValueError( "no starting point -- call `random_points` or `set_points` first" )
        # a model that has ONLY the fused evaluation ( `ProjectedDiracModel` ) imposes the optimizer
        # that consumes it
        fused_only = getattr( model, "fused_only", False )
        optimizer = optimizer if optimizer is not None else self.default_optimizer( max_iter, ftol, min_iter, disp_tol, fused = fused_only )
        # set BEFORE the descent: a `callback` looks at the model along the way
        self.model = model

        p = self.points.raw
        loss_before = float( model.cost( model.wrap( p ) ) )
        nb_steps = [ 0 ]

        # the initial state is only captured once: chaining two steps must not duplicate the
        # junction frame (the starting point of one IS the arrival of the other).
        if self.record and not self.frames:
            self._capture( p )
        if callback is not None:
            callback( -1, model.wrap( p ) )

        def scalar_loss( q ):
            return model.cost( model.wrap( q ) ).value

        def step_callback( step, x ):
            nb_steps[ 0 ] = step + 1
            if self.record and ( step + 1 ) % self.record_every == 0:
                self._capture( x )
            if callback is not None:
                callback( step, model.wrap( x ) )

        t0 = time.time()
        # `FusedLBFGS` consumes `model.value_and_grad` (fused cost+gradient, e.g. the fused
        # kernel of `DiracModel`) instead of a jax-traceable scalar function -- see
        # `optimizers.FusedLBFGS` and `Reconstruction.diracs( backend = "fused" )`.
        if isinstance( optimizer, FusedLBFGS ):
            fused = getattr( model, "value_and_grad", None )
            if fused is None:
                raise ValueError( f"{ model !r } does not support fused evaluation "
                                  "(FusedLBFGS) -- it lacks `value_and_grad`" )
            p_opt = optimizer.minimize( fused, p, callback = step_callback )
        else:
            p_opt = optimizer.minimize( scalar_loss, p, callback = step_callback )
        dt = time.time() - t0

        self.set_points( model.wrap( p_opt ) )
        self.radii = model.radii
        loss_after = float( model.cost( model.wrap( self.points.raw ) ) )

        self.history.append( dict(
            stage = len( self.history ), model = model.name, label = label or model.name,
            optimizer = type( optimizer ).__name__, nb_points = self.nb_points,
            loss_before = loss_before, loss_after = loss_after, floor = model.floor,
            nb_steps = nb_steps[ 0 ], time = dt,
        ) )
        if self.verbose:
            print( self._summary_line( self.history[ -1 ] ) )
        return self

    def diracs( self, with_barycenters: bool | None = None, backend: str = "jax",
               **kwargs ) -> "Reconstruction":
        """One step with the DIRACS model (see `models.DiracModel`). `kwargs` -> `run`.

        `backend`: `"jax"` (default) descends `model.cost` via `jax.grad`, general-purpose (compatible
        with any `optimizer`). `"fused"` instead uses the fused loom kernel
        `dirac_fused.diracs_cost_grad` (`DiracModel.value_and_grad`, cost AND gradient IN A SINGLE
        pass, closed formula -- ~10-30x faster measured on the lung, see
        `benchmarks/execution_speed/benchmark_fused.py`): forces the default optimizer to
        `FusedLBFGS` (see `default_optimizer`), unless `optimizer` is explicitly provided.
        """
        # what goes to the MODEL and not to the descent ( `run` ) -- the per-angle transport settings
        # of `ProjectedDiracModel` ( `background`, `continuation`, ... ). `max_iter` is not one of them:
        # it is L-BFGS's, the Newton cap is called `mass_tol`/`max_iter` INSIDE the model.
        model_kwargs = { k: kwargs.pop( k ) for k in ( "background", "continuation", "mass_tol",
                                                       "kernel_dtype", "strict" ) if k in kwargs }
        model = self.dirac_model( with_barycenters, **model_kwargs )
        # a model that has ONLY the fused evaluation ( `ProjectedDiracModel` ) imposes its optimizer
        if getattr( model, "fused_only", False ):
            backend = "fused"
        if backend == "fused":
            kwargs.setdefault( "optimizer", self.default_optimizer(
                kwargs.get( "max_iter" ), kwargs.get( "ftol" ),
                kwargs.get( "min_iter" ), kwargs.get( "disp_tol", ... ), fused = True ) )
        elif backend != "jax":
            raise ValueError( f"unknown backend: { backend !r } (expected 'jax' or 'fused')" )
        return self.run( model, **kwargs )

    def disks( self, radius: float | None = None, nb_pixels: int | None = None,
               max_chunk_elems: int | None = None, split_before: int = 1,
               split_noise_frac: float = 0.05, **kwargs ) -> "Reconstruction":
        """One step with the DISKS model (see `models.DiskModel`). `kwargs` -> `run` (including
        `min_iter`/`disp_tol`, see `LBFGS` -- useful here: the disks loss has FLAT
        directions -- e.g. a disk can slide without changing the residual as long as it overlaps
        nobody -- where L-BFGS-B may conclude convergence after 0-1 steps).

        Typically chained AFTER `diracs`/`multiscale`: the converged points become the
        disk centers there, with no other transformation.

        `split_before`: if > 1, SUBDIVIDES the current cloud (see `split`) just before the disks
        stage -- more fixed-radius centers give more degrees of freedom to rearrange
        locally. Useful if the inherited cloud (typically one dirac per sinogram bin,
        see `models.sinogram_diracs`) is too sparse for the requested radius and ends up
        LOCKED (overlaps imposed by the neighborhood, no local move
        can lower the loss). `1` (default) changes nothing -- historical behavior.
        """
        if split_before > 1:
            self.split( factor = split_before, noise_frac = split_noise_frac )
        return self.run( self.disk_model( radius, nb_pixels, max_chunk_elems ), **kwargs )

    def anneal_blur( self, blurs = ( 1.0, 0.25, 0.06, 0.015, 0.0 ), model_kwargs = None,
                     stage_callback = None, **kwargs ) -> "Reconstruction":
        """BLURRED projections first, tightened stage by stage: for each `sigma` of
        `blurs` ( as a fraction of the detector extent; `0` = the data as is ), the
        diracs model of the blurred data ( `Radiographs.blurred` / `Sinogram.blurred` ) is descended from
        the current cloud. `kwargs` -> `run` ( `max_iter`, ... ), `model_kwargs` -> `dirac_model`.

        The blur here acts on the OUTER problem -- the POSITIONS: a blurred target has
        gradients that carry far, a badly placed cloud settles into it gently, and each following stage
        starts from a cloud already in place for a barely sharper target.

        Not to be confused with the WIDTH CONTINUATION of `SdotPlanNd` ( `continuation = "auto"` ),
        which blurs the same density but INSIDE a fit, to find the WEIGHTS of a given
        cloud: its steps end on the data AS IS, so it dispenses neither with the
        `background` ( a cell that only sees zeros has no weight that gives it its
        mass ) nor with this blur, which changes the TARGET and hence the landscape of positions. The two
        compose: `model_kwargs` passes `background`/`continuation` to the model of each stage.

        What the blur STILL buys has to be remeasured case by case: on the balls of
        `test_reconstruction_3d`, from the cube, 200 diracs, the sharp data ( background 1e-6 ) ends at
        98 % of the diracs in the balls and a loss of 1.0e-2, versus 99 % and 9.9e-3 after four
        blur stages -- see `notes/2026-09-23-otrec-3d.md`.
        """
        blurred = getattr( self.sinogram, "blurred", None )
        if blurred is None:
            raise TypeError( f"{ type( self.sinogram ).__name__ } cannot be blurred" )
        data = self.sinogram
        try:
            for stage, sigma in enumerate( blurs ):
                self.sinogram = blurred( sigma * getattr( data, "extent_u", data.extent ) ) if sigma > 0 else data
                model = self.dirac_model( **( model_kwargs or {} ) )
                self.run( model, label = f"{ model.name } blur { sigma:g}", **kwargs )
                if stage_callback is not None:
                    stage_callback( stage, sigma, self.points )
        finally:
            self.sinogram = data
        return self

    def multiscale( self, nb_points_final: int, nb_points_init: int = 1000, factor: int = 4,
                    noise_frac: float = 0.05, model: Model | None = None,
                    optimizer_factory = None, stage_callback = None, **kwargs ) -> "Reconstruction":
        """Coarse -> fine: converges at `nb_points_init` points, then repeats { `split` (each point
        -> `factor` noisy children), reconverge } up to `nb_points_final`.

        Motivation: at `nb_points_final` directly (e.g. 1e7), a UNIFORM initial draw must be
        globally rearranged at once by the optimizer (the target mass is concentrated on a
        tiny fraction of the support) -- observed to make the L-BFGS-B line search fail
        (stop after 1-2 iterations, far from convergence). In stages, each level inherits the
        right GLOBAL structure and only has to refine LOCALLY: much better conditioned.

        Starts from the current cloud if there is one (so an existing result can be refined), otherwise
        from a draw of `nb_points_init` points -- in the visual hull when the data can
        draw there (`hull_points`, for `Radiographs`), uniform otherwise. `optimizer_factory( n )`, if
        provided, gives the optimizer of the stage with `n` points; `stage_callback( stage, n, points )`
        is called after convergence of each stage, before the next split.

        In 3D it is also what makes large clouds affordable: each stage starts from an already
        well-placed cloud, so its per-angle fits (`ProjectedDiracModel`) start near
        their solution -- whereas a cloud drawn at once at `nb_points_final` makes them restart from the
        Voronoi at each evaluation.
        """
        model = model or self.default_model()
        if self.points is None:
            if getattr( self.sinogram, "visual_hull_points", None ) is not None:
                self.hull_points( nb_points_init )
            else:
                self.random_points( nb_points_init )

        stage = 0
        while True:
            optimizer = None if optimizer_factory is None else optimizer_factory( self.nb_points )
            self.run( model, optimizer = optimizer, label = f"{ model.name } x{ self.nb_points }", **kwargs )
            if stage_callback is not None:
                stage_callback( stage, self.nb_points, self.points )
            if self.nb_points >= nb_points_final:
                return self

            # `factor` may overshoot the target at the last stage: we then subsample, rather
            # than solving for a non-integer factor.
            next_n = min( self.nb_points * factor, nb_points_final )
            self.split( factor, noise_frac = noise_frac ).subsample( next_n )
            stage += 1

    # -- outputs -----------------------------------------------------------

    def _capture( self, raw ) -> None:
        # float32: the frames are only used for display, and a long trajectory with a large
        # number of points quickly becomes heavier than the reconstruction itself.
        self.frames.append( np.array( raw, dtype = np.float32, copy = True ) )

    def _summary_line( self, h: dict ) -> str:
        floor = f" (floor { h[ 'floor' ]:.8f})" if h[ "floor" ] > 0 else ""
        return ( f"[{ h[ 'label' ] }] { h[ 'nb_points' ] } points, { h[ 'optimizer' ] } : "
                 f"loss { h[ 'loss_before' ]:.6f} -> { h[ 'loss_after' ]:.8f}{ floor }"
                 f"  ({ h[ 'nb_steps' ] } steps, { h[ 'time' ]:.1f}s)" )

    def summary( self ) -> str:
        """The complete history of the steps, one line each."""
        return "\n".join( self._summary_line( h ) for h in self.history )

    def export_html( self, out_path: str, extent: float | None = None, animate: bool | None = None,
                     radii = ..., **kwargs ) -> "Reconstruction":
        """Writes the standalone HTML page of the cloud (see `viz.points_html.export_positions_html`).

        `animate`: replays the recorded trajectory (default: as soon as there are frames, hence as soon
        as `record` was active); `False` only exports the final cloud.

        `radii`: by default the radius of the LAST model played -- fixed and hence EXPORTED after a
        disks step (the disks are then drawn at their real size, the slider being just a scale
        factor), `None` after a diracs step (the slider IS the display radius).
        """
        if self.points is None:
            raise ValueError( "nothing to export -- no point" )
        animate = bool( self.frames ) if animate is None else animate
        if animate and not self.frames:
            raise ValueError( "no frame recorded -- build with `record = True`" )
        export_positions_html(
            self.frames if animate else self.positions,
            extent = float( extent if extent is not None else self.extent ),
            out_path = out_path,
            radii = self.radii if radii is ... else radii,
            **kwargs,
        )
        return self
