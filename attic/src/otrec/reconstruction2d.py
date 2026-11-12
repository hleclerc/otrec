"""2D reconstruction from a sinogram, in jax, on the device: `Reconstruction2d`.

The unknown is a cloud of N objects ( centers [ N, 2 ] ) whose projections, at every angle of the
sinogram, must reproduce the measured profiles. The loss is `sdot.projected1d.ProjectedLoss1d` ( a
sort-free 1D transport on a fine detector grid ); the optimizers are `device_optim` ( the whole loop
compiled, no round trip to the host per step ).

The objects ( `shape` ):
  "diracs"  points ( seen by the grid as tents one fine cell wide )
  "domes"   the radial density ( 1 - |x|^2/r^2 )^( k - 1/2 ) of radius `radius`, whose projection is the
            polynomial ( 1 - s^2/r^2 )^k: k = 1 is the closest to a disk
  "tents"   tents of half-width `radius` at every angle, by sdot's `ContinuousOtPlan1d`: the EXACT transport
            ( sorted breakpoints, no grid )
  "tents_grid"  the same tents, by its grid method ( `scale` cells per bin, no sort )

    rec = Reconstruction2d( sino, seed = 0 ).random_points( 1000 )
    rec.diracs( max_iter = 100 ).domes( radius = 0.05, max_iter = 200 )
    rec.positions, rec.history

Every step starts from the current cloud and replaces it; `history` keeps one line per step, `frames`
( with `record_every` ) snapshots of the trajectory, copied to the host every `record_every` steps only.
Each step first reorders the cloud along a Z curve ( `sdot.projected1d.morton_order` ): the ORDER of
`positions` is not the order of the starting points.

Importing this module enables float64 in jax ( `jax_enable_x64` ): the points stay float32 by default,
but the grid transport and the loss values need float64 ( see `ProjectedLoss1d.grid_dtype` ).
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

# float64 available ( the points stay float32 by default ): the grid transport of `ProjectedLoss1d` and the loss
# values the optimizers compare need it, see `ProjectedLoss1d.grid_dtype`
jax.config.update( "jax_enable_x64", True )

from sdot.projected1d import ProjectedLoss1d, morton_order

from . import device_optim


class Reconstruction2d:
    """`sinogram`: an `otrec.Sinogram` ( or anything with `angles`, `values` [ A, B ], `s_min`, `dw`,
    `extent` ) -- or a LIST of sinograms of the same geometry: K reconstructions solved together, the clouds
    then [ K, N, 2 ] ( one launch of each kernel for all of them: what fills a GPU when N is small ).
    `points`: the starting cloud ( optional ). `scale`: fine grid cells per detector bin. `dtype`: of the
    points ( float32 on a GPU ). `record_every`: keep a frame every that many steps ( 0: none ). `mesh`: a
    `jax.sharding.Mesh` whose devices share the ANGLES ( `ProjectedLoss1d.sharded` ); `backend`: the passes
    over the objects, "kernel" or "jax"."""

    def __init__( self, sinogram, points = None, *, scale = 8, k = 1, radius = None, dtype = jnp.float32,
                  max_iter = 200, ftol = 1e-10, min_iter = 0, disp_tol = None, memory = 10,
                  extent = None, seed = 0, record_every = 0, verbose = False, mesh = None, backend = "kernel" ):
        sinograms = list( sinogram ) if isinstance( sinogram, ( list, tuple ) ) else None
        self.batched = sinograms is not None
        if self.batched:
            sinogram = sinograms[ 0 ]
        self.sinogram = sinogram
        self.angles = np.asarray( sinogram.angles )
        self.values = np.stack( [ np.asarray( s.values ) for s in sinograms ] ) if self.batched else np.asarray( sinogram.values )
        self.K = len( sinograms ) if self.batched else None
        self.mesh, self.backend = mesh, backend
        self.s_min, self.dw = float( sinogram.s_min ), float( sinogram.dw )
        self.scale, self.k, self.radius, self.dtype = int( scale ), int( k ), radius, dtype
        self.max_iter, self.ftol, self.min_iter, self.disp_tol, self.memory = max_iter, ftol, min_iter, disp_tol, memory
        self.extent = float( extent if extent is not None else sinogram.extent )
        self.seed = int( seed )
        self.record_every = int( record_every )
        self.verbose = verbose
        self.points = None
        self.history = [ ]
        self.frames = [ ]
        self.radii = None           # the radius of the last step played ( None after diracs )
        self._losses = { }          # ( shape, radius ) -> ( ProjectedLoss1d, value_and_grad )
        if points is not None:
            self.set_points( points )

    # -- state ----------------------------------------------------------------

    @property
    def nb_points( self ):
        """the number of objects ( of each problem, for a batch )"""
        return 0 if self.points is None else int( self.points.shape[ -2 ] )

    @property
    def positions( self ):
        return np.asarray( self.points )

    def set_points( self, points ):
        if isinstance( points, Reconstruction2d ):
            points = points.points
        p = jnp.asarray( points, self.dtype )
        want = ( self.K, None, 2 ) if self.batched else ( None, 2 )
        if p.ndim != len( want ) or any( w is not None and w != d for w, d in zip( want, p.shape ) ):
            raise ValueError( f"points must have shape { tuple( '?' if w is None else w for w in want ) }, got { tuple( p.shape ) }" )
        self.points = p
        return self

    def _rng( self, seed ):
        if seed is None:
            seed, self.seed = self.seed, self.seed + 1
        return np.random.default_rng( seed )

    def random_points( self, nb_points, extent = None, seed = None ):
        e = float( extent if extent is not None else self.extent )
        lead = ( self.K, ) if self.batched else ( )
        return self.set_points( ( self._rng( seed ).random( ( *lead, nb_points, 2 ) ) - 0.5 ) * e )

    def split( self, factor = 4, noise_frac = 0.05, seed = None ):
        """each point becomes `factor` children, scattered by `noise_frac` of the current spacing"""
        noise = noise_frac * self.extent / np.sqrt( max( 1, self.nb_points ) )
        tiled = np.repeat( self.positions, factor, axis = -2 )
        return self.set_points( tiled + ( self._rng( seed ).random( tiled.shape ) - 0.5 ) * 2 * noise )

    def subsample( self, nb_points, seed = None ):
        if self.nb_points <= nb_points:
            return self
        idx = self._rng( seed ).choice( self.nb_points, nb_points, replace = False )
        return self.set_points( self.positions[ ..., idx, : ] )

    # -- the losses -------------------------------------------------------------

    def loss_function( self, shape = "diracs", radius = None ):
        """( loss( points ), value_and_grad( points ) ) of a model, built once and kept ( the compiled
        optimizer loop is cached on `value_and_grad` )"""
        if shape == "domes":
            radius = float( radius if radius is not None else self.radius )
        if shape in ( "tents", "tents_grid" ):
            radius = float( radius if radius is not None else 0.5 * self.dw )
        key = ( shape, radius )
        if key not in self._losses:
            opts = dict( scale = self.scale, dtype = self.dtype, backend = self.backend )
            if shape in ( "tents", "tents_grid" ):
                # tents of half-width `radius` against the measured profiles, by sdot's `ProjectedSumOfTents` +
                # `ContinuousOtPlan1d`: EXACT ( "tents": no grid, no kink when two tents cross ), or on a fine grid of
                # `scale` cells per bin ( "tents_grid": no sort )
                f = self._exact_tents_loss( radius, method = "grid" if shape == "tents_grid" else "exact" )
                self._losses[ key ] = ( jax.jit( f ), jax.jit( jax.value_and_grad( f ) ) )
                return self._losses[ key ]
            if shape == "diracs":
                L = ProjectedLoss1d( self.angles, self.values, self.s_min, self.dw, shape = "tent", **opts )
            elif shape == "domes":
                L = ProjectedLoss1d( self.angles, self.values, self.s_min, self.dw, shape = "dome", k = self.k,
                                     max_radius = radius, **opts )
            else:
                raise ValueError( f"unknown shape { shape!r} ( diracs, domes, tents, tents_grid )" )
            Lf = L.sharded( self.mesh ) if self.mesh is not None else L
            r = radius if shape == "domes" else L.max_halfwidth
            f = lambda p: Lf( p, jnp.full( p.shape[ :-1 ], r, self.dtype ) )
            self._losses[ key ] = ( jax.jit( f ), jax.jit( jax.value_and_grad( f ) ) )
        return self._losses[ key ]

    def _exact_tents_loss( self, half_width, method = "exact" ):
        if self.batched or self.mesh is not None:
            raise NotImplementedError( "exact tents: one reconstruction on one device for now" )
        from loom.tensor import Axis, ShapeVar
        from sdot import ContinuousOtPlan1d, Image, ProjectedSumOfTents
        normals = jnp.asarray( np.stack( [ np.cos( self.angles ), np.sin( self.angles ) ], 1 ) )
        values = jnp.asarray( self.values )

        def plan( p, with_gradient ):
            num_angle = Axis( ShapeVar( len( self.angles ) ), name = "num_angle" )
            src = ProjectedSumOfTents( points = p, half_widths = jnp.full( p.shape[ :-1 ], half_width, p.dtype ),
                                       normal = normals.astype( p.dtype ), batch_axes = [ num_angle ] )
            dst = Image( values = values, origin = [ self.s_min ], pixel_frame = [ [ self.dw ] ], batch_axes = [ num_angle ] )
            return ContinuousOtPlan1d( src, dst, with_gradient = with_gradient, method = method, scale = self.scale )

        # the gradient comes out of the forward ( `with_gradient` ): one transport per evaluation instead of two
        @jax.custom_vjp
        def f( p ):
            return jnp.sum( jnp.asarray( plan( p, False ).cost ) )

        def f_fwd( p ):
            pl = plan( p, True )
            return jnp.sum( jnp.asarray( pl.cost ) ), jnp.asarray( pl.points_grad )

        def f_bwd( gp, g ):
            return ( g * gp, )

        f.defvjp( f_fwd, f_bwd )
        return f

    def loss( self, shape = None, radius = None, points = None ):
        """the loss of the current cloud ( or of `points` ) for a model ( by default: domes if a radius was
        given, diracs otherwise )"""
        shape = shape or ( "domes" if ( radius or self.radius ) is not None else "diracs" )
        p = self.points if points is None else jnp.asarray( points, self.dtype )
        return float( self.loss_function( shape, radius )[ 0 ]( p ) )

    # -- the steps ----------------------------------------------------------------

    def run( self, shape, radius = None, optimizer = "lbfgs", lr = 0.1, max_iter = None, ftol = None,
             min_iter = None, disp_tol = ..., label = None ):
        if self.points is None:
            raise ValueError( "no starting point: `random_points` or `set_points` first" )
        _, vg = self.loss_function( shape, radius )
        # the objects in Z order ( the loss does not depend on it; the passes over the objects are many times
        # faster ): `positions` come back in that order
        if self.batched:
            self.points = jax.vmap( lambda p: p[ morton_order( p ) ] )( self.points )
        else:
            self.points = self.points[ morton_order( self.points ) ]
        opts = dict( max_iter = self.max_iter if max_iter is None else max_iter,
                     min_iter = self.min_iter if min_iter is None else min_iter,
                     disp_tol = self.disp_tol if disp_tol is ... else disp_tol )
        snap = None
        if self.record_every:
            if not self.frames:
                self.frames.append( self.positions )
            # the host copies start without blocking the next segment
            pending = [ ]
            def snap( k, x ):
                x.copy_to_host_async()
                pending.append( x )
        t0 = time.perf_counter()
        if optimizer == "lbfgs":
            x, info = device_optim.lbfgs( vg, self.points, ftol = self.ftol if ftol is None else ftol,
                                          memory = self.memory, snapshot_every = self.record_every or None,
                                          on_snapshot = snap, **opts )
        elif optimizer == "gd":
            x, info = device_optim.gradient_descent( vg, self.points, lr = lr, ftol = -1.0 if ftol is None else ftol,
                                                     snapshot_every = self.record_every or None,
                                                     on_snapshot = snap, **opts )
        else:
            raise ValueError( f"unknown optimizer { optimizer!r} ( lbfgs, gd )" )
        x.block_until_ready()
        dt = time.perf_counter() - t0
        if self.record_every:
            self.frames.extend( np.asarray( p ) for p in pending )
        self.points = x
        self.radii = radius if shape == "domes" else None
        losses = info[ "losses" ]
        h = dict( label = label or shape, model = shape, optimizer = optimizer, nb_points = self.nb_points,
                  radius = self.radii, loss_before = float( losses[ 0 ] ), loss_after = float( losses[ -1 ] ),
                  losses = losses, nb_steps = info[ "nb_steps" ], nb_evals = info[ "nb_evals" ],
                  reason = info[ "reason" ], time = dt )
        self.history.append( h )
        if self.verbose:
            print( f"  { h[ 'label' ] }: { self.nb_points } points, { h[ 'nb_steps' ] } steps "
                   f"( { h[ 'nb_evals' ] } evals, { h[ 'reason' ] } ), loss { h[ 'loss_before' ]:.4e} -> "
                   f"{ h[ 'loss_after' ]:.4e}, { dt:.2f} s" )
        return self

    def diracs( self, **kw ):
        return self.run( "diracs", **kw )

    def tents( self, radius = None, grid = False, **kw ):
        """tents of half-width `radius` ( half a detector bin by default ), exact or on the fine grid ( `grid` ): see
        `loss_function`"""
        return self.run( "tents_grid" if grid else "tents", radius = radius, **kw )

    def domes( self, radius = None, **kw ):
        r = radius if radius is not None else self.radius
        if r is None:
            raise ValueError( "domes need a radius: `Reconstruction2d( ..., radius = )` or `domes( radius = )`" )
        return self.run( "domes", radius = r, **kw )

    def multiscale( self, nb_points_final, nb_points_init = 50, factor = 4, shape = "diracs", radius = None, **kw ):
        """coarse to fine: `nb_points_init` random points, converged, split by `factor`, converged again...
        up to `nb_points_final` ( the last split subsampled )"""
        if self.points is None or self.nb_points != nb_points_init:
            self.random_points( nb_points_init )
        while True:
            self.run( shape, radius = radius, label = f"{ shape } { self.nb_points }", **kw )
            if self.nb_points >= nb_points_final:
                return self
            self.split( factor )
            self.subsample( nb_points_final )
