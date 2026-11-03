"""3D reconstruction from radiographs: a cloud of diracs whose projections, view by view, are transported onto the
measured images by an EXACT 2D semi-discrete transport ( sdot's Newton on power diagrams, one call per view ).

One evaluation solves the A views one after the other ( each warm started from its weights of the last evaluation ), and
accumulates per dirac, `e_k = b_k - P_k( x )` being the gap between the barycenter of its cell and its projection:

- `r = sum_k J_k^T e_k` and `H = sum_k J_k^T J_k`: the gradient of the cost is `-2 m r` ( the envelope theorem ), and
  the LIFTED LLOYD step is the Gauss-Newton step `dx = H^-1 r`, the point whose projections best match the barycenters
  ( exact in a parallel beam, where `P_k` is linear );
- `S = sum_k |e_k|^2`: `S - r . H^-1 r` is what is left of it at the lifted point, the DISAGREEMENT of the views about
  this dirac ( zero when one point of space explains all its barycenters );
- `sum_k ( w_k - mean( w_k ) )`, the dual weights centered per view: the derivative of the cost with respect to the MASS
  of the dirac ( up to a constant, the total mass being fixed ) -- positive, the dirac would rather shrink; negative,
  grow.

A 2D loss to finish ( `pixel2d` ): per view, the projected diracs as pyramids rasterized exactly on a fine grid against
the pixels as diracs ( `sdot.pixel_transport` ), scipy's L-BFGS on the sum. A SLICED loss, too ( `sliced`, the first stages of a multiscale ): each radiograph projected once more on `T` directions of
its plane, exactly ( `sdot.projected_profiles` ), against the diracs seen as TENTS along the same lines ( the 3D normals
`B_k^T theta`, sdot's `ProjectedSumOfTents` + `ContinuousOtPlan1d`, its grid method by default ): smooth in the
positions, no point where it freezes, the gradient in the same pass, the whole L-BFGS loop on the device.

Two ways to move: `lloyd` ( `x += tau dx`, one evaluation per step ) and `lbfgs` ( scipy on the host, the cost and its
gradient from the same evaluation ); `multiscale` runs Lloyd stage by stage, few diracs first, each stage splitting the
converged cloud ( a dirac far from the object is pulled in while it is cheap to move, instead of staying there ). The loop over the views is Python, each solve a jitted call ( shared by all the
views: same shapes ), the weights of each view kept on the device of that view; views go to the devices in turn
( `k mod nb_devices` ) and jax's asynchronous dispatch overlaps them.
"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from sdot.distributions.Image import Image
from sdot.distributions.SumOfDiracs import SumOfDiracs
from sdot.OtProblem import Iterative, OtProblem
from sdot.projected_profiles import image_profiles
from sdot import device_optim

jax.config.update( "jax_enable_x64", True )

#: `stats[ "status" ]` codes that mean converged ( see `sdot.SdotPlanNd._STATUS` )
_CONVERGED = ( 1, 7 )


def _make_solve( geometry, tol, max_iter, precision, continuation ):
    u_min, v_min, du, dv, _, _ = geometry.detector

    def solve( x, k, w0, values ):
        """view `k`: the cost, the dual weights, and per dirac `e` = its cell's barycenter minus its projection, lifted
        as `J^T e` ( `r` ), `J^T J` ( `H` ), `|e|^2` ( `S` )"""
        n = x.shape[ 0 ]
        uv = geometry.project( x, k )
        img = Image( values = values, origin = [ u_min, v_min ], pixel_frame = [ [ du, 0.0 ], [ 0.0, dv ] ] )
        plan = OtProblem( SumOfDiracs( uv ), img ).solve(
            Iterative( tol = tol / n, max_iter = max_iter, weights0 = w0, precision = precision,
                       continuation = continuation, on_failure = "ignore", differentiable = False ) )
        e = plan.barycenters.raw - uv
        J = geometry.jacobian( x, k )
        st = plan.stats
        return dict( cost = plan.cost.raw if hasattr( plan.cost, "raw" ) else plan.cost, weights = plan.weights.raw,
                     r = jnp.einsum( "nij,ni->nj", J, e ), H = jnp.einsum( "nij,nil->njl", J, J ), S = ( e * e ).sum( 1 ),
                     status = st[ "status" ], nb_iter = st[ "nb_iter" ], nb_diag = st[ "nb_diag" ], start = st[ "start" ],
                     nb_continuation_steps = st[ "nb_continuation_steps" ], min_start_mass = st[ "min_start_mass" ] )

    return jax.jit( solve )


class Reconstruction3d:
    """See the module docstring.

    `radiographs`: a `Radiographs` ( or anything with `.geometry` and `.with_background( b )` ). `points`: `[ n, 3 ]`,
    else `visual_hull_points` / `random_points`. `background`: added to every radiograph, as a fraction of its mean
    ( a cell that only sees zeros cannot get its mass ). `tol`: the mass tolerance of the solves, relative to the mass
    of a dirac. `devices`: the devices the views go to ( all the default backend's by default ).
    """

    def __init__( self, radiographs, points = None, *, background = 1e-6, tol = 1e-4, max_iter = 100, precision = "auto",
                  continuation = "auto", devices = None, verbose = False, freeze_z = False ):
        self.radiographs = radiographs
        self.geometry = radiographs.geometry
        self.background = float( background )
        self.devices = list( devices or jax.devices() )
        vals = radiographs.with_background( self.background )
        self._images = [ jax.device_put( jnp.asarray( vals[ k ] ), self._device( k ) ) for k in range( self.geometry.nb_views ) ]
        self.solver = dict( tol = tol, max_iter = max_iter, precision = precision, continuation = continuation )
        self._solve = _make_solve( self.geometry, **self.solver )
        self.verbose = verbose
        #: the diracs stay in their plane z = const ( a 2D object seen by one-line radiographs: the gradient and the
        #: lifted step lose their z component )
        self.freeze_z = bool( freeze_z )
        self.history = []
        self.frames = []
        self._sliced = { }
        self.points = None
        if points is not None:
            self.set_points( points )

    def configure( self, **solver ):
        """changes the solver's settings ( `tol`, `max_iter`, `precision`, `continuation` ), keeping the cloud and the
        warm starts -- say fp64 and a tight tolerance for the last steps"""
        self.solver.update( solver )
        self._solve = _make_solve( self.geometry, **self.solver )
        return self

    # -- the cloud ---------------------------------------------------------------------------------

    def _device( self, k ):
        return self.devices[ k % len( self.devices ) ]

    def set_points( self, points ):
        self.points = jnp.asarray( points, dtype = jnp.float64 )
        if self.points.ndim != 2 or self.points.shape[ 1 ] != 3:
            raise ValueError( "points must have shape [ n, 3 ]" )
        # the warm start of each view, on its device ( zeros: the solver then picks its own start )
        n = self.points.shape[ 0 ]
        self._weights = [ jax.device_put( jnp.zeros( n ), self._device( k ) ) for k in range( self.geometry.nb_views ) ]
        return self

    def split( self, factor = 4, noise_frac = 0.25, seed = None ):
        """each dirac becomes `factor` children, scattered by `noise_frac` of the mean distance to the nearest neighbour"""
        from scipy.spatial import cKDTree
        x = self.positions
        spacing = float( cKDTree( x ).query( x, k = 2 )[ 0 ][ :, 1 ].mean() ) if len( x ) > 1 else 1.0
        rng = np.random.default_rng( seed if seed is not None else len( self.history ) )
        tiled = np.repeat( x, factor, axis = 0 )
        return self.set_points( tiled + noise_frac * spacing * rng.normal( size = tiled.shape ) )

    def subsample( self, nb_points, seed = None ):
        if self.nb_points <= nb_points:
            return self
        rng = np.random.default_rng( seed if seed is not None else len( self.history ) )
        return self.set_points( self.positions[ rng.choice( self.nb_points, nb_points, replace = False ) ] )

    def visual_hull_points( self, n, seed = 0, **kw ):
        return self.set_points( self.radiographs.visual_hull_points( n, seed = seed, **kw ) )

    def random_points( self, n, seed = 0, extent = None ):
        e = float( self.radiographs.extent if extent is None else extent )
        return self.set_points( ( np.random.default_rng( seed ).random( ( n, 3 ) ) - 0.5 ) * e )

    @property
    def nb_points( self ):
        return int( self.points.shape[ 0 ] )

    @property
    def positions( self ):
        return np.asarray( self.points )

    # -- one evaluation ------------------------------------------------------------------------------

    def evaluate( self, points = None, keep_weights = True ):
        """All the views at `points` ( the current cloud by default ): a dict of host arrays -- `cost`, `cost_per_view`,
        `grad` ( `[ n, 3 ]` ), `step` ( the lifted Lloyd step `dx` ), `disagreement` and `mass_grad` ( `[ n ]` ),
        `status`, `nb_iter`, `nb_diag` ( `[ A ]` ), `time`."""
        x = self.points if points is None else jnp.asarray( points, dtype = jnp.float64 )
        A, n = self.geometry.nb_views, x.shape[ 0 ]
        t0 = time.perf_counter()
        outs = []
        for k in range( A ):   # dispatched without waiting: the views of different devices overlap
            dev = self._device( k )
            outs.append( self._solve( jax.device_put( x, dev ), jax.device_put( jnp.int32( k ), dev ), self._weights[ k ], self._images[ k ] ) )
        if keep_weights:
            self._weights = [ o[ "weights" ] for o in outs ]
        home = self.devices[ 0 ]
        def total( name ):
            return sum( jax.device_put( o[ name ], home ) for o in outs )
        r, H, S = total( "r" ), total( "H" ), total( "S" )
        wc = sum( jax.device_put( o[ "weights" ] - o[ "weights" ].mean(), home ) for o in outs )
        dx = jnp.linalg.solve( H, r[ :, :, None ] )[ :, :, 0 ]
        if self.freeze_z:
            dx, r = dx.at[ :, 2 ].set( 0.0 ), r.at[ :, 2 ].set( 0.0 )
        res = dict(
            cost_per_view = np.array( [ float( o[ "cost" ] ) for o in outs ] ),
            grad = np.asarray( -2.0 / n * r ),
            step = np.asarray( dx ),
            H = np.asarray( H ),
            disagreement = np.asarray( S - ( r * dx ).sum( 1 ) ),
            mass_grad = np.asarray( wc ),
            status = np.array( [ int( o[ "status" ] ) for o in outs ] ),
            nb_iter = np.array( [ int( o[ "nb_iter" ] ) for o in outs ] ),
            nb_diag = np.array( [ int( o[ "nb_diag" ] ) for o in outs ] ),
        )
        res[ "cost" ] = float( res[ "cost_per_view" ].sum() )
        res[ "nb_failed" ] = int( np.sum( ~np.isin( res[ "status" ], _CONVERGED ) ) )
        res[ "time" ] = time.perf_counter() - t0
        return res

    def loss( self ):
        return self.evaluate()[ "cost" ]

    # -- the optimizers --------------------------------------------------------------------------------

    def _record( self, step, ev, x_new, truth ):
        x_old = np.asarray( self.points )
        disp = np.linalg.norm( x_new - x_old, axis = 1 )
        dz = np.abs( x_new[ :, 2 ] - x_old[ :, 2 ] )
        rec = dict( step = step, nb_points = self.nb_points, cost = ev[ "cost" ], cost_per_view = ev[ "cost_per_view" ], time = ev[ "time" ],
                    nb_iter = ev[ "nb_iter" ].mean(), nb_diag = ev[ "nb_diag" ].mean(), nb_failed = ev[ "nb_failed" ],
                    disp_mean = float( disp.mean() ), disp_max = float( disp.max() ),
                    # along the rotation axis: what a cut into independent slices ( 1D transports ) would have to neglect
                    disp_z_mean = float( dz.mean() ), disp_z_max = float( dz.max() ),
                    disagreement_mean = float( ev[ "disagreement" ].mean() ), disagreement_max = float( ev[ "disagreement" ].max() ),
                    mass_grad_std = float( ev[ "mass_grad" ].std() ) )
        if truth is not None:
            rec.update( truth( x_old ) )
        self.history.append( rec )
        if self.verbose:
            print( f"  { step:4d}  n { self.nb_points:7d}  cost { rec[ 'cost' ]:.6e}  disp { rec[ 'disp_mean' ]:.2e}/{ rec[ 'disp_max' ]:.2e}  "
                   f"dz { rec[ 'disp_z_mean' ]:.1e}/{ rec[ 'disp_z_max' ]:.1e}  disagr { rec[ 'disagreement_mean' ]:.2e}  newton { rec[ 'nb_iter' ]:.1f} it { rec[ 'nb_diag' ]:.1f} diag  "
                   f"failed { rec[ 'nb_failed' ] }  { rec[ 'time' ]:.2f} s", flush = True )
        return rec

    def _frame( self, step, ev ):
        self.frames.append( dict( step = step, points = np.asarray( self.points ), disagreement = ev[ "disagreement" ],
                                  mass_grad = ev[ "mass_grad" ], step_vec = ev[ "step" ] ) )

    def lloyd( self, nb_steps = 50, tau = 1.0, disp_tol = 0.0, snapshot_every = 0, truth = None, on_snapshot = None ):
        """`nb_steps` lifted Lloyd steps `x += tau H^-1 r`; stops early when the mean displacement falls under `disp_tol`.
        `truth( points ) -> dict` adds measures against a known object to each history entry ( see `phantoms` );
        `on_snapshot( self )` is called at each snapshot ( a checkpoint, say )."""
        base = len( self.history )   # steps are numbered across the stages
        for it in range( nb_steps ):
            ev = self.evaluate()
            if snapshot_every and it % snapshot_every == 0:
                self._frame( base + it, ev )
                if on_snapshot is not None:
                    on_snapshot( self )
            x_new = np.asarray( self.points ) + tau * ev[ "step" ]
            rec = self._record( base + it, ev, x_new, truth )
            self.set_points_keep( x_new )
            if rec[ "disp_mean" ] < disp_tol:
                break
        ev = self.evaluate()
        self._frame( len( self.history ), ev )
        self._final = ev
        return self

    def multiscale( self, nb_points_final, nb_points_init = 2000, factor = 4, steps = 30, last_steps = None, tau = 1.0,
                    start = "hull", snapshot_every = 0, truth = None, on_snapshot = None, seed = 0, optimizer = "lloyd", sliced = None,
                    pixel = None, **hull ):
        """coarse to fine: `nb_points_init` diracs ( in the visual hull, or `start = "uniform"` in the cube; the current
        cloud when it already has that size ), `steps` Lloyd steps, split by `factor`, `steps` again... up to
        `nb_points_final` ( the last split subsampled ), whose stage takes `last_steps` ( `steps` by default ). `steps`
        may also be a list, the steps of each stage in turn ( its last entry for the stages beyond ). `optimizer`: what each
        stage runs, `lloyd` ( at `tau` ), `plbfgs` ( L-BFGS preconditioned by the Lloyd step ), `sliced` ( the sliced loss,
        `sliced` the options of `sliced` ) or `pixel2d` ( the 2D loss, `pixel` the options of `pixel2d` ) -- or a list,
        the optimizer of each stage in turn ( its last entry for the stages beyond )"""
        if self.points is None or self.nb_points != nb_points_init:
            if start == "hull":
                self.visual_hull_points( nb_points_init, seed = seed, **hull )
            else:
                self.random_points( nb_points_init, seed = seed )
        stage = 0
        while True:
            last = self.nb_points >= nb_points_final
            if isinstance( steps, ( list, tuple ) ):
                nb = steps[ min( stage, len( steps ) - 1 ) ]
            else:
                nb = ( last_steps or steps ) if last else steps
            opt = optimizer[ min( stage, len( optimizer ) - 1 ) ] if isinstance( optimizer, ( list, tuple ) ) else optimizer
            if opt == "plbfgs":
                self.plbfgs( nb, snapshot_every = snapshot_every, truth = truth, on_snapshot = on_snapshot )
            elif opt == "sliced":
                self.sliced( nb, snapshot_every = snapshot_every, truth = truth, on_snapshot = on_snapshot, **( sliced or { } ) )
            elif opt == "pixel2d":
                self.pixel2d( nb, snapshot_every = max( 1, snapshot_every // 10 ), truth = truth, on_snapshot = on_snapshot, **( pixel or { } ) )
            else:
                self.lloyd( nb, tau = tau, snapshot_every = snapshot_every, truth = truth, on_snapshot = on_snapshot )
            stage += 1
            if on_snapshot is not None:
                on_snapshot( self )
            if last:
                return self
            self.split( factor )
            self.subsample( nb_points_final )

    def plbfgs( self, nb_steps = 50, memory = 10, c1 = 1e-4, max_ls = 8, snapshot_every = 0, truth = None,
                on_snapshot = None ):
        """L-BFGS PRECONDITIONED by the lifted Lloyd step: the initial inverse Hessian is, per dirac, M = ( n / 2 ) H^-1, so
        that its plain step is the Lloyd step ( tau = 1 ); the two-loop recursion adds the curvature Lloyd ignores ( how
        the barycenters move with the neighbours ), scaled by gamma = s.y / y.M y. Armijo backtracking from alpha = 1,
        one evaluation per trial ( the warm starts follow the trials ); when it fails ( `max_ls` trials ), a plain Lloyd
        step and the memory dropped ( `alpha` -1 in the history ). A history entry per step ( `nb_evals` counts the
        evaluations )."""
        n = self.nb_points
        base = len( self.history )
        x = self.positions
        ev = self.evaluate()
        S, Y = [ ], [ ]
        nb_evals = 1

        def apply_M( v, H ):
            return 0.5 * n * np.linalg.solve( H, v[ :, :, None ] )[ :, :, 0 ]

        for it in range( nb_steps ):
            g, f, H = ev[ "grad" ], ev[ "cost" ], ev[ "H" ]
            if snapshot_every and it % snapshot_every == 0:
                self._frame( base + it, ev )
                if on_snapshot is not None:
                    on_snapshot( self )
            # the two-loop recursion, with M as the initial inverse Hessian
            q = g.copy()
            al = [ ]
            for s_, y_ in reversed( list( zip( S, Y ) ) ):
                a = ( s_ * q ).sum() / ( y_ * s_ ).sum()
                q -= a * y_
                al.append( a )
            gamma = 1.0
            if S:
                My = apply_M( Y[ -1 ], H )
                gamma = ( S[ -1 ] * Y[ -1 ] ).sum() / ( Y[ -1 ] * My ).sum()
            r = gamma * apply_M( q, H )
            for ( s_, y_ ), a in zip( zip( S, Y ), reversed( al ) ):
                b = ( y_ * r ).sum() / ( y_ * s_ ).sum()
                r += s_ * ( a - b )
            d = -r
            gd = ( g * d ).sum()
            if gd >= 0:   # not a descent direction: back to the Lloyd step, memory dropped
                S, Y = [ ], [ ]
                d = -apply_M( g, H )
                gd = ( g * d ).sum()
            alpha, ok = 1.0, False
            for k in range( max_ls ):
                ev_t = self.evaluate( x + alpha * d )
                nb_evals += 1
                if ev_t[ "cost" ] <= f + c1 * alpha * gd:
                    ok = True
                    break
                alpha *= 0.5
            if not ok:
                # the line search failed ( a cost too rough for it, or a kink ): a plain Lloyd step, memory dropped
                S, Y = [ ], [ ]
                d, alpha = -apply_M( g, H ), 1.0
                ev_t = self.evaluate( x + d )
                nb_evals += 1
            x_new = x + alpha * d
            rec = self._record( base + it, ev, x_new, truth )
            rec.update( nb_evals = nb_evals, alpha = alpha if ok else -1.0, gamma = gamma )
            s_, y_ = x_new - x, ev_t[ "grad" ] - g
            if ( s_ * y_ ).sum() > 1e-12 * np.sqrt( ( s_ * s_ ).sum() * ( y_ * y_ ).sum() ):
                S.append( s_ ); Y.append( y_ )
                S, Y = S[ -memory: ], Y[ -memory: ]
            x, ev = x_new, ev_t
            self.set_points_keep( x )
        self._frame( len( self.history ), ev )
        self._final = ev
        return self

    # -- the sliced loss --------------------------------------------------------------------------

    def sliced_loss( self, nb_directions = 16, radius = None, method = "grid", scale = 8, dtype = jnp.float32 ):
        """`( loss( points ), value_and_grad( points ) )` of the SLICED loss ( see the module ), built once per setting:
        `nb_directions` per view ( angles k pi / T, shifted from a view to the next by the golden ratio ), tents of
        half-width `radius` ( half a pixel by default ), `method` / `scale` those of `ContinuousOtPlan1d`, the tents in
        `dtype` ( float32 by default: the card's fast path ), the gradient of the summed cost from the same pass"""
        g = self.geometry
        u_min, v_min, du, dv, nb_u, nb_v = g.detector
        ds = min( du, dv )
        radius = float( 0.5 * ds if radius is None else radius )
        key = ( int( nb_directions ), radius, method, int( scale ), jnp.dtype( dtype ).name )
        if key in self._sliced:
            return self._sliced[ key ]
        from loom.tensor import Axis, ShapeVar
        from sdot import ContinuousOtPlan1d, Image, ProjectedSumOfTents
        A, T = g.nb_views, int( nb_directions )
        phase = ( np.arange( A ) * 0.6180339887498949 ) % 1.0
        ang = np.pi * ( np.arange( T )[ None, : ] + phase[ :, None ] ) / T
        theta = np.stack( [ np.cos( ang ), np.sin( ang ) ], -1 )                                    # [ A, T, 2 ]
        bases, offsets = np.asarray( g.bases ), np.asarray( g.offsets )
        normals = np.einsum( "atk,akd->atd", theta, bases ).reshape( A * T, 3 )
        # common bins in the tents' coordinate s = n . x = theta . ( uv - offset ): they cover every projected detector
        corners = np.array( [ [ u_min, v_min ], [ u_min + nb_u * du, v_min ], [ u_min, v_min + nb_v * dv ], [ u_min + nb_u * du, v_min + nb_v * dv ] ] )
        proj = np.einsum( "ck,atk->atc", corners, theta ) - np.einsum( "ak,atk->at", offsets, theta )[ ..., None ]
        s0, s1 = proj.min() - ds, proj.max() + ds
        nb = int( np.ceil( ( s1 - s0 ) / ds ) )
        smin_uv = s0 + np.einsum( "ak,atk->at", offsets, theta )                                    # the same bins in uv
        vals = np.asarray( self.radiographs.with_background( self.background ), dtype = np.float64 )
        prof = image_profiles( vals, [ u_min, v_min ], [ [ du, 0.0 ], [ 0.0, dv ] ], theta, smin_uv, ds, nb )
        values = jnp.asarray( prof, dtype ).reshape( A * T, nb ) / ds
        normals = jnp.asarray( normals, dtype )

        def plan( p, with_gradient ):
            num = Axis( ShapeVar( A * T ), name = "num_view_direction" )
            p = p.astype( dtype )
            src = ProjectedSumOfTents( points = p, half_widths = jnp.full( p.shape[ :-1 ], radius, dtype ), normal = normals,
                                       proj_dims = 3, batch_axes = [ num ] )
            dst = Image( values = values, origin = [ s0 ], pixel_frame = [ [ ds ] ], batch_axes = [ num ] )
            return ContinuousOtPlan1d( src, dst, with_gradient = with_gradient, method = method, scale = scale )

        @jax.custom_vjp
        def f( p ):
            return jnp.sum( jnp.asarray( plan( p, False ).cost ).astype( jnp.float64 ) )

        def f_fwd( p ):
            pl = plan( p, True )
            return jnp.sum( jnp.asarray( pl.cost ).astype( jnp.float64 ) ), jnp.asarray( pl.points_grad ).astype( p.dtype )

        def f_bwd( gp, gv ):
            return ( gv.astype( gp.dtype ) * gp, )

        f.defvjp( f_fwd, f_bwd )
        self._sliced[ key ] = ( jax.jit( f ), jax.jit( jax.value_and_grad( f ) ) )
        return self._sliced[ key ]

    def sliced( self, nb_steps = 100, nb_directions = 16, radius = None, method = "grid", scale = 8, dtype = jnp.float32,
                memory = 10, ftol = 1e-7, max_ls = 10, snapshot_every = 0, truth = None, on_snapshot = None ):
        """at most `nb_steps` of L-BFGS on the SLICED loss, on the device ( `sdot.device_optim` ): it stops when the relative
        decrease falls under `ftol` or when the line search fails ( a float32 loss is noisy at the 1e-3 level near
        convergence: past that, the steps cost `max_ls` evaluations each and move nothing ). One history entry per segment
        of `snapshot_every` steps ( the whole run without ), `cost` the sliced loss, `cost_per_view` its value spread evenly
        ( the per-view split would cost a pass ), and the measures of `truth`"""
        _, vg = self.sliced_loss( nb_directions, radius, method, scale, dtype )
        base = len( self.history )
        A = self.geometry.nb_views
        t_last = [ time.perf_counter(), 0 ]

        def record( k, x ):
            self._record_light( base + k, np.asarray( x ), float( vg( jnp.asarray( x ) )[ 0 ] ), "sliced", t_last, truth, on_snapshot )

        x, info = device_optim.lbfgs( vg, self.points, max_iter = nb_steps, ftol = ftol, f_scale = 0.0, memory = memory, max_ls = max_ls,
                                      snapshot_every = snapshot_every or nb_steps, on_snapshot = record )
        if self.verbose:
            print( f"  sliced, { self.nb_points } diracs: { info[ 'nb_steps' ] } steps, { info[ 'nb_evals' ] } evaluations ( { info[ 'reason' ] } )", flush = True )
        if not self.history or self.history[ -1 ][ "step" ] != base + int( info[ "nb_steps" ] ):
            record( int( info[ "nb_steps" ] ), x )
        self.set_points_keep( np.asarray( x ) )
        return self

    def _record_light( self, step, x, cost, model, t_last, truth, on_snapshot ):
        """a history entry and a frame for the optimizers that only know the cost ( `sliced`, `pixel2d` ): the same
        keys as `_record`, the per-view and Newton entries filled in ( `cost_per_view` the cost spread evenly )"""
        A = self.geometry.nb_views
        x_old = np.asarray( self.points )
        disp = np.linalg.norm( x - x_old, axis = 1 )
        dz = np.abs( x[ :, 2 ] - x_old[ :, 2 ] )
        now = time.perf_counter()
        rec = dict( step = step, nb_points = self.nb_points, model = model, cost = cost, cost_per_view = np.full( A, cost / A ),
                    time = now - t_last[ 0 ], nb_iter = 0.0, nb_diag = 0.0, nb_failed = 0, disp_mean = float( disp.mean() ),
                    disp_max = float( disp.max() ), disp_z_mean = float( dz.mean() ), disp_z_max = float( dz.max() ),
                    disagreement_mean = 0.0, disagreement_max = 0.0, mass_grad_std = 0.0 )
        if truth is not None:
            rec.update( truth( x ) )
        self.history.append( rec )
        t_last[ 0 ] = now
        self.points = jnp.asarray( x, jnp.float64 )
        zeros = np.zeros( len( x ) )
        self.frames.append( dict( step = step, points = x, disagreement = zeros, mass_grad = zeros, step_vec = np.zeros_like( x ) ) )
        if self.verbose:
            extra = "  ".join( f"{ m } { rec[ m ]:.4g}" for m in ( "inside", "coverage" ) if m in rec )
            print( f"  { step:5d}  n { self.nb_points:7d}  { model } { cost:.6e}  disp { rec[ 'disp_mean' ]:.2e}  { extra }  "
                   f"{ rec[ 'time' ]:.2f} s", flush = True )
        if on_snapshot is not None:
            on_snapshot( self )
        return rec

    # -- the 2D loss: pyramids on a fine grid against the pixels as diracs ---------------------------

    def pixel_loss( self, scale = 4, radius = None, background = 1e-3, tol = 1e-3, max_iter = 200, cards = None, split = 1 ):
        """per view a `sdot.pixel_transport.PixelTransport2d`: the projected diracs as PYRAMIDS ( products of two tents of
        half-width `radius`, half a pixel by default ) rasterized exactly on a grid of `scale` x `scale` cells per pixel,
        against the pixels of the radiograph as diracs; `background` on both sides; `tol` the mass tolerance of the
        solves relative to a pixel's mean mass; `split`: `split` x `split` diracs per pixel ( one per pixel pulls the
        points to the pixels' centers: along the rotation axis, their z snap to the detector rows ). Built once per setting
        ( the dual weights of each view live in it ).
        `cards`: the views spread over that many cards, a worker process each ( `pixel_worker` ), by default every card
        of this process when there are several ( it should not preallocate them: `XLA_PYTHON_CLIENT_PREALLOCATE=false` );
        on the host, that many worker processes sharing its cores"""
        g = self.geometry
        u_min, v_min, du, dv, nb_u, nb_v = g.detector
        radius = float( 0.5 * min( du, dv ) if radius is None else radius )
        if cards is None:
            cards = sum( d.platform == "gpu" for d in self.devices )
        cards = int( cards ) if cards and cards > 1 else 1
        # the radius is not in the key: a view takes it at each evaluation, so a blur schedule ( `pixel2d` at decreasing radii )
        # keeps the views and warm starts each radius from the weights of the previous one
        key = ( int( scale ), float( background ), float( tol ), cards, int( split ) )
        if getattr( self, "_pixel", None ) is not None and self._pixel[ 0 ] == key:
            self._pixel = ( key, radius, self._pixel[ 2 ] )
        else:
            self.close_pixel_workers()
            vals = self.radiographs.with_background( 0.0 )
            setting = dict( scale = scale, background = background, tol = tol / ( nb_u * nb_v * split * split ), max_iter = max_iter,
                            precision = "fp64", dirac_split = int( split ) )
            if cards > 1:
                from .pixel_worker import PixelWorkers
                on_host = all( d.platform == "cpu" for d in self.devices )
                threads = max( 1, len( os.sched_getaffinity( 0 ) ) // cards ) if on_host and hasattr( os, "sched_getaffinity" ) else None
                views = PixelWorkers( cards, vals, np.asarray( g.bases ), np.asarray( g.offsets ), ( u_min, v_min, du, dv ), setting,
                                      threads = threads )
                if self.verbose:
                    print( f"  pixel2d: { g.nb_views } views over { cards } cards ( { ', '.join( views.cards ) } )", flush = True )
            else:
                from sdot.pixel_transport import PixelTransport2d
                views = [ PixelTransport2d( vals[ k ], [ u_min, v_min ], ( du, dv ), **setting ) for k in range( g.nb_views ) ]
            self._pixel = ( key, radius, views )
        return self._pixel

    def close_pixel_workers( self ):
        """stops the worker processes of the 2D loss, if any"""
        if getattr( self, "_pixel", None ) is not None and hasattr( self._pixel[ 2 ], "close" ):
            self._pixel[ 2 ].close()
        self._pixel = None

    def pixel_value_and_grad( self, points = None, **setting ):
        """the 2D loss summed over the views, and its gradient with respect to the points ( `pixel_loss` for `setting` )"""
        _, radius, views = self.pixel_loss( **setting )
        x = np.asarray( self.points if points is None else points, dtype = np.float64 )
        if hasattr( views, "value_and_grad" ):
            return views.value_and_grad( x, radius )
        total, grad = 0.0, np.zeros( x.shape )
        xj = jnp.asarray( x )
        for k, pt in enumerate( views ):
            c, ( g_uv, _ ) = pt.value_and_grad( self.geometry.project( xj, k ), radius )
            total += c
            # d uv / d x = J_k ( [ n, 2, 3 ] ): the gradient goes back through it
            grad += np.einsum( "ni,nij->nj", np.asarray( g_uv ), np.asarray( self.geometry.jacobian( xj, k ) ) )
        return total, grad

    def pixel2d( self, nb_steps = 20, scale = 4, radius = None, background = 1e-3, tol = 1e-3, ftol = 0.0, cards = None, split = 1,
                 snapshot_every = 1, truth = None, on_snapshot = None, first_move = 0.05 ):
        """at most `nb_steps` of scipy's L-BFGS on the 2D loss ( `pixel_loss` ), one evaluation per trial ( all the views,
        each warm started ); a history entry and a frame every `snapshot_every` accepted steps. `ftol`: scipy's test on the
        decrease relative to max( |f|, 1 ) -- an ABSOLUTE test for this loss ( ~1e-3 ): 0, the default, runs the steps.
        `first_move`: the largest coordinate move of the first trial, in mean nearest-neighbour spacings"""
        from scipy.optimize import minimize
        from scipy.spatial import cKDTree
        setting = dict( scale = scale, radius = radius, background = background, tol = tol, cards = cards, split = split )
        shape = self.points.shape
        base = len( self.history )
        nb_evals = [ 0 ]
        cache = { }
        t_last = [ time.perf_counter() ]

        # scipy's first trial is a step of norm 1 along -g: on the raw coordinates it can throw diracs across several cells
        # ( cost x3.5, a third of the warm solves failed and redone cold, measured at split 4 ). The optimizer works on
        # x / unit, with unit chosen so that this first trial moves no coordinate by more than `first_move` spacings
        x0 = np.asarray( self.points )
        spacing = float( cKDTree( x0 ).query( x0, k = 2 )[ 0 ][ :, 1 ].mean() ) if len( x0 ) > 1 else 1.0
        c0, g0 = self.pixel_value_and_grad( x0, **setting )
        g0 = np.asarray( g0 ).ravel()
        unit = first_move * spacing * np.linalg.norm( g0 ) / max( np.abs( g0 ).max(), 1e-300 )
        first = [ ( c0, g0 ) ]

        def fun( flat ):
            t0 = time.perf_counter()
            if first:
                c, g = first.pop()
            else:
                c, g = self.pixel_value_and_grad( flat.reshape( shape ) * unit, **setting )
            nb_evals[ 0 ] += 1
            cache[ "c" ] = c
            if self.verbose:
                views = self._pixel[ 2 ]
                retries = getattr( views, "nb_cold_retries", None )
                if retries is None:
                    retries = sum( pt.nb_cold_retries for pt in views )
                print( f"    eval { nb_evals[ 0 ]:4d}  { c:.6e}  |g| { np.linalg.norm( g ):.3e}  { time.perf_counter() - t0:.1f} s  "
                       f"cold retries { retries }", flush = True )
            return c, unit * np.asarray( g ).ravel()

        step = [ 0 ]

        def callback( xk ):
            step[ 0 ] += 1
            if step[ 0 ] % max( 1, snapshot_every ) == 0:
                rec = self._record_light( base + step[ 0 ], unit * xk.reshape( shape ), cache[ "c" ], "pixel2d", t_last, truth, on_snapshot )
                rec[ "nb_evals" ] = nb_evals[ 0 ]

        res = minimize( fun, x0.ravel() / unit, jac = True, method = "L-BFGS-B", callback = callback,
                        options = dict( maxiter = nb_steps, ftol = ftol, gtol = 0.0, maxcor = 10 ) )
        if not self.history or self.history[ -1 ][ "step" ] != base + step[ 0 ]:
            self._record_light( base + step[ 0 ], unit * res.x.reshape( shape ), float( res.fun ), "pixel2d", t_last, truth, on_snapshot )
        if self.verbose:
            print( f"  pixel2d, { self.nb_points } diracs: { step[ 0 ] } steps, { nb_evals[ 0 ] } evaluations ( { res.message } )", flush = True )
        self.set_points_keep( unit * res.x.reshape( shape ) )
        return self

    def lbfgs( self, nb_steps = 50, snapshot_every = 0, truth = None, ftol = 1e-12 ):
        """scipy's L-BFGS on the summed cost ( one evaluation per trial of its line search ); a history entry per
        ACCEPTED step ( `nb_evals` counts the trials )."""
        from scipy.optimize import minimize
        shape = self.points.shape
        cache = {}
        nb_evals = [ 0 ]

        def fun( flat ):
            ev = self.evaluate( flat.reshape( shape ) )
            nb_evals[ 0 ] += 1
            cache[ "ev" ], cache[ "x" ] = ev, flat.reshape( shape ).copy()
            return ev[ "cost" ], ev[ "grad" ].ravel()

        step = [ 0 ]

        def callback( xk ):
            ev = cache[ "ev" ]
            if snapshot_every and step[ 0 ] % snapshot_every == 0:
                self._frame( step[ 0 ], ev )
            rec = self._record( step[ 0 ], ev, xk.reshape( shape ), truth )
            rec[ "nb_evals" ] = nb_evals[ 0 ]
            self.set_points_keep( xk.reshape( shape ) )
            step[ 0 ] += 1

        res = minimize( fun, np.asarray( self.points ).ravel(), jac = True, method = "L-BFGS-B", callback = callback,
                        options = dict( maxiter = nb_steps, ftol = ftol, gtol = 0.0, maxcor = 10 ) )
        self.set_points_keep( res.x.reshape( shape ) )
        ev = self.evaluate()
        self._frame( len( self.history ), ev )
        self._final = ev
        return self

    def set_points_keep( self, points ):
        """new positions for the same diracs: the warm starts are kept"""
        self.points = jnp.asarray( points, dtype = jnp.float64 )
