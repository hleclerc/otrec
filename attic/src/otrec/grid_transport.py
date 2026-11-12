"""The 2D loss of a view on the dedicated grid solver of sdot: the measured pixels are the diracs of a grid
( `sdot.SumOfDiracsOnAGrid`, one per pixel, at its centre ), the projected cloud a P1 image ( `sdot.Image( degree = 1 )`,
its nodes at the corners of the pixels ), so that the solve takes `Tuning( grid_solver = "dedicated" )`.

Each point deposits the projection of a radial 3D kernel of radius `R`, sampled at the nodes, the same in every view:

    "smooth"  k( r ) = ( 1 - r^2 / R^2 )^2    ->   p( d ) = 7 / ( 2 pi R^7 ) ( R^2 - d^2 )^( 5 / 2 )
    "ball"    a uniform ball                  ->   p( d ) = 3 / ( 2 pi R^3 ) ( R^2 - d^2 )^( 1 / 2 )

The ball's profile has an infinite slope at its rim: the loss is rough at the scale of a node crossing it ( a central
difference at 1.6e-4 pixel gave half the derivative, 4 views of 64^2 ), where the smooth kernel's vanishes with its slope. Both sides are blurred by a gaussian of width `sigma` ( the GLOBAL blur, decreasing from one stage of
the optimization to the next ) and lifted by a constant, `background` times their mean, which keeps the densities positive
on a connected support ( the damped Newton needs it ). The derivative of the cost with respect to the nodal values is the
Kantorovich potential against each basis function ( `sdot.PowerDiagram.potential_integrals`, envelope theorem ), through the
normalization of the image; JAX takes it back to the positions."""
import collections
import functools
import math
import time

import jax
import jax.numpy as jnp
import numpy as np


def gaussian_matrix( n, sigma ):
    """`[ n, n ]`: the gaussian blur of width `sigma` ( in samples ) of `n` regularly spaced samples, mass conserving away
    from the ends ( the identity for `sigma = 0` )"""
    if sigma <= 0:
        return np.eye( n )
    i = np.arange( n )
    g = np.exp( -0.5 * ( ( i[ :, None ] - i[ None, : ] ) / sigma ) ** 2 )
    return g / ( math.sqrt( 2 * math.pi ) * sigma )


def corner_coeffs( v ):
    """the Bernstein coefficients of the pixels of a P1 image, `[ N, N, 4 ]`, from its nodal values `[ N + 1, N + 1 ]`"""
    return jnp.stack( [ v[ :-1, :-1 ], v[ :-1, 1: ], v[ 1:, :-1 ], v[ 1:, 1: ] ], -1 )


def nodes_of_coeffs( c ):
    """the adjoint of `corner_coeffs`: `[ N, N, 4 ]` -> `[ N + 1, N + 1 ]`"""
    N = c.shape[ 0 ]
    v = jnp.zeros( ( N + 1, N + 1 ), c.dtype )
    v = v.at[ :-1, :-1 ].add( c[ ..., 0 ] ).at[ :-1, 1: ].add( c[ ..., 1 ] )
    return v.at[ 1:, :-1 ].add( c[ ..., 2 ] ).at[ 1:, 1: ].add( c[ ..., 3 ] )


def _deposit( uv, masses, N, h, origin, radius, sigma, background, profile ):
    """the nodal values `[ N + 1, N + 1 ]` ( `GridTransport2d.nodes` )"""
    n = uv.shape[ 0 ]
    m = jnp.full( n, 1.0 / n ) if masses is None else jnp.asarray( masses )
    R = radius
    r = int( math.ceil( R / h ) ) + 1                        # the window of nodes a ball can reach, around its nearest node
    t = ( uv - jnp.asarray( origin ) ) / h                  # node coordinates
    i0 = jnp.round( t ).astype( jnp.int32 )
    off = jnp.arange( -r, r + 1, dtype = jnp.int32 )
    ia = i0[ :, 0, None, None ] + off[ None, :, None ]      # [ n, W, 1 ]
    ib = i0[ :, 1, None, None ] + off[ None, None, : ]      # [ n, 1, W ]
    da = ( ia - t[ :, 0, None, None ] ) * h
    db = ( ib - t[ :, 1, None, None ] ) * h
    q = R * R - da * da - db * db                           # [ n, W, W ]
    inside = q > 0
    # ( out of the support, the samples are masked, not their gradients only: the root's derivative is infinite at 0 )
    qs = jnp.where( inside, q, 1.0 )
    if profile == "smooth":
        prof = jnp.where( inside, qs * qs * jnp.sqrt( qs ), 0.0 ) * ( 7 / ( 2 * math.pi * R ** 7 ) )
    elif profile == "ball":
        prof = jnp.where( inside, jnp.sqrt( qs ), 0.0 ) * ( 3 / ( 2 * math.pi * R ** 3 ) )
    else:
        raise ValueError( f"GridTransport2d: unknown profile { profile !r } ( 'smooth' or 'ball' )" )
    ok = inside & ( ia >= 0 ) & ( ia <= N ) & ( ib >= 0 ) & ( ib <= N )
    flat = jnp.clip( ia, 0, N ) * ( N + 1 ) + jnp.clip( ib, 0, N )
    vals = jnp.where( ok, prof * m[ :, None, None ], 0.0 )
    v = jnp.zeros( ( N + 1 ) * ( N + 1 ), uv.dtype ).at[ flat.reshape( -1 ) ].add( vals.reshape( -1 ) ).reshape( N + 1, N + 1 )
    G = jnp.asarray( gaussian_matrix( N + 1, sigma ) )
    v = G @ v @ G.T
    return v + background * jnp.mean( v )


@functools.lru_cache( maxsize = 32 )
def _model( N, h, origin, radius, sigma, background, profile ):
    """the jitted `( uv, masses ) -> nodes` and `( uv, masses, cotangent ) -> d uv`, compiled once for all the views of a stage
    ( eager, the deposit and its derivative cost 28 ms per view at 2000 points, against 5 ms )"""
    def forward( uv, masses ):
        return _deposit( uv, masses, N, h, origin, radius, sigma, background, profile )

    def backward( uv, masses, cotangent ):
        return jax.vjp( lambda p: forward( p, masses ), uv )[ 1 ]( cotangent )[ 0 ]

    return jax.jit( forward ), jax.jit( backward )


class GridTransport2d:
    """one view: `pixels [ N, N ]` ( N a power of two, from 32 ), the corner of the detector `origin`, the side of a pixel
    `h`. The weights of the last solve are the warm start of the next one. `times`: seconds per phase, added up from one
    evaluation to the next ( may be shared between views ): `model`, `solve` ( with `solve_kernel`, what sdot measures inside ),
    `potential`, `model_backward`."""

    def __init__( self, pixels, origin, h, tol = 1e-9, max_iter = 200, times = None ):
        self.pixels = np.asarray( pixels, dtype = float )
        N = self.pixels.shape[ 0 ]
        if self.pixels.shape != ( N, N ) or N < 32 or N & ( N - 1 ):
            raise ValueError( f"GridTransport2d: N x N pixels, N a power of two from 32 ( got { self.pixels.shape } )" )
        self.N, self.h = N, float( h )
        self.origin = np.asarray( origin, dtype = float )
        self.tol, self.max_iter = float( tol ), int( max_iter )
        self.weights = None
        self.stats = { }
        self._data = { }
        self.nb_cold_retries = 0
        self.times = collections.defaultdict( float ) if times is None else times

    def _lap( self, phase, t0 ):
        """adds the time since `t0` to `phase`; returns the current time"""
        t = time.perf_counter()
        self.times[ phase ] += t - t0
        return t

    # ---- the measure ( diracs ) ------------------------------------------------------------------------------------------

    def data( self, sigma, background ):
        """the masses of the diracs: the pixels blurred by `sigma` ( pixels ) and lifted by `background` times their mean"""
        key = ( float( sigma ), float( background ) )
        if key not in self._data:
            G = gaussian_matrix( self.N, sigma )
            b = G @ self.pixels @ G.T
            self._data = { key: b + background * b.mean() }   # ( one stage at a time )
        return self._data[ key ]

    # ---- the model ( P1 image ) ------------------------------------------------------------------------------------------

    def nodes( self, uv, radius, sigma, background, masses = None, profile = "smooth" ):
        """`[ N + 1, N + 1 ]`: the projected kernels ( `profile` ) of radius `radius` ( physical ) at `uv [ n, 2 ]`, sampled at the nodes,
        blurred by `sigma` ( pixels ), lifted by `background` times their mean. Differentiable ( JAX )."""
        return _model( self.N, self.h, tuple( self.origin.tolist() ), float( radius ), float( sigma ), float( background ), profile )[ 0 ]( jnp.asarray( uv ), masses )

    # ---- the solve and the derivative ------------------------------------------------------------------------------------

    def _problem( self, b, v ):
        import sdot
        frame = [ [ self.h, 0.0 ], [ 0.0, self.h ] ]
        src = sdot.SumOfDiracsOnAGrid( ( self.N, self.N ), weights = b.reshape( -1 ), cell_frame = frame, origin = self.origin )
        img = sdot.Image( values = corner_coeffs( v ), degree = 1, pixel_frame = frame, origin = list( self.origin ) )
        return sdot.OtProblem( src, img )

    def _solve( self, b, v, weights0 = None ):
        import sdot
        tun = sdot.Tuning( grid_solver = "dedicated" )
        w0 = None if weights0 is None else jnp.asarray( weights0 )
        plan = self._problem( b, v ).solve( sdot.Iterative( tol = self.tol, max_iter = self.max_iter, weights0 = w0, on_failure = "ignore", tuning = tun ) )
        self.times[ "solve_kernel" ] += float( np.asarray( plan.stats[ "t_total" ] ) )
        return plan

    def _solve_blur_continuation( self, uv, radius, sigma, background, masses, profile ):
        """a cold solve at a blur where it converges, then down to `sigma`, each level warm started from the previous one.
        ( Cold, the solver fails from sigma = 0.5 pixel down on thin shells, whatever the background: 330 s per view, where
        the chain 1 -> 0.5 -> 0 takes 2.5 s + 4.5 s )"""
        weights = None
        for s in [ s for s in self.CONTINUATION_SIGMAS if s > sigma ] + [ sigma ]:
            v = self.nodes( uv, radius, s, background, masses, profile )
            plan = self._solve( self.data( s, background ), v, weights )
            if self._converged( plan ) or weights is None:
                weights = plan.weights
        return plan

    CONTINUATION_SIGMAS = ( 4.0, 2.0, 1.0, 0.5 )

    @staticmethod
    def _converged( plan ):
        return str( np.asarray( plan.stats[ "status" ] ) ) == "converged"

    def _potential_integrals( self, weights, v ):
        """`[ N + 1, N + 1 ]`: per node, the integral of the Kantorovich potential against its basis function"""
        import sdot
        frame = [ [ self.h, 0.0 ], [ 0.0, self.h ] ]
        c = ( np.arange( self.N ) + 0.5 ) * self.h
        pos = np.stack( np.meshgrid( self.origin[ 0 ] + c, self.origin[ 1 ] + c, indexing = "ij" ), -1 ).reshape( -1, 2 )
        img = sdot.Image( values = corner_coeffs( v ), degree = 1, pixel_frame = frame, origin = list( self.origin ) )
        pd = sdot.PowerDiagram( jnp.asarray( pos ), jnp.asarray( weights ), distribution = img, kernel_dtype = "FP64" )
        return nodes_of_coeffs( jnp.asarray( pd.potential_integrals() ) )

    def value_and_grad( self, uv, radius, sigma, background, masses = None, profile = "smooth" ):
        """the cost and its derivative with respect to `uv [ n, 2 ]`"""
        t = time.perf_counter()
        uv = jnp.asarray( uv, jnp.float64 )
        b = self.data( sigma, background )
        forward, backward = _model( self.N, self.h, tuple( self.origin.tolist() ), float( radius ), float( sigma ), float( background ), profile )
        v = forward( uv, masses ).block_until_ready()
        t = self._lap( "model", t )
        plan = None if self.weights is None else self._solve( b, v, self.weights )
        if plan is None or not self._converged( plan ):
            # ( no weights yet, or a warm start too far from the solution: a continuation in the blur )
            self.nb_cold_retries += 1
            plan = self._solve_blur_continuation( uv, radius, sigma, background, masses, profile )
        self.weights = np.asarray( plan.weights )
        self.stats = { k: np.asarray( s ) for k, s in plan.stats.items() }
        t = self._lap( "solve", t )
        I = self._potential_integrals( self.weights, v ).block_until_ready()
        t = self._lap( "potential", t )
        # the image is normalized by the solver: its mass S = sum_nodes v q, q the integral of a basis function
        q = nodes_of_coeffs( jnp.full( ( self.N, self.N, 4 ), self.h * self.h / 4 ) )
        S = jnp.sum( v * q )
        g_v = I / S - q * jnp.sum( v * I ) / S ** 2
        g_uv = backward( uv, masses, g_v ).block_until_ready()
        self._lap( "model_backward", t )
        return float( np.asarray( plan.cost ) ), g_uv
