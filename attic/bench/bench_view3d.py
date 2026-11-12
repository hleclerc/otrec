"""ONE VIEW of the 3D reconstruction: the 2D semi-discrete transport of `n` projected diracs onto a radiograph ( sdot's
Newton, one call ), cold and warm -- what one evaluation of `Reconstruction3d` costs, times the number of views.

    errand run --no-queue "otrec/bench/bench_view3d.py::solve"                                   # here ( jax CPU )
    errand run --no-queue --env lmo-jax "otrec/bench/bench_view3d.py::solve" --n=1e4,1e5,1e6     # the 2080 Ti
    errand run --no-queue --env lmo-jax "otrec/bench/bench_view3d.py::solve" --platform=cpu      # sdot's CPU path, there

THE PROBLEM: three balls ( `phantoms.balls` ), a `pixels` x `pixels` radiograph, a cloud of `n` points drawn in the
balls ( near the solution ). COLD: zero weights ( the solver picks its own start ). WARM: the cloud moved by a smooth
field of amplitude `move` ( about a late Lloyd step ), started from the cold weights.
"""
from errand import Param, bench


if p := bench( "solve",
               n          = Param( 100000, help = "number of diracs" ),
               pixels     = Param( 256, help = "detector side, in pixels" ),
               move       = Param( 1e-3, help = "the displacement before the warm solve" ),
               background = Param( 1e-6, help = "added to the radiograph, fraction of its mean" ),
               tol        = Param( 1e-4, help = "mass tolerance, relative to a dirac" ),
               precision  = Param( "auto", choices = [ "auto", "fp32", "fp64", "mixed" ] ),
               reps       = Param( 3, help = "repetitions ( the best is kept )" ),
               platform   = Param( "default", choices = [ "default", "cpu" ], help = "cpu: jax ( and sdot ) on the host, even with a card" ) ):
    import os
    import time
    os.environ.setdefault( "XLA_PYTHON_CLIENT_PREALLOCATE", "false" )   # ( the card may be shared with a long run )
    if p.platform == "cpu":
        os.environ[ "JAX_PLATFORMS" ] = "cpu"
        os.environ[ "LOOM_DEVICE" ] = "cpu"
    import jax
    import jax.numpy as jnp
    import numpy as np
    from otrec import phantoms
    from otrec.reconstruction3d import _make_solve

    ph = phantoms.balls( 3, seed = 0 )
    radio = ph.radiographs( nb_angles = 1, nb_pixels = p.pixels )
    geo = radio.geometry
    x = jnp.asarray( ph.sample( int( p.n ), seed = 2 ) )
    img = jnp.asarray( radio.with_background( p.background )[ 0 ] )
    solve = _make_solve( geo, p.tol, 200, p.precision, "auto" )
    k = jnp.int32( 0 )
    # a SMOOTH displacement field of amplitude `move`, as a Lloyd step is ( a random one per point would scramble the cells )
    x2 = x + p.move * jnp.sin( 2.0 * x[ :, ::-1 ] + jnp.array( [ 0.3, 1.1, 2.0 ] ) )

    def run( pts, w ):
        o = solve( pts, k, w, img )
        jax.block_until_ready( o )
        return o

    print( f"device { jax.devices()[ 0 ] }, n={ p.n:.0e} pixels={ p.pixels }² precision={ p.precision }" )
    t0 = time.perf_counter()
    cold = run( x, jnp.zeros( x.shape[ 0 ] ) )
    print( f"  first call { time.perf_counter() - t0:.2f} s" )
    for name, pts, w in [ ( "cold", x, jnp.zeros( x.shape[ 0 ] ) ), ( "warm", x2, cold[ "weights" ] ) ]:
        ts = [ ]
        for _ in range( p.reps ):
            t0 = time.perf_counter()
            o = run( pts, w )
            ts.append( time.perf_counter() - t0 )
        t = min( ts )
        print( f"  { name }  { t * 1e3:9.1f} ms   status { int( o[ 'status' ] ) }  newton { int( o[ 'nb_iter' ] ) } it "
               f"{ int( o[ 'nb_diag' ] ) } diagrams   start { int( o[ 'start' ] ) } ( 0 weights0, 1 voronoi, 2 similarity )  "
               f"continuation { int( o[ 'nb_continuation_steps' ] ) }  min start mass { float( o[ 'min_start_mass' ] ):.2e}   cost { float( o[ 'cost' ] ):.4e}" )
        p.results[ f"{ name }_ms" ] = t * 1e3
        p.results[ f"{ name }_iter" ] = int( o[ "nb_iter" ] )
