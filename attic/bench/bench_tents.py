"""ONE EVALUATION ( value + gradient ) of the exact tents loss: sdot's `ProjectedSumOfTents` against a sinogram, through
`ContinuousOtPlan1d` ( the generic 1D transport, a work-group per view ), on the card or on the host.

    errand run --no-queue --env lmo-jax "otrec/bench/bench_tents.py::eval" --n=5000,50000 --method=exact,grid
"""
from errand import Param, bench


if p := bench( "eval",
               n       = Param( 5000, help = "number of tents" ),
               angles  = Param( 600 ),
               bins    = Param( 256 ),
               width   = Param( 0.5, help = "half-width, in bins" ),
               device  = Param( "default", choices = [ "default", "cpu" ] ),
               method  = Param( "exact", choices = [ "exact", "grid", "diracs" ], help = "diracs: projected1d ( tents one fine cell wide )" ),
               dtype   = Param( "float64", choices = [ "float64", "float32" ], help = "of the points" ),
               scale   = Param( 8, help = "grid method: fine cells per bin" ),
               reps    = Param( 5 ) ):
    import os
    import time
    if p.device == "cpu":
        from otrec.device import use_cpu
        use_cpu()
    import jax
    jax.config.update( "jax_enable_x64", True )
    import jax.numpy as jnp
    import numpy as np
    from otrec.Sinogram import Sinogram
    from otrec.reconstruction2d import Reconstruction2d

    rng = np.random.default_rng( 0 )
    sino = Sinogram( nb_angles = p.angles, nb_bins = p.bins, extent = 2.0 )
    for c, r in zip( rng.uniform( -0.5, 0.5, ( 5, 2 ) ), rng.uniform( 0.1, 0.3, 5 ) ):
        sino.add_disk( center = c, radius = r )
    rec = Reconstruction2d( sino, extent = 1.6, dtype = getattr( np, p.dtype ), scale = int( p.scale ) ).random_points( int( p.n ), seed = 1 )
    shape = { "exact": "tents", "grid": "tents_grid", "diracs": "diracs" }[ p.method ]
    _, vg = rec.loss_function( shape, None if shape == "diracs" else p.width * sino.dw )
    x = rec.points
    t0 = time.perf_counter()
    jax.block_until_ready( vg( x ) )
    first = time.perf_counter() - t0
    ts = [ ]
    for _ in range( p.reps ):
        t0 = time.perf_counter()
        jax.block_until_ready( vg( x ) )
        ts.append( time.perf_counter() - t0 )
    dev = next( iter( jax.tree.leaves( vg( x ) )[ 0 ].devices() ) )
    print( f"device { dev }  { p.method }{ '' if p.method == 'exact' else f' x{ p.scale }' } { p.dtype }  n { p.n }  angles { p.angles }: { min( ts ) * 1e3:.1f} ms ( first call { first:.1f} s )", flush = True )
    p.results[ "ms" ] = min( ts ) * 1e3
