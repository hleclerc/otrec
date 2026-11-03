"""ONE EVALUATION of the 2D loss of `Reconstruction3d` ( pyramids against the pixels as diracs, every view ), warm, on one
card or spread over all of them.

    errand run --no-queue --env n22x2-jax "otrec/bench/bench_pixel2d.py::eval" --devices=1,all
"""
from errand import Param, bench


if p := bench( "eval",
               n       = Param( 50000, help = "number of diracs" ),
               views   = Param( 60 ),
               pixels  = Param( 256 ),
               devices = Param( "all", choices = [ "1", "all" ] ),
               split   = Param( 1, help = "diracs per pixel and per axis" ),
               reps    = Param( 3 ) ):
    import os
    os.environ[ "XLA_PYTHON_CLIENT_PREALLOCATE" ] = "false"     # the cards are shared with the workers
    import time
    import jax
    import numpy as np
    from otrec import phantoms
    from otrec.reconstruction3d import Reconstruction3d

    ph = phantoms.VoronoiFoam( nb_cells = 59, porosity = 0.8, height = 0.2, seed = 0 )
    radio = ph.radiographs( nb_angles = p.views, nb_pixels = p.pixels )
    devs = jax.devices() if p.devices == "all" else jax.devices()[ :1 ]
    x = ph.sample( int( p.n ), seed = 1 ) + np.random.default_rng( 2 ).normal( scale = 0.3 * radio.du, size = ( int( p.n ), 3 ) )
    rec = Reconstruction3d( radio, devices = devs ).set_points( x )
    t0 = time.perf_counter()
    nc = len( devs )
    rec.pixel_value_and_grad( cards = nc, split = int( p.split ) )
    first = time.perf_counter() - t0
    ts = [ ]
    for _ in range( p.reps ):
        x = x + np.random.default_rng( 3 ).normal( scale = 0.01 * radio.du, size = x.shape )
        t0 = time.perf_counter()
        rec.pixel_value_and_grad( x, cards = nc, split = int( p.split ) )
        ts.append( time.perf_counter() - t0 )
    print( f"split { p.split }  { nc } card(s) { [ str( d ) for d in devs ] }  n { p.n }  { p.views } views: { min( ts ):.2f} s ( first { first:.1f} s )", flush = True )
    p.results[ "s" ] = min( ts )
    rec.close_pixel_workers()
