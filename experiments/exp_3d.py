"""A packing of balls in a slab ( `otrec.ball_packing`: a 3D image ), reconstructed from its radiographs and a RANDOM start
( uniform in the cylinder that holds it ), with an HTML page of the steps and the convergence.

    errand run --env torch -o 'otrec/experiments/exp_3d.py::*'                              ( from ~/Projects )

The defaults are a SMALL case ( a laptop ): far too few views for the density. The sizes of interest ( 256^2 pixels and
more, 1e5 to 1e7 shapes ) are for a compute node: the views are solved one after the other on the grid solver."""
from errand import experiment, Param

if p := experiment( "3d",
        n_start = Param( 10_000, help = "shapes of the first stage" ),
        division = Param( 2, help = "each shape is divided in that many after each of the first `splits` stages" ), splits = Param( 1 ),
        views = Param( 12 ), bins = Param( 64, help = "a power of two, from 32" ),
        radius = Param( 0.08, help = "of the shapes ( the image is sampled at the corners of the pixels: not below ~1.5 pixel )" ),
        sigmas = Param( "0.125:0.0625:0.03125", help = "the blur of each stage" ),
        backgrounds = Param( "1e-1:3e-2:1e-2", help = "per stage" ),
        move_tol = Param( 0.04, help = "a stage stops when the shapes move less than that fraction of their radius in 10 iterations" ),
        max_iter = Param( 25, help = "or after that many iterations" ),
        height = Param( 0.4 ), record_every = Param( 2 ), seed = Param( 0 ), device = Param( "default", help = "cpu, cuda, or loom's default" ) ):
    import time

    import numpy as np

    from otrec import Reconstruction, Recorder, Sinogram, ball_packing, viz

    image = ball_packing( 2 * p.bins, dim = 3, height = p.height, ball_radii = ( 0.08, 0.15 ), seed = p.seed )
    sino = Sinogram.from_image( image, p.views, p.bins )

    rec = Reconstruction( sino, radius = p.radius, move_tol = p.move_tol, max_iter = p.max_iter, seed = p.seed,
                          device = None if p.device == "default" else p.device, recorder = Recorder( p.record_every ), verbose = True )
    rec.random_points( p.n_start, radius = 0.8, height = p.height )
    t0 = time.perf_counter()
    # ( the grid solver needs a blur and a background that go down: the stages are written out )
    for i, ( sigma, background ) in enumerate( zip( p.sigmas.split( ":" ), p.backgrounds.split( ":" ) ) ):
        rec.lbfgs( float( sigma ), float( background ) )
        if i < p.splits:
            rec.split( p.division )
    p.results.update( seconds = time.perf_counter() - t0, loss = rec.loss )
    print( p.results )

    # the object, for the page: the centres of a sample of its solid voxels
    solid = np.argwhere( image > 0.5 )
    solid = solid[ np.random.default_rng( 1 ).choice( len( solid ), min( 30_000, len( solid ) ), replace = False ) ]
    viz.export_html( rec, p.out_dir / "cloud.html", title = f"otrec 3D -- a packing of balls, { rec.nb_points } shapes, random start",
                     truth_points = ( solid + 0.5 ) * ( sino.extent / image.shape[ 0 ] ) - sino.extent / 2 )
