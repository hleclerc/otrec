import numpy as np

from errand import test

from otrec import Reconstruction, Recorder, Sinogram, ball_packing, cloud, metrics

if test( "the sizes of a multiscale: n_end at the end, a division between stages, from about n_start" ):
    assert cloud.sizes( 25_000, 100_000, 2 ) == [ 25_000, 50_000, 100_000 ]
    assert cloud.sizes( 30_000, 100_000, 2 ) == [ 25_000, 50_000, 100_000 ] and cloud.sizes( 40_000, 100_000, 2 ) == [ 50_000, 100_000 ]
    assert cloud.sizes( 3, 10, 2 ) == [ 3, 5, 10 ] and cloud.sizes( 7, 10, 2 ) == [ 5, 10 ] and cloud.sizes( 10, 10, 3 ) == [ 10 ]
    assert cloud.sizes( 6_000, 100_000, 4 ) == [ 6_250, 25_000, 100_000 ]
    p = cloud.split( np.random.default_rng( 0 ), np.zeros( ( 3, 2 ) ), 2, 0.1, 5 )
    assert p.shape == ( 5, 2 ) and np.abs( p ).max() < 1
    g = np.stack( np.meshgrid( np.arange( 20.0 ), np.arange( 20.0 ) ), -1 ).reshape( -1, 2 )
    assert np.isclose( cloud.spacing( 0.1 * g ), 0.1, rtol = 1e-2 )

if test( "2D: stages of decreasing blur find a few shapes from a random start" ):
    truth = np.array( [ [ 0.30, 0.10 ], [ -0.25, 0.35 ], [ -0.10, -0.40 ], [ 0.45, -0.30 ] ] )
    sino = Sinogram( 2, 12, 64, extent = 2.0 )
    sino.values = Reconstruction( sino, truth, radius = 0.12 ).image( sigma = 0.0, background = 0.0 )
    rec = Reconstruction( sino, radius = 0.12, move_tol = 1e-3, seed = 1 ).random_points( 4 )
    for sigma in ( 0.25, 0.125, 0.06, 0.03, 0.0 ):
        rec.lbfgs( sigma )
    assert np.linalg.norm( rec.points[ :, None ] - truth[ None ], axis = 2 ).min( 0 ).max() < 1e-2, rec.points
    assert all( l[ "nb_iter" ] < rec.max_iter for l in rec.history )            # ( stopped because nothing moved any more )

if test( "2D: a multiscale gives the image of a packing, at a radius it finds; polygons then lower the loss; the recorder keeps the stages" ):
    image = ball_packing( 64, ball_radii = ( 0.1, 0.2 ) )
    sino = Sinogram.from_image( image, nb_views = 96, nb_bins = 64 )
    rc = Recorder( every = 10 )
    rec = Reconstruction( sino, recorder = rc )
    image_error = rc.metrics[ "image error" ] = metrics.make_image_error( image, sino.extent, rec )
    rec.multiscale( 2_000, 7_500, radius = 0.8 )
    assert [ l[ "nb_points" ] for l in rec.history ] == [ 1_875, 3_750, 7_500 ]
    r = [ l[ "radius" ] for l in rec.history ]
    assert r[ 0 ] > r[ 1 ] > r[ 2 ] == rec.radius and np.isclose( rec.radius, rec.nominal_radius(), rtol = 0.2 ) and np.all( rec.radii == rec.radius ), ( r, rec.nominal_radius() )
    # the room of a ball of the nominal radius, at the density of the matter: its share of the matter, about
    assert 0.7 < np.pi * rec.radius ** 2 / 2.5 / ( image.mean() * sino.extent ** 2 / 7_500 ) < 1.4
    assert image_error( rec.points, rec.radii ) < 0.25 < 0.8 < rc.frames[ 0 ][ "metrics" ][ "image error" ], image_error( rec.points, rec.radii )
    assert metrics.sinogram_residual( rec ) < 0.02
    assert len( rc.losses ) == its_all + 3 if ( its_all := sum( l[ "nb_iter" ] for l in rec.history ) ) else False     # ( every iteration, and the start of each stage )
    assert min( f for _, f in rc.losses ) > rec.loss_fn.floor
    its = [ f[ "iteration" ] for f in rc.frames ]
    assert its[ 0 ] == 0 and its[ -1 ] == sum( l[ "nb_iter" ] for l in rec.history ) and its == sorted( its )
    assert { f[ "stage" ] for f in rc.frames } == { 0, 1, 2 } and [ len( f[ "points" ] ) for f in rc.frames ] == [ f[ "nb_points" ] for f in rc.frames ]
    from otrec.shapes import polygons
    rec.to_polygons( max_iter = 100 )
    X = rec.model.tensor( rec.polygons )
    img = polygons.density( *polygons.vertices( X, rec.model ), polygons.shapes( X, rec.model )[ 1 ], 64, sino.extent )
    assert rec.loss < rec.history[ -1 ][ "loss0" ] and np.isclose( img.sum(), 1.0, rtol = 2e-2 ), rec.history[ -1 ]
    assert np.linalg.norm( rec.points, axis = 1 ).max() + rec.radii.min() <= 1.0 and rec.radii.std() > 0

if test( "3D: a stage lowers the loss of a random cloud" ):
    sino = Sinogram.from_image( ball_packing( 64, dim = 3, ball_radii = ( 0.15, 0.25 ) ), nb_views = 4, nb_bins = 32 )
    rec = Reconstruction( sino, radius = 2.5 * sino.h, background = 1e-1, max_iter = 10, seed = 1 ).random_points( 500, radius = 0.8, height = 0.4 )
    rec.lbfgs( sigma = 2 * sino.h )
    assert rec.loss < 0.7 * rec.history[ 0 ][ "loss0" ], rec.history
