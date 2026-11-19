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
    rec = Reconstruction( sino ).add_random_balls( 4, 0.12 )
    for sigma in ( 0.25, 0.125, 0.06, 0.03, 0.0 ):
        rec.adjust_parameters( centers = True, x_tol = 1e-3, sigma = sigma )
    assert np.linalg.norm( rec.points[ :, None ] - truth[ None ], axis = 2 ).min( 0 ).max() < 1e-2, rec.points
    assert all( l[ "nb_iter" ] < 2000 for l in rec.recorder.stages )            # ( stopped because nothing moved any more )

if test( "2D: a multiscale gives the image of a packing, at a radius it finds; polygons then lower the loss; the recorder keeps the stages" ):
    image = ball_packing( 64, ball_radii = ( 0.1, 0.2 ) )
    sino = Sinogram.from_image( image, nb_views = 96, nb_bins = 64 )
    rc = Recorder( every = 10 )
    rec = Reconstruction( sino, recorder = rc )
    image_error = rc.metrics[ "image error" ] = metrics.make_image_error( image, sino.extent, rec )
    rec.multiscale( 2_000, 7_500 )
    assert [ l[ "nb_points" ] for l in rec.recorder.stages ] == [ 1_875, 3_750, 7_500 ]
    r = [ l[ "radius" ] for l in rec.recorder.stages ]
    assert r[ 0 ] > r[ 1 ] > r[ 2 ] == rec.radius and np.isclose( rec.radius, rec.nominal_radius(), rtol = 0.2 ) and np.all( rec.radii == rec.radius ), ( r, rec.nominal_radius() )
    # the room of a ball of the nominal radius, at the density of the matter: its share of the matter, about
    assert 0.7 < np.pi * rec.radius ** 2 / 2.5 / ( image.mean() * sino.extent ** 2 / 7_500 ) < 1.4
    assert image_error( rec.points, rec.radii ) < 0.25 < 0.8 < rc.frames[ 0 ][ "metrics" ][ "image error" ], image_error( rec.points, rec.radii )
    assert metrics.sinogram_residual( rec ) < 0.02
    assert len( rc.losses ) == its_all + 3 if ( its_all := sum( l[ "nb_iter" ] for l in rec.recorder.stages ) ) else False     # ( every iteration, and the start of each stage )
    assert min( f for _, f in rc.losses ) > rec.loss_fn.floor
    its = [ f[ "iteration" ] for f in rc.frames ]
    assert its[ 0 ] == 0 and its[ -1 ] == sum( l[ "nb_iter" ] for l in rec.recorder.stages ) and its == sorted( its )
    assert { f[ "stage" ] for f in rc.frames } == { 0, 1, 2 } and [ len( f[ "points" ] ) for f in rc.frames ] == [ f[ "nb_points" ] for f in rc.frames ]
    from otrec.shapes import polygons
    rec.fit_polygons( max_iter = 100 )
    X = rec.model.tensor( rec.polygons )
    img = polygons.density( *polygons.vertices( X, rec.model ), polygons.shapes( X, rec.model )[ 1 ], 64, sino.extent )
    assert rec.loss < rec.recorder.stages[ -1 ][ "loss0" ] and np.isclose( img.sum(), 1.0, rtol = 2e-2 ), rec.recorder.stages[ -1 ]
    assert np.linalg.norm( rec.points, axis = 1 ).max() + rec.radii.min() <= 1.0 and rec.radii.std() > 0
    assert rc.frames[ 0 ][ "drawing" ] == { } and rc.frames[ -1 ][ "drawing" ][ "faces" ][ 1 ].shape == ( 8 * rec.nb_points, 3 )      # ( balls as discs, polygons as triangles )

if test( "3D: a stage lowers the loss of a random cloud" ):
    sino = Sinogram.from_image( ball_packing( 64, dim = 3, ball_radii = ( 0.15, 0.25 ) ), nb_views = 4, nb_bins = 32 )
    rec = Reconstruction( sino, background = 1e-1 ).add_random_balls( 500, 2.5 * sino.h )
    rec.adjust_parameters( centers = True, max_iter = 10, sigma = 2 * sino.h )
    assert rec.loss < 0.7 * rec.recorder.stages[ 0 ][ "loss0" ], rec.recorder.stages

if test( "2D: covariances alone leave the centres; the recorder keeps every step with its parameters" ):
    truth = np.array( [ [ 0.30, 0.10 ], [ -0.25, 0.35 ], [ -0.10, -0.40 ], [ 0.45, -0.30 ] ] )
    sino = Sinogram( 2, 12, 64, extent = 2.0 )
    sino.values = Reconstruction( sino, truth, radius = 0.12 ).image( sigma = 0.0, background = 0.0 )
    rec = Reconstruction( sino ).add_random_balls( 4, 0.12 ).adjust_parameters( centers = True, x_tol = 1e-3, sigma = 0.2 )
    centres = rec.points.copy()
    rec.adjust_parameters( centers = False, cov_matrices = True, x_tol = 1e-3, max_iter = 50 )
    assert np.allclose( rec.points, centres ) and rec.ellipses.shape == ( 4, 5 ) and rec.loss <= rec.recorder.stages[ -1 ][ "loss0" ]    # ( balls are the truth here )
    steps = rec.recorder.steps
    assert [ s[ "step" ] for s in steps ] == [ "add_random_balls", "adjust_parameters", "adjust_parameters" ], steps
    assert steps[ 2 ][ "cov_matrices" ] and not steps[ 2 ][ "centers" ] and steps[ 1 ][ "sigma" ] == 0.2 and steps[ 0 ][ "radius" ] == 0.12
    try:
        rec.adjust_parameters( x_tol = 1e-3 )                                 # ( nothing said to move )
        assert False
    except ValueError:
        pass


if test( "balls beyond the border of the domain come back ( projected steps )" ):
    import numpy as np
    from otrec import Reconstruction, Sinogram
    truth = np.array( [ [ 0.1, 0.0 ], [ -0.1, 0.05 ], [ 0.0, -0.1 ], [ 0.05, 0.1 ] ] )
    s = Sinogram( 2, 24, 64, extent = 2.0 )
    s.values = Reconstruction( s, truth, radius = 0.15 ).image( sigma = 0.0, background = 0.0 )
    start = np.array( [ [ 0.98, 0.0 ], [ 0.0, -0.97 ], [ -0.7, 0.7 ], [ 0.2, 0.1 ] ] )   # three where a ball of 0.15 cannot be
    rec = Reconstruction( s, start, radius = 0.15, device = "cpu" )
    rec.adjust_parameters( centers = True, x_tol = 0.01 )
    assert np.linalg.norm( rec.points, axis = 1 ).max() < 0.5, rec.points


if test( "densities: a density each, found from the sinogram of balls of two densities; kept by the stages after" ):
    import numpy as np
    from otrec import Reconstruction, Sinogram
    rng = np.random.default_rng( 3 )
    pts = rng.uniform( -0.5, 0.5, size = ( 30, 2 ) )
    truth = np.where( pts[ :, 0 ] > 0, 3.0, 1.0 )                           # the right half three times as dense
    s = Sinogram( 2, 36, 96, extent = 2.0 )
    ref = Reconstruction( s, pts, radius = 0.08, device = "cpu" )
    ref.densities = truth
    s.values = ref.image( sigma = 0.0, background = 0.0 )
    rec = Reconstruction( s, pts, radius = 0.08, device = "cpu" )
    rec.adjust_parameters( densities = True, x_tol = 1e-4 )
    d = rec.densities / rec.densities[ truth == 1 ].mean()                  # ( a density is known up to a factor )
    assert np.allclose( d, truth, rtol = 0.05 ), d
    rec.adjust_parameters( centers = True, radii = True, x_tol = 0.01 )     # ( kept, fixed )
    assert np.allclose( rec.densities / rec.densities[ truth == 1 ].mean(), d )
    rec.adjust_parameters( centers = True, cov_matrices = True, densities = True, x_tol = 0.01 )
    assert rec.ellipses.shape == ( 30, 5 ) and rec.densities.shape == ( 30, )
    assert "density" in rec.fields()
    rec.adjust_parameters( centers = True, x_tol = 0.01 )                   # ( balls of one density: forgotten )
    assert rec.densities is None


if test( "remove_isolated: the balls alone far from the others go, with their radii and densities" ):
    import numpy as np
    from otrec import Reconstruction, Sinogram
    g = np.stack( np.meshgrid( np.linspace( -0.3, 0.3, 7 ), np.linspace( -0.3, 0.3, 7 ) ), -1 ).reshape( -1, 2 )
    pts = np.concatenate( [ g, [ [ 0.8, 0.0 ], [ -0.6, -0.6 ] ] ] )
    rec = Reconstruction( Sinogram( 2, 8, 32, extent = 2.0 ), pts, radius = 0.05 )
    rec.densities = np.arange( len( pts ), dtype = float ) + 1
    rec.remove_isolated( 1.5 )
    assert rec.nb_points == 49 and np.allclose( rec.points, g ) and np.allclose( rec.densities, np.arange( 49 ) + 1 )
    assert rec.recorder.steps[ -1 ][ "removed" ] == 2


if test( "a removal at the end keeps the ellipses of the others, logs its loss, and records a frame" ):
    import numpy as np
    from otrec import Reconstruction, Recorder, Sinogram
    g = np.stack( np.meshgrid( np.linspace( -0.3, 0.3, 5 ), np.linspace( -0.3, 0.3, 5 ) ), -1 ).reshape( -1, 2 )
    s = Sinogram( 2, 24, 64, extent = 2.0 )
    s.values = Reconstruction( s, g, radius = 0.08 ).image( sigma = 0.0, background = 0.0 )
    rec = Reconstruction( s, np.concatenate( [ g, [ [ 0.8, 0.0 ] ] ] ), radius = 0.08, recorder = Recorder( every = 5 ), device = "cpu" )
    rec.adjust_parameters( cov_matrices = True, max_iter = 10 )                 # ( the centres fixed: the stray one stays )
    n = len( rec.recorder.frames )
    rec.remove_isolated( 1.5 )
    step, fr = rec.recorder.steps[ -1 ], rec.recorder.frames[ -1 ]
    assert step[ "removed" ] >= 1 and rec.ellipses.shape == ( rec.nb_points, 5 ) and np.isfinite( step[ "loss" ] )
    assert len( rec.recorder.frames ) == n + 1 and len( fr[ "points" ] ) == fr[ "nb_points" ] == rec.nb_points and "matrices" in fr[ "drawing" ]


if test( "remove_in_air: the balls a view sees in empty air go, those in the matter stay" ):
    import numpy as np
    from otrec import Reconstruction, Sinogram
    g = np.stack( np.meshgrid( np.linspace( -0.3, 0.3, 7 ), np.linspace( -0.3, 0.3, 7 ) ), -1 ).reshape( -1, 2 )
    s = Sinogram( 2, 36, 96, extent = 2.0 )
    s.values = Reconstruction( s, g, radius = 0.06 ).image( sigma = 0.0, background = 0.0 )
    rec = Reconstruction( s, np.concatenate( [ g, [ [ 0.6, 0.1 ], [ 0.45, 0.45 ] ] ] ), radius = 0.06, device = "cpu" )
    rec.remove_in_air()
    assert rec.nb_points == 49 and np.allclose( rec.points, g )
