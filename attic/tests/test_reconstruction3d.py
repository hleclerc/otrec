"""3D reconstruction ( `Reconstruction3d` ): diracs whose projections are transported, view by view, onto radiographs by the
exact 2D semi-discrete transport of sdot; the lifted Lloyd step and L-BFGS on known phantoms ( `phantoms` ).
"""
import numpy as np

from errand import test

from otrec import phantoms
from otrec.geometry import ParallelGeometry
from otrec.reconstruction3d import Reconstruction3d


if test( "parallel geometry: project and jacobian agree with the radiographs" ):
    import jax
    import jax.numpy as jnp
    radio = phantoms.touching().radiographs( nb_angles = 5, nb_pixels = 16 )
    geo = radio.geometry
    assert isinstance( geo, ParallelGeometry ) and geo.nb_views == 5
    x = jnp.asarray( np.random.default_rng( 0 ).normal( size = ( 7, 3 ) ) )
    ref = radio.project_points( np.asarray( x ) )
    for k in range( 5 ):
        assert np.allclose( geo.project( x, k ), ref[ k ] )
        J = jax.jacfwd( lambda y: geo.project( y[ None ], k )[ 0 ] )( x[ 0 ] )
        assert np.allclose( geo.jacobian( x, k )[ 0 ], J )


if test( "phantoms: samples lie in the object, metrics see it" ):
    for name, make in phantoms.PHANTOMS.items():
        ph = make()
        s = ph.sample( 2000 )
        assert ( ph.distance_to_support( s ) == 0 ).all(), name
        m = ph.metrics( sample_size = 2000 )( ph.sample( 2000, seed = 5 ) )
        assert m[ "inside" ] == 1.0 and m.get( "balance", 0.0 ) < 0.05 and m[ "coverage" ] < 0.1, ( name, m )
    # a hollow ball: nothing sampled in the hole
    h = phantoms.hollow()
    assert ( np.linalg.norm( h.sample( 2000 ) - h.centers[ 1 ], axis = 1 ) >= h.radii[ 1 ] ).all()


if test( "voronoi foam: the porosity, and radiographs that match a sample of it" ):
    vf = phantoms.VoronoiFoam( nb_cells = 60, porosity = 0.8, height = 0.3, nb_mc = 100000 )
    r = vf.radiographs( nb_angles = 2, nb_pixels = 64 )
    assert r.nb_v_host < 64   # the detector is as high as the object, not as wide
    solid = 0.2 * np.pi * 0.8 ** 2 * 0.3
    assert abs( vf.voxel_volume / solid - 1 ) < 0.02 and np.allclose( r.mass(), vf.voxel_volume, rtol = 1e-3 ), ( vf.voxel_volume, r.mass() )
    s = vf.sample( 200000 )
    assert ( vf.distance_to_support( s ) == 0 ).all()
    for k in range( 2 ):
        uv = r.project_points( s )[ k ]
        H = np.histogram2d( uv[ :, 0 ], uv[ :, 1 ], bins = ( r.nb_u_host, r.nb_v_host ),
                            range = [ [ r.u_min, r.u_min + r.extent_u ], [ r.v_min, r.v_min + r.extent_v ] ] )[ 0 ]
        R = np.asarray( r.values[ k ] )
        assert np.abs( H / H.sum() - R / R.sum() ).sum() < 0.12   # Monte-Carlo noise ~0.08 at this sample size


if test( "the object's own sample gives a small cost, consistent gradient and lifted step" ):
    ph = phantoms.balls( 3 )
    radio = ph.radiographs( nb_angles = 6, nb_pixels = 48 )
    good = Reconstruction3d( radio, ph.sample( 1500, seed = 1 ) ).evaluate()
    bad = Reconstruction3d( radio ).random_points( 1500, seed = 1 ).evaluate()
    assert good[ "nb_failed" ] == 0 and bad[ "nb_failed" ] == 0
    assert good[ "cost" ] < bad[ "cost" ] / 20, ( good[ "cost" ], bad[ "cost" ] )
    # parallel beam: the gradient is -2/n H dx, H = sum_k B_k^T B_k
    B = np.asarray( radio.geometry.bases )
    H = np.einsum( "kij,kil->jl", B, B )
    assert np.allclose( bad[ "grad" ], -2 / 1500 * bad[ "step" ] @ H.T, atol = 1e-12 )
    assert ( bad[ "disagreement" ] >= -1e-12 ).all()


if test( "the cost's derivative by finite differences" ):
    ph = phantoms.touching()
    radio = ph.radiographs( nb_angles = 4, nb_pixels = 32 )
    rec = Reconstruction3d( radio, tol = 1e-9 ).visual_hull_points( 200, seed = 2 )
    ev = rec.evaluate()
    rng = np.random.default_rng( 0 )
    d = rng.normal( size = ( 200, 3 ) )
    eps = 1e-5
    x = rec.positions
    fp = rec.evaluate( x + eps * d, keep_weights = False )[ "cost" ]
    fm = rec.evaluate( x - eps * d, keep_weights = False )[ "cost" ]
    fd = ( fp - fm ) / ( 2 * eps )
    an = float( ( ev[ "grad" ] * d ).sum() )
    assert abs( fd - an ) < 1e-3 * abs( an ), ( fd, an )


if test( "the mass derivative: the dual weights, by finite differences" ):
    # moving mass from dirac i to dirac j changes the cost by ( mass_grad_i - mass_grad_j ) dm
    import jax.numpy as jnp
    from sdot.OtProblem import Iterative, OtProblem
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    ph = phantoms.balls( 2, seed = 4 )
    radio = ph.radiographs( nb_angles = 3, nb_pixels = 32 )
    rec = Reconstruction3d( radio, tol = 1e-10 ).visual_hull_points( 100, seed = 3 )
    ev = rec.evaluate()
    geo = radio.geometry
    vals = radio.with_background( 1e-6 )
    i, j, dm = 3, 40, 1e-6

    def cost( masses ):
        c = 0.0
        for k in range( geo.nb_views ):
            img = Image( values = vals[ k ], origin = [ geo.u_min, geo.v_min ], pixel_frame = [ [ geo.du, 0 ], [ 0, geo.dv ] ] )
            uv = np.asarray( geo.project( rec.points, k ) )
            plan = OtProblem( SumOfDiracs( uv, masses ), img ).solve( Iterative( tol = 1e-13, max_iter = 200, differentiable = False ) )
            c += float( plan.cost )
        return c

    m = np.full( 100, 1 / 100 )
    mp, mm = m.copy(), m.copy()
    mp[ i ] += dm; mp[ j ] -= dm
    mm[ i ] -= dm; mm[ j ] += dm
    fd = ( cost( mp ) - cost( mm ) ) / ( 2 * dm )
    an = float( ev[ "mass_grad" ][ i ] - ev[ "mass_grad" ][ j ] )
    assert abs( fd - an ) < 1e-3 * abs( an ) + 1e-9, ( fd, an )


if test( "lifted Lloyd brings the cloud into the balls" ):
    ph = phantoms.balls( 3 )
    radio = ph.radiographs( nb_angles = 8, nb_pixels = 64 )
    truth = ph.metrics( tol = radio.du )
    rec = Reconstruction3d( radio, verbose = True ).visual_hull_points( 2000 )
    rec.lloyd( 15, truth = truth, snapshot_every = 5 )
    h = rec.history
    assert h[ -1 ][ "cost" ] < h[ 0 ][ "cost" ] / 5
    assert h[ -1 ][ "inside" ] > 0.97 and h[ -1 ][ "balance" ] < 0.03, h[ -1 ]
    assert h[ -1 ][ "disagreement_mean" ] < h[ 0 ][ "disagreement_mean" ] / 5
    assert len( rec.frames ) == 4 and rec.frames[ -1 ][ "points" ].shape == ( 2000, 3 )


if test( "multiscale: few diracs first, then split" ):
    ph = phantoms.balls( 3 )
    radio = ph.radiographs( nb_angles = 8, nb_pixels = 64 )
    rec = Reconstruction3d( radio, verbose = True )
    rec.multiscale( 2000, nb_points_init = 125, factor = 4, steps = 6, last_steps = 8, tau = 1.5, truth = ph.metrics( tol = radio.du ),
                    snapshot_every = 100 )
    assert [ h[ "nb_points" ] for h in rec.history ] == [ 125 ] * 6 + [ 500 ] * 6 + [ 2000 ] * 8
    assert rec.history[ -1 ][ "inside" ] > 0.97, rec.history[ -1 ]
    assert [ len( f[ "points" ] ) for f in rec.frames ] == [ 125, 125, 500, 500, 2000, 2000 ]


if test( "L-BFGS lowers the cost too" ):
    ph = phantoms.balls( 3 )
    radio = ph.radiographs( nb_angles = 8, nb_pixels = 64 )
    rec = Reconstruction3d( radio, verbose = True ).visual_hull_points( 2000 )
    rec.lbfgs( 15, truth = ph.metrics() )
    h = rec.history
    assert h[ -1 ][ "cost" ] < h[ 0 ][ "cost" ] / 3, ( h[ 0 ][ "cost" ], h[ -1 ][ "cost" ] )
    assert h[ -1 ][ "nb_evals" ] >= len( h )


if test( "views spread over several devices give the same evaluation" ):
    import jax
    from loom.testing import need
    need( "cpu" )
    if len( jax.devices() ) < 2:
        from errand import skip
        skip( "one device ( XLA_FLAGS=--xla_force_host_platform_device_count=4 for more )" )
    ph = phantoms.balls( 2 )
    radio = ph.radiographs( nb_angles = 4, nb_pixels = 32 )
    x = ph.sample( 300, seed = 1 )
    one = Reconstruction3d( radio, x, devices = jax.devices()[ :1 ] ).evaluate()
    many = Reconstruction3d( radio, x ).evaluate()
    assert np.allclose( one[ "cost" ], many[ "cost" ], rtol = 1e-12 ) and np.allclose( one[ "step" ], many[ "step" ], atol = 1e-12 )


if test( "the sliced loss: the object's sample is better, its gradient by finite differences, L-BFGS lowers it" ):
    import jax
    import jax.numpy as jnp
    ph = phantoms.balls( 3 )
    radio = ph.radiographs( nb_angles = 6, nb_pixels = 48 )
    rec = Reconstruction3d( radio ).random_points( 1500, seed = 1 )
    for method in ( "exact", "grid" ):
        f, vg = rec.sliced_loss( nb_directions = 8, method = method, dtype = jnp.float64 )
        good, bad = float( f( jnp.asarray( ph.sample( 1500, seed = 1 ) ) ) ), float( f( rec.points ) )
        assert good < 0.2 * bad, ( method, good, bad )
        x = rec.points
        v, g = vg( x )
        d, eps = np.random.default_rng( 2 ).normal( size = x.shape ), 1e-6
        fd = ( float( f( x + eps * d ) ) - float( f( x - eps * d ) ) ) / ( 2 * eps )
        an = float( ( np.asarray( g ) * d ).sum() )
        assert abs( fd - an ) < 1e-5 * abs( an ), ( method, fd, an )
    c0 = float( rec.sliced_loss( nb_directions = 8 )[ 0 ]( rec.points ) )
    rec.sliced( 30, nb_directions = 8 )
    c1 = float( rec.sliced_loss( nb_directions = 8 )[ 0 ]( rec.points ) )
    assert c1 < 0.5 * c0, ( c0, c1 )


if test( "the 2D loss ( pyramids against pixels-diracs ): its gradient by finite differences, L-BFGS lowers it" ):
    ph = phantoms.balls( 3 )
    radio = ph.radiographs( nb_angles = 4, nb_pixels = 24 )
    rec = Reconstruction3d( radio ).set_points( ph.sample( 400, seed = 1 ) + np.random.default_rng( 3 ).normal( scale = 0.03, size = ( 400, 3 ) ) )
    setting = dict( tol = 1e-9 )
    c, g = rec.pixel_value_and_grad( **setting )
    x = np.asarray( rec.points )
    d, eps = np.random.default_rng( 2 ).normal( size = x.shape ), 1e-5
    fd = ( rec.pixel_value_and_grad( x + eps * d, **setting )[ 0 ] - rec.pixel_value_and_grad( x - eps * d, **setting )[ 0 ] ) / ( 2 * eps )
    an = float( ( g * d ).sum() )
    assert abs( fd - an ) < 1e-4 * abs( an ), ( fd, an )
    rec.pixel2d( 8, **setting )
    c1 = rec.pixel_value_and_grad( **setting )[ 0 ]
    assert c1 < 0.8 * c, ( c, c1 )


if test( "the 2D loss in worker processes ( views spread over cards ) is the same" ):
    ph = phantoms.balls( 3 )
    radio = ph.radiographs( nb_angles = 4, nb_pixels = 24 )
    x = ph.sample( 400, seed = 1 ) + np.random.default_rng( 3 ).normal( scale = 0.03, size = ( 400, 3 ) )
    one = Reconstruction3d( radio ).set_points( x )
    two = Reconstruction3d( radio ).set_points( x )
    c1, g1 = one.pixel_value_and_grad( tol = 1e-9, cards = 1 )
    c2, g2 = two.pixel_value_and_grad( tol = 1e-9, cards = 2 )
    two.close_pixel_workers()
    assert abs( c1 - c2 ) < 1e-12 * c1 and np.abs( g1 - g2 ).max() < 1e-10 * np.abs( g1 ).max(), ( c1, c2, np.abs( g1 - g2 ).max() )


if test( "the grid loss ( pixels as grid diracs against a P1 image of projected balls ): the dedicated solver, its gradient by finite differences, L-BFGS lowers it" ):
    ph = phantoms.shells( 3 )
    radio = ph.radiographs( nb_angles = 4, nb_pixels = 64 )
    x = ph.sample( 400, seed = 1 ) + np.random.default_rng( 3 ).normal( scale = 0.03, size = ( 400, 3 ) )
    rec = Reconstruction3d( radio ).set_points( x )
    rec.grid_views( tol = 1e-10 )                        # ( the finite differences need converged costs )
    for sigma, bg in ( ( 4.0, 1e-1 ), ( 0.0, 1e-3 ) ):
        c, g = rec.grid_value_and_grad( radius = 2.5, sigma = sigma, background = bg )
        # every view solved by the dedicated solver ( `Tuning( grid_solver = "dedicated" )` would have raised otherwise )
        assert all( str( gt.stats.get( "grid_solver", "" ) ).startswith( "dedicated" ) for gt in rec.grid_views() ), [ gt.stats.get( "grid_solver" ) for gt in rec.grid_views() ]
        d, eps = np.random.default_rng( 2 ).normal( size = x.shape ), 1e-6
        fd = ( rec.grid_value_and_grad( x + eps * d, 2.5, sigma, bg )[ 0 ] - rec.grid_value_and_grad( x - eps * d, 2.5, sigma, bg )[ 0 ] ) / ( 2 * eps )
        an = float( ( g * d ).sum() )
        assert abs( fd - an ) < 1e-3 * abs( an ), ( sigma, fd, an )
    c0 = rec.grid_value_and_grad( radius = 2.5, sigma = 2.0, background = 1e-2 )[ 0 ]
    rec.grid2d( 8, radius = 2.5, sigma = 2.0, background = 1e-2 )
    c1 = rec.grid_value_and_grad( radius = 2.5, sigma = 2.0, background = 1e-2 )[ 0 ]
    assert c1 < 0.8 * c0, ( c0, c1 )
