"""2D reconstruction ( `Reconstruction2d` ): a cloud of diracs or domes whose projections must reproduce
the measured sinogram, optimized on the device ( `device_optim` ) against the sort-free grid transport
of `sdot.projected1d`.

The scenarios of the first version of otrec, ported: a single disk ( diracs inside it give a small
loss; random diracs converge into it ), L-BFGS against gradient descent, coarse-to-fine, and the chain
diracs -> domes that recovers the centers of a few domes.
"""
import numpy as np

from errand import test

from otrec.Sinogram import Sinogram
from otrec.reconstruction2d import Reconstruction2d


def _disk_sinogram( nb_angles = 8, nb_bins = 201, extent = 6.0, center = ( 0.3, -0.2 ), radius = 1.0 ):
    s = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    s.add_disk( center = list( center ), radius = radius )
    return s


TRUTH = np.array( [ [ 0.8, 0.3 ], [ -0.9, 0.6 ], [ 0.1, -1.1 ], [ -0.5, -0.7 ], [ 1.3, -0.4 ] ] )


def _domes_sinogram( radius, nb_angles = 24, nb_bins = 128 ):
    s = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = 6.0 )
    for c in TRUTH:
        s.add_dome( center = c, radius = radius )
    return s


if test( "in_disk_gives_small_loss" ):
    # diracs sampling the target disk reproduce its projections: the loss must be small. Also
    # exercises ZERO-density profiles ( the detector tails ).
    sino = _disk_sinogram()
    rng = np.random.default_rng( 0 )
    pts = []
    while len( pts ) < 400:
        xy = ( rng.random( 2 ) - 0.5 ) * 2
        if xy[ 0 ] ** 2 + xy[ 1 ] ** 2 < 1:
            pts.append( xy + np.array( [ 0.3, -0.2 ] ) )
    rec = Reconstruction2d( sino, np.array( pts ) )
    l = rec.loss()
    uniform = Reconstruction2d( sino ).random_points( 400, extent = 5.0 ).loss()
    assert np.isfinite( l ) and l < 0.05 and l < uniform / 20, ( l, uniform )


if test( "reconstruct_converges" ):
    # from random diracs, gradient descent makes the loss drop and brings the diracs INTO the disk
    center, radius = np.array( [ 0.3, -0.2 ] ), 1.0
    sino = _disk_sinogram( nb_angles = 8, center = center, radius = radius )
    rec = Reconstruction2d( sino, extent = 5.0, record_every = 10 ).random_points( 100, seed = 3 )
    l0 = rec.loss()
    rec.diracs( optimizer = "gd", lr = 0.5, max_iter = 100 )
    l1 = rec.loss()
    assert l1 < l0 / 10, f"the loss did not drop enough : { l0 } -> { l1 }"
    pn = rec.positions
    assert np.allclose( pn.mean( axis = 0 ), center, atol = 0.1 )
    assert ( np.linalg.norm( pn - center, axis = 1 ) <= radius + 0.1 ).mean() > 0.9, "most diracs should be in the disk"
    ( h, ) = rec.history
    assert h[ "model" ] == "diracs" and h[ "nb_points" ] == 100 and h[ "nb_steps" ] == 100
    assert abs( h[ "loss_before" ] - l0 ) < 1e-6 * l0 and abs( h[ "loss_after" ] - l1 ) < 1e-6 * l0
    assert len( rec.frames ) == 11            # the start, then one every 10 steps


if test( "lbfgs_vs_gradient_descent" ):
    # same start, two optimizers: L-BFGS needs far fewer steps for a loss at least as good
    sino = _disk_sinogram( nb_angles = 12 )
    start = Reconstruction2d( sino, extent = 5.0 ).random_points( 80, seed = 42 ).points
    gd = Reconstruction2d( sino, start ).diracs( optimizer = "gd", lr = 0.2, max_iter = 200 ).history[ -1 ]
    lb = Reconstruction2d( sino, start ).diracs( max_iter = 200, ftol = 1e-8 ).history[ -1 ]
    print( f"\n  GD:    { gd[ 'nb_steps' ]:3d} steps, loss { gd[ 'loss_before' ]:.6f} -> { gd[ 'loss_after' ]:.6f}" )
    print( f"  LBFGS: { lb[ 'nb_steps' ]:3d} steps ( { lb[ 'nb_evals' ] } evals ), loss -> { lb[ 'loss_after' ]:.6f} ( { lb[ 'reason' ] } )" )
    # the steps L-BFGS needs to reach the final loss of gradient descent ( it then goes on, lower )
    reach = int( np.argmax( lb[ "losses" ] <= gd[ "loss_after" ] ) )
    print( f"  LBFGS reaches the GD loss in { reach } steps" )
    assert 0 < reach < gd[ "nb_steps" ] / 2
    assert lb[ "loss_after" ] <= gd[ "loss_after" ] * 1.01


if test( "lbfgs_quality" ):
    center, radius = np.array( [ 0.5, 0.1 ] ), 0.8
    sino = _disk_sinogram( nb_angles = 16, center = center, radius = radius )
    rec = Reconstruction2d( sino, extent = 4.0, max_iter = 150, ftol = 1e-9 ).random_points( 100, seed = 123 )
    l0 = rec.loss()
    l = rec.diracs().loss()
    print( f"\n  LBFGS quality: loss { l0:.6f} -> { l:.6f} ( ratio { l / l0:.4f} )" )
    assert l < l0 / 20
    pn = rec.positions
    assert np.allclose( pn.mean( axis = 0 ), center, atol = 0.15 )
    assert ( np.linalg.norm( pn - center, axis = 1 ) <= radius + 0.15 ).mean() > 0.85


if test( "multiscale_refines" ):
    sino = _disk_sinogram( nb_angles = 12 )
    rec = Reconstruction2d( sino, extent = 5.0, max_iter = 40, ftol = 1e-10 )
    rec.multiscale( nb_points_final = 300, nb_points_init = 50, factor = 4 )
    assert rec.nb_points == 300
    assert [ h[ "nb_points" ] for h in rec.history ] == [ 50, 200, 300 ]
    assert rec.history[ -1 ][ "loss_after" ] < rec.history[ 0 ][ "loss_before" ] / 10
    dist = np.linalg.norm( rec.positions - np.array( [ 0.3, -0.2 ] ), axis = 1 )
    assert ( dist <= 1.15 ).mean() > 0.9


if test( "chain_diracs_then_domes" ):
    # diracs first ( no notion of radius ), then the SAME cloud as dome centers: the centers of the
    # five domes of the data are recovered
    radius = 0.4
    sino = _domes_sinogram( radius )
    rec = Reconstruction2d( sino, radius = radius, extent = 3.0, record_every = 5, dtype = np.float64 )
    rec.random_points( len( TRUTH ), seed = 7 )
    rec.diracs( max_iter = 100, ftol = 1e-12 )
    assert rec.radii is None
    l_diracs = rec.loss( "domes" )
    rec.domes( max_iter = 300, ftol = 1e-14 )
    l_domes = rec.history[ -1 ][ "loss_after" ]
    print( f"\n  domes loss { l_diracs:.8f} -> { l_domes:.8f}" )
    assert l_domes < l_diracs / 100, ( l_diracs, l_domes )
    found = rec.positions
    err = [ float( np.min( np.linalg.norm( found - c, axis = 1 ) ) ) for c in TRUTH ]
    assert max( err ) < 0.01, f"centers poorly recovered, errors { err }"
    assert rec.radii == radius
    assert [ h[ "model" ] for h in rec.history ] == [ "diracs", "domes" ]


if test( "min_iter_forces_steps" ):
    # restarted from a converged cloud, L-BFGS stops at once; `min_iter` forces more steps
    radius = 0.4
    sino = _domes_sinogram( radius, nb_angles = 16, nb_bins = 96 )
    rec = Reconstruction2d( sino, radius = radius, extent = 3.0, dtype = np.float64 ).random_points( len( TRUTH ), seed = 11 )
    rec.diracs( max_iter = 100, ftol = 1e-12 ).domes( max_iter = 100, ftol = 1e-8 )
    plain = Reconstruction2d( sino, rec.points, radius = radius, dtype = np.float64 ).domes( max_iter = 50, ftol = 1e-8 )
    forced = Reconstruction2d( sino, rec.points, radius = radius, dtype = np.float64 ).domes( max_iter = 50, ftol = 1e-8, min_iter = 10 )
    n_plain, n_forced = plain.history[ -1 ][ "nb_steps" ], forced.history[ -1 ][ "nb_steps" ]
    print( f"\n  without min_iter: { n_plain } steps -- with min_iter=10: { n_forced } steps" )
    assert n_forced > n_plain and n_forced >= 10


if test( "disp_tol_stops_once_static" ):
    # `disp_tol` takes over AFTER `min_iter`: a huge threshold stops at the first step after it
    radius = 0.4
    sino = _domes_sinogram( radius, nb_angles = 16, nb_bins = 96 )
    rec = Reconstruction2d( sino, radius = radius, extent = 3.0 ).random_points( len( TRUTH ), seed = 11 )
    rec.diracs( max_iter = 100, ftol = 1e-12 )
    rec.domes( max_iter = 50, ftol = 1e-8, min_iter = 5, disp_tol = 1e10 )
    assert rec.history[ -1 ][ "nb_steps" ] == 6, rec.history[ -1 ][ "nb_steps" ]


if test( "a batch of reconstructions" ):
    # three sinograms of the same geometry ( three disks ), reconstructed together: each cloud ends in its disk
    centers = np.array( [ [ 0.3, -0.2 ], [ -0.6, 0.5 ], [ 0.8, 0.7 ] ] )
    sinos = [ _disk_sinogram( nb_angles = 12, center = c, radius = 0.7 ) for c in centers ]
    rec = Reconstruction2d( sinos, extent = 4.0, max_iter = 150 ).random_points( 100, seed = 5 )
    assert rec.points.shape == ( 3, 100, 2 )
    rec.diracs()
    for q, c in enumerate( centers ):
        d = np.linalg.norm( rec.positions[ q ] - c, axis = 1 )
        assert ( d <= 0.8 ).mean() > 0.9, ( q, ( d <= 0.8 ).mean() )


if test( "exact tents: the one-pass gradient against finite differences" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    rng = np.random.default_rng( 0 )
    sino = Sinogram( nb_angles = 12, nb_bins = 64, extent = 2.0 )
    for c, r in zip( rng.uniform( -0.5, 0.5, ( 3, 2 ) ), rng.uniform( 0.1, 0.3, 3 ) ):
        sino.add_disk( center = c, radius = r )
    rec = Reconstruction2d( sino, extent = 1.6, dtype = np.float64 ).random_points( 300, seed = 1 )
    f, vg = rec.loss_function( "tents", 0.5 * sino.dw )
    x = rec.points
    v, g = vg( x )
    assert abs( float( v ) - float( f( x ) ) ) < 1e-12 * float( v )
    eps, d = 1e-6, np.random.default_rng( 2 ).normal( size = x.shape )
    fd = ( float( f( x + eps * d ) ) - float( f( x - eps * d ) ) ) / ( 2 * eps )
    an = float( ( np.asarray( g ) * d ).sum() )
    assert abs( fd - an ) < 1e-6 * abs( an ), ( fd, an )


if test( "tents on the grid: close to the exact tents' loss, the gradient against finite differences" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    rng = np.random.default_rng( 0 )
    sino = Sinogram( nb_angles = 12, nb_bins = 64, extent = 2.0 )
    for c, r in zip( rng.uniform( -0.5, 0.5, ( 3, 2 ) ), rng.uniform( 0.1, 0.3, 3 ) ):
        sino.add_disk( center = c, radius = r )
    rec = Reconstruction2d( sino, extent = 1.6, dtype = np.float64, scale = 32 ).random_points( 300, seed = 1 )
    x = rec.points
    fe, _ = rec.loss_function( "tents", 0.5 * sino.dw )
    fg, vg = rec.loss_function( "tents_grid", 0.5 * sino.dw )
    assert abs( float( fg( x ) ) - float( fe( x ) ) ) < 1e-2 * float( fe( x ) ), ( float( fg( x ) ), float( fe( x ) ) )
    v, g = vg( x )
    eps, d = 1e-7, np.random.default_rng( 2 ).normal( size = x.shape )
    fd = ( float( fg( x + eps * d ) ) - float( fg( x - eps * d ) ) ) / ( 2 * eps )
    an = float( ( np.asarray( g ) * d ).sum() )
    assert abs( fd - an ) < 1e-5 * abs( an ), ( fd, an )
