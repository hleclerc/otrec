"""CT reconstruction with DIRACS (`models.DiracModel`): the unknown is a point cloud whose
projections must reproduce the measured sinogram.

Everything goes through the `Reconstruction` class, which holds the sinogram, the current cloud
and the default parameters, and chains the steps (see `test_chain_diracs_then_disks` for the
diracs -> disks composition, and `test_disks.py` for the disks model alone).
"""
import numpy as np

from otrec.Sinogram import Sinogram
from otrec.Reconstruction import Reconstruction
from otrec.optimizers import GradientDescent, LBFGS
from errand import test
from loom.testing import need


def _disk_sinogram( nb_angles = 8, nb_bins = 201, extent = 6.0, center = ( 0.3, -0.2 ), radius = 1.0 ):
    s = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    s.add_disk( center = list( center ), radius = radius )
    return s


if test( "in_disk_gives_small_loss" ):
    # diracs sampling the target disk reproduce its projections: the loss must be
    # small. Also exercises the OT path on ZERO-density profiles (detector tails).
    sino = _disk_sinogram()

    rng = np.random.default_rng( 0 )
    pts = []
    while len( pts ) < 400:
        xy = ( rng.random( 2 ) - 0.5 ) * 2
        if xy[ 0 ] ** 2 + xy[ 1 ] ** 2 < 1:
            pts.append( xy + np.array( [ 0.3, -0.2 ] ) )
    pts = np.array( pts )

    l = Reconstruction( sino, pts ).loss()
    assert np.isfinite( l )
    assert l < 0.05, f"loss too large for diracs in the disk : { l }"


if test( "reconstruct_converges" ):
    need( "grad" )
    # starting from random diracs, the descent must make the loss drop and bring the diracs
    # INTO the disk (mean position ~ center).
    center, radius = np.array( [ 0.3, -0.2 ] ), 1.0
    sino = _disk_sinogram( nb_angles = 8, center = center, radius = radius )

    nb_steps = 100
    rec = Reconstruction( sino, extent = 5.0 ).random_points( 100, seed = 3 )
    l0 = rec.loss()

    # we capture ~10 snapshots along the descent, for the animation
    frames = [ rec.positions ]
    every = max( 1, nb_steps // 10 )
    def snap( step, pts ):
        if step % every == every - 1 or step == nb_steps - 1:
            frames.append( np.asarray( pts ) )

    rec.diracs( optimizer = GradientDescent( lr = 0.5, nb_steps = nb_steps ), callback = snap )
    l1 = rec.loss()

    assert l1 < l0 / 10, f"the loss did not drop enough : { l0 } -> { l1 }"

    pn = rec.positions
    assert np.allclose( pn.mean( axis = 0 ), center, atol = 0.1 )
    dist = np.linalg.norm( pn - center, axis = 1 )
    assert ( dist <= radius + 0.1 ).mean() > 0.9, "most diracs should be in the disk"

    # the history keeps one line per step played, with the end-to-end losses
    ( h, ) = rec.history
    assert h[ "model" ] == "diracs" and h[ "nb_points" ] == 100
    assert abs( h[ "loss_before" ] - l0 ) < 1e-12 and abs( h[ "loss_after" ] - l1 ) < 1e-12


if test( "lbfgs_vs_gradient_descent" ):
    need( "grad" )
    # Compare the convergence of LBFGS and gradient descent on the same problem.
    # LBFGS should converge faster (fewer iterations) and reach a comparable or better loss.
    center, radius = np.array( [ 0.3, -0.2 ] ), 1.0
    sino = _disk_sinogram( nb_angles = 12, center = center, radius = radius )

    start = Reconstruction( sino, extent = 5.0 ).random_points( 80, seed = 42 ).points
    l0 = Reconstruction( sino, start ).loss()

    # same starting point, two optimizers: two independent `Reconstruction`s
    gd_losses, lbfgs_losses = [], []
    rec_gd = Reconstruction( sino, start )
    rec_gd.diracs( optimizer = GradientDescent( lr = 0.2, nb_steps = 200 ),
                   callback = lambda step, pts: gd_losses.append( rec_gd.loss( points = pts ) ) )
    l_gd = gd_losses[ -1 ]

    rec_lbfgs = Reconstruction( sino, start )
    rec_lbfgs.diracs( optimizer = LBFGS( max_iter = 200, ftol = 1e-8 ),
                      callback = lambda step, pts: lbfgs_losses.append( rec_lbfgs.loss( points = pts ) ) )
    l_lbfgs = lbfgs_losses[ -1 ]

    print( f"\n  GD:    {len(gd_losses):3d} steps, loss {l0:.6f} -> {l_gd:.6f} (ratio {l_gd/l0:.4f})" )
    print( f"  LBFGS: {len(lbfgs_losses):3d} steps, loss {l0:.6f} -> {l_lbfgs:.6f} (ratio {l_lbfgs/l0:.4f})" )
    print( f"  Speedup: {len(gd_losses)/len(lbfgs_losses):.1f}x fewer iterations with LBFGS" )

    # LBFGS should converge in far fewer iterations
    assert len( lbfgs_losses ) < len( gd_losses ) / 2, \
        f"LBFGS should converge faster: {len(lbfgs_losses)} vs {len(gd_losses)}"

    # LBFGS should reach a similar or better loss
    assert l_lbfgs <= l_gd * 1.01, \
        f"LBFGS should reach similar or better loss: {l_lbfgs} vs {l_gd}"


if test( "lbfgs_quality" ):
    need( "grad" )
    # LBFGS alone: check that it reaches a good reconstruction quality. `max_iter`/`ftol`
    # passed at construction define the default L-BFGS of all the steps.
    center, radius = np.array( [ 0.5, 0.1 ] ), 0.8
    sino = _disk_sinogram( nb_angles = 16, center = center, radius = radius )

    rec = Reconstruction( sino, extent = 4.0, max_iter = 150, ftol = 1e-9 ).random_points( 100, seed = 123 )
    l0 = rec.loss()
    l = rec.diracs().loss()

    print( f"\n  LBFGS quality: loss {l0:.6f} -> {l:.6f} (ratio {l/l0:.4f})" )

    # Check sufficient convergence
    assert l < l0 / 20, f"LBFGS should reduce loss significantly: {l0} -> {l}"

    # Check that the diracs are well positioned
    pn = rec.positions
    assert np.allclose( pn.mean( axis = 0 ), center, atol = 0.15 ), "diracs should cluster around center"
    dist = np.linalg.norm( pn - center, axis = 1 )
    assert ( dist <= radius + 0.15 ).mean() > 0.85, "most diracs should be in the disk"


if test( "multiscale_refines" ):
    need( "grad" )
    # coarse -> fine: each stage starts from the structure found by the previous one (`split`), and
    # the history must show an increasing number of points, up to the target.
    sino = _disk_sinogram( nb_angles = 12, center = ( 0.3, -0.2 ), radius = 1.0 )

    rec = Reconstruction( sino, extent = 5.0, max_iter = 40, ftol = 1e-10 )
    rec.multiscale( nb_points_final = 300, nb_points_init = 50, factor = 4 )

    assert rec.nb_points == 300, rec.nb_points
    counts = [ h[ "nb_points" ] for h in rec.history ]
    assert counts == [ 50, 200, 300 ], counts
    assert rec.history[ -1 ][ "loss_after" ] < rec.history[ 0 ][ "loss_before" ] / 10

    pn = rec.positions
    dist = np.linalg.norm( pn - np.array( [ 0.3, -0.2 ] ), axis = 1 )
    assert ( dist <= 1.15 ).mean() > 0.9, "most diracs should be in the disk"


if test( "chain_diracs_then_disks" ):
    need( "grad" )
    # THE use case of chaining: a dirac reconstruction (non-smooth loss) serves as the starting
    # point of a DISK reconstruction (smooth loss), on the SAME point cloud -- each converged
    # dirac becomes a disk center.
    radius = 0.4
    truth = np.array( [ [ 0.8, 0.3 ], [ -0.9, 0.6 ], [ 0.1, -1.1 ], [ -0.5, -0.7 ], [ 1.3, -0.4 ] ] )
    sino = Sinogram( nb_angles = 24, nb_bins = 128, extent = 6.0 )
    for c in truth:
        sino.add_disk( center = list( c ), radius = radius )

    rec = Reconstruction( sino, radius = radius, nb_pixels = 256, extent = 3.0, record = True )
    rec.random_points( len( truth ), seed = 7 )

    # step 1: diracs. The points migrate towards the mass, with no notion of radius.
    rec.diracs( max_iter = 100, ftol = 1e-12 )
    assert rec.radii is None, "the diracs model has no radius of its own"
    l_diracs = rec.loss( rec.disk_model() )         # same cloud, measured with the disks loss

    # step 2: disks, restarting EXACTLY from the previous cloud.
    rec.disks( max_iter = 300, ftol = 1e-14 )
    l_disks = rec.history[ -1 ][ "loss_after" ]
    floor = rec.floor( rec.disk_model() )

    print( f"\n  disks loss { l_diracs:.8f} -> { l_disks:.8f} (floor { floor:.8f})" )
    assert l_disks < l_diracs, f"the disks step must improve the disks loss : { l_diracs } -> { l_disks }"
    assert l_disks < 1.05 * floor, f"final loss { l_disks } above the floor { floor }"

    # the recovered centers, up to matching
    found = rec.positions
    err = [ float( np.min( np.linalg.norm( found - c, axis = 1 ) ) ) for c in truth ]
    assert max( err ) < 0.02, f"centers poorly recovered, errors { err }"

    # the trajectory is CONTINUOUS from one step to the next: a single junction frame, and the
    # exported radius is that of the last step played.
    assert rec.radii == radius
    assert len( rec.frames ) == sum( h[ "nb_steps" ] for h in rec.history ) + 1
    assert [ h[ "model" ] for h in rec.history ] == [ "diracs", "disks" ]


if test( "disks_min_iter_forces_steps" ):
    need( "grad" )
    # `models.DiskModel` has FLAT directions (a disk can slide without changing the residual
    # as long as it overlaps nobody): restarted from an ALREADY converged cloud, L-BFGS-B (native
    # scipy ftol) must stop almost immediately -- `min_iter` must force a minimum number of steps
    # anyway (see the docstring of `LBFGS`).
    radius = 0.4
    truth = np.array( [ [ 0.8, 0.3 ], [ -0.9, 0.6 ], [ 0.1, -1.1 ], [ -0.5, -0.7 ], [ 1.3, -0.4 ] ] )
    sino = Sinogram( nb_angles = 16, nb_bins = 96, extent = 6.0 )
    for c in truth:
        sino.add_disk( center = list( c ), radius = radius )

    rec = Reconstruction( sino, radius = radius, nb_pixels = 128, extent = 3.0 )
    rec.random_points( len( truth ), seed = 11 )
    rec.diracs( max_iter = 100, ftol = 1e-12 )
    rec.disks( max_iter = 100, ftol = 1e-8 )                         # converge a first time

    # restart EXACTLY from the converged cloud, with/without `min_iter`.
    nb_steps_plain = Reconstruction(
        sino, rec.points, radius = radius, nb_pixels = 128, extent = 3.0,
    ).disks( max_iter = 50, ftol = 1e-8 ).history[ -1 ][ "nb_steps" ]

    nb_steps_forced = Reconstruction(
        sino, rec.points, radius = radius, nb_pixels = 128, extent = 3.0,
    ).disks( max_iter = 50, ftol = 1e-8, min_iter = 10 ).history[ -1 ][ "nb_steps" ]

    print( f"\n  without min_iter: { nb_steps_plain } steps -- with min_iter=10: { nb_steps_forced } steps" )
    # NO `>= min_iter` assertion: `min_iter` is a "best effort", not a guarantee. `ftol=0`
    # does not forbid an early stop -- it conditions it on an EXACTLY zero reduction (the
    # L-BFGS-B `factr` test), which a perfectly flat direction achieves, and the DISKS model is
    # full of them. Observed here: scipy stops at 9 steps on
    # `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`, and restarted from that point it
    # takes NO step at all -- it is a truly stationary point for the line search.
    # What is really tested is therefore the effect of `min_iter`, not its exact count.
    assert nb_steps_forced > nb_steps_plain, \
        f"min_iter should force strictly more steps than the native scipy behavior : " \
        f"{ nb_steps_forced } vs { nb_steps_plain }"


if test( "disks_disp_tol_stops_once_static" ):
    need( "grad" )
    # `disp_tol` takes over AFTER `min_iter`: a huge displacement threshold must stop
    # at the first step following `min_iter`, whatever `ftol`/`max_iter`.
    radius = 0.4
    truth = np.array( [ [ 0.8, 0.3 ], [ -0.9, 0.6 ], [ 0.1, -1.1 ], [ -0.5, -0.7 ], [ 1.3, -0.4 ] ] )
    sino = Sinogram( nb_angles = 16, nb_bins = 96, extent = 6.0 )
    for c in truth:
        sino.add_disk( center = list( c ), radius = radius )

    rec = Reconstruction( sino, radius = radius, nb_pixels = 128, extent = 3.0 )
    rec.random_points( len( truth ), seed = 11 )
    rec.diracs( max_iter = 100, ftol = 1e-12 )
    rec.disks( max_iter = 50, ftol = 1e-8, min_iter = 5, disp_tol = 1e10 )

    nb_steps = rec.history[ -1 ][ "nb_steps" ]
    print( f"\n  min_iter=5, huge disp_tol: { nb_steps } steps" )
    assert nb_steps == 6, f"must stop at the very first step following min_iter: { nb_steps } steps"
