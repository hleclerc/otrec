"""3D reconstruction with DIRACS ( `models.ProjectedDiracModel` ): the unknown is a 3D point
cloud whose projections onto the detector, at each angle, must reproduce the measured
RADIOGRAPHS ( `Radiographs` ) -- the counterpart of `test_reconstruction.py` with, at each
angle, a 2D semi-discrete transport ( `SdotPlanNd` ) in place of the 1D transport.

The radiographs are those of a few balls ( `Radiographs.add_sphere` ), whose ground truth is
therefore known.
"""
import numpy as np

from otrec.Radiographs import Radiographs
from otrec.Reconstruction import Reconstruction
from sdot import set_kernel_dtype
from errand import Param, experiment, test
from loom.testing import need

set_kernel_dtype( "FP64" )

CENTERS = np.array( [ [ 0.5, -0.3, 0.2 ], [ -0.6, 0.4, -0.5 ], [ 0.1, 0.5, 0.6 ] ] )
RADIUS = 0.4


def _sphere_radiographs( nb_angles = 3, nb_pixels = 40, extent = 4.0 ):
    r = Radiographs( nb_angles = nb_angles, nb_u = nb_pixels, nb_v = nb_pixels, extent_u = extent )
    for c in CENTERS:
        r.add_sphere( c, RADIUS )
    return r


def _in_spheres( points, margin = 0.1 ):
    """the fraction of the points within `RADIUS + margin` of a center"""
    d = np.linalg.norm( points[ :, None, : ] - CENTERS[ None ], axis = 2 ).min( axis = 1 )
    return float( ( d < RADIUS + margin ).mean() )


if test( "points_in_the_spheres_give_a_small_loss" ):
    need( "cpu" )
    # diracs sampling the balls reproduce their radiographs: the cost must be
    # small compared with that of a uniform cloud
    radio = _sphere_radiographs()
    rng = np.random.default_rng( 0 )
    pts = []
    while len( pts ) < 45:
        c = CENTERS[ len( pts ) % 3 ]
        q = rng.uniform( -RADIUS, RADIUS, 3 )
        if q @ q < RADIUS ** 2:
            pts.append( c + q )
    pts = np.array( pts )

    inside = Reconstruction( radio, pts ).loss()
    uniform = Reconstruction( radio ).random_points( 45, seed = 1 ).loss()
    assert np.isfinite( inside ) and inside < uniform / 5, ( inside, uniform )


if test( "multiscale_refines_up_to_the_requested_count_in_3d" ):
    need( "cpu" )
    # by stages: 40 -> 160 -> 320 diracs ( the last one subsampled ), each stage
    # restarting from the converged cloud of the previous one ( `Reconstruction.multiscale`, the
    # visual hull as starting point ); the 3D model follows the size change ( its warm weights are
    # dropped at each stage ) and the final loss is small
    radio = _sphere_radiographs()
    rec = Reconstruction( radio )
    stages = []
    rec.multiscale( 320, nb_points_init = 40, factor = 4, max_iter = 8,
                    stage_callback = lambda stage, n, pts: stages.append( n ) )
    assert stages == [ 40, 160, 320 ], stages
    assert rec.nb_points == 320
    assert [ h[ "nb_points" ] for h in rec.history ] == stages
    assert rec.history[ -1 ][ "loss_after" ] < rec.history[ 0 ][ "loss_before" ] / 3
    assert _in_spheres( rec.positions ) > 0.85, _in_spheres( rec.positions )


if test( "blur_annealing_reconstructs_from_the_cube" ):
    need( "cpu" )
    # from a cloud drawn in the whole CUBE, the projections blurred first
    # ( `Reconstruction.anneal_blur` ) then tightened bring the diracs into the balls. The blur
    # is no longer there to make the TRANSPORT possible -- `SdotPlanNd`'s width continuation takes
    # care of that, provided the target is > 0 everywhere -- but to soften the landscape that
    # L-BFGS descends. This test checks the chaining of the stages, not their necessity
    # ( measured as weak: `notes/2026-09-23-otrec-3d.md` )
    radio = _sphere_radiographs()
    rec = Reconstruction( radio ).random_points( 60, seed = 1 )
    assert _in_spheres( rec.positions ) < 0.5
    sigmas = []
    rec.anneal_blur( blurs = ( 1.0, 0.25, 0.06, 0.0 ), max_iter = 10,
                     stage_callback = lambda stage, sigma, pts: sigmas.append( sigma ) )
    assert sigmas == [ 1.0, 0.25, 0.06, 0.0 ]
    assert [ h[ "label" ] for h in rec.history ] == [ "diracs 3D blur 1", "diracs 3D blur 0.25", "diracs 3D blur 0.06", "diracs 3D blur 0" ]
    assert _in_spheres( rec.positions ) > 0.8, _in_spheres( rec.positions )
    assert rec.sinogram is radio                      # the sharp data is restored


if test( "reconstruct_converges_in_3d" ):
    need( "cpu" )
    # starting from a uniform cloud, the descent ( L-BFGS on the fused cost + gradient, see
    # `ProjectedDiracModel.value_and_grad` ) must make the cost drop and bring the diracs INTO the
    # balls
    radio = _sphere_radiographs()
    rec = Reconstruction( radio ).random_points( 45, seed = 1 )
    l0 = rec.loss()
    assert _in_spheres( rec.positions ) < 0.5

    rec.diracs( max_iter = 20 )
    l1 = rec.loss()

    assert l1 < l0 / 3, ( l0, l1 )
    assert _in_spheres( rec.positions ) > 0.8, _in_spheres( rec.positions )

    ( h, ) = rec.history
    assert h[ "model" ] == "diracs 3D" and h[ "nb_points" ] == 45


# -- what we LOOK AT -------------------------------------------------------------------------
#
#   ./run experiment test_reconstruction_3d                                  # all
#   ./run experiment "test_reconstruction_3d::rec 3D spheres" --nb-points=20000 --nb-angles=6
#   ./run experiment "test_reconstruction_3d::rec 3D random spheres" --nb-spheres=12
#
# The 3D descent, one step per frame: the HTML page of sdot's `Visualizer` ( the ground-truth
# balls in transparency, the diracs settling into them, the slider to replay ), and the
# SAME scene for ParaView -- a `.pvd` that gathers one `.vtu` per step, to open as is, the
# radius of each point as cell data ( `Glyph`, spheres, `Scale Array = radius` ). Plus the
# convergence curve. On this machine ( 16 loaded cores ), `OMP_NUM_THREADS=4` divides the cost of
# a small kernel by ten -- see `notes/2026-09-14-reconstruction-3d.md`.
#
# WHAT THE SOLVER DOES NOW ( `SdotPlanNd` entirely in C++, see
# `notes/2026-09-22-otplan-cpp.md` ), and what these experiments display at each frame
# ( `ProjectedDiracModel.solver_line()` ):
#   - the Newton step of a ( 2D ) projection comes from the LIMITS, hence without backtracking;
#   - a start that empties cells is no longer a failure: the C++ chooses between the warm weights,
#     the Voronoi and the similarity, and the WIDTH CONTINUATION ( the convolved density, tightened
#     step by step ) takes over the cases where the direct Newton stagnated;
#   - what it does NOT replace: the BACKGROUND ( `--background` ). It softens the path, not the
#     target -- its last step is the data itself, zeros included, and a cell that only sees
#     zeros has no weight that gives it its mass. Measured from the cube ( 200 diracs,
#     6 angles, 64 x 64, `notes/2026-09-23-otrec-3d.md` ): without background, 36 fits out of 48
#     do not converge and the loss GOES UP; with 1e-6, all converge in 22 diagrams and 98 % of
#     the diracs end up in the balls;
#   - the warm weights from one evaluation to the next keep the number of steps to a handful as
#     soon as the cloud no longer moves much -- what the solver line shows plainly ( continuation
#     steps and Voronoi starts = the cloud is still far ).
# The price stays there: one fit per angle and per evaluation, one evaluation per L-BFGS
# step -- count minutes per step at 20,000 diracs and 6 angles, the outputs being rewritten
# every ten steps.

def _random_spheres( nb_spheres, seed, extent = 2.2 ):
    """`nb_spheres` balls of varied radii, DISJOINT, in the cube `[ -extent/2, extent/2 ]^3`"""
    rng = np.random.default_rng( seed )
    centers, radii = [], []
    while len( centers ) < nb_spheres:
        r = rng.uniform( 0.15, 0.45 )
        c = rng.uniform( -extent / 2 + r, extent / 2 - r, 3 )
        if all( np.linalg.norm( c - c2 ) > r + r2 + 0.05 for c2, r2 in zip( centers, radii ) ):
            centers.append( c ); radii.append( r )
    return np.array( centers ), np.array( radii )


def _run_3d( p, centers, radii, stem, multiscale = None, blurs = None ):
    """Common to the experiments: the radiographs of the balls, the recorded descent, the outputs.

    `multiscale = ( nb_points_init, factor )`: the staged descent ( `Reconstruction.multiscale` )
    instead of a cloud drawn at once at `nb_points`. `blurs`: first the BLURRED projections,
    tightened stage by stage ( `Reconstruction.anneal_blur` ), on the starting cloud -- then the
    staged refinement if requested, on the sharp data."""
    import time
    from sdot import Visualizer, write_convergence_html

    radio = Radiographs( nb_angles = p.nb_angles, nb_u = p.nb_pixels, nb_v = p.nb_pixels, extent_u = 4.0 )
    for c, r in zip( centers, radii ):
        radio.add_sphere( c, r )

    rec = Reconstruction( radio, verbose = True, seed = p.seed )
    n0 = multiscale[ 0 ] if multiscale else p.nb_points
    if p.init == "hull":
        rec.hull_points( n0, seed = p.seed )
    else:
        rec.random_points( n0, seed = p.seed )

    def in_spheres( pts, margin = 0.05 ):
        d = np.linalg.norm( pts[ :, None, : ] - centers[ None ], axis = 2 ) - radii[ None, : ]
        return float( ( d.min( axis = 1 ) < margin ).mean() )

    viz = Visualizer( title = f"3D reconstruction -- { p.nb_points } diracs, { p.nb_angles } angles" )
    fractions, times, counts, t0 = [], [], [], time.perf_counter()
    model_kwargs = dict( background = p.background, kernel_dtype = p.kernel, continuation = p.continuation )
    model = rec.dirac_model( **model_kwargs )

    def write_outputs():
        viz.write_html( p.out_dir / f"{ stem }.html" )
        viz.write_vtk( p.out_dir / f"{ stem }.vtk" )
        write_convergence_html(
            { "diracs in the balls ( fraction )": list( zip( times, fractions ) ),
              "number of diracs / final number": list( zip( times, [ c / p.nb_points for c in counts ] ) ) },
            p.out_dir / f"{ stem }_convergence.html",
            title = f"3D reconstruction -- { p.nb_points } diracs", xlabel = "time ( s )", ylabel = "fraction", log_y = False )

    def snap( step, pts ):
        pts = np.asarray( pts )
        if step >= 0 and ( step + 1 ) % p.record_every:
            return
        if fractions:                                     # the very first frame already exists
            viz.new_frame( len( fractions ) )
        viz.add_points( pts, radius = 0.01, color = "#e0a030" )
        for c, r in zip( centers, radii ):
            viz.add_points( c[ None ], radius = r, color = "#4080c0", opacity = 0.2 )
        fractions.append( in_spheres( pts ) )
        counts.append( len( pts ) )
        times.append( time.perf_counter() - t0 )
        print( f"  frame { len( fractions ) - 1 } ( step { step + 1 }, { len( pts ) } diracs ), { times[ -1 ]:.0f} s, "
               f"{ fractions[ -1 ] * 100:.1f} % in the balls", flush = True )
        # what the per-angle fits have cost since the start ( the current model, which
        # `anneal_blur` / `multiscale` build themselves ) -- see the header of this file
        line = getattr( rec.model, "solver_line", None )
        if line is not None:
            print( f"    { line() }", flush = True )
        # a long descent is written ALONG THE WAY: what is done can already be looked at
        if len( fractions ) % 10 == 0:
            write_outputs()

    if blurs:
        rec.anneal_blur( blurs = blurs, max_iter = p.max_iter, callback = snap,
                         model_kwargs = model_kwargs )
    if multiscale:
        # by stages: each stage stops when scipy no longer progresses ( `ftol` ) or at
        # `max_iter` -- the "quasi-convergence" that suffices before refining. After the blur, the
        # starting cloud is ALREADY converged on the sharp data: we refine right away.
        if blurs and blurs[ -1 ] == 0 and rec.nb_points < p.nb_points:
            rec.split( multiscale[ 1 ] ).subsample( min( rec.nb_points, p.nb_points ) )
        rec.multiscale( p.nb_points, nb_points_init = multiscale[ 0 ], factor = multiscale[ 1 ],
                        model = model, max_iter = p.max_iter, callback = snap )
    elif not blurs:
        # `min_iter = max_iter`: all the requested steps, not a scipy stop on an `ftol` that the
        # noise of the internal fits ( `mass_tol` ) triggers too early -- this is an experiment
        rec.run( model, max_iter = p.max_iter, min_iter = p.max_iter, callback = snap, label = "diracs 3D" )

    stages = rec.history
    total = sum( h[ "time" ] for h in stages )
    print( f"  { rec.nb_points } diracs, { p.nb_angles } angles, { len( stages ) } stage(s), "
           f"{ sum( h[ 'nb_steps' ] for h in stages ) } steps in { total:.0f} s: "
           f"loss { stages[ 0 ][ 'loss_before' ]:.3e} -> { stages[ -1 ][ 'loss_after' ]:.3e}, "
           f"{ in_spheres( rec.positions ) * 100:.1f} % of the diracs in the balls" )
    for h in stages:
        print( f"    { h[ 'label' ] }: { h[ 'nb_steps' ] } steps, { h[ 'time' ]:.0f} s, loss { h[ 'loss_before' ]:.3e} -> { h[ 'loss_after' ]:.3e}" )
    p.results[ "loss_before" ], p.results[ "loss_after" ] = stages[ 0 ][ "loss_before" ], stages[ -1 ][ "loss_after" ]
    p.results[ "in_spheres" ] = in_spheres( rec.positions )
    p.results[ "time" ] = total

    write_outputs()
    np.savez( p.out_dir / f"{ stem }_final.npz", positions = rec.positions, centers = centers, radii = radii )
    return rec


_PARAMS = dict(
    nb_points    = Param( 10000, help = "number of diracs" ),
    nb_angles    = Param( 100, help = "number of projection angles" ),
    nb_pixels    = Param( 128, help = "pixels per side of the detector" ),
    max_iter     = Param( 30, help = "number of L-BFGS steps" ),
    record_every = Param( 1, help = "one frame every k steps" ),
    background   = Param( 1e-6, help = "background added to the radiographs, as a fraction of the mean -- "
                                       "ESSENTIAL ( a cell that only sees zeros has no "
                                       "weight that gives it its mass ); `0` to see for yourself" ),
    continuation = Param( "auto", help = "`SdotPlanNd`'s width continuation: auto, always, never" ),
    init         = Param( "hull", help = "starting point: `hull` ( the visual hull ) or `cube`" ),
    kernel       = Param( "FP64", help = "the kernel float ( FP64: what the damping requires; "
                                         "FP32 to see what it costs )" ),
    seed         = Param( 1, help = "seed of the draw" ),
)

if p := experiment( "rec 3D spheres", **_PARAMS ):
    # the three balls of the tests, at large scale
    _run_3d( p, CENTERS, np.full( len( CENTERS ), RADIUS ), "rec_3d_spheres" )

if p := experiment( "rec 3D random spheres", nb_spheres = Param( 8, help = "number of balls" ), **_PARAMS ):
    # a less symmetric phantom: balls of varied radii, drawn at random
    centers, radii = _random_spheres( p.nb_spheres, p.seed + 100 )
    _run_3d( p, centers, radii, "rec_3d_random_spheres" )

if p := experiment( "rec 3D blur",
                    blurs          = Param( "1,0.25,0.06,0.015,0", help = "the blurs, as a fraction of the detector width "
                                                                          "( `0` alone: the sharp data, for comparison )" ),
                    nb_spheres     = Param( 8, help = "number of balls" ),
                    **{ **_PARAMS, "init": Param( "cube", help = "starting point: `cube` ( the whole cube ) or `hull`" ),
                        "max_iter": Param( 20, help = "at most this many L-BFGS steps PER STAGE" ) } ):
    # the BLURRED projections first ( `Reconstruction.anneal_blur` ), from a cloud drawn in
    # the whole cube. What changed: the blur is no longer what makes the TRANSPORT possible --
    # `SdotPlanNd`'s width continuation takes care of that, provided the background is > 0. The blur
    # is left with its other role, softening the landscape that L-BFGS descends, and it is little:
    # at 200 diracs from the cube, the sharp data ( background 1e-6 ) ends with 98 % of the diracs
    # in the balls against 99 % after four blur stages, for a comparable time
    # ( `notes/2026-09-23-otrec-3d.md` ). `--blurs=0` runs the same experiment without blur: it is
    # the comparison to redo on a harder phantom before dispensing with these stages.
    centers, radii = _random_spheres( p.nb_spheres, p.seed + 100 )
    _run_3d( p, centers, radii, "rec_3d_blur", blurs = [ float( b ) for b in str( p.blurs ).split( "," ) ] )

if p := experiment( "rec 3D blur multiscale",
                    blurs          = Param( "1,0.25,0.06,0.015,0", help = "the blurs, as a fraction of the detector width" ),
                    nb_points_init = Param( 500, help = "diracs of the blurred stages and of the first refinement" ),
                    factor         = Param( 4, help = "children per dirac at each refinement" ),
                    nb_spheres     = Param( 8, help = "number of balls" ),
                    **{ **_PARAMS, "init": Param( "cube", help = "starting point: `cube` or `hull`" ),
                        "nb_points": Param( 32000, help = "FINAL number of diracs" ),
                        "max_iter": Param( 20, help = "at most this many L-BFGS steps PER STAGE" ) } ):
    # both: the blur tightened on a small cloud, then the staged refinement on the sharp data
    centers, radii = _random_spheres( p.nb_spheres, p.seed + 100 )
    _run_3d( p, centers, radii, "rec_3d_blur_multiscale", multiscale = ( p.nb_points_init, p.factor ),
             blurs = [ float( b ) for b in str( p.blurs ).split( "," ) ] )

if p := experiment( "rec 3D multiscale",
                    nb_points_init = Param( 500, help = "diracs of the first stage" ),
                    factor         = Param( 4, help = "children per dirac at each refinement" ),
                    nb_spheres     = Param( 8, help = "number of balls" ),
                    **{ **_PARAMS, "nb_points": Param( 32000, help = "FINAL number of diracs" ),
                        "max_iter": Param( 30, help = "at most this many L-BFGS steps PER STAGE" ) } ):
    # BY STAGES ( `Reconstruction.multiscale` ): few diracs first, converged, then each one
    # replaced by `factor` noisy children, reconverged -- up to the requested number. Each stage
    # starts from an already well-placed cloud, so its per-angle transports start near their
    # solution: this is what makes a large cloud tractable, whereas drawing it at once makes every
    # fit restart from the Voronoi ( see the note ). One frame per step, all stages combined --
    # the number of diracs changes from one stage to the next, the curve shows it.
    centers, radii = _random_spheres( p.nb_spheres, p.seed + 100 )
    _run_3d( p, centers, radii, "rec_3d_multiscale", multiscale = ( p.nb_points_init, p.factor ) )
