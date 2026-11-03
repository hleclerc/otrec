"""3D RECONSTRUCTION ON A PHANTOM: watch how the diracs move and how it converges ( `Reconstruction3d`, exact 2D transport per
view ).

    errand run --no-queue -o --env lmo-jax "otrec/src/otrec/experiments/exp_3d.py::3d"                     # a thin foam, multiscale
    errand run --no-queue -o --env lmo-jax "otrec/src/otrec/experiments/exp_3d.py::3d" --n=200000 --height=0.4
    errand run --no-queue -o --env lmo-jax "otrec/src/otrec/experiments/exp_3d.py::3d" --device=cpu        # lmo's CPUs

The default phantom is a Voronoi foam ( `phantoms.VoronoiFoam` ) in a THIN cylinder, so that the 3D view shows every wall.
`optimizer = multiscale` starts with `n_init` diracs and multiplies them by `factor` after each stage of `steps` Lloyd
steps, up to `n` ( whose stage takes `last_steps` ).

Writes in the run's directory: `run.npz` ( history, snapshots of the cloud with the disagreement of the views and the mass
derivative of each dirac, the phantom and a sample of it ), `checkpoint.npz` along the way ( `--resume` takes it ), the
pages of `viz.view3d` ( `index.html` first ), and `summary.json`.
"""
from errand import Param, experiment


if p := experiment( "3d",
                    phantom    = Param( "voronoi", choices = [ "voronoi", "foam", "balls", "hollow", "touching", "thin" ] ),
                    height     = Param( 0.2, help = "voronoi: the height of the cylinder ( radius 0.8 )" ),
                    seed_height = Param( 0.0, help = "voronoi: the height the cells are drawn for ( 0: `height` ); a thinner `height` is then a slab of the same foam" ),
                    cell       = Param( 0.19, help = "voronoi: the size of a cell ( the number of cells follows from the volume )" ),
                    porosity   = Param( 0.8, help = "voronoi: void fraction of the cylinder" ),
                    device     = Param( "default", choices = [ "default", "cpu" ], help = "cpu: the solves on the CPU even where there is a card" ),
                    n          = Param( 50000, help = "number of diracs ( at the end )" ),
                    n_init     = Param( 1000, help = "multiscale: number of diracs of the first stage" ),
                    factor     = Param( 4, help = "multiscale: the diracs are multiplied by that after each stage" ),
                    views      = Param( 60, help = "number of views ( angles over [0, pi) )" ),
                    pixels     = Param( 256, help = "detector width, in pixels ( its height follows the object's )" ),
                    optimizer  = Param( "multiscale", choices = [ "multiscale", "lloyd", "plbfgs", "lbfgs" ], help = "plbfgs: L-BFGS preconditioned by the Lloyd step" ),
                    steps      = Param( 30, help = "optimizer steps ( per stage for multiscale )" ),
                    last_steps = Param( 60, help = "multiscale: the steps of the last stage" ),
                    stage_optimizer = Param( "lloyd", help = "multiscale: what each stage runs, lloyd, plbfgs, sliced ( tents along directions of each radiograph, L-BFGS on the device ) or pixel2d ( pyramids against the pixels as diracs ); ':'-separated, one per stage ( the last for the stages beyond )" ),
                    refine     = Param( 0, help = "pixel2d steps after everything else ( 0: none )" ),
                    pixel_scale = Param( 4, help = "pixel2d: fine cells per pixel and per axis" ),
                    pixel_background = Param( 1e-3, help = "pixel2d: uniform background on both sides, fraction of the mass" ),
                    pixel_tol  = Param( 1e-3, help = "pixel2d: mass tolerance of the solves, relative to a pixel's mean mass" ),
                    pixel_split = Param( 1, help = "pixel2d: split x split diracs per pixel ( 1 pulls z to the detector rows )" ),
                    pixel_radius = Param( 0.5, help = "pixel2d: half-width of the pyramids, in pixels ( 1: no pull to the pixels' centers to first order, the tents summing to a constant on the grid )" ),
                    pixel_blur = Param( "", help = "pixel2d: radii ( pixels ) of blurred stages before `pixel_radius`, separated by colons ( e.g. 10:5:2.5; a comma would make errand sweep them ): the large moves happen where the 2D solves are smooth" ),
                    pixel_blur_steps = Param( 30, help = "pixel2d: steps of each blurred stage" ),
                    pixel_snapshot = Param( 5, help = "pixel2d: a history entry and a frame every that many steps" ),
                    pixel_workers = Param( 0, help = "pixel2d: the views spread over that many worker processes ( 0: one per card, none on the host )" ),
                    directions = Param( 16, help = "sliced: directions per radiograph" ),
                    sliced_method = Param( "grid", choices = [ "grid", "exact" ], help = "sliced: the 1D transport of the tents" ),
                    sliced_scale = Param( 8, help = "sliced, grid: fine cells per bin" ),
                    sliced_dtype = Param( "float32", choices = [ "float32", "float64" ], help = "sliced: the tents' float" ),
                    stage_steps = Param( "", help = "multiscale: the steps of each stage, ':'-separated ( e.g. 2000:1000:500:300; overrides steps / last_steps )" ),
                    tau        = Param( 1.8, help = "lloyd: relaxation of the lifted step" ),
                    start      = Param( "hull", choices = [ "hull", "uniform" ], help = "visual hull or uniform in the cube" ),
                    background = Param( 1e-6, help = "added to the radiographs, fraction of their mean" ),
                    tol        = Param( 1e-4, help = "mass tolerance of the solves, relative to a dirac" ),
                    snapshot   = Param( 10, help = "a snapshot of the cloud every that many steps ( and at the end of each stage )" ),
                    final_steps = Param( 0, help = "Lloyd steps after the rest, with the solver below ( 0: none )" ),
                    final_precision = Param( "fp64", choices = [ "auto", "fp64", "fp32", "mixed" ], help = "the final steps' solver precision" ),
                    final_tol  = Param( 1e-8, help = "the final steps' mass tolerance, relative to a dirac" ),
                    resume     = Param( "", help = "a run.npz or checkpoint.npz to start from ( no comma in the path )" ),
                    seed       = Param( 0 ) ):
    import json
    import os
    import time
    os.environ[ "XLA_PYTHON_CLIENT_PREALLOCATE" ] = "false"     # the cards may be shared with the 2D loss's workers
    if p.device == "cpu":   # before loom and jax are imported
        from otrec.device import use_cpu
        use_cpu()

    import numpy as np

    from otrec import phantoms
    from otrec.reconstruction3d import Reconstruction3d
    from otrec.viz.view3d import write_view

    if p.phantom == "voronoi":
        sh = p.seed_height or p.height
        nb_cells = max( 8, int( round( np.pi * 0.8 ** 2 * sh / p.cell ** 3 ) ) )
        ph = phantoms.VoronoiFoam( nb_cells = nb_cells, porosity = p.porosity, height = p.height, seed = p.seed, seed_height = sh )
    else:
        ph = phantoms.PHANTOMS[ p.phantom ]()
    t0 = time.perf_counter()
    radio = ph.radiographs( nb_angles = p.views, nb_pixels = p.pixels )
    print( f"{ ph.name }: radiographs { p.views } x { radio.nb_u_host } x { radio.nb_v_host } in { time.perf_counter() - t0:.1f} s" )
    truth = ph.metrics( tol = radio.du )
    rec = Reconstruction3d( radio, background = p.background, tol = p.tol, verbose = True )
    hull = dict( extent = ph.extent )
    if p.resume:
        prev = np.load( p.resume )
        rec.set_points( prev[ "x_final" ] )
        print( f"resumed from { p.resume } ( { rec.nb_points } diracs )" )

    def checkpoint( rec ):
        """the cloud so far, in case the run is stopped ( `--resume` takes it )"""
        np.savez( p.out_dir / "checkpoint.npz", x_final = rec.positions, step = len( rec.history ) )

    t0 = time.perf_counter()
    if p.optimizer == "multiscale":
        steps = [ int( k ) for k in p.stage_steps.split( ":" ) ] if p.stage_steps else p.steps
        rec.multiscale( p.n, nb_points_init = rec.nb_points if p.resume else p.n_init, factor = p.factor, steps = steps,
                        last_steps = p.last_steps, tau = p.tau, start = p.start, snapshot_every = p.snapshot,
                        truth = truth, on_snapshot = checkpoint, seed = p.seed, optimizer = p.stage_optimizer.split( ":" ),
                        sliced = dict( nb_directions = p.directions, method = p.sliced_method, scale = p.sliced_scale,
                                       dtype = getattr( np, p.sliced_dtype ) ),
                        pixel = dict( scale = p.pixel_scale, background = p.pixel_background, tol = p.pixel_tol, split = p.pixel_split,
                                      radius = p.pixel_radius * min( radio.du, radio.dv ), cards = p.pixel_workers or None ), **hull )
    else:
        if not p.resume:
            if p.start == "hull":
                rec.visual_hull_points( p.n, seed = p.seed, **hull )
            else:
                rec.random_points( p.n, seed = p.seed )
        if p.optimizer == "lloyd":
            rec.lloyd( p.steps, tau = p.tau, snapshot_every = p.snapshot, truth = truth, on_snapshot = checkpoint )
        elif p.optimizer == "plbfgs":
            rec.plbfgs( p.steps, snapshot_every = p.snapshot, truth = truth, on_snapshot = checkpoint )
        else:
            rec.lbfgs( p.steps, snapshot_every = p.snapshot, truth = truth )
    if p.refine:
        print( f"refinement: { p.refine } pixel2d steps" )
        for r in [ float( r ) for r in str( p.pixel_blur ).split( ":" ) if r.strip() ]:
            print( f"  blurred stage: radius { r } px, { p.pixel_blur_steps } steps", flush = True )
            rec.pixel2d( p.pixel_blur_steps, scale = p.pixel_scale, background = p.pixel_background, tol = p.pixel_tol, split = p.pixel_split,
                         radius = r * min( radio.du, radio.dv ), cards = p.pixel_workers or None,
                         snapshot_every = p.pixel_snapshot, truth = truth, on_snapshot = checkpoint )
        rec.pixel2d( p.refine, scale = p.pixel_scale, background = p.pixel_background, tol = p.pixel_tol, split = p.pixel_split,
                     radius = p.pixel_radius * min( radio.du, radio.dv ), cards = p.pixel_workers or None,
                     snapshot_every = p.pixel_snapshot, truth = truth, on_snapshot = checkpoint )
    if p.final_steps:
        rec.configure( precision = p.final_precision, tol = p.final_tol, max_iter = 200 )
        print( f"final steps: precision { p.final_precision }, tol { p.final_tol }" )
        rec.lloyd( p.final_steps, tau = p.tau, snapshot_every = p.snapshot, truth = truth, on_snapshot = checkpoint )
    wall = time.perf_counter() - t0

    out = p.out_dir
    keys = [ k for k in rec.history[ 0 ] if isinstance( rec.history[ 0 ][ k ], ( int, float, np.number ) ) ]
    hist = { f"h_{ k }": np.array( [ h.get( k, np.nan ) for h in rec.history ] ) for k in keys }
    frames = rec.frames   # their sizes change from a stage to the next: stored end to end, `f_sizes` says where they cut
    np.savez_compressed( out / "run.npz",
                         f_step = np.array( [ f[ "step" ] for f in frames ] ),
                         f_sizes = np.array( [ len( f[ "points" ] ) for f in frames ] ),
                         f_points = np.concatenate( [ f[ "points" ] for f in frames ] ).astype( np.float32 ),
                         f_disagreement = np.concatenate( [ f[ "disagreement" ] for f in frames ] ).astype( np.float32 ),
                         f_mass_grad = np.concatenate( [ f[ "mass_grad" ] for f in frames ] ).astype( np.float32 ),
                         x_final = rec.positions,
                         centers = ph.centers, radii = ph.radii, densities = ph.densities, cylinders = ph.cylinders,
                         truth_sample = ph.sample( min( max( p.n, 50000 ), 200000 ), seed = 9 ).astype( np.float32 ),
                         radiograph0 = np.asarray( radio.values[ 0 ] ), bases0 = np.asarray( radio.bases[ 0 ] ),
                         detector = np.array( [ radio.u_min, radio.v_min, radio.du, radio.dv ] ),
                         cost_per_view = np.stack( [ h[ "cost_per_view" ] for h in rec.history ] ), **hist )
    last = rec.history[ -1 ]
    summary = dict( params = { k: getattr( p, k ) for k in ( "phantom", "height", "seed_height", "cell", "porosity", "n", "n_init", "factor", "views",
                                                             "pixels", "optimizer", "stage_optimizer", "directions", "sliced_method", "sliced_scale", "sliced_dtype", "refine", "pixel_scale", "pixel_background", "pixel_tol", "pixel_split", "pixel_radius", "pixel_blur", "pixel_blur_steps", "pixel_workers", "steps", "last_steps", "stage_steps", "tau", "start", "resume",
                                                             "final_steps", "final_precision", "final_tol" ) },
                    phantom = ph.name, thickness = getattr( ph, "thickness", None ),
                    wall = wall, steps = len( rec.history ), **{ k: last[ k ] for k in keys } )
    # how much the z of the points cling to the detector rows ( the centers of the pixels along the rotation axis ): the
    # fraction in the central half of a row, 0.5 for points spread evenly
    dvz = radio.dv
    phase = ( ( rec.positions[ :, 2 ] - radio.v_min ) / dvz ) % 1.0
    summary[ "z_in_row_center" ] = float( ( np.abs( phase - 0.5 ) < 0.25 ).mean() )
    ( out / "summary.json" ).write_text( json.dumps( summary, indent = 1, default = float ) )
    index = write_view( out / "run.npz", title = f"{ ph.name } — { p.optimizer }, n={ p.n }, { p.views } vues, "
                                                 f"{ radio.nb_u_host }×{ radio.nb_v_host }" )
    print( f"wall { wall:.1f} s -> { index }" )
    for k in ( "cost", "inside", "dist_mean", "coverage", "balance", "disagreement_mean" ):
        if k in last:
            p.results[ k ] = float( last[ k ] )
    p.results[ "wall" ] = wall
