"""A packing of balls in 2D ( `otrec.ball_packing`: an image ), reconstructed from its sinogram and a RANDOM start ( uniform
in the disk that holds it: nothing is known of the holes ), with an HTML page of the steps and the convergence.

    errand run --env torch -o 'otrec/experiments/exp_2d.py::*'                              ( from ~/Projects )
    errand run --env torch -o 'otrec/experiments/exp_2d.py::*' --n_start=250000 --n=1000000 --bins=1024 --views=1440

Take about 1.5 views per bin: with fewer views than bins the data does not determine the density ( see the README )."""
from errand import experiment, Param

if p := experiment( "disks",
        n_start = Param( 1250, help = "shapes of the first stage, about" ),
        n = Param( 15000, help = "shapes at the end" ),
        division = Param( 4, help = "each shape is divided in that many between two stages" ),
        views = Param( 600 ), bins = Param( 1024 ),
        move_tol = Param( 0.01, help = "a stage stops when the shapes move less than that fraction of their radius in 10 iterations" ),
        record_every = Param( 20 ), seed = Param( 0 ), device = Param( "default", help = "cpu, cuda, or loom's default" ) ):

    from otrec import Reconstruction, Recorder, Sinogram, ball_packing, viz

    image = ball_packing( nb = 2 * p.bins, seed = p.seed, gap = 0.025 )
    sino = Sinogram.from_image( image, p.views, p.bins )

    rec = Reconstruction( sino,
        device = None if p.device == "default" else p.device,
        recorder = Recorder( p.record_every, verbose = True )
    )

    rec.multiscale( p.n_start, p.n, p.division, x_tol = p.move_tol )

    rec.recorder.write_html( p.out_dir / "cloud.html",
        title = f"otrec 2D -- a packing of balls, { rec.nb_points } shapes, random start",
        truth_edges = viz.contours( image, sino.extent )
    )


# -- balls, then ELLIPSES ( a covariance each, the profile of the balls ), the steps written out
#     errand run --env torch -o 'otrec/experiments/exp_2d.py::ellipses'                        ( from ~/Projects )
if p := experiment( "ellipses", n = Param( 5000, help = "nb balls" ), radius = Param( 0.01, help = "of the balls" ),
                    views = Param( 720 ), bins = Param( 512 ) ):
    from otrec import Reconstruction, Recorder, Sinogram, ball_packing, viz

    image = ball_packing( nb = 2 * p.bins )
    sino = Sinogram.from_image( image, p.views, p.bins )

    recorder = Recorder( every = 20, verbose = True )

    rec = Reconstruction( sino, recorder = recorder )

    # scale 0
    rec.add_random_balls( p.n // 4, radius = p.radius * 2 )
    rec.adjust_parameters( centers = True, x_tol = 0.4 )

    # scale 1
    rec.split_items( division = 4, noise_ratio = 0.1 )
    rec.adjust_parameters( centers = True, x_tol = 0.1 )

    # radii
    rec.adjust_parameters( centers = True, radii = True, x_tol = 0.01 )

    # ellipses
    rec.adjust_parameters( centers = True, cov_matrices = True, x_tol = 0.01 )

    recorder.write_html(
        p.out_dir / "ellipses.html",
        truth_edges = viz.contours( image, sino.extent )
    )
