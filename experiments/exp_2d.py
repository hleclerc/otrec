"""A packing of balls in 2D ( `otrec.ball_packing`: an image ), reconstructed from its sinogram and a RANDOM start ( uniform
in the disk that holds it: nothing is known of the holes ), with an HTML page of the steps and the convergence.

    errand run --env torch -o 'otrec/experiments/exp_2d.py::*'                              ( from ~/Projects )
    errand run --env torch -o 'otrec/experiments/exp_2d.py::*' --n_start=250000 --n=1000000 --bins=1024 --views=1440

Take about 1.5 views per bin: with fewer views than bins the data does not determine the density ( see the README )."""
from errand import experiment, Param

if p := experiment( "2d",
        n_start = Param( 1250, help = "shapes of the first stage, about" ),
        n = Param( 5000, help = "shapes at the end" ),
        division = Param( 4, help = "each shape is divided in that many between two stages" ),
        views = Param( 360 ), bins = Param( 1024 ),
        radius = Param( 0.003, help = "of the shapes at the end ( the object is a disc of radius 0.8 )" ),
        move_tol = Param( 0.001, help = "a stage stops when the shapes move less than that fraction of their radius in 10 iterations" ),
        record_every = Param( 20 ), seed = Param( 0 ), device = Param( "default", help = "cpu, cuda, or loom's default" ) ):

    from otrec import Reconstruction, Recorder, Sinogram, ball_packing, viz

    image = ball_packing( nb = 2 * p.bins, seed = p.seed, gap = 0.025 )
    sino = Sinogram.from_image( image, p.views, p.bins )

    rec = Reconstruction( sino, 
        radius = p.radius, 
        move_tol = p.move_tol, 
        seed = p.seed, 
        device = None if p.device == "default" else p.device,
        recorder = Recorder( p.record_every, {} ), 
        verbose = True 
    )

    rec.multiscale( p.n_start, p.n, p.division )

    viz.export_html( rec, p.out_dir / "cloud.html", 
        title = f"otrec 2D -- a packing of balls, { rec.nb_points } shapes, random start",
        truth_edges = viz.contours( image, sino.extent ) 
    )
