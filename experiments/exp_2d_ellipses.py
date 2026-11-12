"""Balls, then ELLIPSES ( `Reconstruction.to_ellipses` ): what the walls and the borders of a packing gain when each shape may
stretch and take a mass of its own. Few shapes on purpose, each of a few pixels: the shape of a shape must matter.

    errand run --env lmo-torch -o 'otrec/experiments/exp_2d_ellipses.py::*'                              ( from ~/Projects )

The page has two frames: the balls, then the ellipses, over the outline of the object."""
from errand import experiment, Param

if p := experiment( "ellipses",
        n_start = Param( 1250, help = "shapes of the first stage, about" ), n = Param( 5000, help = "shapes at the end" ),
        division = Param( 2 ), views = Param( 720 ), bins = Param( 512 ),
        radius = Param( 0.008, help = "of the balls at the end ( the object is a disc of radius 0.8 )" ),
        max_ratio = Param( 4.0, help = "a half-axis of an ellipse stays within that factor of `radius`" ),
        move_tol = Param( 0.1, help = "of the stages of balls" ), ellipse_move_tol = Param( 0.01 ), max_iter = Param( 3000 ), seed = Param( 0 ) ):
    import numpy as np
    import torch

    from otrec import Reconstruction, Sinogram, ball_packing, viz
    from otrec.shapes import ellipses
    from sdot.viz.Visualizer import Visualizer

    image = ball_packing( nb = 2 * p.bins, seed = p.seed )
    sino = Sinogram.from_image( image, p.views, p.bins )
    truth = image.reshape( p.bins, 2, p.bins, 2 ).mean( ( 1, 3 ) )

    def image_error( x ):
        """the relative L1 gap, pixel by pixel, between what the shapes say the object is and the object, at the same mass"""
        img = ellipses.density( x, p.radius, p.max_ratio, p.bins, sino.extent, 2 )
        return float( np.abs( img * ( truth.sum() / img.sum() ) - truth ).sum() / truth.sum() )

    rec = Reconstruction( sino, radius = p.radius, move_tol = p.move_tol, max_iter = p.max_iter, seed = p.seed, verbose = True )
    rec.multiscale( p.n_start, p.n, p.division, radius = 0.8 )
    balls = ellipses.from_points( rec.model.tensor( rec.points ) )
    floor = rec.loss_fn.floor
    p.results.update( balls_loss_above_floor = rec.loss - floor, balls_image_error = image_error( balls ) )

    rec.to_ellipses( p.max_ratio, p.ellipse_move_tol )
    x = rec.model.tensor( rec.ellipses )
    _, S, masses = ellipses.unpack( x, p.radius )
    axes = torch.exp( torch.linalg.eigvalsh( S ) ) / p.radius                             # the half-axes, in radii
    p.results.update( ellipses_loss_above_floor = rec.loss - floor, ellipses_image_error = image_error( x ),
                      aspect_ratio_median = float( ( axes[ :, 1 ] / axes[ :, 0 ] ).median() ), aspect_ratio_q90 = float( torch.quantile( axes[ :, 1 ] / axes[ :, 0 ], 0.9 ) ),
                      mass_ratio_q90_over_q10 = float( torch.quantile( masses, 0.9 ) / torch.quantile( masses, 0.1 ) ) )
    for key, value in p.results.items():
        print( f"{ key }: { value:.4g}" )

    page = Visualizer( title = f"otrec 2D -- { p.n } balls, then ellipses", frame_axis = "stage", flat_points = True, playable = False )
    edges = viz.contours( image, sino.extent )
    page.add_points( balls[ :, :2 ].cpu().numpy(), radius = p.radius, color = viz.CLOUD, opacity = 0.5 )
    page.add_edges( *edges, color = viz.TRUTH )
    page.new_frame( 1 )
    page.add_faces( *ellipses.outlines( x.cpu(), p.radius ), color = viz.CLOUD, opacity = 0.5 )
    page.add_edges( *edges, color = viz.TRUTH )
    page.write_html( str( p.out_dir / "ellipses.html" ) )
