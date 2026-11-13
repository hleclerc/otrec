"""Balls at the radius the cloud finds, then POLYGONS ( `to_polygons`, from the Laguerre cells of the
balls; a density each, that stays close to the others ): what the walls and the borders of a packing gain. Few shapes on purpose, each of a few pixels: the shape of a
shape must matter.

    errand run --env lmo-torch -o 'otrec/experiments/exp_2d_polygons.py::*'                              ( from ~/Projects )

The page plays the polygons over the outline of the object: the balls first ( frame -1, each drawn as the disc of
density 1 of the same mass ), the cells the polygons start from ( 0 ), then the iterations. `covered_twice`: the share of
the area of the polygons that lies over another one; `void`: the share of the object no polygon covers ( both read on
the pixels ); `reflex`: the share of the vertices where a polygon is not convex; `max_gap`: the largest angle between two
vertices in a row, seen from their centre ( under 180 degrees, a polygon cannot cross itself ); `density`: of the polygons, 1 the mean one;
`at_<i>`: along the iterations of the polygons."""
from errand import experiment, Param

if p := experiment( "polygons",
        n_start = Param( 1250, help = "shapes of the first stage, about" ), n = Param( 5000, help = "shapes at the end" ),
        division = Param( 2 ), views = Param( 720 ), bins = Param( 512 ), vertices = Param( 8, help = "of a polygon" ),
        max_ratio = Param( 4.0, help = "a vertex stays within that factor of the radius of the balls from its centre" ),
        move_tol = Param( 0.05, help = "of the stages of balls" ), polygon_move_tol = Param( 0.001 ),
        max_iter = Param( 3000 ), seed = Param( 0 ), frames = Param( "25,50,100,200,400,800,1600", help = "iterations of the polygons the page shows" ),

        k = Param( 2.0, help = "the projection of a ball is ( 1 - t^2 )^k; 0.5: true discs, of density 1" ) ):
    import math

    import numpy as np
    import torch

    from otrec import Reconstruction, Sinogram, ball_packing, viz
    from otrec.loss import Loss1d
    from otrec.shapes import polygons, sized
    from sdot.viz.Visualizer import Visualizer

    image = ball_packing( nb = 2 * p.bins, seed = p.seed )
    sino = Sinogram.from_image( image, p.views, p.bins )
    truth = image.reshape( p.bins, 2, p.bins, 2 ).mean( ( 1, 3 ) )
    rec = Reconstruction( sino, k = p.k, move_tol = p.move_tol, max_iter = p.max_iter, seed = p.seed, max_ratio = p.max_ratio, verbose = True )
    md = rec.model
    page = Visualizer( title = f"otrec 2D -- { p.n } balls, then polygons", frame_axis = "iteration of the polygons", frame_value = -1, flat_points = True )
    edges = viz.contours( image, sino.extent )

    def image_error( name, img ):
        """the relative L1 gap, pixel by pixel, between what the shapes say the object is and the object ( at the same mass )"""
        p.results[ f"{ name }_image_error" ] = float( np.abs( img * ( truth.sum() / img.sum() ) - truth ).sum() / truth.sum() )

    def frame( x, value ):
        pos, tri, border = polygons.outlines( *polygons.vertices( x, md ) )
        page.new_frame( value )
        page.add_faces( pos, tri, color = viz.CLOUD, opacity = 0.5 )
        page.add_edges( pos, border, color = viz.CLOUD )
        page.add_edges( *edges, color = viz.TRUTH )

    def report( name, x ):
        c, v = polygons.vertices( x, md )
        w = polygons.shapes( x, md )[ 1 ]
        img = polygons.density( c, v, w, p.bins, sino.extent )
        cover = polygons.density( c, v, torch.ones_like( w ), p.bins, sino.extent ) / ( sino.extent / p.bins ) ** 2  # polygons over each pixel
        d = ( w * polygons.areas( v ).sum() ).cpu().numpy()                                                         # densities, 1: the mean one
        p.results.update( { f"{ name }_density_q10": float( np.quantile( d, 0.1 ) ), f"{ name }_density_q90": float( np.quantile( d, 0.9 ) ) } )
        gaps, reflex = polygons.turns( c, v )
        p.results.update( { f"{ name }_covered_twice": float( np.maximum( cover - 1, 0 ).sum() / cover.sum() ),
                            f"{ name }_void": float( ( truth * np.maximum( 1 - cover, 0 ) ).sum() / truth.sum() ),
                            f"{ name }_reflex": reflex.double().mean().item(), f"{ name }_max_gap_degrees": math.degrees( gaps.max().item() ),
                            f"{ name }_gap_over_180": float( ( gaps.max( 1 ).values > math.pi ).double().mean().item() ) } )
        image_error( name, img )

    rec.multiscale( p.n_start, p.n, p.division, radius = 0.8 )
    p.results[ "radius" ] = rec.radius
    p.results[ "radius_over_disc_of_the_matter" ] = rec.radius / float( np.sqrt( image.mean() * sino.extent ** 2 / p.n / np.pi ) )
    p.results[ "balls_loss_above_floor" ] = rec.loss - rec.loss_fn.floor
    image_error( "balls", sized.density( md.tensor( rec.points ), md.tensor( rec.radii ), p.bins, sino.extent, md.k ) )
    page.add_faces( *sized.outlines( rec.points, rec.radii * math.sqrt( md.unit / math.pi ) ), color = viz.CLOUD, opacity = 0.5 )
    page.add_edges( *edges, color = viz.TRUTH )

    r = md.tensor( rec.radii ) * math.sqrt( md.unit / math.pi )
    start = polygons.from_discs( md.tensor( rec.points ), r, rec.radius, p.vertices, math.pi * r * r )                # ( as `to_polygons` )
    p.results[ "cells_loss_above_floor" ] = Loss1d( md, polygons ).value_and_grad( start, 0.0, rec.background )[ 0 ] - rec.loss_fn.floor
    report( "cells", start )
    frame( start, 0 )

    shown, count = { int( i ) for i in str( p.frames ).split( "," ) }, [ 0 ]
    def each( x, loss ):
        count[ 0 ] += 1
        if count[ 0 ] in shown:
            frame( x, count[ 0 ] )
            c, v = polygons.vertices( x, md )                          # how fast: the loss and the image along the iterations
            p.results[ f"at_{ count[ 0 ] }_loss_above_floor" ] = loss - rec.loss_fn.floor
            image_error( f"at_{ count[ 0 ] }", polygons.density( c, v, polygons.shapes( x, md )[ 1 ], p.bins, sino.extent ) )
    rec.to_polygons( p.vertices, p.polygon_move_tol, callback = each )
    p.results[ "polygons_loss_above_floor" ] = rec.loss - rec.loss_fn.floor
    report( "polygons", md.tensor( rec.polygons ) )
    if count[ 0 ] not in shown:
        frame( md.tensor( rec.polygons ), count[ 0 ] )
    for key, value in p.results.items():
        print( f"{ key }: { value:.4g}" )
    page.write_html( str( p.out_dir / "polygons.html" ) )
