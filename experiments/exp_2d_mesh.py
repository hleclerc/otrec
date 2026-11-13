"""Balls ( `k = 2`: they overlap ), then the CONTOURS of their density and a MESH of the matter between them
( `shapes.mesh` ), then the NODES of the mesh moved to lower the loss ( L-BFGS; only those of the border change it ).

    errand run --env lmo-torch -o 'otrec/experiments/exp_2d_mesh.py::*'                                  ( from ~/Projects )

The page shows the mesh over the outline of the object: as built from the balls ( frame 0 ), then along the iterations
of its nodes. `<stage>_mesh`: the mesh built from the balls of a stage ( `balls`: of one radius; `sized_<i>`: after `i`
iterations with a radius each ); `moved_<i>`: after `i` iterations of its nodes. `area`: of the mesh over that of the
object; `image_error`: the relative L1 gap, pixel by pixel, between the shapes ( the mesh: one density ) and the object;
`smallest_triangle`: over the mean one ( negative: a triangle turned over ); `longest_side`: of the border, over the radius; `contour_sides_missing`: the share of the
sides of the contours that are not sides of the border of the mesh."""
from errand import experiment, Param

if p := experiment( "mesh",
        n_start = Param( 1250, help = "shapes of the first stage, about" ), n = Param( 5000, help = "shapes at the end" ),
        division = Param( 2 ), views = Param( 720 ), bins = Param( 512 ), move_tol = Param( 0.05, help = "of the stages of balls" ),
        sized = Param( "", help = "iterations of the stages of balls with a radius each, one after the other" ),
        sized_move_tol = Param( 0.001 ), move_tol_nodes = Param( 0.001 ), edge = Param( 0.5, help = "of the mesh, over the radius of the balls" ),
        level = Param( 0.5, help = "the contours: where the density of the balls is that share of the typical one" ),
        move = Param( "25,50,100,200,400,800,1600", help = "iterations of the nodes of the last mesh the page shows ( the last one: where it stops )" ),
        max_ratio = Param( 4.0 ), max_iter = Param( 3000 ), seed = Param( 0 ), k = Param( 2.0 ) ):
    import math
    import time

    import numpy as np
    import torch

    from otrec import Reconstruction, Sinogram, ball_packing, viz
    from otrec.lbfgs import lbfgs
    from otrec.loss import Loss1d
    from otrec.shapes import mesh, polygons, sized
    from sdot.viz.Visualizer import Visualizer

    image = ball_packing( nb = 2 * p.bins, seed = p.seed )
    sino = Sinogram.from_image( image, p.views, p.bins )
    truth = image.reshape( p.bins, 2, p.bins, 2 ).mean( ( 1, 3 ) )
    rec = Reconstruction( sino, k = p.k, move_tol = p.move_tol, max_iter = p.max_iter, seed = p.seed, max_ratio = p.max_ratio, verbose = True )
    md = rec.model
    page = Visualizer( title = f"otrec 2D -- { p.n } balls, their contours, a mesh", frame_axis = "iteration of the nodes", frame_value = 0, flat_points = True )
    edges = viz.contours( image, sino.extent )

    def image_error( img ):
        return float( np.abs( img * ( truth.sum() / img.sum() ) - truth ).sum() / truth.sum() )

    def measure( name, kind, x ):
        """the mesh of nodes `x` as it stands: its loss, its image, its area, its smallest triangle ( < 0: turned over )"""
        tri = polygons.areas( x[ kind.triangles ] )
        loss = Loss1d( md, kind ).value_and_grad( x, 0.0, rec.background )[ 0 ]
        p.results.update( { f"{ name }_loss_above_floor": loss - rec.loss_fn.floor, f"{ name }_image_error": image_error( kind.density( x, p.bins, sino.extent ) ),
            f"{ name }_area": tri.sum().item() / ( image.mean() * sino.extent ** 2 ), f"{ name }_smallest_triangle": ( tri.min() / tri.mean() ).item(),
            f"{ name }_longest_side": torch.linalg.vector_norm( x[ kind.sides[ :, 1 ] ] - x[ kind.sides[ :, 0 ] ], dim = 1 ).max().item() / rec.radius } )

    def frame( value, kind, x ):
        nodes, tri = x.cpu().numpy(), kind.triangles.cpu().numpy()
        if value:
            page.new_frame( value )
        page.add_faces( nodes, tri, color = viz.CLOUD, opacity = 0.5 )
        page.add_edges( nodes, np.concatenate( [ tri[ :, [ 0, 1 ] ], tri[ :, [ 1, 2 ] ], tri[ :, [ 2, 0 ] ] ] ), color = viz.CLOUD )
        page.add_edges( *edges, color = viz.TRUTH )

    def report( name ):
        """the balls as they stand, and the mesh of their contours"""
        t0 = time.perf_counter()
        nodes, tri, loops = mesh.from_balls( md.tensor( rec.points ), md.tensor( rec.radii ), md.k, sino.extent, p.edge * rec.radius, p.level )
        seconds = time.perf_counter() - t0
        kind, x = mesh.Mesh( md.tensor( tri ).long() ), md.tensor( nodes )
        sides = { ( int( min( i, j ) ), int( max( i, j ) ) ) for i, j in mesh.border( tri ) }
        wanted = [ ( int( min( i, j ) ), int( max( i, j ) ) ) for l in loops for i, j in zip( l, np.roll( l, -1 ) ) ]
        p.results.update( { f"{ name }_loss_above_floor": rec.loss - rec.loss_fn.floor,
            f"{ name }_image_error": image_error( sized.density( md.tensor( rec.points ), md.tensor( rec.radii ), p.bins, sino.extent, md.k ) ),
            f"{ name }_radius_q10": float( np.quantile( rec.radii, 0.1 ) / rec.radius ), f"{ name }_radius_q90": float( np.quantile( rec.radii, 0.9 ) / rec.radius ),
            f"{ name }_mesh_seconds": seconds, f"{ name }_mesh_nodes": len( nodes ), f"{ name }_mesh_triangles": len( tri ), f"{ name }_mesh_contours": len( loops ),
            f"{ name }_mesh_contour_sides_missing": float( np.mean( [ s not in sides for s in wanted ] ) ) } )
        measure( f"{ name }_mesh", kind, x )
        return kind, x

    rec.multiscale( p.n_start, p.n, p.division, radius = 0.8 )
    p.results[ "radius" ] = rec.radius
    kind, x = report( "balls" )
    done = 0
    for nb in [ int( i ) for i in str( p.sized ).split( "," ) if i ]:
        rec.resize( p.sized_move_tol, nb )
        done += rec.history[ -1 ][ "nb_iter" ]
        kind, x = report( f"sized_{ done }" )
    frame( 0, kind, x )

    # the nodes of the mesh, moved to lower the loss ( those of the border: the others do not change it )
    shown, count, t0 = [ int( i ) for i in str( p.move ).split( "," ) ], [ 0 ], time.perf_counter()
    def each( x, loss ):
        count[ 0 ] += 1
        if count[ 0 ] in shown:
            measure( f"moved_{ count[ 0 ] }", kind, x )
            frame( count[ 0 ], kind, x )
    loss_fn = Loss1d( md, kind )
    x, _, nb_iter, nb_eval, stop = lbfgs( lambda x: loss_fn.value_and_grad( x, 0.0, rec.background ), x, p.move_tol_nodes * rec.radius, shown[ -1 ], callback = each )
    p.results.update( { "move_iterations": nb_iter, "move_evaluations": nb_eval, "move_seconds": time.perf_counter() - t0 } )
    print( "nodes:", stop )
    if nb_iter not in shown:
        measure( "moved_end", kind, x )
        frame( nb_iter, kind, x )
    for key, value in p.results.items():
        print( f"{ key }: { value:.4g}" )
    page.write_html( str( p.out_dir / "mesh.html" ) )
