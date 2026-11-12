"""The time of ONE evaluation of the 2D loss and its gradient ( best of `reps`, after a warm-up ), on random shapes in
the disk of a packing.

    errand run --env torch 'otrec/bench/eval.py::*' --n=100000,1000000              ( from ~/Projects )
    errand run --env lmo-torch 'otrec/bench/eval.py::*' --n=10000000 --device=cuda,cpu"""
from errand import bench, Param

if p := bench( "eval", n = Param( 100_000 ), views = Param( 360 ), bins = Param( 256 ), radius = Param( 0.008, help = "of the shapes" ),
               device = Param( "cpu" ), reps = Param( 5 ) ):
    import time

    import torch

    from otrec import Reconstruction, Sinogram, ball_packing

    sino = Sinogram.from_image( ball_packing( 2 * p.bins ), p.views, p.bins )
    rec = Reconstruction( sino, radius = p.radius, device = p.device ).random_points( p.n, radius = 0.8 )
    points = rec.model.tensor( rec.points )

    def timed( f ):
        best = None
        for _ in range( p.reps + 1 ):                        # ( the first one compiles )
            t = time.perf_counter()
            f()
            if rec.model.device.type == "cuda":
                torch.cuda.synchronize()
            best = time.perf_counter() - t if best is None else min( best, time.perf_counter() - t )
        return best

    p.results[ "seconds" ] = timed( lambda: rec.loss_fn.value_and_grad( points, 0.0, 1e-3 ) )
    p.results[ "seconds_from_numpy" ] = timed( lambda: rec.value_and_grad() )
    print( p.results )
