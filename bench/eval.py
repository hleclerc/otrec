"""The time of ONE evaluation of the 2D loss and its gradient ( best of `reps`, after a warm-up ), on random shapes in
the disk of a packing.

    errand run --env torch 'otrec/bench/eval.py::*' --n=100000,1000000              ( from ~/Projects )
    errand run --env lmo-torch 'otrec/bench/eval.py::*' --n=10000000 --device=cuda,cpu"""
from errand import bench, Param

if p := bench( "eval", n = Param( 100_000 ), views = Param( 360 ), bins = Param( 256 ), radius = Param( 0.008, help = "of the shapes" ),
               device = Param( "cpu" ), reps = Param( 5 ),
               shapes = Param( "balls", help = "balls, or ellipses ( anisotropic: a width per view )" ) ):
    import time

    import torch

    from otrec import Reconstruction, Sinogram, ball_packing

    sino = Sinogram.from_image( ball_packing( 2 * p.bins ), p.views, p.bins )
    rec = Reconstruction( sino, device = p.device ).add_random_balls( p.n, p.radius )
    points = rec.model.tensor( rec.points )
    loss_fn = rec.loss_fn
    if p.shapes == "ellipses":                               # ( stretched up to 2:1, turned at random )
        import numpy as np
        from otrec.loss import Loss1d
        from otrec.shapes import ellipses
        rng = np.random.default_rng( 0 )
        a, t = p.radius * rng.uniform( 0.7, 1.4, p.n ), rng.uniform( 0, np.pi, p.n )
        b = p.radius ** 2 / a
        M = np.stack( [ np.stack( [ a * np.cos( t ), -b * np.sin( t ) ], 1 ), np.stack( [ a * np.sin( t ), b * np.cos( t ) ], 1 ) ], 1 )
        L = np.linalg.cholesky( M @ M.transpose( 0, 2, 1 ) )                  # ( the same ellipse, M lower triangular )
        points = rec.model.tensor( np.concatenate( [ rec.points, L[ :, 0, :1 ], L[ :, 1, :1 ], L[ :, 1, 1: ] ], 1 ) )
        loss_fn = Loss1d( rec.model, ellipses )

    def timed( f ):
        best = None
        for _ in range( p.reps + 1 ):                        # ( the first one compiles )
            t = time.perf_counter()
            f()
            if rec.model.device.type == "cuda":
                torch.cuda.synchronize()
            best = time.perf_counter() - t if best is None else min( best, time.perf_counter() - t )
        return best

    p.results[ "seconds" ] = timed( lambda: loss_fn.value_and_grad( points, 0.0, 1e-3 ) )
    if p.shapes == "balls":
        p.results[ "seconds_from_numpy" ] = timed( lambda: rec.value_and_grad() )
    print( p.results )
