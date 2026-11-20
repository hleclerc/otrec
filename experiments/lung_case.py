"""What `exp_2d_lung.py` needs besides its steps: the clean lung ( `data/lung_clean_2048.png` ), its sinogram, the gap
to it, the recorder of the run ( a scene store, `.sviz`, closed at the end of the process ), and the line that ends it."""
from pathlib import Path

import numpy as np

PNG = Path( __file__ ).parent / "data" / "lung_clean_2048.png"


def truth( nb ):
    """`[ nb, nb ]` in [ 0, 1 ]: the lung averaged to `nb` pixels per side ( the centred square of side 2; the field, the
    disc of radius 0.9 ); `nb` divides 2048"""
    from PIL import Image
    png = np.asarray( Image.open( PNG ) ).astype( float ) / 255
    f = png.shape[ 0 ] // nb
    return png.reshape( nb, f, nb, f ).mean( ( 1, 3 ) )


def sinogram( image, views, bins, noise = 0.0, seed = 0 ):
    """its sinogram ( `Sinogram.from_image` ), with a gaussian noise of deviation `noise` times the mean of a bin"""
    from otrec import Sinogram
    sino = Sinogram.from_image( image, views, bins )
    if noise:
        rng = np.random.default_rng( seed + 1 )
        sino.values = np.clip( sino.values + noise * sino.values.mean() * rng.normal( size = sino.values.shape ), 0, None )
    return sino


def gap( rec, image, nb = 512 ):
    """`( points, radii ) -> ` the relative L1 gap between what the balls draw and the lung, on `nb^2` pixels: the sum of
    the profiles of the balls ( each weighing its area ), scaled to the mass of the lung, against the lung averaged on
    those pixels; 0, the same image; 2, disjoint ones ( never 0 with balls: their profile is a bump, not a plateau )"""
    from otrec.metrics import make_image_error
    f = image.shape[ 0 ] // nb
    return make_image_error( image.reshape( nb, f, nb, f ).mean( ( 1, 3 ) ), rec.sinogram.extent, rec )


def setup( p ):
    """`( image, sino, rec )`: the lung at 2 pixels per bin, its sinogram, a `Reconstruction` of it with its recorder
    ( `p`: views, bins, noise, seed, device, n, record_every, out_dir; the entry's name in the title )"""
    from otrec import Reconstruction
    from otrec.shapes import fused
    fused.PRECISION = getattr( p, "precision", fused.PRECISION )
    image = truth( 2 * p.bins )
    sino = sinogram( image, p.views, p.bins, p.noise, p.seed )
    rec = Reconstruction( sino, device = None if p.device == "default" else p.device, max_ratio = getattr( p, "max_ratio", 4.0 ),
                          fused = bool( getattr( p, "fused", 0 ) ) )
    rec.recorder = recorder( rec, image, p.out_dir, f"otrec 2D -- a lung, { p.n } shapes", p.record_every )
    rec.rng = np.random.default_rng( p.seed )
    return image, sino, rec


def summary( rec, image, t0 ):
    """the line that ends a run: balls, radius, gap to the truth, loss, iterations, time"""
    import time
    st = rec.recorder.stages
    radii = f"radius { rec.radius:.4g}" if np.all( rec.radii == rec.radius ) else f"radii { rec.radii.min():.4g} - { rec.radii.max():.4g}"
    print( f"{ rec.nb_points } balls, { radii }, gap to the truth { gap( rec, image )( rec.points, rec.radii ):.3f},"
           f" loss / floor { st[ -1 ][ 'loss' ] / rec.loss_fn.floor:.4f}, { sum( s[ 'nb_iter' ] for s in st ) } iterations, { time.time() - t0:.0f} s,"
           f" stops { [ s.get( 'stop' ) for s in st ] }" )


def recorder( rec, image, out_dir, title, every = 20 ):
    """a `Recorder` for `rec`: a frame every `every` iterations to the scene store `out_dir / lung.sviz` ( `sviz open`
    while the run goes on, brought back by errand as it grows ), the lung as a layer and its outline, and the gap to it
    as a curve"""
    import errand
    from otrec import Recorder, viz
    store = Path( out_dir ) / "lung.sviz"
    E = rec.sinogram.extent
    return Recorder( every = every, verbose = True, store = store, metrics = { "gap to the truth": gap( rec, image ) },
                     on_frame = lambda: errand.view( store ),
                     page = dict( title = title, images = { "truth": ( image, E ) }, truth_edges = viz.contours( image, E ),
                                  curves = ( "loss", "gap to the truth" ) ) )
