"""A LUNG in 2D, binary and noise-free: `data/lung_clean_2048.png` ( `data/make_lung_png.py`: the big structures of a real
slice at high resolution, its parenchyma made a clean foam -- thin walls, alveoli between balls and cells, a fissure,
airways and solid lumps ), entirely in the field ( a disc of radius 0.9 ). What it tests: big uniform zones next to fine
ones ( walls of about one bin at 512 bins ), and the time it takes. The STRATEGIES are written out below, step by step,
to be played with; `lung_case.py` holds the rest ( the lung, its sinogram, the gap to it, the recorder ), `strategies.py`
the steps that are not in `Reconstruction` yet.

    errand run --env lmo-torch -o 'otrec/experiments/exp_2d_lung.py::*'                      ( from ~/Projects )
    errand run --env lmo-torch -o 'otrec/experiments/exp_2d_lung.py::lung' --n=400000 --bins=1024 --views=1536

    lung        balls at random, then splits ( the recipe of `exp_2d_real` )
    lloyd       the same, each stage until the loss no longer goes down, then Lloyd ( the balls evenly spaced ) and a
                stage again, until that stage is short ( `settle` )
    borders     as `lloyd`, but at each level the balls deep in the matter merge 4 by 4 ( bigger inside ) and only the
                border balls the views contest the most split ( small balls in the recesses: `strategies` )

Each run writes `lung.sviz` ( a scene store, frame by frame: `sviz open` on it, while it goes on )."""
from errand import experiment, Param

PARAMS = dict(
    n = Param( 100000, help = "shapes at the end" ),
    nb_splits = Param( 2, help = "the start has n / 4^nb_splits shapes; each split divides them in 4" ),
    views = Param( 768 ), bins = Param( 512 ),
    x_tol = Param( 0.1, help = "a stage stops when the shapes move less than that fraction of their radius in 10 iterations" ),
    noise = Param( 0.0, help = "gaussian noise on the sinogram, its deviation in fraction of the mean of a bin" ),
    record_every = Param( 20 ), seed = Param( 0 ), device = Param( "default", help = "cpu, cuda, or loom's default" ) )


def imports():
    import sys
    from pathlib import Path
    sys.path.insert( 0, str( Path( __file__ ).parent ) )


if p := experiment( "lung", **PARAMS ):
    import time
    imports()
    import lung_case

    image, sino, rec = lung_case.setup( p )
    t0 = time.time()

    # balls at random, as small as a bin: their centres fitted ( small, they spread as the matter does )
    rec.add_random_balls( p.n // 4 ** p.nb_splits, radius = sino.h )
    rec.adjust_parameters( centers = True, x_tol = p.x_tol )

    # the radius where they fill the matter ( from the distance between neighbours ), the centres fitted again
    rec.radius = rec.nominal_radius()
    rec.adjust_parameters( centers = True, x_tol = p.x_tol )

    # `nb_splits` times: the nominal radius of the settled cloud, then each ball in 4 ( that radius / 2 ), the centres
    # fitted again
    for split in range( p.nb_splits ):
        rec.radius = rec.nominal_radius()
        rec.split_items( division = 4, noise_ratio = 0.5 )
        rec.adjust_parameters( centers = True, x_tol = p.x_tol )

    #   alone in the air, `rec.remove_isolated( 1.5 )` )
    lung_case.summary( rec, image, t0 )


def settle( rec, p ):
    """THE STAGE with Lloyd: the centres fitted until the loss no longer goes down; then, at most `p.rounds` times, the
    spacing made regular ( `relax_spacing` ) and the centres fitted again, until that stage is short ( `p.settled`
    iterations: the regular cloud is one the data accept )"""
    from strategies import relax_spacing
    rec.adjust_parameters( centers = True, radii = "keep", x_tol = p.x_tol, f_tol = p.f_tol )
    for _ in range( p.rounds ):
        relax_spacing( rec, iterations = p.lloyd )
        rec.adjust_parameters( centers = True, radii = "keep", x_tol = p.x_tol, f_tol = p.f_tol )
        if rec.recorder.stages[ -1 ][ "nb_iter" ] <= p.settled:
            break


LLOYD = dict( rounds = Param( 4, help = "relaxations at most per level" ), lloyd = Param( 2, help = "steps of Lloyd in a relaxation" ),
              settled = Param( 40, help = "a stage after a relaxation that short: the cloud has settled" ),
              f_tol = Param( 0.01, help = "a stage stops when 10 iterations take off less than that fraction of the loss above its floor" ) )


if p := experiment( "lloyd", **LLOYD, **PARAMS ):
    import time
    imports()
    import lung_case

    image, sino, rec = lung_case.setup( p )
    t0 = time.time()

    rec.add_random_balls( p.n // 4 ** p.nb_splits, radius = sino.h )
    rec.adjust_parameters( centers = True, x_tol = p.x_tol )               # ( small balls: they spread as the matter does )
    rec.radius = rec.nominal_radius()
    settle( rec, p )
    for split in range( p.nb_splits ):
        rec.radius = rec.nominal_radius()
        rec.split_items( division = 4, noise_ratio = 0.5 )
        settle( rec, p )
    lung_case.summary( rec, image, t0 )


if p := experiment( "borders", share = Param( 0.5, help = "of the border balls split at each level, the most contested" ),
                    max_ratio = Param( 8.0, help = "the biggest ball, in radii of the smallest" ), **LLOYD, **PARAMS ):
    import time
    imports()
    import lung_case
    from strategies import coarsen_interior, refine_borders

    image, sino, rec = lung_case.setup( p )
    area = sino.matter_area()                                              # ( the matter the FBP finds )
    t0 = time.time()

    rec.add_random_balls( p.n // 4 ** p.nb_splits, radius = sino.h )
    rec.adjust_parameters( centers = True, x_tol = p.x_tol )
    rec.radius = rec.nominal_radius()
    settle( rec, p )
    # each level: the balls deep in the matter merged 4 by 4 ( bigger inside ), the border balls the views contest the
    # most split in 4 ( small balls in the recesses only )
    for split in range( p.nb_splits ):
        coarsen_interior( rec, area )
        refine_borders( rec, area, share = p.share )
        settle( rec, p )
    lung_case.summary( rec, image, t0 )
