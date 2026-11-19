# otrec

Tomographic reconstruction by optimal transport, in 2D ( 1D detector ) and 3D ( 2D detector ), on PyTorch and sdot.

The object is a cloud of SHAPES, small pieces of matter of equal masses ( 1e5 to 1e7 of them for a continuous material
full of holes ). For each view, the measured pixels are diracs on a grid ( mass = intensity ) and the shapes are projected
and discretized on an image; the positions of the shapes minimize

    L( points ) = sum_views  W2^2(  diracs of the view ,  image of the projected shapes  )

```python
from otrec import Reconstruction, Recorder, Sinogram, ball_packing

image = ball_packing( 512 )                                           # an object: a disc full of round holes
sino = Sinogram.from_image( image, nb_views = 360, nb_bins = 256 )    # its data ( or Sinogram( ..., values = measured ) )
rec = Reconstruction( sino, recorder = Recorder( verbose = True ) )   # the problem, and what keeps the history
rec.add_random_balls( 25_000, radius = 0.006 )                        # how to solve it: the steps. A random start,
rec.adjust_parameters( centers = True, x_tol = 0.1 )                  # a stage ( the centres, until they no longer move ),
rec.split_items( division = 2, noise_ratio = 1.0 )                    # each ball in two ( the radius / sqrt( 2 ) ),
rec.adjust_parameters( centers = True, x_tol = 0.1 )
rec.adjust_parameters( centers = True, cov_matrices = True, x_tol = 0.01 )  # then ellipses: a covariance each ( 2D )
rec.points, rec.ellipses, rec.recorder.steps                          # [ 50_000, 2 ], [ 50_000, 5 ], every step and its numbers
```

## Layout

Each directory has its own docstring ( `__init__.py` ) that says what its files do.

    src/otrec/
        Sinogram.py          the data: geometry of the views, intensities, the diracs; `from_image`, `fbp` ( for reference )
        Scan.py              a MEASURED parallel-beam slice and its steps: counts -> line integrals, axis, centring, air,
                             beam hardening, clipping, phase contrast, matter bigger than the field ( taken off ),
                             -> Sinogram ( 360 folded onto 180, binned )
        download.py          files fetched on first use into $OTREC_DATA or ~/.cache/otrec; one row of a raw stack by HTTP ranges
        projection.py        the line integrals of an image ( what `from_image` runs )
        Reconstruction.py    the problem, and the steps: add_random_balls, split_items, adjust_parameters ( one stage ), fit_polygons; multiscale ( a recipe )
        lbfgs.py             L-BFGS on tensors, on their device, stopped when the shapes no longer move
        cloud.py             random starts, split, spacing, the sizes of a multiscale ( arrays )
        Recorder.py          the history: every step and its parameters, the losses; frames with metrics, and the HTML page ( write_html )
        metrics.py           image error, residual of the sinogram
        viz.py               the outline of the object ( contours ), the colours of a page
        shapes/              balls, sized ( a radius each ), ellipses ( a covariance each ), polygons, mesh ( triangles between the contours of the balls ); their deposit on bins ( 1D, sdot kernels ) or nodes ( 2D ), blur
        transport/           W2^2 of one view: closed form ( 1D ), sdot's grid solver ( 2D )
        loss/                the sum over the views and its gradient: Model, Loss1d ( any of the shapes ), Loss2d;
                             fields: what the views ask of each ball ( move, dissonance, density asked )
        packing/             ball_packing: the image of a disc ( a cylinder ) minus balls settled by gravity
    experiments/             exp_2d.py, exp_3d.py: a packing reconstructed from a random start ( errand entries );
                             exp_2d.py::ellipses: balls, then ellipses ( centre + covariance ), the steps written out;
                             exp_2d_polygons.py: balls, then polygons; exp_2d_mesh.py: balls, then a mesh of their contours;
                             exp_2d_real.py: measured scans, binary objects from the simplest to the hardest, and a pencil
    bench/                   eval.py: the time of one evaluation of the 2D loss
    tests/                   errand run --no-queue --env torch 'otrec/tests/test_X.py::*'    ( from ~/Projects, one file per call )
    attic/                   everything from before ( not imported )

Who calls whom: `Reconstruction` -> `loss` -> `transport` and `shapes`; `packing`, `metrics` and `viz` sit on the side.
The cloud and the data are numpy arrays between the stages; a stage runs on tensors, on loom's device ( the card where
there is one; `Reconstruction( device = "cpu" | "cuda" )` sets it ).

## How it works

* **Shapes** ( `shapes/` ). The radial density ( 1 - |x|^2 / R^2 )^( k - 1/2 ) projects in any direction onto the
  polynomial c ( 1 - |t|^2 / R^2 )^k, in 2D and in 3D. On a 1D detector it is integrated exactly over each bin ( a P0
  image ) by the fused kernels of sdot: one pass over the shapes for all the views, in float64.
  On a 2D detector it is sampled at the corners of the pixels ( a P1 image; radius not below ~1.5 pixel ).
* **The radius** is not a guess: it is the NOMINAL one, read on the cloud ( `Reconstruction.nominal_radius` ). The mean
  distance `d` from a centre to its 3 nearest neighbours gives the room each ball stands for, `d^dim` ( within 10 % on
  2D packings, whatever the radius the stage ran at, up to twice the nominal one ); the nominal radius is that of a ball
  of that mass whose density at the centre is the density of the matter ( 0.89 d in 2D for k = 2 ). `multiscale` takes it
  again before each stage ( `add_random_balls` takes the radius it is given, `split_items` divides it ). 3D: the same formula, not calibrated.
* **The observable domain**. A shape must stay where every view sees it whole: its centre is pulled back so that centre
  + reach stays in the centred ball of diameter `extent` ( `shapes.balls.inside` ).
* **Transport** ( `transport/` ). 1D: the plan is monotone, W2^2 has a closed form that autograd differentiates
  ( float64, a few ms ). 2D: sdot's dedicated grid solver ( `nb_bins` a power of two from 32 ), warm started from the
  last solve of the view; the derivative with respect to the image is the Kantorovich potential.
* **Recipes** ( `Reconstruction` ). The constructor takes the problem, the steps say how to solve it. A STAGE
  ( `adjust_parameters`: the centres, the radii, the covariances, as asked ) is an L-BFGS, on the device, that stops when 99.9 % of the shapes
  have moved by less than `x_tol` of their radius over 10 iterations ( or after `max_iter` ).
  `multiscale( n_start, n_end, division )` chains stages from a random start: about `n_start` shapes first, each one
  divided in `division` between two stages, `n_end` at the end ( `cloud.sizes` ). Each stage runs at the nominal radius of its cloud. For a stage that starts far from the
  object, `sigma` blurs BOTH measures; `background` lifts both by a fraction of their mean; `n_per_pixel`
  diracs per pixel and per axis.
* **The object of the experiments and tests** ( `packing/` ). `ball_packing( nb, dim )` returns an IMAGE: a disc ( a
  cylinder ) minus a packing of balls. The balls, of various sizes and perfectly slippery, are settled by gravity ( a
  minimum of the potential energy under stiff contacts: 0.85 of the room in 2D, 0.65 in 3D, 4 and 6 contacts per ball ),
  then shrunk by a tenth: the gutters between them are the material. `Sinogram.from_image` integrates the rays through
  any image.

Units: every length given to `Reconstruction` is in WORLD units ( `radius`, `sigma` ) or relative to the shapes
( `x_tol`, a fraction of their radius ). Nothing is counted in pixels: the number of bins is a parameter like another.

## After the balls: polygons ( 2D )

Balls of one radius give the large masses of matter, poorly its walls and its borders. `rec.fit_polygons( nb_vertices )`
then makes each shape a STAR POLYGON of one density ( `shapes/polygons.py`: a centre, a size, for each vertex a distance
and the angle to the next one -- always a polygon that turns once around its centre -- and a mass ).
What is drawn is what is computed: an area covered twice, an area left empty.

Each ball is taken as the disc of density 1 of the same mass, and its polygon starts as the LAGUERRE CELL of that disc
among the others, cut by the disc: the side between two polygons is the common chord of their discs; a border nothing
touches is an arc, cut in sides. The vertices go where the border turns, one every `2 pi / nb_vertices` of turning. The
cell is then scaled around its centre to the area of the disc: the mass of the ball is kept.

A polygon projects as a piecewise linear function ( `deposit_polygons`, `gather_polygons` in `sdot/Projected1d.h`, which
take the vertices themselves ): the curvature of the loss only grows in the few views along a side. The balls keep the
smooth profile `k = 2`: with true discs ( `k = 0.5`, a root in every view ) their last stage takes twice the iterations.

On 5000 shapes, 720 views of 512 bins ( `experiments/exp_2d_polygons.py`, on the card ); the nominal radius found is
0.0105, 1.64 times the radius of the disc of the matter per ball; polygons of 8 vertices:

                              balls ( 16 s )    cells ( the start )    polygons ( 2830 it., 298 s )
    loss above its floor      1.34e-6           1.17e-6                5.2e-8
    image error               14.5 %            13.4 %                 3.4 %
    area covered twice                          7.1 %                  1.0 %
    object left empty                           5.9 %                  4.9 %

The MASSES of the polygons are free, and their densities stay the same all the same: within 1 % of each other at the end
( q10 0.99, q90 1.01 ). With ONE density for all ( masses as the areas ) the transport, which gives each view a unit
mass, makes a polygon that grows take its mass from every other one: the same loss takes twice the iterations

    iteration                 100        200        400        800        1600       the end
    one density               8.3e-7     6.1e-7     3.5e-7     2.1e-7     1.3e-7     1.06e-7 ( 1960 it. ), image error 6.6 %
    free masses               4.4e-7     2.8e-7     1.7e-7     1.0e-7     6.3e-8     5.5e-8 ( 2230 it. ), image error 3.5 %

What was tried and left: vertices free in the plane ( from the cells: 2.4e-7, 8.9 %, one polygon in 23 no longer turning
once around its centre; after the star polygons: nothing moves ), a stage of balls with a radius each before the polygons
( the polygons end at the same place ), cells cut by the whole balls ( those of a border reach out of the matter: 5.5 % ).

## After the balls: a mesh ( 2D )

The balls overlap ( `k = 2` ): their density is smooth, and its level lines are CONTOURS of the matter. `shapes/mesh.py`
takes them where the density is half the typical one, puts nodes along them and inside ( half a radius apart ), and keeps
the Delaunay triangles in the matter; the NODES are then moved to lower the loss. Of one density, the mesh projects as
its BORDER does: the kernels take its sides alone ( open lines, `fused.deposit_polygons( open = True )` ), and only the
nodes of the border move. Same case ( `experiments/exp_2d_mesh.py`: 46e3 triangles, built in 0.3 s ):

                              balls ( 13 s )    the mesh    nodes moved: 25 it.    100        400        670, the end ( 19 s )
    loss above its floor      1.40e-6           4.4e-4      3.9e-6                 8.3e-7     2.0e-7     1.2e-7
    image error               14.7 %            4.6 %       4.4 %                  3.4 %      2.4 %      1.9 %

No triangle turns over ( the smallest stays over 16 % of the mean one ). Balls with a radius each before the contours
change little: 4.2 % for the mesh after 1400 iterations of them. Open: nothing keeps the mesh sound as its border moves
( nodes are free along the border, the inner ones do not follow ); where the Delaunay triangulation misses a side of a
contour ( 0.04 % of them ) the border takes a long side ( 3.5 radii here ); the loss has a kink where a side is along a
view; contours shorter than 3 sides are left out.

## The number of views

With fewer views than bins the problem is under-determined, and it shows. On 256 bins and 5e4 shapes, 45 views fit the
sinogram to 4e-4 and give an image error ( relative L1 gap, pixel by pixel, between the density of the cloud and the
object ) of 42 %: the holes are found, but the mass piles up in the necks between them. 360 views give 8 %, a uniform
density in the material. Take about 1.5 views per bin.

## Experiments

From `~/Projects`, with errand ( `-o` brings the page back under `runs/otrec/experiments/…/latest/cloud.html` ):

    errand run --env torch -o 'otrec/experiments/exp_2d.py::*'      # 2D: 25e3 -> 1e5 shapes, 360 views of 256 bins
    errand run --env torch -o 'otrec/experiments/exp_3d.py::*'      # 3D, a small case: 2e4 shapes, 12 views of 64 x 64 pixels
    errand run --env lmo-torch -o 'otrec/experiments/exp_2d.py::*' --n_start=250000 --n=1000000 --bins=1024 --views=1440

The object is `ball_packing`, the start is random. Every size is a declared parameter ( `errand list`, or the top of each
file ). Each run writes one HTML page: the cloud frame by frame ( every shape as it is: balls at their radius, ellipses, polygons ) over the outline of
the object, and the curves ( loss, image error, blur, number of shapes ); a click in a curve goes to that iteration. The
final numbers go to the results of the run.

## Measured data

A `Scan` is one slice as a camera gives it. Each step makes a new `Scan`, and an example writes them in the order it needs:

```python
scan = Scan.from_counts( projections, flats, darks, angles )      # -log( ( p - dark ) / ( flat - dark ) )
scan = scan.centred( scan.rotation_axis(), radius = 800 )          # on the axis, the disk every view sees whole
scan = scan.minus_air( 0.03 )                                     # the level of the air, view by view
scan = scan.linearized()                                          # beam hardening: p + a p^2, equal masses
sino = scan.clipped().sinogram()                                 # 360 folded onto 180; a bin per pixel
```

When the object is entirely in the field ( the pencil ), two steps replace `clipped`. `zeroed_air( 3.0 )` sets what is
under 3 times the noise of the air to zero. Clipping alone keeps the positive half of the noise, a thin layer of matter
over the whole disk that draws shapes into the air. `equal_masses()` gives every view the mean mass, since every view
sees all the matter.

Such an object also says where the axis is and how the sample moved. Its views have the same mass, and the centre of
mass of view `t` follows `cor + a cos t + b sin t` ( consistency of the Radon transform at orders 0 and 1 ). `aligned()`
fits that sinusoid and shifts each view onto it. On the pencil, the axis given with the data ( 1024.28 ) was 0.9 pixel
off, and the sample moved by 1 pixel rms along the detector. Aligning lowers the loss above its floor by a third.
`rings_removed()` takes off the bias of each pixel of the detector: what the mean view departs from a cubic over 9
pixels. Phase contrast ( fringes at the interfaces, ID19 ) is not handled: a 1D Paganin filter gains little at 512 bins,
and its length would have to be guessed without the distance and the delta/beta of the beam.

A drift of the beam that scales a whole view needs no step, since the loss normalizes each view. The examples give the
URL of their data. `cached_download( url )` keeps the file. `raw_rows( url, shape, row )` fetches only one row of each
image of a raw stack, by HTTP ranges, which turns 12 GB of projections into 10 MB.

`experiments/exp_2d_real.py` covers binary objects ( matter or void ), from the simplest to the hardest. Its strategy is
one function ( `solve` ), written step by step. The first case is a metal foam ( Diamond I12 ), static and entirely in
the field, from the counts of the camera. Next comes sandstone ( SLS TOMCAT ), bigger than the field. Last is bamboo
( ESRF ID19 ), bigger than the field and with the rings of the detector. A pencil ( ESRF ID19 ) sits in the field
but is not binary: it shows what the balls do with densities they cannot have.

## What the views ask of each ball

In each view, the 1D transport sends the mass of a ball somewhere: the view asks it to move by `delta` along its axis.
The gradient sums these requests, and can vanish while the views disagree. `loss/fields.py` gives, per ball:

* `move`: the 2D move that fits the requests best, in radii. It is 0 once a stage has converged;
* `dissonance`: what remains of the requests after that move, in radii. These requests cancel out, so nothing moves,
  but the ball is not where the views want it;
* `density asked`: the mean of data / image over the cells the ball covers, minus 1. This is the change of density one EM
  step would give it ( > 0: the views see more matter there than the balls put ).

* `seen empty`: the fraction of the views that see the ball over air ( the data under it below 25 % of what it puts
  there ). A ball outside the convex hull of the matter has such views, provided the air is at 0 ( `Scan.zeroed_air` ).
  `Reconstruction.remove_in_air()` takes these balls off, and `remove_isolated()` takes off those with no neighbour.

With a density each ( `adjust_parameters( densities = True )` ), a field `density` shows it.

The frames of a `Recorder` keep them ( `Recorder( fields = True )`, the default; balls and sized balls, 2D ). The page
colours the shapes by one of them, with a range over the displayed frame, over all of them, or typed. Its menu also
shows or hides the layers: the outline of the truth, and the images given to `write_html( images = )` ( the FBP in
`exp_2d_real.py` ).

## Speed ( 2D )

One evaluation of the loss and its gradient, 360 views of 256 bins ( `bench/eval.py` ):

    shapes      lmo, CPU      lmo, RTX 2080 Ti
    1e5         0.27 s        0.045 s
    1e6         1.60 s        0.38 s

The whole default experiment ( 25e3, 50e3 then 1e5 shapes, 357 evaluations ): 19 s on the Mac, 6 s on the card, image
error 5.7 %. With 1e6 shapes, 1440 views of 1024 bins, on the card: 51 s, image error 4.7 %.

What it rests on:

* the deposit and its adjoint are everything ( the transport of all the views is 8 ms ): they are sdot's fused kernels.
  Written with tensor operations, an evaluation at 1e5 shapes took 2.4 s on the Mac ( every operation sweeps an array
  `[ views, shapes, window ]`, and autograd keeps them all );
* float64 throughout, for now. Float32 for the shapes ( the bins still in float64 ) is 7 times faster on the card and
  gives the same loss to 5e-8, but its noise stalls a stage before the shapes are still: under `x_tol = 0.01` the
  tolerance is never reached, and where a stage stalls changes from a run to the next on the card. To use again only
  where it is needed. ( With float32 shapes the card took 0.011 s and 0.046 s; the times of the
  whole experiments below were measured that way too. )
* L-BFGS on the device: scipy's cost as much as the loss at 1e6 shapes on the card;
* no blur: from a uniform start in the right disk, stages at 16, 8, 4, 2, 1 pixels of blur leave the image error at 0.4
  after 200 iterations, where stages without blur reach 0.07 in 120.

Shapes stay inside what every view sees. Beyond that border their centre is pulled back onto it, and gets no radial
derivative there. L-BFGS therefore brings each point it tries back inside ( `lbfgs( project = )`, `kind.project` ),
which leaves the loss unchanged. Without that, balls drawn at radius 0 and then given one, or pushed out during a stage,
stayed on the border: 96 balls of 2000 on the pencil, with a loss of 0.37 instead of 0.003.

The stop of a stage: the largest move of a shape is a poor signal ( a few shapes that cross a hole keep it up for
hundreds of iterations after the image has settled: 750 iterations for the first stage at 250e3 shapes, where the
99.9 % quantile stops it at 300 ).

Not gained: a Morton order of the shapes and a larger batch of views change nothing at 256 bins.
