# otrec

Tomographic reconstruction by optimal transport, in 2D ( 1D detector ) and 3D ( 2D detector ), on PyTorch and sdot.

The object is a cloud of SHAPES, small pieces of matter of equal masses ( 1e5 to 1e7 of them for a continuous material
full of holes ). For each view, the measured pixels are diracs on a grid ( mass = intensity ) and the shapes are projected
and discretized on an image; the positions of the shapes minimize

    L( points ) = sum_views  W2^2(  diracs of the view ,  image of the projected shapes  )

```python
from otrec import Reconstruction, Sinogram, ball_packing

image = ball_packing( 512 )                                           # an object: a disc full of round holes
sino = Sinogram.from_image( image, nb_views = 360, nb_bins = 256 )    # its data ( or Sinogram( ..., values = measured ) )
rec = Reconstruction( sino )
rec.multiscale( 25_000, 100_000, radius = 0.8 )                       # random in a disk, then stages: 25e3, 50e3, 1e5 balls
rec.points, rec.radius                                                # [ 100_000, 2 ], the radius the cloud found
```

## Layout

Each directory has its own docstring ( `__init__.py` ) that says what its files do.

    src/otrec/
        Sinogram.py          the data: geometry of the views, intensities, the diracs; `from_image`
        projection.py        the line integrals of an image ( what `from_image` runs )
        Reconstruction.py    the cloud and the steps: random_points, split, lbfgs ( one stage ), multiscale, resize, to_polygons
        lbfgs.py             L-BFGS on tensors, on their device, stopped when the shapes no longer move
        cloud.py             random starts, split, spacing, the sizes of a multiscale ( arrays )
        Recorder.py          frames of a reconstruction, with metrics
        metrics.py           image error, residual of the sinogram
        viz.py               the frames as an HTML page ( sdot.Visualizer )
        shapes/              balls, sized ( a radius each ), polygons, mesh ( triangles between the contours of the balls ); their deposit on bins ( 1D, sdot kernels ) or nodes ( 2D ), blur
        transport/           W2^2 of one view: closed form ( 1D ), sdot's grid solver ( 2D )
        loss/                the sum over the views and its gradient: Model, Loss1d ( any of the shapes ), Loss2d
        packing/             ball_packing: the image of a disc ( a cylinder ) minus balls settled by gravity
    experiments/             exp_2d.py, exp_3d.py: a packing reconstructed from a random start ( errand entries );
                             exp_2d_polygons.py: balls, then polygons; exp_2d_mesh.py: balls, then a mesh of their contours
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
  again before each stage. 3D: the same formula, not calibrated.
* **The observable domain**. A shape must stay where every view sees it whole: its centre is pulled back so that centre
  + reach stays in the centred ball of diameter `extent` ( `shapes.balls.inside` ).
* **Transport** ( `transport/` ). 1D: the plan is monotone, W2^2 has a closed form that autograd differentiates
  ( float64, a few ms ). 2D: sdot's dedicated grid solver ( `nb_bins` a power of two from 32 ), warm started from the
  last solve of the view; the derivative with respect to the image is the Kantorovich potential.
* **Recipes** ( `Reconstruction` ). A STAGE ( `lbfgs` ) is an L-BFGS on the centres, on the device, that stops when
  99.9 % of the shapes have moved by less than `move_tol` of their radius ( 0.1 ) over 10 iterations.
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
( `move_tol`, a fraction of their radius ). Nothing is counted in pixels: the number of bins is a parameter like another.

## After the balls: polygons ( 2D )

Balls of one radius give the large masses of matter, poorly its walls and its borders. `rec.to_polygons( nb_vertices )`
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
file ). Each run writes one HTML page: the cloud frame by frame ( every shape, at its true radius ) over the outline of
the object, and the curves ( loss, image error, blur, number of shapes ); a click in a curve goes to that iteration. The
final numbers go to the results of the run.

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
  gives the same loss to 5e-8, but its noise stalls a stage before the shapes are still: under `move_tol = 0.01` the
  tolerance is never reached, and where a stage stalls changes from a run to the next on the card. To use again only
  where it is needed. ( With float32 shapes the card took 0.011 s and 0.046 s; the times of the
  whole experiments below were measured that way too. )
* L-BFGS on the device: scipy's cost as much as the loss at 1e6 shapes on the card;
* no blur: from a uniform start in the right disk, stages at 16, 8, 4, 2, 1 pixels of blur leave the image error at 0.4
  after 200 iterations, where stages without blur reach 0.07 in 120.

The stop of a stage: the largest move of a shape is a poor signal ( a few shapes that cross a hole keep it up for
hundreds of iterations after the image has settled: 750 iterations for the first stage at 250e3 shapes, where the
99.9 % quantile stops it at 300 ).

Not gained: a Morton order of the shapes and a larger batch of views change nothing at 256 bins.
