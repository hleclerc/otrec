"""The SHAPES: radial kernels whose projection is a POLYNOMIAL of the distance to the projected centre.

In dimension d ( 2 or 3 ), the radial density ( 1 - |x|^2 / R^2 )^( k - 1/2 ) projects along any direction onto

    p( t ) = c ( 1 - |t|^2 / R^2 )^k,        t in R^( d - 1 ), |t| < R,

the same polynomial in every view ( k an integer >= 1; the larger, the smoother ). World 2D: `k = 0.5` is the TRUE disc, of
density 1 ( a root in place of the polynomial ), and polygons are of density 1 as well.

    balls           one radius, equal masses: `x [ n, dim ]`                              } what a cloud is; each gives
    sized           a radius each, a mass that follows it: `x [ n, 3 ]`      ( world 2D )   } `shapes( x, model )`, what the
    polygons        star polygons, a density each: `x [ n, 5 + 2 V ]`               ( world 2D )   } kernels take
    mesh            triangles of one density between the contours of balls: `x [ m, 2 ]`, the nodes   ( world 2D )
    fused           1D detector: the exact mass each bin receives, and the adjoint      -> a P0 image   ( sdot kernels )
    deposit_nodes   2D detector: the density at the corners of the pixels               -> a P1 image   ( autograd )
    blur            the gaussian blur of a row of samples"""
from .blur import gaussian_matrix as gaussian_matrix
from .deposit_nodes import deposit_nodes as deposit_nodes
