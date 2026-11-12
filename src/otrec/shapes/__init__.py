"""The SHAPES: radial kernels whose projection is a POLYNOMIAL of the distance to the projected centre.

In dimension d ( 2 or 3 ), the radial density ( 1 - |x|^2 / R^2 )^( k - 1/2 ) projects along any direction onto

    p( t ) = c ( 1 - |t|^2 / R^2 )^k,        t in R^( d - 1 ), |t| < R,

the same polynomial in every view ( k an integer >= 1; the larger, the smoother ). The `n` shapes of a cloud have equal
masses `1 / n`.

    fused           1D detector: the exact mass each bin receives, and the adjoint      -> a P0 image   ( sdot kernels )
    deposit_nodes   2D detector: the density at the corners of the pixels               -> a P1 image   ( autograd )
    ellipses        1D detector: shapes stretched into ellipses, with masses of their own       ( autograd )
    blur            the gaussian blur of a row of samples"""
from .blur import gaussian_matrix as gaussian_matrix
from .deposit_nodes import deposit_nodes as deposit_nodes
