"""The calls to sdot for a 2D detector: the diracs are those of a grid ( `sdot.SumOfDiracsOnAGrid`, one per cell, at its
centre ), the image is P1 ( `sdot.Image( degree = 1 )`, its nodes at the corners of the cells ), so that the solve takes the
dedicated grid solver. A grid: `N x N` cells ( N a power of two, from 32 ) of side `h` from the corner `origin`.
Everything goes in and out as PyTorch tensors."""
import numpy as np
import torch


def corner_coeffs( v ):
    """the Bernstein coefficients of the cells of a P1 image, `[ N, N, 4 ]`, from its nodal values `[ N + 1, N + 1 ]`"""
    return torch.stack( [ v[ :-1, :-1 ], v[ :-1, 1: ], v[ 1:, :-1 ], v[ 1:, 1: ] ], -1 )


def nodes_of_coeffs( c ):
    """the adjoint of `corner_coeffs`: `[ N, N, 4 ]` -> `[ N + 1, N + 1 ]`"""
    N = c.shape[ 0 ]
    v = torch.zeros( N + 1, N + 1, dtype = c.dtype, device = c.device )
    v[ :-1, :-1 ] += c[ ..., 0 ]
    v[ :-1, 1: ] += c[ ..., 1 ]
    v[ 1:, :-1 ] += c[ ..., 2 ]
    v[ 1:, 1: ] += c[ ..., 3 ]
    return v


def _image( v, origin, h ):
    import sdot
    return sdot.Image( values = corner_coeffs( v ), degree = 1, pixel_frame = [ [ h, 0.0 ], [ 0.0, h ] ], origin = list( origin ) )


def solve( b, v, origin, h, weights0 = None, tol = 1e-9, max_iter = 200 ):
    """the plan between the diracs of masses `b [ N, N ]` and the image of nodal values `v [ N + 1, N + 1 ]`
    ( `plan.cost`, `plan.weights`; see `converged` )"""
    import sdot
    N = b.shape[ 0 ]
    src = sdot.SumOfDiracsOnAGrid( ( N, N ), weights = b.reshape( -1 ), cell_frame = [ [ h, 0.0 ], [ 0.0, h ] ], origin = origin )
    return sdot.OtProblem( src, _image( v, origin, h ) ).solve( sdot.Iterative(
        tol = tol, max_iter = max_iter, weights0 = weights0, on_failure = "ignore", tuning = sdot.Tuning( grid_solver = "dedicated" ) ) )


def converged( plan ):
    return str( np.asarray( plan.stats[ "status" ] ) ) == "converged"


def potential_integrals( weights, v, origin, h ):
    """`[ N + 1, N + 1 ]`: per node, the integral of the Kantorovich potential against its basis function -- the derivative
    of the cost with respect to the nodal values, for an image of fixed mass ( envelope theorem )"""
    import sdot
    N = v.shape[ 0 ] - 1
    c = ( np.arange( N ) + 0.5 ) * h
    pos = np.stack( np.meshgrid( origin[ 0 ] + c, origin[ 1 ] + c, indexing = "ij" ), -1 ).reshape( -1, 2 )
    pd = sdot.PowerDiagram( torch.as_tensor( pos, dtype = v.dtype, device = v.device ), weights, distribution = _image( v, origin, h ), kernel_dtype = "FP64" )
    return nodes_of_coeffs( pd.potential_integrals() )


def node_areas( N, h, like ):
    """`[ N + 1, N + 1 ]`: the integral of each basis function ( the mass of the image is `sum( v * node_areas )` )"""
    return nodes_of_coeffs( torch.full( ( N, N, 4 ), h * h / 4, dtype = like.dtype, device = like.device ) )
