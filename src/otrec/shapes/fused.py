"""The deposit of the shapes on a 1D detector and its adjoint, as FUSED KERNELS ( `sdot/Projected1d.h`, through loom; CPU
or CUDA ): one pass over the shapes for all the views, the projections recomputed on the fly, nothing of size
`[ views, shapes ]` ever stored. Each shape puts in every cell it covers its EXACT mass ( the primitive of its profile ).

Two kinds of shapes: BALLS of one radius and equal masses ( `deposit`, `gather` ), and ELLIPSES with masses of their own
( `deposit_ellipses`, `gather_ellipses`; an ellipse is given by `C = M M^T` as `( c00, c11, c01 )`, M the matrix that maps
the unit ball onto it: its half-width along a view of axis `n` is `sqrt( n.C n )` ).

Everything is in FLOAT64. In float32 for the shapes ( the cells still adding up in float64 ) a card is 7 times faster and
the loss is the same to 5e-8, but its noise then stalls a stage before the shapes are still: a tolerance under a
hundredth of a radius is never reached, and where it stalls changes from a run to the next on a card. To take up again
where speed matters more than the last digits.

A grid: `N` cells of width `h` from `origin`; `normals [ V, 2 ]`: the detector axis of each view."""
import math

import torch


def _call( name, code, normals, h, origin, max_width, k, param3, tensors ):
    """one kernel of `sdot::projected1d`; `tensors`: its other inputs, and its outputs as `( axes, )` to allocate.
    Returns the outputs, the batch axis the kernels work with removed"""
    import loom
    from loom import Axis, RealTensor, ShapeVar
    from loom.compilation.FfiCode import FfiCode
    from sdot.projected1d import _dome_coeffs
    L = int( math.ceil( 2 * max_width / h - 1e-9 ) ) + 1                                 # the cells a shape may cover
    c, a = _dome_coeffs( k )
    outputs = { key: RealTensor[ tuple( Axis( ShapeVar( size ), name = f"{ key }_{ d }" ) for d, size in enumerate( ( 1, *t[ 0 ] ) ) ) + ( dict( size = 64 ), ) ]()
                for key, t in tensors.items() if isinstance( t, tuple ) }
    inputs = { key: t[ None ] for key, t in tensors.items() if not isinstance( t, tuple ) }
    args = ", ".join( [ "queue" ] + [ f"args.{ 'outputs' if key in outputs else 'inputs' }.{ key }" for key in code[ 1 ] ] )
    loom.ffi_call( name, FfiCode.inline( f"sdot::projected1d::{ code[ 0 ].format( k = k, L = L ) }( { args } );", includes = [ "sdot/Projected1d.h" ] ),
                   normals = normals, params = torch.tensor( [ origin, h, max_width, param3, c, *a ], dtype = torch.float64, device = normals.device ),
                   has_dynamic_capacity = False, **inputs, **{ key: loom.out( t ) for key, t in outputs.items() } )
    return [ t.value[ 0 ] for t in outputs.values() ]


# -- balls -----------------------------------------------------------------------------------------------------------------

def _radii( points, radius ):
    return torch.full( ( len( points ), ), radius, dtype = points.dtype, device = points.device )


def deposit( points, normals, N, h, origin, radius, k ):
    """`[ V, N ]`: the mass each cell receives, in each view, from the balls at `points [ n, 2 ]` ( masses `1 / n` )"""
    return _call( "projected1d_deposit", ( "deposit<1,{k},{L}>", ( "points", "radii", "normals", "params", "hist" ) ), normals, h, origin, radius, k, 1 / len( points ),
                  dict( points = points, radii = _radii( points, radius ), hist = ( ( normals.shape[ 0 ], N ), ) ) )[ 0 ]


def gather( points, normals, N, h, origin, radius, k, d_cells ):
    """`[ n, 2 ]`: the derivative with respect to the points of a function whose derivative with respect to the cell masses
    is `d_cells [ V, N ]` ( the adjoint of `deposit` )"""
    # the kernel reads the differences of `d_cells` between neighbour cells, on the N + 1 edges ( 0 on the two end ones )
    dg, n = torch.nn.functional.pad( d_cells[ :, 1: ] - d_cells[ :, :-1 ], ( 1, 1 ) ), len( points )
    return _call( "projected1d_gather", ( "gather<1,{k},{L}>", ( "points", "radii", "normals", "params", "g", "grad_points", "grad_radii" ) ), normals, h, origin, radius, k, 1 / n,
                  dict( points = points, radii = _radii( points, radius ), g = dg, grad_points = ( ( n, 2 ), ), grad_radii = ( ( n, ), ) ) )[ 0 ]


# -- ellipses --------------------------------------------------------------------------------------------------------------

def deposit_ellipses( points, covs, masses, normals, N, h, origin, min_width, max_width, k ):
    """`[ V, N ]`: the mass each cell receives, in each view, from the ellipses at `points [ n, 2 ]`, of matrices
    `covs [ n, 3 ]` and masses `masses [ n ]`; their half-widths are kept in `[ min_width, max_width ]`"""
    return _call( "projected1d_deposit_ellipses", ( "deposit_ellipses<{k},{L}>", ( "points", "covs", "masses", "normals", "params", "hist" ) ), normals, h, origin, max_width, k, min_width,
                  dict( points = points, covs = covs, masses = masses, hist = ( ( normals.shape[ 0 ], N ), ) ) )[ 0 ]


def gather_ellipses( points, covs, masses, normals, N, h, origin, min_width, max_width, k, d_cells ):
    """`( [ n, 2 ], [ n, 3 ], [ n ] )`: the derivatives with respect to the points, the matrices and the masses of a function
    whose derivative with respect to the cell masses is `d_cells [ V, N ]` ( the adjoint of `deposit_ellipses` )"""
    n = len( points )
    return _call( "projected1d_gather_ellipses", ( "gather_ellipses<{k},{L}>", ( "points", "covs", "masses", "normals", "params", "g", "grad_points", "grad_covs", "grad_masses" ) ),
                  normals, h, origin, max_width, k, min_width,
                  dict( points = points, covs = covs, masses = masses, g = d_cells, grad_points = ( ( n, 2 ), ), grad_covs = ( ( n, 3 ), ), grad_masses = ( ( n, ), ) ) )
