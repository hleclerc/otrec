"""The deposit of the shapes on a 1D detector and its adjoint, as FUSED KERNELS ( `sdot/Projected1d.h`, through loom; CPU
or CUDA ): one pass over the shapes for all the views, the projections recomputed on the fly, nothing of size
`[ views, shapes ]` ever stored. Each shape puts in every cell it covers its EXACT mass ( the primitive of its profile ).

Everything is in FLOAT64. In float32 for the shapes ( the cells still adding up in float64 ) a card is 7 times faster and
the loss is the same to 5e-8, but its noise then stalls a stage before the shapes are still: a tolerance under a
hundredth of a radius is never reached, and where it stalls changes from a run to the next on a card. To take up again
where speed matters more than the last digits.

A grid: `N` cells of width `h` from `origin`; `normals [ V, 2 ]`: the detector axis of each view."""
import math

import torch


def _call( name, code, points, normals, N, h, origin, radius, k, outputs, **more ):
    import loom
    from loom.compilation.FfiCode import FfiCode
    from sdot.projected1d import _dome_coeffs
    n, L = points.shape[ 0 ], int( math.ceil( 2 * radius / h - 1e-9 ) ) + 1             # L: the cells a shape may cover
    c, a = _dome_coeffs( k )
    f64 = dict( dtype = torch.float64, device = points.device )
    loom.ffi_call( name, FfiCode.inline( f"sdot::projected1d::{ code.format( tpl = f'1,{ k },{ L }' ) }", includes = [ "sdot/Projected1d.h" ] ),
                   points = points[ None ], radii = torch.full( ( 1, n ), radius, **f64 ), normals = normals,
                   params = torch.tensor( [ origin, h, radius, 1 / n, c, *a ], **f64 ),
                   has_dynamic_capacity = False, **more, **{ key: loom.out( t ) for key, t in outputs.items() } )


def _axes( points, normals, N ):
    from loom import Axis, ShapeVar
    ax = lambda size, name: Axis( ShapeVar( size ), name = name )
    return ax( 1, "num_problem" ), ax( normals.shape[ 0 ], "num_angle" ), ax( points.shape[ 0 ], "num_point" ), ax( N, "num_cell" ), ax( 2, "dim" )


def deposit( points, normals, N, h, origin, radius, k ):
    """`[ V, N ]`: the mass each cell receives, in each view, from the shapes at `points [ n, 2 ]` ( masses `1 / n` )"""
    from loom import RealTensor
    one, views, _, cells, _ = _axes( points, normals, N )
    hist = RealTensor[ one, views, cells, dict( size = 64 ) ]()
    _call( "projected1d_deposit", "deposit<{tpl}>( queue, args.inputs.points, args.inputs.radii, args.inputs.normals, args.inputs.params, args.outputs.hist );",
           points, normals, N, h, origin, radius, k, dict( hist = hist ) )
    return hist.value[ 0 ]


def gather( points, normals, N, h, origin, radius, k, d_cells ):
    """`[ n, 2 ]`: the derivative with respect to the points of a function whose derivative with respect to the
    cell masses is `d_cells [ V, N ]` ( the adjoint of `deposit` )"""
    from loom import RealTensor
    one, views, pts, _, dim = _axes( points, normals, N )
    grad, grad_radii = RealTensor[ one, pts, dim, dict( size = 64 ) ](), RealTensor[ one, pts, dict( size = 64 ) ]()
    # the kernel reads the differences of `d_cells` between neighbour cells, on the N + 1 edges ( 0 on the two end ones )
    dg = torch.nn.functional.pad( d_cells[ :, 1: ] - d_cells[ :, :-1 ], ( 1, 1 ) )
    _call( "projected1d_gather", "gather<{tpl}>( queue, args.inputs.points, args.inputs.radii, args.inputs.normals, args.inputs.params, args.inputs.g, "
           "args.outputs.grad_points, args.outputs.grad_radii );",
           points, normals, N, h, origin, radius, k, dict( grad_points = grad, grad_radii = grad_radii ), g = dg[ None ] )
    return grad.value[ 0 ]
