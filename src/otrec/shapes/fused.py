"""The deposit of the shapes on a 1D detector and its adjoint, as FUSED KERNELS ( `sdot/Projected1d.h`, through loom; CPU
or CUDA ): one pass over the shapes for all the views, the projections recomputed on the fly, nothing of size
`[ views, shapes ]` ever stored. Each shape puts in every cell it covers its EXACT mass ( the primitive of its profile ).

Three kinds of shapes: BALLS of equal masses, each with its radius ( `deposit`, `gather` ), ELLIPSES with masses of their own
( `deposit_ellipses`, `gather_ellipses`; an ellipse is given by `C = M M^T` as `( c00, c11, c01 )`, M the matrix that maps
the unit ball onto it: its half-width along a view of axis `n` is `sqrt( n.C n )` ), and POLYGONS of density 1 given by
their vertices ( `deposit_polygons`, `gather_polygons` ).

The tensors are FLOAT64. `PRECISION` ( balls and ellipses ) says what the kernels compute in float32 ( on a card of the
GeForce kind, float64 is 1/32 of float32: it is what they cost ): 0 nothing; 1 the profile ( the parameter of each edge
and the primitive there ), the projections, windows and sums in float64; 2 the sums too ( along a window and over the
views ); 3 the projections and windows too. The cells always add up in float64. By default 2: 3 times faster on a card
( 100k balls ), errors ~1e-7 relative on the gradient; on the CPU no gain.

A grid: `N` cells of width `h` from `origin`; `normals [ V, 2 ]`: the detector axis of each view."""
import math

import torch


PRECISION = 2


def _dome_coeffs( k ):
    """c_k and the coefficients of G( t ) - 1/2 = c_k sum_j C( k, j ) ( -1 )^j t^( 2j+1 ) / ( 2j+1 ), as `sdot.projected1d`
    has them ( not imported: that module brings jax )"""
    c = math.gamma( k + 1.5 ) / ( math.sqrt( math.pi ) * math.gamma( k + 1 ) )
    return c, [ c * math.comb( k, j ) * ( -1 ) ** j / ( 2 * j + 1 ) for j in range( k + 1 ) ]


def window_cells( max_width, h ):
    """POLYGONS: the cells one may cover, `ceil( 2 max_width / h ) + 1`, ROUNDED UP to 4 steps per octave ( ..., 8, 10,
    12, 14, 16, 20, 24, ...: at most 1.25 times more ): a static size of their kernels, and a radius that changes with
    the number of shapes would otherwise compile them again at each new value. The cells past the support receive
    nothing. ( Balls and ellipses take a window of runtime size: one kernel for every width. )"""
    L = int( math.ceil( 2 * max_width / h - 1e-9 ) ) + 1
    if L <= 5:
        return L
    q = ( 1 << ( ( L - 1 ).bit_length() - 1 ) ) // 4                                  # a quarter of the power of two under L
    return -( -L // q ) * q


def _call( name, code, normals, h, origin, max_width, k, param3, tensors ):
    """one kernel of `sdot::projected1d`; `tensors`: its other inputs, and its outputs as `( axes, )` to allocate.
    Returns the outputs, the batch axis the kernels work with removed"""
    import loom
    import sdot                                                   # ( registers its headers with loom: sdot/Projected1d.h )
    from loom import Axis, RealTensor, ShapeVar
    from loom.compilation.FfiCode import FfiCode
    L = window_cells( max_width, h )                                                  # ( polygons only: `{L}` in their code )
    c, a = _dome_coeffs( k ) if k != 0.5 else ( 2 / math.pi, [ 0.0 ] )           # ( 0.5: true discs, K = 0 for the kernels )
    outputs = { key: RealTensor[ tuple( Axis( ShapeVar( size ), name = f"{ key }_{ d }" ) for d, size in enumerate( ( 1, *t[ 0 ] ) ) ) + ( dict( size = 64 ), ) ]()
                for key, t in tensors.items() if isinstance( t, tuple ) }
    inputs = { key: t[ None ] for key, t in tensors.items() if not isinstance( t, tuple ) }
    args = ", ".join( [ "queue" ] + [ f"args.{ 'outputs' if key in outputs else 'inputs' }.{ key }" for key in code[ 1 ] ] )
    loom.ffi_call( name, FfiCode.inline( f"sdot::projected1d::{ code[ 0 ].format( k = int( k ), L = L, p = int( PRECISION ) ) }( { args } );", includes = [ "sdot/Projected1d.h" ] ),
                   normals = normals, params = torch.tensor( [ origin, h, max_width, param3, c, *a ], dtype = torch.float64, device = normals.device ),
                   has_dynamic_capacity = False, **inputs, **{ key: loom.out( t ) for key, t in outputs.items() } )
    return [ t.value[ 0 ] for t in outputs.values() ]


# -- balls -----------------------------------------------------------------------------------------------------------------

def deposit( points, radii, normals, N, h, origin, max_width, k ):
    """`[ V, N ]`: the mass each cell receives, in each view, from the balls at `points [ n, 2 ]`, of radii `radii [ n ]`
    ( at most `max_width` ) and masses `1 / n`"""
    return _call( "projected1d_deposit", ( "deposit<1,{k},{p}>", ( "points", "radii", "normals", "params", "hist" ) ), normals, h, origin, max_width, k, 1 / len( points ),
                  dict( points = points, radii = radii, hist = ( ( normals.shape[ 0 ], N ), ) ) )[ 0 ]


def gather( points, radii, normals, N, h, origin, max_width, k, d_cells ):
    """`( [ n, 2 ], [ n ] )`: the derivatives with respect to the points and the radii of a function whose derivative with
    respect to the cell masses is `d_cells [ V, N ]` ( the adjoint of `deposit` )"""
    # the kernel reads the differences of `d_cells` between neighbour cells, on the N + 1 edges ( 0 on the two end ones )
    dg, n = torch.nn.functional.pad( d_cells[ :, 1: ] - d_cells[ :, :-1 ], ( 1, 1 ) ), len( points )
    return _call( "projected1d_gather", ( "gather<1,{k},{p}>", ( "points", "radii", "normals", "params", "g", "grad_points", "grad_radii" ) ), normals, h, origin, max_width, k, 1 / n,
                  dict( points = points, radii = radii, g = dg, grad_points = ( ( n, 2 ), ), grad_radii = ( ( n, ), ) ) )


# -- ellipses --------------------------------------------------------------------------------------------------------------

def deposit_ellipses( points, covs, masses, normals, N, h, origin, min_width, max_width, k ):
    """`[ V, N ]`: the mass each cell receives, in each view, from the ellipses at `points [ n, 2 ]`, of matrices
    `covs [ n, 3 ]` and masses `masses [ n ]`; their half-widths are kept in `[ min_width, max_width ]`"""
    return _call( "projected1d_deposit_ellipses", ( "deposit_ellipses<{k},{p}>", ( "points", "covs", "masses", "normals", "params", "hist" ) ), normals, h, origin, max_width, k, min_width,
                  dict( points = points, covs = covs, masses = masses, hist = ( ( normals.shape[ 0 ], N ), ) ) )[ 0 ]


def gather_ellipses( points, covs, masses, normals, N, h, origin, min_width, max_width, k, d_cells ):
    """`( [ n, 2 ], [ n, 3 ], [ n ] )`: the derivatives with respect to the points, the matrices and the masses of a function
    whose derivative with respect to the cell masses is `d_cells [ V, N ]` ( the adjoint of `deposit_ellipses` )"""
    n = len( points )
    return _call( "projected1d_gather_ellipses", ( "gather_ellipses<{k},{p}>", ( "points", "covs", "masses", "normals", "params", "g", "grad_points", "grad_covs", "grad_masses" ) ),
                  normals, h, origin, max_width, k, min_width,
                  dict( points = points, covs = covs, masses = masses, g = d_cells, grad_points = ( ( n, 2 ), ), grad_covs = ( ( n, 3 ), ), grad_masses = ( ( n, ), ) ) )


# -- polygons --------------------------------------------------------------------------------------------------------------

def deposit_polygons( verts, weights, normals, N, h, origin, max_width, open = False ):
    """`[ V, N ]`: the mass each cell receives, in each view, from the polygons of vertices `verts [ n, nb_vertices, 2 ]`
    ( counter-clockwise, simple, no wider than `2 max_width` ) and of densities `weights [ n ]`. `open`: the side from the last
    vertex back to the first is left out -- LINES, each for the area between it and the axis of the detector through the
    origin ( the lines of a closed border, the matter on their left, give what is inside )"""
    nv = verts.shape[ 1 ]
    return _call( "projected1d_deposit_polygons", ( f"deposit_polygons<{ nv },{{L}},{ int( open ) }>", ( "verts", "weights", "normals", "params", "hist" ) ), normals, h, origin, max_width, 0.5, 0.0,
                  dict( verts = verts.reshape( len( verts ), -1 ), weights = weights, hist = ( ( normals.shape[ 0 ], N ), ) ) )[ 0 ]


def gather_polygons( verts, weights, normals, N, h, origin, max_width, d_cells, open = False ):
    """`( [ n, nb_vertices, 2 ], [ n ] )`: the derivatives with respect to the vertices and the densities of a function whose
    derivative with respect to the cell masses is `d_cells [ V, N ]` ( the adjoint of `deposit_polygons` )"""
    n, nv = verts.shape[ :2 ]
    gv, gw = _call( "projected1d_gather_polygons", ( f"gather_polygons<{ nv },{{L}},{ int( open ) }>", ( "verts", "weights", "normals", "params", "g", "grad_verts", "grad_weights" ) ),
                    normals, h, origin, max_width, 0.5, 0.0,
                    dict( verts = verts.reshape( n, -1 ), weights = weights, g = d_cells, grad_verts = ( ( n, 2 * nv ), ), grad_weights = ( ( n, ), ) ) )
    return gv.reshape( n, nv, 2 ), gw
