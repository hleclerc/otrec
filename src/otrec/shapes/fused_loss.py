"""The loss of balls in ONE kernel ( `sdot/Projected1dFused.h`, through loom; CPU or CUDA ): for every view, the deposit of
the balls, the squared Wasserstein distance to the data, and its derivatives with respect to the balls, the image and
its potential kept on the card ( three kernels in the call, the rows in a scratch of the card ). Parallel geometry for now:
the kernel takes the geometry as a parameter ( `Parallel` ), so that a sample that moves or a fan beam adds a geometry,
not a kernel.

    loss, grad_points, grad_radii, grad_masses = balls_loss( points, radii, normals, data, h, origin, max_width, k, background )

`radii`: one value ( every ball that radius, masses 1 / n ), or one per ball with a mass each ( sized balls: `masses`;
`grads`, the derivatives with respect to radii and masses too ). The image is lifted by `background` times its mean;
`data [ V, N ]` as given ( the masses of the diracs at the centres of the cells, lifted already ). What the kernel
computes in float32: `fused.PRECISION` ( 1, the profile; 2, the sums too; 3 is taken as 2 here )."""
import torch

from . import fused
from .fused import _dome_coeffs


def balls_loss( points, radii, normals, data, h, origin, max_width, k, background, masses = None, grads = False, block = 256, tile = 0, debug = 0 ):
    # `tile` ( the card ): 0, three kernels in the call, the rows in global memory ( the default: the fastest ); > 0 the
    # same, that many tiles of views; < 0, one kernel, `-tile` views a block, their rows in shared memory
    """`( loss [ V ], grad_points [ n, 2 ], grad_radii [ n ] or None, grad_masses [ n ] or None )`. `radii`: one value
    ( every ball that radius, masses 1 / n ), or one per ball with `masses` ( sized balls ); `grads`: then the
    derivatives with respect to the radii ( at fixed mass ) and the masses too"""
    import loom
    import sdot                                                   # ( registers its headers with loom )
    from loom import Axis, RealTensor, ShapeVar
    from loom.compilation.FfiCode import FfiCode
    mode = 0 if masses is None else ( 2 if grads else 1 )
    c, a = _dome_coeffs( k ) if k != 0.5 else ( 2 / torch.pi, [ 0.0 ] )
    K = int( k ) if k != 0.5 else 0
    n, V = len( points ), normals.shape[ 0 ]
    M = data.shape[ 1 ]
    card = points.device.type == "cuda"                           # ( the rows in global memory: the card's split path )
    outs = dict( loss = ( V, ), grad_points = ( n, 2 ), grad_radii = ( n if mode == 2 else 1, ), grad_masses = ( n if mode == 2 else 1, ),
                 rows = ( V, M ) if card else ( 1, 1 ) )
    out = { key: RealTensor[ tuple( Axis( ShapeVar( size ), name = f"{ key }_{ d }" ) for d, size in enumerate( ( 1, *shape ) ) ) + ( dict( size = 64 ), ) ]()
            for key, shape in outs.items() }
    names = [ "points", "radii", "masses", "normals", "params", "data", "loss", "grad_points", "grad_radii", "grad_masses", "rows" ]
    args = ", ".join( [ "queue" ] + [ f"args.{ 'outputs' if key in out else 'inputs' }.{ key }" for key in names ] )
    params = torch.tensor( [ origin, h, max_width, background, c, *a ], dtype = torch.float64, device = normals.device )
    ms = masses if masses is not None else radii.reshape( -1 )[ :1 ]
    loom.ffi_call( "projected1d_fused_loss_balls",
                   FfiCode.inline( f"sdot::projected1d_fused::loss_balls<{ K },{ mode },{ min( int( fused.PRECISION ), 2 ) },{ block },{ tile },{ debug }>( { args } );", includes = [ "sdot/Projected1dFused.h" ] ),
                   points = points[ None ], radii = radii.reshape( -1 )[ None ], masses = ms[ None ], normals = normals, params = params, data = data[ None ],
                   has_dynamic_capacity = False, **{ key: loom.out( t ) for key, t in out.items() } )
    loss, gp, gr, gm = ( out[ key ].value[ 0 ] for key in ( "loss", "grad_points", "grad_radii", "grad_masses" ) )
    return loss, gp, ( gr if mode == 2 else None ), ( gm if mode == 2 else None )


def balls_fields( points, radii, normals, data, h, origin, max_width, k, background, masses = None ):
    """`{ "move", "dissonance", "density asked", "seen empty" }`, each `[ n ]`: what the views ask of each ball
    ( `loss.fields`, for the same loss ), in one call ( a deposit, a transport, then the sums of each ball over the views,
    in double )"""
    import loom
    import sdot
    from loom import Axis, RealTensor, ShapeVar
    from loom.compilation.FfiCode import FfiCode
    mode = 0 if masses is None else 1
    c, a = _dome_coeffs( k ) if k != 0.5 else ( 2 / torch.pi, [ 0.0 ] )
    K = int( k ) if k != 0.5 else 0
    n, V, M = len( points ), normals.shape[ 0 ], data.shape[ 1 ]
    card = points.device.type == "cuda"
    outs = dict( fb = ( n, 2 ), fs = ( n, 3 ), rows = ( V, M ) if card else ( 1, 1 ), image = ( V, M ) if card else ( 1, 1 ), sums = ( V, 2 ) if card else ( 1, 1 ) )
    out = { key: RealTensor[ tuple( Axis( ShapeVar( size ), name = f"{ key }_{ d }" ) for d, size in enumerate( ( 1, *shape ) ) ) + ( dict( size = 64 ), ) ]()
            for key, shape in outs.items() }
    names = [ "points", "radii", "masses", "normals", "params", "data", "fb", "fs", "rows", "image", "sums" ]
    args = ", ".join( [ "queue" ] + [ f"args.{ 'outputs' if key in out else 'inputs' }.{ key }" for key in names ] )
    params = torch.tensor( [ origin, h, max_width, background, c, *a ], dtype = torch.float64, device = normals.device )
    ms = masses if masses is not None else radii.reshape( -1 )[ :1 ]
    loom.ffi_call( "projected1d_fused_fields_balls",
                   FfiCode.inline( f"sdot::projected1d_fused::fields_balls<{ K },{ mode }>( { args } );", includes = [ "sdot/Projected1dFused.h" ] ),
                   points = points[ None ], radii = radii.reshape( -1 )[ None ], masses = ms[ None ], normals = normals, params = params, data = data[ None ],
                   has_dynamic_capacity = False, **{ key: loom.out( t ) for key, t in out.items() } )
    fb, fs = out[ "fb" ].value[ 0 ], out[ "fs" ].value[ 0 ]
    r = radii.reshape( -1 ).expand( n ) if radii.numel() == 1 else radii
    A = normals.T @ normals                                                # sum_v n n^T
    D = torch.linalg.solve( A, fb.T ).T                                    # the move that fits the views best
    rest = ( fs[ :, 0 ] - ( D * fb ).sum( 1 ) ).clamp( min = 0 ) / V
    return { "move": torch.linalg.vector_norm( D, dim = 1 ) / r, "dissonance": torch.sqrt( 2 * rest ) / r,
             "density asked": fs[ :, 1 ] / V - 1, "seen empty": fs[ :, 2 ] / V }
