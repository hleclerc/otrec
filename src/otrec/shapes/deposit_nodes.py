import math

import torch



def deposit_nodes( uv, N, h, origin, radius, k ):
    """`[ V, N + 1, N + 1 ]`: the density at the corners of the pixels of a 2D detector ( `N x N` pixels of side `h` from
    the corner `( origin, origin )` ), in each of `V` views, of the `n` shapes ( masses `1 / n` ) projected at
    `uv [ V, n, 2 ]`"""
    w = int( math.ceil( radius / h ) ) + 1                                                # the nodes a shape can reach
    t = ( uv - origin ) / h
    i0 = torch.round( t ).long()
    off = torch.arange( -w, w + 1, device = uv.device )
    ia = i0[ ..., 0, None, None ] + off[ :, None ]                                        # [ V, n, W, 1 ]
    ib = i0[ ..., 1, None, None ] + off[ None, : ]                                        # [ V, n, 1, W ]
    da = ( ia - t[ ..., 0, None, None ] ) * h
    db = ( ib - t[ ..., 1, None, None ] ) * h
    q = torch.clamp( 1.0 - ( da * da + db * db ) / ( radius * radius ), min = 0.0 )
    ok = ( ia >= 0 ) & ( ia <= N ) & ( ib >= 0 ) & ( ib <= N )
    vals = torch.where( ok, q ** k * ( ( k + 1 ) / ( math.pi * radius * radius * uv.shape[ 1 ] ) ), 0.0 )      # ( unit mass in all )
    flat = ia.clamp( 0, N ) * ( N + 1 ) + ib.clamp( 0, N )
    V = uv.shape[ 0 ]
    out = torch.zeros( V, ( N + 1 ) ** 2, dtype = uv.dtype, device = uv.device )
    return out.scatter_add( 1, flat.reshape( V, -1 ), vals.reshape( V, -1 ) ).reshape( V, N + 1, N + 1 )
