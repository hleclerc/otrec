"""Exact squared Wasserstein distance, on a 1D detector, between diracs and a piecewise constant image.

In 1D the optimal plan is monotone: the dirac `i` ( positions sorted ) receives the slice of the image between the quantiles
of the cumulated masses C_( i - 1 ) and C_i. With F the ( piecewise linear ) cumulative of the image and q_i = F^-1( C_i ),

    W2^2 = sum_i  int_( q_( i - 1 ) )^( q_i )  ( x - y_i )^2 f( x ) dx

and on each piece between two consecutive breakpoints ( bin edges and quantiles together ) `f` is a constant: a closed form,
piecewise smooth in the values of the image, that autograd differentiates as it stands ( no solver, no envelope theorem;
the indices are computed on detached values ). Both measures are normalized to a unit mass here."""
import torch


def w2_diracs_vs_bins( y, b, v, origin, h ):
    """`y [ M ]`: sorted positions of the diracs; per view, `b [ V, M ]` their masses ( >= 0 ) and `v [ V, N ]` the mass of
    each bin ( > 0 ) of width `h` from `origin`. Returns `[ V ]`: W2^2 between the two normalized measures of each view."""
    V, N, M = v.shape[ 0 ], v.shape[ 1 ], y.shape[ 0 ]
    f = torch.clamp( v / ( v.sum( 1, keepdim = True ) * h ), min = 1e-300 )                # density per bin, unit mass
    edges = origin + h * torch.arange( N + 1, dtype = v.dtype, device = v.device )
    F = torch.cat( [ torch.zeros( V, 1, dtype = v.dtype, device = v.device ), torch.cumsum( f * h, 1 ) ], 1 )
    C = torch.cumsum( b / b.sum( 1, keepdim = True ), 1 )[ :, :-1 ].contiguous()            # the M - 1 inner thresholds

    j = ( torch.searchsorted( F.detach(), C, right = True ) - 1 ).clamp( 0, N - 1 )
    q = torch.clamp( edges[ j ] + ( C - F.gather( 1, j ) ) / f.gather( 1, j ), edges[ 0 ], edges[ -1 ] )   # the quantiles

    # the breakpoints, by BIN then by position, each edge before the quantiles of its bin. A piece takes the bin of the
    # point it starts at, and the dirac after the quantiles before it: read from that order, never from positions. A
    # quantile sits ON an edge wherever both cumulated masses agree ( a uniform background, one dirac per bin ): a test on
    # positions ( or an unstable sort, as torch's on the CPU ) then gives the piece that ends at q_i the density of bin
    # j - 1, while q_i moves with the density of bin j -- the loss right, its gradient wrong ( 1e-4 relative )
    allp = torch.cat( [ edges.expand( V, -1 ), q ], 1 )
    bins = torch.cat( [ torch.arange( N + 1, device = v.device ).expand( V, -1 ), j ], 1 )
    is_q = torch.cat( [ torch.zeros_like( bins[ :, :N + 1 ] ), torch.ones_like( j ) ], 1 )
    o = torch.sort( allp.detach(), dim = 1, stable = True ).indices
    o = o.gather( 1, torch.sort( bins.gather( 1, o ), dim = 1, stable = True ).indices )
    pts = allp.gather( 1, o )
    lo, hi = pts[ :, :-1 ], pts[ :, 1: ]
    jm = bins.gather( 1, o )[ :, :-1 ].clamp( max = N - 1 )                                  # the bin of each piece
    im = torch.cumsum( is_q.gather( 1, o ), 1 )[ :, :-1 ].clamp( max = M - 1 )               # its dirac
    a, c = hi - y[ im ], lo - y[ im ]
    return ( f.gather( 1, jm ) / 3 * ( a * a * a - c * c * c ) ).sum( 1 )
