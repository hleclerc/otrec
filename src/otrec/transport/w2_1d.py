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

    pts = torch.sort( torch.cat( [ edges.expand( V, -1 ), q ], 1 ), 1 ).values
    lo, hi = pts[ :, :-1 ], pts[ :, 1: ]
    mid = ( 0.5 * ( lo + hi ) ).detach()
    jm = torch.floor( ( mid - origin ) / h ).long().clamp( 0, N - 1 )                        # the bin of each piece
    Fm = ( F.gather( 1, jm ) + f.gather( 1, jm ) * ( mid - edges[ jm ] ) ).detach()
    im = torch.searchsorted( C.detach(), Fm, right = True ).clamp( 0, M - 1 )                # its dirac
    a, c = hi - y[ im ], lo - y[ im ]
    return ( f.gather( 1, jm ) / 3 * ( a * a * a - c * c * c ) ).sum( 1 )
