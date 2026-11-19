"""What the views ASK of each ball, view by view ( world 2D, balls and sized balls ): fields to colour the shapes with.

In a view of axis `n`, the 1D transport sends the mass of a ball at `u = n . centre` to a place whose barycentre is
`u + delta`: the view asks the ball to move by `delta` along `n` ( the derivative of the loss with respect to `u` is
`-2 m delta`, `m` the mass of the ball in the normalized measure ). The gradient adds `delta n` over the views; it can
vanish while the views disagree. Per ball, with `D` the move that fits the `delta` of the views best ( least squares ):

    move          |D|, in radii of the ball: what the views agree on ( 0 once a stage has converged )
    dissonance    the rest, sqrt( 2 mean( ( delta - D . n )^2 ) ), in radii: what they ask and cancel out ( 0 when the
                  ball sits where every view wants it; for views uniform over half a turn, `move` and `dissonance` are
                  the same measure of a request, one coherent, the other not )
    density asked mean( data / image ) - 1 over the cells the ball covers, its profile as the weight: the change of
                  density one step of EM would give it ( > 0: the views see more matter there than the balls put )
    seen empty    the fraction of the views where the data under the ball is less than 25 % of what it puts there ( the
                  background lifts the air to a few percent of a ball ): a
                  ball outside the matter ( its convex hull ) is seen in empty air by some views

ELLIPSES ( `covs` ): in a view of axis `n`, the half-width `sqrt( n.C n )` takes the place of the radius; the moves are
still counted in radii of the disc of the same area. Computed by batches of views, a window of cells around each ball ( `[ views, balls, window ]` ), in torch."""
import math

import torch

from ..shapes.fused import _dome_coeffs


def profile( k ):
    """`( G, p )`: the cumulative of the projected profile of a ball of radius 1 and its density, on `t` in [ -1, 1 ]"""
    if k == 0.5:
        return ( lambda t: 0.5 + ( t * torch.sqrt( 1 - t * t ) + torch.asin( t ) ) / math.pi,
                 lambda t: 2 / math.pi * torch.sqrt( 1 - t * t ) )
    c, a = _dome_coeffs( k )
    return ( lambda t: 0.5 + sum( aj * t ** ( 2 * j + 1 ) for j, aj in enumerate( a ) ),
             lambda t: c * ( 1 - t * t ) ** k )


def per_ball( model, centres, radii, masses, d_cells, totals, ratio, view_batch = 32, covs = None, widths = None ):
    """`{ "move", "dissonance", "density asked" }`, each `[ n ]`. `masses`: of the balls in the deposit ( `[ n ]` );
    `d_cells [ V, N ]`: the derivative of the loss with respect to the mass of each cell of the deposit; `totals [ V ]`:
    the mass of the image of each view, which the loss divides by; `ratio [ V, N ]`: data / image, both normalized as
    the loss compares them. `covs [ n, 3 ]` ( c00, c11, c01 ): ellipses, their half-widths kept in `widths` ( min, max )"""
    A, b, s2, em, empty = sums( model, centres, radii, masses, d_cells, totals, ratio, view_batch, covs, widths )
    V = len( model.axes )
    D = torch.linalg.solve( A, b.T ).T                                                  # [ n, 2 ]
    rest = ( s2 - ( D * b ).sum( 1 ) ).clamp( min = 0 ) / V                            # ( sum_v ( delta - D.n )^2 / V )
    return dict( move = torch.linalg.vector_norm( D, dim = 1 ) / radii,
                 dissonance = torch.sqrt( 2 * rest ) / radii,
                 **{ "density asked": em / V - 1, "seen empty": empty / V } )


def sums( model, centres, radii, masses, d_cells, totals, ratio, view_batch = 32, covs = None, widths = None ):
    """the sums over the views: `sum_v n n^T [ 2, 2 ]`, `sum_v delta n [ n, 2 ]`, `sum_v delta^2 [ n ]`, and
    `sum_v` of the mean of `ratio` over each ball `[ n ]`, the number of views where that mean is under 0.25 `[ n ]`"""
    G, p = profile( model.k )
    N, h, origin = model.N, model.h, model.origin
    axes = model.axes[ :, 0 ]                                                           # [ V, 2 ]
    top = float( radii.max() ) if covs is None else float( torch.sqrt( covs[ :, 0 ] + covs[ :, 1 ] ).max() )
    if widths is not None:
        top = min( top, widths[ 1 ] )
    W = int( math.ceil( 2 * top / h ) ) + 2                                              # cells a shape may cover
    off = torch.arange( W + 1, device = centres.device )
    b = torch.zeros( len( centres ), 2, dtype = centres.dtype, device = centres.device )  # sum_v delta n
    s2 = torch.zeros( len( centres ), dtype = centres.dtype, device = centres.device )   # sum_v delta^2
    em, empty = torch.zeros_like( s2 ), torch.zeros_like( s2 )
    A = axes.T @ axes                                                                   # sum_v n n^T
    for v0 in range( 0, len( axes ), view_batch ):
        n = axes[ v0 : v0 + view_batch ]                                                # [ B, 2 ]
        u = n @ centres.T                                                               # [ B, n ]
        if covs is None:
            w = radii[ None ].expand_as( u )
        else:                                                                           # the half-width in each view
            w = torch.sqrt( ( n[ :, 0:1 ] ** 2 * covs[ None, :, 0 ] + n[ :, 1:2 ] ** 2 * covs[ None, :, 1 ]
                              + 2 * n[ :, 0:1 ] * n[ :, 1:2 ] * covs[ None, :, 2 ] ).clamp( min = 0 ) )
            if widths is not None:
                w = w.clamp( *widths )
        e = torch.floor( ( u - w - origin ) / h ).long()[ ..., None ] + off             # the edges around each shape
        t = ( ( origin + h * e - u[ ..., None ] ) / w[ ..., None ] ).clamp( -1, 1 )
        frac = G( t[ ..., 1: ] ) - G( t[ ..., :-1 ] )                                   # its share in each cell
        cell = e[ ..., :-1 ]
        inside = ( cell >= 0 ) & ( cell < N )
        idx = cell.clamp( 0, N - 1 ).flatten( 1 )
        d = torch.gather( d_cells[ v0 : v0 + view_batch ], 1, idx ).view_as( frac ) * inside
        r = torch.gather( ratio[ v0 : v0 + view_batch ], 1, idx ).view_as( frac )
        # d loss / d u = m sum_c d_c d frac_c / du, d frac_c / du = -( p( t_c+1 ) - p( t_c ) ) / r
        pe = p( t )
        grad = -( masses[ None ] / w ) * ( d * ( pe[ ..., 1: ] - pe[ ..., :-1 ] ) ).sum( -1 )
        delta = -grad * totals[ v0 : v0 + view_batch, None ] / ( 2 * masses[ None ] )   # [ B, n ]
        b += delta.T @ n
        s2 += ( delta * delta ).sum( 0 )
        seen = ( frac * inside * r ).sum( -1 )
        em += seen.sum( 0 )
        empty += ( seen < 0.25 ).sum( 0 )
    return A, b, s2, em, empty
