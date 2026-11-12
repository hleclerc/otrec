"""The EXACT semi-discrete 1D transport, in closed form ( numpy ): diracs of equal masses 1/N onto a piecewise-constant
density ( bins of width `ds` from `s_min` ), angle by angle.

Sorted, the dirac of rank r gets the quantile interval [ r/N, (r+1)/N ] of the density, i.e. the cell [ Q(r/N), Q((r+1)/N) ]
( Q the quantile function, piecewise linear ). From the moments of the density over the cell, its exact barycenter and its
cost; the dual weights by the recurrence at the cell boundaries b: | b - x_r |^2 - w_r = | b - x_{r+1} |^2 - w_{r+1}
( sdot's convention: the derivative of the cost with respect to the mass of a dirac ), centered.

What the 2D reconstruction needs from a view of a 2D object ( `exp_slice2d` ): no solver, no tolerance, no degenerate
diagram -- the reference against which the schemes are compared.
"""
import numpy as np


class Density1d:
    """a piecewise-constant density, normalized to mass 1: `values` [ B ] ( >= 0 ) on the bins of width `ds` from `s_min`"""

    def __init__( self, values, s_min, ds ):
        v = np.asarray( values, dtype = float )
        self.m = v * ds / ( v * ds ).sum()                       # bin masses
        self.e = s_min + ds * np.arange( len( v ) + 1 )          # edges
        self.f = self.m / ds                                     # densities
        self.C = np.concatenate( [ [ 0.0 ], np.cumsum( self.m ) ] )
        # the moments up to each edge: int t f, int t^2 f
        a, b = self.e[ :-1 ], self.e[ 1: ]
        self.M1 = np.concatenate( [ [ 0.0 ], np.cumsum( self.f * ( b ** 2 - a ** 2 ) / 2 ) ] )
        self.M2 = np.concatenate( [ [ 0.0 ], np.cumsum( self.f * ( b ** 3 - a ** 3 ) / 3 ) ] )

    def quantile( self, q ):
        """Q( q ), linear in each bin ( an empty bin is skipped )"""
        j = np.clip( np.searchsorted( self.C, q, side = "right" ) - 1, 0, len( self.m ) - 1 )
        # a bin of zero mass cannot hold q strictly inside: step to the next non-empty one
        with np.errstate( divide = "ignore", invalid = "ignore" ):
            t = np.where( self.m[ j ] > 0, ( q - self.C[ j ] ) / self.m[ j ], 0.0 )
        return self.e[ j ] + np.clip( t, 0, 1 ) * ( self.e[ j + 1 ] - self.e[ j ] )

    def moments( self, y ):
        """( int_{s_min}^y f, int t f, int t^2 f )"""
        j = np.clip( np.searchsorted( self.e, y, side = "right" ) - 1, 0, len( self.m ) - 1 )
        a = self.e[ j ]
        fj = self.f[ j ]
        return ( self.C[ j ] + fj * ( y - a ), self.M1[ j ] + fj * ( y ** 2 - a ** 2 ) / 2,
                 self.M2[ j ] + fj * ( y ** 3 - a ** 3 ) / 3 )


def transport( x, dens ):
    """`x` [ N ] onto `dens`: ( cost, barycenters [ N ], weights [ N ] ), in the order of `x`"""
    n = len( x )
    order = np.argsort( x, kind = "stable" )
    xs = x[ order ]
    b = dens.quantile( np.arange( n + 1 ) / n )                  # the cell boundaries, in rank order
    b[ 0 ], b[ -1 ] = dens.e[ 0 ], dens.e[ -1 ]
    _, m1, m2 = dens.moments( b )
    i1, i2 = np.diff( m1 ), np.diff( m2 )                        # int y, int y^2 over each cell ( mass 1/n each )
    bary = i1 * n
    cost = float( ( i2 - 2 * xs * i1 + xs * xs / n ).sum() )
    # the weights: w_{r+1} = w_r + | b - x_{r+1} |^2 - | b - x_r |^2 at the inner boundaries
    bi = b[ 1:-1 ]
    w = np.concatenate( [ [ 0.0 ], np.cumsum( ( bi - xs[ 1: ] ) ** 2 - ( bi - xs[ :-1 ] ) ** 2 ) ] )
    w -= w.mean()
    out_b, out_w = np.empty( n ), np.empty( n )
    out_b[ order ], out_w[ order ] = bary, w
    return cost, out_b, out_w


_GL_X = np.array( [ -np.sqrt( 0.6 ), 0.0, np.sqrt( 0.6 ) ] )
_GL_W = np.array( [ 5.0, 8.0, 5.0 ] ) / 9.0


def transport_tents( x, w, dens ):
    """`x` [ N ] the centers of TENTS ( 1 - |t| )_+ / w, t = ( y - x ) / w, of mass 1/N each, onto `dens`: the exact W2^2
    between the two densities ( no grid ), ( cost, barycenters [ N ], mass derivative [ N ] ).

    The tents' density is piecewise linear ( breakpoints x - w, x, x + w ), its CDF F piecewise quadratic; merged with the
    preimages of `dens`'s CDF levels, each piece carries a QUADRATIC map T = Q_dens( F ), so a 3-point Gauss-Legendre rule
    is exact for the cost int ( y - T )^2 f ( degree 5 ) and for the prefix integrals of T and y T. The "barycenter" of
    tent i is b_i = int T t_i ( the derivative of the cost with respect to x_i is 2/N ( x_i - b_i ) ); its mass derivative
    the Kantorovich potential phi ( phi' = 2 ( y - T ) ) at its center ( O( w^2 ) )."""
    n = len( x )
    # the sweep over the breakpoints: density f and CDF F at each, slope sigma after each
    p = np.concatenate( [ x - w, x, x + w ] )
    c = np.concatenate( [ np.full( n, 1.0 ), np.full( n, -2.0 ), np.full( n, 1.0 ) ] ) / ( n * w * w )
    o = np.argsort( p, kind = "stable" )
    yb, cb = p[ o ], c[ o ]
    sig = np.cumsum( cb )                                       # slope on [ yb_k, yb_k+1 ]
    dy = np.diff( yb )
    f = np.concatenate( [ [ 0.0 ], np.cumsum( sig[ :-1 ] * dy ) ] )
    f = np.maximum( f, 0.0 )
    F = np.concatenate( [ [ 0.0 ], np.cumsum( f[ :-1 ] * dy + 0.5 * sig[ :-1 ] * dy * dy ) ] )
    F /= F[ -1 ]

    def piece( y ):
        return np.clip( np.searchsorted( yb, y, side = "right" ) - 1, 0, len( yb ) - 2 )

    def cdf( y ):
        k = piece( y )
        t = y - yb[ k ]
        return np.clip( F[ k ] + f[ k ] * t + 0.5 * sig[ k ] * t * t, 0.0, 1.0 ), f[ k ] + sig[ k ] * t

    # the preimages of the target's CDF levels
    lev = dens.C[ 1:-1 ]
    lev = lev[ ( lev > 0 ) & ( lev < 1 ) ]
    k = np.clip( np.searchsorted( F, lev, side = "right" ) - 1, 0, len( yb ) - 2 )
    r = lev - F[ k ]
    disc = np.sqrt( np.maximum( f[ k ] ** 2 + 2 * sig[ k ] * r, 0.0 ) )
    with np.errstate( divide = "ignore", invalid = "ignore" ):
        t = np.where( f[ k ] + disc > 0, 2 * r / ( f[ k ] + disc ), 0.0 )
    pre = yb[ k ] + np.clip( t, 0, np.diff( yb )[ k ] )
    P = np.unique( np.concatenate( [ yb, pre ] ) )
    a, b = P[ :-1 ], P[ 1: ]
    hl = 0.5 * ( b - a )
    yq = 0.5 * ( a + b )[ :, None ] + hl[ :, None ] * _GL_X[ None, : ]          # [ intervals, 3 ]
    Fq, fq = cdf( yq )
    Tq = dens.quantile( Fq )
    wq = hl[ :, None ] * _GL_W[ None, : ]
    cost = float( ( wq * ( yq - Tq ) ** 2 * fq ).sum() )
    P0 = np.concatenate( [ [ 0.0 ], np.cumsum( ( wq * Tq ).sum( 1 ) ) ] )
    P1 = np.concatenate( [ [ 0.0 ], np.cumsum( ( wq * yq * Tq ).sum( 1 ) ) ] )
    Ph = np.concatenate( [ [ 0.0 ], np.cumsum( ( wq * 2 * ( yq - Tq ) ).sum( 1 ) ) ] )   # the potential, up to a constant
    i0, i1, i2 = ( np.searchsorted( P, v ) for v in ( x - w, x, x + w ) )
    bary = ( ( P1[ i1 ] - P1[ i0 ] ) - ( x - w ) * ( P0[ i1 ] - P0[ i0 ] )
             + ( x + w ) * ( P0[ i2 ] - P0[ i1 ] ) - ( P1[ i2 ] - P1[ i1 ] ) ) / ( w * w )
    phi = Ph[ i1 ]
    return cost, bary, phi - phi.mean()


class Views1d:
    """all the views of a 2D object: `axes` [ A, 2 ] ( unit detector axes ), `values` [ A, B ] on bins of width `ds` from
    `s_min`, plus `background` ( a fraction of each view's mean, as `Reconstruction3d` adds )"""

    def __init__( self, axes, values, s_min, ds, background = 0.0 ):
        self.axes = np.asarray( axes, dtype = float )
        vals = np.asarray( values, dtype = float )
        if background:
            vals = vals + background * vals.mean( axis = 1, keepdims = True )
        self.dens = [ Density1d( v, s_min, ds ) for v in vals ]
        self.H = self.axes.T @ self.axes                         # sum_k e_k e_k^T ( 2 x 2 )

    def evaluate( self, x, tent = 0.0 ):
        """`x` [ N, 2 ]: cost, gradient [ N, 2 ], the lifted Lloyd step [ N, 2 ], the mass derivative [ N ], the
        disagreement of the views [ N ]. `tent` > 0: the diracs are tents of that half-width ( `transport_tents` )"""
        n = len( x )
        cost, r, S, wsum = 0.0, np.zeros( ( n, 2 ) ), np.zeros( n ), np.zeros( n )
        for e, d in zip( self.axes, self.dens ):
            s = x @ e
            c, b, w = transport_tents( s, tent, d ) if tent > 0 else transport( s, d )
            cost += c
            r += ( b - s )[ :, None ] * e[ None, : ]
            S += ( b - s ) ** 2
            wsum += w
        step = np.linalg.solve( self.H, r.T ).T
        return dict( cost = cost, grad = -2.0 / n * r, step = step, mass_grad = wsum, disagreement = S - ( r * step ).sum( 1 ) )
