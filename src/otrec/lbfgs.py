"""L-BFGS on tensors, on their device: the two-loop recursion and a backtracking line search."""
import math

import torch


def lbfgs( value_and_grad, x, move_tol, max_iter = 1000, window = 10, quantile = 0.999, history = 10, callback = None ):
    """minimizes `value_and_grad( x ) -> ( float, tensor )` from the tensor `x [ n, dim ]`, until the rows of `x` have moved
    by less than `move_tol` over the last `window` iterations ( a net move: one iteration alone says little, its step may
    be short ) -- all but the `1 - quantile` that moved the most: a few rows on a long way would hold the stop for
    hundreds of iterations. Or until `max_iter`. `callback( x, f )` after each iteration.
    Returns `( x, f, number of iterations, number of evaluations, why it stopped )`: "still" ( the rows no longer move ),
    "stalled" ( no step lowers the function any more: its noise is reached ) or "max_iter"."""
    f, g = value_and_grad( x )
    pairs, nb_eval, it, x_ref, stop = [ ], 1, 0, x, "max_iter"       # pairs: ( s, y, 1 / s.y ), the oldest first
    while it < max_iter:
        # the direction: -H g, H the inverse Hessian the pairs describe ( scaled by s.y / y.y of the last one )
        d, alphas = -g, [ ]
        for s, y, rho in reversed( pairs ):
            alphas.append( rho * torch.sum( s * d ) )
            d = d - alphas[ -1 ] * y
        if pairs:
            d = d * ( torch.sum( pairs[ -1 ][ 0 ] * pairs[ -1 ][ 1 ] ) / torch.sum( pairs[ -1 ][ 1 ] ** 2 ) )
        for ( s, y, rho ), alpha in zip( pairs, reversed( alphas ) ):
            d = d + ( alpha - rho * torch.sum( y * d ) ) * s
        slope = torch.sum( g * d ).item()
        # the step: 1, the first one a unit move; halved until the loss decreases enough ( Armijo )
        t = 1.0 if pairs else 1.0 / max( torch.linalg.vector_norm( g ).item(), 1e-300 )
        while True:
            x1 = x + t * d
            f1, g1 = value_and_grad( x1 )
            nb_eval += 1
            if f1 <= f + 1e-4 * t * slope or t < 1e-10:
                break
            t *= 0.5
        if f1 >= f:                                                   # no progress left along this direction
            stop = "stalled"
            break
        s, y = x1 - x, g1 - g
        sy = torch.sum( s * y ).item()
        if sy > 0:                                                    # ( a pair of negative curvature would break H > 0 )
            pairs = ( pairs + [ ( s, y, 1.0 / sy ) ] )[ -history: ]
        x, f, g, it = x1, f1, g1, it + 1
        if callback:
            callback( x, f )
        if it % window == 0:
            moves = torch.linalg.vector_norm( x - x_ref, dim = 1 )
            if moves.kthvalue( math.ceil( quantile * len( moves ) ) ).values.item() <= move_tol:
                stop = "still"
                break
            x_ref = x
    return x, f, it, nb_eval, stop
