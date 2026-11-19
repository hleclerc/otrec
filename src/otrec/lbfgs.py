"""L-BFGS on tensors, on their device: the two-loop recursion and a backtracking line search."""
import math

import torch


def lbfgs( value_and_grad, x, move_tol, max_iter = 1000, window = 10, quantile = 0.999, history = 10, callback = None, project = None ):
    """minimizes `value_and_grad( x ) -> ( float, tensor )` from the tensor `x [ n, dim ]`, until the rows of `x` have moved
    by less than `move_tol` over the last `window` iterations ( a net move: one iteration alone says little, its step may
    be short ) -- all but the `1 - quantile` that moved the most: a few rows on a long way would hold the stop for
    hundreds of iterations. Or until `max_iter`. `callback( x, f )` after each iteration. `project( x )`: each point tried
    is brought back into a domain where the function only depends on it ( the rows that leave it have no derivative
    towards it otherwise ); its value must not change. The gradient of a row on its border then loses what would take it
    out ( an active set: the directions stay ones that can lower the function ).
    Returns `( x, f, number of iterations, number of evaluations, why it stopped )`: "still" ( the rows no longer move ),
    "stalled" ( no step along the gradient lowers the function any more: its noise is reached ) or "max_iter"."""
    def active( x, g ):
        """`g` without, on each row on the border, the part along which a descent would leave the domain"""
        if project is None:
            return g
        h = 1e-6 / max( g.abs().max().item(), 1e-300 )
        out = ( x - h * g ) - project( x - h * g )                         # where a short descent is brought back from
        n = torch.linalg.vector_norm( out, dim = 1, keepdim = True )
        hit = n > 1e-3 * h * torch.linalg.vector_norm( g, dim = 1, keepdim = True )
        u = out / n.clamp( min = 1e-300 )
        return torch.where( hit, g - torch.sum( g * u, 1, keepdim = True ).clamp( max = 0 ) * u, g )

    f, g = value_and_grad( x )
    g = active( x, g )
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
        if project is not None:
            # the direction too: what of it would leave the domain goes ( the pairs mix the rows, and give back what
            # `active` took off the gradient ); if it is then no longer a descent, the gradient, and the pairs forgotten
            d = -active( x, -d )
            if torch.sum( g * d ).item() >= 0:
                d, pairs = -g, [ ]
        slope = torch.sum( g * d ).item()
        # the step: 1, the first one a unit move; halved until the loss decreases enough ( Armijo )
        t = 1.0 if pairs else 1.0 / max( torch.linalg.vector_norm( d ).item(), 1e-300 )
        while True:
            x1 = x + t * d if project is None else project( x + t * d )
            f1, g1 = value_and_grad( x1 )
            g1 = active( x1, g1 )
            nb_eval += 1
            if f1 <= f + 1e-4 * t * slope or t < 1e-10:
                break
            t *= 0.5
        if f1 >= f:                                                   # no progress left along this direction
            if pairs:                                                 # ( the pairs may be wrong: a row that met a bound
                pairs = [ ]                                           # makes a jump of `s` and `y` ) the gradient first
                continue
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
            # ( not on the first window: the steps are short while the pairs are few, whatever is left to do )
            if it > window and moves.kthvalue( math.ceil( quantile * len( moves ) ) ).values.item() <= move_tol:
                stop = "still"
                break
            x_ref = x
    return x, f, it, nb_eval, stop
