import numpy as np
import torch

from ..shapes import gaussian_matrix
from . import grid_solver


class GridView:
    """One view of a 2D detector: `masses [ N, N ]` of the diracs ( N a power of two, from 32 ), at the centres of the cells
    of side `h` from the corner `origin`. The weights of the last solve are the warm start of the next one."""

    #: the blurs ( in cells ) a cold solve goes down through: cold, the solver fails on sharp images
    CONTINUATION_SIGMAS = ( 4.0, 2.0, 1.0, 0.5 )

    def __init__( self, masses, origin, h ):
        self.masses = np.asarray( masses, dtype = float )
        N = self.masses.shape[ 0 ]
        if self.masses.shape != ( N, N ) or N < 32 or N & ( N - 1 ):
            raise ValueError( f"GridView: N x N diracs, N a power of two from 32 ( got { self.masses.shape } )" )
        self.N, self.h, self.origin = N, float( h ), np.asarray( origin, dtype = float )
        self.weights, self.converged = None, False

    def data( self, sigma, background, like ):
        """the masses of the diracs, blurred by `sigma` ( cells ) and lifted by `background` times their mean"""
        G = gaussian_matrix( self.N, sigma )
        b = G @ self.masses @ G.T
        return torch.as_tensor( b + background * b.mean(), dtype = like.dtype, device = like.device )

    def value_and_grad( self, nodes_of_sigma, sigma, background ):
        """`nodes_of_sigma( s )`: the nodal values `[ N + 1, N + 1 ]` of the image at the blur `s` ( in cells ), detached.
        Returns the cost at `sigma` and its derivative with respect to the nodal values."""
        v = nodes_of_sigma( sigma )
        solve = lambda s, weights: grid_solver.solve( self.data( s, background, v ), nodes_of_sigma( s ), self.origin, self.h, weights )
        plan = None if self.weights is None else solve( sigma, self.weights )
        if plan is None or not grid_solver.converged( plan ):
            # no weights yet, or a warm start too far from the solution: from a blur where it converges, down to `sigma`
            weights = None
            for s in [ s for s in self.CONTINUATION_SIGMAS if s > sigma ] + [ sigma ]:
                plan = solve( s, weights )
                if grid_solver.converged( plan ) or weights is None:
                    weights = plan.weights
        self.weights, self.converged = plan.weights.detach(), grid_solver.converged( plan )
        # the solver normalizes the image: its mass is S = sum( v q )
        I, q = grid_solver.potential_integrals( self.weights, v, self.origin, self.h ), grid_solver.node_areas( self.N, self.h, v )
        S = torch.sum( v * q )
        return float( plan.cost ), I / S - q * torch.sum( v * I ) / S ** 2
