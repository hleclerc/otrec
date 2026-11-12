import torch

from ..shapes import fused
from ..transport import w2_diracs_vs_bins


class Loss1d:
    """World 2D. The deposit of the shapes and its adjoint are fused kernels ( `shapes.fused` ): what costs. Between them,
    the transport of every view has a closed form ( `transport.w2_1d` ) that autograd differentiates on `[ views, cells ]`."""

    def __init__( self, model ):
        self.model = model
        #: what the loss cannot go under: each dirac sits at the centre of its cell, and the image is uniform in a cell --
        #: spreading a dirac over its own cell costs h^2 / 12 per unit of mass, in every view
        self.floor = model.sinogram.nb_views * model.h ** 2 / 12
        self.diracs = model.origin + model.h * ( torch.arange( model.N, dtype = torch.float64, device = model.device ) + 0.5 )

    def value_and_grad( self, points, sigma, background ):
        md = self.model
        normals = md.axes[ :, 0 ]
        mu = fused.deposit( points, normals, *md.grid ).requires_grad_( True )
        loss = w2_diracs_vs_bins( self.diracs, md.data( sigma, background ), md.finish( mu, sigma, background ), md.origin, md.h ).sum()
        return loss.item(), fused.gather( points, normals, *md.grid, torch.autograd.grad( loss, mu )[ 0 ] )
