import torch

from ..shapes import balls
from ..transport import w2_diracs_vs_bins


class Loss1d:
    """World 2D. The deposit of the shapes and its adjoint are fused kernels ( `shapes.fused` ): what costs. Between them,
    the transport of every view has a closed form ( `transport.w2_1d` ) that autograd differentiates on `[ views, cells ]`.
    `kind`: what the rows of `x` are -- `shapes.balls`, `shapes.sized` or `shapes.polygons`, which give the tensors the kernels take
    ( `shapes` ), autograd taking the derivatives back from them to `x`."""

    def __init__( self, model, kind = balls ):
        self.model, self.kind = model, kind
        #: what the loss cannot go under: each dirac sits at the centre of its cell, and the image is uniform in a cell --
        #: spreading a dirac over its own cell costs h^2 / 12 per unit of mass, in every view
        self.floor = model.sinogram.nb_views * model.h ** 2 / 12
        self.diracs = model.origin + model.h * ( torch.arange( model.N, dtype = torch.float64, device = model.device ) + 0.5 )

    def value_and_grad( self, x, sigma, background ):
        md = self.model
        x = x.detach().requires_grad_( True )
        shapes = self.kind.shapes( x, md )
        frozen = [ t.detach() for t in shapes ]
        mu = self.kind.deposit( *frozen, md ).requires_grad_( True )
        loss = w2_diracs_vs_bins( self.diracs, md.data( sigma, background ), md.finish( mu, sigma, background ), md.origin, md.h ).sum()
        d_shapes = self.kind.gather( *frozen, md, torch.autograd.grad( loss, mu )[ 0 ] )
        return loss.item(), torch.autograd.grad( shapes, x, d_shapes )[ 0 ]
