import torch

from ..shapes import ellipses, fused
from ..transport import w2_diracs_vs_bins


class EllipseLoss1d:
    """World 2D, a cloud of ELLIPSES ( `shapes.ellipses`: `x [ n, 6 ]` ): the same transport as `Loss1d`, between the fused
    kernels of the ellipses ( `shapes.fused` ), which give the derivatives with respect to the centres, the matrices
    `exp( 2 S )` and the masses; autograd takes them back to `x`. `r0`: the radius the ellipses are measured against;
    `max_ratio`: how far from it a half-width may go."""

    def __init__( self, model, max_ratio = 4.0 ):
        self.model, self.r0, self.max_ratio = model, model.radius, float( max_ratio )
        self.floor = model.sinogram.nb_views * model.h ** 2 / 12
        self.diracs = model.origin + model.h * ( torch.arange( model.N, dtype = torch.float64, device = model.device ) + 0.5 )

    def value_and_grad( self, x, sigma, background ):
        md = self.model
        x = x.detach().requires_grad_( True )
        centres, S, masses = ellipses.unpack( x, self.r0 )
        shapes = ( centres, ellipses.covariances( S ), masses )
        args = ( md.axes[ :, 0 ], md.N, md.h, md.origin, self.r0 / self.max_ratio, self.r0 * self.max_ratio, md.k )
        mu = fused.deposit_ellipses( *[ t.detach() for t in shapes ], *args ).requires_grad_( True )
        loss = w2_diracs_vs_bins( self.diracs, md.data( sigma, background ), md.finish( mu, sigma, background ), md.origin, md.h ).sum()
        d_shapes = fused.gather_ellipses( *[ t.detach() for t in shapes ], *args, torch.autograd.grad( loss, mu )[ 0 ] )
        return loss.item(), torch.autograd.grad( shapes, x, d_shapes )[ 0 ]
