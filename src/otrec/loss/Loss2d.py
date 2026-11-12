import torch

from ..transport import GridView


class Loss2d:
    """World 3D: each view is solved on sdot's grid solver ( `transport.GridView`, warm started from its last solve ), which
    gives the cost and its derivative with respect to the nodal values of the image; autograd takes it back to the points."""

    def __init__( self, model ):
        md, s = model, model.sinogram
        self.model = md
        self.floor = 0.0                                         # ( no closed form: see `Loss1d.floor` )
        self.views = [ GridView( b, ( md.origin, md.origin ), md.h ) for b in s.refined( s.values, md.n_per_pixel ) ]

    def value_and_grad( self, points, sigma, background ):
        md, ns = self.model, self.model.n_per_pixel
        points = points.detach().requires_grad_( True )
        total, grad = 0.0, torch.zeros_like( points )
        for views in md.batches():
            images, cotangents = md.images( points, sigma, background, views ), [ ]
            for i, nv in enumerate( range( views.start, views.stop ) ):
                def nodes( s_cells ):                              # ( a cold solve asks for other blurs than `sigma` )
                    if s_cells == sigma * ns:
                        return images[ i ].detach()
                    with torch.no_grad():
                        return md.images( points, s_cells / ns, background, slice( nv, nv + 1 ) )[ 0 ]
                cost, d_nodes = self.views[ nv ].value_and_grad( nodes, sigma * ns, background )
                total += cost
                cotangents.append( d_nodes )
            grad += torch.autograd.grad( images, points, torch.stack( cotangents ) )[ 0 ]
        return total, grad
