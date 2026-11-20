import torch

from ..shapes import balls, sized
from ..shapes.fused_loss import balls_fields, balls_loss


class LossFused:
    """World 2D, BALLS of one radius: the same loss as `Loss1d( model, balls )` ( W2^2 of each view, summed ), in ONE kernel
    ( `shapes.fused_loss` ): the deposit, the transport and the gather fused, nothing of size `[ views, cells ]` through
    torch. Without blur and on the window of the data only ( `sigma = 0`, `model.pad = 0` ): `Loss1d` otherwise."""

    def __init__( self, model ):
        self.model, self.kind = model, balls
        self.floor = model.sinogram.nb_views * model.h ** 2 / 12
        from .Loss1d import Loss1d
        self._fallback = Loss1d( model, balls )

    def value_and_grad( self, x, sigma, background ):
        md = self.model
        if sigma or md.pad:
            return self._fallback.value_and_grad( x, sigma, background )
        x = x.detach().requires_grad_( True )
        c = balls.shapes( x, md )[ 0 ]
        normals, N, h, origin = md.detector()
        loss, gp, _, _ = balls_loss( c.detach(), md.tensor( [ md.radius ] ), normals, md.data( 0.0, background ), h, origin, md.radius, md.k, background )
        return loss.sum().item(), torch.autograd.grad( c, x, gp )[ 0 ]

    def fields( self, x, sigma, background ):
        """`{ name: [ n ] }`, numpy: what the views ask of each ball ( `Loss1d.fields`, in one call )"""
        md = self.model
        if sigma or md.pad:
            return self._fallback.fields( x, sigma, background )
        with torch.no_grad():
            c = balls.shapes( x, md )[ 0 ]
            normals, N, h, origin = md.detector()
            f = balls_fields( c, md.tensor( [ md.radius ] ), normals, md.data( 0.0, background ), h, origin, md.radius, md.k, background )
        return { k: v.cpu().numpy() for k, v in f.items() }


class LossFusedSized:
    """World 2D, SIZED balls ( `shapes.sized`: a radius each, the mass following the size, a density each when the rows
    have one ): the loss of `Loss1d( model, sized )` in one kernel. `grads`: the radii ( and densities ) may move -- the
    derivatives with respect to the radii and the masses are computed; without, only those of the centres ( a stage
    whose radii are kept: `adjust_parameters( radii = "keep" )` ). Without blur and on the window only, else `Loss1d`."""

    def __init__( self, model, grads = True ):
        self.model, self.kind, self.grads = model, sized, grads
        self.floor = model.sinogram.nb_views * model.h ** 2 / 12
        from .Loss1d import Loss1d
        self._fallback = Loss1d( model, sized )

    def value_and_grad( self, x, sigma, background ):
        md = self.model
        if sigma or md.pad:
            return self._fallback.value_and_grad( x, sigma, background )
        x = x.detach().requires_grad_( True )
        c, C, m = sized.shapes( x, md )                                  # ( C = ( r^2, r^2, 0 ), m normalized )
        r = torch.sqrt( C[ :, 0 ] )
        normals, N, h, origin = md.detector()
        loss, gp, gr, gm = balls_loss( c.detach(), r.detach(), normals, md.data( 0.0, background ), h, origin, md.radius * md.max_ratio, md.k,
                                       background, masses = m.detach(), grads = self.grads )
        if not self.grads:
            return loss.sum().item(), torch.autograd.grad( c, x, gp, retain_graph = False )[ 0 ]
        return loss.sum().item(), torch.autograd.grad( ( c, r, m ), x, ( gp, gr, gm ) )[ 0 ]

    def fields( self, x, sigma, background ):
        """`{ name: [ n ] }`, numpy: what the views ask of each ball, and its density when it has one"""
        md = self.model
        if sigma or md.pad:
            return self._fallback.fields( x, sigma, background )
        with torch.no_grad():
            c, C, m = sized.shapes( x, md )
            normals, N, h, origin = md.detector()
            f = balls_fields( c, torch.sqrt( C[ :, 0 ] ), normals, md.data( 0.0, background ), h, origin, md.radius * md.max_ratio, md.k, background, masses = m )
            if x.shape[ 1 ] > 3:
                f[ "density" ] = sized.weights( x, md )
        return { k: v.cpu().numpy() for k, v in f.items() }
