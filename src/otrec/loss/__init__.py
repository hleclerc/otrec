"""The LOSS: the sum over the views of W2^2( diracs of the view, image of the projected shapes ).

    Model     the image of the shapes in the views ( deposit, blur, background ) and the masses of the diracs
    Loss1d    world 2D: fused deposit kernels around a closed form per view
    Loss2d    world 3D: one solve per view on sdot's grid solver, then the chain rule through the model

`sigma` is the blur, here in PIXELS of the sinogram ( `Reconstruction` takes it in world units ), applied to BOTH measures; `background` lifts both by that fraction of their
mean, so that the image stays positive everywhere."""
from .Loss1d import Loss1d as Loss1d
from .Loss2d import Loss2d as Loss2d
from .Model import Model as Model


def make_loss( model ):
    """the loss of the dimension of the sinogram: `.value_and_grad( points, sigma, background )`, on tensors"""
    return Loss1d( model ) if model.sinogram.dim == 2 else Loss2d( model )
