"""How far a cloud is from the object it should describe."""
import numpy as np


def make_image_error( image, extent, rec ):
    """2D: `( points, radii ) -> ` the relative L1 gap, pixel by pixel, between the density of the balls and
    `image [ nb, nb ]` ( the object ), both of the same mass"""
    from .shapes import sized
    md = rec.model
    def image_error( points, radii ):
        img = sized.density( md.tensor( points ), md.tensor( radii ), image.shape[ 0 ], extent, md.k )
        return float( np.abs( img * ( image.sum() / img.sum() ) - image ).sum() / image.sum() )
    return image_error


def sinogram_residual( rec ):
    """2D: the relative L1 distance between the sinogram and the image of the shapes, both of unit mass per view, without
    blur nor background ( 0: the same sinogram, 2: disjoint ones )"""
    v = rec.image( sigma = 0.0, background = 0.0 )
    b = rec.sinogram.refined( rec.sinogram.values, rec.model.n_per_pixel )
    return float( np.abs( v / v.sum( 1, keepdims = True ) - b / b.sum( 1, keepdims = True ) ).sum( 1 ).mean() )
