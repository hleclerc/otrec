"""How far a cloud is from the object it should describe."""
import numpy as np


def density_image( points, radius, k, nb, extent, sub = 4 ):
    """2D: the density of the cloud ( the sum of its shapes, each of mass 1 ) summed over each of the `nb^2` pixels of the
    centred square of side `extent`: a histogram of the centers on `sub` cells per pixel, convolved by the shape"""
    from scipy.signal import fftconvolve
    half, R = extent / 2, radius / ( extent / nb ) * sub
    H = np.histogram2d( points[ :, 0 ], points[ :, 1 ], bins = nb * sub, range = [ [ -half, half ] ] * 2 )[ 0 ]
    g = np.arange( -int( R ) - 1, int( R ) + 2 )
    K = np.maximum( 1 - ( g[ :, None ] ** 2 + g[ None ] ** 2 ) / R ** 2, 0 ) ** ( k - 0.5 )
    return fftconvolve( H, K / K.sum(), mode = "same" ).reshape( nb, sub, nb, sub ).sum( ( 1, 3 ) )


def make_image_error( image, extent, radius, k = 2 ):
    """2D: `points -> ` the relative L1 gap, pixel by pixel, between the density of the cloud and `image [ nb, nb ]` ( the
    object ), both of the same mass"""
    def image_error( points ):
        img = density_image( points, radius, k, image.shape[ 0 ], extent )
        return float( np.abs( img * ( image.sum() / img.sum() ) - image ).sum() / image.sum() )
    return image_error


def sinogram_residual( rec, points = None ):
    """2D: the relative L1 distance between the sinogram and the image of the shapes, both of unit mass per view, without
    blur nor background ( 0: the same sinogram, 2: disjoint ones )"""
    v = rec.image( points, 0.0, 0.0 )
    b = rec.sinogram.refined( rec.sinogram.values, rec.model.n_per_pixel )
    return float( np.abs( v / v.sum( 1, keepdims = True ) - b / b.sum( 1, keepdims = True ) ).sum( 1 ).mean() )
