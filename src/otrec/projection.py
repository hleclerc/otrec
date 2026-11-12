"""The parallel projections of an IMAGE, for views that turn about z: the line integrals along the rays, the detector
axis `u` being the horizontal unit vector `e` of the view ( and `v = z` in 3D ). A ray is sampled every pixel of the
image with a bilinear interpolation; `sub` rays per bin."""
import numpy as np
import torch


def project_image( image, extent, directions, nb_bins, sub = 2 ):
    """the MASS each pixel of the detector ( centred, of side `extent`, `nb_bins` per axis ) receives from `image`
    ( `[ nx, nx ]`, or `[ nx, nx, nz ]`, on the centred square or cube of side `extent` ), for each view of
    `directions [ nb_views, 2 ]`: `[ nb_views, nb_bins ]`, or `[ nb_views, nb_bins, nb_bins ]` ( u, v )"""
    image = torch.as_tensor( np.asarray( image ), dtype = torch.float32 )
    layers = image[ None ] if image.ndim == 2 else image.permute( 2, 0, 1 )               # [ nz, nx, ny ]
    nb, half, ds = layers.shape[ 1 ], extent / 2, extent / nb_bins
    h = extent / nb
    us = -half + ds / sub * ( torch.arange( nb_bins * sub ) + 0.5 )                        # the rays
    ts = -half + h * ( torch.arange( nb ) + 0.5 )                                          # the samples along a ray

    def at( i, j ):
        ok = ( i >= 0 ) & ( i < nb ) & ( j >= 0 ) & ( j < nb )
        return torch.where( ok, layers[ :, i.clamp( 0, nb - 1 ), j.clamp( 0, nb - 1 ) ], 0.0 )

    def view( e ):
        # p = u e + t e_perp, e_perp = ( -e_y, e_x ); in pixel index units
        x = ( us[ :, None ] * e[ 0 ] - ts[ None, : ] * e[ 1 ] + half ) / h - 0.5
        y = ( us[ :, None ] * e[ 1 ] + ts[ None, : ] * e[ 0 ] + half ) / h - 0.5
        i0, j0 = torch.floor( x ).long(), torch.floor( y ).long()
        fx, fy = x - i0, y - j0
        rays = ( at( i0, j0 ) * ( 1 - fx ) * ( 1 - fy ) + at( i0 + 1, j0 ) * fx * ( 1 - fy )
                 + at( i0, j0 + 1 ) * ( 1 - fx ) * fy + at( i0 + 1, j0 + 1 ) * fx * fy ).sum( 2 ) * h   # [ nz, rays ]
        bins = rays.reshape( -1, nb_bins, sub ).sum( 2 ) * ( ds / sub )                    # [ nz, u ]: mass per unit of z
        if image.ndim == 2:
            return bins[ 0 ]
        # the layers gathered in the rows of the detector: [ u, v ]
        return torch.nn.functional.adaptive_avg_pool1d( bins.T[ None ], nb_bins )[ 0 ] * ds

    return torch.stack( [ view( e ) for e in torch.as_tensor( np.asarray( directions ), dtype = torch.float32 ) ] ).double().numpy()
