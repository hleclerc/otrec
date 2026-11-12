import numpy as np

from errand import test

from otrec import Sinogram

if test( "2D: a disc projects as 2 sqrt( R^2 - s^2 ) around its projected centre" ):
    nb, R, c = 256, 0.5, np.array( [ 0.2, -0.1 ] )
    g = -1 + 2 / nb * ( np.arange( nb ) + 0.5 )
    image = np.clip( 0.5 + ( R - np.hypot( g[ :, None ] - c[ 0 ], g[ None, : ] - c[ 1 ] ) ) / ( 2 / nb ), 0, 1 )
    s = Sinogram.from_image( image, nb_views = 5, nb_bins = 64 )
    centres = s.edges[ :-1 ] + s.h / 2 - s.project( c[ None ] )[ :, 0 ]                  # [ view, bin ]
    gap = np.abs( s.values / s.h - 2 * np.sqrt( np.maximum( R * R - centres ** 2, 0 ) ) )
    assert gap.mean() < 5e-3 and gap.max() < 0.15, ( gap.mean(), gap.max() )      # ( the largest at the rim, where the slope is infinite )
    assert np.allclose( s.values.sum( 1 ), np.pi * R * R, rtol = 2e-3 )

if test( "3D: a ball; every view carries its mass, and so do its diracs, around its projected centre" ):
    nb, R, c = 64, 0.4, np.array( [ 0.2, -0.1, 0.3 ] )
    g = -1 + 2 / nb * ( np.arange( nb ) + 0.5 )
    d = np.sqrt( ( g[ :, None, None ] - c[ 0 ] ) ** 2 + ( g[ None, :, None ] - c[ 1 ] ) ** 2 + ( g[ None, None, : ] - c[ 2 ] ) ** 2 )
    s = Sinogram.from_image( np.clip( 0.5 + ( R - d ) / ( 2 / nb ), 0, 1 ), nb_views = 3, nb_bins = 32 )
    assert s.dim == 3 and np.allclose( s.values.sum( ( 1, 2 ) ), 4 / 3 * np.pi * R ** 3, rtol = 1e-2 ), s.values.sum( ( 1, 2 ) )
    pos, masses = s.diracs( 2 )
    assert pos.shape == ( 64, ) and masses.shape == ( 3, 64, 64 ) and np.allclose( masses.sum( ( 1, 2 ) ), s.values.sum( ( 1, 2 ) ) )
    uv = s.project( c[ None ] )[ :, 0 ]
    m = masses.sum( ( 1, 2 ) )
    assert np.allclose( np.einsum( "vij,i->v", masses, pos ) / m, uv[ :, 0 ], atol = 5e-3 )
    assert np.allclose( np.einsum( "vij,j->v", masses, pos ) / m, uv[ :, 1 ], atol = 5e-3 )
