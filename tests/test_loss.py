import numpy as np

from errand import test

from otrec import Reconstruction, Sinogram, ball_packing

TRUTH = np.array( [ [ 0.30, 0.10 ], [ -0.25, 0.35 ], [ -0.10, -0.40 ], [ 0.45, -0.30 ] ] )


def sino_of_shapes( points, radius, nb_views = 12, nb_bins = 64 ):
    """2D: the sinogram a cloud of shapes gives exactly ( its own image, without blur nor background )"""
    s = Sinogram( 2, nb_views, nb_bins, extent = 2.0 )
    s.values = Reconstruction( s, points, radius = radius ).image( sigma = 0.0, background = 0.0 )
    return s


def finite_difference( rec, p, sigma, eps, nb = 4, seed = 0 ):
    """( directional derivatives from the gradient, the same by central differences ) along `nb` random directions"""
    rng = np.random.default_rng( seed )
    g = rec.value_and_grad( p, sigma )[ 1 ]
    an, fd = [ ], [ ]
    for _ in range( nb ):
        d = rng.normal( size = p.shape )
        an.append( np.sum( g * d ) )
        fd.append( ( rec.value_and_grad( p + eps * d, sigma )[ 0 ] - rec.value_and_grad( p - eps * d, sigma )[ 0 ] ) / ( 2 * eps ) )
    return np.array( an ), np.array( fd )


if test( "2D: at the truth, only the cost of spreading each dirac over its bin remains" ):
    rec = Reconstruction( sino_of_shapes( TRUTH, 0.12 ), TRUTH, radius = 0.12 )
    f = rec.value_and_grad()[ 0 ]
    floor = rec.sinogram.nb_views * rec.sinogram.h ** 2 / 12
    assert f < 1.05 * floor, ( f, floor )
    assert rec.value_and_grad( TRUTH + 0.05 )[ 0 ] > 2 * f

if test( "2D: one shape moved by d pays ( d . axis )^2 more in every view" ):
    s = sino_of_shapes( np.array( [ [ 0.1, 0.0 ] ] ), 0.15, nb_views = 6, nb_bins = 128 )
    rec = Reconstruction( s, [ [ 0.1, 0.0 ] ], radius = 0.15, background = 0.0 )
    d = np.array( [ 0.2, 0.1 ] )
    f0, f1 = rec.value_and_grad()[ 0 ], rec.value_and_grad( np.array( [ [ 0.1, 0.0 ] ] ) + d )[ 0 ]
    assert np.isclose( f1 - f0, np.sum( ( s.axes[ :, 0 ] @ d ) ** 2 ), rtol = 1e-2 ), ( f1 - f0 )

if test( "2D: the gradient matches finite differences" ):
    for n, sigma in ( ( 1, 0.0 ), ( 2, 0.05 ) ):
        rec = Reconstruction( sino_of_shapes( TRUTH, 0.12 ), radius = 0.12, n_per_pixel = n, seed = 3 ).random_points( 6 )
        an, fd = finite_difference( rec, rec.points, sigma, eps = 1e-6 )
        assert np.allclose( an, fd, rtol = 1e-4, atol = 1e-8 ), ( an, fd )

if test( "3D: the gradient matches finite differences" ):
    sino = Sinogram.from_image( ball_packing( 64, dim = 3, ball_radii = ( 0.15, 0.25 ) ), nb_views = 2, nb_bins = 32 )
    rec = Reconstruction( sino, radius = 0.2, background = 1e-2, seed = 0 ).random_points( 6, radius = 0.6, height = 0.3 )
    an, fd = finite_difference( rec, rec.points, sino.h, eps = 1e-4, nb = 2 )
    assert np.allclose( an, fd, rtol = 5e-2, atol = 1e-6 ), ( an, fd )
