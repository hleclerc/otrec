"""`Reconstruction2d( ..., mesh = )`: the angles split between devices ( 4 CPU devices, forced before jax starts: a file
of its own )."""
import os

os.environ[ "XLA_FLAGS" ] = os.environ.get( "XLA_FLAGS", "" ) + " --xla_force_host_platform_device_count=4"

from errand import test


if test( "a reconstruction with its angles on four devices follows the one on a single device" ):
    import jax
    import numpy as np
    from jax.sharding import Mesh
    from loom.testing import need
    from otrec.Sinogram import Sinogram
    from otrec.reconstruction2d import Reconstruction2d
    need( "cpu" )
    sino = Sinogram( nb_angles = 16, nb_bins = 128, extent = 6.0 )
    sino.add_disk( center = [ 0.3, -0.2 ], radius = 1.0 )
    mesh = Mesh( np.array( jax.devices() ), ( "angles", ) )
    one = Reconstruction2d( sino, extent = 5.0, dtype = np.float64 ).random_points( 200, seed = 1 ).diracs( max_iter = 30 )
    four = Reconstruction2d( sino, extent = 5.0, dtype = np.float64, mesh = mesh ).random_points( 200, seed = 1 ).diracs( max_iter = 30 )
    # the same steps, up to the order of the sums between devices: equal for the first steps, then the roundings
    # grow through the line searches ( a diracs loss is not smooth ) -- both runs must converge as well
    a, b = one.history[ -1 ][ "losses" ], four.history[ -1 ][ "losses" ]
    assert np.allclose( a[ :8 ], b[ :8 ], rtol = 1e-8 ), ( a, b )
    assert a[ -1 ] < a[ 0 ] / 1000 and b[ -1 ] < b[ 0 ] / 1000 and abs( a[ -1 ] - b[ -1 ] ) < 0.2 * a[ -1 ], ( a[ -1 ], b[ -1 ] )
