from errand import test


if test( "a 360 degree scan, off-centre: its axis found, it gives back the sinogram over 180 degrees" ):
    import numpy as np
    from otrec import Scan, Sinogram, ball_packing
    ref = Sinogram.from_image( ball_packing( 128 ), 60, 64 )                      # 180 degrees, the axis at the centre
    # the same object on a wider detector ( 5 more pixels to the right ), over 360 degrees: t + pi is t mirrored
    values = np.concatenate( [ ref.values, ref.values[ :, ::-1 ] ] ) / ref.h                # line integrals
    scan = Scan( np.pad( values, ( ( 0, 0 ), ( 0, 5 ) ) ), np.pi * np.arange( 120 ) / 60 )
    assert abs( scan.rotation_axis() - 31.5 ) < 0.1                                # ( within a tenth of a pixel )
    scan = scan.centred( 31.5 )
    assert scan.nb_pixels == 64 and scan.is_centred
    s = scan.sinogram( nb_bins = 64, nb_views = 60 )
    assert np.allclose( s.values, ref.values ) and np.allclose( s.axes, ref.axes )
    s = scan.sinogram( nb_bins = 32, nb_views = 30 )                              # pixels and views averaged
    assert np.allclose( s.values, ref.values.reshape( 30, 2, 32, 2 ).sum( 3 ).mean( 1 ) )
    assert np.isclose( s.fbp().sum(), ref.fbp().sum() / 4, rtol = 0.05 )           # ( the same density, 4 times fewer pixels )

if test( "counts -> line integrals; the air level taken off; beam hardening undone; negatives clipped" ):
    import numpy as np
    from otrec import Scan
    p = np.zeros( ( 3, 40 ) )
    p[ 0, 15:25 ] = 1.0                                                          # 3 views of the same mass, in 3 shapes
    p[ 1, 10:30 ] = 0.5
    p[ 2, 18:22 ] = 2.5
    hard = p - 0.1 * p ** 2                                                      # what a polychromatic beam measures
    flat, dark = np.full( ( 2, 40 ), 1000.0 ), np.full( ( 2, 40 ), 10.0 )
    counts = dark[ 0 ] + ( flat[ 0 ] - dark[ 0 ] ) * np.exp( -( hard + 0.05 ) )  # and the air at 0.05
    scan = Scan.from_counts( counts, flat, dark, np.arange( 3 ) * np.pi / 3 )
    assert np.allclose( scan.values, hard + 0.05 )
    scan = scan.minus_air( 0.1 )
    assert np.allclose( scan.values, hard )
    s = scan.linearized().values.sum( 1 )
    assert np.ptp( s ) < np.ptp( hard.sum( 1 ) ) / 5                             # ( one coefficient: not exact )
    assert ( Scan( p - 0.5, scan.angles ).clipped().values >= 0 ).all()


if test( "zeroed_air: the air at 0, the matter kept; equal_masses: one mass per view" ):
    import numpy as np
    from otrec import Scan
    rng = np.random.default_rng( 0 )
    V, N = 40, 200
    x = np.arange( N ) - ( N - 1 ) / 2
    matter = np.clip( 1 - ( x / 50 ) ** 2, 0, None ) * 0.5                       # in the middle, air on both sides
    values = matter * ( 1 + 0.05 * np.sin( np.arange( V ) ) )[ :, None ] + 0.01 * rng.normal( size = ( V, N ) )
    scan = Scan( values, np.linspace( 0, np.pi, V, endpoint = False ) )
    z = scan.zeroed_air( 3.0, 0.1 )
    assert ( z.values[ :, :40 ] == 0 ).mean() > 0.99 and ( z.values[ :, -40: ] == 0 ).mean() > 0.99
    assert ( z.values >= 0 ).all()
    inside = np.abs( x ) < 40
    assert np.allclose( z.values[ :, inside ], values[ :, inside ] )             # ( well above the noise: untouched )
    m = z.equal_masses().values.sum( 1 )
    assert np.allclose( m, m.mean() ) and np.isclose( m.mean(), z.values.sum( 1 ).mean() )


if test( "aligned: the axis and the motion of the sample from the centres of mass; rings_removed: the bias of each pixel" ):
    import numpy as np
    from otrec import Scan
    rng = np.random.default_rng( 1 )
    V, N, R, cor = 180, 128, 12.0, 61.3
    t = np.linspace( 0, np.pi, V, endpoint = False )
    motion = rng.normal( scale = 1.0, size = V )
    centre = cor + 20 * np.cos( t - 0.4 ) + motion                       # a disk off the axis, the sample shaking
    edges = np.arange( N + 1 ) - 0.5
    F = lambda u: R * R * ( np.arcsin( np.clip( u / R, -1, 1 ) ) + np.clip( u / R, -1, 1 ) * np.sqrt( 1 - np.clip( u / R, -1, 1 ) ** 2 ) )
    values = np.diff( F( edges[ None ] - centre[ :, None ] ), axis = 1 )   # the exact chord integrated over each pixel
    a = Scan( values, t ).aligned()
    assert abs( a.cor - cor ) < 0.05, a.cor
    # ( what of the motion is a sinusoid of the angle cannot be told from the path of the object: the rest comes back )
    A = np.stack( [ np.ones( V ), np.cos( t ), np.sin( t ) ], 1 )
    rest = motion - A @ np.linalg.lstsq( A, motion, rcond = None )[ 0 ]
    assert np.std( a.shifts + rest ) < 0.05 * np.std( rest )
    # ( what departs from a cubic over 9 pixels goes: a bias that varies slowly along the detector stays )
    bias = 0.02 * values.max() * rng.normal( size = N )
    r = Scan( values + bias, t ).rings_removed()
    from scipy.signal import savgol_filter
    # ( and a little of the object: where its mean view has a kink, at the ends of the band the disk sweeps )
    assert np.std( ( r.values - values ).mean( 0 ) - savgol_filter( bias, 9, 3, mode = "interp" ) ) < 0.2 * bias.std()


if test( "sinogram( reduction ): that many times fewer bins and views" ):
    import numpy as np
    from otrec import Scan
    s = Scan( np.random.default_rng( 0 ).random( ( 120, 64 ) ), np.linspace( 0, np.pi, 120, endpoint = False ) )
    full, coarse = s.sinogram(), s.sinogram( reduction = 4 )
    assert ( full.nb_views, full.nb_bins ) == ( 120, 64 ) and ( coarse.nb_views, coarse.nb_bins ) == ( 30, 16 )
    assert np.isclose( coarse.values.sum() * 4, full.values.sum() )            # ( views averaged, bins summed )
