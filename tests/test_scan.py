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
