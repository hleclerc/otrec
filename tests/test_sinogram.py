import numpy as np

from reconstruction.Sinogram import Sinogram
from sdot import SumOfDiracs1d, OtPlan1d
from sdot.testing import test


# -- construction / état initial -----------------------------------------

if test( "init_is_zero" ):
    s = Sinogram( nb_angles = 4, nb_bins = 16, extent = 6.0 )
    assert s.nb_angles.value == 4
    assert s.nb_bins.value == 16
    assert np.asarray( s.values ).shape == ( 4, 16 )
    assert np.all( np.asarray( s.values ) == 0.0 )
    assert np.allclose( np.asarray( s.mass() ), 0.0 )

if test( "geometry_helpers" ):
    s = Sinogram( nb_angles = 3, nb_bins = 10, extent = 4.0, detector_center = 1.0 )
    assert np.isclose( s.dw, 0.4 )
    assert np.isclose( s.s_min, 1.0 - 2.0 )
    assert s.bin_edges.shape == ( 11, )
    assert s.bin_centers.shape == ( 10, )
    # les centres sont au milieu des bords consécutifs
    assert np.allclose( s.bin_centers, 0.5 * ( s.bin_edges[ :-1 ] + s.bin_edges[ 1: ] ) )
    # angles régulièrement répartis sur [0, pi)
    assert np.allclose( s.angles, np.pi * np.arange( 3 ) / 3 )

if test( "project_points" ):
    s = Sinogram( nb_angles = 2, nb_bins = 8, extent = 4.0 )  # angles 0 et pi/2
    sp = np.asarray( s.project_points( [ [ 1.0, 2.0 ], [ -0.5, 0.3 ] ] ) )  # Tensor -> np
    assert sp.shape == ( 2, 2 )
    # angle 0 -> normale (1,0) -> s = x ; angle pi/2 -> normale (0,1) -> s = y
    assert np.allclose( sp[ 0 ], [ 1.0, -0.5 ] )
    assert np.allclose( sp[ 1 ], [ 2.0, 0.3 ] )


# -- add_disk : invariants physiques -------------------------------------

if test( "add_disk_mass_conservation" ):
    # la transformée de Radon préserve l'intégrale : masse par angle == densité·pi·r²
    s = Sinogram( nb_angles = 5, nb_bins = 401, extent = 8.0 )
    s.add_disk( center = [ 0.4, -0.3 ], radius = 1.2, density = 2.0 )
    expected = 2.0 * np.pi * 1.2 ** 2
    assert np.allclose( np.asarray( s.mass() ), expected, atol = 1e-6 )

if test( "add_disk_centroid" ):
    # le centroïde du profil == centre du disque projeté sur le détecteur
    center = np.array( [ 0.5, -0.3 ] )
    s = Sinogram( nb_angles = 6, nb_bins = 401, extent = 8.0 )
    s.add_disk( center = center, radius = 1.0 )
    vals = np.asarray( s.values )
    m = np.asarray( s.mass() )
    centroid = ( vals * s.bin_centers[ None, : ] ).sum( axis = 1 ) * s.dw / m
    assert np.allclose( centroid, s.normals @ center, atol = 1e-3 )

if test( "add_disk_is_additive" ):
    s = Sinogram( nb_angles = 3, nb_bins = 201, extent = 8.0 )
    s.add_disk( center = [ 0.0, 0.0 ], radius = 1.0 )
    s.add_disk( center = [ 0.5, 0.2 ], radius = 0.7, density = 3.0 )
    expected = np.pi * 1.0 ** 2 + 3.0 * np.pi * 0.7 ** 2
    assert np.allclose( np.asarray( s.mass() ), expected, atol = 1e-5 )

if test( "add_disk_returns_self_and_nonnegative" ):
    s = Sinogram( nb_angles = 4, nb_bins = 64, extent = 6.0 )
    assert s.add_disk( center = [ 0.2, 0.1 ], radius = 1.0 ) is s
    assert np.all( np.asarray( s.values ) >= 0.0 )


# -- consommation : image( k ) -------------------------------------------

if test( "image_mass_matches" ):
    s = Sinogram( nb_angles = 3, nb_bins = 201, extent = 8.0 )
    s.add_disk( center = [ 0.3, -0.2 ], radius = 1.0, density = 1.5 )
    for k in range( 3 ):
        assert np.isclose( float( s.image( k ).mass ), float( s.mass( k ) ), atol = 1e-6 )

if test( "image_plugs_into_otplan1d" ):
    # smoke test : le profil sert de distribution cible face à des diracs projetés
    s = Sinogram( nb_angles = 2, nb_bins = 101, extent = 8.0 )
    s.add_disk( center = [ 0.0, 0.0 ], radius = 1.0 )

    points = np.array( [ [ -0.4, 0.1 ], [ 0.3, -0.2 ], [ 0.1, 0.5 ] ] )
    positions = s.project_points( points )[ 0 ]              # angle 0

    otp = OtPlan1d( SumOfDiracs1d( positions = positions ), s.image( 0 ) )
    cost = float( otp.cost )
    assert np.isfinite( cost ) and cost >= 0.0

if test( "single_angle_batched_image" ):
    # régression : un axe de batch d'EXTENT 1 ne doit pas faire passer les membres PARTAGÉS
    # (origin/frame) de l'image batchée pour des entrées batchées. Sinon leurs NOMS gagnent
    # `num_angle` sans que le buffer gagne le rang -> `Image::measure()` (kernel `mass`) ne compile
    # plus (`without_index` sur un `Tuple<>`). Le chemin batché doit tourner à nb_angles = 1.
    s = Sinogram( nb_angles = 1, nb_bins = 101, extent = 8.0 )
    s.add_disk( center = [ 0.0, 0.0 ], radius = 1.0 )

    projected = s.project_points( np.array( [ [ -0.4, 0.1 ], [ 0.3, -0.2 ], [ 0.1, 0.5 ] ] ) )  # [ 1, 3 ]
    src = SumOfDiracs1d( positions = projected, batch_axes = [ s.num_angle ] )
    cost = OtPlan1d( src, s.batched_image() ).cost
    assert np.all( np.isfinite( np.asarray( cost ) ) )


# -- validation des arguments --------------------------------------------

if test( "invalid_construction" ):
    for kwargs in ( dict( nb_angles = 0, nb_bins = 8, extent = 1.0 ),
                    dict( nb_angles = 4, nb_bins = 0, extent = 1.0 ),
                    dict( nb_angles = 4, nb_bins = 8, extent = 0.0 ) ):
        try:
            Sinogram( **kwargs )
        except ValueError:
            pass
        else:
            raise AssertionError( f"attendu ValueError pour { kwargs }" )

if test( "invalid_add_disk" ):
    s = Sinogram( nb_angles = 2, nb_bins = 8, extent = 4.0 )
    for args in ( dict( center = [ 0, 0, 0 ], radius = 1.0 ),   # mauvaise shape
                  dict( center = [ 0, 0 ], radius = 0.0 ) ):    # rayon nul
        try:
            s.add_disk( **args )
        except ValueError:
            pass
        else:
            raise AssertionError( f"attendu ValueError pour { args }" )
