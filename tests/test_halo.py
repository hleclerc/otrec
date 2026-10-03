"""The HALO (`halo.py`): the footprint on the sinogram of the matter outside the field of view.

Three levels, from the most local to the most global: the projection OPERATOR of a cell (compared
with the analytic projection of an annulus), the positive FIT that recovers a known outside mass,
and finally the full ALTERNATION, judged on what motivates the whole module -- the voids
preserved in the reconstruction.
"""
import numpy as np

from otrec.halo import ( Halo, alternate, interior_values, mass_profile,
                                  scan_interior_mass, void_fraction )
from otrec.Sinogram import Sinogram
from errand import test
from loom.testing import need


def _annulus_sinogram( a, b, nb_angles = 5, nb_bins = 256, extent = 10.0 ):
    """EXACT projection of a centered annulus [ a, b ], as the difference of two disks."""
    s = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    s.add_disk( center = [ 0.0, 0.0 ], radius = b )
    s.add_disk( center = [ 0.0, 0.0 ], radius = a, density = -1.0 )
    return s


if test( "halo_operator_matches_annulus" ):
    # a halo with ONE ring and ONE sector IS an annulus: its operator must reproduce the analytic
    # projection (exact, via `add_disk`). Also checks the exact radial integration and the
    # deposit of the ramps through the primitive.
    a, b, extent, nb_bins = 1.0, 2.0, 10.0, 256
    ref = _annulus_sinogram( a, b, nb_bins = nb_bins, extent = extent )
    halo = Halo( ref, outer_radius = b, inner_radius = a, growth = b / a, nb_sectors = 1,
                 nb_coarse_bins = 64 )
    assert halo.nb_cells == 1, f"expected a single sector, got { halo.nb_cells }"

    got = halo.operator[ 0 ]                                          # [ nb_angles, nb_coarse ]
    exp = np.asarray( ref.values ).reshape( got.shape[ 0 ], got.shape[ 1 ], halo.group ).mean( axis = 2 )

    # mass: the annulus fits in the detector, it must equal its area at all angles
    area = np.pi * ( b * b - a * a )
    mass = got.sum( axis = 1 ) * halo.coarse_dw
    assert np.allclose( mass, area, rtol = 1e-3 ), f"mass { mass } != area { area }"

    # shape: the radial integration is exact, only the angular quadrature remains
    err = np.abs( got - exp ).max() / exp.max()
    assert err < 0.02, f"profile too far from the analytic annulus: relative error { err }"

    # a centered annulus projects the SAME thing at all angles
    assert np.abs( got - got[ 0 ] ).max() / exp.max() < 0.02, "projection not invariant across angles"


if test( "halo_operator_sector_partition" ):
    # the sectors of an annulus form a PARTITION: their operators must sum back to
    # that of the whole annulus (at equal density).
    a, b = 1.0, 2.0
    ref = _annulus_sinogram( a, b )
    whole = Halo( ref, outer_radius = b, inner_radius = a, growth = b / a, nb_sectors = 1, nb_coarse_bins = 64 )
    split = Halo( ref, outer_radius = b, inner_radius = a, growth = b / a, nb_sectors = 8, nb_coarse_bins = 64 )
    assert split.nb_cells > 1

    summed = split.operator.sum( axis = 0 )
    err = np.abs( summed - whole.operator[ 0 ] ).max() / whole.operator[ 0 ].max()
    assert err < 0.02, f"the sum of the sectors does not give back the annulus : { err }"
    assert np.isclose( split.areas.sum(), whole.areas.sum(), rtol = 1e-12 )


if test( "interior_values_conserves_mass" ):
    # the projection of the cloud must carry EXACTLY the requested mass, at all angles (linear
    # deposit onto the two neighboring bins) -- this is what makes the residual interpretable.
    sino = Sinogram( nb_angles = 7, nb_bins = 128, extent = 8.0 )
    pts = np.random.default_rng( 0 ).normal( scale = 0.6, size = ( 500, 2 ) )
    vals = interior_values( sino, pts, mass = 3.0 )
    assert np.allclose( vals.sum( axis = 1 ) * sino.dw, 3.0, rtol = 1e-9 )

    # with a radius, we go through `DiskProjector`: same mass
    vals = interior_values( sino, pts, mass = 3.0, radius = 0.2 )
    assert np.allclose( vals.sum( axis = 1 ) * sino.dw, 3.0, rtol = 1e-3 )


if test( "halo_fit_recovers_outside_mass" ):
    # ground truth: one disk INSIDE the field, one disk OUTSIDE. We give the halo the exact residual
    # (measured minus the projection of the inner disk) and it must recover the outside mass.
    extent, nb_bins, nb_angles = 4.0, 400, 60
    inside, outside, r_out = ( 0.2, -0.1 ), ( 2.6, 0.4 ), 0.5

    full = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    full.add_disk( center = list( inside ), radius = 0.6 )
    full.add_disk( center = list( outside ), radius = r_out )

    only_in = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    only_in.add_disk( center = list( inside ), radius = 0.6 )

    residual = np.asarray( full.values ) - np.asarray( only_in.values )
    per_angle = mass_profile( full )
    target = per_angle - per_angle.min()

    halo = Halo( full, outer_radius = 4.0, nb_coarse_bins = 100 )
    halo.fit( residual, target_mass = target )

    # the halo only sees what falls in the detector: that is the mass that is constrained
    got, exp = halo.visible_mass(), residual.sum( axis = 1 ) * full.dw
    err = np.abs( got - exp ).max() / exp.max()
    assert err < 0.12, f"visible halo mass poorly recovered : { err }"

    # and it is indeed placed OUTSIDE, on the right side
    weight_by_ring = { }
    for w, ( _, _, p0, p1, _ ) in zip( halo.weights, halo.cells ):
        weight_by_ring[ 0.5 * ( p0 + p1 ) ] = weight_by_ring.get( 0.5 * ( p0 + p1 ), 0.0 ) + w
    best = max( weight_by_ring, key = weight_by_ring.get )
    expected_phi = np.arctan2( outside[ 1 ], outside[ 0 ] ) % ( 2 * np.pi )
    gap = abs( ( best - expected_phi + np.pi ) % ( 2 * np.pi ) - np.pi )
    assert gap < 1.0, f"halo placed at φ={ best:.2f} instead of { expected_phi:.2f}"


if test( "halo_corrected_equalizes_mass" ):
    # after correction, `∫p_θ` must be MUCH flatter: it is the direct measure of the leakage.
    extent, nb_bins, nb_angles = 4.0, 400, 60
    full = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    full.add_disk( center = [ 0.2, -0.1 ], radius = 0.6 )
    full.add_disk( center = [ 2.6, 0.4 ], radius = 0.5 )
    full.add_disk( center = [ -2.2, -1.5 ], radius = 0.4 )

    only_in = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    only_in.add_disk( center = [ 0.2, -0.1 ], radius = 0.6 )

    per_angle = mass_profile( full )
    halo = Halo( full, outer_radius = 4.0, nb_coarse_bins = 100 )
    halo.fit( np.asarray( full.values ) - np.asarray( only_in.values ),
              target_mass = per_angle - per_angle.min() )

    # three COMPACT outside blobs, the most demanding case for a deliberately coarse mesh
    # (an extended outside object, the real case, does much better) -- hence a factor 3 and
    # not an order of magnitude.
    before = per_angle.std() / per_angle.mean()
    after_profile = mass_profile( halo.corrected() )
    after = after_profile.std() / after_profile.mean()
    assert after < before / 3, f"mass per angle not equalized enough: { before:.4f} -> { after:.4f}"


if test( "alternate_preserves_voids" ):
    need( "grad" )
    # end to end, on what motivates the module: an object with holes INSIDE the field, matter
    # OUTSIDE. Without a halo, the excess mass is redistributed in the field and fills the holes;
    # with one, the holes must come back.
    extent, nb_bins, nb_angles = 4.0, 300, 90
    rng = np.random.default_rng( 0 )

    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    holes = [ ( 0.55, 0.55 ), ( -0.55, 0.55 ), ( 0.55, -0.55 ), ( -0.55, -0.55 ) ]
    sino.add_disk( center = [ 0.0, 0.0 ], radius = 1.4 )                       # solid object
    for h in holes:
        sino.add_disk( center = list( h ), radius = 0.32, density = -1.0 )     # ... with holes
    for c in [ ( 2.6, 0.5 ), ( -2.4, -1.0 ), ( 0.3, 2.7 ) ]:                   # matter OUTSIDE
        sino.add_disk( center = list( c ), radius = 0.5 )

    def solve( rec ):
        return rec.diracs( max_iter = 60 )

    common = dict( outer_radius = 4.0, extent = extent, seed = 1, nb_points = 600,
                   max_residual_points = 20_000,
                   halo_kwargs = dict( nb_coarse_bins = 100 ) )
    plain, _ = alternate( sino, solve, nb_outer = 1, **common )                # null halo
    fixed, halo = alternate( sino, solve, nb_outer = 3, **common )

    # the halo must have found something, and flattened the mass per angle
    assert halo.mass() > 0, "halo stayed null"
    raw, cor = mass_profile( sino ), mass_profile( halo.corrected() )
    assert cor.std() / cor.mean() < raw.std() / raw.mean() / 3, "mass per angle not equalized"

    # ... and the holes must be emptier. We measure them where they are, not globally.
    def in_holes( pts ):
        pts = np.asarray( pts )
        inside = [ ( ( pts - np.array( h ) ) ** 2 ).sum( axis = 1 ) < 0.25 ** 2 for h in holes ]
        return float( np.any( inside, axis = 0 ).mean() )

    assert in_holes( fixed.positions ) < in_holes( plain.positions ) * 0.7, (
        f"the holes did not empty out: { in_holes( plain.positions ):.4f} -> "
        f"{ in_holes( fixed.positions ):.4f}" )
    # ... and fewer points must linger outside the object (the excess mass went there too)
    def outside_frac( pts ):
        return float( ( np.linalg.norm( np.asarray( pts ), axis = 1 ) > 1.5 ).mean() )
    assert outside_frac( fixed.positions ) < outside_frac( plain.positions )
    assert void_fraction( fixed.positions, extent ) > void_fraction( plain.positions, extent )


if test( "scan_interior_mass_is_monotone" ):
    # `M_in` being the decisive parameter (see `halo.scan_interior_mass`), the scan must at least
    # be consistent: the less mass is attributed to the inside, the more the halo takes.
    extent, nb_bins, nb_angles = 4.0, 400, 60
    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    sino.add_disk( center = [ 0.0, 0.0 ], radius = 1.0 )
    sino.add_disk( center = [ 2.6, 0.4 ], radius = 0.5 )

    rng = np.random.default_rng( 0 )
    pts = rng.normal( scale = 0.5, size = ( 3000, 2 ) )
    halo = Halo( sino, outer_radius = 4.0, nb_coarse_bins = 100 )
    before = halo.weights.copy()

    scan = scan_interior_mass( halo, pts )
    assert np.all( np.diff( scan[ "halo_mass" ] ) < 0 ), (
        f"halo mass not decreasing in M_in: { scan[ 'halo_mass' ] }" )
    assert np.array_equal( halo.weights, before ), "the scan must not modify the halo"


if test( "void_fraction_is_calibrated" ):
    rng = np.random.default_rng( 0 )
    n, extent = 4000, 2.0
    # a uniform cloud over the WHOLE domain, at the default grid (√n per side): ~1/e void
    full = ( rng.random( ( n, 2 ) ) - 0.5 ) * extent
    assert 0.3 < void_fraction( full, extent ) < 0.45

    # the same cloud packed into the left half: it can no longer fill anything on the right
    half = full.copy()
    half[ :, 0 ] = half[ :, 0 ] / 2 - extent / 4
    assert void_fraction( half, extent ) > void_fraction( full, extent ) + 0.15
