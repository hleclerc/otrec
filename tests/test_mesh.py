"""The graded MESH (`mesh.py`): the tiling, the operator, then the mass split.

The first three tests concern EXACT properties (tiling, projected mass, adjoint) and are tight
accordingly; the last one judges what the step is really for -- recovering the share of mass that
is outside the field of view.
"""
import numpy as np

from otrec.mesh import GradedMesh, scan_exterior_scale
from otrec.Sinogram import Sinogram
from errand import test
from loom.testing import need


def _sino( nb_angles = 24, nb_bins = 512, extent = 10.0 ):
    return Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )


if test( "mesh_is_an_exact_tiling" ):
    # no hole and no overlap: each bin of the fine grid must belong to exactly ONE
    # cell. This is the property everything else depends on -- a plane counted twice would skew
    # the mass split, which is the goal of the step.
    mesh = GradedMesh( _sino(), outer_radius = 6.0, inner_radius = 2.0, cell_size = 0.125 )
    assert mesh.nb_levels > 1, "the mesh must really be graded for the test to make sense"

    h, r = mesh.cell_size, mesh.outer_radius
    k = ( np.arange( -int( 2 * r / h ), int( 2 * r / h ) ) + 0.5 ) * h
    x, y = np.meshgrid( k, k, indexing = "ij" )
    keep = np.hypot( x, y ) < r
    fx, fy = x[ keep ], y[ keep ]

    count = np.zeros( len( fx ), dtype = int )
    for ( cx, cy ), s in zip( mesh.centers, mesh.sizes ):
        count += ( np.abs( fx - cx ) < s / 2 ) & ( np.abs( fy - cy ) < s / 2 )
    assert count.min() == 1 and count.max() == 1, (
        f"coverage per fine bin: { count.min() }..{ count.max() } (expected exactly 1)" )
    assert np.isclose( mesh.areas.sum(), len( fx ) * h * h, rtol = 1e-12 )


if test( "mesh_projects_a_disk_exactly" ):
    # the projection of a disk tiled with cells must have the MASS of this tiling, at all
    # angles -- this is what validates the analytic trapezoid and the linear deposit.
    sino = _sino()
    sino.add_disk( center = [ 0.0, 0.0 ], radius = 3.0 )
    mesh = GradedMesh( sino, outer_radius = 5.0, inner_radius = 3.0, cell_size = 0.1 )

    inside = np.linalg.norm( mesh.centers, axis = 1 ) < 3.0
    w = inside.astype( float )
    got = mesh.project( w )
    mass = got.sum( axis = 1 ) * mesh.coarse_dw
    assert np.allclose( mass, mesh.areas[ inside ].sum(), rtol = 1e-9 ), (
        f"projected mass { mass.min() }..{ mass.max() } != area { mesh.areas[ inside ].sum() }" )

    # ... and the shape must follow the analytic projection of the disk, up to the staircase of the tiling
    exp = np.asarray( sino.values ).reshape( mesh.nb_angles, mesh.nb_coarse, mesh.group ).mean( axis = 2 )
    assert np.abs( got - exp ).max() / exp.max() < 0.10


if test( "mesh_adjoint_is_exact" ):
    # `backproject` must be the EXACT adjoint of `project` (symmetric trapezoid + matrix deposit),
    # otherwise the solver does not converge to the right point.
    mesh = GradedMesh( _sino(), outer_radius = 6.0, inner_radius = 2.0, cell_size = 0.2 )
    rng = np.random.default_rng( 0 )
    a = rng.random( mesh.nb_cells )
    b = rng.random( ( mesh.nb_angles, mesh.nb_coarse ) )
    lhs, rhs = float( ( mesh.project( a ) * b ).sum() ), float( a @ mesh.backproject( b ) )
    assert abs( lhs - rhs ) <= 1e-9 * abs( rhs ), f"wrong adjoint: { lhs } != { rhs }"


if test( "mesh_trapezoid_conserves_area" ):
    # the convolution kernel is the projection of a square: its mass equals the area, at any angle.
    mesh = GradedMesh( _sino( nb_angles = 16 ), outer_radius = 3.0, inner_radius = 2.0, cell_size = 0.5 )
    for h in ( 0.3, 0.5, 1.1 ):
        k = mesh._trapezoid( h )
        assert np.allclose( k.sum( axis = 1 ) * mesh.coarse_dw, h * h, rtol = 1e-9 ), (
            f"trapezoid mass != h² for h={ h }" )


if test( "mesh_solve_splits_inside_from_outside" ):
    # what the step is for: an object wider than the detector, for which we want to know what
    # share of mass is INSIDE the field of view -- the quantity that `halo.alternate` had to guess.
    extent, fov = 4.0, 2.0
    sino = Sinogram( nb_angles = 120, nb_bins = 400, extent = extent )
    sino.add_disk( center = [ 0.0, 0.0 ], radius = 1.5 )
    holes = [ ( 0.6, 0.6 ), ( -0.6, 0.6 ), ( 0.6, -0.6 ), ( -0.6, -0.6 ), ( 0.0, 0.0 ) ]
    for h in holes:
        sino.add_disk( center = list( h ), radius = 0.3, density = -1.0 )
    for c in [ ( 2.8, 0.6 ), ( -2.5, -1.2 ), ( 0.4, 2.9 ), ( -1.8, 2.2 ) ]:
        sino.add_disk( center = list( c ), radius = 0.6, density = 0.5 )

    true_inside = np.pi * ( 1.5 ** 2 - len( holes ) * 0.3 ** 2 )
    true_outside = 4 * np.pi * 0.6 ** 2 * 0.5

    mesh = GradedMesh( sino, outer_radius = 4.0, cell_size = 0.08, nb_coarse_bins = 200 )
    mesh.solve( smooth = 3e-2 )                    # L2 + conjugate gradient, the default

    got_in = mesh.interior_mass()
    got_out = mesh.mass() - got_in
    assert abs( got_in / true_inside - 1 ) < 0.10, (
        f"interior mass { got_in:.3f} for { true_inside:.3f} expected" )
    assert abs( got_out / true_outside - 1 ) < 0.25, (
        f"exterior mass { got_out:.3f} for { true_outside:.3f} expected" )

    # and the corrected sinogram must have a much more constant mass per angle
    raw = np.asarray( sino.mass() )
    cor = np.asarray( mesh.corrected().mass() )
    assert cor.std() / cor.mean() < raw.std() / raw.mean() / 4, (
        f"mass per angle not equalized enough: { raw.std() / raw.mean():.4f} -> "
        f"{ cor.std() / cor.mean():.4f}" )


if test( "scan_exterior_scale_is_coherent" ):
    need( "grad" )
    # `alpha` is the degree of freedom that remains open after solving on the mesh: the more
    # exterior is removed, the less mass remains inside, and the more the clipping at 0 bites. The
    # scan must at least respect that -- it does NOT CLAIM to find the right `alpha`, see
    # `scan_exterior_scale`.
    sino = Sinogram( nb_angles = 60, nb_bins = 200, extent = 4.0 )
    sino.add_disk( center = [ 0.0, 0.0 ], radius = 1.5 )
    for c in [ ( 2.8, 0.6 ), ( -2.5, -1.2 ) ]:
        sino.add_disk( center = list( c ), radius = 0.6, density = 0.5 )

    mesh = GradedMesh( sino, outer_radius = 4.0, cell_size = 0.15, nb_coarse_bins = 100 )
    mesh.solve( smooth = 3e-2 )

    scan = scan_exterior_scale(
        mesh, lambda rec: rec.random_points( 200, seed = 1 ).diracs( max_iter = 5 ),
        alphas = [ 0.0, 0.6, 1.2 ] )

    assert np.all( np.diff( scan[ "interior_mass" ] ) < 0 ), (
        f"interior mass not decreasing in alpha: { scan[ 'interior_mass' ] }" )
    assert np.all( np.diff( scan[ "clipped" ] ) >= 0 ), (
        f"clipping not increasing in alpha: { scan[ 'clipped' ] }" )
    assert len( scan[ "clouds" ] ) == 3 and all( c.shape[ 1 ] == 2 for c in scan[ "clouds" ] )
