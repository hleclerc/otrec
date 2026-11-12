"""`Radiographs`: the data of a 3D reconstruction -- one 2D image per angle, accumulated from
balls whose projection is known ( `add_sphere` ). The counterpart of `test_sinogram.py`."""
import numpy as np

from otrec.Radiographs import Radiographs
from sdot import OtPlan, SumOfDiracs
from errand import test
from loom.testing import need


if test( "init_is_zero" ):
    r = Radiographs( nb_angles = 4, nb_u = 16, nb_v = 12, extent_u = 6.0, extent_v = 4.0 )
    assert r.nb_angles.value == 4 and r.nb_u.value == 16 and r.nb_v.value == 12
    assert np.asarray( r.values ).shape == ( 4, 16, 12 )
    assert np.all( np.asarray( r.values ) == 0.0 )
    assert np.allclose( r.mass(), 0.0 )
    assert np.isclose( r.du, 6.0 / 16 ) and np.isclose( r.dv, 4.0 / 12 )
    assert np.isclose( r.u_min, -3.0 ) and np.isclose( r.v_min, -2.0 )
    assert np.allclose( r.u_centers, 0.5 * ( r.u_edges[ :-1 ] + r.u_edges[ 1: ] ) )


if test( "project_points" ):
    # angles 0 and pi/2: at 0 the ray is `x`, the detector sees ( y, z ); at pi/2 the ray is `y`
    # and the detector sees ( -x, z )
    r = Radiographs( nb_angles = 2, nb_u = 8, nb_v = 8, extent_u = 4.0 )
    pts = np.array( [ [ 1.0, 2.0, 3.0 ], [ -0.5, 0.3, 0.1 ] ] )
    uv = r.project_points( pts )
    assert uv.shape == ( 2, 2, 2 )
    assert np.allclose( uv[ 0 ], [ [ 2.0, 3.0 ], [ 0.3, 0.1 ] ] )
    assert np.allclose( uv[ 1 ], [ [ -1.0, 3.0 ], [ 0.5, 0.1 ] ] )

    # `unproject_grad` is the transpose of `project_points`: `< P p, g > == < p, P^T g >`
    rng = np.random.default_rng( 0 )
    g = rng.normal( size = uv.shape )
    assert np.isclose( ( uv * g ).sum(), ( pts * r.unproject_grad( g ) ).sum() )


if test( "add_sphere_mass_and_shape" ):
    # the mass per angle is the volume of the ball ( up to quadrature errors, on the pixels
    # crossed by the edge ), the same at all angles; the maximum is at the projected center and
    # equals the diameter; outside the shadow, zero.
    r = Radiographs( nb_angles = 6, nb_u = 96, nb_v = 96, extent_u = 6.0 )
    center, radius = np.array( [ 0.4, -0.3, 0.2 ] ), 1.0
    r.add_sphere( center, radius )

    vol = 4 / 3 * np.pi * radius ** 3
    assert np.allclose( r.mass(), vol, rtol = 2e-3 ), r.mass() / vol

    vals = np.asarray( r.values )
    uv0 = r.project_points( center[ None ] )[ :, 0 ]
    for k in range( 6 ):
        i, j = np.unravel_index( vals[ k ].argmax(), vals[ k ].shape )
        assert abs( r.u_centers[ i ] - uv0[ k, 0 ] ) <= r.du and abs( r.v_centers[ j ] - uv0[ k, 1 ] ) <= r.dv
        assert vals[ k ].max() <= 2 * radius + 1e-12 and vals[ k ].max() > 1.9 * radius
        far = ( r.u_centers[ :, None ] - uv0[ k, 0 ] ) ** 2 + ( r.v_centers[ None, : ] - uv0[ k, 1 ] ) ** 2 > ( radius + r.du + r.dv ) ** 2
        assert np.all( vals[ k ][ far ] == 0 )

    # two balls add up, a density weights
    r2 = Radiographs( nb_angles = 6, nb_u = 96, nb_v = 96, extent_u = 6.0 )
    r2.add_sphere( center, radius, density = 2.0 ).add_sphere( -center, 0.5 )
    assert np.allclose( r2.mass(), 2 * vol + 4 / 3 * np.pi * 0.5 ** 3, rtol = 3e-3 )


if test( "images_feed_a_2d_transport" ):
    need( "cpu" )
    # a radiograph is a 2D `Image` whose orientation follows ( u, v ): the barycenter of the
    # cell of a single dirac is the projected center of the ball -- which checks both
    # `origin` / `frame` and that `OtPlan` consumes it as is
    r = Radiographs( nb_angles = 3, nb_u = 40, nb_v = 30, extent_u = 4.0, extent_v = 3.0 )
    center = np.array( [ 0.6, -0.2, 0.35 ] )
    r.add_sphere( center, 0.5 )
    uv0 = r.project_points( center[ None ] )[ :, 0 ]
    for k in range( 3 ):
        plan = OtPlan( SumOfDiracs( np.array( [ [ 0.1, 0.1 ] ] ) ), r.image( k ), max_iter = 1 )
        _, bary, m = plan.transport()
        assert np.isclose( m[ 0 ], 1.0 ) and np.allclose( bary[ 0 ], uv0[ k ], atol = 1e-3 ), ( bary, uv0[ k ] )


if test( "visual_hull_points_project_on_matter" ):
    # each point drawn in the visual hull projects, at ALL angles, onto a nonzero pixel
    r = Radiographs( nb_angles = 5, nb_u = 64, nb_v = 64, extent_u = 4.0 )
    r.add_sphere( [ 0.5, -0.3, 0.2 ], 0.5 ).add_sphere( [ -0.6, 0.4, -0.5 ], 0.3 )
    pts = r.visual_hull_points( 500, seed = 3 )
    assert pts.shape == ( 500, 3 )
    vals = np.asarray( r.values )
    uv = r.project_points( pts )
    iu = np.floor( ( uv[ ..., 0 ] - r.u_min ) / r.du ).astype( int )
    iv = np.floor( ( uv[ ..., 1 ] - r.v_min ) / r.dv ).astype( int )
    assert np.all( vals[ np.arange( 5 )[ :, None ], iu, iv ] > 0 )
    # and with five angles, the hull is almost the balls themselves
    d = np.minimum( np.linalg.norm( pts - [ 0.5, -0.3, 0.2 ], axis = 1 ) - 0.5,
                    np.linalg.norm( pts - [ -0.6, 0.4, -0.5 ], axis = 1 ) - 0.3 )
    assert ( d < 0.15 ).mean() > 0.9
