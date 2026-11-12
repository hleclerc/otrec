import numpy as np

from errand import test

from otrec import ball_packing
from otrec.packing.gravity import contacts, settle

if test( "gravity gives a dense packing: 4 contacts per disc, 6 per ball, hardly any overlap" ):
    for dim, half_width, floor, density, nb_contacts in ( ( 2, 0.9, -0.9, 0.80, 3.8 ), ( 3, 0.9, -0.35, 0.58, 5.5 ) ):
        rng = np.random.default_rng( 0 )
        radii = rng.uniform( 0.05, 0.1, 250 if dim == 2 else 500 ) * ( 1 if dim == 2 else 1.5 )
        x = settle( radii, dim, half_width, floor )
        pairs, overlap = contacts( x, radii )
        assert overlap.max() < 0.05 * radii.mean(), overlap.max()
        assert ( np.abs( x[ :, :-1 ] ) + radii[ :, None ] < half_width + 0.01 ).all() and ( x[ :, -1 ] - radii > floor - 0.01 ).all()
        # away from the walls and from the free surface
        top = np.sort( x[ :, -1 ] )[ len( x ) // 2 ]
        core = ( np.abs( x[ :, :-1 ] ).max( 1 ) < half_width - 0.2 ) & ( x[ :, -1 ] > floor + 0.2 ) & ( x[ :, -1 ] < top )
        assert np.bincount( pairs.ravel(), minlength = len( x ) )[ core ].mean() > nb_contacts
        p = np.concatenate( [ rng.uniform( -half_width + 0.2, half_width - 0.2, ( 50_000, dim - 1 ) ), rng.uniform( floor + 0.2, top, ( 50_000, 1 ) ) ], 1 )
        inside = ( np.linalg.norm( p[ :, None ] - x[ None ], axis = 2 ) < radii[ None ] ).any( 1 ).mean()
        assert inside > density, inside

if test( "ball_packing: an image in [ 0, 1 ], a disc ( a cylinder ) with a good share of holes" ):
    image = ball_packing( 128 )
    assert image.shape == ( 128, 128 ) and image.min() >= 0 and image.max() <= 1
    assert 0.2 < image.sum() * ( 2 / 128 ) ** 2 / ( np.pi * 0.8 ** 2 ) < 0.45
    volume = ball_packing( 64, dim = 3, ball_radii = ( 0.1, 0.18 ) )
    assert volume.shape == ( 64, 64, 64 ) and 0.4 < volume.sum() * ( 2 / 64 ) ** 3 / ( np.pi * 0.8 ** 2 * 0.4 ) < 0.65
    assert volume[ :, :, :10 ].max() == 0                                     # ( nothing under the cylinder )
