import numpy as np

from errand import test

from otrec import Reconstruction, Sinogram, ball_packing

TRUTH = np.array( [ [ 0.30, 0.10 ], [ -0.25, 0.35 ], [ -0.10, -0.40 ], [ 0.45, -0.30 ] ] )


def sino_of_shapes( points, radius, nb_views = 12, nb_bins = 64, k = 2 ):
    """2D: the sinogram a cloud of shapes gives exactly ( its own image, without blur nor background )"""
    s = Sinogram( 2, nb_views, nb_bins, extent = 2.0 )
    s.values = Reconstruction( s, points, radius = radius, k = k ).image( sigma = 0.0, background = 0.0 )
    return s


def finite_difference( fun, x, eps, nb = 4, seed = 0 ):
    """( directional derivatives from the gradient, the same by central differences ) along `nb` random directions;
    `fun( x ) -> ( value, gradient )`, on arrays"""
    rng = np.random.default_rng( seed )
    g = fun( x )[ 1 ]
    an, fd = [ ], [ ]
    for _ in range( nb ):
        d = rng.normal( size = x.shape )
        an.append( np.sum( g * d ) )
        fd.append( ( fun( x + eps * d )[ 0 ] - fun( x - eps * d )[ 0 ] ) / ( 2 * eps ) )
    return np.array( an ), np.array( fd )


def on_arrays( rec, loss_fn ):
    def fun( x ):
        f, g = loss_fn.value_and_grad( rec.model.tensor( x ), 0.0, 1e-3 )
        return f, g.cpu().numpy()
    return fun


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
        rec = Reconstruction( sino_of_shapes( TRUTH, 0.12 ), n_per_pixel = n ).add_random_balls( 6, 0.12 )
        an, fd = finite_difference( lambda p: rec.value_and_grad( p, sigma ), rec.points, eps = 1e-6 )
        assert np.allclose( an, fd, rtol = 1e-4, atol = 1e-8 ), ( an, fd )

if test( "3D: the gradient matches finite differences" ):
    sino = Sinogram.from_image( ball_packing( 64, dim = 3, ball_radii = ( 0.15, 0.25 ) ), nb_views = 2, nb_bins = 32 )
    rec = Reconstruction( sino, background = 1e-2 ).add_random_balls( 6, 0.2 )
    an, fd = finite_difference( lambda p: rec.value_and_grad( p, sino.h ), rec.points, eps = 1e-4, nb = 2 )
    assert np.allclose( an, fd, rtol = 5e-2, atol = 1e-6 ), ( an, fd )

if test( "a shape that would leave what the views see is pulled back inside" ):
    import torch
    from otrec.shapes import balls
    rec = Reconstruction( sino_of_shapes( TRUTH, 0.12 ), [ [ 0.95, 0.0 ], [ 0.0, -3.0 ], [ 0.2, 0.1 ] ], radius = 0.12 )
    points = balls.shapes( rec.model.tensor( rec.points ), rec.model )[ 0 ]
    assert torch.allclose( torch.linalg.vector_norm( points, dim = 1 ) + 0.12, rec.model.tensor( [ 1.0, 1.0, 0.05 ** 0.5 + 0.12 ] ) )
    assert np.isclose( rec.image( background = 0.0 ).sum( 1 ), 1.0 ).all()                      # ( nothing lost on the detector )

if test( "2D sized balls: equal, the loss of the balls; the gradient matches finite differences" ):
    from otrec.loss import Loss1d
    from otrec.shapes import sized
    rec = Reconstruction( sino_of_shapes( TRUTH, 0.12 ), max_ratio = 3.0 ).add_random_balls( 6, 0.12 )
    fun = on_arrays( rec, Loss1d( rec.model, sized ) )
    x = rec._sized().cpu().numpy()
    assert np.isclose( fun( x )[ 0 ], rec.value_and_grad()[ 0 ], rtol = 1e-9 )
    an, fd = finite_difference( fun, x + 0.03 * np.random.default_rng( 0 ).normal( size = x.shape ), eps = 1e-6 )
    assert np.allclose( an, fd, rtol = 1e-4, atol = 1e-8 ), ( an, fd )

if test( "a ball twice as large weighs four times as much; the density of the balls has their masses" ):
    import torch
    from otrec.shapes import sized
    rec = Reconstruction( sino_of_shapes( TRUTH, 0.12 ), TRUTH[ :2 ], radius = 0.1 )
    rec.radii = np.array( [ 0.1, 0.2 ] )
    assert torch.allclose( sized.shapes( rec._sized(), rec.model )[ 2 ], rec.model.tensor( [ 0.2, 0.8 ] ) )
    img = sized.density( rec.model.tensor( rec.points ), rec.model.tensor( rec.radii ), 64, 2.0, 2, sub = 8 )
    assert np.isclose( img.sum(), 1.0, rtol = 2e-2 )
    assert np.isclose( img[ 32:, : ].sum() / img.sum(), 0.2, rtol = 2e-2 )                   # ( the small one, alone where x > 0 )

if test( "2D true discs ( k = 0.5 ): the loss of the truth, the gradients match finite differences" ):
    from otrec.loss import Loss1d
    from otrec.shapes import balls, sized
    rec = Reconstruction( sino_of_shapes( TRUTH, 0.12, k = 0.5 ), TRUTH, radius = 0.12, k = 0.5, max_ratio = 3.0 )
    f, floor = rec.value_and_grad()[ 0 ], rec.sinogram.nb_views * rec.sinogram.h ** 2 / 12
    assert f < 1.05 * floor, ( f, floor )
    rec.points = None                                                       # ( a new cloud: 6 random balls )
    rec.add_random_balls( 6, 0.12 )
    rng = np.random.default_rng( 0 )
    for kind, x in ( ( balls, rec.model.tensor( rec.points ) ), ( sized, rec._sized() ) ):
        fun = on_arrays( rec, Loss1d( rec.model, kind ) )
        x = x.cpu().numpy()
        an, fd = finite_difference( fun, x + 0.03 * rng.normal( size = x.shape ), eps = 1e-6 )
        assert np.allclose( an, fd, rtol = 1e-3, atol = 1e-7 ), ( kind, an, fd )

if test( "polygons from discs: a disc alone is a regular polygon, two that overlap meet on their common chord" ):
    import torch
    from otrec.shapes import polygons
    rec = Reconstruction( sino_of_shapes( TRUTH, 0.12 ), radius = 0.1 )
    md, T = rec.model, rec.model.tensor
    c = T( [ [ -0.5, 0.0 ], [ 0.1, 0.0 ], [ 0.2, 0.0 ] ] )
    verts = polygons.vertices( polygons.from_discs( c, T( [ 0.1, 0.1, 0.1 ] ), 0.1, 16 ), md )[ 1 ]
    assert torch.allclose( torch.linalg.vector_norm( verts[ 0 ] - c[ 0 ], dim = 1 ), T( 0.1 ) )
    assert np.isclose( polygons.areas( verts )[ 0 ].item(), 8 * 0.01 * np.sin( np.pi / 8 ) )                # ( the regular 16-gon )
    assert np.isclose( verts[ 1, :, 0 ].max().item(), 0.15, atol = 1e-9 ) and np.isclose( verts[ 2, :, 0 ].min().item(), 0.15, atol = 1e-9 )
    lens = 2 * 0.01 * np.arccos( 0.5 ) - 0.05 * np.sqrt( 0.03 )
    assert np.isclose( polygons.areas( verts )[ 1: ].sum().item(), 2 * np.pi * 0.01 - lens, rtol = 4e-2 )   # ( the union of the two discs )
    x = polygons.from_discs( c, T( [ 0.1, 0.1, 0.1 ] ), 0.1, 16, T( [ 0.01, 0.02, 0.03 ] ) )
    verts, w = polygons.shapes( x, md )
    assert torch.allclose( w, w[ :1 ].expand( 3 ) ) and np.isclose( ( w * polygons.areas( verts ) ).sum().item(), 1.0 )     # ( one density, a mass of 1 )
    assert torch.allclose( polygons.areas( verts ), T( [ 0.01, 0.02, 0.03 ] ) )                              # ( scaled to the areas asked for )
    cv = polygons.vertices( polygons.from_discs( c, T( [ 0.1, 0.1, 0.1 ] ), 0.1, 16 ), md )
    pos, tri, edges = polygons.outlines( *cv )
    assert pos.shape == ( 51, 2 ) and tri.shape == ( 48, 3 ) and edges.shape == ( 48, 2 ) and not polygons.turns( *cv )[ 1 ].any()

if test( "2D star polygons: at the discs they come from, about their loss; the density has the mass; the gradient matches finite differences" ):
    from otrec.loss import Loss1d
    from otrec.shapes import polygons
    rec = Reconstruction( sino_of_shapes( TRUTH, 0.12, k = 0.5 ), TRUTH, radius = 0.12, k = 0.5, max_ratio = 3.0 )
    md, f = rec.model, rec.value_and_grad()[ 0 ]
    x = polygons.from_discs( md.tensor( rec.points ), md.tensor( rec.radii ), 0.12, 24 )
    assert np.isclose( Loss1d( md, polygons ).value_and_grad( x, 0.0, 1e-3 )[ 0 ], f, rtol = 2e-2 )
    rec.points = None                                                       # ( a new cloud: 6 random balls )
    rec.add_random_balls( 6, 0.12 )
    x = polygons.from_discs( md.tensor( rec.points ), md.tensor( rec.radii ), 0.12, 6 ).cpu().numpy()
    x = x + 0.02 * np.random.default_rng( 0 ).normal( size = x.shape )
    X = md.tensor( x )
    w = polygons.shapes( X, md )[ 1 ]
    assert np.isclose( polygons.density( *polygons.vertices( X, md ), w, 64, 2.0, sub = 8 ).sum(), 1.0, rtol = 2e-2 ) and w.std() > 0
    an, fd = finite_difference( on_arrays( rec, Loss1d( md, polygons ) ), x, eps = 1e-6 )
    assert np.allclose( an, fd, rtol = 1e-4, atol = 1e-8 ), ( an, fd )

if test( "2D mesh from balls: the area of a disc full of balls, its contour on the border of the mesh, the gradient on the nodes" ):
    from otrec.loss import Loss1d
    import torch
    from otrec.shapes import mesh, polygons
    rec = Reconstruction( sino_of_shapes( TRUTH, 0.12 ), radius = 0.05 )
    md = rec.model
    g = 0.05 * np.stack( np.meshgrid( np.arange( -12, 13 ), np.arange( -12, 13 ), indexing = "ij" ), 2 ).reshape( -1, 2 )
    g = g[ np.linalg.norm( g, axis = 1 ) < 0.5 ]
    nodes, tri, loops = mesh.from_balls( md.tensor( g ), md.tensor( np.full( len( g ), 0.05 ) ), 2, 2.0, 0.025 )
    kind = mesh.Mesh( md.tensor( tri ).long() )
    verts, w = kind.shapes( md.tensor( nodes ), md )
    tris = md.tensor( nodes )[ kind.triangles ]                 # the border alone projects as the triangles do
    assert torch.allclose( kind.deposit( verts, w, md ), polygons.deposit( tris, w[ :1 ].expand( len( tris ) ), md ), atol = 1e-12 )
    assert len( loops ) == 1 and np.isclose( 1 / w[ 0 ].item(), np.pi * 0.25, rtol = 5e-2 ) and np.isclose( kind.density( md.tensor( nodes ), 64, 2.0 ).sum(), 1.0, rtol = 1e-2 )
    sides = { ( int( min( i, j ) ), int( max( i, j ) ) ) for i, j in mesh.border( tri ) }
    missing = np.mean( [ ( int( min( i, j ) ), int( max( i, j ) ) ) not in sides for i, j in zip( loops[ 0 ], np.roll( loops[ 0 ], -1 ) ) ] )
    assert missing < 0.02 and abs( len( sides ) / len( loops[ 0 ] ) - 1 ) < 0.1, ( missing, len( sides ), len( loops[ 0 ] ) )
    an, fd = finite_difference( on_arrays( rec, Loss1d( md, kind ) ), nodes, eps = 1e-6 )
    assert np.allclose( an, fd, rtol = 1e-2, atol = 1e-3 ), ( an, fd )         # ( loose: the loss has a kink where a side is along a view )

if test( "2D ellipses: round ones are the balls they come from; the gradient matches finite differences" ):
    from otrec.loss import Loss1d
    from otrec.shapes import ellipses
    rec = Reconstruction( sino_of_shapes( TRUTH, 0.12 ), TRUTH, radius = 0.12 )
    md, f = rec.model, rec.value_and_grad()[ 0 ]
    x = ellipses.from_discs( md.tensor( rec.points ), md.tensor( rec.radii ), md )
    assert np.isclose( Loss1d( md, ellipses ).value_and_grad( x, 0.0, 1e-3 )[ 0 ], f, rtol = 1e-9 )
    x = x.cpu().numpy() + [ 0, 0, 0.01, 0.03, -0.02 ] * np.random.default_rng( 0 ).normal( size = x.shape )
    an, fd = finite_difference( on_arrays( rec, Loss1d( md, ellipses ) ), x, eps = 1e-6 )
    assert np.allclose( an, fd, rtol = 1e-4, atol = 1e-8 ), ( an, fd )


if test( "fields: what each view asks of the balls adds up to the gradient" ):
    import torch
    from otrec.loss import fields
    from otrec.shapes import balls
    sino = Sinogram.from_image( ball_packing( nb = 128, seed = 0, gap = 0.025 ), 45, 64 )
    rec = Reconstruction( sino, device = "cpu" )
    rec.add_random_balls( 200, radius = 0.08 )
    md = rec.model
    x = md.tensor( rec.points )
    centres = balls.shapes( x, md )[ 0 ]
    radii, masses = torch.full_like( centres[ :, 0 ], md.radius ), torch.full_like( centres[ :, 0 ], 1 / len( centres ) )
    d = torch.as_tensor( np.random.default_rng( 0 ).normal( size = ( sino.nb_views, md.N ) ) )
    # with `d` as the derivative of a function of the cells, sum_v delta_v n_v gives back its gradient ( `gather` )
    ones = torch.ones( sino.nb_views, dtype = d.dtype )
    _, b, _, _, _ = fields.sums( md, centres, radii, masses, d, ones, torch.ones_like( d ), view_batch = 7 )
    assert torch.allclose( -2 * masses[ :, None ] * b, balls.gather( centres, md, d )[ 0 ], rtol = 1e-10, atol = 1e-12 )
    out = fields.per_ball( md, centres, radii, masses, d, ones, torch.ones_like( d ) )
    assert out[ "density asked" ].abs().max() < 1e-12                            # ( data = image: nothing asked )
    f = rec.fields()
    assert set( f ) == { "move", "dissonance", "density asked", "seen empty" } and all( v.shape == ( 200, ) for v in f.values() )


if test( "fields of ellipses: what each view asks adds up to the gradient of the centres" ):
    import torch
    from otrec.loss import Loss1d, fields
    from otrec.shapes import ellipses
    sino = Sinogram.from_image( ball_packing( nb = 128, seed = 0, gap = 0.025 ), 45, 64 )
    rec = Reconstruction( sino, device = "cpu" )
    rec.add_random_balls( 100, radius = 0.08 )
    md = rec.model
    x = ellipses.from_discs( md.tensor( rec.points ), md.tensor( rec.radii ), md )
    x[ :, 3: ] = md.tensor( np.random.default_rng( 1 ).normal( scale = 0.02, size = ( 100, 2 ) ) )   # stretched
    centres, C, masses = ellipses.shapes( x, md )
    d = torch.as_tensor( np.random.default_rng( 0 ).normal( size = ( sino.nb_views, md.N ) ) )
    widths = ellipses.window( md )[ :2 ]
    _, b, _, _, _ = fields.sums( md, centres, ellipses.discs( x, md )[ 1 ], masses, d, torch.ones( sino.nb_views, dtype = d.dtype ),
                              torch.ones_like( d ), covs = C, widths = widths )
    grad = ellipses.gather( centres, C, masses, md, d )[ 0 ]
    assert torch.allclose( -2 * masses[ :, None ] * b, grad, rtol = 1e-8, atol = 1e-12 )
    assert set( Loss1d( md, ellipses ).fields( x, 0.0, 1e-3 ) ) == { "move", "dissonance", "density asked", "seen empty" }
