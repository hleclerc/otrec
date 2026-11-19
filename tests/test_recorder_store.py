import numpy as np

from errand import test

from otrec import Reconstruction, Recorder, Sinogram


def _run( recorder ):
    truth = np.array( [ [ 0.30, 0.10 ], [ -0.25, 0.35 ], [ -0.10, -0.40 ], [ 0.45, -0.30 ] ] )
    sino = Sinogram( 2, 12, 64, extent = 2.0 )
    sino.values = Reconstruction( sino, truth, radius = 0.12 ).image( sigma = 0.0, background = 0.0 )
    rec = Reconstruction( sino, recorder = recorder ).add_random_balls( 4, 0.12 )
    for sigma in ( 0.25, 0.06, 0.0 ):
        rec.adjust_parameters( centers = True, x_tol = 1e-3, sigma = sigma )
    rec.adjust_parameters( centers = True, cov_matrices = True, max_iter = 20 )         # ( ellipses )
    return rec


if test( "a recorder with a store: the frames of the page, written as they come, the curves too; nothing kept but what the curves need" ):
    import tempfile
    from sdot.viz.store import read_scene
    page = Recorder( every = 5 )
    _run( page )
    path = tempfile.mkdtemp() + "/run.sviz"
    told = [ ]
    edges = ( np.array( [ [ -1, -1 ], [ 1, -1 ], [ 1, 1 ] ], np.float32 ), np.array( [ [ 0, 1 ], [ 1, 2 ] ] ) )
    stored = Recorder( every = 5, store = path, on_frame = lambda: told.append( 1 ),
                       page = dict( title = "t", truth_edges = edges, images = { "fbp": ( np.random.rand( 8, 8 ), 2.0 ) } ) )
    _run( stored )
    stored.close()
    assert told and all( "points" not in fr for fr in stored.frames )
    s = read_scene( path )
    m = s.manifest
    assert m[ "complete" ] and m[ "title" ] == "t" and len( s ) == len( page.frames ) == len( stored.frames )
    assert [ s.value( i ) for i in range( len( s ) ) ] == [ fr[ "iteration" ] for fr in page.frames ]
    for i, fr in enumerate( page.frames ):
        a, _ = s.frame( i )
        assert np.allclose( a[ "pos" ][ a[ "pnt_v" ] ], fr[ "points" ], atol = 1e-4 ), i               # ( 16 bits )
        if "matrices" in fr[ "drawing" ]:
            assert np.allclose( a[ "pnt_m" ], fr[ "drawing" ][ "matrices" ].reshape( -1, 4 ), atol = 1e-3 ), i
    # the curves: those of the page ( `add_curve` drops what a log scale cannot show, on both sides )
    ref = page.write_html( tempfile.mkdtemp() + "/p.html" )
    assert [ ( c[ "name" ], len( c[ "y" ] ) ) for c in m[ "curves" ] ] == [ ( c[ "name" ], len( c[ "y" ] ) ) for c in ref._curves ]
    for c, r in zip( m[ "curves" ], ref._curves ):
        assert np.allclose( c[ "y" ], r[ "y" ] ) and np.allclose( c[ "x" ], r[ "x" ] ), c[ "name" ]
    base, bmeta = s.base()
    assert len( bmeta[ "images" ] ) == 1 and len( base[ "edge_v" ] ) == 2                       # ( the image, the truth: in the base )
    try:
        stored.write_html( tempfile.mkdtemp() + "/p.html" )
        assert False, "a store is not a page"
    except ValueError:
        pass
