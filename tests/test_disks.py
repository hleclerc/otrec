"""CT reconstruction with DISKS (`models.DiskModel` + `disks.DiskProjector`): the unknown is the
position of 2D disks of FIXED radius, compared with the measured sinogram seen as weighted diracs.

These cases are INDEPENDENT of the dirac reconstruction: none of them needs to run it first
(see `test_reconstruction.py` for the dirac model and for chaining the two). Each starts from a
synthetic sinogram built by `Sinogram.add_disk`, whose ground truth is therefore known.
"""
import os

import numpy as np

from otrec.Sinogram import Sinogram
from otrec.Reconstruction import Reconstruction
from otrec.disks import DiskProjector
from otrec.models import DiskModel, sinogram_diracs
from otrec.optimizers import LBFGS
from otrec.viz.points_html import export_positions_html
import loom
from errand import test
from loom.testing import need


EXTENT = 6.0


def _sinogram( centers, radius, nb_angles = 16, nb_bins = 128, extent = EXTENT ):
    s = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    for c in centers:
        s.add_disk( center = list( c ), radius = radius )
    return s


def _match_error( found, truth ):
    """Position error after greedy matching (closest first) of `found` onto `truth`."""
    found, truth = np.asarray( found ), np.asarray( truth )
    used, err = set(), []
    for c in truth:
        d = np.linalg.norm( found - c, axis = 1 )
        for i in np.argsort( d ):
            if int( i ) not in used:
                used.add( int( i ) )
                err.append( float( d[ i ] ) )
                break
    return np.array( err )


def _html_path( name ):
    """Export path of the visualizations, under `applications/reconstruction/benchmarks/results/`
    (already present and ignored by git). `SDOT_NO_HTML=1` disables the export."""
    if os.environ.get( "SDOT_NO_HTML" ):
        return None
    here = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
    out = os.path.join( here, "benchmarks", "results" )
    os.makedirs( out, exist_ok = True )
    return os.path.join( out, name )


if test( "disk_projection_matches_radon" ):
    # `DiskProjector.values` must reproduce, on the detector grid, exactly what
    # `Sinogram.add_disk` accumulates -- it is the SAME chord integral, one in non-differentiated
    # numpy, the other in differentiable `Tensor` algebra.
    truth = np.array( [ [ 0.3, -0.2 ], [ -1.2, 0.7 ] ] )
    radius = 1.0
    sino = _sinogram( truth, radius, nb_angles = 8, nb_bins = 64 )

    proj = DiskProjector( sino, radius = radius )
    got = np.asarray( proj.values( truth ) )
    assert np.allclose( got, np.asarray( sino.values ), atol = 1e-12 ), \
        f"max deviation { np.max( np.abs( got - np.asarray( sino.values ) ) ) }"

    # the mass per angle is exactly `nb_disks * pi * r^2` (exact integration per bin)
    mass = got.sum( axis = 1 ) * proj.dw
    assert np.allclose( mass, len( truth ) * np.pi * radius ** 2, atol = 1e-10 ), mass

    # splitting the sum over the disks does not change the result
    chunked = DiskProjector( sino, radius = radius, max_chunk_elems = 1 )
    assert np.allclose( np.asarray( chunked.values( truth ) ), got, atol = 1e-12 )


if test( "disk_projection_gradient" ):
    need( "grad" )
    # the derivative of the projection with respect to the centers is supplied BY HAND (first-order
    # surrogate, see `DiskProjector._values_of`): we check it against finite differences.
    truth = np.array( [ [ 0.3, -0.2 ], [ -1.0, 0.7 ] ] )
    radius = 1.0
    sino = _sinogram( truth, radius, nb_angles = 4, nb_bins = 64 )
    proj = DiskProjector( sino, radius = radius )

    def scalar( c ):
        return loom.ops().sum( proj.values( c ).value ** 2 )

    g = np.asarray( loom.grad( scalar )( loom.array( truth ) ) )
    assert np.all( np.isfinite( g ) ), f"gradient not finite: { g }"

    eps = 1e-6
    fd = np.zeros_like( truth )
    for i in range( truth.shape[ 0 ] ):
        for j in range( 2 ):
            hp, hm = truth.copy(), truth.copy()
            hp[ i, j ] += eps
            hm[ i, j ] -= eps
            fd[ i, j ] = ( float( scalar( loom.array( hp ) ) ) - float( scalar( loom.array( hm ) ) ) ) / ( 2 * eps )

    rel = np.abs( g - fd ) / np.maximum( np.abs( fd ), 1.0 )
    assert rel.max() < 1e-3, f"gradient != finite differences:\n{ g }\n{ fd }"


if test( "disk_chunked_matches_unchunked" ):
    need( "grad" )
    # splitting into chunks (`loom.fold` + `loom.checkpoint`, which bounds the backward memory
    # peak -- see `DiskProjector.values`) must be NEUTRAL: same values, same gradient. The case
    # that matters is the one where the chunk size does NOT divide the number of disks: the last
    # chunk is then padded with zero-weight filler centers, whose mass and gradient must not
    # show through.
    truth = np.array( [ [ 0.3, -0.2 ], [ -1.2, 0.7 ], [ 0.9, 0.4 ], [ -0.4, -0.8 ], [ 1.1, -1.0 ] ] )
    radius = 0.7
    sino = _sinogram( truth, radius, nb_angles = 4, nb_bins = 64 )

    whole = DiskProjector( sino, radius = radius )
    assert whole._chunk_size( len( truth ) ) == len( truth ), "this case must fit in one chunk"

    def scalar( proj, c ):
        return loom.ops().sum( proj.values( c ).value ** 2 )

    ref_v = np.asarray( whole.values( truth ) )
    ref_g = np.asarray( loom.grad( lambda c: scalar( whole, c ) )( loom.array( truth ) ) )

    per_disk = whole.nb_angles * ( whole.nb_pixels + 1 )
    for chunk in ( 1, 2, 3, 4 ):                 # 4 = uneven chunks, 2 = last one half empty
        proj = DiskProjector( sino, radius = radius, max_chunk_elems = chunk * per_disk )
        assert proj._chunk_size( len( truth ) ) == chunk

        v = np.asarray( proj.values( truth ) )
        assert np.allclose( v, ref_v, atol = 1e-12 ), f"chunk={ chunk } : deviation { np.max( np.abs( v - ref_v ) ) }"

        g = np.asarray( loom.grad( lambda c: scalar( proj, c ) )( loom.array( truth ) ) )
        assert np.allclose( g, ref_g, atol = 1e-10 ), f"chunk={ chunk } : gradient\n{ g }\n!=\n{ ref_g }"


if test( "disk_loss_floor_at_truth" ):
    # at the ground truth, the loss is EXACTLY the quantization floor:
    # the diracs condense each detector bin into its center, which costs dw^2/12 per angle.
    truth = np.array( [ [ 0.3, -0.2 ], [ -1.2, 0.7 ] ] )
    radius = 1.0
    sino = _sinogram( truth, radius, nb_angles = 8, nb_bins = 64 )
    rec = Reconstruction( sino, truth, radius = radius )

    l = rec.loss()                      # `radius` fixed -> the default model is the disk one
    floor = rec.floor()
    assert abs( l - floor ) < 1e-10 * max( 1.0, floor ), f"{ l } != { floor }"

    # ... and it GROWS as soon as we move away, monotonically
    prev = l
    for shift in ( 0.05, 0.1, 0.2, 0.4 ):
        cur = rec.loss( points = truth + shift )
        assert cur > prev, f"loss not increasing at { shift } : { cur } <= { prev }"
        prev = cur


if test( "disk_image_mass_shortcut" ):
    # `DiskProjector.image` supplies `current_mass` by hand (sum * bin width) instead of
    # letting `Image._update_current_mass` compute it with a C++ kernel -- this is what keeps
    # the whole loss in the Jax graph. Here we check that the shortcut gives the SAME mass
    # as the kernel it replaces, and the same loss.
    truth = np.array( [ [ 0.3, -0.2 ], [ -1.2, 0.7 ] ] )
    radius = 1.0
    sino = _sinogram( truth, radius, nb_angles = 6, nb_bins = 64 )
    model = DiskModel( sino, radius = radius )
    proj = model.projector

    img = proj.image( truth )
    shortcut = np.asarray( img.mass )

    # the same image, but without `current_mass`: `mass` then goes through the kernel
    from sdot import Image, OtPlan1d
    plain = Image( values = proj.values( truth ), origin = [ proj.s_min ], frame = [ [ proj.dw ] ],
                   batch_axes = [ sino.num_angle ] )
    via_kernel = np.asarray( plain.mass )

    assert shortcut.shape == via_kernel.shape, f"{ shortcut.shape } != { via_kernel.shape }"
    assert np.allclose( shortcut, via_kernel, rtol = 1e-12 ), f"{ shortcut } != { via_kernel }"

    # and the loss is identical end to end
    l_shortcut = float( model.cost( truth ) )
    l_plain = float( OtPlan1d( sinogram_diracs( sino ), plain ).cost.sum() )
    assert abs( l_shortcut - l_plain ) < 1e-12 * max( 1.0, abs( l_plain ) ), f"{ l_shortcut } != { l_plain }"


if test( "disk_loss_gradient" ):
    need( "grad" )
    # gradient of the full LOSS (hence through `OtPlan1d`) vs finite differences.
    truth = np.array( [ [ 0.4, -0.3 ], [ -0.9, 0.5 ] ] )
    radius = 0.8
    sino = _sinogram( truth, radius, nb_angles = 6, nb_bins = 96 )
    model = DiskModel( sino, radius = radius )

    start = truth + np.array( [ [ 0.25, -0.15 ], [ -0.2, 0.3 ] ] )

    def scalar( c ):
        return model.cost( model.wrap( c ) ).value

    g = np.asarray( loom.grad( scalar )( loom.array( start ) ) )
    assert np.all( np.isfinite( g ) ), f"gradient not finite: { g }"

    eps = 1e-5
    fd = np.zeros_like( start )
    for i in range( start.shape[ 0 ] ):
        for j in range( 2 ):
            hp, hm = start.copy(), start.copy()
            hp[ i, j ] += eps
            hm[ i, j ] -= eps
            fd[ i, j ] = ( float( scalar( loom.array( hp ) ) ) - float( scalar( loom.array( hm ) ) ) ) / ( 2 * eps )

    rel = np.abs( g - fd ) / np.maximum( np.abs( fd ), 1e-3 )
    assert rel.max() < 1e-3, f"gradient != finite differences:\n{ g }\n{ fd }\nrel { rel }"


if test( "disk_reconstruct_recovers_centers" ):
    need( "grad" )
    # reference case: 5 disks of known radius, recovered from a random draw.
    radius = 0.4
    truth = np.array( [ [ 0.8, 0.3 ], [ -0.9, 0.6 ], [ 0.1, -1.1 ], [ -0.5, -0.7 ], [ 1.3, -0.4 ] ] )
    sino = _sinogram( truth, radius, nb_angles = 24, nb_bins = 128 )

    # image grid 2x finer than the detector: the radius (0.4) only covers ~8 detector bins
    rec = Reconstruction( sino, radius = radius, nb_pixels = 256, extent = 3.0,
                          max_iter = 300, ftol = 1e-14, record = True )
    rec.random_points( len( truth ), seed = 7 )
    l0 = rec.loss()

    rec.disks()
    l1 = rec.loss()
    floor = rec.floor()

    print( f"\n  loss { l0:.6f} -> { l1:.8f} (floor { floor:.8f}, { len( rec.frames ) } frames)" )
    assert l1 < l0 / 100, f"the loss did not drop enough : { l0 } -> { l1 }"
    assert l1 < 1.05 * floor, f"final loss { l1 } above the floor { floor }"

    err = _match_error( rec.positions, truth )
    assert err.max() < 0.02, f"centers poorly recovered, errors { err }"

    # -- visualization: FIXED radii exported, hence used as is when drawing ------
    if ( path := _html_path( "disks_recover.html" ) ):
        rec.export_html( path, extent = EXTENT, title = "disks: convergence towards the centers", fps = 8 )


if test( "disk_reconstruct_many" ):
    need( "grad" )
    # denser: 40 small-radius disks drawn in a ring, reconstructed from a uniform draw.
    # We do not aim for exact matching (the optimum is not unique at this density), only a loss
    # close to the floor and a support that becomes the ring again.
    radius = 0.18
    rng = np.random.default_rng( 3 )
    theta = 2 * np.pi * rng.random( 40 )
    rad = 1.2 + 0.25 * rng.random( 40 )
    truth = np.stack( [ rad * np.cos( theta ), rad * np.sin( theta ) ], axis = 1 )

    sino = _sinogram( truth, radius, nb_angles = 32, nb_bins = 192 )

    rec = Reconstruction( sino, radius = radius, nb_pixels = 384, extent = 3.0,
                          max_iter = 400, ftol = 1e-14, record = True )
    rec.random_points( len( truth ), seed = 11 )
    l0 = rec.loss()

    l1 = rec.disks().loss()
    floor = rec.floor()
    print( f"\n  loss { l0:.6f} -> { l1:.8f} (floor { floor:.8f}, { len( rec.frames ) } frames)" )

    assert l1 < l0 / 50, f"the loss did not drop enough : { l0 } -> { l1 }"

    dist = np.linalg.norm( rec.positions, axis = 1 )
    assert ( ( dist > 0.9 ) & ( dist < 1.7 ) ).mean() > 0.9, \
        f"the disks should fall back onto the ring, radii { np.round( np.sort( dist ), 2 ) }"

    if ( path := _html_path( "disks_ring.html" ) ):
        rec.export_html( path, extent = EXTENT, title = "disks: ring", fps = 8 )


if test( "disk_finer_image_grid_lowers_floor" ):
    # the image grid is a FREE choice (independent of the detector): refining it does not change
    # the loss noticeably here, but must remain stable -- this is what allows choosing it fine
    # enough to represent a small radius.
    radius = 0.3
    truth = np.array( [ [ 0.5, 0.2 ], [ -0.6, -0.4 ] ] )
    sino = _sinogram( truth, radius, nb_angles = 8, nb_bins = 64 )
    rec = Reconstruction( sino, truth, radius = radius )

    grids = ( 64, 128, 256, 512 )
    losses = [ rec.loss( rec.disk_model( nb_pixels = n ) ) for n in grids ]
    print( f"\n  loss at the truth by nb_pixels 64/128/256/512 : { [ f'{ l:.3e}' for l in losses ] }" )
    floor = rec.floor()
    for l, n in zip( losses, grids ):
        assert np.isfinite( l ) and abs( l - floor ) < 0.05 * floor, \
            f"nb_pixels={ n } : { l }, expected ~{ floor }"


if test( "disk_html_export_radii" ):
    need( "grad" )
    # the HTML export must embed the radius and the "scaled" mode (fixed radii, cf. `disks.py`).
    pts = np.array( [ [ 0.0, 0.0 ], [ 1.0, 0.5 ], [ -0.7, 0.3 ] ] )

    path = _html_path( "disks_export_check.html" )
    if path is None:
        path = os.path.join( "/tmp", "disks_export_check.html" )

    export_positions_html( pts, extent = 4.0, out_path = path, radii = 0.25 )
    html = open( path ).read()
    assert "const SCALED = true;" in html
    assert "const R0 = 0.25;" in html
    assert "const RADII = null;" in html, "uniform radius: nothing to encode per point"

    # variable radii -> one encoded block per point
    export_positions_html( pts, extent = 4.0, out_path = path, radii = [ 0.1, 0.2, 0.3 ] )
    html = open( path ).read()
    assert "const SCALED = true;" in html
    assert "const RADII = decodeF32(" in html

    # without `radii`: historical behavior intact (the slider IS the radius)
    export_positions_html( pts, extent = 4.0, out_path = path )
    html = open( path ).read()
    assert "const SCALED = false;" in html
    assert "const RADII = null;" in html
    assert "const R0 = 1.0;" in html

    # ... and this is exactly what `Reconstruction.export_html` chooses depending on the last model
    sino = _sinogram( pts, 0.25, nb_angles = 4, nb_bins = 32 )
    Reconstruction( sino, pts, radius = 0.25 ).disks( max_iter = 1 ).export_html( path, extent = 4.0 )
    assert "const R0 = 0.25;" in open( path ).read()
    Reconstruction( sino, pts ).diracs( max_iter = 1 ).export_html( path, extent = 4.0 )
    assert "const SCALED = false;" in open( path ).read()
