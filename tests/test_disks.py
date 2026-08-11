"""Reconstruction CT par DISQUES (`models.DiskModel` + `disks.DiskProjector`) : l'inconnue est la
position de disques 2D de rayon FIXE, comparés au sinogramme mesuré vu comme des diracs pondérés.

Ces cas sont INDÉPENDANTS de la reconstruction par diracs : aucun n'a besoin de la faire tourner
d'abord (voir `test_reconstruction.py` pour le modèle diracs et pour l'enchaînement des deux).
Chacun part d'un sinogramme synthétique construit par `Sinogram.add_disk`, dont on connaît donc
la vérité terrain.
"""
import os

import numpy as np

from reconstruction.Sinogram import Sinogram
from reconstruction.Reconstruction import Reconstruction
from reconstruction.disks import DiskProjector
from reconstruction.models import DiskModel, sinogram_diracs
from reconstruction.optimizers import LBFGS
from reconstruction.viz.points_html import export_positions_html
from sdot import driver
from sdot.testing import test


EXTENT = 6.0


def _sinogram( centers, radius, nb_angles = 16, nb_bins = 128, extent = EXTENT ):
    s = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    for c in centers:
        s.add_disk( center = list( c ), radius = radius )
    return s


def _match_error( found, truth ):
    """Erreur de position après appariement glouton (plus proche d'abord) de `found` sur `truth`."""
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
    """Chemin d'export des visualisations, sous `applications/reconstruction/benchmarks/results/`
    (déjà présent et ignoré par git). `SDOT_NO_HTML=1` désactive l'export."""
    if os.environ.get( "SDOT_NO_HTML" ):
        return None
    here = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
    out = os.path.join( here, "benchmarks", "results" )
    os.makedirs( out, exist_ok = True )
    return os.path.join( out, name )


if test( "disk_projection_matches_radon" ):
    # `DiskProjector.values` doit reproduire, sur la grille du détecteur, exactement ce que
    # `Sinogram.add_disk` accumule -- c'est la MÊME intégrale de corde, l'une en numpy non
    # différentiée, l'autre en algèbre `Tensor` différentiable.
    truth = np.array( [ [ 0.3, -0.2 ], [ -1.2, 0.7 ] ] )
    radius = 1.0
    sino = _sinogram( truth, radius, nb_angles = 8, nb_bins = 64 )

    proj = DiskProjector( sino, radius = radius )
    got = np.asarray( proj.values( truth ) )
    assert np.allclose( got, np.asarray( sino.values ), atol = 1e-12 ), \
        f"écart max { np.max( np.abs( got - np.asarray( sino.values ) ) ) }"

    # la masse par angle vaut exactement `nb_disques * pi * r^2` (intégration exacte par case)
    mass = got.sum( axis = 1 ) * proj.dw
    assert np.allclose( mass, len( truth ) * np.pi * radius ** 2, atol = 1e-10 ), mass

    # découper la somme sur les disques ne change pas le résultat
    chunked = DiskProjector( sino, radius = radius, max_chunk_elems = 1 )
    assert np.allclose( np.asarray( chunked.values( truth ) ), got, atol = 1e-12 )


if test( "disk_projection_gradient" ):
    # la dérivée de la projection par rapport aux centres est fournie À LA MAIN (surrogate du
    # premier ordre, voir `DiskProjector._values_of`) : on la confronte aux différences finies.
    truth = np.array( [ [ 0.3, -0.2 ], [ -1.0, 0.7 ] ] )
    radius = 1.0
    sino = _sinogram( truth, radius, nb_angles = 4, nb_bins = 64 )
    proj = DiskProjector( sino, radius = radius )

    def scalar( c ):
        return driver.sum( proj.values( c ).tensor ** 2 )

    g = np.asarray( driver.grad( scalar )( driver.array( truth ) ) )
    assert np.all( np.isfinite( g ) ), f"gradient non fini : { g }"

    eps = 1e-6
    fd = np.zeros_like( truth )
    for i in range( truth.shape[ 0 ] ):
        for j in range( 2 ):
            hp, hm = truth.copy(), truth.copy()
            hp[ i, j ] += eps
            hm[ i, j ] -= eps
            fd[ i, j ] = ( float( scalar( driver.array( hp ) ) ) - float( scalar( driver.array( hm ) ) ) ) / ( 2 * eps )

    rel = np.abs( g - fd ) / np.maximum( np.abs( fd ), 1.0 )
    assert rel.max() < 1e-3, f"gradient != differences finies :\n{ g }\n{ fd }"


if test( "disk_chunked_matches_unchunked" ):
    # le découpage en tranches (`driver.fold` + `driver.checkpoint`, ce qui borne le pic mémoire du
    # backward -- voir `DiskProjector.values`) doit être NEUTRE : mêmes valeurs, même gradient. Le
    # cas qui compte est celui où la taille de tranche ne divise PAS le nombre de disques : la
    # dernière est alors complétée par des centres de remplissage à poids nul, dont ni la masse ni
    # le gradient ne doivent transparaître.
    truth = np.array( [ [ 0.3, -0.2 ], [ -1.2, 0.7 ], [ 0.9, 0.4 ], [ -0.4, -0.8 ], [ 1.1, -1.0 ] ] )
    radius = 0.7
    sino = _sinogram( truth, radius, nb_angles = 4, nb_bins = 64 )

    whole = DiskProjector( sino, radius = radius )
    assert whole._chunk_size( len( truth ) ) == len( truth ), "ce cas doit tenir en une tranche"

    def scalar( proj, c ):
        return driver.sum( proj.values( c ).tensor ** 2 )

    ref_v = np.asarray( whole.values( truth ) )
    ref_g = np.asarray( driver.grad( lambda c: scalar( whole, c ) )( driver.array( truth ) ) )

    per_disk = whole.nb_angles * ( whole.nb_pixels + 1 )
    for chunk in ( 1, 2, 3, 4 ):                 # 4 = tranches inégales, 2 = dernière à moitié vide
        proj = DiskProjector( sino, radius = radius, max_chunk_elems = chunk * per_disk )
        assert proj._chunk_size( len( truth ) ) == chunk

        v = np.asarray( proj.values( truth ) )
        assert np.allclose( v, ref_v, atol = 1e-12 ), f"chunk={ chunk } : écart { np.max( np.abs( v - ref_v ) ) }"

        g = np.asarray( driver.grad( lambda c: scalar( proj, c ) )( driver.array( truth ) ) )
        assert np.allclose( g, ref_g, atol = 1e-10 ), f"chunk={ chunk } : gradient\n{ g }\n!=\n{ ref_g }"


if test( "disk_loss_floor_at_truth" ):
    # à la vérité terrain, la perte vaut EXACTEMENT le plancher de quantification :
    # les diracs condensent chaque case détecteur en son centre, ce qui coûte dw^2/12 par angle.
    truth = np.array( [ [ 0.3, -0.2 ], [ -1.2, 0.7 ] ] )
    radius = 1.0
    sino = _sinogram( truth, radius, nb_angles = 8, nb_bins = 64 )
    rec = Reconstruction( sino, truth, radius = radius )

    l = rec.loss()                      # `radius` fixé -> le modèle par défaut est celui des disques
    floor = rec.floor()
    assert abs( l - floor ) < 1e-10 * max( 1.0, floor ), f"{ l } != { floor }"

    # ... et elle CROÎT dès qu'on s'éloigne, de façon monotone
    prev = l
    for shift in ( 0.05, 0.1, 0.2, 0.4 ):
        cur = rec.loss( points = truth + shift )
        assert cur > prev, f"perte non croissante en { shift } : { cur } <= { prev }"
        prev = cur


if test( "disk_image_mass_shortcut" ):
    # `DiskProjector.image` fournit `current_mass` à la main (somme * largeur de case) au lieu de
    # laisser `Image._update_current_mass` la calculer par un kernel C++ -- c'est ce qui garde
    # toute la perte dans le graphe Jax. On vérifie ici que le raccourci donne bien la MÊME masse
    # que le kernel qu'il remplace, et la même perte.
    truth = np.array( [ [ 0.3, -0.2 ], [ -1.2, 0.7 ] ] )
    radius = 1.0
    sino = _sinogram( truth, radius, nb_angles = 6, nb_bins = 64 )
    model = DiskModel( sino, radius = radius )
    proj = model.projector

    img = proj.image( truth )
    shortcut = np.asarray( img.mass )

    # la même image, mais sans `current_mass` : `mass` passe alors par le kernel
    from sdot import Image, OtPlan1d
    plain = Image( values = proj.values( truth ), origin = [ proj.s_min ], frame = [ [ proj.dw ] ],
                   batch_axes = [ sino.num_angle ] )
    via_kernel = np.asarray( plain.mass )

    assert shortcut.shape == via_kernel.shape, f"{ shortcut.shape } != { via_kernel.shape }"
    assert np.allclose( shortcut, via_kernel, rtol = 1e-12 ), f"{ shortcut } != { via_kernel }"

    # et la perte est identique de bout en bout
    l_shortcut = float( model.cost( truth ) )
    l_plain = float( OtPlan1d( sinogram_diracs( sino ), plain ).cost.sum() )
    assert abs( l_shortcut - l_plain ) < 1e-12 * max( 1.0, abs( l_plain ) ), f"{ l_shortcut } != { l_plain }"


if test( "disk_loss_gradient" ):
    # gradient de la PERTE complète (donc à travers `OtPlan1d`) vs différences finies.
    truth = np.array( [ [ 0.4, -0.3 ], [ -0.9, 0.5 ] ] )
    radius = 0.8
    sino = _sinogram( truth, radius, nb_angles = 6, nb_bins = 96 )
    model = DiskModel( sino, radius = radius )

    start = truth + np.array( [ [ 0.25, -0.15 ], [ -0.2, 0.3 ] ] )

    def scalar( c ):
        return model.cost( model.wrap( c ) ).tensor

    g = np.asarray( driver.grad( scalar )( driver.array( start ) ) )
    assert np.all( np.isfinite( g ) ), f"gradient non fini : { g }"

    eps = 1e-5
    fd = np.zeros_like( start )
    for i in range( start.shape[ 0 ] ):
        for j in range( 2 ):
            hp, hm = start.copy(), start.copy()
            hp[ i, j ] += eps
            hm[ i, j ] -= eps
            fd[ i, j ] = ( float( scalar( driver.array( hp ) ) ) - float( scalar( driver.array( hm ) ) ) ) / ( 2 * eps )

    rel = np.abs( g - fd ) / np.maximum( np.abs( fd ), 1e-3 )
    assert rel.max() < 1e-3, f"gradient != differences finies :\n{ g }\n{ fd }\nrel { rel }"


if test( "disk_reconstruct_recovers_centers" ):
    # cas de référence : 5 disques de rayon connu, retrouvés à partir d'un tirage aléatoire.
    radius = 0.4
    truth = np.array( [ [ 0.8, 0.3 ], [ -0.9, 0.6 ], [ 0.1, -1.1 ], [ -0.5, -0.7 ], [ 1.3, -0.4 ] ] )
    sino = _sinogram( truth, radius, nb_angles = 24, nb_bins = 128 )

    # grille image 2x plus fine que le détecteur : le rayon (0.4) ne couvre que ~8 cases détecteur
    rec = Reconstruction( sino, radius = radius, nb_pixels = 256, extent = 3.0,
                          max_iter = 300, ftol = 1e-14, record = True )
    rec.random_points( len( truth ), seed = 7 )
    l0 = rec.loss()

    rec.disks()
    l1 = rec.loss()
    floor = rec.floor()

    print( f"\n  perte { l0:.6f} -> { l1:.8f} (plancher { floor:.8f}, { len( rec.frames ) } frames)" )
    assert l1 < l0 / 100, f"la perte n'a pas suffisamment chuté : { l0 } -> { l1 }"
    assert l1 < 1.05 * floor, f"perte finale { l1 } au-dessus du plancher { floor }"

    err = _match_error( rec.positions, truth )
    assert err.max() < 0.02, f"centres mal retrouvés, erreurs { err }"

    # -- visualisation : rayons FIXES exportés, donc utilisés tels quels au dessin ------
    if ( path := _html_path( "disks_recover.html" ) ):
        rec.export_html( path, extent = EXTENT, title = "disques : convergence vers les centres", fps = 8 )


if test( "disk_reconstruct_many" ):
    # plus dense : 40 disques de petit rayon tirés dans un anneau, reconstruits depuis un tirage
    # uniforme. On ne vise pas l'appariement exact (l'optimum n'est pas unique à cette densité),
    # seulement une perte proche du plancher et un support qui redevient l'anneau.
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
    print( f"\n  perte { l0:.6f} -> { l1:.8f} (plancher { floor:.8f}, { len( rec.frames ) } frames)" )

    assert l1 < l0 / 50, f"la perte n'a pas suffisamment chuté : { l0 } -> { l1 }"

    dist = np.linalg.norm( rec.positions, axis = 1 )
    assert ( ( dist > 0.9 ) & ( dist < 1.7 ) ).mean() > 0.9, \
        f"les disques devraient retomber sur l'anneau, rayons { np.round( np.sort( dist ), 2 ) }"

    if ( path := _html_path( "disks_ring.html" ) ):
        rec.export_html( path, extent = EXTENT, title = "disques : anneau", fps = 8 )


if test( "disk_finer_image_grid_lowers_floor" ):
    # la grille image est un choix LIBRE (indépendant du détecteur) : la raffiner ne change pas la
    # perte de façon sensible ici, mais doit rester stable -- c'est ce qui autorise à la choisir
    # assez fine pour représenter un petit rayon.
    radius = 0.3
    truth = np.array( [ [ 0.5, 0.2 ], [ -0.6, -0.4 ] ] )
    sino = _sinogram( truth, radius, nb_angles = 8, nb_bins = 64 )
    rec = Reconstruction( sino, truth, radius = radius )

    grids = ( 64, 128, 256, 512 )
    losses = [ rec.loss( rec.disk_model( nb_pixels = n ) ) for n in grids ]
    print( f"\n  perte à la vérité selon nb_pixels 64/128/256/512 : { [ f'{ l:.3e}' for l in losses ] }" )
    floor = rec.floor()
    for l, n in zip( losses, grids ):
        assert np.isfinite( l ) and abs( l - floor ) < 0.05 * floor, \
            f"nb_pixels={ n } : { l }, attendu ~{ floor }"


if test( "disk_html_export_radii" ):
    # l'export HTML doit embarquer le rayon et le mode "échelle" (rayons fixes, cf. `disks.py`).
    pts = np.array( [ [ 0.0, 0.0 ], [ 1.0, 0.5 ], [ -0.7, 0.3 ] ] )

    path = _html_path( "disks_export_check.html" )
    if path is None:
        path = os.path.join( "/tmp", "disks_export_check.html" )

    export_positions_html( pts, extent = 4.0, out_path = path, radii = 0.25 )
    html = open( path ).read()
    assert "const SCALED = true;" in html
    assert "const R0 = 0.25;" in html
    assert "const RADII = null;" in html, "rayon uniforme : rien à encoder par point"

    # rayons variables -> un bloc encodé par point
    export_positions_html( pts, extent = 4.0, out_path = path, radii = [ 0.1, 0.2, 0.3 ] )
    html = open( path ).read()
    assert "const SCALED = true;" in html
    assert "const RADII = decodeF32(" in html

    # sans `radii` : comportement historique intact (le slider EST le rayon)
    export_positions_html( pts, extent = 4.0, out_path = path )
    html = open( path ).read()
    assert "const SCALED = false;" in html
    assert "const RADII = null;" in html
    assert "const R0 = 1.0;" in html

    # ... et c'est exactement ce que `Reconstruction.export_html` choisit selon le dernier modèle
    sino = _sinogram( pts, 0.25, nb_angles = 4, nb_bins = 32 )
    Reconstruction( sino, pts, radius = 0.25 ).disks( max_iter = 1 ).export_html( path, extent = 4.0 )
    assert "const R0 = 0.25;" in open( path ).read()
    Reconstruction( sino, pts ).diracs( max_iter = 1 ).export_html( path, extent = 4.0 )
    assert "const SCALED = false;" in open( path ).read()
