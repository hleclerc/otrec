"""Reconstruction CT par DIRACS (`models.DiracModel`) : l'inconnue est un nuage de points dont
les projections doivent reproduire le sinogramme mesuré.

Tout passe par la classe `Reconstruction`, qui porte le sinogramme, le nuage courant et les
paramètres par défaut, et enchaîne les étapes (voir `test_chain_diracs_then_disks` pour la
composition diracs -> disques, et `test_disks.py` pour le modèle disques seul).
"""
import numpy as np

from otrec.Sinogram import Sinogram
from otrec.Reconstruction import Reconstruction
from otrec.optimizers import GradientDescent, LBFGS
from loom.testing import test


def _disk_sinogram( nb_angles = 8, nb_bins = 201, extent = 6.0, center = ( 0.3, -0.2 ), radius = 1.0 ):
    s = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    s.add_disk( center = list( center ), radius = radius )
    return s


if test( "in_disk_gives_small_loss" ):
    # des diracs échantillonnant le disque cible reproduisent ses projections : la perte doit être
    # petite. Exerce aussi le chemin OT sur des profils à densité NULLE (queues du détecteur).
    sino = _disk_sinogram()

    rng = np.random.default_rng( 0 )
    pts = []
    while len( pts ) < 400:
        xy = ( rng.random( 2 ) - 0.5 ) * 2
        if xy[ 0 ] ** 2 + xy[ 1 ] ** 2 < 1:
            pts.append( xy + np.array( [ 0.3, -0.2 ] ) )
    pts = np.array( pts )

    l = Reconstruction( sino, pts ).loss()
    assert np.isfinite( l )
    assert l < 0.05, f"perte trop grande pour des diracs dans le disque : { l }"


if test( "reconstruct_converges" ):
    # à partir de diracs aléatoires, la descente doit faire chuter la perte et amener les diracs
    # DANS le disque (position moyenne ~ centre).
    center, radius = np.array( [ 0.3, -0.2 ] ), 1.0
    sino = _disk_sinogram( nb_angles = 8, center = center, radius = radius )

    nb_steps = 100
    rec = Reconstruction( sino, extent = 5.0 ).random_points( 100, seed = 3 )
    l0 = rec.loss()

    # on capture ~10 instantanés le long de la descente, pour l'animation
    frames = [ rec.positions ]
    every = max( 1, nb_steps // 10 )
    def snap( step, pts ):
        if step % every == every - 1 or step == nb_steps - 1:
            frames.append( np.asarray( pts ) )

    rec.diracs( optimizer = GradientDescent( lr = 0.5, nb_steps = nb_steps ), callback = snap )
    l1 = rec.loss()

    assert l1 < l0 / 10, f"la perte n'a pas suffisamment chuté : { l0 } -> { l1 }"

    pn = rec.positions
    assert np.allclose( pn.mean( axis = 0 ), center, atol = 0.1 )
    dist = np.linalg.norm( pn - center, axis = 1 )
    assert ( dist <= radius + 0.1 ).mean() > 0.9, "la plupart des diracs devraient être dans le disque"

    # l'historique garde une ligne par étape jouée, avec les pertes de bout en bout
    ( h, ) = rec.history
    assert h[ "model" ] == "diracs" and h[ "nb_points" ] == 100
    assert abs( h[ "loss_before" ] - l0 ) < 1e-12 and abs( h[ "loss_after" ] - l1 ) < 1e-12


if test( "lbfgs_vs_gradient_descent" ):
    # Comparer la convergence de LBFGS et gradient descent sur le même problème.
    # LBFGS devrait converger plus vite (moins d'itérations) et atteindre une perte comparable ou meilleure.
    center, radius = np.array( [ 0.3, -0.2 ] ), 1.0
    sino = _disk_sinogram( nb_angles = 12, center = center, radius = radius )

    start = Reconstruction( sino, extent = 5.0 ).random_points( 80, seed = 42 ).points
    l0 = Reconstruction( sino, start ).loss()

    # même point de départ, deux optimiseurs : deux `Reconstruction` indépendantes
    gd_losses, lbfgs_losses = [], []
    rec_gd = Reconstruction( sino, start )
    rec_gd.diracs( optimizer = GradientDescent( lr = 0.2, nb_steps = 200 ),
                   callback = lambda step, pts: gd_losses.append( rec_gd.loss( points = pts ) ) )
    l_gd = gd_losses[ -1 ]

    rec_lbfgs = Reconstruction( sino, start )
    rec_lbfgs.diracs( optimizer = LBFGS( max_iter = 200, ftol = 1e-8 ),
                      callback = lambda step, pts: lbfgs_losses.append( rec_lbfgs.loss( points = pts ) ) )
    l_lbfgs = lbfgs_losses[ -1 ]

    print( f"\n  GD:    {len(gd_losses):3d} steps, loss {l0:.6f} -> {l_gd:.6f} (ratio {l_gd/l0:.4f})" )
    print( f"  LBFGS: {len(lbfgs_losses):3d} steps, loss {l0:.6f} -> {l_lbfgs:.6f} (ratio {l_lbfgs/l0:.4f})" )
    print( f"  Speedup: {len(gd_losses)/len(lbfgs_losses):.1f}x fewer iterations with LBFGS" )

    # LBFGS devrait converger en beaucoup moins d'itérations
    assert len( lbfgs_losses ) < len( gd_losses ) / 2, \
        f"LBFGS should converge faster: {len(lbfgs_losses)} vs {len(gd_losses)}"

    # LBFGS devrait atteindre une perte similaire ou meilleure
    assert l_lbfgs <= l_gd * 1.01, \
        f"LBFGS should reach similar or better loss: {l_lbfgs} vs {l_gd}"


if test( "lbfgs_quality" ):
    # LBFGS seul : vérifier qu'il atteint une bonne qualité de reconstruction. `max_iter`/`ftol`
    # passés à la construction définissent le L-BFGS par défaut de toutes les étapes.
    center, radius = np.array( [ 0.5, 0.1 ] ), 0.8
    sino = _disk_sinogram( nb_angles = 16, center = center, radius = radius )

    rec = Reconstruction( sino, extent = 4.0, max_iter = 150, ftol = 1e-9 ).random_points( 100, seed = 123 )
    l0 = rec.loss()
    l = rec.diracs().loss()

    print( f"\n  LBFGS quality: loss {l0:.6f} -> {l:.6f} (ratio {l/l0:.4f})" )

    # Vérifier convergence suffisante
    assert l < l0 / 20, f"LBFGS should reduce loss significantly: {l0} -> {l}"

    # Vérifier que les diracs sont bien positionnés
    pn = rec.positions
    assert np.allclose( pn.mean( axis = 0 ), center, atol = 0.15 ), "diracs should cluster around center"
    dist = np.linalg.norm( pn - center, axis = 1 )
    assert ( dist <= radius + 0.15 ).mean() > 0.85, "most diracs should be in the disk"


if test( "multiscale_refines" ):
    # grossier -> fin : chaque étage part de la structure trouvée par le précédent (`split`), et
    # l'historique doit montrer un nombre de points croissant, jusqu'à la cible.
    sino = _disk_sinogram( nb_angles = 12, center = ( 0.3, -0.2 ), radius = 1.0 )

    rec = Reconstruction( sino, extent = 5.0, max_iter = 40, ftol = 1e-10 )
    rec.multiscale( nb_points_final = 300, nb_points_init = 50, factor = 4 )

    assert rec.nb_points == 300, rec.nb_points
    counts = [ h[ "nb_points" ] for h in rec.history ]
    assert counts == [ 50, 200, 300 ], counts
    assert rec.history[ -1 ][ "loss_after" ] < rec.history[ 0 ][ "loss_before" ] / 10

    pn = rec.positions
    dist = np.linalg.norm( pn - np.array( [ 0.3, -0.2 ] ), axis = 1 )
    assert ( dist <= 1.15 ).mean() > 0.9, "la plupart des diracs devraient être dans le disque"


if test( "chain_diracs_then_disks" ):
    # LE cas d'usage du chaînage : une reconstruction en diracs (perte non lisse) sert de point de
    # départ à une reconstruction en DISQUES (perte lisse), sur le MÊME nuage de points -- chaque
    # dirac convergé devient un centre de disque.
    radius = 0.4
    truth = np.array( [ [ 0.8, 0.3 ], [ -0.9, 0.6 ], [ 0.1, -1.1 ], [ -0.5, -0.7 ], [ 1.3, -0.4 ] ] )
    sino = Sinogram( nb_angles = 24, nb_bins = 128, extent = 6.0 )
    for c in truth:
        sino.add_disk( center = list( c ), radius = radius )

    rec = Reconstruction( sino, radius = radius, nb_pixels = 256, extent = 3.0, record = True )
    rec.random_points( len( truth ), seed = 7 )

    # étape 1 : diracs. Les points migrent vers la masse, sans notion de rayon.
    rec.diracs( max_iter = 100, ftol = 1e-12 )
    assert rec.radii is None, "le modèle diracs n'a pas de rayon propre"
    l_diracs = rec.loss( rec.disk_model() )         # même nuage, mesuré avec la perte disques

    # étape 2 : disques, en repartant EXACTEMENT du nuage précédent.
    rec.disks( max_iter = 300, ftol = 1e-14 )
    l_disks = rec.history[ -1 ][ "loss_after" ]
    floor = rec.floor( rec.disk_model() )

    print( f"\n  perte disques { l_diracs:.8f} -> { l_disks:.8f} (plancher { floor:.8f})" )
    assert l_disks < l_diracs, f"l'étape disques doit améliorer la perte disques : { l_diracs } -> { l_disks }"
    assert l_disks < 1.05 * floor, f"perte finale { l_disks } au-dessus du plancher { floor }"

    # les centres retrouvés, à l'appariement près
    found = rec.positions
    err = [ float( np.min( np.linalg.norm( found - c, axis = 1 ) ) ) for c in truth ]
    assert max( err ) < 0.02, f"centres mal retrouvés, erreurs { err }"

    # la trajectoire est CONTINUE d'une étape à l'autre : une seule frame de jonction, et le rayon
    # exporté est celui de la dernière étape jouée.
    assert rec.radii == radius
    assert len( rec.frames ) == sum( h[ "nb_steps" ] for h in rec.history ) + 1
    assert [ h[ "model" ] for h in rec.history ] == [ "diracs", "disques" ]


if test( "disks_min_iter_forces_steps" ):
    # `models.DiskModel` a des directions PLATES (un disque peut glisser sans changer le résidu
    # tant qu'il ne chevauche personne) : reparti d'un nuage DÉJÀ convergé, L-BFGS-B (ftol scipy
    # natif) doit s'arrêter quasi tout de suite -- `min_iter` doit forcer un nombre de pas minimum
    # malgré ça (voir la docstring de `LBFGS`).
    radius = 0.4
    truth = np.array( [ [ 0.8, 0.3 ], [ -0.9, 0.6 ], [ 0.1, -1.1 ], [ -0.5, -0.7 ], [ 1.3, -0.4 ] ] )
    sino = Sinogram( nb_angles = 16, nb_bins = 96, extent = 6.0 )
    for c in truth:
        sino.add_disk( center = list( c ), radius = radius )

    rec = Reconstruction( sino, radius = radius, nb_pixels = 128, extent = 3.0 )
    rec.random_points( len( truth ), seed = 11 )
    rec.diracs( max_iter = 100, ftol = 1e-12 )
    rec.disks( max_iter = 100, ftol = 1e-8 )                         # converge une première fois

    # repartir EXACTEMENT du nuage convergé, avec/sans `min_iter`.
    nb_steps_plain = Reconstruction(
        sino, rec.points, radius = radius, nb_pixels = 128, extent = 3.0,
    ).disks( max_iter = 50, ftol = 1e-8 ).history[ -1 ][ "nb_steps" ]

    nb_steps_forced = Reconstruction(
        sino, rec.points, radius = radius, nb_pixels = 128, extent = 3.0,
    ).disks( max_iter = 50, ftol = 1e-8, min_iter = 10 ).history[ -1 ][ "nb_steps" ]

    print( f"\n  sans min_iter: { nb_steps_plain } pas -- avec min_iter=10: { nb_steps_forced } pas" )
    assert nb_steps_forced >= 10, f"min_iter=10 doit forcer au moins 10 pas : { nb_steps_forced }"
    assert nb_steps_forced > nb_steps_plain, \
        f"min_iter devrait forcer strictement plus de pas que le comportement scipy natif : " \
        f"{ nb_steps_forced } vs { nb_steps_plain }"


if test( "disks_disp_tol_stops_once_static" ):
    # `disp_tol` reprend la main APRÈS `min_iter` : un seuil de déplacement énorme doit arrêter
    # dès le premier pas suivant `min_iter`, quel que soit `ftol`/`max_iter`.
    radius = 0.4
    truth = np.array( [ [ 0.8, 0.3 ], [ -0.9, 0.6 ], [ 0.1, -1.1 ], [ -0.5, -0.7 ], [ 1.3, -0.4 ] ] )
    sino = Sinogram( nb_angles = 16, nb_bins = 96, extent = 6.0 )
    for c in truth:
        sino.add_disk( center = list( c ), radius = radius )

    rec = Reconstruction( sino, radius = radius, nb_pixels = 128, extent = 3.0 )
    rec.random_points( len( truth ), seed = 11 )
    rec.diracs( max_iter = 100, ftol = 1e-12 )
    rec.disks( max_iter = 50, ftol = 1e-8, min_iter = 5, disp_tol = 1e10 )

    nb_steps = rec.history[ -1 ][ "nb_steps" ]
    print( f"\n  min_iter=5, disp_tol énorme : { nb_steps } pas" )
    assert nb_steps == 6, f"doit s'arrêter au tout premier pas suivant min_iter : { nb_steps } pas"
