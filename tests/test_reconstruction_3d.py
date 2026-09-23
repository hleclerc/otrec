"""Reconstruction 3D par DIRACS ( `models.ProjectedDiracModel` ) : l'inconnue est un nuage de
points 3D dont les projections sur le détecteur, à chaque angle, doivent reproduire les
RADIOGRAPHIES mesurées ( `Radiographs` ) -- le pendant de `test_reconstruction.py` avec, à chaque
angle, un transport semi-discret 2D ( `OtPlan` ) à la place du transport 1D.

Les radiographies sont celles de quelques boules ( `Radiographs.add_sphere` ), dont on connaît
donc la vérité terrain.
"""
import numpy as np

from otrec.Radiographs import Radiographs
from otrec.Reconstruction import Reconstruction
from sdot import set_kernel_dtype
from loom.testing import Param, experiment, test

set_kernel_dtype( "FP64" )

CENTERS = np.array( [ [ 0.5, -0.3, 0.2 ], [ -0.6, 0.4, -0.5 ], [ 0.1, 0.5, 0.6 ] ] )
RADIUS = 0.4


def _sphere_radiographs( nb_angles = 3, nb_pixels = 40, extent = 4.0 ):
    r = Radiographs( nb_angles = nb_angles, nb_u = nb_pixels, nb_v = nb_pixels, extent_u = extent )
    for c in CENTERS:
        r.add_sphere( c, RADIUS )
    return r


def _in_spheres( points, margin = 0.1 ):
    """la fraction des points à moins de `RADIUS + margin` d'un centre"""
    d = np.linalg.norm( points[ :, None, : ] - CENTERS[ None ], axis = 2 ).min( axis = 1 )
    return float( ( d < RADIUS + margin ).mean() )


if test( "points_in_the_spheres_give_a_small_loss" ):
    # des diracs échantillonnant les boules reproduisent leurs radiographies : le coût doit être
    # petit devant celui d'un nuage uniforme
    radio = _sphere_radiographs()
    rng = np.random.default_rng( 0 )
    pts = []
    while len( pts ) < 45:
        c = CENTERS[ len( pts ) % 3 ]
        q = rng.uniform( -RADIUS, RADIUS, 3 )
        if q @ q < RADIUS ** 2:
            pts.append( c + q )
    pts = np.array( pts )

    inside = Reconstruction( radio, pts ).loss()
    uniform = Reconstruction( radio ).random_points( 45, seed = 1 ).loss()
    assert np.isfinite( inside ) and inside < uniform / 5, ( inside, uniform )


if test( "multiscale_refines_up_to_the_requested_count_in_3d" ):
    # par étages : 40 -> 160 -> 320 diracs ( le dernier sous-échantillonné ), chaque étage
    # repartant du nuage convergé du précédent ( `Reconstruction.multiscale`, l'enveloppe visuelle
    # comme point de départ ) ; le modèle 3D suit le changement de taille ( ses poids chauds sont
    # abandonnés à chaque étage ) et la perte finale est petite
    radio = _sphere_radiographs()
    rec = Reconstruction( radio )
    stages = []
    rec.multiscale( 320, nb_points_init = 40, factor = 4, max_iter = 8,
                    stage_callback = lambda stage, n, pts: stages.append( n ) )
    assert stages == [ 40, 160, 320 ], stages
    assert rec.nb_points == 320
    assert [ h[ "nb_points" ] for h in rec.history ] == stages
    assert rec.history[ -1 ][ "loss_after" ] < rec.history[ 0 ][ "loss_before" ] / 3
    assert _in_spheres( rec.positions ) > 0.85, _in_spheres( rec.positions )


if test( "blur_annealing_reconstructs_from_the_cube" ):
    # depuis un nuage tiré dans tout le CUBE, les projections floutées d'abord
    # ( `Reconstruction.anneal_blur` ) puis resserrées amènent les diracs dans les boules. Le flou
    # ne sert plus à rendre le TRANSPORT possible -- la continuation en largeur d'`OtPlan` s'en
    # charge, pourvu que la cible soit > 0 partout -- mais à adoucir le paysage que L-BFGS descend.
    # C'est l'enchaînement des étages que ce test vérifie, pas leur nécessité ( mesurée faible :
    # `notes/2026-09-23-otrec-3d.md` )
    radio = _sphere_radiographs()
    rec = Reconstruction( radio ).random_points( 60, seed = 1 )
    assert _in_spheres( rec.positions ) < 0.5
    sigmas = []
    rec.anneal_blur( blurs = ( 1.0, 0.25, 0.06, 0.0 ), max_iter = 10,
                     stage_callback = lambda stage, sigma, pts: sigmas.append( sigma ) )
    assert sigmas == [ 1.0, 0.25, 0.06, 0.0 ]
    assert [ h[ "label" ] for h in rec.history ] == [ "diracs 3D flou 1", "diracs 3D flou 0.25", "diracs 3D flou 0.06", "diracs 3D flou 0" ]
    assert _in_spheres( rec.positions ) > 0.8, _in_spheres( rec.positions )
    assert rec.sinogram is radio                      # la donnée nette est rendue


if test( "reconstruct_converges_in_3d" ):
    # à partir d'un nuage uniforme, la descente ( L-BFGS sur le coût + gradient fusionnés, voir
    # `ProjectedDiracModel.value_and_grad` ) doit faire chuter le coût et amener les diracs DANS les
    # boules
    radio = _sphere_radiographs()
    rec = Reconstruction( radio ).random_points( 45, seed = 1 )
    l0 = rec.loss()
    assert _in_spheres( rec.positions ) < 0.5

    rec.diracs( max_iter = 20 )
    l1 = rec.loss()

    assert l1 < l0 / 3, ( l0, l1 )
    assert _in_spheres( rec.positions ) > 0.8, _in_spheres( rec.positions )

    ( h, ) = rec.history
    assert h[ "model" ] == "diracs 3D" and h[ "nb_points" ] == 45


# -- ce qu'on REGARDE ------------------------------------------------------------------------
#
#   ./run experiment test_reconstruction_3d                                  # toutes
#   ./run experiment "test_reconstruction_3d::rec 3D spheres" --nb-points=20000 --nb-angles=6
#   ./run experiment "test_reconstruction_3d::rec 3D random spheres" --nb-spheres=12
#
# La descente en 3D, un pas par image : la page HTML du `Visualizer` de sdot ( les boules de la
# vérité terrain en transparence, les diracs qui s'y rangent, le curseur pour rejouer ), et la
# MÊME scène pour ParaView -- un `.pvd` qui rassemble un `.vtu` par pas, à ouvrir tel quel, le
# rayon de chaque point en donnée de cellule ( `Glyph`, sphères, `Scale Array = radius` ). Plus la
# courbe de convergence. Sur cette machine ( 16 coeurs chargés ), `OMP_NUM_THREADS=4` divise par
# dix le coût d'un petit kernel -- voir `notes/2026-09-14-reconstruction-3d.md`.
#
# CE QUE LE SOLVEUR FAIT MAINTENANT ( `OtPlan` entièrement en C++, voir
# `notes/2026-09-22-otplan-cpp.md` ), et que ces expériences affichent à chaque image
# ( `ProjectedDiracModel.solver_line()` ) :
#   - le pas de Newton d'une projection ( 2D ) vient des LIMITES, donc sans reculs ;
#   - un départ qui vide des cellules n'est plus un échec : le C++ choisit entre les poids chauds,
#     le Voronoï et la similitude, et la CONTINUATION EN LARGEUR ( la densité convolée, resserrée
#     étape par étape ) reprend les cas où le Newton direct stagnait ;
#   - ce qu'elle ne remplace PAS : le FOND ( `--background` ). Elle adoucit le chemin, pas la
#     cible -- sa dernière étape est la donnée elle-même, zéros compris, et une cellule qui ne voit
#     que des zéros n'a aucun poids qui lui donne sa masse. Mesuré depuis le cube ( 200 diracs,
#     6 angles, 64 x 64, `notes/2026-09-23-otrec-3d.md` ) : sans fond, 36 ajustements sur 48 ne
#     convergent pas et la perte MONTE ; avec 1e-6, tous convergent en 22 diagrammes et 98 % des
#     diracs finissent dans les boules ;
#   - les poids chauds d'une évaluation à l'autre gardent le nombre de pas à quelques unités dès
#     que le nuage ne bouge plus beaucoup -- ce que la ligne du solveur montre en clair ( des
#     étapes de continuation et des départs au Voronoï = le nuage est encore loin ).
# Le prix en reste là : un ajustement par angle et par évaluation, une évaluation par pas de
# L-BFGS -- compter des minutes par pas à 20 000 diracs et 6 angles, les sorties étant réécrites
# tous les dix pas.

def _random_spheres( nb_spheres, seed, extent = 2.2 ):
    """`nb_spheres` boules de rayons variés, DISJOINTES, dans le cube `[ -extent/2, extent/2 ]^3`"""
    rng = np.random.default_rng( seed )
    centers, radii = [], []
    while len( centers ) < nb_spheres:
        r = rng.uniform( 0.15, 0.45 )
        c = rng.uniform( -extent / 2 + r, extent / 2 - r, 3 )
        if all( np.linalg.norm( c - c2 ) > r + r2 + 0.05 for c2, r2 in zip( centers, radii ) ):
            centers.append( c ); radii.append( r )
    return np.array( centers ), np.array( radii )


def _run_3d( p, centers, radii, stem, multiscale = None, blurs = None ):
    """Commun aux expériences : les radiographies des boules, la descente enregistrée, les sorties.

    `multiscale = ( nb_points_init, factor )` : la descente par étages ( `Reconstruction.multiscale` )
    au lieu d'un nuage tiré d'un coup à `nb_points`. `blurs` : d'abord les projections FLOUTÉES,
    resserrées étage par étage ( `Reconstruction.anneal_blur` ), sur le nuage de départ -- puis le
    raffinement par étages s'il est demandé, sur la donnée nette."""
    import time
    from sdot import Visualizer, write_convergence_html

    radio = Radiographs( nb_angles = p.nb_angles, nb_u = p.nb_pixels, nb_v = p.nb_pixels, extent_u = 4.0 )
    for c, r in zip( centers, radii ):
        radio.add_sphere( c, r )

    rec = Reconstruction( radio, verbose = True, seed = p.seed )
    n0 = multiscale[ 0 ] if multiscale else p.nb_points
    if p.init == "hull":
        rec.hull_points( n0, seed = p.seed )
    else:
        rec.random_points( n0, seed = p.seed )

    def in_spheres( pts, margin = 0.05 ):
        d = np.linalg.norm( pts[ :, None, : ] - centers[ None ], axis = 2 ) - radii[ None, : ]
        return float( ( d.min( axis = 1 ) < margin ).mean() )

    viz = Visualizer( title = f"reconstruction 3D -- { p.nb_points } diracs, { p.nb_angles } angles" )
    fractions, times, counts, t0 = [], [], [], time.perf_counter()
    model_kwargs = dict( background = p.background, kernel_dtype = p.kernel, continuation = p.continuation )
    model = rec.dirac_model( **model_kwargs )

    def write_outputs():
        viz.write_html( p.out_dir / f"{ stem }.html" )
        viz.write_vtk( p.out_dir / f"{ stem }.vtk" )
        write_convergence_html(
            { "diracs dans les boules ( fraction )": list( zip( times, fractions ) ),
              "nombre de diracs / nombre final": list( zip( times, [ c / p.nb_points for c in counts ] ) ) },
            p.out_dir / f"{ stem }_convergence.html",
            title = f"reconstruction 3D -- { p.nb_points } diracs", xlabel = "temps ( s )", ylabel = "fraction", log_y = False )

    def snap( step, pts ):
        pts = np.asarray( pts )
        if step >= 0 and ( step + 1 ) % p.record_every:
            return
        if fractions:                                     # la toute première image existe déjà
            viz.new_frame( len( fractions ) )
        viz.add_points( pts, radius = 0.01, color = "#e0a030" )
        for c, r in zip( centers, radii ):
            viz.add_points( c[ None ], radius = r, color = "#4080c0", opacity = 0.2 )
        fractions.append( in_spheres( pts ) )
        counts.append( len( pts ) )
        times.append( time.perf_counter() - t0 )
        print( f"  image { len( fractions ) - 1 } ( pas { step + 1 }, { len( pts ) } diracs ), { times[ -1 ]:.0f} s, "
               f"{ fractions[ -1 ] * 100:.1f} % dans les boules", flush = True )
        # ce que les ajustements par angle ont coûté depuis le début ( le modèle en cours, que
        # `anneal_blur` / `multiscale` construisent eux-mêmes ) -- voir l'en-tête de ce fichier
        line = getattr( rec.model, "solver_line", None )
        if line is not None:
            print( f"    { line() }", flush = True )
        # une descente longue s'écrit EN COURS DE ROUTE : ce qui est fait est déjà regardable
        if len( fractions ) % 10 == 0:
            write_outputs()

    if blurs:
        rec.anneal_blur( blurs = blurs, max_iter = p.max_iter, callback = snap,
                         model_kwargs = model_kwargs )
    if multiscale:
        # par étages : chaque étage s'arrête quand scipy ne progresse plus ( `ftol` ) ou à
        # `max_iter` -- la « quasi-convergence » qui suffit avant de raffiner. Après le flou, le
        # nuage de départ est DÉJÀ convergé sur la donnée nette : on raffine tout de suite.
        if blurs and blurs[ -1 ] == 0 and rec.nb_points < p.nb_points:
            rec.split( multiscale[ 1 ] ).subsample( min( rec.nb_points, p.nb_points ) )
        rec.multiscale( p.nb_points, nb_points_init = multiscale[ 0 ], factor = multiscale[ 1 ],
                        model = model, max_iter = p.max_iter, callback = snap )
    elif not blurs:
        # `min_iter = max_iter` : tous les pas demandés, pas un arrêt de scipy sur un `ftol` que le
        # bruit des ajustements internes ( `mass_tol` ) déclenche trop tôt -- c'est une expérience
        rec.run( model, max_iter = p.max_iter, min_iter = p.max_iter, callback = snap, label = "diracs 3D" )

    stages = rec.history
    total = sum( h[ "time" ] for h in stages )
    print( f"  { rec.nb_points } diracs, { p.nb_angles } angles, { len( stages ) } étage(s), "
           f"{ sum( h[ 'nb_steps' ] for h in stages ) } pas en { total:.0f} s : "
           f"perte { stages[ 0 ][ 'loss_before' ]:.3e} -> { stages[ -1 ][ 'loss_after' ]:.3e}, "
           f"{ in_spheres( rec.positions ) * 100:.1f} % des diracs dans les boules" )
    for h in stages:
        print( f"    { h[ 'label' ] } : { h[ 'nb_steps' ] } pas, { h[ 'time' ]:.0f} s, perte { h[ 'loss_before' ]:.3e} -> { h[ 'loss_after' ]:.3e}" )
    p.results[ "loss_before" ], p.results[ "loss_after" ] = stages[ 0 ][ "loss_before" ], stages[ -1 ][ "loss_after" ]
    p.results[ "in_spheres" ] = in_spheres( rec.positions )
    p.results[ "time" ] = total

    write_outputs()
    np.savez( p.out_dir / f"{ stem }_final.npz", positions = rec.positions, centers = centers, radii = radii )
    return rec


_PARAMS = dict(
    nb_points    = Param( 10000, help = "nombre de diracs" ),
    nb_angles    = Param( 100, help = "nombre d'angles de projection" ),
    nb_pixels    = Param( 128, help = "pixels par côté du détecteur" ),
    max_iter     = Param( 30, help = "nombre de pas de L-BFGS" ),
    record_every = Param( 1, help = "une image toutes les k pas" ),
    background   = Param( 1e-6, help = "fond ajouté aux radiographies, en fraction de la moyenne -- "
                                       "INDISPENSABLE ( une cellule qui ne voit que des zéros n'a pas "
                                       "de poids qui lui donne sa masse ) ; `0` pour le constater" ),
    continuation = Param( "auto", help = "la continuation en largeur d'`OtPlan` : auto, always, never" ),
    init         = Param( "hull", help = "point de départ : `hull` ( l'enveloppe visuelle ) ou `cube`" ),
    kernel       = Param( "FP64", help = "le flottant du noyau ( FP64 : ce que l'amortissement demande ; "
                                         "FP32 pour voir ce qu'il en coûte )" ),
    seed         = Param( 1, help = "graine du tirage" ),
)

if p := experiment( "rec 3D spheres", **_PARAMS ):
    # les trois boules des tests, à grande échelle
    _run_3d( p, CENTERS, np.full( len( CENTERS ), RADIUS ), "rec_3d_spheres" )

if p := experiment( "rec 3D random spheres", nb_spheres = Param( 8, help = "nombre de boules" ), **_PARAMS ):
    # un fantôme moins symétrique : des boules de rayons variés, tirées au hasard
    centers, radii = _random_spheres( p.nb_spheres, p.seed + 100 )
    _run_3d( p, centers, radii, "rec_3d_random_spheres" )

if p := experiment( "rec 3D blur",
                    blurs          = Param( "1,0.25,0.06,0.015,0", help = "les flous, en fraction de la largeur du détecteur "
                                                                          "( `0` seul : la donnée nette, pour comparer )" ),
                    nb_spheres     = Param( 8, help = "nombre de boules" ),
                    **{ **_PARAMS, "init": Param( "cube", help = "point de départ : `cube` ( tout le cube ) ou `hull`" ),
                        "max_iter": Param( 20, help = "pas de L-BFGS au plus PAR ÉTAGE" ) } ):
    # les projections FLOUTÉES d'abord ( `Reconstruction.anneal_blur` ), depuis un nuage tiré dans
    # tout le cube. Ce qui a changé : le flou n'est plus ce qui rend le TRANSPORT possible -- la
    # continuation en largeur d'`OtPlan` s'en charge, à condition d'un fond > 0. Il ne reste au flou
    # que son autre rôle, adoucir le paysage que L-BFGS descend, et c'est peu : à 200 diracs depuis
    # le cube, la donnée nette ( fond 1e-6 ) finit à 98 % des diracs dans les boules contre 99 %
    # après quatre étages de flou, pour un temps comparable ( `notes/2026-09-23-otrec-3d.md` ).
    # `--blurs=0` fait tourner la même expérience sans flou : c'est la comparaison à refaire sur un
    # fantôme plus dur avant de se passer de ces étages.
    centers, radii = _random_spheres( p.nb_spheres, p.seed + 100 )
    _run_3d( p, centers, radii, "rec_3d_blur", blurs = [ float( b ) for b in str( p.blurs ).split( "," ) ] )

if p := experiment( "rec 3D blur multiscale",
                    blurs          = Param( "1,0.25,0.06,0.015,0", help = "les flous, en fraction de la largeur du détecteur" ),
                    nb_points_init = Param( 500, help = "diracs des étages floutés et du premier raffinement" ),
                    factor         = Param( 4, help = "enfants par dirac à chaque raffinement" ),
                    nb_spheres     = Param( 8, help = "nombre de boules" ),
                    **{ **_PARAMS, "init": Param( "cube", help = "point de départ : `cube` ou `hull`" ),
                        "nb_points": Param( 32000, help = "nombre de diracs FINAL" ),
                        "max_iter": Param( 20, help = "pas de L-BFGS au plus PAR ÉTAGE" ) } ):
    # les deux : le flou resserré sur un petit nuage, puis le raffinement par étages sur la donnée nette
    centers, radii = _random_spheres( p.nb_spheres, p.seed + 100 )
    _run_3d( p, centers, radii, "rec_3d_blur_multiscale", multiscale = ( p.nb_points_init, p.factor ),
             blurs = [ float( b ) for b in str( p.blurs ).split( "," ) ] )

if p := experiment( "rec 3D multiscale",
                    nb_points_init = Param( 500, help = "diracs du premier étage" ),
                    factor         = Param( 4, help = "enfants par dirac à chaque raffinement" ),
                    nb_spheres     = Param( 8, help = "nombre de boules" ),
                    **{ **_PARAMS, "nb_points": Param( 32000, help = "nombre de diracs FINAL" ),
                        "max_iter": Param( 30, help = "pas de L-BFGS au plus PAR ÉTAGE" ) } ):
    # PAR ÉTAGES ( `Reconstruction.multiscale` ) : peu de diracs d'abord, convergés, puis chacun
    # remplacé par `factor` enfants bruités, reconvergés -- jusqu'au nombre demandé. Chaque étage
    # part d'un nuage déjà bien placé, donc ses transports par angle démarrent près de leur
    # solution : c'est ce qui rend un gros nuage abordable, là où le tirer d'un coup fait repartir
    # chaque ajustement du Voronoï ( voir la note ). Une image par pas, tous étages confondus --
    # le nombre de diracs change d'un étage à l'autre, la courbe le montre.
    centers, radii = _random_spheres( p.nb_spheres, p.seed + 100 )
    _run_3d( p, centers, radii, "rec_3d_multiscale", multiscale = ( p.nb_points_init, p.factor ) )
