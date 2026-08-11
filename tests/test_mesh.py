"""Le MAILLAGE gradué (`mesh.py`) : le pavage, l'opérateur, puis le partage de masse.

Les trois premiers tests portent sur des propriétés EXACTES (pavage, masse projetée, adjoint) et
sont serrés en conséquence ; le dernier juge ce à quoi sert vraiment l'étape -- retrouver la part
de masse qui est hors du champ de vue.
"""
import numpy as np

from otrec.mesh import GradedMesh, scan_exterior_scale
from otrec.Sinogram import Sinogram
from loom.testing import test


def _sino( nb_angles = 24, nb_bins = 512, extent = 10.0 ):
    return Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )


if test( "mesh_is_an_exact_tiling" ):
    # ni trou ni recouvrement : chaque case de la grille fine doit appartenir à exactement UNE
    # cellule. C'est la propriété dont tout le reste dépend -- un plan compté deux fois fausserait
    # le partage de masse, qui est le but de l'étape.
    mesh = GradedMesh( _sino(), outer_radius = 6.0, inner_radius = 2.0, cell_size = 0.125 )
    assert mesh.nb_levels > 1, "le maillage doit vraiment être gradué pour que le test ait un sens"

    h, r = mesh.cell_size, mesh.outer_radius
    k = ( np.arange( -int( 2 * r / h ), int( 2 * r / h ) ) + 0.5 ) * h
    x, y = np.meshgrid( k, k, indexing = "ij" )
    keep = np.hypot( x, y ) < r
    fx, fy = x[ keep ], y[ keep ]

    count = np.zeros( len( fx ), dtype = int )
    for ( cx, cy ), s in zip( mesh.centers, mesh.sizes ):
        count += ( np.abs( fx - cx ) < s / 2 ) & ( np.abs( fy - cy ) < s / 2 )
    assert count.min() == 1 and count.max() == 1, (
        f"couverture par case fine : { count.min() }..{ count.max() } (attendu exactement 1)" )
    assert np.isclose( mesh.areas.sum(), len( fx ) * h * h, rtol = 1e-12 )


if test( "mesh_projects_a_disk_exactly" ):
    # la projection d'un disque pavé de cellules doit avoir la MASSE de ce pavage, à tous les
    # angles -- c'est ce qui valide le trapèze analytique et le dépôt linéaire.
    sino = _sino()
    sino.add_disk( center = [ 0.0, 0.0 ], radius = 3.0 )
    mesh = GradedMesh( sino, outer_radius = 5.0, inner_radius = 3.0, cell_size = 0.1 )

    inside = np.linalg.norm( mesh.centers, axis = 1 ) < 3.0
    w = inside.astype( float )
    got = mesh.project( w )
    mass = got.sum( axis = 1 ) * mesh.coarse_dw
    assert np.allclose( mass, mesh.areas[ inside ].sum(), rtol = 1e-9 ), (
        f"masse projetée { mass.min() }..{ mass.max() } != aire { mesh.areas[ inside ].sum() }" )

    # ... et la forme doit suivre la projection analytique du disque, à l'escalier du pavage près
    exp = np.asarray( sino.values ).reshape( mesh.nb_angles, mesh.nb_coarse, mesh.group ).mean( axis = 2 )
    assert np.abs( got - exp ).max() / exp.max() < 0.10


if test( "mesh_adjoint_is_exact" ):
    # `backproject` doit être l'adjoint EXACT de `project` (trapèze symétrique + dépôt matriciel),
    # sans quoi le solveur ne converge pas vers le bon point.
    mesh = GradedMesh( _sino(), outer_radius = 6.0, inner_radius = 2.0, cell_size = 0.2 )
    rng = np.random.default_rng( 0 )
    a = rng.random( mesh.nb_cells )
    b = rng.random( ( mesh.nb_angles, mesh.nb_coarse ) )
    lhs, rhs = float( ( mesh.project( a ) * b ).sum() ), float( a @ mesh.backproject( b ) )
    assert abs( lhs - rhs ) <= 1e-9 * abs( rhs ), f"adjoint faux : { lhs } != { rhs }"


if test( "mesh_trapezoid_conserves_area" ):
    # le noyau de convolution est la projection d'un carré : sa masse vaut l'aire, à tout angle.
    mesh = GradedMesh( _sino( nb_angles = 16 ), outer_radius = 3.0, inner_radius = 2.0, cell_size = 0.5 )
    for h in ( 0.3, 0.5, 1.1 ):
        k = mesh._trapezoid( h )
        assert np.allclose( k.sum( axis = 1 ) * mesh.coarse_dw, h * h, rtol = 1e-9 ), (
            f"masse du trapèze != h² pour h={ h }" )


if test( "mesh_solve_splits_inside_from_outside" ):
    # ce à quoi sert l'étape : un objet plus large que le détecteur, dont on veut savoir quelle
    # part de masse est DANS le champ de vue -- la quantité que `halo.alternate` devait deviner.
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
    mesh.solve( smooth = 3e-2 )                    # L2 + gradient conjugué, le défaut

    got_in = mesh.interior_mass()
    got_out = mesh.mass() - got_in
    assert abs( got_in / true_inside - 1 ) < 0.10, (
        f"masse intérieure { got_in:.3f} pour { true_inside:.3f} attendue" )
    assert abs( got_out / true_outside - 1 ) < 0.25, (
        f"masse extérieure { got_out:.3f} pour { true_outside:.3f} attendue" )

    # et le sinogramme corrigé doit avoir une masse par angle bien plus constante
    raw = np.asarray( sino.mass() )
    cor = np.asarray( mesh.corrected().mass() )
    assert cor.std() / cor.mean() < raw.std() / raw.mean() / 4, (
        f"masse par angle pas assez égalisée : { raw.std() / raw.mean():.4f} -> "
        f"{ cor.std() / cor.mean():.4f}" )


if test( "scan_exterior_scale_is_coherent" ):
    # `alpha` est le degré de liberté qui reste ouvert après la résolution sur maillage : plus on
    # retire d'extérieur, moins il reste de masse à l'intérieur, et plus l'écrêtage à 0 mord. Le
    # balayage doit au moins respecter ça -- il ne PRÉTEND pas trouver le bon `alpha`, voir
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
        f"masse intérieure non décroissante en alpha : { scan[ 'interior_mass' ] }" )
    assert np.all( np.diff( scan[ "clipped" ] ) >= 0 ), (
        f"écrêtage non croissant en alpha : { scan[ 'clipped' ] }" )
    assert len( scan[ "clouds" ] ) == 3 and all( c.shape[ 1 ] == 2 for c in scan[ "clouds" ] )
