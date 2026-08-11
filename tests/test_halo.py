"""Le HALO (`halo.py`) : l'empreinte sur le sinogramme de la matière hors du champ de vue.

Trois niveaux, du plus local au plus global : l'OPÉRATEUR de projection d'une cellule (comparé à
la projection analytique d'un anneau), l'AJUSTEMENT positif qui retrouve une masse extérieure
connue, et enfin l'ALTERNANCE complète, jugée sur ce qui motive tout le module -- les vides
préservés dans la reconstruction.
"""
import numpy as np

from otrec.halo import ( Halo, alternate, interior_values, mass_profile,
                                  scan_interior_mass, void_fraction )
from otrec.Sinogram import Sinogram
from loom.testing import test


def _annulus_sinogram( a, b, nb_angles = 5, nb_bins = 256, extent = 10.0 ):
    """Projection EXACTE d'un anneau [ a, b ] centré, par différence de deux disques."""
    s = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    s.add_disk( center = [ 0.0, 0.0 ], radius = b )
    s.add_disk( center = [ 0.0, 0.0 ], radius = a, density = -1.0 )
    return s


if test( "halo_operator_matches_annulus" ):
    # un halo à UN anneau et UN secteur EST un anneau : son opérateur doit reproduire la projection
    # analytique (exacte, via `add_disk`). Vérifie du même coup l'intégration radiale exacte et le
    # dépôt des rampes par la primitive.
    a, b, extent, nb_bins = 1.0, 2.0, 10.0, 256
    ref = _annulus_sinogram( a, b, nb_bins = nb_bins, extent = extent )
    halo = Halo( ref, outer_radius = b, inner_radius = a, growth = b / a, nb_sectors = 1,
                 nb_coarse_bins = 64 )
    assert halo.nb_cells == 1, f"attendu un seul secteur, obtenu { halo.nb_cells }"

    got = halo.operator[ 0 ]                                          # [ nb_angles, nb_coarse ]
    exp = np.asarray( ref.values ).reshape( got.shape[ 0 ], got.shape[ 1 ], halo.group ).mean( axis = 2 )

    # masse : l'anneau tient dans le détecteur, elle doit valoir son aire à tous les angles
    area = np.pi * ( b * b - a * a )
    mass = got.sum( axis = 1 ) * halo.coarse_dw
    assert np.allclose( mass, area, rtol = 1e-3 ), f"masse { mass } != aire { area }"

    # forme : l'intégration radiale est exacte, il ne reste que la quadrature angulaire
    err = np.abs( got - exp ).max() / exp.max()
    assert err < 0.02, f"profil trop loin de l'anneau analytique : erreur relative { err }"

    # un anneau centré projette la MÊME chose à tous les angles
    assert np.abs( got - got[ 0 ] ).max() / exp.max() < 0.02, "projection non invariante par angle"


if test( "halo_operator_sector_partition" ):
    # les secteurs d'un anneau forment une PARTITION : leurs opérateurs doivent se resommer en
    # celui de l'anneau entier (à densité égale).
    a, b = 1.0, 2.0
    ref = _annulus_sinogram( a, b )
    whole = Halo( ref, outer_radius = b, inner_radius = a, growth = b / a, nb_sectors = 1, nb_coarse_bins = 64 )
    split = Halo( ref, outer_radius = b, inner_radius = a, growth = b / a, nb_sectors = 8, nb_coarse_bins = 64 )
    assert split.nb_cells > 1

    summed = split.operator.sum( axis = 0 )
    err = np.abs( summed - whole.operator[ 0 ] ).max() / whole.operator[ 0 ].max()
    assert err < 0.02, f"la somme des secteurs ne redonne pas l'anneau : { err }"
    assert np.isclose( split.areas.sum(), whole.areas.sum(), rtol = 1e-12 )


if test( "interior_values_conserves_mass" ):
    # la projection du nuage doit porter EXACTEMENT la masse demandée, à tous les angles (dépôt
    # linéaire sur les deux cases voisines) -- c'est ce qui rend le résidu interprétable.
    sino = Sinogram( nb_angles = 7, nb_bins = 128, extent = 8.0 )
    pts = np.random.default_rng( 0 ).normal( scale = 0.6, size = ( 500, 2 ) )
    vals = interior_values( sino, pts, mass = 3.0 )
    assert np.allclose( vals.sum( axis = 1 ) * sino.dw, 3.0, rtol = 1e-9 )

    # avec un rayon, on passe par `DiskProjector` : même masse
    vals = interior_values( sino, pts, mass = 3.0, radius = 0.2 )
    assert np.allclose( vals.sum( axis = 1 ) * sino.dw, 3.0, rtol = 1e-3 )


if test( "halo_fit_recovers_outside_mass" ):
    # vérité terrain : un disque DANS le champ, un disque DEHORS. On donne au halo le résidu exact
    # (mesuré moins la projection du disque intérieur) et il doit retrouver la masse extérieure.
    extent, nb_bins, nb_angles = 4.0, 400, 60
    inside, outside, r_out = ( 0.2, -0.1 ), ( 2.6, 0.4 ), 0.5

    full = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    full.add_disk( center = list( inside ), radius = 0.6 )
    full.add_disk( center = list( outside ), radius = r_out )

    only_in = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    only_in.add_disk( center = list( inside ), radius = 0.6 )

    residual = np.asarray( full.values ) - np.asarray( only_in.values )
    per_angle = mass_profile( full )
    target = per_angle - per_angle.min()

    halo = Halo( full, outer_radius = 4.0, nb_coarse_bins = 100 )
    halo.fit( residual, target_mass = target )

    # le halo ne voit que ce qui tombe dans le détecteur : c'est cette masse-là qui est contrainte
    got, exp = halo.visible_mass(), residual.sum( axis = 1 ) * full.dw
    err = np.abs( got - exp ).max() / exp.max()
    assert err < 0.12, f"masse visible du halo mal retrouvée : { err }"

    # et il se place bien DEHORS, du bon côté
    weight_by_ring = { }
    for w, ( _, _, p0, p1, _ ) in zip( halo.weights, halo.cells ):
        weight_by_ring[ 0.5 * ( p0 + p1 ) ] = weight_by_ring.get( 0.5 * ( p0 + p1 ), 0.0 ) + w
    best = max( weight_by_ring, key = weight_by_ring.get )
    expected_phi = np.arctan2( outside[ 1 ], outside[ 0 ] ) % ( 2 * np.pi )
    gap = abs( ( best - expected_phi + np.pi ) % ( 2 * np.pi ) - np.pi )
    assert gap < 1.0, f"halo placé à φ={ best:.2f} au lieu de { expected_phi:.2f}"


if test( "halo_corrected_equalizes_mass" ):
    # après correction, `∫p_θ` doit être BEAUCOUP plus plat : c'est la mesure directe de la fuite.
    extent, nb_bins, nb_angles = 4.0, 400, 60
    full = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    full.add_disk( center = [ 0.2, -0.1 ], radius = 0.6 )
    full.add_disk( center = [ 2.6, 0.4 ], radius = 0.5 )
    full.add_disk( center = [ -2.2, -1.5 ], radius = 0.4 )

    only_in = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    only_in.add_disk( center = [ 0.2, -0.1 ], radius = 0.6 )

    per_angle = mass_profile( full )
    halo = Halo( full, outer_radius = 4.0, nb_coarse_bins = 100 )
    halo.fit( np.asarray( full.values ) - np.asarray( only_in.values ),
              target_mass = per_angle - per_angle.min() )

    # trois blobs extérieurs COMPACTS, c'est le cas le plus exigeant pour un maillage volontairement
    # grossier (un objet extérieur étendu, le cas réel, lui va bien mieux) -- d'où un facteur 3 et
    # non un ordre de grandeur.
    before = per_angle.std() / per_angle.mean()
    after_profile = mass_profile( halo.corrected() )
    after = after_profile.std() / after_profile.mean()
    assert after < before / 3, f"masse par angle pas assez égalisée : { before:.4f} -> { after:.4f}"


if test( "alternate_preserves_voids" ):
    # bout en bout, sur ce qui motive le module : un objet troué DANS le champ, de la matière
    # DEHORS. Sans halo, l'excédent de masse est redistribué dans le champ et bouche les trous ;
    # avec, les trous doivent revenir.
    extent, nb_bins, nb_angles = 4.0, 300, 90
    rng = np.random.default_rng( 0 )

    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    holes = [ ( 0.55, 0.55 ), ( -0.55, 0.55 ), ( 0.55, -0.55 ), ( -0.55, -0.55 ) ]
    sino.add_disk( center = [ 0.0, 0.0 ], radius = 1.4 )                       # objet plein
    for h in holes:
        sino.add_disk( center = list( h ), radius = 0.32, density = -1.0 )     # ... troué
    for c in [ ( 2.6, 0.5 ), ( -2.4, -1.0 ), ( 0.3, 2.7 ) ]:                   # matière DEHORS
        sino.add_disk( center = list( c ), radius = 0.5 )

    def solve( rec ):
        return rec.diracs( max_iter = 60 )

    common = dict( outer_radius = 4.0, extent = extent, seed = 1, nb_points = 600,
                   max_residual_points = 20_000,
                   halo_kwargs = dict( nb_coarse_bins = 100 ) )
    plain, _ = alternate( sino, solve, nb_outer = 1, **common )                # halo nul
    fixed, halo = alternate( sino, solve, nb_outer = 3, **common )

    # le halo doit avoir trouvé quelque chose, et aplati la masse par angle
    assert halo.mass() > 0, "halo resté nul"
    raw, cor = mass_profile( sino ), mass_profile( halo.corrected() )
    assert cor.std() / cor.mean() < raw.std() / raw.mean() / 3, "masse par angle pas égalisée"

    # ... et les trous doivent être plus vides. On les mesure là où ils sont, pas globalement.
    def in_holes( pts ):
        pts = np.asarray( pts )
        inside = [ ( ( pts - np.array( h ) ) ** 2 ).sum( axis = 1 ) < 0.25 ** 2 for h in holes ]
        return float( np.any( inside, axis = 0 ).mean() )

    assert in_holes( fixed.positions ) < in_holes( plain.positions ) * 0.7, (
        f"les trous ne se sont pas vidés : { in_holes( plain.positions ):.4f} -> "
        f"{ in_holes( fixed.positions ):.4f}" )
    # ... et moins de points doivent traîner hors de l'objet (la masse en trop y allait aussi)
    def outside_frac( pts ):
        return float( ( np.linalg.norm( np.asarray( pts ), axis = 1 ) > 1.5 ).mean() )
    assert outside_frac( fixed.positions ) < outside_frac( plain.positions )
    assert void_fraction( fixed.positions, extent ) > void_fraction( plain.positions, extent )


if test( "scan_interior_mass_is_monotone" ):
    # `M_in` étant le paramètre décisif (voir `halo.scan_interior_mass`), le balayage doit au moins
    # être cohérent : moins on attribue de masse à l'intérieur, plus le halo en prend.
    extent, nb_bins, nb_angles = 4.0, 400, 60
    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    sino.add_disk( center = [ 0.0, 0.0 ], radius = 1.0 )
    sino.add_disk( center = [ 2.6, 0.4 ], radius = 0.5 )

    rng = np.random.default_rng( 0 )
    pts = rng.normal( scale = 0.5, size = ( 3000, 2 ) )
    halo = Halo( sino, outer_radius = 4.0, nb_coarse_bins = 100 )
    before = halo.weights.copy()

    scan = scan_interior_mass( halo, pts )
    assert np.all( np.diff( scan[ "halo_mass" ] ) < 0 ), (
        f"masse du halo non décroissante en M_in : { scan[ 'halo_mass' ] }" )
    assert np.array_equal( halo.weights, before ), "le balayage ne doit pas modifier le halo"


if test( "void_fraction_is_calibrated" ):
    rng = np.random.default_rng( 0 )
    n, extent = 4000, 2.0
    # un nuage uniforme sur TOUT le domaine, à la grille par défaut (√n de côté) : ~1/e de vide
    full = ( rng.random( ( n, 2 ) ) - 0.5 ) * extent
    assert 0.3 < void_fraction( full, extent ) < 0.45

    # le même nuage tassé dans la moitié gauche : il ne peut plus rien remplir à droite
    half = full.copy()
    half[ :, 0 ] = half[ :, 0 ] / 2 - extent / 4
    assert void_fraction( half, extent ) > void_fraction( full, extent ) + 0.15
