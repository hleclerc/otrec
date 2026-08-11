"""Vérifie la référence SYCL fusionnée (`dirac_sycl.diracs_cost_grad`) contre le chemin PUR JAX
existant (`models.DiracModel.cost` + `jax.grad`) : même formule (transport optimal 1D
semi-discret, diracs de masse égale), donc coût ET gradient doivent coïncider à la précision
flottante près -- voir `dirac_sycl.py` pour pourquoi un seul kernel fwd-only suffit ici (pas de
`bwd_code`, le gradient est écrit directement par la formule fermée).
"""
import numpy as np

from reconstruction.Sinogram import Sinogram
from reconstruction.models import DiracModel
from reconstruction.dirac_sycl import diracs_cost_grad, subspace_hessian, MAX_DIRS
from sdot import driver
from sdot.testing import test


def _disk_sinogram( nb_angles = 8, nb_bins = 201, extent = 6.0, center = ( 0.3, -0.2 ), radius = 1.0 ):
    s = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    s.add_disk( center = list( center ), radius = radius )
    return s


if test( "diracs_cost_grad_matches_pure_jax" ):
    sino = _disk_sinogram()
    rng = np.random.default_rng( 0 )
    pts = ( rng.random( ( 137, 2 ) ) - 0.5 ) * 2

    cost_sycl, grad_sycl = diracs_cost_grad( pts, sino )

    model = DiracModel( sino )
    def scalar_loss( p ):
        return model.cost( model.wrap( p ) ).tensor
    cost_jax = float( driver.jit( scalar_loss )( pts ) )
    grad_jax = np.asarray( driver.jit( driver.grad( scalar_loss ) )( pts ) )

    assert np.isfinite( cost_sycl )
    assert abs( cost_sycl - cost_jax ) < 1e-8 * max( 1.0, abs( cost_jax ) ), \
        f"coût SYCL { cost_sycl } != coût Jax { cost_jax }"
    assert np.allclose( grad_sycl, grad_jax, atol = 1e-6, rtol = 1e-5 ), \
        f"gradient SYCL != gradient Jax, écart max { np.max( np.abs( grad_sycl - grad_jax ) ) }"


if test( "diracs_cost_grad_single_angle" ):
    # cas limite : un seul angle -- vérifie que le batching sur `num_angle` dégénère correctement.
    sino = _disk_sinogram( nb_angles = 1 )
    rng = np.random.default_rng( 1 )
    pts = ( rng.random( ( 23, 2 ) ) - 0.5 ) * 2

    cost_sycl, grad_sycl = diracs_cost_grad( pts, sino )

    model = DiracModel( sino )
    def scalar_loss( p ):
        return model.cost( model.wrap( p ) ).tensor
    cost_jax = float( driver.jit( scalar_loss )( pts ) )
    grad_jax = np.asarray( driver.jit( driver.grad( scalar_loss ) )( pts ) )

    assert abs( cost_sycl - cost_jax ) < 1e-8 * max( 1.0, abs( cost_jax ) )
    assert np.allclose( grad_sycl, grad_jax, atol = 1e-6, rtol = 1e-5 )


def _random_directions( rng, shape, m ):
    """`[MAX_DIRS,*shape]`, zero-paddée au-delà de `m` directions actives, chacune de norme 1 --
    voir `optimizers.SubspaceNewtonLBFGS` pour la normalisation en contexte réel (des gradients
    normalisés, ici de simples directions aléatoires : `subspace_hessian` ne connaît pas leur
    origine, seule leur valeur compte)."""
    directions = np.zeros( ( MAX_DIRS, ) + shape )
    for i in range( m ):
        d = rng.standard_normal( shape )
        directions[ i ] = d / np.linalg.norm( d )
    return directions


if test( "subspace_hessian_b_matches_grad_dot_directions" ):
    # `b_i` du sous-espace DOIT être exactement `directions[i] . grad_sycl` -- même somme
    # (angle par angle, dirac par dirac) que `diracs_cost_grad` accumule déjà pour `grad`, juste
    # projetée sur `directions[i]` au lieu d'être scatterée sur les points bruts (voir la dérivation
    # par la règle de la chaîne dans la docstring de `subspace_hessian`).
    sino = _disk_sinogram()
    rng = np.random.default_rng( 2 )
    pts = ( rng.random( ( 97, 2 ) ) - 0.5 ) * 2

    _, grad_sycl = diracs_cost_grad( pts, sino )

    m = 3
    directions = _random_directions( rng, pts.shape, m )
    H, b = subspace_hessian( pts, directions, sino )

    assert H.shape == ( MAX_DIRS, MAX_DIRS )
    assert b.shape == ( MAX_DIRS, )

    expected_b = np.einsum( "ind,nd->i", directions, grad_sycl )
    assert np.allclose( b, expected_b, atol = 1e-6, rtol = 1e-5 ), \
        f"b != directions . grad, écart max { np.max( np.abs( b - expected_b ) ) }"

    # slots inactifs (zero-paddés au-delà de m) : contribution nulle sur ces lignes/colonnes.
    assert np.allclose( b[ m: ], 0 )
    assert np.allclose( H[ m:, : ], 0 )
    assert np.allclose( H[ :, m: ], 0 )

    # symétrie : H_ij et H_ji sont accumulées INDÉPENDAMMENT dans le kernel (pas de symétrisation
    # forcée), donc ce test couvre aussi un bug d'indexation i/j inversé.
    assert np.allclose( H[ :m, :m ], H[ :m, :m ].T, atol = 1e-6 ), \
        f"H non symétrique, écart max { np.max( np.abs( H[ :m, :m ] - H[ :m, :m ].T ) ) }"


if test( "subspace_hessian_matches_finite_difference" ):
    # H doit être la dérivée EXACTE de b le long de chaque direction stockée (même approximation
    # "assignation figée" des deux côtés -- voir la docstring de `subspace_hessian`) : perturber
    # `pts` de `eps * directions[i]` revient à `a = eps * e_i` dans les coordonnées du sous-espace.
    sino = _disk_sinogram()
    rng = np.random.default_rng( 3 )
    pts = ( rng.random( ( 61, 2 ) ) - 0.5 ) * 2

    m = 2
    directions = _random_directions( rng, pts.shape, m )
    H, b0 = subspace_hessian( pts, directions, sino )

    eps = 1e-4
    for i in range( m ):
        _, b_eps = subspace_hessian( pts + eps * directions[ i ], directions, sino )
        fd_col = ( b_eps - b0 ) / eps
        assert np.allclose( fd_col[ :m ], H[ :m, i ], atol = 1e-2, rtol = 1e-1 ), \
            f"colonne { i } de H (diff. finie) { fd_col[ :m ] } != H[:,{ i }] { H[ :m, i ] }"
