"""Checks the fused loom kernel (`dirac_fused.diracs_cost_grad`) against the existing PURE JAX
path (`models.DiracModel.cost` + `jax.grad`): same formula (semi-discrete 1D optimal transport,
equal-mass diracs), so cost AND gradient must coincide up to floating-point precision -- see
`dirac_fused.py` for why a single fwd-only kernel is enough here (no `bwd_code`, the gradient is
written directly by the closed-form formula).
"""
import numpy as np

from otrec.Sinogram import Sinogram
from otrec.models import DiracModel
from otrec.dirac_fused import diracs_cost_grad, subspace_hessian, MAX_DIRS
from loom import driver
from errand import test
from loom.testing import need


def _disk_sinogram( nb_angles = 8, nb_bins = 201, extent = 6.0, center = ( 0.3, -0.2 ), radius = 1.0 ):
    s = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    s.add_disk( center = list( center ), radius = radius )
    return s


if test( "diracs_cost_grad_matches_pure_jax" ):
    need( "grad" )
    sino = _disk_sinogram()
    rng = np.random.default_rng( 0 )
    pts = ( rng.random( ( 137, 2 ) ) - 0.5 ) * 2

    cost_fused, grad_fused = diracs_cost_grad( pts, sino )

    model = DiracModel( sino )
    def scalar_loss( p ):
        return model.cost( model.wrap( p ) ).value
    cost_jax = float( driver.jit( scalar_loss )( pts ) )
    grad_jax = np.asarray( driver.jit( driver.grad( scalar_loss ) )( pts ) )

    assert np.isfinite( cost_fused )
    assert abs( cost_fused - cost_jax ) < 1e-8 * max( 1.0, abs( cost_jax ) ), \
        f"Fused cost { cost_fused } != Jax cost { cost_jax }"
    assert np.allclose( grad_fused, grad_jax, atol = 1e-6, rtol = 1e-5 ), \
        f"Fused gradient != Jax gradient, max deviation { np.max( np.abs( grad_fused - grad_jax ) ) }"


if test( "diracs_cost_grad_single_angle" ):
    need( "grad" )
    # edge case: a single angle -- checks that batching over `num_angle` degenerates correctly.
    sino = _disk_sinogram( nb_angles = 1 )
    rng = np.random.default_rng( 1 )
    pts = ( rng.random( ( 23, 2 ) ) - 0.5 ) * 2

    cost_fused, grad_fused = diracs_cost_grad( pts, sino )

    model = DiracModel( sino )
    def scalar_loss( p ):
        return model.cost( model.wrap( p ) ).value
    cost_jax = float( driver.jit( scalar_loss )( pts ) )
    grad_jax = np.asarray( driver.jit( driver.grad( scalar_loss ) )( pts ) )

    assert abs( cost_fused - cost_jax ) < 1e-8 * max( 1.0, abs( cost_jax ) )
    assert np.allclose( grad_fused, grad_jax, atol = 1e-6, rtol = 1e-5 )


def _random_directions( rng, shape, m ):
    """`[MAX_DIRS,*shape]`, zero-padded beyond `m` active directions, each of norm 1 --
    see `optimizers.SubspaceNewtonLBFGS` for the normalization in a real context (normalized
    gradients; here just random directions: `subspace_hessian` does not know where they come
    from, only their value matters)."""
    directions = np.zeros( ( MAX_DIRS, ) + shape )
    for i in range( m ):
        d = rng.standard_normal( shape )
        directions[ i ] = d / np.linalg.norm( d )
    return directions


if test( "subspace_hessian_b_matches_grad_dot_directions" ):
    # the subspace `b_i` MUST be exactly `directions[i] . grad_fused` -- same sum
    # (angle by angle, dirac by dirac) that `diracs_cost_grad` already accumulates for `grad`, just
    # projected onto `directions[i]` instead of being scattered onto the raw points (see the
    # chain-rule derivation in the docstring of `subspace_hessian`).
    sino = _disk_sinogram()
    rng = np.random.default_rng( 2 )
    pts = ( rng.random( ( 97, 2 ) ) - 0.5 ) * 2

    _, grad_fused = diracs_cost_grad( pts, sino )

    m = 3
    directions = _random_directions( rng, pts.shape, m )
    H, b = subspace_hessian( pts, directions, sino )

    assert H.shape == ( MAX_DIRS, MAX_DIRS )
    assert b.shape == ( MAX_DIRS, )

    expected_b = np.einsum( "ind,nd->i", directions, grad_fused )
    assert np.allclose( b, expected_b, atol = 1e-6, rtol = 1e-5 ), \
        f"b != directions . grad, max deviation { np.max( np.abs( b - expected_b ) ) }"

    # inactive slots (zero-padded beyond m): zero contribution on these rows/columns.
    assert np.allclose( b[ m: ], 0 )
    assert np.allclose( H[ m:, : ], 0 )
    assert np.allclose( H[ :, m: ], 0 )

    # symmetry: H_ij and H_ji are accumulated INDEPENDENTLY in the kernel (no forced
    # symmetrization), so this test also covers a swapped i/j indexing bug.
    assert np.allclose( H[ :m, :m ], H[ :m, :m ].T, atol = 1e-6 ), \
        f"H not symmetric, max deviation { np.max( np.abs( H[ :m, :m ] - H[ :m, :m ].T ) ) }"


if test( "subspace_hessian_matches_finite_difference" ):
    # H must be the EXACT derivative of b along each stored direction (same "frozen assignment"
    # approximation on both sides -- see the docstring of `subspace_hessian`): perturbing `pts` by
    # `eps * directions[i]` amounts to `a = eps * e_i` in the subspace coordinates.
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
            f"column { i } of H (finite diff.) { fd_col[ :m ] } != H[:,{ i }] { H[ :m, i ] }"
