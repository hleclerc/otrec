"""Fused reference (loom `FfiCode` kernel, CPU for now) of the cost + gradient of the DIRACS model (`models.DiracModel`),
in a SINGLE fwd-only `loom.ffi_call` -- without `backward`, hence without going through Jax autodiff: the
gradient is written directly by the kernel, closed formula `(point - barycenter) * direction`,
instead of a `jax.grad` through `_pure_jax_cost1d.cost_1d_ot` (the default path,
`Image.try_update_sdotplan1d`) or the fwd/bwd pair of `SdotPlan1d.cxx` (the general C++ path).

Motivation: compare the speed of a kernel that fuses BOTH passes (cost AND gradient, the
same formula as `SdotPlan1d.cxx::sweep_outputs_bwd`'s barycentre-recompute branch, but computed
ONCE instead of twice) against the pure-Jax pipeline currently used by
`Reconstruction.diracs`. Deliberately simpler than `SdotPlan1d`: a single work-item per angle
(same LSD radix sort on a key+index packet as `SdotPlan1d.cxx::sort_diracs`, but WITHOUT its group
cooperation -- a single thread suffices since it processes the whole angle -- and not the
`udp_at`/`cell_cum_mass` walk -- a simple sequential `udp_start` suffices for the same reason).
Defer group cooperation to a later stage if this reference proves promising
at larger scale.

Reuses `ProjectedSumOfDiracs`/`Image` AS IS (same C++ structs, same `position(i)`/`udp_start`/
`udp_cont` methods as `SdotPlan1d.cxx`) -- only the orchestration (sort + sweep
+ gradient scatter) is new.

Turned out promising enough (10-30x faster than the Jax path measured on the lung, see
`benchmarks/execution_speed/benchmark_fused.py`) to be WIRED IN: `models.DiracModel.value_and_grad`
exposes `diracs_cost_grad` with the same contract that `optimizers.FusedLBFGS` expects, and
`Reconstruction.diracs( backend = "fused" )` routes to it (see `experiments/lung_alveoli.py`).
"""
import numpy as np

import loom
import loom
from loom import Tensor, Axis, CtShapeVar, RealTensor, IntTensor
from loom.compilation.FfiCode import FfiCode
from sdot.distributions.ProjectedSumOfDiracs import ProjectedSumOfDiracs

from .Sinogram import Sinogram

# max number of directions of the `subspace_hessian` subspace -- see its docstring. Fixed (not a
# runtime `ShapeVar`): a single compiled kernel serves all calls, `optimizers.SubspaceNewtonLBFGS`
# zero-padding beyond the number of directions actually stored.
MAX_DIRS = 5


def diracs_cost_grad( points, sinogram: Sinogram ):
    """`(cost, grad)` of the DIRACS model for `points` (`[n,2]`, Tensor or array) against
    `sinogram`, computed by the fused kernel above -- `cost` a Python float,
    `grad` a numpy array `[n,2]` (same sign convention as `jax.grad(model.cost)`, so a gradient
    descent step is `points - lr * grad`).
    """
    pts = points if isinstance( points, Tensor ) else RealTensor( points )

    # `src`/`dst` normalized to mass 1 -- exactly what `SdotPlan1d.__init__` does before
    # sweeping (see its docstring): without it the `udp_cont` walk (whose per-dirac `w` takes
    # must exactly exhaust the total mass of the image) would not sweep the whole image.
    # `src` has no explicit weights -> `normalized_version()` would give UNIFORM weights
    # `1 / n`, exactly what the kernel computes itself (`w = TF(1)/TF(n)`) -- no need to
    # materialize a weight tensor just for that.
    src = ProjectedSumOfDiracs( points = pts, normal = sinogram.normals_t,
                                batch_axes = [ sinogram.num_angle ] )
    dst = sinogram.batched_image().normalized_version()

    sorted_idx = IntTensor[ sinogram.num_angle, src.num_dirac ]()
    radix_tmp = IntTensor[ sinogram.num_angle, src.num_dirac ]()
    cost = RealTensor[ sinogram.num_angle ]()
    grad = RealTensor[ src.num_dirac, src.proj_dim ]()

    loom.ffi_call(
        "diracs_fused_cost_grad",
        FfiCode.per_item(
            includes = [ "loom/support/atomic_add.h" ],
            # ( `grad` is SHARED -- the points are the same at all angles -- and accumulated by
            # `atomic_add`, so it must start from zero. No need to say so here anymore: every shared
            # floating-point output of a batched call is seeded with zero automatically, see
            # `CallArg_Tensor.cpp_seed_member`. )
            code = """
            {
                const SI n = SI( inputs.src.points.shape( 0 ) );
                auto order = scratch.sorted_idx( batch_index );
                auto tmp   = scratch.radix_tmp( batch_index );

                // view sliced at THIS angle, built ONCE -- `inputs.src( batch_index )` would otherwise redo
                // this resolution (per-angle normal, etc.) at EACH comparison of the sort
                // (O(n log n) times) and at each step of the `udp_cont` walk (O(n) times).
                auto s = inputs.src( batch_index );

                // on-the-fly projection (no materialized [nb_angles, n] array): the same
                // method as the general C++ path, `ProjectedSumOfDiracs::position`.
                auto proj = [&]( SI i ) { return s.position( i ); };

                // Iterative O(n) LSD radix sort, NO comparator (no recursion -- same
                // SSCP constraint that ruled out std::sort/introsort, see SdotPlan1d.cxx::sort_diracs,
                // of which this block is a simplified version WITHOUT group cooperation, a single
                // work-item processing the whole angle). We PACK, in each int64 slot of `order`,
                // an ordered float32 key (IEEE bits flipped to preserve ordering) in the
                // high bits + the dirac index in the low bits, then sort THIS PACKET
                // directly (contiguous, no indirection) instead of comparing indirectly via
                // `proj( order( i ) )` -- which recomputed the projection dot product at
                // EACH comparison of the previous heap sort (O(n log n) times) instead of once
                // per dirac here.
                for ( SI i = 0; i < n; ++i ) {
                    const float f = float( proj( i ) );
                    uint32_t u = __builtin_bit_cast( uint32_t, f );
                    u ^= ( u & 0x80000000u ) ? 0xFFFFFFFFu : 0x80000000u;
                    order( i ) = ( SI( u >> 1 ) << 32 ) | SI( i );
                }

                constexpr int NB_BITS    = 8;
                constexpr int NB_BUCKETS = 1 << NB_BITS;
                constexpr int NB_PASSES  = 32 / NB_BITS;
                static_assert( NB_PASSES % 2 == 0 ); // the final result must land back in `order`

                auto radix_pass = [&]( auto &&from, auto &&to, int shift ) {
                    SI count[ NB_BUCKETS ] = { 0 };
                    for ( SI i = 0; i < n; ++i )
                        ++count[ ( SI( from( i ) ) >> shift ) & ( NB_BUCKETS - 1 ) ];
                    SI sum = 0;
                    for ( int b = 0; b < NB_BUCKETS; ++b ) {
                        const SI c = count[ b ];
                        count[ b ] = sum;
                        sum += c;
                    }
                    for ( SI i = 0; i < n; ++i ) {
                        const SI key = from( i );
                        const int b = ( key >> shift ) & ( NB_BUCKETS - 1 );
                        to( count[ b ]++ ) = key;
                    }
                };
                for ( int p = 0; p < NB_PASSES; ++p ) {
                    const int shift = 32 + p * NB_BITS;
                    if ( p % 2 == 0 ) radix_pass( order, tmp, shift );
                    else              radix_pass( tmp, order, shift );
                }

                inputs.dst( batch_index ).with_defaults( [&]( auto &&img ) {
                    using TF = DECAYED_TYPE_OF( img.values )::TF;
                    const TF w = TF( 1 ) / TF( n );

                    // a single work-item for the whole angle -> no need for `udp_at`/`cell_cum_mass`
                    // (the walk starts from the very beginning, as `udp_at( cell_cum_mass, 0 )` would).
                    auto udp = img.udp_start();
                    TF local_cost = 0;
                    for ( SI k = 0; k < n; ++k ) {
                        const SI di = order( k ) & 0xFFFFFFFFll; // decode: low bits = original index
                        const TF p = proj( di );

                        // cost AND barycenter (first moment) IN A SINGLE walk -- no
                        // `barycenters` buffer [nb_angles, n], no second bwd pass.
                        TF moment = 0;
                        img.udp_cont( udp, w, [&]( auto &&item ) {
                            local_cost += item.w2_dist( p );
                            moment += item.first_moment();
                        } );
                        const TF b = moment / w;

                        // d outputs.cost / d position(i) = 2 w (p - b) ; d position / d point = normal
                        // (see `ProjectedSumOfDiracs::add_position_grad`) -- same formula, but
                        // written directly into `outputs.grad` (raw output) rather than via the
                        // `grad_src` indirection that the Jax differentiation protocol would build for a bwd.
                        const TF grad_s = TF( 2 ) * w * ( p - b );
                        atomic_add( outputs.grad( num_dirac = di, proj_dim = 0 ).ref(),
                                    TF( grad_s * TF( s.normal( proj_dim = 0 ) ) ) );
                        atomic_add( outputs.grad( num_dirac = di, proj_dim = 1 ).ref(),
                                    TF( grad_s * TF( s.normal( proj_dim = 1 ) ) ) );
                    }
                    outputs.cost( batch_index ) = local_cost;
                } );
            }
            """,
        ),
        src = src,
        dst = dst,
        sorted_idx = loom.scratch( sorted_idx ),
        radix_tmp = loom.scratch( radix_tmp ),
        cost = loom.out( cost ),
        grad = loom.out( grad ),
        has_dynamic_capacity = False,
    )

    return float( np.asarray( cost.value ).sum() ), np.asarray( grad.value )


def diracs_cost( points, sinogram: Sinogram ):
    """`cost` ALONE of the DIRACS model -- SAME formula and SAME sort + sweep kernel as
    `diracs_cost_grad` above, but without the gradient computation (no moment/barycenter per
    dirac, no atomic scatter): for the "cost only" evaluations of a line search
    (see `experiments.lung_alveoli._parabolic_bracket`), where the gradient would be thrown away
    anyway. Saves the `first_moment()` computation and the per-dirac `atomic_add`s -- the sort stays
    identical (it is what dominates the cost, see `otplan1d-kernel-profile`).
    """
    pts = points if isinstance( points, Tensor ) else RealTensor( points )

    src = ProjectedSumOfDiracs( points = pts, normal = sinogram.normals_t,
                                batch_axes = [ sinogram.num_angle ] )
    dst = sinogram.batched_image().normalized_version()

    sorted_idx = IntTensor[ sinogram.num_angle, src.num_dirac ]()
    radix_tmp = IntTensor[ sinogram.num_angle, src.num_dirac ]()
    cost = RealTensor[ sinogram.num_angle ]()

    loom.ffi_call(
        "diracs_fused_cost_only",
        FfiCode.per_item(
            code = """
            {
                const SI n = SI( inputs.src.points.shape( 0 ) );
                auto order = scratch.sorted_idx( batch_index );
                auto tmp   = scratch.radix_tmp( batch_index );

                auto s = inputs.src( batch_index );
                auto proj = [&]( SI i ) { return s.position( i ); };

                for ( SI i = 0; i < n; ++i ) {
                    const float f = float( proj( i ) );
                    uint32_t u = __builtin_bit_cast( uint32_t, f );
                    u ^= ( u & 0x80000000u ) ? 0xFFFFFFFFu : 0x80000000u;
                    order( i ) = ( SI( u >> 1 ) << 32 ) | SI( i );
                }

                constexpr int NB_BITS    = 8;
                constexpr int NB_BUCKETS = 1 << NB_BITS;
                constexpr int NB_PASSES  = 32 / NB_BITS;
                static_assert( NB_PASSES % 2 == 0 );

                auto radix_pass = [&]( auto &&from, auto &&to, int shift ) {
                    SI count[ NB_BUCKETS ] = { 0 };
                    for ( SI i = 0; i < n; ++i )
                        ++count[ ( SI( from( i ) ) >> shift ) & ( NB_BUCKETS - 1 ) ];
                    SI sum = 0;
                    for ( int b = 0; b < NB_BUCKETS; ++b ) {
                        const SI c = count[ b ];
                        count[ b ] = sum;
                        sum += c;
                    }
                    for ( SI i = 0; i < n; ++i ) {
                        const SI key = from( i );
                        const int b = ( key >> shift ) & ( NB_BUCKETS - 1 );
                        to( count[ b ]++ ) = key;
                    }
                };
                for ( int p = 0; p < NB_PASSES; ++p ) {
                    const int shift = 32 + p * NB_BITS;
                    if ( p % 2 == 0 ) radix_pass( order, tmp, shift );
                    else              radix_pass( tmp, order, shift );
                }

                inputs.dst( batch_index ).with_defaults( [&]( auto &&img ) {
                    using TF = DECAYED_TYPE_OF( img.values )::TF;
                    const TF w = TF( 1 ) / TF( n );

                    auto udp = img.udp_start();
                    TF local_cost = 0;
                    for ( SI k = 0; k < n; ++k ) {
                        const SI di = order( k ) & 0xFFFFFFFFll;
                        const TF p = proj( di );
                        img.udp_cont( udp, w, [&]( auto &&item ) {
                            local_cost += item.w2_dist( p );
                        } );
                    }
                    outputs.cost( batch_index ) = local_cost;
                } );
            }
            """,
        ),
        src = src,
        dst = dst,
        sorted_idx = loom.scratch( sorted_idx ),
        radix_tmp = loom.scratch( radix_tmp ),
        cost = loom.out( cost ),
        has_dynamic_capacity = False,
    )

    return float( np.asarray( cost.value ).sum() )


def subspace_hessian( points, directions, sinogram: Sinogram ):
    """`(H, b)` -- Hessian `[MAX_DIRS,MAX_DIRS]` and gradient `[MAX_DIRS]` of the DIRACS model
    RESTRICTED to the subspace spanned by `directions` (`[MAX_DIRS,n,2]`, zero-padded beyond the
    number of actually active directions -- see `optimizers.SubspaceNewtonLBFGS`), i.e. of
    `a -> loss( points + sum_i a_i * directions[i] )` evaluated at `a = 0`.

    Same sort + `udp_start`/`udp_cont` sweep as `diracs_cost_grad` (same `sorted_idx`/
    `radix_tmp` scratch, RECOMPUTED here rather than reused as residuals -- this kernel runs once
    per OUTER step of `SubspaceNewtonLBFGS`, not per inner scipy step, so the extra O(n)
    per angle is negligible against the gain -- sharing the residuals would be a
    later optimization if this prototype proves worthwhile).

    Relies on the SAME "frozen assignment" approximation as `grad_s = 2w(p-b)` in
    `diracs_cost_grad` (the barycenter `b` of each dirac treated as constant with respect to its
    position): with the assignment fixed, the 1D-OT cost of a dirac is EXACTLY quadratic in its
    projected position `p`, and `p` is itself AFFINE in `a` (`p(a) = p(0) + sum_i a_i * e_i`,
    `e_i = directions[i][dirac]·normal`). Hence, per angle and per dirac, in A SINGLE pass:
    `b_i += grad_s * e_i` (subspace gradient) and `H_ij += 2*w*e_i*e_j` (EXACT Hessian of
    this local model, NOT a finite difference) -- a sum of rank-1 `w*e*e^T` terms, so `H` is
    PSD by construction (never any negative curvature to handle on the solver side).
    """
    pts = points if isinstance( points, Tensor ) else RealTensor( points )
    dirs = directions if isinstance( directions, Tensor ) else RealTensor( directions )

    src = ProjectedSumOfDiracs( points = pts, normal = sinogram.normals_t,
                                batch_axes = [ sinogram.num_angle ] )
    dst = sinogram.batched_image().normalized_version()

    # two DISTINCT axes (same shared MAX_DIRS count): `H` is square [MAX_DIRS,MAX_DIRS], so
    # its two dimensions cannot share A SINGLE `Axis` object (the kernel's named indexing,
    # `H( dir_index_i = i, dir_index_j = j )`, needs two distinct names -- and
    # `Tensor._dim_index` could not resolve the ambiguity between two occurrences of the same axis).
    max_dirs = CtShapeVar( MAX_DIRS )
    dir_index_i = Axis( max_dirs, name = "dir_index_i" )
    dir_index_j = Axis( max_dirs, name = "dir_index_j" )

    sorted_idx = IntTensor[ sinogram.num_angle, src.num_dirac ]()
    radix_tmp = IntTensor[ sinogram.num_angle, src.num_dirac ]()
    directions_t = RealTensor[ dir_index_i, src.num_dirac, src.proj_dim ]( dirs )
    H = RealTensor[ dir_index_i, dir_index_j ]()
    b = RealTensor[ dir_index_i ]()

    loom.ffi_call(
        "diracs_subspace_hessian",
        FfiCode.per_item(
            includes = [ "loom/support/atomic_add.h" ],
            # ( `H`/`b` are SHARED -- the diracs are the same at all angles -- and
            # accumulated by `atomic_add`. Like `grad` above, they are seeded with zero automatically. )
            code = f"""
            {{
                constexpr SI MAX_DIRS = { MAX_DIRS };
                const SI n = SI( inputs.src.points.shape( 0 ) );
                auto order = scratch.sorted_idx( batch_index );
                auto tmp   = scratch.radix_tmp( batch_index );

                // same sliced view + same LSD radix sort as `diracs_cost_grad` -- see its
                // comments for the details (key+index packet, no recursive comparator).
                auto s = inputs.src( batch_index );
                auto proj = [&]( SI i ) {{ return s.position( i ); }};

                for ( SI i = 0; i < n; ++i ) {{
                    const float f = float( proj( i ) );
                    uint32_t u = __builtin_bit_cast( uint32_t, f );
                    u ^= ( u & 0x80000000u ) ? 0xFFFFFFFFu : 0x80000000u;
                    order( i ) = ( SI( u >> 1 ) << 32 ) | SI( i );
                }}

                constexpr int NB_BITS    = 8;
                constexpr int NB_BUCKETS = 1 << NB_BITS;
                constexpr int NB_PASSES  = 32 / NB_BITS;
                static_assert( NB_PASSES % 2 == 0 );

                auto radix_pass = [&]( auto &&from, auto &&to, int shift ) {{
                    SI count[ NB_BUCKETS ] = {{ 0 }};
                    for ( SI i = 0; i < n; ++i )
                        ++count[ ( SI( from( i ) ) >> shift ) & ( NB_BUCKETS - 1 ) ];
                    SI sum = 0;
                    for ( int bucket = 0; bucket < NB_BUCKETS; ++bucket ) {{
                        const SI c = count[ bucket ];
                        count[ bucket ] = sum;
                        sum += c;
                    }}
                    for ( SI i = 0; i < n; ++i ) {{
                        const SI key = from( i );
                        const int bkt = ( key >> shift ) & ( NB_BUCKETS - 1 );
                        to( count[ bkt ]++ ) = key;
                    }}
                }};
                for ( int p = 0; p < NB_PASSES; ++p ) {{
                    const int shift = 32 + p * NB_BITS;
                    if ( p % 2 == 0 ) radix_pass( order, tmp, shift );
                    else              radix_pass( tmp, order, shift );
                }}

                inputs.dst( batch_index ).with_defaults( [&]( auto &&img ) {{
                    using TF = DECAYED_TYPE_OF( img.values )::TF;
                    const TF w = TF( 1 ) / TF( n );

                    // same walk as `diracs_cost_grad`, but ONLY `first_moment()` is read (not
                    // `w2_dist` -- no cost to compute here): `img.udp_start()`/`udp_cont()`
                    // are `const` on `img` (never mutated), so this FRESH and
                    // independent walk behaves identically to that of `diracs_cost_grad`.
                    auto udp = img.udp_start();
                    for ( SI k = 0; k < n; ++k ) {{
                        const SI di = order( k ) & 0xFFFFFFFFll;
                        const TF p = proj( di );

                        TF moment = 0;
                        img.udp_cont( udp, w, [&]( auto &&item ) {{
                            moment += item.first_moment();
                        }} );
                        const TF bary = moment / w;
                        const TF grad_s = TF( 2 ) * w * ( p - bary );

                        // projection of each stored direction onto the normal of THIS angle, at
                        // THIS dirac -- `e_i = inputs.directions[i][di]·normal`, the affine coefficient of
                        // `p(a)` in `a_i` (see the function docstring).
                        TF e[ MAX_DIRS ];
                        for ( SI i = 0; i < MAX_DIRS; ++i )
                            e[ i ] = TF( inputs.directions( dir_index_i = i, num_dirac = di, proj_dim = 0 ) ) * TF( s.normal( proj_dim = 0 ) )
                                   + TF( inputs.directions( dir_index_i = i, num_dirac = di, proj_dim = 1 ) ) * TF( s.normal( proj_dim = 1 ) );

                        for ( SI i = 0; i < MAX_DIRS; ++i ) {{
                            atomic_add( outputs.b( dir_index_i = i ).ref(), grad_s * e[ i ] );
                            for ( SI j = 0; j < MAX_DIRS; ++j )
                                atomic_add( outputs.H( dir_index_i = i, dir_index_j = j ).ref(), TF( 2 ) * w * e[ i ] * e[ j ] );
                        }}
                    }}
                }} );
            }}
            """,
        ),
        src = src,
        dst = dst,
        directions = directions_t,
        sorted_idx = loom.scratch( sorted_idx ),
        radix_tmp = loom.scratch( radix_tmp ),
        H = loom.out( H ),
        b = loom.out( b ),
        has_dynamic_capacity = False,
    )

    return np.asarray( H.value ), np.asarray( b.value )
