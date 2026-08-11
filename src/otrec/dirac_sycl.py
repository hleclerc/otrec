"""Référence SYCL (CPU pour l'instant) du coût + gradient du modèle DIRACS (`models.DiracModel`),
en UN SEUL `driver.call` fwd-only -- sans `bwd_code`, donc sans passer par l'autodiff Jax : le
gradient est écrit directement par le kernel, formule fermée `(point - barycentre) * direction`,
au lieu d'un `jax.grad` à travers `_pure_jax_cost1d.cost_1d_ot` (le chemin par défaut,
`Image.try_update_otplan1d`) ou du couple fwd/bwd `OtPlan1d.cxx` (le chemin C++ général).

Motivation : comparer la vitesse d'un noyau qui fusionne les DEUX passes (coût ET gradient, la
même formule que `OtPlan1d.cxx::sweep_outputs_bwd`'s barycentre-recompute branch, mais calculé
UNE FOIS au lieu de deux) contre le pipeline pur Jax actuellement utilisé par
`Reconstruction.diracs`. Volontairement plus simple que `OtPlan1d` : un seul work-item par angle
(même tri radix LSD par paquet clé+index que `OtPlan1d.cxx::sort_diracs`, mais SANS sa coopération
de groupe -- un seul thread suffit puisqu'il traite tout l'angle -- et pas la marche
`udp_at`/`cell_cum_mass` -- un simple `udp_start` séquentiel suffit pour la même raison).
Repousser la coopération de groupe à une étape ultérieure si cette référence s'avère prometteuse
à plus grande échelle.

Réutilise `ProjectedSumOfDiracs`/`Image` TELS QUELS (mêmes structs C++, mêmes méthodes
`position(i)`/`udp_start`/`udp_cont` que `OtPlan1d.cxx`) -- seule l'orchestration (tri + balayage
+ dispersion du gradient) est nouvelle.

S'est avéré assez prometteur (10-30x plus rapide que le chemin Jax mesuré sur le poumon, voir
`benchmarks/execution_speed/benchmark_fused.py`) pour être BRANCHÉ : `models.DiracModel.value_and_grad`
expose `diracs_cost_grad` avec le même contrat que `optimizers.FusedLBFGS` attend, et
`Reconstruction.diracs( backend = "sycl" )` route dessus (voir `experiments/lung_alveoli.py`).
"""
import numpy as np

from loom import Tensor, Axis, CtShapeVar, driver
from loom.compilation.FfiCode import FfiCodeParallel
from sdot.distributions.ProjectedSumOfDiracs import ProjectedSumOfDiracs

from .Sinogram import Sinogram

# nombre max de directions du sous-espace de `subspace_hessian` -- voir sa docstring. Fixe (pas un
# `ShapeVar` runtime) : un seul noyau compilé sert tous les appels, `optimizers.SubspaceNewtonLBFGS`
# zero-paddant au-delà du nombre de directions réellement stockées.
MAX_DIRS = 5


def diracs_cost_grad( points, sinogram: Sinogram ):
    """`(cost, grad)` du modèle DIRACS pour `points` (`[n,2]`, Tensor ou tableau) face à
    `sinogram`, calculés par le kernel SYCL fusionné ci-dessus -- `cost` un flottant Python,
    `grad` un tableau numpy `[n,2]` (même convention de signe que `jax.grad(model.cost)`, un pas
    de descente de gradient fait donc `points - lr * grad`).
    """
    pts = points if isinstance( points, Tensor ) else Tensor( points )

    # `src`/`dst` normalisés à masse 1 -- exactement ce que fait `OtPlan1d.__init__` avant de
    # balayer (voir sa docstring) : sans ça la marche `udp_cont` (dont les prises `w` par dirac
    # doivent épuiser exactement la masse totale de l'image) ne balaierait pas l'image en entier.
    # `src` n'a pas de poids explicites -> `normalized_version()` donnerait des poids UNIFORMES
    # `1 / n`, exactement ce que le kernel calcule lui-même (`w = TF(1)/TF(n)`) -- inutile de
    # matérialiser un tenseur de poids rien que pour ça.
    src = ProjectedSumOfDiracs( points = pts, normal = sinogram.normals_t,
                                batch_axes = [ sinogram.num_angle ] )
    dst = sinogram.batched_image().normalized_version()

    sorted_idx = Tensor[ sinogram.num_angle, src.num_dirac, dict( dtype = int ) ]()
    radix_tmp = Tensor[ sinogram.num_angle, src.num_dirac, dict( dtype = int ) ]()
    cost = Tensor[ sinogram.num_angle ]()
    grad = Tensor[ src.num_dirac, src.proj_dim ]()

    driver.call(
        FfiCodeParallel(
            name = "diracs_fused_cost_grad",
            includes = [ "loom/support/atomic_add.h" ],
            # zéro AVANT la boucle parallèle sur les angles : `grad` est PARTAGÉ (les points sont
            # les mêmes à tous les angles) et accumulé par `atomic_add` -- un buffer FFI fraîchement
            # alloué n'est pas garanti démarrer à zéro (voir `ProjectedSumOfDiracs::zero_position_grad`).
            fwd_setup_code = "grad.fill_with( queue, 0 );",
            fwd_code = """
            {
                const SI n = SI( src.points.shape( 0 ) );
                auto order = sorted_idx( batch_index );
                auto tmp   = radix_tmp( batch_index );

                // vue tranchée à CET angle, construite UNE FOIS -- `src( batch_index )` referait
                // sinon cette résolution (normale par angle, etc.) à CHAQUE comparaison du tri
                // (O(n log n) fois) et à chaque pas de la marche `udp_cont` (O(n) fois).
                auto s = src( batch_index );

                // projection à la volée (pas de tableau [nb_angles, n] matérialisé) : la même
                // méthode que le chemin C++ général, `ProjectedSumOfDiracs::position`.
                auto proj = [&]( SI i ) { return s.position( i ); };

                // Tri radix LSD itératif O(n), PAS de comparateur (pas de récursion -- même
                // contrainte SSCP qui a écarté std::sort/introsort, voir OtPlan1d.cxx::sort_diracs,
                // dont ce bloc est une version simplifiée SANS coopération de groupe, un seul
                // work-item traitant tout l'angle). On PAQUETTE, dans chaque case int64 de `order`,
                // une clé float32 ordonnée (bits IEEE retournés pour préserver l'ordre) dans les
                // bits hauts + l'indice du dirac dans les bits bas, puis on trie CE PAQUET
                // directement (contigu, sans indirection) au lieu de comparer indirectement via
                // `proj( order( i ) )` -- ce qui recalculait le produit scalaire de la projection à
                // CHAQUE comparaison du tri par tas précédent (O(n log n) fois) au lieu d'une seule
                // fois par dirac ici.
                for ( SI i = 0; i < n; ++i ) {
                    const float f = float( proj( i ) );
                    uint32_t u = __builtin_bit_cast( uint32_t, f );
                    u ^= ( u & 0x80000000u ) ? 0xFFFFFFFFu : 0x80000000u;
                    order( i ) = ( SI( u >> 1 ) << 32 ) | SI( i );
                }

                constexpr int NB_BITS    = 8;
                constexpr int NB_BUCKETS = 1 << NB_BITS;
                constexpr int NB_PASSES  = 32 / NB_BITS;
                static_assert( NB_PASSES % 2 == 0 ); // le résultat final doit retomber dans `order`

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

                dst( batch_index ).with_defaults( [&]( auto &&img ) {
                    using TF = DECAYED_TYPE_OF( img.values )::TF;
                    const TF w = TF( 1 ) / TF( n );

                    // un seul work-item pour tout l'angle -> pas besoin de `udp_at`/`cell_cum_mass`
                    // (la marche part du tout début, comme `udp_at( cell_cum_mass, 0 )` le ferait).
                    auto udp = img.udp_start();
                    TF local_cost = 0;
                    for ( SI k = 0; k < n; ++k ) {
                        const SI di = order( k ) & 0xFFFFFFFFll; // décode : bits bas = indice d'origine
                        const TF p = proj( di );

                        // coût ET barycentre (premier moment) EN UNE SEULE marche -- pas de
                        // buffer `barycenters` [nb_angles, n], pas de second passage bwd.
                        TF moment = 0;
                        img.udp_cont( udp, w, [&]( auto &&item ) {
                            local_cost += item.w2_dist( p );
                            moment += item.first_moment();
                        } );
                        const TF b = moment / w;

                        // d cost / d position(i) = 2 w (p - b) ; d position / d point = normal
                        // (voir `ProjectedSumOfDiracs::add_position_grad`) -- même formule, mais
                        // écrite directement dans `grad` (sortie brute) plutôt que via l'indirection
                        // `grad_src` que le protocole de différentiation Jax construirait pour un bwd.
                        const TF grad_s = TF( 2 ) * w * ( p - b );
                        atomic_add( grad( num_dirac = di, proj_dim = 0 ).ref(),
                                    TF( grad_s * TF( s.normal( proj_dim = 0 ) ) ) );
                        atomic_add( grad( num_dirac = di, proj_dim = 1 ).ref(),
                                    TF( grad_s * TF( s.normal( proj_dim = 1 ) ) ) );
                    }
                    cost( batch_index ) = local_cost;
                } );
            }
            """,
        ),
        output_attributes = [ "cost", "grad", "sorted_idx", "radix_tmp" ],
        scratch_attributes = [ "sorted_idx", "radix_tmp" ],
        has_dynamic_capacity = False,
        src = src,
        dst = dst,
        sorted_idx = sorted_idx,
        radix_tmp = radix_tmp,
        cost = cost,
        grad = grad,
    )

    return float( np.asarray( cost.value ).sum() ), np.asarray( grad.value )


def diracs_cost( points, sinogram: Sinogram ):
    """`cost` SEUL du modèle DIRACS -- MÊME formule et MÊME kernel de tri + balayage que
    `diracs_cost_grad` ci-dessus, mais sans le calcul de gradient (pas de moment/barycentre par
    dirac, pas de dispersion atomique) : pour les évaluations "coût seul" d'une recherche de pas
    (voir `experiments.lung_alveoli._parabolic_bracket`), où le gradient serait de toute façon
    jeté. Économise le calcul de `first_moment()` et les `atomic_add` par dirac -- le tri reste
    identique (c'est lui qui domine le coût, voir `otplan1d-kernel-profile`).
    """
    pts = points if isinstance( points, Tensor ) else Tensor( points )

    src = ProjectedSumOfDiracs( points = pts, normal = sinogram.normals_t,
                                batch_axes = [ sinogram.num_angle ] )
    dst = sinogram.batched_image().normalized_version()

    sorted_idx = Tensor[ sinogram.num_angle, src.num_dirac, dict( dtype = int ) ]()
    radix_tmp = Tensor[ sinogram.num_angle, src.num_dirac, dict( dtype = int ) ]()
    cost = Tensor[ sinogram.num_angle ]()

    driver.call(
        FfiCodeParallel(
            name = "diracs_fused_cost_only",
            fwd_code = """
            {
                const SI n = SI( src.points.shape( 0 ) );
                auto order = sorted_idx( batch_index );
                auto tmp   = radix_tmp( batch_index );

                auto s = src( batch_index );
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

                dst( batch_index ).with_defaults( [&]( auto &&img ) {
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
                    cost( batch_index ) = local_cost;
                } );
            }
            """,
        ),
        output_attributes = [ "cost", "sorted_idx", "radix_tmp" ],
        scratch_attributes = [ "sorted_idx", "radix_tmp" ],
        has_dynamic_capacity = False,
        src = src,
        dst = dst,
        sorted_idx = sorted_idx,
        radix_tmp = radix_tmp,
        cost = cost,
    )

    return float( np.asarray( cost.value ).sum() )


def subspace_hessian( points, directions, sinogram: Sinogram ):
    """`(H, b)` -- Hessienne `[MAX_DIRS,MAX_DIRS]` et gradient `[MAX_DIRS]` du modèle DIRACS
    RESTREINT au sous-espace engendré par `directions` (`[MAX_DIRS,n,2]`, zero-paddé au-delà du
    nombre de directions réellement actives -- voir `optimizers.SubspaceNewtonLBFGS`), c-à-d de
    `a -> loss( points + sum_i a_i * directions[i] )` évalués en `a = 0`.

    Même tri + balayage `udp_start`/`udp_cont` que `diracs_cost_grad` (mêmes `sorted_idx`/
    `radix_tmp` scratch, RECALCULÉS ici plutôt que réutilisés en résidus -- ce noyau tourne une
    fois par pas EXTERNE de `SubspaceNewtonLBFGS`, pas par pas scipy interne, donc le O(n)
    supplémentaire par angle est negligeable face au gain -- partager les résidus serait une
    optimisation ultérieure si ce prototype s'avère payant).

    Repose sur la MÊME approximation "assignation figée" que `grad_s = 2w(p-b)` dans
    `diracs_cost_grad` (le barycentre `b` de chaque dirac traité comme constant vis-à-vis de sa
    position) : à assignation fixée, le coût 1D-OT d'un dirac est EXACTEMENT quadratique en sa
    position projetée `p`, et `p` est elle-même AFFINE en `a` (`p(a) = p(0) + sum_i a_i * e_i`,
    `e_i = directions[i][dirac]·normal`). D'où, par angle et par dirac, en UNE SEULE passe :
    `b_i += grad_s * e_i` (gradient du sous-espace) et `H_ij += 2*w*e_i*e_j` (Hessienne EXACTE de
    ce modèle local, PAS une différence finie) -- somme de termes `w*e*e^T` en rang 1, donc `H` est
    PSD par construction (jamais de courbure négative à gérer côté solveur).
    """
    pts = points if isinstance( points, Tensor ) else Tensor( points )
    dirs = directions if isinstance( directions, Tensor ) else Tensor( directions )

    src = ProjectedSumOfDiracs( points = pts, normal = sinogram.normals_t,
                                batch_axes = [ sinogram.num_angle ] )
    dst = sinogram.batched_image().normalized_version()

    # deux axes DISTINCTS (même compte MAX_DIRS partagé) : `H` est carrée [MAX_DIRS,MAX_DIRS], donc
    # ses deux dimensions ne peuvent pas partager UN SEUL objet `Axis` (l'indexation nommée du
    # kernel, `H( dir_index_i = i, dir_index_j = j )`, a besoin de deux noms distincts -- et
    # `Tensor._dim_index` ne saurait pas lever l'ambiguïté entre deux occurrences du même axe).
    max_dirs = CtShapeVar( MAX_DIRS )
    dir_index_i = Axis( max_dirs, name = "dir_index_i" )
    dir_index_j = Axis( max_dirs, name = "dir_index_j" )

    sorted_idx = Tensor[ sinogram.num_angle, src.num_dirac, dict( dtype = int ) ]()
    radix_tmp = Tensor[ sinogram.num_angle, src.num_dirac, dict( dtype = int ) ]()
    directions_t = Tensor[ dir_index_i, src.num_dirac, src.proj_dim ]( dirs )
    H = Tensor[ dir_index_i, dir_index_j ]()
    b = Tensor[ dir_index_i ]()

    driver.call(
        FfiCodeParallel(
            name = "diracs_subspace_hessian",
            includes = [ "loom/support/atomic_add.h" ],
            # zéro AVANT la boucle parallèle sur les angles -- `H`/`b` sont PARTAGÉS (les diracs
            # sont les mêmes à tous les angles) et accumulés par `atomic_add`, voir
            # `diracs_cost_grad.fwd_setup_code` pour la même raison sur `grad`.
            fwd_setup_code = "H.fill_with( queue, 0 ); b.fill_with( queue, 0 );",
            fwd_code = f"""
            {{
                constexpr SI MAX_DIRS = { MAX_DIRS };
                const SI n = SI( src.points.shape( 0 ) );
                auto order = sorted_idx( batch_index );
                auto tmp   = radix_tmp( batch_index );

                // même vue tranchée + même tri radix LSD que `diracs_cost_grad` -- voir ses
                // commentaires pour le détail (paquet clé+indice, pas de comparateur récursif).
                auto s = src( batch_index );
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
                    for ( int b = 0; b < NB_BUCKETS; ++b ) {{
                        const SI c = count[ b ];
                        count[ b ] = sum;
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

                dst( batch_index ).with_defaults( [&]( auto &&img ) {{
                    using TF = DECAYED_TYPE_OF( img.values )::TF;
                    const TF w = TF( 1 ) / TF( n );

                    // même marche que `diracs_cost_grad`, mais SEUL `first_moment()` est lu (pas
                    // `w2_dist` -- pas de coût à calculer ici) : `img.udp_start()`/`udp_cont()`
                    // sont `const` sur `img` (jamais mutées), donc cette marche FRAÎCHE et
                    // indépendante se comporte identiquement à celle de `diracs_cost_grad`.
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

                        // projection de chaque direction stockée sur la normale de CET angle, à
                        // CE dirac -- `e_i = directions[i][di]·normal`, le coefficient affine de
                        // `p(a)` en `a_i` (voir la docstring de la fonction).
                        TF e[ MAX_DIRS ];
                        for ( SI i = 0; i < MAX_DIRS; ++i )
                            e[ i ] = TF( directions( dir_index_i = i, num_dirac = di, proj_dim = 0 ) ) * TF( s.normal( proj_dim = 0 ) )
                                   + TF( directions( dir_index_i = i, num_dirac = di, proj_dim = 1 ) ) * TF( s.normal( proj_dim = 1 ) );

                        for ( SI i = 0; i < MAX_DIRS; ++i ) {{
                            atomic_add( b( dir_index_i = i ).ref(), grad_s * e[ i ] );
                            for ( SI j = 0; j < MAX_DIRS; ++j )
                                atomic_add( H( dir_index_i = i, dir_index_j = j ).ref(), TF( 2 ) * w * e[ i ] * e[ j ] );
                        }}
                    }}
                }} );
            }}
            """,
        ),
        output_attributes = [ "H", "b", "sorted_idx", "radix_tmp" ],
        scratch_attributes = [ "sorted_idx", "radix_tmp" ],
        has_dynamic_capacity = False,
        src = src,
        dst = dst,
        directions = directions_t,
        sorted_idx = sorted_idx,
        radix_tmp = radix_tmp,
        H = H,
        b = b,
    )

    return np.asarray( H.value ), np.asarray( b.value )
