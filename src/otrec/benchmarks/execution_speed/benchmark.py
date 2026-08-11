"""Execution-speed benchmark : temps d'exécution réel d'un problème de reconstruction représentatif.

Pas une comparaison de qualité d'optimiseurs (voir ../optimizers/benchmark.py pour ça) -- ici on
ne regarde que le TEMPS : coût de compilation JIT (premier appel) puis débit en régime permanent
(steady state, une fois le kernel compilé), pour `loss` seule (forward) et pour son gradient
(forward + backward, via `driver.grad`). Le but est de comparer le MÊME problème sur différents
hardwares/backends (CPU local, `lmo` CUDA/CPU -- voir les cibles `bench`/`bench_lmo` de
`.private/Makefile`) avant de chercher des optimisations spécifiques.

Usage direct : `python -m applications.reconstruction.benchmarks.execution_speed.benchmark [options]`
(dans l'env `vfs` -- voir `.private/Makefile`'s `bench`/`bench_lmo`).
"""

import argparse
import time

from loom import driver

from ...Reconstruction import Reconstruction
from ...Sinogram import Sinogram
from ...models import DiracModel


def _sync( x ):
    """Force la complétion d'un calcul device asynchrone (jax/torch) pour un chrono fiable."""
    block_until_ready = getattr( x, "block_until_ready", None )
    if block_until_ready is not None:
        return block_until_ready()
    import numpy as np
    return np.asarray( x )


def _time_steady( func, x, nb_calls: int ):
    """Chronomètre `func` : 1er appel (trace + compile + exécution) séparé du régime permanent.

    Renvoie `( t_compile, t_steady_per_call )`, en secondes.
    """
    t0 = time.perf_counter()
    _sync( func( x ) )
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range( nb_calls ):
        _sync( func( x ) )
    t_steady = ( time.perf_counter() - t0 ) / nb_calls

    return t_compile, t_steady


def benchmark_execution_speed(
    nb_angles: int = 600,
    nb_bins: int = 150,
    nb_diracs: int = 10000,
    extent: float = 6.0,
    nb_calls: int = 10,
    seed: int = 0,
    verbose: bool = True,
):
    """Construit un problème de reconstruction représentatif et chronomètre `loss` et son gradient.

    `DiracModel.cost` est un unique `OtPlan1d` BATCHÉ sur les angles (voir `models.py`) -- ce chrono
    mesure donc le débit du kernel batché réel, pas une boucle Python par angle. On appelle le
    MODÈLE directement (et non une étape de `Reconstruction`) : ici on ne veut chronométrer que le
    coût et son gradient, sans optimiseur autour. Renvoie un dict de timings, imprimé par
    `__main__` sous forme de rapport.
    """
    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    sino.add_disk( center = [ 0.3, -0.2 ], radius = 1.0 )

    positions = Reconstruction( sino, extent = extent ).random_points( nb_diracs, seed = seed ).points.raw

    model = DiracModel( sino )
    model_bary = DiracModel( sino, with_barycenters = True )

    def scalar_loss( p ):
        return model.cost( model.wrap( p ) ).tensor

    # `positions` est le SEUL argument tracé : ce gradient ne porte déjà que sur les positions
    # des diracs, jamais sur le sinogramme (fixe, capturé par la closure). Avec `with_barycenters
    # = True`, le backward d'`OtPlan1d` lit les barycentres stockés au lieu de re-trier + re-
    # balayer chaque angle (voir `OtPlan1d.__init__`'s docstring / [[projected-source-fusion]]) --
    # comparé ci-dessous au cas par défaut (`with_barycenters = False`) pour mesurer le gain.
    def scalar_loss_bary( p ):
        return model_bary.cost( model_bary.wrap( p ) ).tensor

    loss_j = driver.jit( scalar_loss )
    grad_j = driver.jit( driver.grad( scalar_loss ) )
    grad_bary_j = driver.jit( driver.grad( scalar_loss_bary ) )

    if verbose:
        print( f"device={driver.device!r}  framework={driver.framework!r}" )
        print( f"problem: nb_angles={nb_angles}  nb_bins={nb_bins}  nb_diracs={nb_diracs}" )

    t_compile_loss, t_steady_loss = _time_steady( loss_j, positions, nb_calls )
    t_compile_grad, t_steady_grad = _time_steady( grad_j, positions, nb_calls )
    t_compile_grad_bary, t_steady_grad_bary = _time_steady( grad_bary_j, positions, nb_calls )

    # un `OtPlan1d` (taille nb_diracs) par angle est résolu à chaque appel de loss/grad (batché).
    nb_ot_solves = nb_angles

    print( driver.ftype.cpp_name )

    result = {
        "nb_angles": nb_angles, "nb_bins": nb_bins, "nb_diracs": nb_diracs,
        "device": repr( driver.device ), "framework": repr( driver.framework ),
        "nb_calls": nb_calls,
        "t_compile_loss": t_compile_loss, "t_steady_loss": t_steady_loss,
        "t_compile_grad": t_compile_grad, "t_steady_grad": t_steady_grad,
        "t_compile_grad_bary": t_compile_grad_bary, "t_steady_grad_bary": t_steady_grad_bary,
        "ot_solves_per_s_loss": nb_ot_solves / t_steady_loss,
        "ot_solves_per_s_grad": nb_ot_solves / t_steady_grad,
        "ot_solves_per_s_grad_bary": nb_ot_solves / t_steady_grad_bary,
    }

    if verbose:
        print( f"  loss (forward)                        : compile={t_compile_loss:8.3f}s  "
               f"steady={t_steady_loss * 1e3:8.3f}ms/call  ({result['ot_solves_per_s_loss']:10.0f} OT-solves/s)" )
        print( f"  grad (fwd+bwd, recompute barycenters)  : compile={t_compile_grad:8.3f}s  "
               f"steady={t_steady_grad * 1e3:8.3f}ms/call  ({result['ot_solves_per_s_grad']:10.0f} OT-solves/s)" )
        print( f"  grad (fwd+bwd, with_barycenters=True)  : compile={t_compile_grad_bary:8.3f}s  "
               f"steady={t_steady_grad_bary * 1e3:8.3f}ms/call  ({result['ot_solves_per_s_grad_bary']:10.0f} OT-solves/s)" )

    return result


def _parse_args():
    p = argparse.ArgumentParser( description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter )
    p.add_argument( "--nb-angles", type = int, default = 600 )
    p.add_argument( "--nb-bins", type = int, default = 150 )
    p.add_argument( "--nb-diracs", type = int, default = 10000 )
    p.add_argument( "--extent", type = float, default = 6.0 )
    p.add_argument( "--nb-calls", type = int, default = 10, help = "nombre d'appels chronométrés en régime permanent" )
    p.add_argument( "--seed", type = int, default = 0 )
    return p.parse_args()


if __name__ == "__main__":
    driver.ftype = "FP32"
    
    args = _parse_args()
    benchmark_execution_speed(
        nb_angles = args.nb_angles,
        nb_bins = args.nb_bins,
        nb_diracs = args.nb_diracs,
        extent = args.extent,
        nb_calls = args.nb_calls,
        seed = args.seed,
    )
