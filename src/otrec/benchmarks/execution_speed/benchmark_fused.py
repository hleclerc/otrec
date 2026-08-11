"""Compare le chemin PUR JAX (`models.DiracModel.cost` + `jax.grad`, ce que `Reconstruction.diracs`
utilise aujourd'hui) au kernel SYCL fusionné, fwd-only (`dirac_sycl.diracs_cost_grad`, voir sa
docstring) -- même formule (coût + gradient du modèle DIRACS), sur le même problème.

CPU pour l'instant (voir `dirac_sycl.py`) : lancer avec `SDOT_DEVICE=cpu JAX_PLATFORMS=cpu`,
sans quoi la moitié Jax tournerait sur GPU pendant que le kernel SYCL compile pour le CPU (même
piège que `.private/Makefile`'s `D=cpu` pour `bench`).

Usage : `SDOT_DEVICE=cpu JAX_PLATFORMS=cpu python -m applications.reconstruction.benchmarks.execution_speed.benchmark_fused`
"""
import argparse
import time

from loom import driver

from ...Reconstruction import Reconstruction
from ...Sinogram import Sinogram
from ...models import DiracModel
from ...dirac_sycl import diracs_cost_grad


def _sync( x ):
    block_until_ready = getattr( x, "block_until_ready", None )
    if block_until_ready is not None:
        return block_until_ready()
    import numpy as np
    return np.asarray( x )


def _time_steady( func, x, nb_calls: int ):
    t0 = time.perf_counter()
    _sync( func( x ) )
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range( nb_calls ):
        _sync( func( x ) )
    t_steady = ( time.perf_counter() - t0 ) / nb_calls

    return t_compile, t_steady


def benchmark_fused_vs_jax(
    nb_angles: int = 600,
    nb_bins: int = 150,
    nb_diracs: int = 10000,
    extent: float = 6.0,
    nb_calls: int = 10,
    seed: int = 0,
    verbose: bool = True,
):
    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    sino.add_disk( center = [ 0.3, -0.2 ], radius = 1.0 )

    positions = Reconstruction( sino, extent = extent ).random_points( nb_diracs, seed = seed ).points.raw

    # chemin Jax par défaut -- celui de `Reconstruction.diracs`/`optimizers.LBFGS`
    # (`with_barycenters=False`, voir `Reconstruction.dirac_model`).
    model = DiracModel( sino )
    def scalar_loss( p ):
        return model.cost( model.wrap( p ) ).tensor
    grad_j = driver.jit( driver.grad( scalar_loss ) )

    # kernel SYCL fusionné : un seul appel donne (coût, gradient) -- pas de `driver.jit`/`driver.grad`
    # séparés (rien à tracer, la formule est déjà écrite en dur dans le kernel).
    def fused( p ):
        return diracs_cost_grad( p, sino )[ 1 ]

    if verbose:
        print( f"device={driver.device!r}  framework={driver.framework!r}" )
        print( f"problem: nb_angles={nb_angles}  nb_bins={nb_bins}  nb_diracs={nb_diracs}" )

    t_compile_jax, t_steady_jax = _time_steady( grad_j, positions, nb_calls )
    t_compile_sycl, t_steady_sycl = _time_steady( fused, positions, nb_calls )

    result = {
        "nb_angles": nb_angles, "nb_bins": nb_bins, "nb_diracs": nb_diracs,
        "t_compile_jax": t_compile_jax, "t_steady_jax": t_steady_jax,
        "t_compile_sycl": t_compile_sycl, "t_steady_sycl": t_steady_sycl,
    }

    if verbose:
        print( f"  jax.grad  (pur Jax, chemin Reconstruction actuel) : "
               f"compile={t_compile_jax:8.3f}s  steady={t_steady_jax * 1e3:8.3f}ms/call" )
        print( f"  fused     (SYCL fwd-only, formule fermée)         : "
               f"compile={t_compile_sycl:8.3f}s  steady={t_steady_sycl * 1e3:8.3f}ms/call" )
        print( f"  speedup steady state : {t_steady_jax / t_steady_sycl:.2f}x" )

    return result


def _parse_args():
    p = argparse.ArgumentParser( description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter )
    p.add_argument( "--nb-angles", type = int, default = 600 )
    p.add_argument( "--nb-bins", type = int, default = 150 )
    p.add_argument( "--nb-diracs", type = int, default = 10000 )
    p.add_argument( "--extent", type = float, default = 6.0 )
    p.add_argument( "--nb-calls", type = int, default = 10 )
    p.add_argument( "--seed", type = int, default = 0 )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    benchmark_fused_vs_jax(
        nb_angles = args.nb_angles,
        nb_bins = args.nb_bins,
        nb_diracs = args.nb_diracs,
        extent = args.extent,
        nb_calls = args.nb_calls,
        seed = args.seed,
    )
