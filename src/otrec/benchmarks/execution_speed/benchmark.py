"""Execution-speed benchmark: real execution time of a representative reconstruction problem.

Not an optimizer quality comparison (see ../optimizers/benchmark.py for that) -- here we
only look at TIME: JIT compilation cost (first call) then steady-state throughput
(once the kernel is compiled), for `loss` alone (forward) and for its gradient
(forward + backward, via `driver.grad`). The goal is to compare the SAME problem on different
hardware/backends (local CPU, `lmo` CUDA/CPU -- see the `bench`/`bench_lmo` targets of
`.private/Makefile`) before looking for specific optimizations.

Direct usage: `python -m applications.reconstruction.benchmarks.execution_speed.benchmark [options]`
(in the `vfs` env -- see `.private/Makefile`'s `bench`/`bench_lmo`).
"""

import argparse
import time

from loom import driver

from ...Reconstruction import Reconstruction
from ...Sinogram import Sinogram
from ...models import DiracModel


def _sync( x ):
    """Force completion of an asynchronous device computation (jax/torch) for reliable timing."""
    block_until_ready = getattr( x, "block_until_ready", None )
    if block_until_ready is not None:
        return block_until_ready()
    import numpy as np
    return np.asarray( x )


def _time_steady( func, x, nb_calls: int ):
    """Time `func`: first call (trace + compile + execution) kept separate from steady state.

    Returns `( t_compile, t_steady_per_call )`, in seconds.
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
    """Build a representative reconstruction problem and time `loss` and its gradient.

    `DiracModel.cost` is a single `SdotPlan1d` BATCHED over the angles (see `models.py`) -- this timing
    therefore measures the throughput of the real batched kernel, not a Python loop per angle. We call the
    MODEL directly (rather than a `Reconstruction` step): here we only want to time the
    cost and its gradient, with no optimizer around. Returns a dict of timings, printed by
    `__main__` as a report.
    """
    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    sino.add_disk( center = [ 0.3, -0.2 ], radius = 1.0 )

    positions = Reconstruction( sino, extent = extent ).random_points( nb_diracs, seed = seed ).points.raw

    model = DiracModel( sino )
    model_bary = DiracModel( sino, with_barycenters = True )

    def scalar_loss( p ):
        return model.cost( model.wrap( p ) ).value

    # `positions` is the ONLY traced argument: this gradient is already only w.r.t. the positions
    # of the diracs, never w.r.t. the sinogram (fixed, captured by the closure). With `with_barycenters
    # = True`, the backward of `SdotPlan1d` reads the stored barycenters instead of re-sorting + re-
    # sweeping each angle (see `SdotPlan1d.__init__`'s docstring / [[projected-source-fusion]]) --
    # compared below with the default case (`with_barycenters = False`) to measure the gain.
    def scalar_loss_bary( p ):
        return model_bary.cost( model_bary.wrap( p ) ).value

    loss_j = driver.jit( scalar_loss )
    grad_j = driver.jit( driver.grad( scalar_loss ) )
    grad_bary_j = driver.jit( driver.grad( scalar_loss_bary ) )

    if verbose:
        print( f"device={driver.device!r}  framework={driver.framework!r}" )
        print( f"problem: nb_angles={nb_angles}  nb_bins={nb_bins}  nb_diracs={nb_diracs}" )

    t_compile_loss, t_steady_loss = _time_steady( loss_j, positions, nb_calls )
    t_compile_grad, t_steady_grad = _time_steady( grad_j, positions, nb_calls )
    t_compile_grad_bary, t_steady_grad_bary = _time_steady( grad_bary_j, positions, nb_calls )

    # one `SdotPlan1d` (size nb_diracs) per angle is solved at each loss/grad call (batched).
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
    p.add_argument( "--nb-calls", type = int, default = 10, help = "number of timed steady-state calls" )
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
