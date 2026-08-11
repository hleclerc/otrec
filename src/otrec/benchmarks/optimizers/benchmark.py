"""Comprehensive benchmark of optimization algorithms."""

import numpy as np
import time
from pathlib import Path

from sdot import Tensor

# trois points : ce module est sous `benchmarks/optimizers/`, l'app est deux niveaux au-dessus
from ...Reconstruction import Reconstruction
from ...Sinogram import Sinogram
from ...optimizers import GradientDescent, GradientDescentLineSearch, LBFGS, Adam


def benchmark_optimizers(
    nb_angles: int = 100,
    nb_bins: int = 100,
    nb_diracs: int = 10000,
    extent: float = 6.0,
    seed: int = 0,
    verbose: bool = True
):
    """Benchmark all optimizers on a large problem.

    Returns:
        dict with results for each optimizer
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Benchmark: {nb_angles} angles, {nb_bins} bins, {nb_diracs} diracs")
        print(f"{'='*70}")

    # Create sinogram with a disk
    sino = Sinogram(nb_angles=nb_angles, nb_bins=nb_bins, extent=extent)
    sino.add_disk(center=[0.3, -0.2], radius=1.0)

    # Initial positions -- partagées par tous les optimiseurs (chacun repart du MÊME nuage)
    positions_init = Reconstruction(sino, extent=extent).random_points(nb_diracs, seed=seed).points
    initial_loss = Reconstruction(sino, positions_init).loss()

    results = {}

    optimizers_to_test = [
        ("Gradient Descent", GradientDescent(lr=0.2, nb_steps=500)),
        ("GD + Line Search", GradientDescentLineSearch(lr=1.0, nb_steps=300)),
        ("Adam", Adam(lr=0.1, nb_steps=300, grad_clip=10.0)),
        ("L-BFGS", LBFGS(max_iter=200, ftol=1e-8)),
    ]

    for name, optimizer in optimizers_to_test:
        if verbose:
            print(f"\n--- {name} ---")

        losses = []
        start_time = time.time()
        rec = Reconstruction(sino, positions_init, extent=extent)

        def make_callback(name):
            def callback(step, pos):
                l = rec.loss(points=pos)
                losses.append(l)
                if verbose and (step + 1) % max(1, (step + 100) // 20) == 0:
                    elapsed = time.time() - start_time
                    print(f"  Step {step + 1:3d}: loss = {l:.8f} ({elapsed:.1f}s)")
            return callback

        t0 = time.time()
        try:
            rec.diracs(optimizer=optimizer, callback=make_callback(name))
            t_total = time.time() - t0
            final_loss = rec.loss()

            results[name] = {
                'optimizer': optimizer,
                'positions': rec.points,
                'losses': losses,
                'final_loss': final_loss,
                'time': t_total,
                'nb_steps': len(losses),
                'success': True
            }

            improvement = (initial_loss - final_loss) / initial_loss * 100
            if verbose:
                print(f"  ✓ Final loss: {final_loss:.8f}")
                print(f"  ✓ Time: {t_total:.2f}s")
                print(f"  ✓ Improvement: {improvement:.1f}%")
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            results[name] = {
                'optimizer': optimizer,
                'success': False,
                'error': str(e)
            }

    return results, initial_loss


def print_comparison(results_list, problem_specs):
    """Print comparison table of results."""
    print(f"\n\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")

    for (nb_angles, nb_bins, nb_diracs), results, init_loss in results_list:
        print(f"\nProblem: {nb_angles}×{nb_bins}, {nb_diracs} diracs | Initial loss: {init_loss:.6f}")
        print(f"{'Optimizer':<20} {'Steps':<10} {'Time':<10} {'Final Loss':<15} {'Speedup':<10} {'Quality':<10}")
        print("-" * 100)

        times = {name: r['time'] for name, r in results.items() if r['success']}
        if times:
            baseline_time = times[list(times.keys())[0]]

        for name, result in results.items():
            if not result['success']:
                print(f"{name:<20} {'FAILED':<10}")
            else:
                speedup = baseline_time / result['time'] if times else 1.0
                quality_vs_best = result['final_loss'] / min(r['final_loss'] for r in results.values() if r['success'])
                print(f"{name:<20} {result['nb_steps']:<10} {result['time']:<10.2f} {result['final_loss']:<15.8f} {speedup:<10.2f}x {quality_vs_best:<10.2f}x")


def plot_convergence(results, init_loss, output_path: str = "convergence.png"):
    """Plot convergence curves for all optimizers."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"Gradient Descent": "C0", "GD + Line Search": "C1", "Adam": "C2", "L-BFGS": "C3"}

    # Plot 1: Loss vs iteration
    for name, result in results.items():
        if result['success']:
            ax1.semilogy(result['losses'], label=name, linewidth=2, color=colors.get(name))

    ax1.axhline(init_loss, color='gray', linestyle='--', alpha=0.5, label='Initial loss')
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Loss (log scale)", fontsize=12)
    ax1.set_title("Convergence: Loss vs Iteration", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot 2: Loss vs time
    for name, result in results.items():
        if result['success']:
            if len(result['losses']) > 0:
                time_per_step = result['time'] / len(result['losses'])
                time_array = np.cumsum([time_per_step] * len(result['losses']))
                ax2.semilogy(time_array, result['losses'], label=name, linewidth=2, marker='o', markersize=3, color=colors.get(name))

    ax2.axhline(init_loss, color='gray', linestyle='--', alpha=0.5, label='Initial loss')
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Loss (log scale)", fontsize=12)
    ax2.set_title("Convergence: Loss vs Time", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Convergence plot saved to {output_path}")
    plt.close()


def plot_performance_scaling(results_list, output_path: str = "scaling.png"):
    """Plot performance vs problem size."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    problem_sizes = []
    for (nb_angles, nb_bins, nb_diracs), _, _ in results_list:
        problem_sizes.append(nb_diracs)

    colors = {"Gradient Descent": "C0", "GD + Line Search": "C1", "Adam": "C2", "L-BFGS": "C3"}

    # Plot 1: Time vs problem size
    ax = axes[0, 0]
    for name in ["Gradient Descent", "GD + Line Search", "Adam", "L-BFGS"]:
        times = []
        for (_, _, _), results, _ in results_list:
            if name in results and results[name]['success']:
                times.append(results[name]['time'])
            else:
                times.append(None)
        valid = [(s, t) for s, t in zip(problem_sizes, times) if t is not None]
        if valid:
            s, t = zip(*valid)
            ax.loglog(s, t, marker='o', label=name, linewidth=2, markersize=8, color=colors.get(name))

    ax.set_xlabel("Number of Diracs", fontsize=11)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_title("Execution Time vs Problem Size", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Iterations vs problem size
    ax = axes[0, 1]
    for name in ["Gradient Descent", "GD + Line Search", "Adam", "L-BFGS"]:
        steps = []
        for (_, _, _), results, _ in results_list:
            if name in results and results[name]['success']:
                steps.append(results[name]['nb_steps'])
            else:
                steps.append(None)
        valid = [(s, t) for s, t in zip(problem_sizes, steps) if t is not None]
        if valid:
            s, t = zip(*valid)
            ax.loglog(s, t, marker='s', label=name, linewidth=2, markersize=8, color=colors.get(name))

    ax.set_xlabel("Number of Diracs", fontsize=11)
    ax.set_ylabel("Iterations", fontsize=11)
    ax.set_title("Iterations vs Problem Size", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Time per iteration
    ax = axes[1, 0]
    for name in ["Gradient Descent", "GD + Line Search", "Adam", "L-BFGS"]:
        times_per_iter = []
        for (_, _, _), results, _ in results_list:
            if name in results and results[name]['success']:
                t_per_iter = results[name]['time'] / results[name]['nb_steps']
                times_per_iter.append(t_per_iter)
            else:
                times_per_iter.append(None)
        valid = [(s, t) for s, t in zip(problem_sizes, times_per_iter) if t is not None]
        if valid:
            s, t = zip(*valid)
            ax.loglog(s, t, marker='^', label=name, linewidth=2, markersize=8, color=colors.get(name))

    ax.set_xlabel("Number of Diracs", fontsize=11)
    ax.set_ylabel("Time per Iteration (s)", fontsize=11)
    ax.set_title("Time per Iteration vs Problem Size", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Final loss quality
    ax = axes[1, 1]
    for name in ["Gradient Descent", "GD + Line Search", "Adam", "L-BFGS"]:
        losses = []
        for (_, _, _), results, init_loss in results_list:
            if name in results and results[name]['success']:
                rel_loss = results[name]['final_loss'] / init_loss
                losses.append(rel_loss)
            else:
                losses.append(None)
        valid = [(s, l) for s, l in zip(problem_sizes, losses) if l is not None]
        if valid:
            s, l = zip(*valid)
            ax.loglog(s, l, marker='D', label=name, linewidth=2, markersize=8, color=colors.get(name))

    ax.set_xlabel("Number of Diracs", fontsize=11)
    ax.set_ylabel("Final Loss / Initial Loss", fontsize=11)
    ax.set_title("Solution Quality vs Problem Size", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Scaling plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Run benchmarks on multiple problem sizes
    problems = [
        (20, 50, 500),      # Small
        (50, 75, 2500),     # Medium
        (100, 100, 10000),  # Large
    ]

    results_list = []
    # sorties à côté de ce script (applications/reconstruction/benchmarks/results/), pas dans le CWD
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    for nb_angles, nb_bins, nb_diracs in problems:
        results, init_loss = benchmark_optimizers(
            nb_angles=nb_angles,
            nb_bins=nb_bins,
            nb_diracs=nb_diracs
        )
        results_list.append(((nb_angles, nb_bins, nb_diracs), results, init_loss))

        # Plot convergence for the largest problem
        if nb_diracs == 10000:
            plot_convergence(results, init_loss, output_path=str(output_dir / "convergence_10k.png"))

    # Print summary table
    print_comparison(results_list, problems)

    # Plot scaling behavior
    plot_performance_scaling(results_list, output_path=str(output_dir / "scaling.png"))

    print(f"\n✓ All results saved to {output_dir}/")
