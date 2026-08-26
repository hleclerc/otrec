"""Compare convergence of different optimization algorithms."""

import numpy as np
import time
from pathlib import Path

from .Reconstruction import Reconstruction
from .Sinogram import Sinogram
from .optimizers import GradientDescent, LBFGS


def test_convergence(sinogram: Sinogram, nb_diracs: int = 50, seed: int = 0):
    """Test and compare convergence of different optimizers.

    Returns:
        dict with convergence metrics for each optimizer
    """
    print(f"Testing convergence with {nb_diracs} diracs...")
    # un même nuage de départ, puis une `Reconstruction` par optimiseur (elles sont indépendantes :
    # chacune porte son propre nuage courant)
    positions_init = Reconstruction(sinogram, extent=1.0).random_points(nb_diracs, seed=seed).points
    initial_loss = Reconstruction(sinogram, positions_init).loss()

    results = {}

    # Test Gradient Descent
    print("\n=== Gradient Descent (lr=0.2, 500 steps) ===")
    gd_losses = []

    rec_gd = Reconstruction(sinogram, positions_init)

    def gd_callback(step, pos):
        l = rec_gd.loss(points=pos)
        gd_losses.append(l)
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}: loss = {l:.6f}")

    t0 = time.time()
    rec_gd.diracs(optimizer=GradientDescent(lr=0.2, nb_steps=500), callback=gd_callback)
    gd_time = time.time() - t0
    final_loss_gd = rec_gd.loss()

    results['gradient_descent'] = {
        'positions': rec_gd.points,
        'losses': gd_losses,
        'final_loss': final_loss_gd,
        'time': gd_time,
        'nb_steps': len(gd_losses)
    }

    print(f"Final loss: {final_loss_gd:.6f} (improved by {100 * (1 - final_loss_gd / initial_loss):.1f}%)")
    print(f"Time: {gd_time:.2f}s")

    # Test LBFGS
    print("\n=== L-BFGS (max_iter=200) ===")
    lbfgs_losses = []

    rec_lbfgs = Reconstruction(sinogram, positions_init)

    def lbfgs_callback(step, pos):
        l = rec_lbfgs.loss(points=pos)
        lbfgs_losses.append(l)
        if (step + 1) % 10 == 0:
            print(f"  Iteration {step + 1}: loss = {l:.6f}")

    t0 = time.time()
    rec_lbfgs.diracs(optimizer=LBFGS(max_iter=200, ftol=1e-8), callback=lbfgs_callback)
    lbfgs_time = time.time() - t0
    final_loss_lbfgs = rec_lbfgs.loss()

    results['lbfgs'] = {
        'positions': rec_lbfgs.points,
        'losses': lbfgs_losses,
        'final_loss': final_loss_lbfgs,
        'time': lbfgs_time,
        'nb_steps': len(lbfgs_losses)
    }

    print(f"Final loss: {final_loss_lbfgs:.6f} (improved by {100 * (1 - final_loss_lbfgs / initial_loss):.1f}%)")
    print(f"Time: {lbfgs_time:.2f}s")

    # Summary
    print("\n=== Summary ===")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"\nGradient Descent: {len(gd_losses)} steps, {gd_time:.2f}s, final loss {final_loss_gd:.6f}")
    print(f"L-BFGS:          {len(lbfgs_losses)} steps, {lbfgs_time:.2f}s, final loss {final_loss_lbfgs:.6f}")

    speedup = gd_time / lbfgs_time if lbfgs_time > 0 else float('inf')
    improvement = (final_loss_gd - final_loss_lbfgs) / final_loss_gd * 100 if final_loss_gd > 0 else 0
    print(f"\nL-BFGS speedup: {speedup:.1f}x")
    print(f"L-BFGS loss improvement: {improvement:.1f}%")

    return results


def save_convergence_plot(results, output_path: str = "convergence.txt"):
    """Save convergence curves as text table."""
    max_steps = max(len(r['losses']) for r in results.values())

    with open(output_path, 'w') as f:
        f.write("Step\tGD Loss\t\tLBFGS Loss\n")
        for step in range(max_steps):
            gd_loss = results['gradient_descent']['losses'][step] if step < len(results['gradient_descent']['losses']) else None
            lbfgs_loss = results['lbfgs']['losses'][step] if step < len(results['lbfgs']['losses']) else None

            gd_str = f"{gd_loss:.6f}" if gd_loss is not None else "-"
            lbfgs_str = f"{lbfgs_loss:.6f}" if lbfgs_loss is not None else "-"
            f.write(f"{step}\t{gd_str}\t\t{lbfgs_str}\n")

    print(f"\nConvergence curves saved to {output_path}")


if __name__ == "__main__":
    # Example usage: create a simple test sinogram
    from .Sinogram import Sinogram

    # You would need to create a real Sinogram here
    # For now, this is a template
    print("Run this script with a proper Sinogram instance")
