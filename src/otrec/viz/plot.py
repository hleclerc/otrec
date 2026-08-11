"""Comprehensive benchmark of optimization algorithms."""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import time

from sdot import Tensor

from ..optimizers import GradientDescent, GradientDescentLineSearch, LBFGS, Adam
from ..Reconstruction import Reconstruction
from ..Sinogram import Sinogram


def plot(
    nb_angles: int = 200,
    nb_bins: int = 200,
    nb_diracs: int = 1000,
    extent: float = 6.0,
    seed: int = 0,
    verbose: bool = True
):

    # Create sinogram with a disk
    sino = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    sino.add_disk( center = [ 0.3, -0.2 ], radius = 1.0 )

    # Initial positions
    rec = Reconstruction( sino, extent = extent ).random_points( nb_diracs, seed = seed )
    initial_loss = rec.loss()

    results = {}

    print(f"\nPhase 1: LBFGS")
    print(f"  Initial loss: {initial_loss:.8f}")

    losses_lbfgs = []
    start_time_lbfgs = time.time()
    def callback_lbfgs(step, pos):
        l = rec.loss( points = pos )
        losses_lbfgs.append( l )
        if verbose and (step + 1) % max(1, (step + 100) // 20) == 0:
            elapsed = time.time() - start_time_lbfgs
            print(f"  Step {step + 1:3d}: loss = {l:.8f} ({elapsed:.1f}s)")

    rec.diracs( optimizer = LBFGS( max_iter = 200, ftol = 1e-10 ), callback = callback_lbfgs )
    loss_after_lbfgs = rec.loss()
    print(f"  Loss after LBFGS: {loss_after_lbfgs:.8f}")

    # print(f"\nPhase 2: Adam (warm-start from LBFGS)")
    # print(f"  Initial loss for Adam: {loss_after_lbfgs:.8f}")

    # losses_adam = []
    # start_time_adam = time.time()
    # def callback_adam(step, pos):
    #     l = float( loss( sino, pos ).tensor )
    #     losses_adam.append( l )
    #     if verbose and (step + 1) % max(1, (step + 100) // 20) == 0:
    #         elapsed = time.time() - start_time_adam
    #         print(f"  Step {step + 1:3d}: loss = {l:.8f} ({elapsed:.1f}s)")

    # rec.diracs( optimizer = Adam( lr = 0.1, nb_steps = 300, grad_clip = 10.0 ), callback = callback_adam )
    # final_loss = rec.loss()
    #
    plt.plot( rec.positions[ :, 0 ], rec.positions[ :, 1 ], '.' )
    plt.show()

plot()
