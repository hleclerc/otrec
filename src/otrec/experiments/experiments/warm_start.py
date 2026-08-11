"""Warm-start : vérifie que le nuage passe bien d'un optimiseur au suivant.

C'est la propriété qui rend `Reconstruction` chaînable : chaque étape part EXACTEMENT du nuage
laissé par la précédente (ici LBFGS puis Adam ; c'est le même mécanisme qui permet d'enchaîner
diracs -> disques, voir `disks_demo.py --diracs`).
"""

import numpy as np

from ..Reconstruction import Reconstruction
from ..Sinogram import Sinogram
from ..optimizers import LBFGS, Adam

# Create test problem
sino = Sinogram(nb_angles=20, nb_bins=20, extent=6.0)
sino.add_disk(center=[0.3, -0.2], radius=1.0)

rec = Reconstruction(sino, extent=6.0).random_points(100, seed=42)
positions_init = rec.positions
loss_init = rec.loss()

print(f"Initial setup:")
print(f"  positions_init shape: {positions_init.shape}")
print(f"  positions_init dtype: {positions_init.dtype}")
print(f"  initial loss: {loss_init:.8f}")
print(f"  first position: {positions_init[0]}")

# Phase 1: LBFGS (just a few iterations to make positions change)
print(f"\n=== Phase 1: LBFGS ===")
rec.diracs(optimizer=LBFGS(max_iter=3, ftol=1e-8))
positions_after_lbfgs = rec.positions
loss_after_lbfgs = rec.loss()

print(f"After LBFGS:")
print(f"  positions_after_lbfgs shape: {positions_after_lbfgs.shape}")
print(f"  positions_after_lbfgs dtype: {positions_after_lbfgs.dtype}")
print(f"  loss: {loss_after_lbfgs:.8f}")
print(f"  first position: {positions_after_lbfgs[0]}")
print(f"  position change: {np.linalg.norm(positions_after_lbfgs[0] - positions_init[0]):.8f}")

# Phase 2: Adam starting from LBFGS result -- même objet, on enchaîne simplement l'étape
print(f"\n=== Phase 2: Adam (warm-start) ===")
print(f"  Expected initial loss for Adam: {loss_after_lbfgs:.8f}")

callback_losses = []
def track_callback(step, pos):
    l = rec.loss(points=pos)
    callback_losses.append((step, l))
    if step == -1:
        print(f"  Callback initial loss (step=-1): {l:.8f}")
    elif step == 0:
        print(f"  Callback step 0 loss: {l:.8f}")

rec.diracs(optimizer=Adam(lr=0.1, nb_steps=3, grad_clip=10.0), callback=track_callback)
positions_after_adam = rec.positions
loss_after_adam = rec.loss()

print(f"After Adam:")
print(f"  positions_after_adam shape: {positions_after_adam.shape}")
print(f"  loss: {loss_after_adam:.8f}")
print(f"  first position: {positions_after_adam[0]}")

# Diagnostics
print(f"\n=== Diagnostics ===")
print(f"Loss trajectory:")
print(f"  Initial:      {loss_init:.8f}")
print(f"  After LBFGS:  {loss_after_lbfgs:.8f} (reduction: {(loss_init - loss_after_lbfgs):.8f})")
print(f"  After Adam:   {loss_after_adam:.8f} (reduction: {(loss_after_lbfgs - loss_after_adam):.8f})")

if callback_losses:
    print(f"\nCallback losses:")
    for step, l in callback_losses[:5]:
        print(f"  Step {step}: {l:.8f}")

# l'historique de `Reconstruction` dit la même chose, une ligne par étape
print(f"\n{rec.summary()}")

# Check if positions actually changed
pos_diff = np.linalg.norm(positions_after_adam - positions_after_lbfgs)
print(f"\nPosition norm difference (Adam vs LBFGS): {pos_diff:.8e}")

if pos_diff < 1e-6 and loss_after_adam >= loss_after_lbfgs:
    print("❌ WARNING: Positions did NOT change in Adam phase!")
else:
    print("✓ Positions changed correctly")
