"""Vérifie le modèle DISQUES de `HcReconstruction` (`use_disks`), sur les deux backends.

`use_disks( radius, shape = "disk" | "triangle" )` : le backend `jax` sait projeter les DEUX
formes (grille `nb_pixels`, même maths que `disks.DiskProjector`/`models.DiskModel` pour "disk" ;
profil triangulaire fermé pour "triangle", voir `_triangle_mass_angle`) -- exprès, pour pouvoir
comparer `jax(shape="triangle")` et `sycl` sur EXACTEMENT le même profil cible. Le backend `sycl`
ne sait balayer QUE "triangle" (noyau CONTINU, pas de grille -- voir `hc_ot_sycl.cpp`) ; demander
`shape="disk"` avec `backend="sycl"` doit lever une erreur claire à `_init`.

- `jax` + `shape="disk"` : comparé directement au coût `models.DiskModel` (loom/sdot) sur la même
  géométrie, et à un gradient par différences finies.
- `sycl` (+ `shape="triangle"` implicite) : vérifié contre un gradient par différences finies ET
  contre `_ot1d_disks_angle` (jax) évalué sur une discrétisation TRÈS fine du même profil
  triangulaire (référence indépendante du balayage C++), qui doit converger vers la même valeur.
- `jax` + `shape="triangle"` vs `sycl` : les deux calculent maintenant le MÊME modèle (profil
  triangulaire) par deux chemins indépendants (grille fine vs balayage continu exact) -- doivent
  s'accorder à la résolution de grille près.

Voir [[HcReconstruction disks model]] pour le contexte : ces deux chemins reprennent, en dehors
de loom/sdot, le même principe que `models.DiskModel` -- le sinogramme mesuré devient des diracs
pondérés fixes, et les points (centres de disques) paramètrent la cible continue.
"""
import numpy as np

from otrec.Sinogram import Sinogram
from otrec.models import DiskModel
from otrec.HcReconstruction import HcReconstruction
from loom.testing import test


def _phantom_sinogram(nb_angles=24, nb_bins=150, extent=8.0):
    s = Sinogram(nb_angles=nb_angles, nb_bins=nb_bins, extent=extent)
    s.add_disk([0.4, -0.2], 1.6, density=1.0)
    s.add_disk([-0.8, 0.6], 0.7, density=-0.3)
    return s


def _finite_diff_grad(loss_only, centers, eps=1e-3):
    fd = np.zeros_like(centers, dtype=np.float64)
    for i in range(centers.shape[0]):
        for d in range(2):
            cp = centers.copy(); cp[i, d] += eps
            cm = centers.copy(); cm[i, d] -= eps
            fd[i, d] = (loss_only(cp.astype(np.float32)) - loss_only(cm.astype(np.float32))) / (2 * eps)
    return fd


if test("hc_disks_jax_matches_loom_sdot_diskmodel"):
    nb_angles, nb_bins, extent = 24, 150, 8.0
    radius = 0.5
    sino = _phantom_sinogram(nb_angles, nb_bins, extent)
    rng = np.random.default_rng(0)
    centers = (rng.random((10, 2)) - 0.5) * extent * 0.6

    dm = DiskModel(sino, radius=radius, nb_pixels=nb_bins)
    cost_ref = float(dm.cost(centers))

    hc = HcReconstruction(nb_angles=nb_angles, nb_bins=nb_bins, extent=extent, backend="jax")
    hc.values = np.asarray(sino.values, dtype=np.float32)
    hc.use_disks(radius=radius, nb_pixels=nb_bins)

    cost_hc, grad_hc = hc.loss_grad(centers.astype(np.float32))

    assert np.isfinite(cost_hc)
    assert abs(cost_hc - cost_ref) < 1e-5 * max(1.0, abs(cost_ref)), \
        f"coût jax { cost_hc } != coût DiskModel { cost_ref }"

    fd = _finite_diff_grad(hc._loss_only, centers)
    assert np.allclose(grad_hc, fd, atol=1e-2, rtol=1e-2), \
        f"gradient jax != différences finies, écart max { np.max(np.abs(grad_hc - fd)) }"


if test("hc_disks_jax_floor_and_model_switch"):
    # `use_diracs`/`use_disks` doivent (dé)geler `_ready` correctement, et TOUS les
    # `line_search_*` (génériques sur `loss_grad`/`_loss_only`) doivent fonctionner sans
    # modification pour le modèle disques -- on n'en teste qu'un sous-ensemble ici (fumée).
    hc = HcReconstruction(nb_angles=16, nb_bins=80, extent=6.0, backend="jax")
    sino = _phantom_sinogram(16, 80, 6.0)
    hc.values = np.asarray(sino.values, dtype=np.float32)

    assert hc.floor == 0.0
    hc.use_disks(radius=0.4)
    assert hc.floor > 0.0

    pts = hc.random_points(20, seed=0)
    p1 = hc.line_search(pts.copy(), max_iter=3, verbose=False)
    p2 = hc.line_search_lbfgs(pts.copy(), max_iter=3, verbose=False)
    p3 = hc.line_search_quad2d(pts.copy(), max_iter=3, verbose=False)
    for p in (p1, p2, p3):
        assert p.shape == pts.shape
        assert np.all(np.isfinite(p))

    hc.use_diracs()
    assert hc.floor == 0.0
    p4 = hc.line_search(pts.copy(), max_iter=3, verbose=False)
    assert np.all(np.isfinite(p4))


if test("hc_disks_sycl_matches_finite_difference"):
    nb_angles, nb_bins, extent = 5, 220, 10.0
    radius = 0.6
    rng = np.random.default_rng(7)
    ndisks = 11
    centers = (rng.random((ndisks, 2)) - 0.5) * extent * 0.6

    hc = HcReconstruction(nb_angles=nb_angles, nb_bins=nb_bins, extent=extent, backend="sycl")
    hc.values = (rng.random((nb_angles, nb_bins)).astype(np.float32) + 0.05)
    hc.use_disks(radius=radius, shape="triangle")

    cost_sycl, grad_sycl = hc.loss_grad(centers.astype(np.float32))
    assert np.isfinite(cost_sycl)
    assert np.all(np.isfinite(grad_sycl))

    fd = _finite_diff_grad(hc._loss_only, centers, eps=3e-3)
    assert np.allclose(grad_sycl, fd, atol=2e-3, rtol=5e-2), \
        f"gradient sycl != différences finies, écart max { np.max(np.abs(grad_sycl - fd)) }"


if test("hc_disks_sycl_matches_fine_grid_reference"):
    # Référence INDÉPENDANTE du balayage C++ : construit le même profil triangulaire (somme de
    # tentes) sur une grille très fine et calcule le coût OT1D directement (recherche de
    # cumulée, en float64 -- pas `_ot1d_disks_angle`/jax qui perd trop de précision en float32
    # à cette résolution). Doit converger vers la valeur du noyau continu.
    nb_angles, nb_bins, extent = 1, 300, 10.0
    radius = 0.7
    rng = np.random.default_rng(3)
    ndisks = 9
    centers = (rng.random((ndisks, 2)) - 0.5) * extent * 0.6

    hc = HcReconstruction(nb_angles=nb_angles, nb_bins=nb_bins, extent=extent, backend="sycl")
    hc.values = (rng.random((nb_angles, nb_bins)).astype(np.float32) + 0.1)
    hc.use_disks(radius=radius, shape="triangle")

    cost_sycl = hc._loss_only(centers.astype(np.float32))

    nx, ny = hc.normals[0]
    s0 = centers[:, 0] * nx + centers[:, 1] * ny
    M = ndisks * radius
    sino_row = np.asarray(hc.values[0], dtype=np.float64)
    w_src = sino_row / sino_row.sum()
    q_src = np.cumsum(w_src)
    bin_centers = hc.bin_centers

    lo, hi = (s0 - radius).min(), (s0 + radius).max()
    pad = 0.05 * (hi - lo)
    n_fine = 2_000_000
    u = np.linspace(lo - pad, hi + pad, n_fine)
    f = np.sum(np.maximum(0.0, 1 - np.abs(u[None, :] - s0[:, None]) / radius), axis=0)
    du = u[1] - u[0]
    F = np.cumsum(f) * du
    t = F / M
    idx = np.searchsorted(q_src, t, side="right").clip(0, nb_bins - 1)
    T = bin_centers[idx]
    g = (u - T) ** 2
    cost_ref = np.trapezoid(g * f, u) / M

    assert abs(cost_sycl - cost_ref) < 1e-3 * max(1.0, abs(cost_ref)), \
        f"coût sycl { cost_sycl } != référence fine-grid { cost_ref }"


if test("hc_disks_jax_disk_vs_triangle_differ"):
    # `shape` doit réellement changer le modèle -- filet de sécurité contre un dispatch qui
    # ignorerait silencieusement le paramètre (les deux formes n'ont aucune raison de tomber sur
    # le même coût sur des données non triviales).
    nb_angles, nb_bins, extent = 16, 100, 6.0
    radius = 0.4
    sino = _phantom_sinogram(nb_angles, nb_bins, extent)
    rng = np.random.default_rng(5)
    centers = (rng.random((8, 2)) - 0.5) * extent * 0.6

    hc = HcReconstruction(nb_angles=nb_angles, nb_bins=nb_bins, extent=extent, backend="jax")
    hc.values = np.asarray(sino.values, dtype=np.float32)

    hc.use_disks(radius=radius, shape="disk")
    cost_disk = hc._loss_only(centers.astype(np.float32))

    hc.use_disks(radius=radius, shape="triangle")
    cost_triangle = hc._loss_only(centers.astype(np.float32))

    assert np.isfinite(cost_disk) and np.isfinite(cost_triangle)
    assert abs(cost_disk - cost_triangle) > 1e-3 * max(1.0, abs(cost_disk))


if test("hc_disks_jax_triangle_matches_sycl_triangle"):
    # Même profil (triangle) des deux côtés désormais -- doivent s'accorder à la résolution de
    # grille jax près (grossière ici pour rester rapide, d'où une tolérance large).
    nb_angles, nb_bins, extent = 12, 180, 8.0
    radius = 0.5
    rng = np.random.default_rng(11)
    ndisks = 7
    centers = (rng.random((ndisks, 2)) - 0.5) * extent * 0.6
    values = (rng.random((nb_angles, nb_bins)).astype(np.float32) + 0.1)

    hc_jax = HcReconstruction(nb_angles=nb_angles, nb_bins=nb_bins, extent=extent, backend="jax")
    hc_jax.values = values
    hc_jax.use_disks(radius=radius, shape="triangle", nb_pixels=4000)
    cost_jax = hc_jax._loss_only(centers.astype(np.float32))

    hc_sycl = HcReconstruction(nb_angles=nb_angles, nb_bins=nb_bins, extent=extent, backend="sycl")
    hc_sycl.values = values
    hc_sycl.use_disks(radius=radius, shape="triangle")
    cost_sycl = hc_sycl._loss_only(centers.astype(np.float32))

    assert abs(cost_jax - cost_sycl) < 5e-2 * max(1.0, abs(cost_sycl)), \
        f"coût jax(triangle) { cost_jax } != coût sycl(triangle) { cost_sycl }"
