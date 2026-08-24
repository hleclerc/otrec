"""Vérifie le modèle POLYGONE de `HcReconstruction` (`use_polygon`, jax only).

Contrairement à `use_disks(shape="triangle")` -- un PROFIL de densité radiale sur un support
circulaire (voir `cost.jax_disks`'s docstring) --, `use_polygon(n_sides, radius)` modélise un
VRAI polygone régulier : chaque point est `[x, y, theta]`, `theta` étant une variable
d'optimisation à part entière (voir `cost.jax_polygon`). La silhouette dépend donc de l'angle de
projection, contrairement au cas disque/triangle-profil (silhouette toujours circulaire).

Le coeur des maths (`_polygon_mass_angle`, une primitive close par arête via le théorème de
Green) a été dérivé, testé, et un vrai bug de signe (mauvais sens de clip selon le signe de `dx`
le long d'une arête) trouvé et corrigé via un cross-check numérique -- voir le premier test
ci-dessous, qui reprend ce cross-check comme garde-fou permanent.
"""
import numpy as np
import jax.numpy as jnp

from otrec.HcReconstruction import HcReconstruction
from otrec.HcReconstruction.cost.jax_polygon import _polygon_mass_angle
from loom.testing import test


def _mc_polygon_mass(cx, cy, radius, theta, n_sides, nx, ny, pix_edges, rng, nsamp=2_000_000):
    """Référence indépendante : rasterise le polygone (test point-in-convex-polygon par
    demi-plans) et histogramme sa projection sur (nx, ny) -- même convention MASSE (pas densité)
    que `_polygon_mass_angle` (`mass = H[1:] - H[:-1]`, pas de division par la largeur de bin)."""
    box = radius * 1.05
    pts = rng.uniform(-box, box, size=(nsamp, 2))
    k = np.arange(n_sides)
    gamma = theta + 2 * np.pi * k / n_sides
    verts = np.stack([radius * np.cos(gamma), radius * np.sin(gamma)], axis=1)
    inside = np.ones(nsamp, dtype=bool)
    for i in range(n_sides):
        a, b = verts[i], verts[(i + 1) % n_sides]
        edge = b - a
        normal = np.array([-edge[1], edge[0]])
        inside &= ((pts - a) @ normal) >= 0
    proj = (pts[:, 0] + cx) * nx + (pts[:, 1] + cy) * ny
    hist, _ = np.histogram(proj[inside], bins=pix_edges)
    cell_area = (2 * box) ** 2 / nsamp
    return hist * cell_area


if test("hc_polygon_projection_matches_reference"):
    rng = np.random.default_rng(0)
    pix_edges = np.linspace(-3.5, 3.5, 141)
    for n_sides, radius, cx, cy, theta, angle in [
        (3, 2.0, 0.3, -0.6, 0.7, 1.1),
        (5, 1.3, -0.5, 0.2, -0.2, 2.3),
        (4, 1.0, 0.0, 0.0, 0.4, 0.0),
    ]:
        nx, ny = np.cos(angle), np.sin(angle)
        points = jnp.asarray([[cx, cy, theta]], dtype=jnp.float32)
        closed_form = np.asarray(_polygon_mass_angle(
            points, n_sides, radius, nx, ny, jnp.asarray(pix_edges, dtype=jnp.float32)))
        ref = _mc_polygon_mass(cx, cy, radius, theta, n_sides, nx, ny, pix_edges, rng)

        assert closed_form.min() > -1e-4, \
            f"n_sides={n_sides}: masse négative {closed_form.min()} -- support non convexe ?"
        assert np.abs(closed_form.sum() - ref.sum()) < 5e-3 * ref.sum(), \
            f"n_sides={n_sides}: masse totale {closed_form.sum():.4f} != référence MC {ref.sum():.4f}"
        assert np.max(np.abs(closed_form - ref)) < 0.02, \
            f"n_sides={n_sides}: écart max par bin {np.max(np.abs(closed_form - ref)):.4f} (bruit MC attendu ~0.002-0.005)"


if test("hc_polygon_rotation_symmetry"):
    # Un n-gone régulier tourné de 2*pi/n_sides est le MÊME polygone -- invariant fort,
    # indépendant du test Monte-Carlo ci-dessus.
    n_sides, radius = 5, 1.7
    pix_edges = jnp.asarray(np.linspace(-3.0, 3.0, 121), dtype=jnp.float32)
    nx, ny = 0.6, 0.8
    theta0 = 0.35
    p0 = jnp.asarray([[0.2, -0.1, theta0]], dtype=jnp.float32)
    p1 = jnp.asarray([[0.2, -0.1, theta0 + 2 * np.pi / n_sides]], dtype=jnp.float32)
    m0 = np.asarray(_polygon_mass_angle(p0, n_sides, radius, nx, ny, pix_edges))
    m1 = np.asarray(_polygon_mass_angle(p1, n_sides, radius, nx, ny, pix_edges))
    assert np.allclose(m0, m1, atol=1e-5), \
        f"tourner theta de 2*pi/n_sides change le profil, écart max {np.max(np.abs(m0-m1))}"


if test("hc_polygon_gradient_matches_finite_difference"):
    nb_angles, nb_bins, extent = 24, 150, 8.0
    radius, n_sides = 0.5, 3
    rng = np.random.default_rng(5)

    hc = HcReconstruction(nb_angles=nb_angles, nb_bins=nb_bins, extent=extent, backend="jax")
    hc.sinogram.values = (rng.random((nb_angles, nb_bins)).astype(np.float32) + 0.1)
    hc.use_polygon(n_sides=n_sides, radius=radius, nb_pixels=nb_bins)

    pts = np.concatenate([
        (rng.random((8, 2)) - 0.5) * extent * 0.6,
        rng.uniform(0, 2 * np.pi, size=(8, 1)),
    ], axis=1).astype(np.float32)

    cost0, grad = hc.cost_model.cost_grad(pts)
    assert np.isfinite(cost0) and np.all(np.isfinite(grad))

    eps = 1e-3
    fd = np.zeros_like(pts, dtype=np.float64)
    for i in range(pts.shape[0]):
        for j in range(3):
            p2, p1 = pts.copy(), pts.copy()
            p2[i, j] += eps
            p1[i, j] -= eps
            fd[i, j] = (hc.cost_model.cost(p2) - hc.cost_model.cost(p1)) / (2 * eps)

    assert np.allclose(grad, fd, atol=2e-2, rtol=1e-1), \
        f"gradient (x,y,theta) != différences finies, écart max {np.max(np.abs(grad - fd))}"
    # colonne theta spécifiquement -- le point de cette feature.
    assert np.allclose(grad[:, 2], fd[:, 2], atol=2e-2, rtol=1e-1), \
        f"gradient de theta seul != différences finies, écart max {np.max(np.abs(grad[:,2] - fd[:,2]))}"


if test("hc_polygon_differs_from_disk"):
    # `use_polygon` doit réellement calculer un coût différent de `use_disks` -- filet de
    # sécurité contre un dispatch qui tomberait silencieusement sur le mauvais modèle.
    nb_angles, nb_bins, extent = 16, 100, 6.0
    radius, n_sides = 0.4, 3
    rng = np.random.default_rng(9)
    values = (rng.random((nb_angles, nb_bins)).astype(np.float32) + 0.1)
    pts2d = (rng.random((8, 2)) - 0.5) * extent * 0.6
    pts3d = np.concatenate([pts2d, rng.uniform(0, 2 * np.pi, size=(8, 1))], axis=1).astype(np.float32)

    hc = HcReconstruction(nb_angles=nb_angles, nb_bins=nb_bins, extent=extent, backend="jax")
    hc.sinogram.values = values

    hc.use_disks(radius=radius, shape="disk")
    cost_disk = hc.cost_model.cost(pts2d.astype(np.float32))

    hc.use_polygon(n_sides=n_sides, radius=radius)
    cost_polygon = hc.cost_model.cost(pts3d)

    assert np.isfinite(cost_disk) and np.isfinite(cost_polygon)
    assert abs(cost_disk - cost_polygon) > 1e-3 * max(1.0, abs(cost_disk))


if test("hc_polygon_pipeline_smoke"):
    # Bout-en-bout via optim.pipeline : le coût baisse ET theta bouge vraiment (pas coincé à
    # son init aléatoire) -- confirme que le gradient de theta est bien utilisé par LBFGS, pas
    # seulement correct isolément (test précédent).
    hc, *_ = HcReconstruction.make_lung_phantom(
        nb_alveoli=5, backend="jax", nb_angles=40, nb_bins=100, record=True)
    points = hc.run_pipeline(
        "lbfgs(polygon(n_sides=3; radius_factor=0.5))",
        nb_diracs=12, seed=2, max_iter=6, verbose=False)

    assert points.shape == (12, 3)
    assert np.all(np.isfinite(points))
    loss_first = hc.recorder.loss_history[0]["cost"]
    loss_last = hc.recorder.loss_history[-1]["cost"]
    assert loss_last < loss_first, f"le coût n'a pas baissé : {loss_first} -> {loss_last}"

    theta_init = hc.recorder.frames[0][:, 2]
    theta_final = points[:, 2]
    assert np.max(np.abs(theta_final - theta_init)) > 1e-4, \
        "theta n'a pas bougé -- le gradient de theta n'est peut-être pas utilisé par LBFGS"
