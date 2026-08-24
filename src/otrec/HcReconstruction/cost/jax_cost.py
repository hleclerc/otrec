"""Jax-backed `CostModel`s: one class per model, each a thin, jitted wrapper
around the pure math in `jax_diracs.py`/`jax_disks.py`.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .base import CostModel
from .jax_diracs import ot_all_angles
from .jax_disks import MASS_FNS, disk_ot_all_angles
from .jax_polygon import polygon_ot_all_angles


class JaxDiracsCost(CostModel):
    """DIRACS model: points ARE equal-mass diracs (the moving SOURCE); the
    sinogram is the fixed piecewise-constant TARGET.
    """

    def __init__(self, sinogram):
        self.sinogram = sinogram
        g = sinogram.geometry
        normals_j = jnp.asarray(g.normals)
        values_j = jnp.asarray(sinogram.values)
        edges_j = jnp.asarray(g.bin_edges)

        @jax.jit
        def cost_grad_jax(points_j):
            return ot_all_angles(points_j, normals_j, values_j, edges_j)
        self._cost_grad_jax = cost_grad_jax

    def cost(self, points: np.ndarray) -> float:
        cost, _ = self._cost_grad_jax(jnp.asarray(points))
        return float(cost)

    def cost_grad(self, points: np.ndarray):
        cost, grad = self._cost_grad_jax(jnp.asarray(points))
        return float(cost), np.asarray(grad)


class JaxDisksCost(CostModel):
    """DISKS model (mirrors `models.DiskModel`): points are the centres of
    fixed-`radius` disks (the unknown TARGET, a piecewise-constant density on
    a `nb_pixels`-wide grid); the measured sinogram becomes fixed weighted
    diracs at its bin centres (the fixed SOURCE) — source/target roles flip
    relative to `JaxDiracsCost`, see `models.DiskModel`'s docstring.

    `shape` : "disk" (the true circular chord profile, see
    `jax_disks._disk_mass_angle`) or "triangle" (a tent of support
    half-width `radius`, see `jax_disks._triangle_mass_angle`) — the SYCL
    disks kernel (`cost.sycl_cost.SyclDisksCost`) is continuous and only
    knows how to sweep "triangle" in closed form; `cost.factory.build_cost_model`
    enforces that at construction time, not this class.
    """

    def __init__(self, sinogram, radius: float, nb_pixels: int, shape: str = "disk"):
        if shape not in MASS_FNS:
            raise ValueError(f"unknown disks shape {shape!r} "
                             f"(expected one of {sorted(MASS_FNS)})")
        self.sinogram = sinogram
        self.radius = float(radius)
        g = sinogram.geometry
        mass_fn = MASS_FNS[shape]
        radius_f = self.radius
        normals_j = jnp.asarray(g.normals)
        values_j = jnp.asarray(sinogram.values)
        bin_centers_j = jnp.asarray(g.bin_centers)
        pix_edges_j = jnp.asarray(g.pixel_edges(nb_pixels))
        cost_grad_fn = jax.value_and_grad(disk_ot_all_angles, argnums=0)

        @jax.jit
        def cost_grad_jax(points_j):
            return cost_grad_fn(points_j, radius_f, normals_j, values_j,
                                bin_centers_j, pix_edges_j, mass_fn)
        self._cost_grad_jax = cost_grad_jax

    def cost(self, points: np.ndarray) -> float:
        cost, _ = self._cost_grad_jax(jnp.asarray(points))
        return float(cost)

    def cost_grad(self, points: np.ndarray):
        cost, grad = self._cost_grad_jax(jnp.asarray(points))
        return float(cost), np.asarray(grad)

    @property
    def frame_radius(self) -> float:
        return self.radius


class JaxPolygonCost(CostModel):
    """POLYGON model: points are `[x, y, theta]` — the centre AND orientation
    of a real regular `n_sides`-gon of fixed circumradius `radius` (unlike
    `JaxDisksCost`'s `shape="triangle"`, which is a radial density profile on
    a circularly-symmetric disk, not an actual polygon — see
    `jax_polygon._polygon_mass_angle`'s docstring). `theta` is a genuine
    optimized variable alongside position; its gradient falls out of
    ordinary jax autodiff, no special-casing needed anywhere else in the
    `LineSearch`/`optim.pipeline` machinery (already shape-agnostic).
    """

    def __init__(self, sinogram, n_sides: int, radius: float, nb_pixels: int):
        self.sinogram = sinogram
        self.n_sides = int(n_sides)
        self.radius = float(radius)
        g = sinogram.geometry
        n_sides_i = self.n_sides
        radius_f = self.radius
        normals_j = jnp.asarray(g.normals)
        values_j = jnp.asarray(sinogram.values)
        bin_centers_j = jnp.asarray(g.bin_centers)
        pix_edges_j = jnp.asarray(g.pixel_edges(nb_pixels))
        cost_grad_fn = jax.value_and_grad(polygon_ot_all_angles, argnums=0)

        @jax.jit
        def cost_grad_jax(points_j):
            return cost_grad_fn(points_j, n_sides_i, radius_f, normals_j, values_j,
                                bin_centers_j, pix_edges_j)
        self._cost_grad_jax = cost_grad_jax

    def cost(self, points: np.ndarray) -> float:
        cost, _ = self._cost_grad_jax(jnp.asarray(points))
        return float(cost)

    def cost_grad(self, points: np.ndarray):
        cost, grad = self._cost_grad_jax(jnp.asarray(points))
        return float(cost), np.asarray(grad)

    @property
    def frame_radius(self) -> float:
        return self.radius
