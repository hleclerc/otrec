"""The one place that knows which `CostModel` a (model, shape) combo
maps to — everything else just depends on the `CostModel` interface.
"""
from .base import CostModel
from .jax_cost import JaxDiracsCost, JaxDisksCost, JaxPolygonCost


def build_cost_model(sinogram, *, model: str,
                     radius: float | None = None, nb_pixels: int | None = None,
                     shape: str = "disk", n_sides: int | None = None) -> CostModel:
    """Build the (pure Jax) `CostModel` for `model`
    ("diracs" | "disks" | "polygon"). `radius` is required for "disks"/
    "polygon"; `nb_pixels` (disks/polygon only, defaults to
    `sinogram.geometry.nb_bins`) is the projection grid's own finesse;
    `shape` ("disk" | "triangle", "disks" only) selects the per-disk radial
    density profile. `n_sides` is required for "polygon" (a REAL regular
    n-gon with its own optimized orientation, see `cost.jax_polygon` —
    distinct from "disks"' `shape="triangle"`, which is a profile on a
    circularly-symmetric disk, not an actual polygon).
    """
    if model == "disks" and radius is None:
        raise ValueError("the disks model needs a radius")
    if model == "polygon" and (radius is None or n_sides is None):
        raise ValueError("the polygon model needs a radius and n_sides")

    if model == "diracs":
        return JaxDiracsCost(sinogram)
    if model == "disks":
        return JaxDisksCost(sinogram, radius,
                            nb_pixels or sinogram.geometry.nb_bins, shape)
    if model == "polygon":
        return JaxPolygonCost(sinogram, n_sides, radius,
                              nb_pixels or sinogram.geometry.nb_bins)
    raise ValueError(f"unknown model {model!r} (expected 'diracs', 'disks' or 'polygon')")
