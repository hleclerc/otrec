"""The one place that knows which `CostModel` a (backend, model, shape) combo
maps to — everything else just depends on the `CostModel` interface.
"""
from .base import CostModel
from .jax_cost import JaxDiracsCost, JaxDisksCost, JaxPolygonCost
from .sycl_cost import SyclDiracsCost, SyclDisksCost


def build_cost_model(sinogram, *, backend: str, model: str,
                     radius: float | None = None, nb_pixels: int | None = None,
                     shape: str = "disk", n_sides: int | None = None) -> CostModel:
    """Build the `CostModel` for `backend` ("jax" | "sycl") x `model`
    ("diracs" | "disks" | "polygon"). `radius` is required for "disks"/
    "polygon"; `nb_pixels" (jax disks/polygon only, defaults to
    `sinogram.geometry.nb_bins`) is the projection grid's own finesse;
    `shape` ("disk" | "triangle", jax "disks" only — sycl's disks kernel is
    continuous and only sweeps "triangle") selects the per-disk radial
    density profile. `n_sides` is required for "polygon" (a REAL regular
    n-gon with its own optimized orientation, see `cost.jax_polygon` —
    distinct from "disks"' `shape="triangle"`, which is a profile on a
    circularly-symmetric disk, not an actual polygon) — jax only, no sycl
    kernel for it yet.
    """
    if model == "disks" and radius is None:
        raise ValueError("the disks model needs a radius")
    if model == "polygon" and (radius is None or n_sides is None):
        raise ValueError("the polygon model needs a radius and n_sides")

    backend = backend.lower()
    if backend == "jax":
        if model == "diracs":
            return JaxDiracsCost(sinogram)
        if model == "disks":
            return JaxDisksCost(sinogram, radius,
                                nb_pixels or sinogram.geometry.nb_bins, shape)
        if model == "polygon":
            return JaxPolygonCost(sinogram, n_sides, radius,
                                  nb_pixels or sinogram.geometry.nb_bins)
        raise ValueError(f"unknown model {model!r} (expected 'diracs', 'disks' or 'polygon')")

    if backend == "sycl":
        if model == "diracs":
            return SyclDiracsCost(sinogram)
        if model == "disks":
            if shape != "triangle":
                raise ValueError(
                    f"the sycl backend's disks kernel is CONTINUOUS and only knows how to "
                    f"sweep the triangle profile in closed form (see cost/hc_ot_sycl.cpp) — "
                    f"got shape={shape!r}. Use backend='jax' for shape='disk', "
                    f"or shape='triangle' here.")
            return SyclDisksCost(sinogram, radius)
        if model == "polygon":
            raise ValueError("the sycl backend does not support the polygon model yet "
                             "(use backend='jax')")
        raise ValueError(f"unknown model {model!r} (expected 'diracs', 'disks' or 'polygon')")

    raise ValueError(f"unknown backend {backend!r} (expected 'jax' or 'sycl')")
