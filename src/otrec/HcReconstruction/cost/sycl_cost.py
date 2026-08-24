"""SYCL-backed `CostModel`s: one class per model, each owning its own pinned
(contiguous float32) arrays and ctypes calls — no shared branching on model
kind, unlike the compiled kernel itself (which does dispatch internally,
see `hc_ot_sycl.cpp`).
"""
import ctypes

import numpy as np

from .base import CostModel
from .sycl_kernel import load_kernel_lib


class SyclDiracsCost(CostModel):
    """DIRACS model, standalone compiled SYCL kernel — see `jax_cost.JaxDiracsCost`
    for the model itself; this class only differs in backend."""

    def __init__(self, sinogram):
        self.sinogram = sinogram
        self._lib = load_kernel_lib()
        g = sinogram.geometry
        self._normals = np.ascontiguousarray(g.normals, dtype=np.float32)
        self._values = np.ascontiguousarray(sinogram.values, dtype=np.float32)
        self._edges = np.ascontiguousarray(g.bin_edges, dtype=np.float32)

    @property
    def _nb_angles(self) -> int:
        return self._normals.shape[0]

    @property
    def _nb_bins(self) -> int:
        return self._values.shape[1]

    def cost(self, points: np.ndarray) -> float:
        pts = np.ascontiguousarray(points, dtype=np.float32)
        return self._lib.hc_ot_cost(
            ctypes.c_void_p(pts.ctypes.data), ctypes.c_int(pts.shape[0]),
            ctypes.c_void_p(self._normals.ctypes.data), ctypes.c_int(self._nb_angles),
            ctypes.c_void_p(self._values.ctypes.data), ctypes.c_int(self._nb_bins),
            ctypes.c_void_p(self._edges.ctypes.data),
        )

    def cost_grad(self, points: np.ndarray):
        pts = np.ascontiguousarray(points, dtype=np.float32)
        grad = np.zeros_like(pts)
        cost = self._lib.hc_ot_cost_grad(
            ctypes.c_void_p(pts.ctypes.data), ctypes.c_int(pts.shape[0]),
            ctypes.c_void_p(self._normals.ctypes.data), ctypes.c_int(self._nb_angles),
            ctypes.c_void_p(self._values.ctypes.data), ctypes.c_int(self._nb_bins),
            ctypes.c_void_p(self._edges.ctypes.data),
            ctypes.c_void_p(grad.ctypes.data),
        )
        return float(cost), grad


class SyclDisksCost(CostModel):
    """DISKS model, standalone compiled SYCL kernel. CONTINUOUS: no
    projection grid, the kernel sweeps the disk union's Radon profile exactly
    (see `hc_ot_sycl.cpp`'s docstring) — and only knows how to do that for the
    TRIANGLE profile (`jax_disks._triangle_mass_angle`'s target), not the true
    circular chord; `cost.factory.build_cost_model` refuses `shape="disk"`
    with this backend before ever constructing one of these.
    """

    def __init__(self, sinogram, radius: float):
        self.sinogram = sinogram
        self.radius = float(radius)
        self._lib = load_kernel_lib()
        g = sinogram.geometry
        self._normals = np.ascontiguousarray(g.normals, dtype=np.float32)
        self._values = np.ascontiguousarray(sinogram.values, dtype=np.float32)
        # continuous kernel needs the source diracs' positions (bin CENTRES, no grid).
        self._bin_centers = np.ascontiguousarray(g.bin_centers, dtype=np.float32)

    @property
    def _nb_angles(self) -> int:
        return self._normals.shape[0]

    @property
    def _nb_bins(self) -> int:
        return self._values.shape[1]

    def cost(self, points: np.ndarray) -> float:
        pts = np.ascontiguousarray(points, dtype=np.float32)
        return self._lib.hc_ot_disks_cost(
            ctypes.c_void_p(pts.ctypes.data), ctypes.c_int(pts.shape[0]),
            ctypes.c_void_p(self._normals.ctypes.data), ctypes.c_int(self._nb_angles),
            ctypes.c_void_p(self._values.ctypes.data), ctypes.c_int(self._nb_bins),
            ctypes.c_void_p(self._bin_centers.ctypes.data),
            ctypes.c_float(self.radius),
        )

    def cost_grad(self, points: np.ndarray):
        pts = np.ascontiguousarray(points, dtype=np.float32)
        grad = np.zeros_like(pts)
        cost = self._lib.hc_ot_disks_cost_grad(
            ctypes.c_void_p(pts.ctypes.data), ctypes.c_int(pts.shape[0]),
            ctypes.c_void_p(self._normals.ctypes.data), ctypes.c_int(self._nb_angles),
            ctypes.c_void_p(self._values.ctypes.data), ctypes.c_int(self._nb_bins),
            ctypes.c_void_p(self._bin_centers.ctypes.data),
            ctypes.c_float(self.radius),
            ctypes.c_void_p(grad.ctypes.data),
        )
        return float(cost), grad

    @property
    def frame_radius(self) -> float:
        return self.radius
