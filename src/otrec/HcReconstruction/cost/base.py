"""The `CostModel` contract every OT backend/model combination implements."""
from abc import ABC, abstractmethod

import numpy as np


class CostModel(ABC):
    """OT cost(+gradient) of a point cloud against one `Sinogram`.

    Concrete subclasses (`cost.jax_cost.JaxDiracsCost`/`JaxDisksCost`,
    `cost.sycl_cost.SyclDiracsCost`/`SyclDisksCost`) each own exactly one
    (backend, model) combination and are independently constructible from a
    `Sinogram` alone — `cost.factory.build_cost_model` just picks the right
    one. Every concrete subclass stores the `Sinogram` it was built from as
    `self.sinogram`.
    """

    sinogram: "object"  # otrec.HcReconstruction.sinogram.Sinogram

    @abstractmethod
    def cost(self, points: np.ndarray) -> float:
        """Cost only — cheaper than `cost_grad` for the bracket phase of a
        line search."""

    @abstractmethod
    def cost_grad(self, points: np.ndarray) -> tuple[float, np.ndarray]:
        """(cost, grad [n, 2]) for `points` ([n, 2])."""

    @property
    def extent(self) -> float:
        """Detector extent of the geometry this cost model was built from —
        line searches use it to pick an initial, scale-appropriate step size."""
        return self.sinogram.geometry.extent

    @property
    def frame_radius(self) -> float:
        """Disk radius to tag recorded frames with (0.0 for diracs models,
        see `optim.recorder.Recorder`) — lets `export.export_html` draw
        markers at their true model size."""
        return 0.0
