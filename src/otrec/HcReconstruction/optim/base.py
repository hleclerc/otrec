"""The `LineSearch` contract every optimization algorithm implements."""
from abc import ABC, abstractmethod

import numpy as np

from ..cost.base import CostModel
from .recorder import Recorder


class LineSearch(ABC):
    """A single-resolution point-cloud optimizer: repeatedly steps `points`
    downhill against a `CostModel`, appending each step to a `Recorder`.

    Each concrete subclass (`GradientDescent`, `ConjugateGradient`, `LBFGS`,
    `Quad2D`, `GQuad2D`, `Grid2DOracle`, `Grid3DOracle`) is independently
    constructible and configurable — algorithm-specific knobs (e.g. `LBFGS`'s
    `memory`) are constructor arguments, not `run()` kwargs, so an instance
    fully describes "which method, with which settings" and can be passed
    around as a value (see `optim.pipeline.MultiscaleStage`, whose `inner`
    is built by `optim.pipeline.parse_pipeline`).

    `recorder` is optional: pass `None` (the default) to run standalone, with
    no bookkeeping — `HcReconstruction.optimize` always supplies its own.
    """
    algo: str

    @abstractmethod
    def run(self, points: np.ndarray, cost_model: CostModel,
            recorder: Recorder | None = None, *,
            max_iter: int = 60, ftol: float = 1e-10, verbose: bool = True,
            plateau_frac: float | None = None) -> np.ndarray:
        """Return the optimized `points` ([n, 2])."""
