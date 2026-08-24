"""Measured piecewise-constant sinogram over a `CtGeometry` — pure numpy."""
import numpy as np

from .geometry import CtGeometry


class Sinogram:
    """`values[a, b]` is the measured density in angle `a`'s detector bin `b`.

    Starts at zero; `add_disk` accumulates the Radon projection of a uniform
    disk (the only way this class knows how to build up a phantom — see
    `phantom.py` for a full lung phantom built from many calls).
    """

    def __init__(self, geometry: CtGeometry):
        self.geometry = geometry
        self.values = np.zeros((geometry.nb_angles, geometry.nb_bins), dtype=np.float32)

    def add_disk(self, center, radius: float, density: float = 1.0) -> "Sinogram":
        """Add the Radon projection of a uniform disk. Returns self."""
        g = self.geometry
        center = np.asarray(center, dtype=float)
        r = float(radius)
        s0 = g.normals @ center
        edges = g.bin_edges[None, :] - s0[:, None]
        t = np.clip(edges, -r, r)
        G = t * np.sqrt(np.maximum(r * r - t * t, 0.0)) \
            + r * r * np.arcsin(t / r)
        self.values += (density * (G[:, 1:] - G[:, :-1]) / g.dw).astype(np.float32)
        return self
