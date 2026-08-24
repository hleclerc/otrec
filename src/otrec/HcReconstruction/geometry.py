"""CT detector/angle geometry — the value object every other piece is built from."""
import numpy as np


class CtGeometry:
    """`nb_angles` projection angles evenly spaced over [0, pi), each with
    `nb_bins` detector bins spanning [-extent/2, extent/2].

    Pure geometry, no measured data: `Sinogram` wraps one of these with a
    `values` array, and every `CostModel` is built from a `Sinogram`.
    """

    def __init__(self, nb_angles: int, nb_bins: int, extent: float):
        self.nb_angles = int(nb_angles)
        self.nb_bins = int(nb_bins)
        self.extent = float(extent)
        self.dw = self.extent / self.nb_bins
        self.s_min = -self.extent / 2

        self.angles = np.pi * np.arange(self.nb_angles) / self.nb_angles
        self.normals = np.stack([np.cos(self.angles), np.sin(self.angles)], axis=1)
        self.bin_edges = self.s_min + self.dw * np.arange(self.nb_bins + 1)
        self.bin_centers = self.s_min + self.dw * (np.arange(self.nb_bins) + 0.5)

    def pixel_edges(self, nb_pixels: int) -> np.ndarray:
        """Edges of an alternate-resolution grid over the same detector
        extent — used by the disks model's projection grid (see
        `cost.jax_cost.JaxDisksCost`), independent of `nb_bins`."""
        dw_pix = self.extent / nb_pixels
        return self.s_min + dw_pix * np.arange(nb_pixels + 1)
