"""`HcReconstruction` — a thin orchestrator over independently-usable pieces.

One class, two OT backends (pure Jax or a standalone SYCL fused kernel), no
loom/sdot deps. Each responsibility below is its OWN class, constructible and
testable on its own — `HcReconstruction` only wires them together and adds a
few convenience methods for the common path:

- `geometry.CtGeometry` — detector/angle geometry (pure value object).
- `sinogram.Sinogram` — a `CtGeometry` + measured values, built up via `add_disk`.
- `cost.base.CostModel` (`cost.factory.build_cost_model`) — OT cost(+grad) of a
  point cloud against a `Sinogram`; one concrete class per (backend, model):
  `cost.jax_cost.JaxDiracsCost`/`JaxDisksCost`,
  `cost.sycl_cost.SyclDiracsCost`/`SyclDisksCost`.
- `optim.base.LineSearch` — an optimization algorithm (`optim.gradient_line_search
  .GradientDescent`/`ConjugateGradient`, `optim.lbfgs.LBFGS`, `optim.quad2d.Quad2D`/
  `GQuad2D`, `optim.grid_oracle.Grid2DOracle`/`Grid3DOracle`); each is a small,
  independently configurable object — `hc.optimize(LBFGS(memory=20), points)`.
- `optim.recorder.Recorder` — per-step frame/history bookkeeping, shared by
  every `LineSearch.run` call on this reconstruction.
- `optim.pipeline.Stage` (`optim.pipeline.parse_pipeline`) — a composable
  point-cloud transformation (a single `LineSearch` run, or a coarse -> fine
  `multiscale(...)` driver over an arbitrary inner `Stage`); a whole
  optimization run is one `Stage` expression, see `run_pipeline`.

Usage
-----
    hc = HcReconstruction(nb_angles=300, nb_bins=1000, extent=44.0,
                          backend="sycl", record=True)
    hc.add_disk([0, 0], 10.0)
    hc.add_disk([2, 3], 1.0, density=-1.0)

    # or: lung phantom
    hc, lobes, alveoli = HcReconstruction.make_lung_phantom(nb_alveoli=1000)

    points = hc.random_points(5000, seed=1)
    result = hc.optimize(GradientDescent(), points, max_iter=60)
    hc.run_pipeline("multiscale(lbfgs(diracs); nb_points_init=100)", nb_diracs=5000)
    hc.export_html("out.html")
"""
import numpy as np

from .cost.factory import build_cost_model
from .export import export_html
from .geometry import CtGeometry
from .optim.base import LineSearch
from .optim.pipeline import RunCtx, Stage, parse_pipeline
from .optim.recorder import Recorder
from .phantom import hex_sites
from .point_cloud import random_points, split_points
from .sinogram import Sinogram


class HcReconstruction:
    """Simplified CT reconstruction from a piecewise-constant sinogram.

    See the module docstring for the usage example and for which class owns
    which responsibility.
    """

    def __init__(self, nb_angles: int, nb_bins: int, extent: float, *,
                 backend: str = "jax", record: bool = False,
                 model: str = "diracs", radius: float | None = None,
                 nb_pixels: int | None = None, shape: str = "disk"):
        self.geometry = CtGeometry(nb_angles, nb_bins, extent)
        self.sinogram = Sinogram(self.geometry)
        self.recorder = Recorder(enabled=record)

        # -- cost model config: "diracs" (points ARE equal-mass diracs, sinogram is the
        # fixed piecewise-constant target) or "disks" (points are FIXED-radius disk
        # centers, the measured sinogram becomes fixed weighted diracs — see
        # `models.DiskModel`'s docstring for why the roles flip). `radius` is required
        # for "disks"; `nb_pixels` is the disk-projection grid's own finesse (JAX
        # backend only — the SYCL backend's disks kernel is continuous, no grid, see
        # `cost.sycl_cost.SyclDisksCost`). `shape` : the per-disk PROFILE used by the
        # disks model -- "disk" (the true circular chord) or "triangle" (a tent of the
        # same support half-width `radius`, see `cost.jax_disks._triangle_mass_angle`).
        # JAX supports both, so the two can be compared directly; the SYCL kernel is
        # continuous and only knows how to sweep the triangle profile in closed form,
        # so `shape="disk"` with `backend="sycl"` raises when the cost model is built.
        self.backend = backend.lower()
        self.model = model
        self.radius = float(radius) if radius is not None else None
        self.nb_pixels = int(nb_pixels) if nb_pixels is not None else self.geometry.nb_bins
        self.shape = shape
        self.n_sides = None
        self._cost_model = None

    # -- cost model: lazy, rebuilt whenever use_diracs/use_disks/use_polygon changes config --

    @property
    def cost_model(self):
        if self._cost_model is None:
            self._cost_model = build_cost_model(
                self.sinogram, backend=self.backend, model=self.model,
                radius=self.radius, nb_pixels=self.nb_pixels, shape=self.shape,
                n_sides=self.n_sides)
        return self._cost_model

    def use_diracs(self) -> "HcReconstruction":
        """Switch to the DIRACS model. Returns self."""
        self.model = "diracs"
        self._cost_model = None
        return self

    def use_disks(self, radius: float, nb_pixels: int | None = None,
                  shape: str | None = None) -> "HcReconstruction":
        """Switch to the DISKS model (mirrors `models.DiskModel`): points
        become the centres of fixed-`radius` disks. `shape=None` keeps
        whatever was set before (the constructor's own default is "disk").
        Returns self.
        """
        self.model = "disks"
        self.radius = float(radius)
        if nb_pixels is not None:
            self.nb_pixels = int(nb_pixels)
        if shape is not None:
            self.shape = shape
        self._cost_model = None
        return self

    def use_polygon(self, n_sides: int, radius: float,
                    nb_pixels: int | None = None) -> "HcReconstruction":
        """Switch to the POLYGON model (see `cost.jax_polygon`): points
        become `[x, y, theta]` — the centre AND orientation of a real regular
        `n_sides`-gon of fixed circumradius `radius`. Distinct from
        `use_disks(shape="triangle")`, which is a radial density PROFILE on a
        circularly-symmetric disk, not an actual polygon. Jax backend only.
        Returns self.
        """
        self.model = "polygon"
        self.n_sides = int(n_sides)
        self.radius = float(radius)
        if nb_pixels is not None:
            self.nb_pixels = int(nb_pixels)
        self._cost_model = None
        return self

    @property
    def floor(self) -> float:
        """Incompressible cost floor of the current model (0 for diracs).
        For disks: the QUANTIZATION floor of treating the measured sinogram
        as diracs at its bin centres instead of its true continuous profile
        — `dw²/12` per angle (see `models.DiskModel.floor`'s docstring)."""
        if self.model != "disks":
            return 0.0
        return self.geometry.nb_angles * self.geometry.dw ** 2 / 12

    # -- sinogram convenience --------------------------------------------------

    def add_disk(self, center, radius: float, density: float = 1.0) -> "HcReconstruction":
        """Add the Radon projection of a uniform disk to `self.sinogram`. Returns self."""
        self.sinogram.add_disk(center, radius, density)
        return self

    # -- point cloud -------------------------------------------------------

    def random_points(self, n: int, seed: int = 0) -> np.ndarray:
        """Generate `n` random 2D points within the detector extent."""
        return random_points(self.geometry.extent, n, seed)

    def split(self, factor: int = 4, noise_frac: float = 0.05) -> np.ndarray:
        """Replace each point of the last recorded frame by `factor` noisy
        children. Returns the new point cloud [n*factor, 2]."""
        if self.recorder.positions is None:
            raise ValueError("no points to split")
        return split_points(self.recorder.positions, self.geometry.extent, factor, noise_frac)

    # -- optimization --------------------------------------------------------

    def optimize(self, method: LineSearch, points, **kwargs) -> np.ndarray:
        """Run one `LineSearch` at the current resolution. Returns optimized
        `points` [n, 2]. Records into `self.recorder` when `record=True`."""
        return method.run(points, self.cost_model, self.recorder, **kwargs)

    def run_pipeline(self, pipeline: Stage | str, points: np.ndarray | None = None, *,
                     nb_diracs: int | None = None, seed: int = 1, max_iter: int = 60,
                     ftol: float = 1e-10, verbose: bool = True,
                     multiscale_plateau_frac: float = 0.01, nb_diracs_init: int = 100,
                     factor: int = 4, split_noise_frac: float = 0.05) -> np.ndarray:
        """Run a `Stage` (or a DSL string, see `optim.pipeline.parse_pipeline`)
        against this reconstruction.

        `points` seeds a pipeline with NO `multiscale(...)` node (defaults to
        `nb_diracs` random points, `seed`); a `multiscale(...)` node ignores
        it and generates its own coarse start (`nb_diracs_init` points).
        Every other keyword becomes the matching `optim.pipeline.RunCtx`
        field — the pipeline-wide default any DSL node can override for its
        own subtree.
        """
        if isinstance(pipeline, str):
            pipeline = parse_pipeline(pipeline)
        nb_diracs = nb_diracs if nb_diracs is not None else 2000
        if points is None:
            points = self.random_points(nb_diracs, seed=seed)
        ctx = RunCtx(max_iter=max_iter, ftol=ftol, verbose=verbose,
                    multiscale_plateau_frac=multiscale_plateau_frac,
                    nb_diracs_init=nb_diracs_init, nb_diracs_final=nb_diracs,
                    factor=factor, seed=seed, split_noise_frac=split_noise_frac)
        return pipeline.run(self, points, ctx)

    # -- phantom factory -----------------------------------------------------

    @classmethod
    def make_lung_phantom(cls, nb_angles=600, nb_bins=2000, extent=44.0,
                          nb_alveoli=1000, alveolus_radius=2.75, scale=1.0,
                          seed=0, **kwargs) -> tuple["HcReconstruction", list, list]:
        """Build a lung phantom. Returns ``(hc, lobes, alveoli)``."""
        if alveolus_radius is None:
            alveolus_radius = 0.075 * scale
        lobes = [(np.array([0.0, 0.0]), 0.5 * extent * scale)]
        hc = cls(nb_angles=nb_angles, nb_bins=nb_bins, extent=extent, **kwargs)
        for center, radius in lobes:
            hc.add_disk(center=center, radius=radius, density=1.0)
        sites, rng = hex_sites(lobes, alveolus_radius, seed=seed)
        nb_alveoli = min(nb_alveoli, len(sites))
        rng.shuffle(sites)
        sites = sites[:nb_alveoli]
        radii = alveolus_radius * (0.8 + 0.2 * rng.random(nb_alveoli))
        for center, radius in zip(sites, radii):
            hc.add_disk(center=center, radius=float(radius), density=-1.0)
        alv = list(zip(sites, radii))
        print(f"  phantom: {nb_alveoli} alveoli")
        return hc, lobes, alv

    # -- display -------------------------------------------------------------

    def export_html(self, out_path: str, **kwargs) -> "HcReconstruction":
        """Export the recorded run to a self-contained HTML animation. When
        the CURRENT model is "polygon", automatically passes `n_sides` (see
        `export.export_html`) so the true rotated polygon is drawn — no
        manual marker needed, the pipeline's own model drives the render."""
        if self.model == "polygon" and "n_sides" not in kwargs:
            kwargs["n_sides"] = self.n_sides
        export_html(self.recorder, self.geometry, out_path, **kwargs)
        return self
