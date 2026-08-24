"""A tiny embedded-Python DSL for composing `HcReconstruction` optimization
pipelines out of independent `Stage` objects — one parametrizable expression
instead of one bespoke `experiments/exp_hc_*.py` file per combination.

Grammar
-------
Ordinary Python expressions, evaluated in a restricted namespace (see
`parse_pipeline`) containing only the names below — no hand-written parser:

- model specs: `diracs` (the dirac model), `disk(...)`/`triangle(...)`
  (the disks model, a radial density PROFILE on an otherwise
  circularly-symmetric disk — `shape="disk"`/`"triangle"` preset, see
  `Disks`), and `polygon(n_sides=...; ...)` (a REAL regular n-gon with its
  own optimized orientation, see `Polygon` — deliberately not named
  `triangle`, which already means something different above).
- line-search stages: `gd(model, **kwargs)`, `pr(...)`, `lbfgs(...)`,
  `quad2d(...)`, `gquad2d(...)`, `grid2d(...)`, `grid3d(...)` — each wraps
  one `LineSearch` class (see `_LINE_SEARCH_CLASSES`); `**kwargs` are that
  class's constructor arguments (e.g. `lbfgs(triangle(...); memory=20)`).
- `multiscale(stage, **kwargs)` — wraps a `Stage` to run coarse -> fine
  (see `MultiscaleStage`).
- `a + b` — sequence: run `b` on `a`'s output, at the SAME resolution
  (`Stage.__add__`).

Example — dirac multiscale with a triangle-disks LBFGS polish after every
stage (mirrors the old `experiments/exp_hc_multiscale_triangles.py`), then
one more polish at the very end:

    multiscale(gd(diracs) + lbfgs(triangle(radius_factor=0.5)); nb_points_init=100) + lbfgs(triangle(radius_factor=0.5))

NOTE the `;` where you'd write `,` in plain Python: `./run`'s CLI splits any
`--flag=a,b,c` into a cartesian-product sweep on a bare comma (see
`loom.cli.main._expand_param_combos`) BEFORE this module ever sees the
string — a literal `,` inside `--pipeline=...` would get shredded into two
bogus sweep values. `parse_pipeline` translates `;` -> `,` before `eval`, so
`;` is this DSL's one and only argument/kwarg separator, always (even when
building a `Stage` from plain Python, for consistency — there's only one
syntax to remember).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace as _dc_replace

import numpy as np

from .gradient_line_search import ConjugateGradient, GradientDescent
from .grid_oracle import Grid2DOracle, Grid3DOracle
from .lbfgs import LBFGS
from .quad2d import GQuad2D, Quad2D

# -- model specs -------------------------------------------------------------


class ModelSpec(ABC):
    """What cost model a `Stage` should run against, resolved against the
    CURRENT point count `n` — a triangle disk's radius, for instance,
    shrinks as the cloud gets denser (see `Disks.radius_for`)."""

    @abstractmethod
    def apply(self, hc, n: int) -> None:
        """Switch `hc` to this model (`hc.use_diracs()`/`hc.use_disks(...)`)."""

    def prepare_points(self, points: np.ndarray, seed: int) -> np.ndarray:
        """Adapt `points` to whatever column layout this model needs, before
        a `LineSearchStage` runs. Default: no-op (every model but `Polygon`
        works on plain `[n, 2]` position arrays) — see `Polygon.prepare_points`
        for the one model that needs an extra column."""
        return points


class _Diracs(ModelSpec):
    def apply(self, hc, n):
        hc.use_diracs()

    def __repr__(self):
        return "diracs"


#: the DIRACS model spec — a singleton, referenced directly in DSL expressions.
diracs = _Diracs()


class Disks(ModelSpec):
    """The DISKS model, at a fixed `radius` or one that SHRINKS with the
    point count as `radius_factor * extent / sqrt(n)` — the same "typical
    local spacing at n points" heuristic `HcReconstruction.split`'s own
    jitter scale uses, so e.g. triangles start roughly touching their
    neighbours at the coarsest stage and shrink to match as the cloud
    gets denser at each finer multiscale stage.
    """

    def __init__(self, shape: str = "disk", *, radius: float | None = None,
                radius_factor: float | None = None, nb_pixels: int | None = None):
        if (radius is None) == (radius_factor is None):
            raise ValueError("Disks needs exactly one of radius or radius_factor")
        self.shape = shape
        self.radius = radius
        self.radius_factor = radius_factor
        self.nb_pixels = nb_pixels

    def radius_for(self, hc, n: int) -> float:
        if self.radius is not None:
            return self.radius
        return self.radius_factor * hc.geometry.extent / np.sqrt(max(n, 1))

    def apply(self, hc, n):
        hc.use_disks(radius=self.radius_for(hc, n), shape=self.shape, nb_pixels=self.nb_pixels)

    def __repr__(self):
        spec = f"radius={self.radius}" if self.radius is not None else f"radius_factor={self.radius_factor}"
        return f"{self.shape}({spec})"


def disk(*, radius: float | None = None, radius_factor: float | None = None,
        nb_pixels: int | None = None) -> Disks:
    """The DISKS model with the true circular-chord profile (jax backend only)."""
    return Disks("disk", radius=radius, radius_factor=radius_factor, nb_pixels=nb_pixels)


def triangle(*, radius: float | None = None, radius_factor: float | None = None,
            nb_pixels: int | None = None) -> Disks:
    """The DISKS model with the tent/triangle profile (jax AND sycl backends)."""
    return Disks("triangle", radius=radius, radius_factor=radius_factor, nb_pixels=nb_pixels)


class Polygon(ModelSpec):
    """A REAL regular `n_sides`-gon per point (see `cost.jax_polygon`) — its
    own orientation is a genuine optimized variable, not a fixed display
    shape. Distinct from `Disks(shape="triangle")`: that is a radial density
    profile on an otherwise circularly-symmetric disk (same silhouette as
    `Disks(shape="disk")`), this is an actual polygon with a real silhouette
    that depends on the projection angle. Jax backend only (`use_polygon`
    raises for `backend="sycl"`, via `cost.factory.build_cost_model`).

    `radius`/`radius_factor` — same "fixed, or shrinks with point count"
    choice as `Disks`.
    """

    def __init__(self, n_sides: int, *, radius: float | None = None,
                radius_factor: float | None = None, nb_pixels: int | None = None):
        if (radius is None) == (radius_factor is None):
            raise ValueError("Polygon needs exactly one of radius or radius_factor")
        self.n_sides = int(n_sides)
        self.radius = radius
        self.radius_factor = radius_factor
        self.nb_pixels = nb_pixels

    def radius_for(self, hc, n: int) -> float:
        if self.radius is not None:
            return self.radius
        return self.radius_factor * hc.geometry.extent / np.sqrt(max(n, 1))

    def apply(self, hc, n):
        hc.use_polygon(n_sides=self.n_sides, radius=self.radius_for(hc, n),
                       nb_pixels=self.nb_pixels)

    def prepare_points(self, points: np.ndarray, seed: int) -> np.ndarray:
        """`[n, 2]` points get a fresh random orientation column appended
        (deterministic from `seed`) — a regular n-gon's own symmetry means
        `[0, 2*pi/n_sides)` already covers every distinct orientation.
        `[n, 3]` points (continuing a polygon pipeline across a multiscale
        split) pass through unchanged."""
        points = np.asarray(points, dtype=np.float32)
        if points.shape[1] >= 3:
            return points
        rng = np.random.default_rng(seed)
        theta0 = rng.uniform(0.0, 2.0 * np.pi / self.n_sides,
                             size=(points.shape[0], 1)).astype(np.float32)
        return np.concatenate([points, theta0], axis=1)

    def __repr__(self):
        spec = f"radius={self.radius}" if self.radius is not None else f"radius_factor={self.radius_factor}"
        return f"polygon(n_sides={self.n_sides}, {spec})"


def polygon(n_sides: int, *, radius: float | None = None,
           radius_factor: float | None = None, nb_pixels: int | None = None) -> Polygon:
    """The POLYGON model — a real regular `n_sides`-gon with its own
    optimized orientation. See `Polygon`."""
    return Polygon(n_sides, radius=radius, radius_factor=radius_factor, nb_pixels=nb_pixels)


# -- run context ---------------------------------------------------------


@dataclass
class RunCtx:
    """Pipeline-wide defaults, threaded through every `Stage.run` — an
    experiment script builds one from its CLI params; individual DSL nodes
    (a `LineSearchStage`'s `max_iter=`/`ftol=`, a `MultiscaleStage`'s
    `nb_points_init=`/etc.) override single fields for their own subtree via
    `replace()`, everything else keeps flowing through unchanged.
    """
    max_iter: int = 60
    ftol: float = 1e-10
    verbose: bool = True
    #: live value handed to every `LineSearchStage` — `None` (full
    #: convergence) at the top level; a `MultiscaleStage` overrides it to its
    #: own `multiscale_plateau_frac` for every non-final stage, `None` again
    #: for the final one.
    plateau_frac: float | None = None
    #: `MultiscaleStage`'s OWN early-stop threshold, unless a node overrides
    #: it directly (`multiscale(...; plateau_frac=0.005)`).
    multiscale_plateau_frac: float = 0.01
    nb_diracs_init: int = 100
    nb_diracs_final: int = 2000
    factor: int = 4
    seed: int = 1
    split_noise_frac: float = 0.05

    def replace(self, **kwargs) -> "RunCtx":
        return _dc_replace(self, **kwargs)


# -- stages ----------------------------------------------------------------


class Stage(ABC):
    """A point-cloud transformation: `points_out = stage.run(hc, points_in, ctx)`.
    `+` sequences stages; `LineSearchStage`/`MultiscaleStage` are the two
    concrete leaves/wrappers — see the module docstring."""

    @abstractmethod
    def run(self, hc, points: np.ndarray, ctx: RunCtx) -> np.ndarray:
        ...

    def __add__(self, other: "Stage") -> "Stage":
        return SequentialStage([*self._flatten(), *other._flatten()])

    def _flatten(self) -> list["Stage"]:
        return [self]


class SequentialStage(Stage):
    """Run each sub-stage in turn, feeding one's output into the next's input."""

    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def _flatten(self):
        return list(self.stages)

    def run(self, hc, points, ctx):
        for stage in self.stages:
            points = stage.run(hc, points, ctx)
        return points

    def __repr__(self):
        return " + ".join(repr(s) for s in self.stages)


_LINE_SEARCH_CLASSES = {
    "gd": GradientDescent, "pr": ConjugateGradient, "lbfgs": LBFGS,
    "quad2d": Quad2D, "gquad2d": GQuad2D,
    "grid2d": Grid2DOracle, "grid3d": Grid3DOracle,
}


class LineSearchStage(Stage):
    """One `LineSearch` run, at whatever `model` resolves to for the CURRENT
    point count, to convergence (`ctx.max_iter`/`ctx.ftol`/`ctx.plateau_frac`
    — pipeline-wide unless overridden here via `max_iter=`/`ftol=`).
    `**ctor_kwargs` are the underlying `LineSearch` class's own constructor
    arguments (e.g. `lbfgs(diracs; memory=20)`, `gd(diracs; instrument_iters=4)`).
    """

    def __init__(self, algo: str, model: ModelSpec, *, max_iter: int | None = None,
                ftol: float | None = None, **ctor_kwargs):
        self.algo = algo
        self.model = model
        self.max_iter = max_iter
        self.ftol = ftol
        self.ctor_kwargs = ctor_kwargs

    def run(self, hc, points, ctx):
        points = self.model.prepare_points(points, ctx.seed)
        n = len(points)
        self.model.apply(hc, n)
        if ctx.verbose:
            if isinstance(self.model, (Disks, Polygon)):
                model_desc = f"{self.model!r} -> radius={self.model.radius_for(hc, n):.4g}"
            else:
                model_desc = repr(self.model)
            print(f"-- stage: {self.algo} / {model_desc}, n={n} points --")
        method = _LINE_SEARCH_CLASSES[self.algo](**self.ctor_kwargs)
        return hc.optimize(
            method, points,
            max_iter=self.max_iter if self.max_iter is not None else ctx.max_iter,
            ftol=self.ftol if self.ftol is not None else ctx.ftol,
            verbose=ctx.verbose, plateau_frac=ctx.plateau_frac)

    def __repr__(self):
        extra = "".join(f"; {k}={v}" for k, v in self.ctor_kwargs.items())
        return f"{self.algo}({self.model!r}{extra})"


def _line_search_factory(algo: str):
    def factory(model: ModelSpec, *, max_iter: int | None = None,
               ftol: float | None = None, **ctor_kwargs) -> LineSearchStage:
        return LineSearchStage(algo, model, max_iter=max_iter, ftol=ftol, **ctor_kwargs)
    factory.__name__ = algo
    factory.__doc__ = f"`Stage` factory for `{_LINE_SEARCH_CLASSES[algo].__name__}` — see `LineSearchStage`."
    return factory


#: DSL line-search factories — `gd(diracs)`, `lbfgs(triangle(...); memory=20)`, ...
gd = _line_search_factory("gd")
pr = _line_search_factory("pr")
lbfgs = _line_search_factory("lbfgs")
quad2d = _line_search_factory("quad2d")
gquad2d = _line_search_factory("gquad2d")
grid2d = _line_search_factory("grid2d")
grid3d = _line_search_factory("grid3d")


def _absorb_stage_history(recorder, hist_from, tim_from, stage, step_base, time_base, dt):
    """Tag `recorder.loss_history[hist_from:]`/`recorder.timings[tim_from:]`
    (whatever a multiscale stage's `run` just appended) with `stage`, and
    offset their `step`/`time` so they continue monotonically from the
    running `step_base`/`time_base`. Returns the updated
    `(step_base, time_base)`."""
    max_step = step_base - 1
    for h in recorder.loss_history[hist_from:]:
        h["stage"] = stage
        h["step"] += step_base
        h["time"] += time_base
        max_step = max(max_step, h["step"])
    for row in recorder.timings[tim_from:]:
        row["stage"] = stage
        row["step"] += step_base
    return max_step + 1, time_base + dt


class MultiscaleStage(Stage):
    """Coarse -> fine: run `inner` at `nb_points_init` points, then repeat
    {split each point into `factor` noisy children, rerun `inner`} until
    `nb_points_final` (defaults to `ctx.nb_diracs_final`, the pipeline's
    `--nb-diracs`). Every stage but the last runs `inner` with
    `plateau_frac` (defaults to `ctx.multiscale_plateau_frac`) — early-stop
    as soon as one step's gain drops below that fraction of the stage's
    first gain, since polishing an intermediate resolution past that point
    is wasted work the next split makes moot anyway. Only the FINAL stage
    converges fully (`plateau_frac=None`).

    `inner` is an arbitrary `Stage`, not just a bare line search — so
    interleaving a refinement in a DIFFERENT model between dirac stages
    (e.g. a triangle-disks LBFGS polish, see the module docstring's example)
    is just sequential composition (`gd(diracs) + lbfgs(triangle(...))`),
    with no special-cased hook needed.

    IGNORES whatever `points` it's called with — multiscale always starts
    its own coarse random cloud (`nb_points_init` points, `ctx.seed`).
    Accumulates continuously into `hc.recorder`'s history/timings across ALL
    stages (see `_absorb_stage_history`), so the usual loss-vs-step /
    loss-vs-time plots and `HcReconstruction.export_html` animation read as
    one continuous run.
    """

    def __init__(self, inner: Stage, *, nb_points_init: int | None = None,
                nb_points_final: int | None = None, factor: int | None = None,
                plateau_frac: float | None = None, split_noise_frac: float | None = None):
        self.inner = inner
        self.nb_points_init = nb_points_init
        self.nb_points_final = nb_points_final
        self.factor = factor
        self.plateau_frac = plateau_frac
        self.split_noise_frac = split_noise_frac

    def run(self, hc, points, ctx):
        nb_init = self.nb_points_init if self.nb_points_init is not None else ctx.nb_diracs_init
        nb_final = self.nb_points_final if self.nb_points_final is not None else ctx.nb_diracs_final
        factor = self.factor if self.factor is not None else ctx.factor
        plateau_frac = self.plateau_frac if self.plateau_frac is not None else ctx.multiscale_plateau_frac
        split_noise_frac = self.split_noise_frac if self.split_noise_frac is not None else ctx.split_noise_frac

        recorder = hc.recorder
        rng = np.random.default_rng(ctx.seed)
        points = hc.random_points(min(nb_init, nb_final), seed=ctx.seed)
        n = points.shape[0]

        step_base, time_base, stage = 0, 0.0, 0
        while True:
            is_final = n >= nb_final
            if ctx.verbose:
                print(f"== multiscale stage {stage}: {n} points"
                     f"{'  [FINAL]' if is_final else ''} ==")
            stage_ctx = ctx.replace(plateau_frac=None if is_final else plateau_frac)

            hist_from, tim_from = len(recorder.loss_history), len(recorder.timings)
            t0 = time.time()
            points = self.inner.run(hc, points, stage_ctx)
            dt = time.time() - t0
            step_base, time_base = _absorb_stage_history(
                recorder, hist_from, tim_from, stage, step_base, time_base, dt)

            if is_final:
                return points

            next_n = min(n * factor, nb_final)
            children = hc.split(factor=factor, noise_frac=split_noise_frac)
            if children.shape[0] > next_n:
                idx = rng.choice(children.shape[0], next_n, replace=False)
                children = children[idx]
            points = children
            n = points.shape[0]
            stage += 1

    def __repr__(self):
        return f"multiscale({self.inner!r})"


def multiscale(inner: Stage, **kwargs) -> MultiscaleStage:
    """`Stage` factory for `MultiscaleStage` — see its docstring."""
    return MultiscaleStage(inner, **kwargs)


# -- parsing -----------------------------------------------------------

_NAMESPACE = dict(
    diracs=diracs, disk=disk, triangle=triangle, polygon=polygon,
    gd=gd, pr=pr, lbfgs=lbfgs, quad2d=quad2d, gquad2d=gquad2d,
    grid2d=grid2d, grid3d=grid3d, multiscale=multiscale,
)


def parse_pipeline(expr: str) -> Stage:
    """Parse a pipeline DSL expression (see the module docstring) into a
    `Stage` tree, by `eval`-ing it in a namespace containing ONLY the DSL
    names above (`__builtins__` stripped) — reusing Python's own expression
    grammar instead of a hand-written parser, for a small closed vocabulary,
    not a general-purpose `eval`.
    """
    code = expr.replace(";", ",")
    try:
        stage = eval(code, {"__builtins__": {}}, dict(_NAMESPACE))
    except Exception as e:
        raise ValueError(f"invalid pipeline expression {expr!r}: {e}") from e
    if not isinstance(stage, Stage):
        raise ValueError(f"pipeline expression {expr!r} did not evaluate to a Stage "
                         f"(got {type(stage).__name__})")
    return stage
