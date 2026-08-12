"""Simplified CT reconstruction — one class, two OT backends, no loom/sdot deps.

`HcReconstruction` builds sinograms (pure numpy), computes OT cost+gradient via
pure Jax OR a standalone SYCL fused kernel (compiled with acpp, loaded via ctypes),
solves via line search, and exports results.

The SYCL kernel (`hc_ot_sycl.cpp`) is compiled ONCE into a shared library
(`hc_ot_sycl.dylib` on macOS, `hc_ot_sycl.so` on Linux) and loaded at first use.
No `Tensor`, no `Image`, no `driver.call` — the API is pure numpy / raw float pointers.
"""
import ctypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp


# ---- pure Jax 1D OT (one angle) — analytical, no scan/fori_loop -----------

def _ot1d_angle(proj, bin_mass, bin_edges):
    """1D OT cost + gradient, fully parallel (no scan, no fori_loop).

    Uses the integrated quantile function M(q) = ∫₀^q Q(t) dt of the target,
    evaluated at dirac quantile boundaries via searchsorted + prefix sums.
    Barycenter i = (M((i+1)/n) − M(i/n)) / w  — all vectorised.
    """
    n = proj.shape[0]
    m = bin_mass.shape[0]
    w = 1.0 / n
    dw = bin_edges[1] - bin_edges[0]

    order = jnp.argsort(proj)
    s = proj[order]

    # cumulative target mass at bin boundaries
    cum     = jnp.cumsum(bin_mass)                           # [m]
    cum_sta = jnp.concatenate([jnp.zeros(1), cum[:-1]])      # [m]

    # prefix sum of M at bin boundaries:  M(cum[j]) = Σ_{k<j} bin_mass[k]·center[k]
    bin_center = bin_edges[:-1] + dw / 2
    prefix_M = jnp.concatenate([
        jnp.zeros(1), jnp.cumsum(bin_mass * bin_center)[:-1]])  # [m]

    # ---- M(q) = prefix_M[j] + bin_mass[j]·(edges[j]·f + dw·f²/2) ----
    #   where j = searchsorted(cum, q),  f = (q − cum_sta[j]) / bin_mass[j]
    def M_at(q):
        j = jnp.clip(jnp.searchsorted(cum, q, side='right'), 0, m - 1)
        bm = bin_mass[j]
        f = jnp.where(bm > 0, (q - cum_sta[j]) / bm, 0.0)
        return prefix_M[j] + bm * (bin_edges[j] * f + dw * f * f / 2)

    # dirac quantile boundaries
    i = jnp.arange(n)
    q0 = i * w
    q1 = q0 + w

    M0 = M_at(q0)
    M1 = M_at(q1)
    bary = (M1 - M0) / w                               # [n]

    grad_sorted = 2.0 * w * (s - bary)                 # [n]
    grad = grad_sorted[jnp.argsort(order)]

    # cost = W2²(μ, ν) = w·Σ s_i² − 2w·Σ s_i·bary_i + total_2nd_moment(ν)
    w2_nu = jnp.sum(bin_mass * bin_center ** 2) + dw * dw / 12
    cost = w * jnp.sum(s ** 2) - 2 * w * jnp.sum(s * bary) + w2_nu

    return cost, grad


@jax.jit
def _ot_all_angles(points, normals, sino_vals, bin_edges):
    """Pure Jax OT for all angles.  vmapped over angles."""
    proj = points @ normals.T                            # [n, A]
    proj = proj.T                                        # [A, n]
    total = sino_vals.sum(axis=1, keepdims=True)         # [A, 1]
    bin_mass = sino_vals / total                         # [A, m]
    costs, grads_s = jax.vmap(_ot1d_angle, in_axes=(0, 0, None))(
        proj, bin_mass, bin_edges)
    return costs.sum(), grads_s.T @ normals              # [n, A] @ [A, 2]


# ---- SYCL kernel compilation and loading -----------------------------------

_KERNEL_DIR = Path(__file__).resolve().parent
_KERNEL_CPP  = _KERNEL_DIR / "hc_ot_sycl.cpp"
# Shared library extension: .dylib on macOS, .so on Linux (the container).
_SO_EXT     = ".dylib" if sys.platform == "darwin" else ".so"
_KERNEL_SO   = _KERNEL_DIR / f"hc_ot_sycl{_SO_EXT}"


def _find_acpp():
    """Return the path to the `acpp` compiler binary.

    Locates an already-built acpp through loom's AdaptiveCpp cache (which honours
    SDOT_CACHE_DIR in containers, plus XDG / macOS / /tmp fallbacks), then falls
    back to `acpp` on PATH.
    """
    try:
        from loom.compilation.adaptive_cpp import acpp_path, usable_backend_set
    except ImportError:
        return shutil.which("acpp") or "acpp"

    # A "full" build (CUDA/HIP/...) also compiles the omp target we use here, so
    # prefer it before the CPU-only "minimal" build.
    for profile in ("full", "minimal"):
        backends = usable_backend_set(profile, ())
        if backends is not None:
            p = acpp_path(profile, backends)
            if p.is_file():
                return str(p)

    return shutil.which("acpp") or "acpp"


def _libomp_include_flags():
    """Extra -I flags so <omp.h> resolves (macOS only — Homebrew libomp is keg-only).

    Linux/acpp already provide OpenMP headers; the container ships libomp-20-dev.
    """
    if sys.platform != "darwin":
        return []
    for p in ("/opt/homebrew/opt/libomp/include", "/usr/local/opt/libomp/include"):
        if Path(p).is_dir():
            return ["-I", p]
    return []


def _compile_sycl_kernel():
    """Compile hc_ot_sycl.cpp → shared lib (once; recompiles if source is newer)."""
    if _KERNEL_SO.exists() and _KERNEL_SO.stat().st_mtime >= _KERNEL_CPP.stat().st_mtime:
        return

    acpp = _find_acpp()
    cmd = [
         acpp, "--acpp-targets=omp", "-std=c++20", "-O3", "-ffast-math",
        "-fPIC", "-shared",
        *_libomp_include_flags(),
        "-o", str(_KERNEL_SO), str(_KERNEL_CPP),
    ]
    subprocess.run(cmd, check=True)


def _load_sycl_kernel():
    """Return a ctypes CDLL with the two SYCL entry points."""
    lib = ctypes.CDLL(str(_KERNEL_SO))

    # double hc_ot_cost_grad(
    #     const float* points, int n,
    #     const float* normals, int nb_angles,
    #     const float* sino_vals, int nb_bins,
    #     const float* bin_edges,
    #     float* grad)
    lib.hc_ot_cost_grad.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.hc_ot_cost_grad.restype = ctypes.c_double

    # double hc_ot_cost(
    #     const float* points, int n,
    #     const float* normals, int nb_angles,
    #     const float* sino_vals, int nb_bins,
    #     const float* bin_edges)
    lib.hc_ot_cost.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.hc_ot_cost.restype = ctypes.c_double

    return lib


# ---- HcReconstruction ------------------------------------------------------

class HcReconstruction:
    """Simplified CT reconstruction from a piecewise-constant sinogram.

    Usage
    -----
        hc = HcReconstruction(nb_angles=300, nb_bins=1000, extent=44.0,
                              backend="sycl", record=True)
        hc.add_disk([0, 0], 10.0)
        hc.add_disk([2, 3], 1.0, density=-1.0)

        # or: lung phantom
        hc, lobes, alveoli = HcReconstruction.make_lung_phantom(nb_alveoli=1000)

        points = hc.random_points(5000, seed=1)
        result = hc.line_search(points, max_iter=60)
        hc.export_html("out.html")
    """

    def __init__(self, nb_angles: int, nb_bins: int, extent: float,
                 backend: str = "jax", record: bool = False):
        self.extent = float(extent)
        self.nb_angles = int(nb_angles)
        self.nb_bins   = int(nb_bins)
        self.dw = self.extent / nb_bins
        self.s_min = -self.extent / 2

        angles = np.pi * np.arange(nb_angles) / nb_angles
        self.angles = angles
        self.normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        self.bin_edges   = self.s_min + self.dw * np.arange(nb_bins + 1)
        self.bin_centers = self.s_min + self.dw * (np.arange(nb_bins) + 0.5)

        self.values = np.zeros((nb_angles, nb_bins), dtype=np.float32)

        self.backend = backend.lower()
        self._ready  = False

        self.record = record
        self.frames: list[np.ndarray] = []
        self.loss_history: list[dict] = []
        self.timings: list[dict] = []         # per-step bracket + grad wall time

    # -- sinogram (pure numpy) -----------------------------------------------

    def add_disk(self, center, radius: float, density: float = 1.0
                 ) -> "HcReconstruction":
        """Add the Radon projection of a uniform disk. Returns self."""
        center = np.asarray(center, dtype=float)
        r = float(radius)
        s0 = self.normals @ center
        edges = self.bin_edges[None, :] - s0[:, None]
        t = np.clip(edges, -r, r)
        G = t * np.sqrt(np.maximum(r * r - t * t, 0.0)) \
            + r * r * np.arcsin(t / r)
        self.values += (density * (G[:, 1:] - G[:, :-1]) / self.dw
                        ).astype(np.float32)
        return self

    # -- phantom factory -----------------------------------------------------

    @classmethod
    def make_lung_phantom( cls, nb_angles=300, nb_bins=1000, extent=44.0,
                          nb_alveoli=1000, alveolus_radius=0.75, scale=1.0,
                          seed=0, **kwargs ):
        """Build a lung phantom. Returns ``(hc, lobes, alveoli)``."""
        if alveolus_radius is None:
            alveolus_radius = 0.075 * scale
        lobes = [(np.array([0.0, 0.0]), 0.5 * extent * scale)]
        hc = cls(nb_angles=nb_angles, nb_bins=nb_bins, extent=extent, **kwargs)
        for center, radius in lobes:
            hc.add_disk(center=center, radius=radius, density=1.0)
        sites, rng = cls._hex_sites(lobes, alveolus_radius, seed=seed)
        nb_alveoli = min(nb_alveoli, len(sites))
        rng.shuffle(sites)
        sites = sites[:nb_alveoli]
        radii = alveolus_radius * (0.8 + 0.2 * rng.random(nb_alveoli))
        for center, radius in zip(sites, radii):
            hc.add_disk(center=center, radius=float(radius), density=-1.0)
        alv = list(zip(sites, radii))
        print(f"  phantom: {nb_alveoli} alveoli")
        return hc, lobes, alv

    @staticmethod
    def _hex_sites(lobes, max_radius, spacing_factor=2.15, jitter_factor=0.02,
                   seed=0):
        """Candidate alveolus centres on a jittered hexagonal grid."""
        rng = np.random.default_rng(seed)
        spacing = spacing_factor * max_radius
        row_h = spacing * np.sqrt(3) / 2
        jitter_amp = jitter_factor * spacing
        bound = max(r + float(np.linalg.norm(c)) for c, r in lobes) + spacing
        rows = int(2 * bound / row_h) + 2
        cols = int(2 * bound / spacing) + 2
        ii, jj = np.meshgrid(np.arange(-rows, rows), np.arange(-cols, cols),
                             indexing="ij")
        y = ii.ravel() * row_h
        x = jj.ravel() * spacing + np.where(ii.ravel() % 2, spacing / 2, 0.0)
        pts = np.stack([x, y], axis=1)
        pts = pts[(np.abs(pts[:, 0]) <= bound) & (np.abs(pts[:, 1]) <= bound)]
        pts += (rng.random(pts.shape) * 2 - 1) * jitter_amp
        inside = np.zeros(len(pts), dtype=bool)
        for center, radius in lobes:
            d = np.linalg.norm(pts - center[None, :], axis=1)
            inside |= d <= (radius - max_radius * 1.05)
        return pts[inside], rng

    # -- point cloud ---------------------------------------------------------

    def random_points(self, n: int, seed: int = 0) -> np.ndarray:
        """Generate `n` random 2D points within the detector extent."""
        rng = np.random.default_rng(seed)
        return rng.normal(0, self.extent / 6, size=(n, 2)).astype(np.float32)

    def split(self, factor: int = 4, noise_frac: float = 0.05) -> np.ndarray:
        """Replace each point by `factor` children with small uniform noise.

        Returns the new point cloud [n*factor, 2].
        """
        if self.positions is None:
            raise ValueError("no points to split")
        rng = np.random.default_rng()
        n = self.positions.shape[0]
        noise_scale = noise_frac * self.extent / np.sqrt(max(1, n))
        tiled = np.repeat(self.positions, factor, axis=0)
        return tiled + (rng.random(tiled.shape).astype(np.float32) - 0.5) * 2 * noise_scale

    # -- lazy init -----------------------------------------------------------

    def _init(self):
        if self._ready:
            return

        if self.backend == "jax":
            normals_j = jnp.asarray(self.normals)
            values_j  = jnp.asarray(self.values)
            edges_j   = jnp.asarray(self.bin_edges)

            @jax.jit
            def fn(points_j):
                return _ot_all_angles(points_j, normals_j, values_j, edges_j)
            self._cost_grad_jax = fn

        else:  # "sycl" — standalone compiled kernel, loaded via ctypes
            assert self.backend == "sycl", "jax or sycl"

            _compile_sycl_kernel()
            self._lib = _load_sycl_kernel()
            # pin numpy arrays (float32, contiguous) for ctypes
            self._normals_p = np.ascontiguousarray(
                self.normals, dtype=np.float32)
            self._values_p  = np.ascontiguousarray(
                self.values,  dtype=np.float32)
            self._edges_p   = np.ascontiguousarray(
                self.bin_edges, dtype=np.float32)

        self._ready = True

    # -- OT interface --------------------------------------------------------

    def _loss_only(self, points: np.ndarray) -> float:
        """Cost only — for the bracket phase of line search."""
        self._init()
        if self.backend == "jax":
            cost, _ = self._cost_grad_jax(jnp.asarray(points))
            return float(cost)

        pts = np.ascontiguousarray(points, dtype=np.float32)
        return self._lib.hc_ot_cost(
            ctypes.c_void_p(pts.ctypes.data),
            ctypes.c_int(pts.shape[0]),
            ctypes.c_void_p(self._normals_p.ctypes.data),
            ctypes.c_int(self.nb_angles),
            ctypes.c_void_p(self._values_p.ctypes.data),
            ctypes.c_int(self.nb_bins),
            ctypes.c_void_p(self._edges_p.ctypes.data),
        )

    def loss_grad(self, points: np.ndarray):
        """(float cost, ndarray grad [n, 2]) for `points` ([n, 2])."""
        self._init()
        if self.backend == "jax":
            cost, grad = self._cost_grad_jax(jnp.asarray(points))
            return float(cost), np.asarray(grad)

        pts = np.ascontiguousarray(points, dtype=np.float32)
        grad = np.zeros_like(pts)

        cost = self._lib.hc_ot_cost_grad(
            ctypes.c_void_p(pts.ctypes.data),
            ctypes.c_int(pts.shape[0]),
            ctypes.c_void_p(self._normals_p.ctypes.data),
            ctypes.c_int(self.nb_angles),
            ctypes.c_void_p(self._values_p.ctypes.data),
            ctypes.c_int(self.nb_bins),
            ctypes.c_void_p(self._edges_p.ctypes.data),
            ctypes.c_void_p(grad.ctypes.data),
        )
        return float(cost), grad

    # -- solver --------------------------------------------------------------

    def line_search(self, points, max_iter: int = 60, ftol: float = 1e-10,
                    verbose: bool = True):
        """Gradient descent with parabolic line search.

        If ``self.record`` is True, captures frames and loss history.
        Returns optimized `points` [n, 2].
        """

        self._init()

        p = np.asarray(points, dtype=np.float32).copy()
        t0 = time.time()
        cost, g = self.loss_grad(p)
        t_grad = (time.time() - t0) * 1000
        g = np.asarray(g)

        t_start = time.time()
        self._record_step( -1, p, cost, t_start )

        g_norms = np.linalg.norm(g, axis=-1)
        mean_n = float(np.mean(g_norms)) if len(g_norms) else 1.0
        a = 0.01 * self.extent / mean_n if mean_n > 0 else 0.0

        for step in range(max_iter):
            t_bracket = time.time()
            a, cost_half, cost_a, tries = self._bracket(p, cost, g, a)
            t_bracket = (time.time() - t_bracket) * 1000
            t = self._parabola_vertex(cost, cost_half, cost_a, a)

            p_new = p - t * g
            t_grad = time.time()
            cost_new, g_new = self.loss_grad(p_new)
            t_grad = (time.time() - t_grad) * 1000
            g_new = np.asarray(g_new, dtype=np.float32)
            a = 2.0 * t

            self._record_step(step, p_new, cost_new, t_start)
            self.timings.append(dict(
                step=step, n_points=p.shape[0], backend=self.backend,
                bracket_ms=round(t_bracket, 1), grad_ms=round(t_grad, 1)))

            if verbose: #  and (step % 10 == 0 or step == max_iter - 1):
                disp = float(np.max(np.abs(p_new - p)))
                print(f"  step {step:3d}: loss={cost_new:.6f}, "
                      f"max disp={disp:.4g}  "
                      f"(bracket tries={tries}, t={t:.4g})")

            if abs(cost - cost_new) < ftol:
                if verbose:
                    print(f"  converged (step {step + 1}): |Δcost| < {ftol}")
                return p_new

            p, cost, g = p_new, cost_new, g_new

        return p

    def _record_step(self, step, pts, cost, t_start):
        if not self.record:
            return
        if step == -1 and self.frames:
            return
        self.frames.append(np.array(pts, dtype=np.float32, copy=True))
        self.loss_history.append(dict(
            step=step, cost=float(cost), time=time.time() - t_start))

    # -- bracket & parabola --------------------------------------------------

    def _bracket(self, p, cost0, g, a, max_tries=30):
        """Find `a` s.t. a/2 is the lowest-cost of (0, a/2, a) along -g."""
        cost_half = self._loss_only((p - (a / 2) * g).astype(np.float32))
        cost_a    = self._loss_only((p - a * g).astype(np.float32))

        for tries in range(max_tries):
            if cost_half <= cost0 and cost_half <= cost_a:
                break
            if cost_a <= cost_half and cost_a <= cost0:
                a *= 2
                cost_half, cost_a = cost_a, self._loss_only(
                    (p - a * g).astype(np.float32))
            else:
                a /= 2
                cost_a, cost_half = cost_half, self._loss_only(
                    (p - (a / 2) * g).astype(np.float32))

        return a, cost_half, cost_a, tries

    @staticmethod
    def _parabola_vertex(cost0, cost_half, cost_a, a):
        """Vertex of parabola through (0, cost0), (a/2, cost_half), (a, cost_a)."""
        h = a / 2
        denom = cost0 - 2 * cost_half + cost_a
        if denom <= 0:
            return h
        return min(max(h + h * (cost0 - cost_a) / (2 * denom), 0.0), a)

    # -- display -------------------------------------------------------------

    @property
    def positions(self) -> np.ndarray | None:
        """Current point cloud [n, 2]."""
        if not self.frames:
            return None
        return self.frames[-1]

    def summary(self) -> str:
        """One line per recorded step."""
        if not self.loss_history:
            return "(no history)"
        lines = []
        for h in self.loss_history:
            s = h["step"]
            tag = "init" if s == -1 else f"step {s}"
            lines.append(f"  [{tag}] loss={h['cost']:.6f}  ({h['time']:.1f}s)")
        return "\n".join(lines)

    def export_html(self, out_path: str, *,
                    animate: bool | None = None,
                    point_radius: float | None = None,
                    title: str = "reconstruction",
                    **kwargs) -> "HcReconstruction":
        """Write a self-contained HTML page showing the point cloud."""
        from otrec.viz.points_html import export_positions_html

        if not self.frames:
            raise ValueError("nothing to export — no frames (record=True?)")

        animate = bool(self.frames) if animate is None else animate
        export_positions_html(
            self.frames if animate else self.positions,
            extent=float(self.extent),
            out_path=out_path,
            point_radius=point_radius,
            title=title,
            **kwargs,
        )
        return self
