"""Simplified CT reconstruction — one class, two OT backends, no loom/sdot deps.

`HcReconstruction` builds sinograms (pure numpy), computes OT cost+gradient via
pure Jax OR a standalone SYCL fused kernel (compiled with acpp, loaded via ctypes),
solves via line search, and exports results.

The SYCL kernel (`hc_ot_sycl.cpp`) is compiled ONCE into a shared library
(`hc_ot_sycl.dylib` on macOS, `hc_ot_sycl.so` on Linux) and loaded at first use.
No `Tensor`, no `Image`, no `driver.call` — the API is pure numpy / raw float pointers.
"""
import ctypes
import functools
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


# ---- pure Jax 1D OT, DISKS variant (one angle) -----------------------------
#
# Mirrors `models.DiskModel`: the roles of source/target FLIP relative to
# `_ot1d_angle` above. The measured sinogram becomes the (fixed) SOURCE —
# weighted diracs at the bin centers, one per detector bin, no shape of its
# own — and the disk centers (the unknown) parametrise the TARGET, a
# piecewise-constant density built by projecting the union of fixed-radius
# disks onto a `nb_pixels`-wide grid (same chord-integral math as
# `disks.DiskProjector._values_of`, ported to plain arrays). Because the
# source's positions/weights are FIXED, no sort is needed here (bin centers
# are trivially sorted by construction) — unlike `_ot1d_angle`, which sorts
# the points because THEY are the (moving) source.

def _disk_mass_angle(centers, radius, nx, ny, pix_edges):
    """Projected MASS (not density) of the union of `radius`-disks at one
    angle, per pixel of `pix_edges` (`[nb_pixels + 1]`). `[nb_pixels]`.

    Same chord-integral primitive as `disks.DiskProjector._values_of`
    (`G(t) = t*sqrt(r^2-t^2) + r^2*asin(t/r)`), with the same NaN-safe
    surrogate derivative (`H = stop_grad(G - chord*u) + stop_grad(chord)*u`)
    — see that method's docstring for why the plain autodiff of `G` is
    numerically unusable near a case/disk edge.
    """
    s0 = centers[:, 0] * nx + centers[:, 1] * ny                # [ndisks]
    u = pix_edges[None, :] - s0[:, None]                        # [ndisks, nb_pixels+1]
    t = jnp.clip(u, -radius, radius)
    sq = jnp.sqrt(jnp.maximum(radius * radius - t * t, 0.0))
    chord = 2.0 * sq
    G = t * sq + radius * radius * jnp.arcsin(jnp.clip(t / radius, -1.0, 1.0))
    H = jax.lax.stop_gradient(G - chord * u) + jax.lax.stop_gradient(chord) * u
    mass = H[:, 1:] - H[:, :-1]                                 # [ndisks, nb_pixels]
    return jnp.sum(mass, axis=0)                                # [nb_pixels]


def _ot1d_disks_angle(sino_row, bin_centers_src, mass_tgt, pix_edges):
    """1D OT cost between the FIXED sinogram diracs (`sino_row` weights at
    `bin_centers_src`) and the disk-projected piecewise-constant density
    (`mass_tgt` over `pix_edges`, differentiable w.r.t. the disk centers
    through `mass_tgt`) — SOURCE/TARGET roles flipped relative to
    `_ot1d_angle` (see `models.DiskModel`'s docstring, "ils en ÉCHANGENT les
    rôles").

    A direct port of `sdot.distributions._pure_jax_cost1d.cost_1d_ot` — the
    closed-form pure-Jax path `OtPlan1d` itself uses for a dirac-source vs.
    `Image` pair — since `HcReconstruction` avoids importing loom/sdot (see
    the module docstring). Ported rather than hand re-derived: a from-scratch
    derivation (accumulating one GLOBAL target second-moment plus a per-
    dirac barycenter) agrees on well-behaved inputs but silently disagrees
    from this reference whenever the source carries a NEGATIVE weight (e.g.
    a phantom with a negative-density disk, so a `sino_row` entry < 0) —
    `cost_1d_ot`'s per-matched-window accumulation of the target's 2nd
    moment (`dM2`, not a global constant split off) is what `OtPlan1d`'s own
    C++ kernel agrees with in that case too (a known, matched quirk, see
    `cost_1d_ot`'s own docstring — not "more correct" OT theory, just the
    behaviour this reimplementation must match). Both sides are normalized
    to mass 1 internally.

    FLOAT32 FRAGILITY (found running `experiments.exp_hc_multiscale_triangles`, a real lung
    phantom): when `mass_tgt` has many THIN near-zero-mass gaps (e.g. late multiscale stages,
    where triangles pack close to their touching radius over an `nb_pixels`-fine grid), the
    `cumsum`/`searchsorted` below can lose enough precision in plain float32 to produce a
    NEGATIVE cost for some matched window — impossible in exact arithmetic (every window's
    contribution is `integral (x-p)^2 dnu(x) >= 0`). Confirmed float32-specific: the exact same
    inputs under `jax.config.update("jax_enable_x64", True)` give a sane, positive cost matching
    the (double-precision) sycl kernel. No mitigation lives HERE (would need call-site
    `jax.config`, out of scope for a pure function) — callers running the disks/triangle model
    at fine resolution should enable jax x64.
    """
    m = mass_tgt.shape[0]
    dw = pix_edges[1] - pix_edges[0]
    w_norm = sino_row / jnp.sum(sino_row)                       # [nb_bins], source mass fractions
    mass_norm = mass_tgt / jnp.maximum(jnp.sum(mass_tgt), 1e-30)  # [nb_pixels], target mass fractions

    C   = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass_norm)])
    M1e = jnp.concatenate([jnp.zeros(1), jnp.cumsum(
        mass_norm / dw * (pix_edges[1:] ** 2 - pix_edges[:-1] ** 2) / 2)])
    M2e = jnp.concatenate([jnp.zeros(1), jnp.cumsum(
        mass_norm / dw * (pix_edges[1:] ** 3 - pix_edges[:-1] ** 3) / 3)])

    order = jnp.argsort(bin_centers_src)
    p_sorted = bin_centers_src[order]
    w_sorted = w_norm[order]
    W = jnp.concatenate([jnp.zeros(1), jnp.cumsum(w_sorted)])

    c_idx = jnp.clip(jnp.searchsorted(C[1:], W, side="left"), 0, m - 1)
    y_c = mass_norm[c_idx] / dw
    e_c = pix_edges[c_idx]
    safe_y = jnp.where(y_c != 0, y_c, 1.0)
    t = jnp.where(y_c != 0, e_c + (W - C[c_idx]) / safe_y, e_c)

    M0_t = C[c_idx] + y_c * (t - e_c)
    M1_t = M1e[c_idx] + y_c * (t ** 2 - e_c ** 2) / 2
    M2_t = M2e[c_idx] + y_c * (t ** 3 - e_c ** 3) / 3

    dM0 = M0_t[1:] - M0_t[:-1]
    dM1 = M1_t[1:] - M1_t[:-1]
    dM2 = M2_t[1:] - M2_t[:-1]

    return jnp.sum(dM2 - 2 * p_sorted * dM1 + p_sorted ** 2 * dM0)


def _triangle_mass_angle(centers, radius, nx, ny, pix_edges):
    """Projected MASS of a union of TRIANGULAR ("tent") profiles, one per
    disk centre, `[nb_pixels]` — the alternative `shape` to `_disk_mass_angle`
    (see `HcReconstruction.use_disks`). `tent(v) = max(0, 1-|v|/r)`: height 1
    at the peak, support half-width `r` (so area `r`, translation-invariant,
    same normalisation convention as `hc_ot_sycl.cpp`'s continuous kernel —
    the only shape it knows how to sweep in closed form, hence `shape` being
    selectable HERE too: it lets a jax run and a sycl run be compared on the
    literal same target shape).

    Purely polynomial in the clipped local coordinate `v` — no sqrt/asin, so
    (unlike `_disk_mass_angle`) no NaN-safe surrogate derivative is needed:
    `P` below is the exact antiderivative of `tent`, built piecewise from
    `v` clipped to `[-r, r]`, continuous and correctly saturating (`P=0` for
    `v<=-r`, `P=r` for `v>=r`) by construction.
    """
    s0 = centers[:, 0] * nx + centers[:, 1] * ny                # [ndisks]
    v = pix_edges[None, :] - s0[:, None]                        # [ndisks, nb_pixels+1]
    vr = jnp.clip(v, -radius, radius)
    left  = vr + vr * vr / (2 * radius) + radius / 2            # antiderivative, v in [-r, 0]
    right = radius / 2 + vr - vr * vr / (2 * radius)            # antiderivative, v in [0, r]
    P = jnp.where(v <= 0, left, right)
    mass = P[:, 1:] - P[:, :-1]                                 # [ndisks, nb_pixels]
    return jnp.sum(mass, axis=0)                                # [nb_pixels]


#: per-disk profile, selected by `HcReconstruction.use_disks`'s `shape`.
_MASS_FNS = { "disk": _disk_mass_angle, "triangle": _triangle_mass_angle }


def _disk_cost_angle(centers, radius, nx, ny, sino_row, bin_centers_src, pix_edges, mass_fn):
    mass_tgt = mass_fn(centers, radius, nx, ny, pix_edges)
    return _ot1d_disks_angle(sino_row, bin_centers_src, mass_tgt, pix_edges)


def _disk_ot_all_angles(centers, radius, normals, sino_vals, bin_centers_src, pix_edges, mass_fn):
    """Pure Jax OT for all angles, DISKS variant. `centers` `[ndisks, 2]` is
    the only differentiated argument (see `HcReconstruction._init`, which
    wraps this in `jax.value_and_grad(argnums=0)`). `mass_fn` (one of
    `_MASS_FNS`'s values) is a plain Python callable, closed over by
    `functools.partial` before `vmap` rather than passed as a jax value."""
    cost_angle = functools.partial(_disk_cost_angle, mass_fn=mass_fn)
    costs = jax.vmap(cost_angle, in_axes=(None, None, 0, 0, 0, None, None))(
        centers, radius, normals[:, 0], normals[:, 1], sino_vals, bin_centers_src, pix_edges)
    return costs.sum()


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

    # double hc_ot_disks_cost_grad(
    #     const float* centers, int ndisks,
    #     const float* normals, int nb_angles,
    #     const float* sino_vals, int nb_bins,
    #     const float* bin_centers,
    #     float radius,
    #     float* grad)
    lib.hc_ot_disks_cost_grad.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    lib.hc_ot_disks_cost_grad.restype = ctypes.c_double

    # double hc_ot_disks_cost(
    #     const float* centers, int ndisks,
    #     const float* normals, int nb_angles,
    #     const float* sino_vals, int nb_bins,
    #     const float* bin_centers,
    #     float radius)
    lib.hc_ot_disks_cost.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_float,
    ]
    lib.hc_ot_disks_cost.restype = ctypes.c_double

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
                 backend: str = "jax", record: bool = False,
                 model: str = "diracs", radius: float | None = None,
                 nb_pixels: int | None = None, shape: str = "disk"):
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

        # -- model: "diracs" (points ARE equal-mass diracs, sinogram is the
        # fixed piecewise-constant target) or "disks" (points are FIXED-radius
        # disk centers, the measured sinogram becomes fixed weighted diracs —
        # see `models.DiskModel`'s docstring for why the roles flip). `radius`
        # is required for "disks"; `nb_pixels` is the disk-projection grid's
        # own finesse (JAX backend only — the SYCL backend's disks kernel is
        # continuous, no grid, see `use_disks`). `shape` : the per-disk PROFILE
        # used by the disks model -- "disk" (the true circular chord) or
        # "triangle" (a tent of the same support half-width `radius`, see
        # `_triangle_mass_angle`). JAX supports both, so the two can be
        # compared directly; the SYCL kernel is continuous and only knows how
        # to sweep the triangle profile in closed form (see `hc_ot_sycl.cpp`),
        # so `shape="disk"` with `backend="sycl"` raises at `_init` time.
        self.model = model
        self.radius = float(radius) if radius is not None else None
        self.nb_pixels = int(nb_pixels) if nb_pixels is not None else self.nb_bins
        self.shape = shape

        self.record = record
        self.frames: list[np.ndarray] = []
        self.loss_history: list[dict] = []
        self.timings: list[dict] = []         # per-step bracket + grad wall time
        self.debug_scans: list[dict] = []     # line-search instrumentation (see line_search)

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

    # -- model switch (diracs <-> disks) --------------------------------------

    def use_diracs(self) -> "HcReconstruction":
        """Switch to the DIRACS model: points are equal-mass diracs, the
        sinogram is the fixed piecewise-constant target (the original
        behaviour of this class). Returns self."""
        self.model = "diracs"
        self._ready = False
        return self

    def use_disks(self, radius: float, nb_pixels: int | None = None,
                  shape: str | None = None) -> "HcReconstruction":
        """Switch to the DISKS model (mirrors `models.DiskModel`): points
        become the centres of fixed-`radius` disks, and the measured
        sinogram becomes fixed weighted diracs at its bin centres — see
        `models.DiskModel`'s docstring for why the source/target roles flip
        relative to the diracs model.

        `nb_pixels` : finesse of the disk-projection grid (JAX backend only,
        default `self.nb_bins`).

        `shape` : "disk" (default, the true circular chord profile) or
        "triangle" (a tent of support half-width `radius` — see
        `_triangle_mass_angle`), `None` keeps whatever was set before (the
        constructor's own default is "disk"). JAX supports both, precisely so
        a triangle-shaped run can be compared apples-to-apples against a
        disk-shaped one. The SYCL backend's disks kernel is CONTINUOUS (no
        grid at all, see `hc_ot_sycl.cpp`'s `hc_ot_disks_cost_grad`) and only
        knows how to sweep the TRIANGLE profile in closed form — it raises at
        `_init` time if asked for `shape="disk"`.

        Returns self.
        """
        self.model = "disks"
        self.radius = float(radius)
        if nb_pixels is not None:
            self.nb_pixels = int(nb_pixels)
        if shape is not None:
            self.shape = shape
        self._ready = False
        return self

    @property
    def disk_pixel_edges(self) -> np.ndarray:
        """Edges of the disk-projection grid, `[nb_pixels + 1]` — same
        detector extent as `bin_edges`, but possibly a different resolution
        (see `use_disks`'s `nb_pixels`)."""
        dw_pix = self.extent / self.nb_pixels
        return self.s_min + dw_pix * np.arange(self.nb_pixels + 1)

    @property
    def floor(self) -> float:
        """Incompressible cost floor of the current model (0 for diracs).
        For disks: the QUANTIZATION floor of treating the measured sinogram
        as diracs at its bin centres instead of its true continuous profile
        — `dw²/12` per angle (see `models.DiskModel.floor`'s docstring)."""
        if self.model != "disks":
            return 0.0
        return self.nb_angles * self.dw ** 2 / 12

    # -- phantom factory -----------------------------------------------------

    @classmethod
    def make_lung_phantom( cls, nb_angles=600, nb_bins=2000, extent=44.0,
                          nb_alveoli=1000, alveolus_radius=2.75, scale=1.0,
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

        if self.model == "disks" and self.radius is None:
            raise ValueError("the disks model needs a radius — call `use_disks(radius=...)`")

        if self.backend == "jax":
            normals_j = jnp.asarray(self.normals)
            values_j  = jnp.asarray(self.values)

            if self.model == "diracs":
                edges_j = jnp.asarray(self.bin_edges)

                @jax.jit
                def fn(points_j):
                    return _ot_all_angles(points_j, normals_j, values_j, edges_j)
                self._cost_grad_jax = fn

            else:  # "disks"
                if self.shape not in _MASS_FNS:
                    raise ValueError(f"unknown disks shape { self.shape !r } "
                                     f"(expected one of { sorted(_MASS_FNS) })")
                mass_fn = _MASS_FNS[self.shape]
                radius = float(self.radius)
                bin_centers_src_j = jnp.asarray(self.bin_centers)
                pix_edges_j = jnp.asarray(self.disk_pixel_edges)
                cost_grad_fn = jax.value_and_grad(_disk_ot_all_angles, argnums=0)

                @jax.jit
                def fn(points_j):
                    return cost_grad_fn(points_j, radius, normals_j, values_j,
                                        bin_centers_src_j, pix_edges_j, mass_fn)
                self._cost_grad_jax = fn

        else:  # "sycl" — standalone compiled kernel, loaded via ctypes
            assert self.backend == "sycl", "jax or sycl"
            if self.model == "disks" and self.shape != "triangle":
                raise ValueError(
                    f"the sycl backend's disks kernel is CONTINUOUS and only knows how to "
                    f"sweep the triangle profile in closed form (see hc_ot_sycl.cpp) — "
                    f"got shape={ self.shape !r }. Use backend='jax' for shape='disk', "
                    f"or use_disks( ..., shape='triangle' ) here.")

            _compile_sycl_kernel()
            self._lib = _load_sycl_kernel()
            # pin numpy arrays (float32, contiguous) for ctypes
            self._normals_p = np.ascontiguousarray(
                self.normals, dtype=np.float32)
            self._values_p  = np.ascontiguousarray(
                self.values,  dtype=np.float32)
            if self.model == "diracs":
                self._edges_p = np.ascontiguousarray(
                    self.bin_edges, dtype=np.float32)
            else:  # "disks" — continuous kernel needs the source diracs'
                   # positions (bin CENTRES, no grid — see `use_disks`).
                self._bin_centers_p = np.ascontiguousarray(
                    self.bin_centers, dtype=np.float32)

        self._ready = True

    # -- OT interface --------------------------------------------------------

    def _loss_only(self, points: np.ndarray) -> float:
        """Cost only — for the bracket phase of line search."""
        self._init()
        if self.backend == "jax":
            cost, _ = self._cost_grad_jax(jnp.asarray(points))
            return float(cost)

        pts = np.ascontiguousarray(points, dtype=np.float32)
        if self.model == "diracs":
            return self._lib.hc_ot_cost(
                ctypes.c_void_p(pts.ctypes.data),
                ctypes.c_int(pts.shape[0]),
                ctypes.c_void_p(self._normals_p.ctypes.data),
                ctypes.c_int(self.nb_angles),
                ctypes.c_void_p(self._values_p.ctypes.data),
                ctypes.c_int(self.nb_bins),
                ctypes.c_void_p(self._edges_p.ctypes.data),
            )

        return self._lib.hc_ot_disks_cost(
            ctypes.c_void_p(pts.ctypes.data),
            ctypes.c_int(pts.shape[0]),
            ctypes.c_void_p(self._normals_p.ctypes.data),
            ctypes.c_int(self.nb_angles),
            ctypes.c_void_p(self._values_p.ctypes.data),
            ctypes.c_int(self.nb_bins),
            ctypes.c_void_p(self._bin_centers_p.ctypes.data),
            ctypes.c_float(self.radius),
        )

    def loss_grad(self, points: np.ndarray):
        """(float cost, ndarray grad [n, 2]) for `points` ([n, 2])."""
        self._init()
        if self.backend == "jax":
            cost, grad = self._cost_grad_jax(jnp.asarray(points))
            return float(cost), np.asarray(grad)

        pts = np.ascontiguousarray(points, dtype=np.float32)
        grad = np.zeros_like(pts)

        if self.model == "diracs":
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

        cost = self._lib.hc_ot_disks_cost_grad(
            ctypes.c_void_p(pts.ctypes.data),
            ctypes.c_int(pts.shape[0]),
            ctypes.c_void_p(self._normals_p.ctypes.data),
            ctypes.c_int(self.nb_angles),
            ctypes.c_void_p(self._values_p.ctypes.data),
            ctypes.c_int(self.nb_bins),
            ctypes.c_void_p(self._bin_centers_p.ctypes.data),
            ctypes.c_float(self.radius),
            ctypes.c_void_p(grad.ctypes.data),
        )
        return float(cost), grad

    # -- solver --------------------------------------------------------------

    def line_search(self, points, max_iter: int = 60, ftol: float = 1e-10,
                    verbose: bool = True, instrument_iters: int = 0,
                    plateau_frac: float | None = None):
        """Gradient descent with parabolic line search.

        If ``self.record`` is True, captures frames and loss history.

        ``instrument_iters`` — see ``_run_line_search``. ``plateau_frac`` —
        see ``_run_line_search``.

        Returns optimized `points` [n, 2].
        """
        def make_direction(g, s_prev, g_prev):
            return g, {}

        return self._run_line_search(points, max_iter, ftol, verbose,
                                     instrument_iters, algo="gd",
                                     make_direction=make_direction,
                                     plateau_frac=plateau_frac)

    def line_search_pr(self, points, max_iter: int = 60, ftol: float = 1e-10,
                       verbose: bool = True, instrument_iters: int = 0,
                       plateau_frac: float | None = None):
        """Nonlinear conjugate gradient (Polak-Ribière+) with parabolic line search.

        Search direction ``s_k = g_k - beta_k · s_{k-1}`` (``s_0 = g_0``), so the
        step ``p - t·s_k`` moves along ``-g_k + beta_k·s_{k-1}`` — the classic CG
        combination of the current gradient and the *previous search direction*.
        ``beta_k = max(0, <g_k, g_k - g_{k-1}> / <g_{k-1}, g_{k-1}>)`` (Polak-Ribière+;
        the ``max(0, ·)`` auto-restarts to plain gradient descent whenever the PR
        ratio goes negative — the standard safeguard against divergence).

        ``instrument_iters`` — see ``_run_line_search``.

        Behaviour is recorded the same way as ``line_search`` (``self.loss_history``
        / ``self.timings``, both tagged ``algo="pr"``; ``timings`` additionally
        carries ``beta`` and ``restarted`` per step) so runs of ``line_search`` and
        ``line_search_pr`` can be compared directly.

        Returns optimized `points` [n, 2].
        """
        def make_direction(g, s_prev, g_prev):
            if s_prev is None:
                return g, dict(beta=0.0, restarted=True)
            denom = float(np.sum(g_prev * g_prev))
            beta = float(np.sum(g * (g - g_prev)) / denom) if denom > 0 else 0.0
            restarted = beta <= 0.0
            beta = max(0.0, beta)
            return g - beta * s_prev, dict(beta=round(beta, 4), restarted=restarted)

        return self._run_line_search(points, max_iter, ftol, verbose,
                                     instrument_iters, algo="pr",
                                     make_direction=make_direction)

    def _run_line_search(self, points, max_iter, ftol, verbose, instrument_iters,
                         algo, make_direction, plateau_frac: float | None = None):
        """Shared bracket/parabola line-search loop, parametrised by search direction.

        ``make_direction(g, s_prev, g_prev) -> (s, diag)`` computes the search
        direction ``s`` (used as ``p - t·s``) for the current step from the current
        gradient ``g`` and the previous iteration's direction/gradient (``None`` on
        the first step). ``diag`` is a dict of extra per-step fields (e.g. PR's
        ``beta``) merged into the ``self.timings`` row for that step.

        ``instrument_iters`` — for the first N steps, additionally: (1) do a fine
        1D scan of cost(p - t·s) to compare the true minimum along -s against the
        3-point parabola vertex actually used, and (2) do a 2D grid scan combining
        the search direction with the *previous* step's direction (the displacement
        just taken), to see whether a step off that axis would have done better.
        Results are appended to ``self.debug_scans``.

        ``plateau_frac`` — if set, stop as soon as one step's gain
        (``cost_before - cost_after``) drops below ``plateau_frac`` times the
        FIRST step's gain, instead of running to ``max_iter``/``ftol``. Meant
        for intermediate stages of ``line_search_multiscale``: at a given
        resolution, further iterations mostly polish detail that the next
        (finer) stage will redo anyway, so it's wasted work — the same
        rationale as ``ftol``, just RELATIVE (a fixed absolute ``ftol``
        doesn't transfer across dirac counts, whose loss magnitude differs by
        orders of magnitude).
        """
        self._init()

        p = np.asarray(points, dtype=np.float32).copy()
        cost, g = self.loss_grad(p)
        g = np.asarray(g)

        t_start = time.time()
        self._record_step(-1, p, cost, t_start, algo=algo)

        g_norms = np.linalg.norm(g, axis=-1)
        mean_n = float(np.mean(g_norms)) if len(g_norms) else 1.0
        a = 0.01 * self.extent / mean_n if mean_n > 0 else 0.0

        self.debug_scans = []
        s_prev = None      # previous search direction, for the CG recursion
        g_prev = None       # previous gradient, for the CG recursion
        disp_prev = None    # previous step's displacement, for the 2D scan + colinearity
        first_gain = None    # this stage's step-0 gain, plateau_frac's reference

        for step in range(max_iter):
            s, diag = make_direction(g, s_prev, g_prev)

            t_bracket = time.time()
            a, cost_half, cost_a, tries = self._bracket(p, cost, s, a)
            t_bracket = (time.time() - t_bracket) * 1000
            d0 = -float(np.sum(g * s))
            t = self._parabola_vertex(cost, cost_half, cost_a, a, d0=d0)

            if step < instrument_iters:
                self._instrument_step(step, p, s, disp_prev, a, t, cost)

            p_new = p - t * s
            disp = p_new - p
            dir_cos = self._direction_cosine(disp, disp_prev)
            mean_disp = float(np.mean(np.linalg.norm(disp, axis=-1)))
            disp_prev = disp
            t_grad = time.time()
            cost_new, g_new = self.loss_grad(p_new)
            t_grad = (time.time() - t_grad) * 1000
            g_new = np.asarray(g_new, dtype=np.float32)
            a = 2.0 * t

            self._record_step(step, p_new, cost_new, t_start, algo=algo,
                              dir_cos=dir_cos, mean_disp=mean_disp)
            self.timings.append(dict(
                step=step, n_points=p.shape[0], backend=self.backend, algo=algo,
                t=round(float(t), 4), tries=tries, dir_cos=dir_cos,
                mean_disp=round(mean_disp, 6),
                bracket_ms=round(t_bracket, 1), grad_ms=round(t_grad, 1), **diag))

            if verbose: #  and (step % 10 == 0 or step == max_iter - 1):
                disp_mag = float(np.max(np.abs(p_new - p)))
                extra = "".join(f", {k}={v:.3g}" if isinstance(v, float) else f", {k}={v}"
                                for k, v in diag.items())
                cos_str = f", cos={dir_cos:+.3f}" if dir_cos is not None else ""
                print(f"  [{algo}] step {step:3d}: loss={cost_new:.6f}, "
                      f"max disp={disp_mag:.4g}, mean disp={mean_disp:.4g}  "
                      f"(bracket tries={tries}, t={t:.4g}{extra}{cos_str})")

            if abs(cost - cost_new) < ftol:
                if verbose:
                    print(f"  [{algo}] converged (step {step + 1}): |Δcost| < {ftol}")
                return p_new

            gain = cost - cost_new
            if first_gain is None:
                first_gain = gain
            elif plateau_frac is not None and first_gain > 0 and gain < plateau_frac * first_gain:
                if verbose:
                    print(f"  [{algo}] plateau (step {step + 1}): gain {gain:.3g} < "
                          f"{plateau_frac:.0%} of this stage's first gain ({first_gain:.3g})")
                return p_new

            s_prev, g_prev = s, g
            p, cost, g = p_new, cost_new, g_new

        return p

    # -- L-BFGS ----------------------------------------------------------------

    @staticmethod
    def _lbfgs_direction(g, s_hist, y_hist):
        """Two-loop recursion: ``H_k^{-1}·g`` from the last ``len(s_hist)``
        curvature pairs (Nocedal & Wright). Returns a field shaped like ``g``
        — used as ``p - t·direction``, so on empty history (nothing accepted
        yet) this degrades exactly to plain gradient descent.
        """
        q = g.ravel().astype(np.float64).copy()
        m = len(s_hist)
        alphas = np.empty(m)
        rhos = np.empty(m)
        for i in range(m - 1, -1, -1):
            s, y = s_hist[i], y_hist[i]
            rhos[i] = 1.0 / np.dot(y, s)
            alphas[i] = rhos[i] * np.dot(s, q)
            q -= alphas[i] * y
        if m > 0:
            s_last, y_last = s_hist[-1], y_hist[-1]
            gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
        else:
            gamma = 1.0
        r = gamma * q
        for i in range(m):
            s, y = s_hist[i], y_hist[i]
            beta = rhos[i] * np.dot(y, r)
            r += s * (alphas[i] - beta)
        return r.reshape(g.shape).astype(np.float32)

    def line_search_lbfgs(self, points, max_iter: int = 60, ftol: float = 1e-10,
                          verbose: bool = True, memory: int = 10,
                          plateau_frac: float | None = None):
        """L-BFGS: direction from the two-loop recursion over the last
        ``memory`` (position-delta, gradient-delta) pairs (``_lbfgs_direction``),
        combined with the SAME bracket+parabola step-size selection as every
        other method here — so runs differ only in DIRECTION, the actual point
        of this comparison suite, not in how the step length is chosen.

        Skips updating the curvature memory whenever ``s_k·y_k <= 0`` (the
        standard L-BFGS safeguard: that pair would make the implied Hessian
        approximation indefinite) — the step for that iteration still comes
        from the recursion over the EXISTING memory, only the update is
        dropped, same auto-restart spirit as ``line_search_pr``'s Polak-Ribière+
        safeguard. Memory is capped at ``memory`` pairs (oldest dropped first).

        ``plateau_frac`` — see ``_run_line_search``.

        Returns optimized `points` [n, 2].
        """
        self._init()

        p = np.asarray(points, dtype=np.float32).copy()
        cost, g = self.loss_grad(p)
        g = np.asarray(g, dtype=np.float32)

        t_start = time.time()
        self._record_step(-1, p, cost, t_start, algo="lbfgs")

        g_norms = np.linalg.norm(g, axis=-1)
        mean_n = float(np.mean(g_norms)) if len(g_norms) else 1.0
        a = 0.01 * self.extent / mean_n if mean_n > 0 else 0.0

        s_hist: list = []   # position deltas p_{k+1} - p_k, flattened float64
        y_hist: list = []   # gradient deltas g_{k+1} - g_k, flattened float64
        disp_prev = None
        first_gain = None

        for step in range(max_iter):
            direction = self._lbfgs_direction(g, s_hist, y_hist)

            t_bracket = time.time()
            a, cost_half, cost_a, tries = self._bracket(p, cost, direction, a)
            t_bracket = (time.time() - t_bracket) * 1000
            d0 = -float(np.sum(g * direction))
            t = self._parabola_vertex(cost, cost_half, cost_a, a, d0=d0)

            p_new = p - t * direction
            disp = p_new - p
            dir_cos = self._direction_cosine(disp, disp_prev)
            mean_disp = float(np.mean(np.linalg.norm(disp, axis=-1)))
            disp_prev = disp
            t_grad = time.time()
            cost_new, g_new = self.loss_grad(p_new)
            t_grad = (time.time() - t_grad) * 1000
            g_new = np.asarray(g_new, dtype=np.float32)
            a = 2.0 * t

            s_k = disp.ravel().astype(np.float64)
            y_k = (g_new - g).ravel().astype(np.float64)
            sy = float(np.dot(s_k, y_k))
            restarted = sy <= 1e-12
            if not restarted:
                s_hist.append(s_k); y_hist.append(y_k)
                if len(s_hist) > memory:
                    s_hist.pop(0); y_hist.pop(0)

            self._record_step(step, p_new, cost_new, t_start, algo="lbfgs",
                              dir_cos=dir_cos, mean_disp=mean_disp)
            self.timings.append(dict(
                step=step, n_points=p.shape[0], backend=self.backend, algo="lbfgs",
                t=round(float(t), 4), tries=tries, dir_cos=dir_cos,
                mean_disp=round(mean_disp, 6), mem=len(s_hist), restarted=restarted,
                bracket_ms=round(t_bracket, 1), grad_ms=round(t_grad, 1)))

            if verbose:
                disp_mag = float(np.max(np.abs(p_new - p)))
                cos_str = f", cos={dir_cos:+.3f}" if dir_cos is not None else ""
                print(f"  [lbfgs] step {step:3d}: loss={cost_new:.6f}, "
                      f"max disp={disp_mag:.4g}, mean disp={mean_disp:.4g}  "
                      f"(bracket tries={tries}, t={t:.4g}, mem={len(s_hist)}{cos_str})")

            if abs(cost - cost_new) < ftol:
                if verbose:
                    print(f"  [lbfgs] converged (step {step + 1}): |Δcost| < {ftol}")
                return p_new

            gain = cost - cost_new
            if first_gain is None:
                first_gain = gain
            elif plateau_frac is not None and first_gain > 0 and gain < plateau_frac * first_gain:
                if verbose:
                    print(f"  [lbfgs] plateau (step {step + 1}): gain {gain:.3g} < "
                          f"{plateau_frac:.0%} of this stage's first gain ({first_gain:.3g})")
                return p_new

            p, cost, g = p_new, cost_new, g_new

        return p

    # -- fine 2D grid: gradient x previous-direction -------------------------

    def _scan_grid2d(self, p, g, d_prev, t_hi, n_grid_2d, tp_range=(-0.5, 1.5),
                     max_expand=4):
        """Fine grid + local refine of cost(p + t_prev·d_prev - t_g·g).

        Used both diagnostically (``_instrument_step``, where the result is only
        logged) and as the actual step driver of ``line_search_grid2d`` (the
        reference/oracle method that always takes the grid's global minimum).

        If the refined optimum lands on a boundary of the search domain (``t_hi``
        or ``tp_range``), that boundary is grown (doubled / extended by its own
        width) and the grid+refine is redone — up to ``max_expand`` times — so a
        minimum that genuinely lies further out isn't silently clipped. ``t_g``'s
        lower bound stays fixed at 0 (stepping backward along the gradient isn't
        a useful direction to search).

        Returns ``(tg_best, tp_best, cost_best, tg_vals, tp_vals, cost_grid)`` —
        the grid/cost_grid returned are those of the LAST (possibly grown) pass.
        """
        # local import: scipy is only needed for this opt-in debug/reference path,
        # the class otherwise stays pure numpy/jax (see module docstring).
        from scipy.optimize import minimize

        def cost_2d(x):
            tg, tp = x
            return self._loss_only((p + tp * d_prev - tg * g).astype(np.float32))

        tp_lo, tp_hi = tp_range
        for _expand in range(max_expand + 1):
            tg_vals = np.linspace(0.0, t_hi, n_grid_2d)
            tp_vals = np.linspace(tp_lo, tp_hi, n_grid_2d)
            cost_grid = np.empty((n_grid_2d, n_grid_2d), dtype=np.float64)
            for i, tp in enumerate(tp_vals):
                for j, tg in enumerate(tg_vals):
                    cost_grid[i, j] = cost_2d((tg, tp))
            ij_best = np.unravel_index(int(np.argmin(cost_grid)), cost_grid.shape)

            # refine the grid's argmin with a local (bounded) search — the grid
            # is coarse (n_grid_2d points/axis) and can undershoot a sharp minimum.
            x0 = (tg_vals[ij_best[1]], tp_vals[ij_best[0]])
            res = minimize(cost_2d, x0=x0, method="Nelder-Mead",
                           bounds=[(0.0, t_hi), (tp_lo, tp_hi)],
                           options=dict(xatol=1e-6 * max(t_hi, 1.0), fatol=1e-10))
            tg_best, tp_best = float(res.x[0]), float(res.x[1])

            edge_g    = tg_best > t_hi * (1 - 1e-3)
            edge_p_hi = tp_best > tp_hi - (tp_hi - tp_lo) * 1e-3
            edge_p_lo = tp_best < tp_lo + (tp_hi - tp_lo) * 1e-3
            if not (edge_g or edge_p_hi or edge_p_lo):
                break
            if edge_g:
                t_hi *= 2.0
            if edge_p_hi:
                tp_hi += (tp_hi - tp_lo)
            if edge_p_lo:
                tp_lo -= (tp_hi - tp_lo)

        return tg_best, tp_best, float(res.fun), tg_vals, tp_vals, cost_grid

    def line_search_grid2d(self, points, max_iter: int = 60, ftol: float = 1e-10,
                           verbose: bool = True, n_grid_2d: int = 15,
                           plateau_frac: float | None = None):
        """Reference/oracle line search: always takes the global minimum of the
        fine 2D grid over ``(t_g, t_prev)`` combining the gradient direction with
        the *previous* step's displacement (``_scan_grid2d``) as the real step —
        not just for diagnostics. This upper-bounds how much a combined-direction
        line search (e.g. ``line_search_pr``) could possibly gain, at the cost of
        ``O(n_grid_2d²)`` extra loss evaluations per step (plus a local refine).

        Meant to be run once and cached (see ``viz.line_search_compare.save_run``),
        not for production use. The first step has no previous direction yet, so
        it falls back to a plain 1D bracket/parabola step (same as ``line_search``).

        ``plateau_frac`` — see ``_run_line_search``.

        Returns optimized `points` [n, 2].
        """
        self._init()

        p = np.asarray(points, dtype=np.float32).copy()
        cost, g = self.loss_grad(p)
        g = np.asarray(g)

        t_start = time.time()
        self._record_step(-1, p, cost, t_start, algo="grid2d")

        g_norms = np.linalg.norm(g, axis=-1)
        mean_n = float(np.mean(g_norms)) if len(g_norms) else 1.0
        a = 0.01 * self.extent / mean_n if mean_n > 0 else 0.0

        d_prev = None
        first_gain = None

        for step in range(max_iter):
            if d_prev is None:
                a, cost_half, cost_a, _tries = self._bracket(p, cost, g, a)
                t = self._parabola_vertex(cost, cost_half, cost_a, a, d0=-float(np.sum(g * g)))
                p_new = p - t * g
                diag = dict(tg=round(float(t), 4), tp=0.0)
            else:
                t_hi = max(2.0 * a, 1e-12)
                tg_best, tp_best, _, *_ = self._scan_grid2d(p, g, d_prev, t_hi, n_grid_2d)
                p_new = p + tp_best * d_prev - tg_best * g
                a = 2.0 * tg_best if tg_best > 1e-9 else a
                diag = dict(tg=round(tg_best, 4), tp=round(tp_best, 4))

            disp = p_new - p
            dir_cos = self._direction_cosine(disp, d_prev)
            mean_disp = float(np.mean(np.linalg.norm(disp, axis=-1)))
            d_prev = disp
            t_grad = time.time()
            cost_new, g_new = self.loss_grad(p_new)
            t_grad = (time.time() - t_grad) * 1000
            g_new = np.asarray(g_new, dtype=np.float32)

            self._record_step(step, p_new, cost_new, t_start, algo="grid2d",
                              dir_cos=dir_cos, mean_disp=mean_disp)
            self.timings.append(dict(
                step=step, n_points=p.shape[0], backend=self.backend, algo="grid2d",
                dir_cos=dir_cos, mean_disp=round(mean_disp, 6),
                grad_ms=round(t_grad, 1), **diag))

            if verbose:
                disp_mag = float(np.max(np.abs(p_new - p)))
                cos_str = f", cos={dir_cos:+.3f}" if dir_cos is not None else ""
                print(f"  [grid2d] step {step:3d}: loss={cost_new:.6f}, "
                      f"max disp={disp_mag:.4g}, mean disp={mean_disp:.4g}  "
                      f"(t_g={diag['tg']:.4g}, t_prev={diag['tp']:.4g}{cos_str})")

            if abs(cost - cost_new) < ftol:
                if verbose:
                    print(f"  [grid2d] converged (step {step + 1}): |Δcost| < {ftol}")
                return p_new

            gain = cost - cost_new
            if first_gain is None:
                first_gain = gain
            elif plateau_frac is not None and first_gain > 0 and gain < plateau_frac * first_gain:
                if verbose:
                    print(f"  [grid2d] plateau (step {step + 1}): gain {gain:.3g} < "
                          f"{plateau_frac:.0%} of this stage's first gain ({first_gain:.3g})")
                return p_new

            p, cost, g = p_new, cost_new, g_new

        return p

    # -- fine 3D grid: gradient x last two previous directions ---------------

    def _scan_grid3d(self, p, g, d_prev, d_prev2, t_hi, n_grid_3d,
                     tp_range=(-0.5, 1.5), tp2_range=(-0.5, 1.5), max_expand=3):
        """Fine 3D grid + local refine of cost(p + t_prev·d_prev + t_prev2·d_prev2 - t_g·g).

        3-axis generalisation of ``_scan_grid2d``: combines the gradient
        direction with the *two* previous steps' displacements. Grows any axis
        whose refined optimum lands on its boundary, same rationale as
        ``_scan_grid2d``. O(n_grid_3d³) loss evaluations per call — meant for
        ``line_search_grid3d``, an even more expensive reference/oracle than
        ``line_search_grid2d``; run once and cache.

        Returns ``(tg_best, tp_best, tp2_best, cost_best)``.
        """
        # local import: scipy is only needed for this opt-in debug/reference path,
        # the class otherwise stays pure numpy/jax (see module docstring).
        from scipy.optimize import minimize

        def cost_3d(x):
            tg, tp, tp2 = x
            return self._loss_only(
                (p + tp * d_prev + tp2 * d_prev2 - tg * g).astype(np.float32))

        tp_lo, tp_hi = tp_range
        tp2_lo, tp2_hi = tp2_range

        for _expand in range(max_expand + 1):
            tg_vals  = np.linspace(0.0, t_hi, n_grid_3d)
            tp_vals  = np.linspace(tp_lo, tp_hi, n_grid_3d)
            tp2_vals = np.linspace(tp2_lo, tp2_hi, n_grid_3d)
            cost_grid = np.empty((n_grid_3d, n_grid_3d, n_grid_3d), dtype=np.float64)
            for k, tp2 in enumerate(tp2_vals):
                for i, tp in enumerate(tp_vals):
                    for j, tg in enumerate(tg_vals):
                        cost_grid[k, i, j] = cost_3d((tg, tp, tp2))
            kij_best = np.unravel_index(int(np.argmin(cost_grid)), cost_grid.shape)

            x0 = (tg_vals[kij_best[2]], tp_vals[kij_best[1]], tp2_vals[kij_best[0]])
            res = minimize(cost_3d, x0=x0, method="Nelder-Mead",
                           bounds=[(0.0, t_hi), (tp_lo, tp_hi), (tp2_lo, tp2_hi)],
                           options=dict(xatol=1e-6 * max(t_hi, 1.0), fatol=1e-10))
            tg_best, tp_best, tp2_best = float(res.x[0]), float(res.x[1]), float(res.x[2])

            edge_g     = tg_best  > t_hi  * (1 - 1e-3)
            edge_p_hi  = tp_best  > tp_hi  - (tp_hi  - tp_lo)  * 1e-3
            edge_p_lo  = tp_best  < tp_lo  + (tp_hi  - tp_lo)  * 1e-3
            edge_p2_hi = tp2_best > tp2_hi - (tp2_hi - tp2_lo) * 1e-3
            edge_p2_lo = tp2_best < tp2_lo + (tp2_hi - tp2_lo) * 1e-3
            if not (edge_g or edge_p_hi or edge_p_lo or edge_p2_hi or edge_p2_lo):
                break
            if edge_g:
                t_hi *= 2.0
            if edge_p_hi:
                tp_hi += (tp_hi - tp_lo)
            if edge_p_lo:
                tp_lo -= (tp_hi - tp_lo)
            if edge_p2_hi:
                tp2_hi += (tp2_hi - tp2_lo)
            if edge_p2_lo:
                tp2_lo -= (tp2_hi - tp2_lo)

        return tg_best, tp_best, tp2_best, float(res.fun)

    def line_search_grid3d(self, points, max_iter: int = 60, ftol: float = 1e-10,
                           verbose: bool = True, n_grid_3d: int = 7,
                           plateau_frac: float | None = None):
        """Reference/oracle line search, one level up from ``line_search_grid2d``:
        combines the gradient direction with the *two* previous steps'
        displacements (``_scan_grid3d``), always taking that 3D grid's global
        minimum as the real step. Upper-bounds ``line_search_grid2d`` the same
        way ``line_search_grid2d`` upper-bounds ``line_search_pr`` — if this
        curve doesn't improve meaningfully over grid2d's, a third direction
        isn't worth chasing.

        O(n_grid_3d³) loss evaluations per step — much slower than grid2d; run
        once and cache. Falls back to ``line_search``'s 1D step on the first
        iteration (no previous direction yet) and to ``line_search_grid2d``'s
        2D scan on the second (only one previous direction yet).

        ``plateau_frac`` — see ``_run_line_search``.

        Returns optimized `points` [n, 2].
        """
        self._init()

        p = np.asarray(points, dtype=np.float32).copy()
        cost, g = self.loss_grad(p)
        g = np.asarray(g)

        t_start = time.time()
        self._record_step(-1, p, cost, t_start, algo="grid3d")

        g_norms = np.linalg.norm(g, axis=-1)
        mean_n = float(np.mean(g_norms)) if len(g_norms) else 1.0
        a = 0.01 * self.extent / mean_n if mean_n > 0 else 0.0

        d_prev = None
        d_prev2 = None
        first_gain = None

        for step in range(max_iter):
            if d_prev is None:
                a, cost_half, cost_a, _tries = self._bracket(p, cost, g, a)
                t = self._parabola_vertex(cost, cost_half, cost_a, a, d0=-float(np.sum(g * g)))
                p_new = p - t * g
                diag = dict(tg=round(float(t), 4), tp=0.0, tp2=0.0)
            elif d_prev2 is None:
                t_hi = max(2.0 * a, 1e-12)
                tg_best, tp_best, _, *_ = self._scan_grid2d(p, g, d_prev, t_hi, n_grid_3d)
                p_new = p + tp_best * d_prev - tg_best * g
                a = 2.0 * tg_best if tg_best > 1e-9 else a
                diag = dict(tg=round(tg_best, 4), tp=round(tp_best, 4), tp2=0.0)
            else:
                t_hi = max(2.0 * a, 1e-12)
                tg_best, tp_best, tp2_best, _ = self._scan_grid3d(
                    p, g, d_prev, d_prev2, t_hi, n_grid_3d)
                p_new = p + tp_best * d_prev + tp2_best * d_prev2 - tg_best * g
                a = 2.0 * tg_best if tg_best > 1e-9 else a
                diag = dict(tg=round(tg_best, 4), tp=round(tp_best, 4), tp2=round(tp2_best, 4))

            disp = p_new - p
            dir_cos = self._direction_cosine(disp, d_prev)
            mean_disp = float(np.mean(np.linalg.norm(disp, axis=-1)))
            d_prev2 = d_prev
            d_prev = disp
            t_grad = time.time()
            cost_new, g_new = self.loss_grad(p_new)
            t_grad = (time.time() - t_grad) * 1000
            g_new = np.asarray(g_new, dtype=np.float32)

            self._record_step(step, p_new, cost_new, t_start, algo="grid3d",
                              dir_cos=dir_cos, mean_disp=mean_disp)
            self.timings.append(dict(
                step=step, n_points=p.shape[0], backend=self.backend, algo="grid3d",
                dir_cos=dir_cos, mean_disp=round(mean_disp, 6),
                grad_ms=round(t_grad, 1), **diag))

            if verbose:
                disp_mag = float(np.max(np.abs(p_new - p)))
                cos_str = f", cos={dir_cos:+.3f}" if dir_cos is not None else ""
                print(f"  [grid3d] step {step:3d}: loss={cost_new:.6f}, "
                      f"max disp={disp_mag:.4g}, mean disp={mean_disp:.4g}  "
                      f"(t_g={diag['tg']:.4g}, t_prev={diag['tp']:.4g}, "
                      f"t_prev2={diag['tp2']:.4g}{cos_str})")

            if abs(cost - cost_new) < ftol:
                if verbose:
                    print(f"  [grid3d] converged (step {step + 1}): |Δcost| < {ftol}")
                return p_new

            gain = cost - cost_new
            if first_gain is None:
                first_gain = gain
            elif plateau_frac is not None and first_gain > 0 and gain < plateau_frac * first_gain:
                if verbose:
                    print(f"  [grid3d] plateau (step {step + 1}): gain {gain:.3g} < "
                          f"{plateau_frac:.0%} of this stage's first gain ({first_gain:.3g})")
                return p_new

            p, cost, g = p_new, cost_new, g_new

        return p

    # -- cheap 2D-quadratic step: 6-point cross+diagonal fit -----------------

    def _quad2d_vertex(self, p, g, d_prev, cost0, hg, hp=0.5):
        """6-point cross+diagonal quadratic fit of cost(p + tp·d_prev - tg·g)
        around (tg, tp) = (0, 0); returns the analytic vertex ``(tg, tp)`` of
        the fitted paraboloid — a 2D generalisation of ``_parabola_vertex``'s
        3-point fit, using only 5 extra evaluations (vs. ``_scan_grid2d``'s
        O(n²) grid + Nelder-Mead refine). Falls back to a plain gradient
        half-step if the fitted Hessian isn't positive-definite (no genuine
        local minimum to jump to — e.g. still on a downhill slope).

        The linear terms ``bx``, ``by`` use the EXACT directional derivatives
        at (0, 0) instead of the finite-difference estimate ``(cxp-cxm)/2hg``
        — both probe directions are straight lines through ``p`` (``-g`` and
        ``d_prev``), so ``bx = -g·g`` and ``by = g·d_prev`` are free from the
        gradient already in hand. (A cheaper 3-probe variant, dropping
        ``cxm``/``cym`` and estimating ``axx``/``ayy`` one-sided against
        these exact linear terms, was tried and measured WORSE empirically —
        the one-sided curvature estimate's O(hg) truncation error outweighed
        the saved evaluations at this step scale — so ``cxm``/``cym`` stay,
        keeping the accurate central-difference curvature.)
        """
        def cost_2d(tg, tp):
            return self._loss_only((p + tp * d_prev - tg * g).astype(np.float32))

        bx = -float(np.sum(g * g))
        by = float(np.sum(g * d_prev))

        cxp = cost_2d(hg, 0.0)
        cxm = cost_2d(-hg, 0.0)
        cyp = cost_2d(0.0, hp)
        cym = cost_2d(0.0, -hp)
        cd  = cost_2d(hg, hp)

        axx = (cxp - 2 * cost0 + cxm) / (hg * hg)
        ayy = (cyp - 2 * cost0 + cym) / (hp * hp)
        axy = (cd - cost0 - bx * hg - by * hp
              - 0.5 * axx * hg * hg - 0.5 * ayy * hp * hp) / (hg * hp)

        det = axx * ayy - axy * axy
        if axx > 0 and det > 0:
            tg, tp = np.linalg.solve(np.array([[axx, axy], [axy, ayy]]),
                                     -np.array([bx, by]))
            tg = float(np.clip(tg, 0.0, 4 * hg))
            tp = float(np.clip(tp, -2.0, 2.0))
        else:
            tg, tp = hg, 0.0

        return tg, tp

    def line_search_quad2d(self, points, max_iter: int = 60, ftol: float = 1e-10,
                           verbose: bool = True, plateau_frac: float | None = None):
        """Cheap 2D-quadratic line search: same idea as ``line_search_grid2d``
        (combine the gradient direction with the *previous* step's
        displacement) but instead of a full grid + Nelder-Mead refine, fits a
        paraboloid to 6 points (a cross + one diagonal corner, using the
        gradient's exact linear terms, see ``_quad2d_vertex``) and jumps to
        its analytic vertex — O(1) extra evaluations per step, cheap enough
        for production use (unlike ``line_search_grid2d``/``line_search_grid3d``).

        Falls back to a plain 1D bracket/parabola step (same as
        ``line_search``) on the first iteration, when there is no previous
        direction yet.

        ``plateau_frac`` — see ``_run_line_search``.

        Returns optimized `points` [n, 2].
        """
        self._init()

        p = np.asarray(points, dtype=np.float32).copy()
        cost, g = self.loss_grad(p)
        g = np.asarray(g)

        t_start = time.time()
        self._record_step(-1, p, cost, t_start, algo="quad2d")

        g_norms = np.linalg.norm(g, axis=-1)
        mean_n = float(np.mean(g_norms)) if len(g_norms) else 1.0
        a = 0.01 * self.extent / mean_n if mean_n > 0 else 0.0

        d_prev = None
        first_gain = None

        for step in range(max_iter):
            if d_prev is None:
                a, cost_half, cost_a, _tries = self._bracket(p, cost, g, a)
                t = self._parabola_vertex(cost, cost_half, cost_a, a, d0=-float(np.sum(g * g)))
                p_new = p - t * g
                diag = dict(tg=round(float(t), 4), tp=0.0)
            else:
                tg, tp = self._quad2d_vertex(p, g, d_prev, cost, hg=a / 2)
                p_new = p + tp * d_prev - tg * g
                a = 2.0 * tg if tg > 1e-9 else a
                diag = dict(tg=round(tg, 4), tp=round(tp, 4))

            disp = p_new - p
            dir_cos = self._direction_cosine(disp, d_prev)
            mean_disp = float(np.mean(np.linalg.norm(disp, axis=-1)))
            d_prev = disp
            t_grad = time.time()
            cost_new, g_new = self.loss_grad(p_new)
            t_grad = (time.time() - t_grad) * 1000
            g_new = np.asarray(g_new, dtype=np.float32)

            self._record_step(step, p_new, cost_new, t_start, algo="quad2d",
                              dir_cos=dir_cos, mean_disp=mean_disp)
            self.timings.append(dict(
                step=step, n_points=p.shape[0], backend=self.backend, algo="quad2d",
                dir_cos=dir_cos, mean_disp=round(mean_disp, 6),
                grad_ms=round(t_grad, 1), **diag))

            if verbose:
                disp_mag = float(np.max(np.abs(p_new - p)))
                cos_str = f", cos={dir_cos:+.3f}" if dir_cos is not None else ""
                print(f"  [quad2d] step {step:3d}: loss={cost_new:.6f}, "
                      f"max disp={disp_mag:.4g}, mean disp={mean_disp:.4g}  "
                      f"(t_g={diag['tg']:.4g}, t_prev={diag['tp']:.4g}{cos_str})")

            if abs(cost - cost_new) < ftol:
                if verbose:
                    print(f"  [quad2d] converged (step {step + 1}): |Δcost| < {ftol}")
                return p_new

            gain = cost - cost_new
            if first_gain is None:
                first_gain = gain
            elif plateau_frac is not None and first_gain > 0 and gain < plateau_frac * first_gain:
                if verbose:
                    print(f"  [quad2d] plateau (step {step + 1}): gain {gain:.3g} < "
                          f"{plateau_frac:.0%} of this stage's first gain ({first_gain:.3g})")
                return p_new

            p, cost, g = p_new, cost_new, g_new

        return p

    # -- gradient-informed quad2d: 2nd axis from the bracket, not history -----

    def line_search_gquad2d(self, points, max_iter: int = 60, ftol: float = 1e-10,
                            verbose: bool = True, plateau_frac: float | None = None):
        """Like ``line_search_quad2d``, but the 2D fit's *second* axis comes
        from a GRADIENT evaluated during the bracket phase, not from the
        previous accepted step's displacement.

        ``quad2d``/``grid2d``/``grid3d`` all need a previous step before they
        have a second direction to combine with ``-g`` — their first
        iteration falls back to plain gradient descent. Here, ``_bracket``
        already evaluates the LOSS at the far trial point ``p - a·g``; this
        method additionally evaluates the GRADIENT there (``g_a``, one extra
        ``loss_grad`` call — cheap, same cost class as ``quad2d``'s 5 extra
        probes). ``g_a`` differs from ``g`` by exactly the curvature the loss
        has along that line, so ``d2 = -a·g_a`` (what a gradient step of the
        SAME scale ``a`` would do FROM the endpoint) is a genuinely new
        direction, scaled to the same order of magnitude as the bracket's own
        displacement — available from the VERY FIRST step, no history needed.

        Feeds ``(g, d2)`` into the same 6-point paraboloid fit as
        ``line_search_quad2d`` (``_quad2d_vertex``) and jumps to its analytic
        vertex.

        ``plateau_frac`` — see ``_run_line_search``.

        Returns optimized `points` [n, 2].
        """
        self._init()

        p = np.asarray(points, dtype=np.float32).copy()
        cost, g = self.loss_grad(p)
        g = np.asarray(g, dtype=np.float32)

        t_start = time.time()
        self._record_step(-1, p, cost, t_start, algo="gquad2d")

        g_norms = np.linalg.norm(g, axis=-1)
        mean_n = float(np.mean(g_norms)) if len(g_norms) else 1.0
        a = 0.01 * self.extent / mean_n if mean_n > 0 else 0.0

        disp_prev = None
        first_gain = None

        for step in range(max_iter):
            a, _cost_half, _cost_a, tries = self._bracket(p, cost, g, a)
            _, g_a = self.loss_grad((p - a * g).astype(np.float32))
            g_a = np.asarray(g_a, dtype=np.float32)
            d2 = -a * g_a

            tg, tp = self._quad2d_vertex(p, g, d2, cost, hg=a / 2)
            p_new = p + tp * d2 - tg * g
            a = 2.0 * tg if tg > 1e-9 else a
            diag = dict(tg=round(tg, 4), tp=round(tp, 4), tries=tries)

            disp = p_new - p
            dir_cos = self._direction_cosine(disp, disp_prev)
            mean_disp = float(np.mean(np.linalg.norm(disp, axis=-1)))
            disp_prev = disp
            t_grad = time.time()
            cost_new, g_new = self.loss_grad(p_new)
            t_grad = (time.time() - t_grad) * 1000
            g_new = np.asarray(g_new, dtype=np.float32)

            self._record_step(step, p_new, cost_new, t_start, algo="gquad2d",
                              dir_cos=dir_cos, mean_disp=mean_disp)
            self.timings.append(dict(
                step=step, n_points=p.shape[0], backend=self.backend, algo="gquad2d",
                dir_cos=dir_cos, mean_disp=round(mean_disp, 6),
                grad_ms=round(t_grad, 1), **diag))

            if verbose:
                disp_mag = float(np.max(np.abs(p_new - p)))
                cos_str = f", cos={dir_cos:+.3f}" if dir_cos is not None else ""
                print(f"  [gquad2d] step {step:3d}: loss={cost_new:.6f}, "
                      f"max disp={disp_mag:.4g}, mean disp={mean_disp:.4g}  "
                      f"(t_g={diag['tg']:.4g}, t_prev={diag['tp']:.4g}{cos_str})")

            if abs(cost - cost_new) < ftol:
                if verbose:
                    print(f"  [gquad2d] converged (step {step + 1}): |Δcost| < {ftol}")
                return p_new

            gain = cost - cost_new
            if first_gain is None:
                first_gain = gain
            elif plateau_frac is not None and first_gain > 0 and gain < plateau_frac * first_gain:
                if verbose:
                    print(f"  [gquad2d] plateau (step {step + 1}): gain {gain:.3g} < "
                          f"{plateau_frac:.0%} of this stage's first gain ({first_gain:.3g})")
                return p_new

            p, cost, g = p_new, cost_new, g_new

        return p

    # -- coarse -> fine multiscale driver -------------------------------------

    def _absorb_stage_history(self, hist_from, tim_from, stage, step_base, time_base, dt):
        """Tag `self.loss_history[hist_from:]`/`self.timings[tim_from:]` (whatever a stage's
        `run` — or a `line_search_multiscale` `stage_callback` — just appended) with `stage`,
        and offset their `step`/`time` so they continue monotonically from the running
        `step_base`/`time_base`. Returns the updated `(step_base, time_base)`. Factored out of
        `line_search_multiscale` since a `stage_callback`'s own history needs the EXACT same
        treatment as the stage's own `run` — see that method's docstring.
        """
        max_step = step_base - 1
        for h in self.loss_history[hist_from:]:
            h["stage"] = stage
            h["step"] += step_base
            h["time"] += time_base
            max_step = max(max_step, h["step"])
        for row in self.timings[tim_from:]:
            row["stage"] = stage
            row["step"] += step_base
        return max_step + 1, time_base + dt

    def line_search_multiscale(self, method: str, nb_diracs_final: int,
                               nb_diracs_init: int = 100, factor: int = 4,
                               seed: int = 1, plateau_frac: float = 0.01,
                               max_iter: int = 60, ftol: float = 1e-10,
                               verbose: bool = True, n_grid_2d: int = 15,
                               n_grid_3d: int = 7, lbfgs_memory: int = 10,
                               split_noise_frac: float = 0.05, stage_callback=None):
        """Coarse -> fine: converge at ``nb_diracs_init`` points, then repeat
        { ``split`` (each point -> ``factor`` noisy children), reconverge }
        until ``nb_diracs_final``.

        Motivation: at a uniform random start, a single line-search method has
        to move points across the whole domain to find the right GLOBAL
        arrangement — this is exactly where the animation shows successive
        directions staying suspiciously close to each other (the gradient
        alone gives a poor descent direction there). Starting coarse (~100
        diracs) is cheap enough to sort out that global arrangement fast, and
        each finer stage then only has to refine LOCALLY around structure the
        previous stage already got right.

        Every stage except the last stops as soon as it PLATEAUS rather than
        running to ``max_iter``/``ftol`` (see ``plateau_frac`` on
        ``_run_line_search`` et al.) — polishing an intermediate resolution
        past that point is wasted, the next split makes it moot anyway. Only
        the FINAL stage (``nb_diracs_final`` points) runs to genuine
        convergence: ``plateau_frac`` is disabled there, ``max_iter``/``ftol``
        apply as usual.

        ``method`` : one of "gd" | "pr" | "lbfgs" | "quad2d" | "gquad2d" |
        "grid2d" | "grid3d" (same names as ``exp_hc_line_search_compare``).

        Accumulates continuously into ``self.loss_history``/``self.timings``/
        ``self.frames`` across ALL stages (each row tagged with a ``stage``
        index; ``step`` and ``time`` are offset to stay monotonic across
        stage boundaries, so the usual loss-vs-step / loss-vs-time plots and
        ``export_html`` animation read as one continuous run — frames are
        allowed to change point count between stages, see
        ``viz.points_html.export_positions_html``).

        ``stage_callback(stage, n, points) -> new_points | None`` : if given,
        called after EVERY stage (including the final one) converges, before
        the next split — the natural place to interleave a DIFFERENT model's
        refinement between dirac stages (e.g. a few disks/triangle LBFGS
        iterations at a shrinking radius, see
        ``experiments.exp_hc_multiscale_triangles``): switch ``self.model``
        inside the callback (``use_disks``/``use_diracs``), run whatever
        line-search you like from ``points``, switch back, and return the
        refined points. Returning ``None`` leaves ``points`` unchanged (a
        pure side effect / diagnostic probe instead of a feed-forward
        refinement). Any history/timings the callback appends (e.g. via its
        own ``line_search_lbfgs`` call) are tagged into the SAME ``stage`` and
        folded into the running ``step``/``time`` offsets, exactly like the
        stage's own ``run`` — so the combined trajectory still reads as one
        continuous timeline. NOTE: like the dirac-only path, feeding a
        refinement forward across stages relies on ``self.split`` reading
        ``self.frames[-1]`` — ``record=True`` is required for a multi-stage
        call regardless of ``stage_callback``.

        Returns the final optimized points ``[nb_diracs_final, 2]``.
        """
        fns = dict(
            gd = lambda pts, **kw: self.line_search(pts, **kw),
            pr = lambda pts, **kw: self.line_search_pr(pts, **kw),
            lbfgs = lambda pts, **kw: self.line_search_lbfgs(pts, memory=lbfgs_memory, **kw),
            quad2d = lambda pts, **kw: self.line_search_quad2d(pts, **kw),
            gquad2d = lambda pts, **kw: self.line_search_gquad2d(pts, **kw),
            grid2d = lambda pts, **kw: self.line_search_grid2d(pts, n_grid_2d=n_grid_2d, **kw),
            grid3d = lambda pts, **kw: self.line_search_grid3d(pts, n_grid_3d=n_grid_3d, **kw),
        )
        if method not in fns:
            raise ValueError(f"unknown method {method!r} (expected one of {sorted(fns)})")
        run = fns[method]

        rng = np.random.default_rng(seed)
        points = self.random_points(min(nb_diracs_init, nb_diracs_final), seed=seed)
        n = points.shape[0]

        step_base = 0    # keeps loss_history/timings 'step' monotonic across stages
        time_base = 0.0   # keeps loss_history 'time' cumulative across stages
        stage = 0
        while True:
            is_final = n >= nb_diracs_final
            if verbose:
                print(f"== multiscale stage {stage}: {n} diracs"
                     f"{'  [FINAL]' if is_final else ''} ==")

            hist_from = len(self.loss_history)
            tim_from = len(self.timings)
            t_stage0 = time.time()
            points = run(points, max_iter=max_iter, ftol=ftol, verbose=verbose,
                        plateau_frac=None if is_final else plateau_frac)
            stage_dt = time.time() - t_stage0
            step_base, time_base = self._absorb_stage_history(
                hist_from, tim_from, stage, step_base, time_base, stage_dt)

            if stage_callback is not None:
                hist_from = len(self.loss_history)
                tim_from = len(self.timings)
                t_cb0 = time.time()
                new_points = stage_callback(stage, n, points)
                cb_dt = time.time() - t_cb0
                if new_points is not None:
                    points = np.asarray(new_points, dtype=np.float32)
                step_base, time_base = self._absorb_stage_history(
                    hist_from, tim_from, stage, step_base, time_base, cb_dt)

            if is_final:
                return points

            next_n = min(n * factor, nb_diracs_final)
            children = self.split(factor=factor, noise_frac=split_noise_frac)
            if children.shape[0] > next_n:
                idx = rng.choice(children.shape[0], next_n, replace=False)
                children = children[idx]
            points = children
            n = points.shape[0]
            stage += 1

    # -- line-search instrumentation ------------------------------------------

    def _instrument_step(self, step, p, g, d_prev, a, t_parabola, cost0,
                         n_scan_1d: int = 41, n_grid_2d: int = 21):
        """Record how far the parabola vertex is from the true minimum.

        ``ts`` spans ``[0, 2a]`` — the same support the 3-point parabola
        (0, a/2, a) was fit on doubled, since ``a`` itself was set to ``2·t``
        of the previous accepted step, so the true minimum can land past it.

        The 2D grid explores ``p + t_prev·d_prev - t_g·g``: t_prev=0 is exactly
        the pure-gradient line the parabola searches, so comparing the grid's
        global minimum to the best point on that t_prev=0 axis tells us whether
        mixing in the previous step's direction would have helped.
        """
        # local import: scipy is only needed for this opt-in debug path, the
        # class otherwise stays pure numpy/jax (see module docstring).
        from scipy.optimize import minimize_scalar

        t_hi = max(2.0 * a, 1e-12)

        def cost_1d(tt):
            return self._loss_only((p - tt * g).astype(np.float32))

        # coarse curve, for plotting only — NOT the ground truth (grid resolution
        # can miss a sharp minimum, see cost_true_1d vs. minimize_scalar below).
        ts = np.linspace(0.0, t_hi, n_scan_1d)
        costs_1d = np.array([cost_1d(tt) for tt in ts])

        res_1d = minimize_scalar(cost_1d, bounds=(0.0, t_hi), method="bounded",
                                 options=dict(xatol=1e-6 * max(t_hi, 1.0)))

        record = dict(
            step=step, a=float(a), cost0=float(cost0), ts=ts, costs_1d=costs_1d,
            t_parabola=float(t_parabola),
            cost_parabola=float(cost_1d(t_parabola)),
            t_true_1d=float(res_1d.x),
            cost_true_1d=float(res_1d.fun),
        )

        if d_prev is not None:
            tg_best, tp_best, cost_best, tg_vals, tp_vals, cost_grid = \
                self._scan_grid2d(p, g, d_prev, t_hi, n_grid_2d)
            record.update(
                tg_vals=tg_vals, tp_vals=tp_vals, cost_grid=cost_grid,
                tg_best=tg_best, tp_best=tp_best, cost_grid_best=cost_best,
            )

        self.debug_scans.append(record)

    @staticmethod
    def _direction_cosine(disp, disp_prev):
        """Cosine similarity between this step's and the previous step's
        displacement field, each flattened over all points/dims — 1 means the
        two steps moved every point in (nearly) the same direction, 0 is
        orthogonal, -1 is a reversal.

        None when there's no previous step (first step of a run/stage) or the
        point count changed (a multiscale split) — the two fields then live in
        different-dimensional spaces and aren't comparable.
        """
        if disp_prev is None or disp_prev.shape != disp.shape:
            return None
        a, b = disp.ravel().astype(np.float64), disp_prev.ravel().astype(np.float64)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return None
        return float(np.dot(a, b) / (na * nb))

    def _record_step(self, step, pts, cost, t_start, algo="gd", dir_cos=None,
                     mean_disp=None):
        if not self.record:
            return
        if step == -1 and self.frames:
            return
        self.frames.append(np.array(pts, dtype=np.float32, copy=True))
        self.loss_history.append(dict(
            step=step, cost=float(cost), time=time.time() - t_start, algo=algo,
            dir_cos=dir_cos, mean_disp=mean_disp))

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
    def _parabola_vertex(cost0, cost_half, cost_a, a, d0=None):
        """Vertex of a polynomial fit through (0, cost0), (a/2, cost_half), (a, cost_a).

        ``d0`` — the analytic directional derivative ``d/dt cost(p - t·s)`` at
        ``t=0`` — is free at every call site (just ``-g·s`` from the gradient
        and direction already in hand before the bracket runs). When given,
        this upgrades the fit from a plain 3-point QUADRATIC to a Hermite
        CUBIC through ``(f(0), f'(0)=d0, f(a/2), f(a))`` — one more
        constraint than the quadratic, at zero extra evaluations. Falls back
        to the quadratic vertex (the previous behaviour) if the cubic has no
        interior minimum (degenerate leading coefficient, negative
        discriminant, or its critical point isn't a min / falls outside
        ``[0, a]``).
        """
        h = a / 2
        if d0 is not None and h > 0:
            dh = cost_half - cost0 - d0 * h
            da = cost_a - cost0 - d0 * a
            c3 = (da - 4 * dh) / (4 * h ** 3)
            c2 = dh / (h * h) - c3 * h
            disc = c2 * c2 - 3 * c3 * d0
            if c3 != 0 and disc >= 0:
                sq = disc ** 0.5
                for t in ((-c2 + sq) / (3 * c3), (-c2 - sq) / (3 * c3)):
                    if 0.0 <= t <= a and 2 * c2 + 6 * c3 * t > 0:
                        return t

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
