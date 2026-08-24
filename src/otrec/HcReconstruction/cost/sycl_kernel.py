"""SYCL kernel compilation and ctypes loading.

`hc_ot_sycl.cpp` (this directory) is compiled ONCE into a shared library
(`hc_ot_sycl.dylib` on macOS, `hc_ot_sycl.so` on Linux) and loaded via ctypes.
No `Tensor`, no `Image`, no `driver.call` — standalone, pure numpy / raw float
pointers. `sycl_cost.py`'s `SyclDiracsCost`/`SyclDisksCost` are the only
callers; this module just gets a usable `CDLL` into their hands.
"""
import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

_KERNEL_DIR = Path(__file__).resolve().parent
_KERNEL_CPP = _KERNEL_DIR / "hc_ot_sycl.cpp"
# Shared library extension: .dylib on macOS, .so on Linux (the container).
_SO_EXT   = ".dylib" if sys.platform == "darwin" else ".so"
_KERNEL_SO = _KERNEL_DIR / f"hc_ot_sycl{_SO_EXT}"


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


def compile_kernel():
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


def load_kernel_lib() -> ctypes.CDLL:
    """Compile (if needed) and return a ctypes CDLL with the four SYCL entry points."""
    compile_kernel()
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
