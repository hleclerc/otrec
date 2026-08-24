"""Point cloud generation and multiscale splitting — pure numpy, no
dependency on `HcReconstruction` beyond the plain numbers it passes in."""
import numpy as np


def random_points(extent: float, n: int, seed: int = 0) -> np.ndarray:
    """Generate `n` random 2D points within the detector extent."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, extent / 6, size=(n, 2)).astype(np.float32)


def split_points(points: np.ndarray, extent: float, factor: int = 4,
                 noise_frac: float = 0.05) -> np.ndarray:
    """Replace each point by `factor` children with small uniform noise.

    `points` may be `[n, 2]` (position only) or `[n, 3]` (position + a
    polygon orientation `theta`, see `optim.pipeline.Polygon`) — the spatial
    columns `[:2]` always get the usual `noise_frac * extent / sqrt(n)`
    jitter; column `2`, if present, gets an independent ANGULAR jitter of
    `noise_frac * 2*pi` instead (reuses `noise_frac` rather than adding a
    separate knob — children shouldn't all inherit the exact same
    orientation as their parent).

    Returns the new point cloud `[n*factor, ndim]`.
    """
    rng = np.random.default_rng()
    n = points.shape[0]
    noise_scale = noise_frac * extent / np.sqrt(max(1, n))
    tiled = np.repeat(points, factor, axis=0)
    noise = (rng.random(tiled.shape).astype(np.float32) - 0.5) * 2
    noise[:, :2] *= noise_scale
    if tiled.shape[1] >= 3:
        noise[:, 2] *= noise_frac * 2.0 * np.pi
    return tiled + noise
