"""Lung phantom site layout: a jittered hexagonal grid of alveolus candidates.

`HcReconstruction.make_lung_phantom` is the actual factory (it needs `cls(...)`
to build the geometry/sinogram); this module only holds the pure-geometry
helper it calls.
"""
import numpy as np


def hex_sites(lobes, max_radius, spacing_factor: float = 2.15,
             jitter_factor: float = 0.02, seed: int = 0):
    """Candidate alveolus centres on a jittered hexagonal grid, restricted to
    the interior of `lobes` (a list of `(center, radius)` disks). Returns
    `(sites, rng)` — the caller reuses `rng` to pick radii for the sites it keeps.
    """
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
