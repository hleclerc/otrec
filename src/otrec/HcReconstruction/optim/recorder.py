"""Per-step bookkeeping shared by every `LineSearch`: point-cloud frames (for
HTML animation), loss/timing history (for plots), and opt-in debug scans.
"""
import time

import numpy as np


class Recorder:
    """Accumulates state across one or more `LineSearch.run` calls.

    Disabled (`enabled=False`) recorders track nothing — cheap to always
    create and pass around, so every `LineSearch.run` can assume it has one
    (see `LineSearch`'s docstring: `recorder=None` defaults to a disabled one).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.frames: list[np.ndarray] = []
        self.frame_radii: list[float] = []   # disks radius per frame (0.0 for diracs)
        self.loss_history: list[dict] = []
        self.timings: list[dict] = []         # per-step bracket + grad wall time
        self.debug_scans: list[dict] = []     # line-search instrumentation, see optim.instrumentation

    @property
    def positions(self) -> np.ndarray | None:
        """Most recently recorded point cloud [n, 2], or None."""
        return self.frames[-1] if self.frames else None

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

    @staticmethod
    def direction_cosine(disp, disp_prev):
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

    def record_step(self, step, points, cost, t_start, *, algo: str, radius: float = 0.0,
                    dir_cos=None, mean_disp=None):
        if not self.enabled:
            return
        if step == -1 and self.frames:
            return
        self.frames.append(np.array(points, dtype=np.float32, copy=True))
        self.frame_radii.append(float(radius))
        self.loss_history.append(dict(
            step=step, cost=float(cost), time=time.time() - t_start, algo=algo,
            dir_cos=dir_cos, mean_disp=mean_disp))
