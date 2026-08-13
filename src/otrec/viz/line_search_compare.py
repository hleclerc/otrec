"""Persist and compare `HcReconstruction` line-search runs across experiments.

Each named run (e.g. "gd", "pr", "grid2d") is saved as one JSON file in a results
directory; saving under the same name OVERWRITES the previous result there — the
point, e.g. after improving that method's implementation and wanting to replace
its stale curve. Loading scans the directory for every `*.json` file, so a single
new experiment run can regenerate a comparison plot spanning every method ever
saved there, including expensive reference runs (like the fine 2D grid oracle)
that don't need to be re-run just because another method changed.

Each run's HTML animation (see ``HcReconstruction.export_html``) is saved
alongside its JSON as ``{results_dir}/{name}.html`` by the calling experiment
script (not by this module — ``save_run`` only persists the numeric history),
so the point-cloud trajectory of any accumulated run can be inspected visually.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_run(results_dir, name: str, meta: dict, loss_history: list, timings: list) -> Path:
    """Save one named run, overwriting `{results_dir}/{name}.json` if present."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    payload = dict(
        name=name, saved_at=time.time(), meta=meta,
        loss_history=loss_history, timings=timings,
    )
    out_path.write_text(json.dumps(payload))
    return out_path


def load_runs(results_dir) -> dict:
    """Return ``{name: payload}`` for every `*.json` file in `results_dir`."""
    out_dir = Path(results_dir)
    runs = {}
    for f in sorted(out_dir.glob("*.json")):
        payload = json.loads(f.read_text())
        runs[payload["name"]] = payload
    return runs


def _warn_config_mismatches(runs: dict):
    """Runs compare fairly only if they solved the same problem — flag it if not."""
    keys = ("nb_diracs", "nb_angles", "nb_alveoli", "backend", "seed")
    configs = {name: {k: r["meta"].get(k) for k in keys} for name, r in runs.items()}
    ref_name, ref = next(iter(configs.items()))
    for name, cfg in configs.items():
        if cfg != ref:
            print(f"  ⚠ run {name!r} config differs from {ref_name!r}: {cfg} vs {ref}")


# Reference/oracle methods evaluate O(n_grid^k) points per step, so their wall
# time is orders of magnitude above any production method's — mixing them into
# the loss-vs-time panel collapses every other curve onto the y-axis. They stay
# in the loss-vs-step panel (that's the whole point of having them: an upper
# bound on achievable convergence), just excluded from the time axis.
_SLOW_METHODS = {"grid2d", "grid3d"}


def plot_comparison(runs: dict, out_path):
    """2x2 grid: loss-vs-step, loss-vs-wall-time, direction-colinearity-vs-step
    and mean-displacement-vs-step, for every run in `runs`.

    The colinearity panel plots ``dir_cos`` — cos(displacement_k, displacement_{k-1})
    over the whole point cloud (see ``HcReconstruction._direction_cosine``) — a
    value near +1 means consecutive steps moved (almost) every point the same
    way, i.e. the line search is repeatedly re-taking the same step instead of
    exploring; a value near 0 or negative means directions are genuinely
    changing. The displacement panel plots ``mean_disp`` — the mean per-dirac
    Euclidean displacement that step (log scale, since it typically decays over
    orders of magnitude as the optimizer converges). Both are ``None`` on the
    first step of a run/multiscale stage where there's nothing prior to compare
    against (colinearity) or, for displacement, never (a step's own size is
    always defined) — plotted as ``NaN`` (not skipped), so a missing value
    shows as a real gap in the line rather than silently connecting two
    unrelated stages.

    Wall-time-side, runs whose ``meta.method`` is in ``_SLOW_METHODS`` are
    dropped (see its docstring) — they still appear in the other panels.
    """
    if not runs:
        return
    _warn_config_mismatches(runs)

    names = sorted(runs)
    palette = plt.cm.tab10.colors
    color_of = {name: palette[i % len(palette)] for i, name in enumerate(names)}
    time_names = [n for n in names if runs[n]["meta"].get("method") not in _SLOW_METHODS]
    skipped = [n for n in names if n not in time_names]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))

    ax = axes[0, 0]
    for name in names:
        hist = [h for h in runs[name]["loss_history"] if h["step"] >= 0]
        ax.plot([h["step"] for h in hist], [h["cost"] for h in hist],
               "-o", ms=3, color=color_of[name], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("cost")
    ax.set_title("loss vs. step")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name in time_names:
        hist = [h for h in runs[name]["loss_history"] if h["step"] >= 0]
        ax.plot([h["time"] for h in hist], [h["cost"] for h in hist],
               "-o", ms=3, color=color_of[name], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("cost")
    title = "loss vs. time"
    if skipped:
        title += f"  (excl. {', '.join(skipped)}: too slow to share this axis)"
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name in names:
        hist = [h for h in runs[name]["loss_history"] if h["step"] >= 0]
        if not hist:
            continue
        steps = [h["step"] for h in hist]
        cos = [h["dir_cos"] if h.get("dir_cos") is not None else float("nan") for h in hist]
        ax.plot(steps, cos, "-o", ms=3, color=color_of[name], label=name)
    ax.axhline(1.0, color="0.6", lw=0.7, ls=":")
    ax.axhline(0.0, color="0.6", lw=0.7, ls="-")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("step")
    ax.set_ylabel(r"cos(step$_k$, step$_{k-1}$)")
    ax.set_title("direction colinearity", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for name in names:
        hist = [h for h in runs[name]["loss_history"] if h["step"] >= 0]
        if not hist:
            continue
        steps = [h["step"] for h in hist]
        md = [h["mean_disp"] if h.get("mean_disp") is not None else float("nan") for h in hist]
        ax.plot(steps, md, "-o", ms=3, color=color_of[name], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("mean |Δp| per dirac")
    ax.set_title("mean displacement", fontsize=9)
    ax.legend(fontsize=8)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
