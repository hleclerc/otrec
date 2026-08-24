"""HTML visualization export — a free function over a `Recorder` + `CtGeometry`."""
import numpy as np


def export_html(recorder, geometry, out_path: str, *,
                animate: bool | None = None,
                point_radius: float | None = None,
                title: str = "reconstruction",
                n_sides: int | None = None,
                **kwargs):
    """Write a self-contained HTML page showing the point cloud in `recorder`.

    Unless the caller already passes `radii` (or `vertex_offsets`) explicitly, this
    defaults `radii` from `recorder.frame_radii` — the disks radius recorded alongside each
    frame (see `optim.recorder.Recorder.record_step`) — so markers are drawn at their TRUE model
    size instead of the generic display-only default, even across a run that interleaves
    diracs stages (radius 0) with disks/triangle stages at different (e.g. shrinking) radii
    (see `optim.pipeline.MultiscaleStage`, whose `inner` `Stage` can switch models freely
    between sub-stages). Skipped entirely for a
    pure-diracs run (all recorded radii are 0) so the historical "slider IS the radius"
    behaviour is unchanged.

    `n_sides` (typically passed automatically by `HcReconstruction.export_html`
    when its current model is "polygon", see `optim.pipeline.Polygon`): when
    given and the recorded frames are `[n, 3]` (`x, y, theta` — a POLYGON
    run, see `cost.jax_polygon`), each point's TRUE rotated regular
    `n_sides`-gon is derived from its `theta` column and `frame_radii`, and
    passed as `vertex_offsets` (see `viz.points_html.export_positions_html`)
    — the model drives its own rendering, no `marker` needed. `positions`
    then gets only the `[:, :2]` columns (`vertex_offsets` is world-unit
    offsets from the center, `theta` itself isn't a canvas coordinate).
    Ignored if the caller already passes `radii`/`vertex_offsets` explicitly,
    or if frames are plain `[n, 2]`. Out of scope: a single export mixing
    polygon and non-polygon frames (a pipeline that switches models mid-run).
    """
    from otrec.viz.points_html import export_positions_html

    if not recorder.frames:
        raise ValueError("nothing to export — no frames (record=True?)")

    animate = bool(recorder.frames) if animate is None else animate
    frame_list = recorder.frames if animate else [recorder.positions]
    is_polygon = (n_sides is not None and "radii" not in kwargs
                 and "vertex_offsets" not in kwargs
                 and frame_list and frame_list[0].shape[1] >= 3)

    if is_polygon:
        radii_by_frame = recorder.frame_radii if animate else recorder.frame_radii[-1:]
        unit_k = np.arange(n_sides)
        unit = np.stack([np.cos(2 * np.pi * unit_k / n_sides),
                         np.sin(2 * np.pi * unit_k / n_sides)], axis=1)  # [n_sides, 2]
        vo_by_frame = []
        for f, r in zip(frame_list, radii_by_frame):
            theta = f[:, 2]
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot_x = cos_t[:, None] * unit[None, :, 0] - sin_t[:, None] * unit[None, :, 1]
            rot_y = sin_t[:, None] * unit[None, :, 0] + cos_t[:, None] * unit[None, :, 1]
            vo_by_frame.append((np.stack([rot_x, rot_y], axis=-1) * r).astype(np.float32))
        kwargs["vertex_offsets"] = vo_by_frame
        frames_arg = [f[:, :2] for f in frame_list] if animate else frame_list[0][:, :2]
    else:
        frames_arg = recorder.frames if animate else recorder.positions
        if "radii" not in kwargs and "vertex_offsets" not in kwargs:
            radii_by_frame = recorder.frame_radii if animate else recorder.frame_radii[-1:]
            if radii_by_frame and any(radii_by_frame) and len(radii_by_frame) == len(frame_list):
                kwargs["radii"] = [
                    np.full(len(f), r, dtype=np.float32)
                    for f, r in zip(frame_list, radii_by_frame)
                ]

    export_positions_html(
        frames_arg,
        extent=float(geometry.extent),
        out_path=out_path,
        point_radius=point_radius,
        title=title,
        **kwargs,
    )
