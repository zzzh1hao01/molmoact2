"""Rerun blueprint for bimanual-YAM episodes.

Layout (adapted from the so100-hackathon reference ``blueprint.py``):

* top: 3D dual-URDF view (left) + a vertical stack of the three cameras (right),
* bottom: tabs with one position-vs-goal ``TimeSeriesView`` per arm (sliding
  cursor-relative window) and a diagnostics grid (velocities / whatever else the
  collection hook logs under the diagnostics subpaths).

Entity-path contract (must match the collection hook in
``YAM/gello_software/experiments/launch_yaml_collect_data.py``):

* ``camera/top``, ``camera/left``, ``camera/right`` -- JPEG ``EncodedImage`` per tick
  (``front_camera_rgb`` maps to ``camera/top``, per ``YAM/molmoact_to_lerobot_v30.py``).
* ``left_arm/position``, ``right_arm/position`` -- 7 scalars each (14-D state split
  ``[left_joint1..6, left_gripper | right_joint1..6, right_gripper]``).
* ``left_arm/goal``, ``right_arm/goal`` -- the gello leader command (the action).
* URDF subtrees live under the same ``left_arm``/``right_arm`` prefixes (Phase 0's
  ``yam_rerun.urdf_yam``); their *visual geometry* paths are passed in explicitly so
  the 3D view shows meshes only, not scalar/transform clutter.

Footgun (from the port plan): a blueprint file must NOT be rewritten while it is
registered on the catalog -- use :func:`register_dataset_blueprint`, which goes through
``takes.save_dataset_blueprint`` (write-once) + ``takes.register_blueprint``.
"""

from __future__ import annotations

from pathlib import Path

import rerun.blueprint as rrb

from yam_rerun.takes import register_blueprint, save_dataset_blueprint

ARM_NAMES = ("left_arm", "right_arm")
CAMERA_PATHS = ("camera/top", "camera/left", "camera/right")

# entity subpath -> plot title, for the diagnostics tab. Everything here is logged but
# irrelevant to training (which only consumes positions + images) -- it's rig health.
# The collection hook currently logs `velocity` (from RobotEnv's joint_velocities);
# i2rt motor telemetry (temperature/current) can be added later under the same scheme.
DIAGNOSTICS: dict[str, str] = {
    "velocity": "joint velocity (rad/s)",
}


def create_blueprint(
    *,
    camera_paths: tuple[str, ...] = CAMERA_PATHS,
    visual_paths: list[str] | None = None,
    window_seconds: float = 10.0,
) -> rrb.Blueprint:
    """Sliding-window layout for realtime viewing and catalog playback.

    ``visual_paths`` are the URDF visual-geometry subtree roots (one per arm); with none
    given the 3D pane is omitted and the cameras take the full top row.
    """
    time_ranges = rrb.VisibleTimeRange(
        "time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-window_seconds),
        end=rrb.TimeRangeBoundary.cursor_relative(),
    )

    # <arm>/goal is the gello leader command (lerobot's "action"); <arm>/position is the
    # follower's measured state. Overlaying them makes tracking lag/error visible.
    # <arm>/pred* only exists on episodes with eval layers (tools/eval_policy.py;
    # pred = default checkpoint, pred1k etc. = comparison checkpoints) -- including
    # them here is harmless otherwise and overlays the policies' predictions.
    position_tabs = [
        rrb.TimeSeriesView(
            name=f"{name} position vs goal",
            origin=name,
            contents=["+ $origin/position", "+ $origin/goal", "+ $origin/pred", "+ $origin/pred1k"],
            time_ranges=time_ranges,
        )
        for name in ARM_NAMES
    ]
    diagnostics = rrb.Grid(
        *[
            rrb.TimeSeriesView(name=f"{name} {title}", origin=f"{name}/{subpath}", time_ranges=time_ranges)
            for name in ARM_NAMES
            for subpath, title in DIAGNOSTICS.items()
        ],
        grid_columns=max(1, len(DIAGNOSTICS)),
        name="diagnostics",
    )
    tabs = rrb.Tabs(*position_tabs, diagnostics, active_tab=0)

    arms_3d = (
        rrb.Spatial3DView(
            name="arms",
            origin="/",
            # Include ONLY the URDF visual meshes: no cameras, collision meshes, or
            # transform/scalar entities cluttering the view's entity tree. Ancestor
            # transforms still apply -- contents filters visibility, not the hierarchy.
            # ghost_* mirrors the arm paths for episodes with an eval layer
            # (tools/eval_policy.py): the policy's predicted trajectory rendered as
            # translucent ghost arms; harmless when absent.
            contents=[f"+ /{path.lstrip('/')}/**" for path in visual_paths or []]
            + [f"+ /ghost_{path.lstrip('/')}/**" for path in visual_paths or []]
            + [f"+ /ghost1k_{path.lstrip('/')}/**" for path in visual_paths or []]
            + ["+ /labels/**"],
            # Default eye BEHIND the arms (operator/top-camera viewpoint; the arms
            # reach toward +x, left arm anchored at +y). Without this the viewer
            # auto-frames from the far side and the arms read as left/right-swapped
            # relative to the camera panes.
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=(-1.4, 0.0, 1.0),
                look_target=(0.25, 0.0, 0.2),
                eye_up=(0.0, 0.0, 1.0),
            ),
        )
        if visual_paths
        else None
    )
    camera_views = [rrb.Spatial2DView(name=path.rsplit("/", 1)[-1], origin=path) for path in camera_paths]

    # Cameras stack vertically so the 3D view keeps most of the width (3:2 split).
    cameras = rrb.Vertical(*camera_views) if len(camera_views) > 1 else (camera_views[0] if camera_views else None)
    panes = [pane for pane in (arms_3d, cameras) if pane is not None]
    if not panes:
        return rrb.Blueprint(tabs, collapse_panels=True)
    spatial = rrb.Horizontal(*panes, column_shares=[3, 2]) if len(panes) == 2 else panes[0]
    return rrb.Blueprint(rrb.Vertical(spatial, tabs, row_shares=[2, 1]), collapse_panels=True)


def register_dataset_blueprint(
    catalog_uri: str,
    recordings_dir: Path,
    dataset: str,
    *,
    visual_paths: list[str] | None = None,
    window_seconds: float = 10.0,
) -> bool:
    """Make this layout the dataset's default blueprint on the catalog.

    Write-once on disk (``recordings/<dataset>/blueprint/blueprint.rrd``; delete the file
    to regenerate) and registered at most once per catalog lifetime -- the file is never
    rewritten while registered (doing so breaks the live registration with 'malformed
    response'). Returns True if this call actually set the default.
    """
    blueprint = create_blueprint(visual_paths=visual_paths, window_seconds=window_seconds)
    path = save_dataset_blueprint(recordings_dir, dataset, blueprint)
    return register_blueprint(catalog_uri, dataset, path)
