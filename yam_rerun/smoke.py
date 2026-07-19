"""Smoke test: animate both YAM arms from a synthetic 14-D sine trajectory.

Headless (CI) usage — writes a .rrd you can open or query later:

    /Users/zhihao/molmoact2/.venv-rerun/bin/python -m yam_rerun.smoke --save out.rrd

Interactive usage — opens the native viewer:

    /Users/zhihao/molmoact2/.venv-rerun/bin/python -m yam_rerun.smoke --spawn

Each joint sweeps a sine around the middle of its URDF limits (grippers sweep
0..1, logged as scalars only — no gripper joint in yam.urdf). Alongside the FK
transforms, the 7-D per-arm state is logged as ``left_arm/position`` and
``right_arm/position`` scalars, matching the Phase-1 entity naming in
rerun-yam-port-plan.md.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import rerun as rr

from yam_rerun.urdf_yam import GRIPPER_STATE_INDEX, STATE_DIM, DualYam

APP_ID = "yam"


def synthetic_state(robot: DualYam, t: float) -> np.ndarray:
    """A smooth in-limits 14-D state at time ``t`` (seconds)."""
    state = np.zeros(STATE_DIM, dtype=np.float32)
    for arm_phase, arm in zip((0.0, math.pi / 2), robot.arms):
        for joint_index, (joint, state_index) in enumerate(
            zip(arm.joints, arm.state_indices)
        ):
            mid = (joint.limit_lower + joint.limit_upper) / 2.0
            amp = 0.45 * (joint.limit_upper - joint.limit_lower) / 2.0
            freq = 0.25 + 0.1 * joint_index  # Hz; slightly detuned per joint
            phase = arm_phase + joint_index * math.pi / 6
            state[state_index] = mid + amp * math.sin(2 * math.pi * freq * t + phase)
        # Gripper: normalized 0..1 open/close sweep (scalar only; no URDF joint).
        state[GRIPPER_STATE_INDEX[arm.name]] = 0.5 + 0.5 * math.sin(
            2 * math.pi * 0.5 * t + arm_phase
        )
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", type=Path, default=None, metavar="OUT.rrd",
                        help="write the recording to this .rrd (headless)")
    parser.add_argument("--spawn", action="store_true",
                        help="open the native Rerun viewer")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds of trajectory")
    parser.add_argument("--fps", type=float, default=30.0, help="trajectory sample rate")
    parser.add_argument("--spacing", type=float, default=0.6,
                        help="distance between the two arm bases (meters, along y)")
    args = parser.parse_args()
    if args.save is None and not args.spawn:
        parser.error("pass --save OUT.rrd (headless) and/or --spawn (viewer)")

    rec = rr.RecordingStream(APP_ID, recording_id=f"smoke_{int(time.time())}")
    if args.save is not None:
        rec.save(str(args.save))
    if args.spawn:
        rec.spawn()

    robot = DualYam.create(spacing=args.spacing)
    robot.log_static(rec)

    n_frames = int(args.duration * args.fps)
    t0 = time.time()
    for frame in range(n_frames):
        t = frame / args.fps
        rec.set_time("time", timestamp=t0 + t)
        state = synthetic_state(robot, t)
        robot.log_state(rec, state)
        for arm in robot.arms:
            lo, hi = arm.state_indices[0], arm.gripper_state_index
            rec.log(f"{arm.name}/position", rr.Scalars(state[lo : hi + 1]))

    rec.flush()
    if args.save is not None:
        print(f"wrote {n_frames} frames ({args.duration:g}s @ {args.fps:g}Hz) to {args.save}")
    if args.spawn:
        print("viewer spawned; ctrl-C to exit")


if __name__ == "__main__":
    main()
