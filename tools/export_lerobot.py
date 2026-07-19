"""Export a recorded YAM catalog dataset to LeRobot v3 format (the Train step).

Query-native exporter: the export set is defined by a catalog query (tag filter over
``property:episode:tag``), and each episode's frames come from one
``filter_segments(id).filter_contents([4 joint streams + 3 cams]).reader(index="time",
fill_latest_at=True)`` round-trip -- the exact alignment machinery
``tools/episode_metrics.py`` scores with.

Two halves, two interpreters:

1. **Staging** (this script; rerun-sdk 0.34.1 env, no lerobot): per episode, stack
   ``left_arm/*`` + ``right_arm/*`` into 14-D ``state``/``action`` (float32, radians --
   NO unit conversion, YAM is radians end-to-end and normalization lives in the
   checkpoint's ``norm_stats.json``), drop incomplete leading rows, write
   ``state.npy``/``action.npy`` + per-camera JPEGs + ``manifest.json``.
2. **Writing** (``tools/_export_lerobot_writer.py``; the lerobot env -- e.g. the robot
   workstation's ``conda ai2_yam`` interpreter, where ``YAM/lerobot`` 0.4.3 is
   installed): builds the v3 dataset with ``LeRobotDataset.create / add_frame /
   save_episode / finalize`` using the exact schema of ``YAM/molmoact_to_lerobot_v30.py``
   (STATE_DIM_NAMES ordering, observation.images.{top,left,right}, robot_type
   molmoact_dual_arm).

Usage::

    # stage only (works in the rerun env, no lerobot needed):
    python tools/export_lerobot.py export --dataset towels --repo-id you/towels \
        --stage-only --stage-dir /tmp/towels-stage

    # stage + write in one go (writer runs as a subprocess in the lerobot env):
    python tools/export_lerobot.py export --dataset towels --repo-id you/towels \
        --writer-python "$(conda run -n ai2_yam which python)"

    # later, write a previously staged dir from inside the lerobot env directly:
    python tools/_export_lerobot_writer.py --stage /tmp/towels-stage --root datasets

    # parity gate (needs real robot data recorded through BOTH paths):
    python tools/export_lerobot.py diff --a datasets/you/towels --b <h5-converted v3 dir>

Only episodes tagged "Good episode" are exported by default (``--tag ""`` for all).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import _yam_catalog as yc

WRITER = Path(__file__).resolve().parent / "_export_lerobot_writer.py"


# --- Staging (rerun env) --------------------------------------------------------------


def stage_episode(dataset, episode: yc.Episode, out_dir: Path) -> int:
    """Query one episode and write state/action arrays + per-camera JPEGs to out_dir.

    Frame semantics (ground truth: molmoact_to_lerobot_v30.py with
    action_mode="next_joint_fields"): state = measured joints (``<arm>/position``),
    action = commanded goal joints (``<arm>/goal``), both 14-D left-then-right.
    ``fill_latest_at=True`` carries the latest value of every stream onto each row;
    rows before every stream has produced a value are dropped (incomplete leading rows).
    """
    entities = list(yc.JOINT_ENTITIES) + list(yc.CAMERA_ENTITIES)
    df = yc.episode_frames(dataset, episode.segment_id, entities)
    if df.empty:
        return 0

    out_dir.mkdir(parents=True)
    np.save(out_dir / "state.npy", yc.stack14(df, *yc.POSITION_ENTITIES).astype(np.float32))
    np.save(out_dir / "action.npy", yc.stack14(df, *yc.GOAL_ENTITIES).astype(np.float32))
    for camera in yc.CAMERA_ENTITIES:
        column = yc.blob_column(df, camera)
        cam_dir = out_dir / yc.CAMERA_KEYS[camera]
        cam_dir.mkdir()
        for index, blob in enumerate(df[column].to_numpy()):
            (cam_dir / f"{index:06d}.jpg").write_bytes(yc.blob_bytes(blob))
    return len(df)


def stage_dataset(args: argparse.Namespace, stage_dir: Path) -> int:
    client = yc.connect(args.catalog_port)
    dataset = client.get_dataset(name=args.dataset)
    episodes = yc.list_episodes(dataset, tag=args.tag if args.tag else None)
    if not episodes:
        raise SystemExit(f"no episodes{f' tagged {args.tag!r}' if args.tag else ''} in dataset '{args.dataset}'")
    print(f"exporting {len(episodes)} episode(s) from '{args.dataset}' "
          f"(state={'+'.join(yc.POSITION_ENTITIES)} action={'+'.join(yc.GOAL_ENTITIES)} "
          f"cameras={','.join(yc.CAMERA_ENTITIES)})")
    print("units: radians, unconverted (YAM convention; normalization lives in norm_stats.json)")

    staged = []
    for episode in episodes:
        frames = stage_episode(dataset, episode, stage_dir / episode.segment_id)
        if frames == 0:
            print(f"  {episode.name}: no complete frames, skipped")
            continue
        print(f"  {episode.name}: {frames} frames")
        staged.append({"dir": episode.segment_id, "name": episode.name, "task": episode.task})
    if not staged:
        raise SystemExit("nothing to export")

    (stage_dir / "manifest.json").write_text(json.dumps({
        "repo_id": args.repo_id,
        "fps": args.fps,
        "robot_type": "molmoact_dual_arm",   # matches molmoact_to_lerobot_v30.py
        "vcodec": args.vcodec,
        "state_names": yc.STATE_DIM_NAMES,   # 14-D left-then-right, verbatim from v30
        "cameras": [yc.CAMERA_KEYS[camera] for camera in yc.CAMERA_ENTITIES],  # top,left,right
        "task_instruction": args.task_instruction,
        "episodes": staged,
    }, indent=2))
    return len(staged)


def cmd_export(args: argparse.Namespace) -> None:
    output = args.root / args.repo_id
    if output.exists():
        raise SystemExit(f"{output} already exists -- remove it first to re-export")

    if args.stage_dir is not None:
        stage_dir = args.stage_dir
        if stage_dir.exists() and any(stage_dir.iterdir()):
            raise SystemExit(f"stage dir {stage_dir} is not empty")
        stage_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        stage_dir = Path(tempfile.mkdtemp(prefix="lerobot-stage-"))
        cleanup = not args.stage_only

    try:
        count = stage_dataset(args, stage_dir)
        print(f"staged {count} episode(s) -> {stage_dir}")
        if args.stage_only:
            print("\n--stage-only: stopping before the LeRobot writer. To finish, run inside the lerobot env:")
            print(f"  python {WRITER} --stage {stage_dir} --root {args.root}")
            return
        if args.writer_python is None:
            raise SystemExit(
                "--writer-python is required to run the LeRobot writer (it needs the lerobot env, "
                "e.g. --writer-python \"$(conda run -n ai2_yam which python)\"), "
                "or pass --stage-only and run tools/_export_lerobot_writer.py yourself."
            )
        command = [args.writer_python, str(WRITER), "--stage", str(stage_dir), "--root", str(args.root)]
        print(f"handing off to the LeRobot writer: {' '.join(command)}", flush=True)
        result = subprocess.run(command)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        print(f"\ndone: {output}")
    finally:
        if cleanup:
            shutil.rmtree(stage_dir, ignore_errors=True)


# --- Parity diff (needs real data recorded through both paths) ------------------------


def _load_v3_frames(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise SystemExit(f"{root} has no data/chunk-*/file-*.parquet -- not a LeRobot v3 dataset?")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def cmd_diff(args: argparse.Namespace) -> None:
    """Parity gate: diff state/action arrays + frame counts between two v3 datasets
    (catalog export vs the old h5 -> v3 converter). Reads the parquet directly, so it
    runs in the rerun env; videos are compared by frame count only.

    NOTE: this needs one real episode converted through BOTH paths -- run it on the
    robot workstation once Phase 1 has recorded real data in parallel with the
    JSON/h5 pipeline."""
    a, b = _load_v3_frames(args.a), _load_v3_frames(args.b)
    print(f"a: {args.a} -- {len(a)} frames, {a['episode_index'].nunique()} episode(s)")
    print(f"b: {args.b} -- {len(b)} frames, {b['episode_index'].nunique()} episode(s)")
    if len(a) != len(b):
        print(f"FRAME COUNT MISMATCH: {len(a)} vs {len(b)}")
    n = min(len(a), len(b))
    ok = True
    for key in ("observation.state", "action"):
        va = np.stack(a[key].to_numpy()[:n]).astype(np.float64)
        vb = np.stack(b[key].to_numpy()[:n]).astype(np.float64)
        if va.shape != vb.shape:
            print(f"{key}: SHAPE MISMATCH {va.shape} vs {vb.shape}")
            ok = False
            continue
        max_abs = float(np.abs(va - vb).max()) if n else 0.0
        status = "OK" if max_abs <= args.atol else "DIVERGES"
        if max_abs > args.atol:
            ok = False
        print(f"{key}: shape {va.shape}, max |a-b| = {max_abs:.3e}  [{status}, atol={args.atol:g}]")
    raise SystemExit(0 if ok and len(a) == len(b) else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="stage catalog episodes (+ optionally run the LeRobot v3 writer)")
    export.add_argument("--dataset", required=True, help="catalog dataset to export")
    export.add_argument("--repo-id", required=True, help="output repo id, e.g. your-hf-user/towels")
    export.add_argument("--tag", default="Good episode", help='only export episodes with this tag (--tag "" for all)')
    export.add_argument("--fps", type=int, default=30, help="frame rate stamped on the dataset (collection hz)")
    export.add_argument("--vcodec", default="h264", choices=["h264", "hevc", "libsvtav1"])
    export.add_argument("--root", type=Path, default=yc.REPO_ROOT / "datasets", help="output dir; dataset lands in <root>/<repo-id>")
    export.add_argument("--task-instruction", default=None, help="override every episode's task with one instruction")
    export.add_argument("--stage-only", action="store_true", help="stop after staging npy+JPEGs (no lerobot needed)")
    export.add_argument("--stage-dir", type=Path, default=None, help="persistent staging dir (default: temp dir)")
    export.add_argument("--writer-python", default=None, help="python interpreter of the lerobot env for the writer subprocess")
    export.add_argument("--catalog-port", type=int, default=yc.DEFAULT_CATALOG_PORT)
    export.set_defaults(func=cmd_export)

    diff = sub.add_parser("diff", help="parity gate: diff two LeRobot v3 datasets (old h5 path vs catalog export)")
    diff.add_argument("--a", type=Path, required=True, help="first v3 dataset root")
    diff.add_argument("--b", type=Path, required=True, help="second v3 dataset root")
    diff.add_argument("--atol", type=float, default=1e-6, help="max |a-b| tolerated on state/action")
    diff.set_defaults(func=cmd_diff)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except ConnectionError as error:
        raise SystemExit(f"cannot reach the catalog -- is yam_rerun/server.py running? ({error})") from None
