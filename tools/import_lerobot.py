"""Import a LeRobot v3 dataset into the Rerun catalog (the inverse of export_lerobot.py).

Turns each LeRobot episode into a standard YAM take: a ``recordings/<dataset>/<episode>.rrd``
following the Phase-1 logging contract (``camera/{top,left,right}`` JPEG ``EncodedImage``,
``{left,right}_arm/{position,goal}`` 7-D ``Scalars``, URDF FK transforms, ``time`` timeline,
``property:episode:*`` metadata), so imported demonstrations are indistinguishable from
episodes recorded live -- the viewer blueprint, ``query_dataset.py``, ``episode_metrics.py``
and even a re-export all work on them unchanged.

Reads the v3 layout directly (``meta/info.json``, ``meta/episodes/chunk-*/file-*.parquet``,
``data/chunk-*/file-*.parquet``, ``videos/<key>/chunk-*/file-*.mp4``) with pandas + PyAV --
deliberately NO ``lerobot`` import, so it runs in the ``.venv-rerun`` interpreter like every
other tool here. v3 concatenates episodes into shared mp4 files; each episode's frames are
the ``[from_timestamp, to_timestamp)`` slice recorded in the episode metadata, decoded and
re-encoded to JPEG (matching what the live collection hook logs).

Frame semantics (same ground truth as the exporter, molmoact_to_lerobot_v30.py):
``observation.state`` -> ``<arm>/position`` (measured, drives FK), ``action`` -> ``<arm>/goal``
(commanded). 14-D left-then-right, radians, unconverted.

Usage (catalog registration is optional by design -- yam_rerun/server.py re-registers
everything under ``recordings/`` on startup, so importing while the server is down is fine)::

    # import every episode of a v3 dataset as recordings/<name>/episode_NNN.rrd:
    python tools/import_lerobot.py --source datasets/you/towels

    # subset, custom catalog dataset name, pre-blessed for re-export:
    python tools/import_lerobot.py --source datasets/you/towels --dataset towels_demos \
        --episodes 0,2,5 --tag "Good episode"
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _yam_catalog as yc

sys.path.insert(0, str(yc.REPO_ROOT))  # tools/ scripts run with sys.path[0] == tools/

from yam_rerun import blueprint as bp  # noqa: E402
from yam_rerun import takes  # noqa: E402
from yam_rerun.urdf_yam import STATE_DIM, STATE_DIM_NAMES, DualYam  # noqa: E402

import rerun as rr  # noqa: E402

# v3 path templates (fallbacks; info.json's data_path/video_path win when present).
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


# --- Reading the v3 layout (no lerobot import) ----------------------------------------


def load_info(source: Path) -> dict:
    info_path = source / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"{source} has no meta/info.json -- not a LeRobot v3 dataset?")
    return json.loads(info_path.read_text())


def load_episodes_meta(source: Path) -> pd.DataFrame:
    files = sorted((source / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise SystemExit(f"{source} has no meta/episodes/chunk-*/file-*.parquet -- not v3?")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def video_features(info: dict, camera_map: dict[str, str]) -> dict[str, str]:
    """LeRobot video feature key -> camera entity path (observation.images.top -> camera/top).

    ``camera_map`` renames key stems (e.g. ``{"wrist_1": "right"}`` puts
    ``observation.images.wrist_1`` under ``camera/right``).
    """
    keys = [key for key, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]
    if not keys:
        raise SystemExit("dataset has no video features -- image-in-parquet datasets are not supported")
    unknown = set(camera_map) - {key.rsplit(".", 1)[-1] for key in keys}
    if unknown:
        raise SystemExit(f"--camera-map names not in the dataset's video features: {sorted(unknown)}")
    return {
        key: f"camera/{camera_map.get(key.rsplit('.', 1)[-1], key.rsplit('.', 1)[-1])}"
        for key in keys
    }


def check_state_features(info: dict) -> None:
    """The 14-D contract is what makes the URDF/FK + entity split valid -- enforce it."""
    features = info.get("features", {})
    for key in ("observation.state", "action"):
        feature = features.get(key)
        if feature is None:
            raise SystemExit(f"dataset has no '{key}' feature")
        shape = tuple(feature.get("shape", ()))
        if shape != (STATE_DIM,):
            raise SystemExit(
                f"'{key}' has shape {shape}, expected ({STATE_DIM},) -- this importer is "
                f"specific to the bimanual-YAM 14-D contract (molmoact_to_lerobot_v30.py)"
            )
        names = feature.get("names")
        if names and list(names) != list(STATE_DIM_NAMES):
            print(f"warning: '{key}' names differ from STATE_DIM_NAMES; importing by position:")
            print(f"  theirs: {list(names)}")


def episode_frames(source: Path, info: dict, ep: pd.Series) -> pd.DataFrame:
    data_path = info.get("data_path") or DEFAULT_DATA_PATH
    file = source / data_path.format(
        chunk_index=int(ep["data/chunk_index"]), file_index=int(ep["data/file_index"])
    )
    df = pd.read_parquet(file)
    df = df[df["episode_index"] == int(ep["episode_index"])]
    return df.sort_values("frame_index").reset_index(drop=True)


def stack_rows(df: pd.DataFrame, key: str) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32) for v in df[key]])


def episode_task(ep: pd.Series) -> str:
    tasks = ep.get("tasks")
    if tasks is None:
        return ""
    if isinstance(tasks, str):
        return tasks
    values = list(np.asarray(tasks).ravel())
    return str(values[0]) if values else ""


# --- Video slice -> JPEG frames -------------------------------------------------------


def decode_video_slice(
    path: Path, from_ts: float, to_ts: float, fps: float, jpeg_quality: int
) -> list[bytes]:
    """Decode the episode's ``[from_ts, to_ts)`` slice of a (shared, concatenated) mp4.

    Seeks to the keyframe at/before ``from_ts`` (v3 episodes are encoded one at a time
    and concatenated, so episode boundaries are keyframes), then decodes forward,
    keeping frames whose presentation time falls inside the slice -- with a half-frame
    tolerance so exact-boundary timestamps land on the right side.
    """
    import av

    half_frame = 0.5 / fps
    frames: list[bytes] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if from_ts > 0:
            container.seek(int(from_ts * av.time_base), backward=True)
        for frame in container.decode(stream):
            t = frame.time
            if t is None or t < from_ts - half_frame:
                continue
            if t >= to_ts - half_frame:
                break
            buffer = io.BytesIO()
            frame.to_image().save(buffer, "JPEG", quality=jpeg_quality)
            frames.append(buffer.getvalue())
    return frames


# --- One episode -> one take ----------------------------------------------------------


def import_episode(
    args: argparse.Namespace,
    info: dict,
    cameras: dict[str, str],
    robot: DualYam,
    ep: pd.Series,
) -> Path | None:
    index = int(ep["episode_index"])
    name = f"episode_{index:03d}"
    frames = episode_frames(args.source, info, ep)
    if frames.empty:
        print(f"  {name}: no rows in data parquet, skipped")
        return None
    state = stack_rows(frames, "observation.state")
    action = stack_rows(frames, "action")
    if args.swap_arms:
        state = np.concatenate([state[:, 7:14], state[:, :7]], axis=1)
        action = np.concatenate([action[:, 7:14], action[:, :7]], axis=1)
    timestamps = frames["timestamp"].to_numpy(dtype=np.float64)
    fps = float(info.get("fps") or 30.0)

    jpegs: dict[str, list[bytes]] = {}
    count = len(state)
    for key, entity in cameras.items():
        video_path = args.source / (info.get("video_path") or DEFAULT_VIDEO_PATH).format(
            video_key=key,
            chunk_index=int(ep[f"videos/{key}/chunk_index"]),
            file_index=int(ep[f"videos/{key}/file_index"]),
        )
        jpegs[entity] = decode_video_slice(
            video_path,
            float(ep[f"videos/{key}/from_timestamp"]),
            float(ep[f"videos/{key}/to_timestamp"]),
            fps,
            args.jpeg_quality,
        )
        if len(jpegs[entity]) != len(state):
            print(f"  {name}: {entity} has {len(jpegs[entity])} video frames for {len(state)} rows")
            count = min(count, len(jpegs[entity]))

    task = (args.task_instruction or episode_task(ep)).strip()
    path = takes.episode_path(args.recordings_dir, args.dataset, name)
    rec = takes.begin_take(path, episode=path.stem, dataset=args.dataset, task=task, proxy_uri=None)
    robot.log_static(rec)
    for arm, names in (("left_arm", STATE_DIM_NAMES[:7]), ("right_arm", STATE_DIM_NAMES[7:])):
        rec.log(f"{arm}/position", rr.SeriesLines(names=list(names)), static=True)
        rec.log(f"{arm}/goal", rr.SeriesLines(names=[f"{n} goal" for n in names]), static=True)

    # LeRobot timestamps are episode-relative; re-base onto the wall clock at import
    # time so the ``time`` timeline looks like every live-recorded take's.
    base = time.time() - float(timestamps[0])
    for i in range(count):
        rec.set_time("time", timestamp=base + float(timestamps[i]))
        rec.log("left_arm/position", rr.Scalars(state[i, :7]))
        rec.log("right_arm/position", rr.Scalars(state[i, 7:14]))
        rec.log("left_arm/goal", rr.Scalars(action[i, :7]))
        rec.log("right_arm/goal", rr.Scalars(action[i, 7:14]))
        robot.log_state(rec, state[i])
        for entity, blobs in jpegs.items():
            rec.log(entity, rr.EncodedImage(contents=blobs[i], media_type="image/jpeg"))

    takes.finish_take(rec, dataset=args.dataset, task=task, tag=args.tag, proxy_uri=None)
    takes.optimize_rrd(path)
    print(f"  {name}: {count} frames -> {path}")
    return path


# --- CLI ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="LeRobot v3 dataset root (the dir holding meta/, data/, videos/)")
    parser.add_argument("--dataset", default=None, help="catalog dataset name (default: source dir name)")
    parser.add_argument("--episodes", default=None, help="comma-separated LeRobot episode indices (default: all)")
    parser.add_argument("--tag", default="", help='tag stamped on every imported episode (e.g. "Good episode"; default: untagged)')
    parser.add_argument("--task-instruction", default=None, help="override every episode's task string")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="JPEG re-encode quality (75 = live collection hook default)")
    parser.add_argument("--camera-map", default="", help='rename video key stems to camera entities, e.g. "wrist_1=right,wrist_2=left"')
    parser.add_argument("--swap-arms", action="store_true", help="state/action dims 0-6 are the RIGHT arm (can0-first datasets); swap halves so left_arm/* entities stay physically left")
    parser.add_argument("--recordings-dir", type=Path, default=takes.DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--no-register", action="store_true", help="skip catalog registration (startup rescan picks the files up later)")
    parser.add_argument("--catalog-port", type=int, default=yc.DEFAULT_CATALOG_PORT)
    args = parser.parse_args()

    info = load_info(args.source)
    check_state_features(info)
    camera_map = dict(part.split("=", 1) for part in args.camera_map.split(",") if part.strip())
    cameras = video_features(info, camera_map)
    args.dataset = takes.sanitize_name(args.dataset or args.source.name)

    episodes = load_episodes_meta(args.source)
    if args.episodes is not None:
        wanted = {int(part) for part in args.episodes.split(",") if part.strip()}
        episodes = episodes[episodes["episode_index"].isin(wanted)]
        missing = wanted - set(int(i) for i in episodes["episode_index"])
        if missing:
            raise SystemExit(f"episode indices not in the dataset: {sorted(missing)}")
    if episodes.empty:
        raise SystemExit("no episodes to import")

    print(f"importing {len(episodes)} episode(s) from {args.source} -> "
          f"{args.recordings_dir / args.dataset} (cameras: {', '.join(cameras.values())})")
    robot = DualYam.create()
    imported: list[Path] = []
    for _, ep in episodes.sort_values("episode_index").iterrows():
        path = import_episode(args, info, cameras, robot, ep)
        if path is not None:
            imported.append(path)
    if not imported:
        raise SystemExit("nothing imported")

    if args.no_register:
        print(f"done: {len(imported)} episode(s); not registered (server startup rescan will pick them up)")
        return
    catalog_uri = yc.catalog_uri(args.catalog_port)
    try:
        for path in imported:
            takes.register_rrd(catalog_uri, args.dataset, path)
        bp.register_dataset_blueprint(
            catalog_uri, args.recordings_dir, args.dataset,
            visual_paths=[arm.visual_geometries_path for arm in robot.arms],
        )
    except Exception as error:
        print(f"warning: catalog registration failed ({error}); "
              f"the files are on disk and yam_rerun/server.py will register them on startup")
        return
    print(f"done: {len(imported)} episode(s) registered to dataset '{args.dataset}'")


if __name__ == "__main__":
    main()
