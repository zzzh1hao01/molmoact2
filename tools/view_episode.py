"""Visualise one YAM episode in the Rerun viewer: animated dual-URDF + cameras + plots.

Two episode sources, one visualisation path (both re-log through ``yam_rerun.urdf_yam``
FK and the standard blueprint, so the result looks exactly like a live-recorded take):

* **Raw episode dir** (the workstation's ``data/<ep>/`` layout: ``data.npz`` with
  ``state``/``action`` (T,14) + ``t`` (T,) wall-clock seconds, one ``<camera>.mp4`` per
  camera, ``meta.json``): decoded with PyAV, re-encoded JPEG.
* **Catalog episode** (``--dataset``/``--episode``): frames come from the query API --
  ``filter_segments(...).filter_contents([joints + cams]).reader(index="time",
  fill_latest_at=True)`` -- the same round-trip ``episode_metrics.py`` scores with. If no
  catalog is reachable, an ephemeral in-process ``rr.server.Server`` is spun up over
  ``--recordings-dir``, so this works without ``yam_rerun/server.py`` running.

Outputs (combine freely): ``--spawn`` native viewer (default when nothing else is given),
``--serve`` browser viewer (the https://rerun.io/viewer flow, kept alive until Ctrl-C),
``--save out.rrd`` (headless; drop it under ``recordings/<dataset>/`` to make it a
catalog episode).

Optional 2D object detection: ``--detect "cup, bottle, towel"`` runs open-vocabulary
YOLO-World over the camera frames and logs ``rr.Boxes2D`` overlays under
``camera/<name>/detections`` -- visible in the camera panes and queryable like any other
entity once saved. Needs ``ultralytics`` + the ultralytics CLIP fork in the venv (see
yam_rerun/README.md); the episode itself has no depth/calibration, so boxes live in
image space only.

Usage::

    # raw episode dir -> native viewer
    python tools/view_episode.py data/ep01

    # raw episode dir -> browser viewer + .rrd on disk
    python tools/view_episode.py data/ep01 --serve --save /tmp/ep01.rrd

    # episode already in the catalog (or just on disk under recordings/)
    python tools/view_episode.py --dataset import_test --episode episode_000
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import _yam_catalog as yc

sys.path.insert(0, str(yc.REPO_ROOT))  # tools/ scripts run with sys.path[0] == tools/

from yam_rerun import blueprint as bp  # noqa: E402
from yam_rerun.urdf_yam import STATE_DIM, STATE_DIM_NAMES, DualYam  # noqa: E402

import rerun as rr  # noqa: E402

# Raw-dir camera file stem -> catalog camera entity. Same 3-cam contract as
# examples/yam/host_server_yam.py ([top, left, right]); wrist_1/wrist_2 are the
# left/right wrist cameras in the workstation's collection layout.
RAW_CAMERA_ENTITIES = {"top": "camera/top", "wrist_1": "camera/left", "wrist_2": "camera/right"}
EPHEMERAL_CATALOG_PORT = 51990


@dataclass
class EpisodeData:
    """Everything the visualiser needs, whichever source it came from."""

    name: str
    task: str
    times: np.ndarray  # (T,) float64 wall-clock seconds
    state: np.ndarray  # (T, 14) float32 -- measured, drives FK
    action: np.ndarray | None  # (T, 14) float32 -- commanded goals
    jpegs: dict[str, list[bytes]]  # camera entity path -> JPEG bytes per frame


# --- Source 1: raw episode dir --------------------------------------------------------


def decode_all_frames(path: Path, jpeg_quality: int) -> list[bytes]:
    import av

    frames: list[bytes] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            buffer = io.BytesIO()
            frame.to_image().save(buffer, "JPEG", quality=jpeg_quality)
            frames.append(buffer.getvalue())
    return frames


def load_raw_episode(source: Path, jpeg_quality: int) -> EpisodeData:
    npz = np.load(source / "data.npz")
    state = np.asarray(npz["state"], dtype=np.float32)
    action = np.asarray(npz["action"], dtype=np.float32) if "action" in npz.files else None
    times = np.asarray(npz["t"], dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise SystemExit(f"{source}/data.npz state has shape {state.shape}, expected (T, {STATE_DIM})")

    meta = json.loads((source / "meta.json").read_text()) if (source / "meta.json").exists() else {}
    cameras = meta.get("cameras") or sorted(p.stem for p in source.glob("*.mp4"))
    jpegs: dict[str, list[bytes]] = {}
    for camera in cameras:
        entity = RAW_CAMERA_ENTITIES.get(camera, f"camera/{camera}")
        jpegs[entity] = decode_all_frames(source / f"{camera}.mp4", jpeg_quality)
        if len(jpegs[entity]) != len(state):
            print(f"note: {camera}.mp4 has {len(jpegs[entity])} frames for {len(state)} state rows")
    return EpisodeData(
        name=source.name,
        task=str(meta.get("task") or meta.get("language_instruction") or ""),
        times=times,
        state=state,
        action=action,
        jpegs=jpegs,
    )


# --- Source 2: catalog episode (query API) --------------------------------------------


def connect_catalog(port: int, recordings_dir: Path, dataset_name: str):
    """The long-lived catalog if it answers; otherwise an ephemeral in-process one.

    Returns (client, server) -- the server (or None) must be kept alive while querying.
    """
    try:
        client = yc.connect(port)
        client.all_datasets()  # force a round-trip; the constructor doesn't connect
        return client, None
    except Exception:
        pass
    rrds = sorted(str(p) for p in (recordings_dir / yc.sanitize_name(dataset_name)).glob("*.rrd"))
    if not rrds:
        raise SystemExit(
            f"no catalog on port {port} and no .rrd files under "
            f"{recordings_dir / yc.sanitize_name(dataset_name)} for an ephemeral one"
        )
    print(f"no catalog on port {port}; serving {len(rrds)} .rrd file(s) in-process")
    server = rr.server.Server(port=EPHEMERAL_CATALOG_PORT, datasets={yc.sanitize_name(dataset_name): rrds})
    return rr.catalog.CatalogClient(yc.catalog_uri(EPHEMERAL_CATALOG_PORT)), server


def load_catalog_episode(args: argparse.Namespace) -> EpisodeData:
    client, server = connect_catalog(args.catalog_port, args.recordings_dir, args.dataset)
    dataset = client.get_dataset(name=yc.sanitize_name(args.dataset))
    episodes = yc.list_episodes(dataset, episode=args.episode)
    if not episodes:
        known = ", ".join(ep.name for ep in yc.list_episodes(dataset)) or "(none)"
        raise SystemExit(f"episode {args.episode!r} not in dataset {args.dataset!r}; episodes: {known}")
    episode = episodes[0]

    entities = list(yc.JOINT_ENTITIES) + list(yc.CAMERA_ENTITIES)
    df = yc.episode_frames(dataset, episode.segment_id, entities)
    if df.empty:
        raise SystemExit(f"episode {episode.name} has no complete frames")
    times = pd.to_datetime(df[yc.TIME_INDEX]).astype("int64").to_numpy() / 1e9
    jpegs = {
        camera: [yc.blob_bytes(cell) for cell in df[yc.blob_column(df, camera)].to_numpy()]
        for camera in yc.CAMERA_ENTITIES
    }
    del server  # queries done; the ephemeral catalog (if any) can go
    return EpisodeData(
        name=episode.name,
        task=episode.task,
        times=times,
        state=yc.stack14(df, *yc.POSITION_ENTITIES),
        action=yc.stack14(df, *yc.GOAL_ENTITIES),
        jpegs=jpegs,
    )


# --- Optional 2D detection overlays ---------------------------------------------------

# Stable per-class box colors (cycled when there are more prompts than entries).
DETECT_PALETTE = [
    (230, 90, 60), (60, 160, 230), (90, 200, 90), (230, 190, 60),
    (190, 90, 220), (80, 210, 200), (240, 130, 180), (150, 150, 150),
]


def run_detections(
    data: EpisodeData, prompts: list[str], conf: float, every: int, device: str | None
) -> dict[str, dict[int, tuple[np.ndarray, list[str], list[tuple[int, int, int]]]]]:
    """YOLO-World over every ``every``-th frame of each camera.

    Returns ``{camera entity: {frame index: (boxes XYXY (N,4), labels, colors)}}`` --
    an entry with N == 0 still appears, so stale boxes get cleared at that timestamp.
    """
    try:
        from ultralytics import YOLOWorld
    except ImportError as error:
        raise SystemExit(
            "--detect needs ultralytics (+ the ultralytics CLIP fork) in this venv:\n"
            "  uv pip install -p .venv-rerun ultralytics 'git+https://github.com/ultralytics/CLIP.git'\n"
            f"(import failed: {error})"
        ) from None
    from PIL import Image

    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    # Keep the auto-downloaded weights in a cache dir: ultralytics drops the .pt
    # wherever the cwd happens to be, so first use downloads there and we move it.
    weights = Path.home() / ".cache" / "ultralytics" / "yolov8s-worldv2.pt"
    if weights.exists():
        model = YOLOWorld(str(weights))
    else:
        model = YOLOWorld("yolov8s-worldv2.pt")
        downloaded = Path("yolov8s-worldv2.pt")
        if downloaded.exists():
            weights.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), weights)
    model.set_classes(prompts)

    results: dict[str, dict[int, tuple[np.ndarray, list[str], list[tuple[int, int, int]]]]] = {}
    total = sum(len(range(0, len(blobs), every)) for blobs in data.jpegs.values())
    done = 0
    for entity, blobs in data.jpegs.items():
        per_frame: dict[int, tuple[np.ndarray, list[str], list[tuple[int, int, int]]]] = {}
        for index in range(0, len(blobs), every):
            rgb = np.asarray(Image.open(io.BytesIO(blobs[index])).convert("RGB"))
            result = model.predict(rgb[..., ::-1], conf=conf, device=device, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4), np.float32)
            classes = [int(b.cls) for b in result.boxes]
            labels = [f"{result.names[c]} {float(b.conf):.2f}" for c, b in zip(classes, result.boxes)]
            colors = [DETECT_PALETTE[c % len(DETECT_PALETTE)] for c in classes]
            per_frame[index] = (boxes, labels, colors)
            done += 1
            if done % 200 == 0:
                print(f"  detection: {done}/{total} frames")
        results[entity] = per_frame
    print(f"  detection: {done}/{total} frames done ({device}, conf>={conf:g}, every {every})")
    return results


# --- Re-log + view --------------------------------------------------------------------


def log_episode(
    rec: rr.RecordingStream,
    data: EpisodeData,
    robot: DualYam,
    detections: dict[str, dict[int, tuple[np.ndarray, list[str], list[tuple[int, int, int]]]]] | None = None,
) -> None:
    robot.log_static(rec)
    if data.task:
        rec.log("/task", rr.TextDocument(data.task), static=True)
    for arm, names in (("left_arm", STATE_DIM_NAMES[:7]), ("right_arm", STATE_DIM_NAMES[7:])):
        rec.log(f"{arm}/position", rr.SeriesLines(names=list(names)), static=True)
        if data.action is not None:
            rec.log(f"{arm}/goal", rr.SeriesLines(names=[f"{n} goal" for n in names]), static=True)

    for i in range(len(data.state)):
        rec.set_time("time", timestamp=float(data.times[i]))
        rec.log("left_arm/position", rr.Scalars(data.state[i, :7]))
        rec.log("right_arm/position", rr.Scalars(data.state[i, 7:14]))
        if data.action is not None:
            rec.log("left_arm/goal", rr.Scalars(data.action[i, :7]))
            rec.log("right_arm/goal", rr.Scalars(data.action[i, 7:14]))
        robot.log_state(rec, data.state[i])
        for entity, blobs in data.jpegs.items():
            if i < len(blobs):
                rec.log(entity, rr.EncodedImage(contents=blobs[i], media_type="image/jpeg"))
            hit = (detections or {}).get(entity, {}).get(i)
            if hit is not None:
                boxes, labels, colors = hit
                rec.log(
                    f"{entity}/detections",
                    rr.Boxes2D(array=boxes, array_format=rr.Box2DFormat.XYXY, labels=labels, colors=colors),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path, help="raw episode dir (holding data.npz + <camera>.mp4)")
    parser.add_argument("--dataset", default=None, help="catalog dataset (instead of a raw dir)")
    parser.add_argument("--episode", default=None, help="episode name in --dataset, e.g. episode_01")
    parser.add_argument("--spawn", action="store_true", help="open the native viewer (default output)")
    parser.add_argument("--serve", action="store_true", help="serve the browser viewer until Ctrl-C")
    parser.add_argument("--save", type=Path, default=None, metavar="OUT.rrd", help="also write a .rrd")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--detect", default=None, metavar='"cup, bottle, ..."',
                        help="open-vocabulary prompts; logs Boxes2D overlays on the camera views")
    parser.add_argument("--detect-conf", type=float, default=0.2, help="detection confidence threshold")
    parser.add_argument("--detect-every", type=int, default=1, help="run detection on every Nth frame")
    parser.add_argument("--detect-device", default=None, help="torch device (default: mps/cuda if available, else cpu)")
    parser.add_argument("--window-seconds", type=float, default=10.0, help="sliding plot window in the blueprint")
    parser.add_argument("--recordings-dir", type=Path, default=yc.RECORDINGS_DIR)
    parser.add_argument("--catalog-port", type=int, default=yc.DEFAULT_CATALOG_PORT)
    args = parser.parse_args()

    if (args.source is None) == (args.dataset is None):
        parser.error("pass either a raw episode dir OR --dataset/--episode")
    if args.dataset is not None and args.episode is None:
        parser.error("--dataset needs --episode")
    if not (args.spawn or args.serve or args.save):
        args.spawn = True

    if args.source is not None:
        if not (args.source / "data.npz").exists():
            parser.error(f"{args.source} has no data.npz -- not a raw episode dir")
        data = load_raw_episode(args.source, args.jpeg_quality)
    else:
        data = load_catalog_episode(args)
    duration = float(data.times[-1] - data.times[0]) if len(data.times) > 1 else 0.0
    print(f"{data.name}: {len(data.state)} frames, {duration:.1f}s, "
          f"cameras: {', '.join(data.jpegs) or '(none)'}"
          + (f", task: {data.task!r}" if data.task else ""))

    detections = None
    if args.detect:
        prompts = [p.strip() for p in args.detect.split(",") if p.strip()]
        if not prompts:
            parser.error("--detect got no prompts")
        print(f"detecting: {', '.join(prompts)}")
        detections = run_detections(data, prompts, args.detect_conf, max(1, args.detect_every), args.detect_device)

    robot = DualYam.create()
    blueprint = bp.create_blueprint(
        camera_paths=tuple(data.jpegs),
        visual_paths=[arm.visual_geometries_path for arm in robot.arms],
        window_seconds=args.window_seconds,
    )

    def new_recording() -> rr.RecordingStream:
        rec = rr.RecordingStream(
            "yam", recording_id=f"view-{yc.sanitize_name(data.name)}-{time.time_ns()}"
        )
        rec.send_recording_name(data.name)
        return rec

    # save/spawn share one recording (sinks tee); serve_grpc installs its own sink and
    # must run BEFORE logging, so it gets a recording of its own.
    if args.save or args.spawn:
        rec = new_recording()
        if args.spawn and args.save:
            rec.spawn(connect=False)
            rec.set_sinks(
                rr.GrpcSink(url="rerun+http://127.0.0.1:9876/proxy"),
                rr.FileSink(str(args.save)),
            )
        elif args.save:
            rec.save(str(args.save))
        else:
            rec.spawn()
        rec.send_blueprint(blueprint)
        log_episode(rec, data, robot, detections)
        rec.flush()
        if args.save:
            print(f"wrote {args.save}")
        if args.spawn:
            print("viewer spawned")

    if args.serve:
        rec = new_recording()
        uri = rr.serve_grpc(recording=rec)
        rec.send_blueprint(blueprint)
        log_episode(rec, data, robot, detections)
        rec.flush()
        rr.serve_web_viewer(connect_to=uri)
        print("browser viewer up; Ctrl-C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
