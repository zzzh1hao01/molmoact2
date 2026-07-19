"""Offline policy evaluation through the catalog: replay recorded observations to a
policy server and write predictions back as an ``eval`` layer.

The loop this enables (no robot needed): for every episode of a catalog dataset,
query the aligned observation frames (query API), send them to a MolmoAct2 policy
server (the ``/act`` wire protocol of ``examples/yam/host_server_yam.py`` /
``experiments/modal_serve.py``), and log the predicted action chunks back INTO the
catalog on the episode's own timeline:

- ``left_arm/pred`` / ``right_arm/pred``  7-D ``Scalars`` at each future frame time,
  so the viewer overlays predicted-vs-commanded trajectories on the existing plots;
- ``property:eval:*`` divergence scores (RMSE against the recorded ``goal`` stream,
  per arm, gripper MAE, request latency), so ``query_dataset.py --full`` ranks
  episodes by how far the policy strays from the demonstration.

Everything is written as a separate ``eval`` catalog layer (same mechanism as
``metrics``): the take .rrd files are never touched, re-running replaces the layer,
and ``yam_rerun/server.py`` re-registers ``recordings/<ds>/eval/*.rrd`` on startup.

Arm-order note: takes store left-arm-first (dims 0-6 = left), but checkpoints
fine-tuned on can0-first datasets (e.g. Shivakumr/yams via ``--swap-arms`` import)
expect right-arm-first state and return right-arm-first actions. ``--arm-order
right-first`` (the default) swaps halves both ways at the wire; use ``left-first``
for checkpoints trained on left-first data.

Usage::

    python tools/eval_policy.py --dataset yams \
        --url https://<app>.modal.run --episodes episode_000,episode_003
    python tools/eval_policy.py --dataset yams --url http://localhost:8202  # all episodes
    python tools/query_dataset.py --dataset yams --full                     # see eval:* columns
"""

from __future__ import annotations

import argparse
import io
import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd

import _yam_catalog as yc

import sys

sys.path.insert(0, str(yc.REPO_ROOT))  # tools/ scripts run with sys.path[0] == tools/

from yam_rerun.urdf_yam import STATE_DIM_NAMES, DualYam  # noqa: E402

import rerun as rr  # noqa: E402  (env prepared by _yam_catalog import)

ARM_DIMS = np.r_[0:6, 7:13]  # revolute joints; 6/13 are grippers
GRIPPER_DIMS = np.asarray(yc.GRIPPER_DIMS)
GHOST_RGBA = (255, 140, 0, 110)  # translucent orange: the policy's "ghost arms"


def layer_suffix(layer: str) -> str:
    """'eval' -> '' (plain pred/ghost_); 'eval1k' -> '1k' (pred1k/ghost1k_)."""
    return layer[4:] if layer.startswith("eval") else f"_{layer}"


def swap_halves(vec: np.ndarray) -> np.ndarray:
    """[left(7), right(7)] <-> [right(7), left(7)] on the last axis."""
    return np.concatenate([vec[..., 7:14], vec[..., :7]], axis=-1)


def decode_jpeg(blob: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.uint8)


def request_actions(url: str, cams: dict[str, np.ndarray], instruction: str, state: np.ndarray) -> np.ndarray:
    import json_numpy
    import requests

    payload = {
        "top_cam": cams["camera/top"],
        "left_cam": cams["camera/left"],
        "right_cam": cams["camera/right"],
        "instruction": instruction,
        "state": state.astype(np.float32),
    }
    response = requests.post(
        f"{url.rstrip('/')}/act",
        data=json_numpy.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=300,
    )
    response.raise_for_status()
    body = json_numpy.loads(response.text)
    return np.asarray(body["actions"] if isinstance(body, dict) else body, dtype=np.float32)


def eval_episode(args: argparse.Namespace, dataset, episode, ghost: DualYam | None) -> dict[str, float] | None:
    df = yc.episode_frames(dataset, episode.segment_id, list(yc.JOINT_ENTITIES) + list(yc.CAMERA_ENTITIES))
    if df.empty:
        print(f"  {episode.name}: no complete frames, skipped")
        return None
    times = pd.to_datetime(df[yc.TIME_INDEX]).astype("int64").to_numpy() / 1e9
    state = yc.stack14(df, *yc.POSITION_ENTITIES)   # left-first, as logged
    goal = yc.stack14(df, *yc.GOAL_ENTITIES)
    blob_cols = {camera: yc.blob_column(df, camera) for camera in yc.CAMERA_ENTITIES}
    instruction = (args.instruction or episode.task or "").strip()
    right_first = args.arm_order == "right-first"

    count = len(state)
    preds = np.full((count, 14), np.nan, dtype=np.float32)
    latencies: list[float] = []
    i = 0
    while i < count:
        cams = {cam: decode_jpeg(yc.blob_bytes(df[col].iloc[i])) for cam, col in blob_cols.items()}
        wire_state = swap_halves(state[i]) if right_first else state[i]
        started = time_mod.perf_counter()
        actions = request_actions(args.url, cams, instruction, wire_state)
        latencies.append((time_mod.perf_counter() - started) * 1e3)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise SystemExit(f"policy returned shape {actions.shape}, expected (N, 14)")
        if right_first:
            actions = swap_halves(actions)
        stride = min(args.every or len(actions), len(actions))
        take = min(stride, count - i)
        preds[i : i + take] = actions[:take]
        i += take

    valid = ~np.isnan(preds[:, 0])
    error = preds[valid] - goal[valid]
    scores = {
        "pred_rmse": float(np.sqrt(np.mean(error[:, ARM_DIMS] ** 2))),
        "pred_max": float(np.max(np.abs(error[:, ARM_DIMS]))),
        "left_rmse": float(np.sqrt(np.mean(error[:, 0:6] ** 2))),
        "right_rmse": float(np.sqrt(np.mean(error[:, 7:13] ** 2))),
        "gripper_mae": float(np.mean(np.abs(error[:, GRIPPER_DIMS]))),
        "latency_ms": float(np.mean(latencies)),
        "requests": len(latencies),
        "checkpoint": args.checkpoint_label,
    }

    pred = f"pred{layer_suffix(args.layer)}"
    layer_path = args.recordings_dir / dataset.name / args.layer / f"{episode.name}.rrd"
    layer_path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(yc.APP_ID, recording_id=episode.segment_id)
    rec.set_sinks(rr.FileSink(str(layer_path)))
    for arm, names in (("left_arm", STATE_DIM_NAMES[:7]), ("right_arm", STATE_DIM_NAMES[7:])):
        rec.log(f"{arm}/{pred}", rr.SeriesLines(names=[f"{n} {pred}" for n in names]), static=True)
    if ghost is not None:
        # Translucent "ghost arms" driven by the predicted joints, overlaid on the
        # demo robot in the same 3D view (blueprint includes ghost*/** visual meshes).
        ghost.log_static(rec)
        ghost.tint(rec, args.ghost_rgba)
    for k in range(count):
        if not valid[k]:
            continue
        rec.set_time(yc.TIME_INDEX, timestamp=float(times[k]))
        rec.log(f"left_arm/{pred}", rr.Scalars(preds[k, :7]))
        rec.log(f"right_arm/{pred}", rr.Scalars(preds[k, 7:14]))
        if ghost is not None:
            ghost.log_state(rec, preds[k])
    rec.send_property(args.layer, rr.AnyValues(**scores))
    rec.flush()
    rec.disconnect()
    print(f"  {episode.name}: rmse={scores['pred_rmse']:.4f} max={scores['pred_max']:.4f} "
          f"latency={scores['latency_ms']:.0f}ms ({len(latencies)} requests) -> {layer_path}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="catalog dataset to evaluate")
    parser.add_argument("--url", required=True, help="policy server base URL (POST <url>/act)")
    parser.add_argument("--episodes", default=None, help="comma-separated episode names (default: all)")
    parser.add_argument("--tag", default=None, help='only episodes with this tag (e.g. "Good episode")')
    parser.add_argument("--instruction", default=None, help="override the episode task string")
    parser.add_argument("--every", type=int, default=None, help="frames per policy request (default: the returned chunk length)")
    parser.add_argument("--arm-order", choices=("right-first", "left-first"), default="right-first",
                        help="the CHECKPOINT's state/action convention (takes are always left-first)")
    parser.add_argument("--checkpoint-label", default="", help="free-form label stamped as property:eval:checkpoint")
    parser.add_argument("--no-ghost", action="store_true", help="skip the 3D ghost-arm overlay (predictions still plot in 2D)")
    parser.add_argument("--layer", default="eval", help="catalog layer name; use e.g. 'eval1k' to keep several checkpoints side by side (subdir, property:<layer>:*, <arm>/pred<suffix>, ghost<suffix>_*)")
    parser.add_argument("--ghost-color", default="255,140,0,110", help='ghost mesh RGBA, e.g. "0,190,255,110" for a second checkpoint')
    parser.add_argument("--recordings-dir", type=Path, default=None)
    parser.add_argument("--catalog-port", type=int, default=yc.DEFAULT_CATALOG_PORT)
    args = parser.parse_args()

    if args.recordings_dir is None:
        args.recordings_dir = yc.REPO_ROOT / "recordings"
    client = yc.connect(args.catalog_port)
    dataset = client.get_dataset(name=yc.sanitize_name(args.dataset))
    episodes = yc.list_episodes(dataset, tag=args.tag)
    if args.episodes:
        wanted = {part.strip() for part in args.episodes.split(",") if part.strip()}
        episodes = [ep for ep in episodes if ep.name in wanted]
        missing = wanted - {ep.name for ep in episodes}
        if missing:
            raise SystemExit(f"episodes not in dataset '{args.dataset}': {sorted(missing)}")
    if not episodes:
        raise SystemExit("no episodes to evaluate")

    print(f"evaluating {len(episodes)} episode(s) of '{dataset.name}' against {args.url} "
          f"(checkpoint convention: {args.arm_order})")
    args.ghost_rgba = tuple(int(part) for part in args.ghost_color.split(","))
    # NOTE: underscore, not "ghost/": UrdfTree escapes a "/" in entity_path_prefix
    # into a literal path component, disconnecting the meshes from the blueprint.
    ghost = None if args.no_ghost else DualYam.create(prefix=f"ghost{layer_suffix(args.layer)}_", label="")
    rows = []
    for episode in episodes:
        scores = eval_episode(args, dataset, episode, ghost)
        if scores is not None:
            rows.append({"episode": episode.name, **{k: v for k, v in scores.items() if k != "checkpoint"}})

    layer_dir = args.recordings_dir / dataset.name / args.layer
    paths = sorted(layer_dir.glob("*.rrd"))
    yc.register_layer(client, dataset.name, args.layer, paths)
    print(f"registered {len(paths)} {args.layer} layer file(s) to dataset '{dataset.name}'")
    if rows:
        table = pd.DataFrame(rows).sort_values("pred_rmse", ascending=False)
        yc.print_table(table.round(4))


if __name__ == "__main__":
    main()
