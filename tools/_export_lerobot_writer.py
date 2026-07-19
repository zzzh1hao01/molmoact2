"""LeRobot-side half of the catalog export (spawned by tools/export_lerobot.py).

Runs inside the lerobot environment (the only interpreter with ``lerobot`` installed --
on the robot workstation that is conda ``ai2_yam`` with ``YAM/lerobot`` 0.4.3; this
script deliberately imports NO rerun). Reads a staged directory
(``state.npy``/``action.npy`` + per-camera JPEG dirs + ``manifest.json``) and writes a
LeRobot v3 dataset with the exact schema of ``YAM/molmoact_to_lerobot_v30.py``:

- ``observation.state`` / ``action``: float32 (14,), names = STATE_DIM_NAMES
  (left_joint1..6, left_gripper, right_joint1..6, right_gripper), radians unconverted
- ``observation.images.{top,left,right}``: video features, names [height,width,channels]
- robot_type ``molmoact_dual_arm``, ``use_videos=True``, then ``finalize()`` and the
  same quantile-column sanitization for the online visualizer.

Standalone usage (e.g. after ``export --stage-only``)::

    python tools/_export_lerobot_writer.py --stage <stage-dir> --root datasets
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


def build(stage: Path, root: Path) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = json.loads((stage / "manifest.json").read_text())
    repo_id: str = manifest["repo_id"]
    state_names: list[str] = manifest["state_names"]
    cameras: list[str] = manifest["cameras"]
    task_override = manifest.get("task_instruction")

    # Same feature schema as molmoact_to_lerobot_v30.create_lerobot_dataset_v30.
    features: dict = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
    }
    for camera in cameras:
        first = next((stage / manifest["episodes"][0]["dir"] / camera).glob("*.jpg"))
        with Image.open(first) as img:
            width, height = img.size
            channels = len(img.getbands())
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        }

    output = Path(root) / repo_id
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory '{output}' is not empty")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=manifest["fps"],
        features=features,
        root=output,
        robot_type=manifest.get("robot_type", "molmoact_dual_arm"),
        use_videos=True,
        batch_encoding_size=1,
        vcodec=manifest.get("vcodec", "h264"),
    )

    episodes = manifest["episodes"]
    for number, episode in enumerate(episodes, start=1):
        episode_dir = stage / episode["dir"]
        state = np.load(episode_dir / "state.npy")
        action = np.load(episode_dir / "action.npy")
        task = (task_override or episode.get("task") or "perform the task").strip() or "perform the task"
        for index in range(len(state)):
            frame = {"observation.state": state[index], "action": action[index], "task": task}
            for camera in cameras:
                with Image.open(episode_dir / camera / f"{index:06d}.jpg") as img:
                    frame[f"observation.images.{camera}"] = img.convert("RGB")
            dataset.add_frame(frame)
        start = time.perf_counter()
        dataset.save_episode()
        print(f"[{number}/{len(episodes)}] {episode['name']}: wrote {len(state)} frames "
              f"(encoded in {time.perf_counter() - start:.0f}s)", flush=True)
        gc.collect()

    print("finalizing v3 dataset...", flush=True)
    dataset.finalize()
    sanitize_online_viz_meta(output)
    return output


def sanitize_online_viz_meta(output: Path) -> None:
    """Drop quantile-only columns from episode metadata for broader viewer compatibility
    (same as molmoact_to_lerobot_v30.sanitize_episode_metadata_for_online_viz)."""
    try:
        import pandas as pd
    except Exception:
        print("warning: pandas unavailable; skipping metadata sanitization")
        return
    episodes_root = output / "meta" / "episodes"
    dropped = 0
    for parquet in sorted(episodes_root.glob("chunk-*/file-*.parquet")) if episodes_root.exists() else []:
        df = pd.read_parquet(parquet)
        drop_cols = [c for c in df.columns if c.endswith(("/q01", "/q10", "/q50", "/q90", "/q99"))]
        if drop_cols:
            df.drop(columns=drop_cols).to_parquet(parquet, index=False)
            dropped += len(drop_cols)
    if dropped:
        print(f"sanitized episode metadata for the online visualizer ({dropped} columns dropped)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True, help="staging dir written by export_lerobot.py")
    parser.add_argument("--root", type=Path, required=True, help="output root; dataset lands in <root>/<repo-id>")
    args = parser.parse_args()
    output = build(args.stage, args.root)
    print(f"done: {output}")


if __name__ == "__main__":
    main()
