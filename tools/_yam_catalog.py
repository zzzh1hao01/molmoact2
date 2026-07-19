"""Shared helpers for the YAM catalog CLIs (query / metrics / export).

Everything in ``tools/`` talks to the local Rerun catalog served by
``yam_rerun/server.py`` (default port 51234) and codes against the Phase-1
logging contract:

- cameras   ``camera/top``, ``camera/left``, ``camera/right``  (JPEG ``rr.EncodedImage``)
- state     ``left_arm/position``, ``right_arm/position``      (7-D ``rr.Scalars`` each)
- action    ``left_arm/goal``, ``right_arm/goal``              (7-D ``rr.Scalars`` each)
- timeline  ``time``
- episode properties ``dataset`` / ``task`` / ``tag`` stamped via
  ``send_property("episode", ...)`` -> ``property:episode:*`` columns

The 14-D ordering is the ground truth of ``YAM/molmoact_to_lerobot_v30.py``:
left arm first (6 joints + gripper), then right arm. YAM is radians end-to-end,
so no unit conversion happens anywhere in these tools.

Layer conventions (must stay in sync with ``yam_rerun/takes.py``):

- base recordings   ``recordings/<dataset>/<episode>.rrd``
- metadata edits    ``recordings/<dataset>/edits/<episode>.rrd``    (layer "edits")
- metric scores     ``recordings/<dataset>/metrics/<episode>.rrd``  (layer "metrics")

Edits/metrics files reuse the episode's recording id (== catalog segment id) so
registering them as a named layer overlays new ``property:*`` columns on the
original recording without touching the data. Each file is rewritten wholesale
on update; property stamps must always send the FULL field set (a narrower
later stamp loses to a wider earlier one).
"""

from __future__ import annotations

import dataclasses
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The catalog client refuses localhost tokens unless we opt out of the host check.
# Must be set before rerun is imported.
os.environ.setdefault("RERUN_INSECURE_SKIP_HOST_CHECK", "1")

import rerun as rr  # noqa: E402

APP_ID = "yam"
DEFAULT_CATALOG_PORT = 51234
REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = REPO_ROOT / "recordings"

SEGMENT_TAGS = ("Good episode", "Bad episode", "Needs review")

# --- Entity-path contract (Phase 1 logging) -------------------------------------------
POSITION_ENTITIES = ("left_arm/position", "right_arm/position")
GOAL_ENTITIES = ("left_arm/goal", "right_arm/goal")
CAMERA_ENTITIES = ("camera/top", "camera/left", "camera/right")
JOINT_ENTITIES = POSITION_ENTITIES + GOAL_ENTITIES

# LeRobot feature key per camera entity. Must match YAM/molmoact_to_lerobot_v30.py's
# CAMERA_FEATURE_KEYS (observation.images.{top,left,right}) == the training mixture
# tag `yam_dual_molmoact2` and the released MolmoAct2-BimanualYAM datasets.
CAMERA_KEYS = {
    "camera/top": "top",
    "camera/left": "left",
    "camera/right": "right",
}

# 14-D dim names, verbatim from YAM/molmoact_to_lerobot_v30.py (STATE_DIM_NAMES).
# state/action = concat([left_arm (7,), right_arm (7,)]).
STATE_DIM_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_gripper",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_gripper",
]
ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # the 12 revolute joints
GRIPPER_DIMS = [6, 13]

TIME_INDEX = "time"


# --- Catalog access -------------------------------------------------------------------


def catalog_uri(port: int = DEFAULT_CATALOG_PORT) -> str:
    return f"rerun+http://localhost:{port}"


def connect(port: int = DEFAULT_CATALOG_PORT) -> rr.catalog.CatalogClient:
    return rr.catalog.CatalogClient(catalog_uri(port))


def sanitize_name(name: str) -> str:
    """Filesystem- and catalog-safe name (same rule as yam_rerun/takes.py)."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return cleaned or time.strftime("%Y%m%d-%H%M%S")


def flatten(value: Any) -> Any:
    """Property columns are list-typed (one value per layer); show the first."""
    if isinstance(value, str) or value is None:
        return value
    try:
        return value[0] if len(value) else None
    except TypeError:
        return value


@dataclasses.dataclass
class Episode:
    segment_id: str
    name: str
    task: str
    tag: str
    dataset: str
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None


def list_episodes(dataset: rr.catalog.DatasetEntry, *, tag: str | None = None, episode: str | None = None) -> list[Episode]:
    """One Episode per catalog segment, newest metadata layer winning, sorted by name."""
    table = dataset.segment_table().to_pandas()
    episodes: list[Episode] = []
    for _, row in table.iterrows():
        row_tag = flatten(row.get("property:episode:tag")) or ""
        name = str(flatten(row.get("property:RecordingInfo:name")) or row["rerun_segment_id"])
        if tag is not None and row_tag != tag:
            continue
        if episode is not None and name != episode:
            continue
        episodes.append(
            Episode(
                segment_id=str(row["rerun_segment_id"]),
                name=name,
                task=str(flatten(row.get("property:episode:task")) or ""),
                tag=str(row_tag),
                dataset=str(flatten(row.get("property:episode:dataset")) or dataset.name),
                start=row.get("time:start"),
                end=row.get("time:end"),
            )
        )
    return sorted(episodes, key=lambda ep: ep.name)


# --- Reader-column helpers ------------------------------------------------------------


def scalar_column(df: pd.DataFrame, entity: str) -> str:
    return _column(df, entity, ":Scalars:scalars")


def blob_column(df: pd.DataFrame, entity: str) -> str:
    return _column(df, entity, ":EncodedImage:blob")


def _column(df: pd.DataFrame, entity: str, suffix: str) -> str:
    """Reader columns are named ``/entity/path:Archetype:field``."""
    matches = [name for name in df.columns if name.endswith(suffix) and entity in name]
    if not matches:
        raise SystemExit(f"column '{entity}{suffix}' missing from the query result (got: {list(df.columns)})")
    return matches[0]


def episode_frames(
    dataset: rr.catalog.DatasetEntry,
    segment_id: str,
    entities: list[str],
    *,
    index_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """One episode's aligned frame table: one row per tick, latest values filled forward.

    ``filter_contents`` restricts the reader to exactly the requested entity columns
    (single round-trip for all of them); ``fill_latest_at=True`` carries each stream's
    most recent value onto every row, so rows logged as separate events collapse into
    complete frames. Rows are deduped per timestamp (keep last == most filled) and
    leading rows where any stream has not produced a value yet are dropped.

    ``index_values`` (datetime64[ns] array) switches the reader to resampling mode:
    one row per requested timestamp (``using_index_values``), which is how the
    demo-vs-rollout comparison puts every episode on the same fixed-rate grid.
    """
    view = dataset.filter_segments(segment_id).filter_contents(list(entities))
    if index_values is not None:
        df = view.reader(index=TIME_INDEX, using_index_values=index_values, fill_latest_at=True).to_pandas()
    else:
        df = view.reader(index=TIME_INDEX, fill_latest_at=True).to_pandas()
    if df.empty:
        return df
    df = df.sort_values(TIME_INDEX).drop_duplicates(subset=TIME_INDEX, keep="last")
    required = [
        _column(df, entity, ":EncodedImage:blob" if entity in CAMERA_ENTITIES else ":Scalars:scalars")
        for entity in entities
    ]
    return df.dropna(subset=required).reset_index(drop=True)


def times_seconds(df: pd.DataFrame) -> np.ndarray:
    """The ``time`` index as float seconds from episode start."""
    ts = pd.to_datetime(df[TIME_INDEX]).astype("int64").to_numpy()
    return (ts - ts[0]) / 1e9


def stack14(df: pd.DataFrame, left_entity: str, right_entity: str) -> np.ndarray:
    """(T, 14) float32 in molmoact_to_lerobot_v30 order: left arm (7) then right arm (7).

    Scalars cells come back as lists / object arrays -> np.stack per arm, then concat.
    """
    left = np.stack([np.asarray(v, dtype=np.float32) for v in df[scalar_column(df, left_entity)]])
    right = np.stack([np.asarray(v, dtype=np.float32) for v in df[scalar_column(df, right_entity)]])
    if left.shape[1] != 7 or right.shape[1] != 7:
        raise SystemExit(f"expected 7-D per arm, got left={left.shape} right={right.shape}")
    return np.concatenate([left, right], axis=1)


def blob_bytes(cell: Any) -> bytes:
    """An EncodedImage blob cell -> raw JPEG bytes (cells are nested one list level)."""
    data = np.asarray(cell)
    if data.dtype == object:
        data = data[0]
    return bytes(np.asarray(data, dtype=np.uint8))


# --- Write-back layers (edits / metrics) ----------------------------------------------


def edits_path(dataset: str, episode_name: str) -> Path:
    return RECORDINGS_DIR / sanitize_name(dataset) / "edits" / f"{sanitize_name(episode_name)}.rrd"


def metrics_path(dataset: str, episode_name: str) -> Path:
    return RECORDINGS_DIR / sanitize_name(dataset) / "metrics" / f"{sanitize_name(episode_name)}.rrd"


def _write_property_rrd(path: Path, *, recording_id: str, property_name: str, values: dict[str, Any]) -> None:
    """A tiny .rrd carrying only property stamps, reusing the episode's recording id
    (== segment id) so a layer registration overlays the columns onto the episode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(APP_ID, recording_id=recording_id)
    rec.set_sinks(rr.FileSink(str(path)))
    rec.send_property(property_name, rr.AnyValues(**values))
    rec.flush()
    rec.disconnect()


def write_episode_edits(path: Path, *, recording_id: str, dataset: str, task: str, tag: str) -> None:
    """Rewrite an episode's curation properties (always the FULL episode field set)."""
    _write_property_rrd(path, recording_id=recording_id, property_name="episode", values={"dataset": dataset, "task": task, "tag": tag})


def write_episode_metrics(path: Path, *, recording_id: str, scores: dict[str, Any]) -> None:
    """Write the metric scoreboard as a separate ``metrics`` property (own layer, so it
    never fights the human-curated ``edits`` layer over the same chunks)."""
    _write_property_rrd(path, recording_id=recording_id, property_name="metrics", values=scores)


def register_layer(client: rr.catalog.CatalogClient, dataset_name: str, layer: str, paths: list[Path]) -> None:
    dataset = client.create_dataset(sanitize_name(dataset_name), exist_ok=True)
    dataset.register(
        [path.resolve().as_uri() for path in paths],
        layer_name=layer,
        on_duplicate=rr.catalog.OnDuplicateSegmentLayer.REPLACE,
    ).wait()


# --- Terminal table printing (no extra deps) ------------------------------------------


def print_table(df: pd.DataFrame, *, max_cell: int = 60) -> None:
    """Fixed-width terminal table; long cells truncated with an ellipsis."""
    if df.empty:
        print("(empty)")
        return
    headers = [str(c) for c in df.columns]
    rows = [[_clip(str(v), max_cell) for v in row] for row in df.itertuples(index=False)]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def _clip(text: str, max_cell: int) -> str:
    return text if len(text) <= max_cell else text[: max_cell - 1] + "…"


def human_size(size_bytes: float) -> str:
    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"
