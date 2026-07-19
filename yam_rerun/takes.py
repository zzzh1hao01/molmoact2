"""Recording *takes*: YAM episodes written to ``recordings/<dataset>/<episode>.rrd``.

Vendored (near-copy) from the so100-hackathon reference (``src/so100_hackathon/takes.py``,
rerun-sdk 0.34.1) and adapted for the bimanual YAM. Shared between the long-lived local
server (``yam_rerun/server.py``) and the gello collection hook
(``YAM/gello_software/experiments/launch_yaml_collect_data.py``). A take is:

1. a fresh :class:`rerun.RecordingStream` with the episode's metadata stamped on as
   *recording properties* (they become ``property:...`` columns in the catalog),
2. teed to disk (and optionally the live gRPC proxy) while the data source logs into it,
3. on stop: the tag property is sent, the file sink is closed, the ``.rrd`` compacted
   with ``rerun rrd optimize``, and the file registered to the local catalog.

Registration is optional by design: the server re-registers everything found under
``recordings/`` on startup, so files recorded while it was down are picked up then.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

# The catalog client refuses localhost tokens unless we opt out of the host check.
# Set here (not only in server.py) because the collection hook also registers episodes.
os.environ.setdefault("RERUN_INSECURE_SKIP_HOST_CHECK", "1")

APP_ID = "yam"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDINGS_DIR = REPO_ROOT / "recordings"
"""Repo-relative home of ``<dataset>/<episode>.rrd`` files."""

DEFAULT_GRPC_PORT = 9876
DEFAULT_CATALOG_PORT = 51234  # NOTE: Server(port=0) is broken on 0.34.1 -- fixed port.
DEFAULT_CONTROL_PORT = 8001  # 8000 is taken by the DROID inference-server convention.
DEFAULT_PROXY_URI = f"rerun+http://localhost:{DEFAULT_GRPC_PORT}/proxy"
DEFAULT_CATALOG_URI = f"rerun+http://localhost:{DEFAULT_CATALOG_PORT}"

SEGMENT_TAGS = ("Good episode", "Bad episode", "Needs review")
"""Suggested curation tags; free-form strings are accepted everywhere."""


def sanitize_name(name: str) -> str:
    """A filesystem- and catalog-safe version of a user-provided name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return cleaned or time.strftime("%Y%m%d-%H%M%S")


def episode_path(recordings_dir: Path, dataset: str, episode: str) -> Path:
    """``recordings/<dataset>/<episode>.rrd``, suffixed ``-2``, ``-3``, ... on collision."""
    directory = recordings_dir / sanitize_name(dataset)
    stem = sanitize_name(episode)
    path = directory / f"{stem}.rrd"
    counter = 2
    while path.exists():
        path = directory / f"{stem}-{counter}.rrd"
        counter += 1
    return path


def next_episode(recordings_dir: Path, dataset: str) -> str:
    """The next free ``episode_NN`` id for the dataset: highest existing number + 1.

    Ids are never reused -- deleting ``episode_03`` does NOT make the next take
    ``episode_03`` again, so an episode id always refers to one take, forever.
    """
    directory = recordings_dir / sanitize_name(dataset)
    highest = 0
    if directory.is_dir():
        for path in directory.glob("episode_*.rrd"):
            match = re.match(r"episode_(\d+)", path.stem)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"episode_{highest + 1:02d}"


def scan_recordings(recordings_dir: Path) -> dict[str, list[str]]:
    """Map each ``recordings/<dataset>/`` subdir to its ``.rrd`` files (for the startup scan)."""
    datasets: dict[str, list[str]] = {}
    if recordings_dir.is_dir():
        for child in sorted(recordings_dir.iterdir()):
            if child.is_dir():
                rrds = sorted(str(p) for p in child.glob("*.rrd"))
                if rrds:
                    datasets[child.name] = rrds
    return datasets


def begin_take(path: Path, *, episode: str, dataset: str, task: str, proxy_uri: str | None) -> rr.RecordingStream:
    """Create the take's recording stream: sinks wired, metadata stamped, ready to log into.

    The recording id doubles as the catalog segment id, so it is derived from the file stem
    (unique within the dataset by construction, see :func:`episode_path`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(
        APP_ID,
        recording_id=f"{sanitize_name(dataset)}-{path.stem}",
        # Small frequent chunks: keeps live-viewer latency low; `rerun rrd optimize` on
        # stop re-compacts the file (the .rrd-size risk note in the port plan).
        batcher_config=rr.ChunkBatcherConfig.LOW_LATENCY(),
    )
    sinks: list[rr.GrpcSink | rr.FileSink] = [rr.FileSink(str(path))]
    if proxy_uri is not None:
        sinks.insert(0, rr.GrpcSink(url=proxy_uri))
    rec.set_sinks(*sinks)

    rec.send_recording_name(episode)
    stamp_properties(rec, dataset=dataset, task=task, tag="")
    # The task doubles as the LeRobot task string (read back by the Phase-3 exporter).
    rec.log("/task", rr.TextDocument(task), static=True)
    return rec


def stamp_properties(rec: rr.RecordingStream, *, dataset: str, task: str, tag: str) -> None:
    """Stamp the episode properties onto ``rec`` -- always the FULL set, never a subset.

    The catalog does not resolve conflicting stamps per component: a later stamp with
    FEWER columns loses to an earlier, wider one (verified on 0.33 -- a ``tag``-only
    stamp never overrides a ``tag`` sent earlier together with ``task``). Stamping every
    column every time keeps the chunks the same shape, where plain latest-wins applies.
    """
    rec.send_property("episode", rr.AnyValues(dataset=dataset, task=task, tag=tag))


def finish_take(rec: rr.RecordingStream, *, dataset: str, task: str, tag: str, proxy_uri: str | None) -> None:
    """Stamp the final properties, then close the file sink (flushes the ``.rrd`` footer)."""
    stamp_properties(rec, dataset=dataset, task=task, tag=tag)
    # Swapping sinks drops the FileSink; keep streaming to the proxy if there is one.
    if proxy_uri is not None:
        rec.set_sinks(rr.GrpcSink(url=proxy_uri))
    else:
        rec.flush()
        rec.disconnect()


def edits_path(recordings_dir: Path, dataset: str, episode_stem: str) -> Path:
    """Where an episode's property edits live: ``recordings/<dataset>/edits/<episode>.rrd``.

    Kept out of the dataset folder itself so the startup scan does not register the
    edits file as a base-layer recording.
    """
    return recordings_dir / sanitize_name(dataset) / "edits" / f"{episode_stem}.rrd"


def write_edits(path: Path, *, recording_id: str, task: str, tag: str) -> None:
    """Write an *edits layer* recording: just the updated episode properties.

    It reuses the episode's recording id (== catalog segment id), so registering it
    under a separate layer overlays the same ``property:episode:...`` columns on top of
    the values baked into the original ``.rrd`` -- rewriting the metadata without
    touching the data. Rewritten wholesale on every update (one file per episode).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(APP_ID, recording_id=recording_id)
    rec.set_sinks(rr.FileSink(str(path)))
    rec.send_property("episode", rr.AnyValues(task=task, tag=tag))
    rec.flush()
    rec.disconnect()


def register_layer(catalog_uri: str, dataset_name: str, layer_name: str, paths: list[Path]) -> None:
    """Register property overlay files under ``layer_name``, replacing any previous ones."""
    client = rr.catalog.CatalogClient(catalog_uri)
    dataset = client.create_dataset(sanitize_name(dataset_name), exist_ok=True)
    dataset.register(
        [path.resolve().as_uri() for path in paths],
        layer_name=layer_name,
        on_duplicate=rr.catalog.OnDuplicateSegmentLayer.REPLACE,
    ).wait()


def register_edits(catalog_uri: str, dataset_name: str, paths: list[Path]) -> None:
    """Register edits files as the ``edits`` layer, replacing any previous edits."""
    register_layer(catalog_uri, dataset_name, "edits", paths)


def scan_layer_files(recordings_dir: Path, subdir: str) -> dict[str, list[Path]]:
    """Map each dataset to its ``<subdir>/*.rrd`` overlay files (re-registered on startup)."""
    found: dict[str, list[Path]] = {}
    if recordings_dir.is_dir():
        for child in sorted(recordings_dir.iterdir()):
            files = sorted((child / subdir).glob("*.rrd")) if child.is_dir() else []
            if files:
                found[child.name] = files
    return found


def scan_edits(recordings_dir: Path) -> dict[str, list[Path]]:
    """Map each dataset to its ``edits/*.rrd`` files (re-registered on startup)."""
    return scan_layer_files(recordings_dir, "edits")


def save_dataset_blueprint(recordings_dir: Path, dataset: str, blueprint: rrb.Blueprint) -> Path:
    """Write the dataset's blueprint to ``recordings/<dataset>/blueprint/blueprint.rrd``.

    One blueprint per dataset, written ONCE: an existing file is left untouched. The
    catalog reads it lazily from disk (rewriting it breaks the live registration with
    'malformed response'), and it may carry user customizations -- delete the file to
    have the next recording regenerate it. Kept in its own subdir so neither the
    episode scan nor the edits scan picks it up."""
    path = recordings_dir / sanitize_name(dataset) / "blueprint" / "blueprint.rrd"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        blueprint.save(APP_ID, str(path))
    return path


def register_blueprint(catalog_uri: str, dataset_name: str, path: Path) -> bool:
    """Set the dataset's default blueprint, unless it already has one.

    Without a default blueprint, episodes opened from the catalog get the viewer's
    heuristic layout. The blueprint is the same for every recording of the dataset, so
    it is registered at most once per catalog lifetime (the in-process catalog forgets
    it on shutdown; the startup scan re-registers from disk)."""
    client = rr.catalog.CatalogClient(catalog_uri)
    dataset = client.create_dataset(sanitize_name(dataset_name), exist_ok=True)
    if dataset.default_blueprint() is not None:
        return False
    dataset.register_blueprint(path.resolve().as_uri(), set_default=True)
    return True


def scan_blueprints(recordings_dir: Path) -> dict[str, Path]:
    """Map each dataset to its saved default blueprint (re-registered on startup)."""
    blueprints: dict[str, Path] = {}
    if recordings_dir.is_dir():
        for child in sorted(recordings_dir.iterdir()):
            path = child / "blueprint" / "blueprint.rrd"
            if child.is_dir() and path.exists():
                blueprints[child.name] = path
    return blueprints


def optimize_rrd(path: Path) -> None:
    """Compact the recording's chunks in place via ``rerun rrd optimize``."""
    tmp = path.with_name(path.name + ".tmp")
    proc = subprocess.run(
        [sys.executable, "-m", "rerun", "rrd", "optimize", str(path), "-o", str(tmp)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"rerun rrd optimize failed: {proc.stderr.strip()}")
    os.replace(tmp, path)
    print(f"[optimize]  compacted {path}", flush=True)


def register_rrd(catalog_uri: str, dataset_name: str, path: Path) -> dict[str, object]:
    """Register the ``.rrd`` into the catalog dataset, returning ids + viewer deep links."""
    client = rr.catalog.CatalogClient(catalog_uri)
    dataset = client.create_dataset(sanitize_name(dataset_name), exist_ok=True)
    handle = dataset.register([path.resolve().as_uri()])
    result = handle.wait()
    segment_ids = [str(seg) for seg in (getattr(result, "segment_ids", []) or [])]
    viewer_urls = [dataset.segment_url(seg) for seg in segment_ids]
    return {
        "dataset": sanitize_name(dataset_name),
        "uri": path.resolve().as_uri(),
        "segment_ids": segment_ids,
        # Deep links a running web viewer can `open()` directly.
        "viewer_urls": viewer_urls,
    }
