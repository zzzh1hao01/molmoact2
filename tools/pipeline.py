"""One-shot pipeline: recordings (or a LeRobot v3 dataset) -> catalog -> viewer -> query.

Chains the individual tools so a single command goes from data on disk to an open,
queryable viewer session:

1. (only with a ``source`` argument) resolve it -- a local v3 root, or a HF hub repo
   id fetched via ``hf download`` -- and run ``import_lerobot.py``: one
   ``recordings/<dataset>/episode_NNN.rrd`` per episode (skipped when the .rrd files
   already exist; ``--reimport`` forces it)
2. ``yam_rerun.server``   -- started if not already answering on the control port
   (its startup scan registers every ``recordings/<dataset>/*.rrd`` on disk); if it
   is already up, a ``POST /rescan`` registers any fresh .rrd files instead
3. curated dataset blueprints are registered (idempotent) -- for ``--dataset``, or
   for every dataset folder found under ``--recordings-dir``
4. the native viewer opens on the catalog (``--no-viewer`` to skip)
5. the episode table (or the dataset list) is printed via the query API (``--no-query``
   to skip)

Usage::

    # just serve + view + query whatever is already under recordings/:
    python tools/pipeline.py

    # HF hub dataset, can0-first arms, wrist_1/wrist_2 cameras (the Shivakumr/yams shape):
    python tools/pipeline.py Shivakumr/yams \
        --camera-map "wrist_1=right,wrist_2=left" --swap-arms

    # local v3 root, defaults:
    python tools/pipeline.py ~/data/towels --dataset towels

Afterwards, curate + filter::

    curl -X POST http://localhost:8001/episode/update \
        -d '{"dataset": "yams", "episode": "episode_003", "tag": "Good episode"}'
    python tools/query_dataset.py --dataset yams --tag "Good episode"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import _yam_catalog as yc

sys.path.insert(0, str(yc.REPO_ROOT))  # tools/ scripts run with sys.path[0] == tools/

from yam_rerun import takes  # noqa: E402

TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_CONTROL_PORT = 8001


# --- Step 1: source -------------------------------------------------------------------


def resolve_source(source: str) -> Path:
    path = Path(source).expanduser()
    if path.exists():
        return path
    if "/" not in source or source.startswith((".", "/")):
        raise SystemExit(f"source {source!r} is neither a local path nor a HF repo id")
    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf is None:
        raise SystemExit(f"{source!r} looks like a HF repo id but no `hf` CLI is on PATH")
    print(f"[pipeline]  downloading hub dataset {source} ...")
    result = subprocess.run(
        [hf, "download", source, "--repo-type", "dataset"],
        check=True, capture_output=True, text=True,
    )
    snapshot = Path(result.stdout.strip().splitlines()[-1])
    if not (snapshot / "meta" / "info.json").exists():
        raise SystemExit(f"downloaded snapshot {snapshot} has no meta/info.json")
    return snapshot


# --- Step 2: import -------------------------------------------------------------------


def ensure_imported(args: argparse.Namespace, source: Path) -> None:
    out_dir = args.recordings_dir / args.dataset
    existing = sorted(out_dir.glob("*.rrd"))
    if existing and not args.reimport:
        print(f"[pipeline]  {len(existing)} .rrd file(s) already in {out_dir}, skipping import "
              f"(--reimport to redo)")
        return
    command = [
        sys.executable, str(TOOLS_DIR / "import_lerobot.py"),
        "--source", str(source), "--dataset", args.dataset,
        "--recordings-dir", str(args.recordings_dir),
        "--jpeg-quality", str(args.jpeg_quality),
        "--no-register",  # step 3 (server start or rescan) registers everything
    ]
    if args.episodes:
        command += ["--episodes", args.episodes]
    if args.camera_map:
        command += ["--camera-map", args.camera_map]
    if args.swap_arms:
        command += ["--swap-arms"]
    if args.tag:
        command += ["--tag", args.tag]
    subprocess.run(command, check=True, cwd=TOOLS_DIR)


# --- Step 3: server -------------------------------------------------------------------


def control_get(port: int, route: str, *, post: bool = False, timeout: float = 3.0) -> dict | None:
    request = urllib.request.Request(
        f"http://localhost:{port}{route}", method="POST" if post else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def ensure_server(args: argparse.Namespace) -> None:
    if control_get(args.control_port, "/status") is not None:
        print("[pipeline]  server already running; rescanning for new .rrd files")
        control_get(args.control_port, "/rescan", post=True, timeout=120)
        return
    log_path = args.recordings_dir / ".server.log"
    print(f"[pipeline]  starting yam_rerun.server (log: {log_path})")
    with open(log_path, "ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "yam_rerun.server",
             "--catalog-port", str(args.catalog_port), "--control-port", str(args.control_port)],
            cwd=yc.REPO_ROOT, stdout=log, stderr=log, start_new_session=True,
        )
    deadline = time.time() + 180
    while time.time() < deadline:
        if control_get(args.control_port, "/status") is not None:
            return
        time.sleep(2)
    raise SystemExit(f"server did not answer on port {args.control_port}; see {log_path}")


# --- Step 4: blueprint ----------------------------------------------------------------


def ensure_blueprints(args: argparse.Namespace) -> None:
    from yam_rerun import blueprint as bp
    from yam_rerun.urdf_yam import DualYam

    if args.dataset:
        names = [args.dataset]
    else:  # every dataset folder on disk that actually holds takes
        names = sorted(
            d.name for d in args.recordings_dir.iterdir()
            if d.is_dir() and any(d.glob("*.rrd"))
        )
    robot = DualYam.create()
    visual_paths = [arm.visual_geometries_path for arm in robot.arms]
    for name in names:
        bp.register_dataset_blueprint(
            yc.catalog_uri(args.catalog_port), args.recordings_dir, name,
            visual_paths=visual_paths,
        )
        print(f"[pipeline]  blueprint registered for dataset '{name}'")


# --- Steps 5 + 6: viewer + query ------------------------------------------------------


def open_viewer(args: argparse.Namespace) -> None:
    rerun_bin = Path(sys.executable).with_name("rerun")
    subprocess.Popen(
        [str(rerun_bin), "--port", "auto", yc.catalog_uri(args.catalog_port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    print(f"[pipeline]  viewer opening on {yc.catalog_uri(args.catalog_port)}")


def query_summary(args: argparse.Namespace) -> None:
    command = [sys.executable, str(TOOLS_DIR / "query_dataset.py"), "--catalog-port", str(args.catalog_port)]
    if args.dataset:  # without --dataset this lists every dataset in the catalog
        command += ["--dataset", args.dataset]
        if args.tag:
            command += ["--tag", args.tag]
    subprocess.run(command, check=True, cwd=TOOLS_DIR)


# --- CLI ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default=None,
                        help="LeRobot v3 root (dir with meta/, data/, videos/) or a HF hub repo id; "
                             "omit to serve/view/query the recordings already on disk")
    parser.add_argument("--dataset", default=None, help="catalog dataset name (default: source dir/repo name)")
    parser.add_argument("--episodes", default=None, help="comma-separated episode indices (default: all)")
    parser.add_argument("--camera-map", default="", help='rename video key stems, e.g. "wrist_1=right,wrist_2=left"')
    parser.add_argument("--swap-arms", action="store_true", help="dims 0-6 are the RIGHT arm; swap halves on import")
    parser.add_argument("--tag", default="", help="tag stamped on imported episodes; also filters the final query")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--reimport", action="store_true", help="re-run the import even if .rrd files exist")
    parser.add_argument("--no-viewer", action="store_true", help="don't open the native viewer")
    parser.add_argument("--no-query", action="store_true", help="don't print the episode table at the end")
    parser.add_argument("--recordings-dir", type=Path, default=takes.DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--catalog-port", type=int, default=yc.DEFAULT_CATALOG_PORT)
    parser.add_argument("--control-port", type=int, default=DEFAULT_CONTROL_PORT)
    args = parser.parse_args()

    if args.source is not None:
        source = resolve_source(args.source)
        args.dataset = takes.sanitize_name(args.dataset or Path(args.source).name)
        ensure_imported(args, source)
    elif args.dataset:
        args.dataset = takes.sanitize_name(args.dataset)
    ensure_server(args)
    ensure_blueprints(args)
    if not args.no_viewer:
        open_viewer(args)
    if not args.no_query:
        query_summary(args)
    name = args.dataset or "<dataset>"
    print(f"\n[pipeline]  done. next steps:\n"
          f"  tag:    curl -X POST http://localhost:{args.control_port}/episode/update "
          f"-d '{{\"dataset\": \"{name}\", \"episode\": \"episode_000\", \"tag\": \"Good episode\"}}'\n"
          f"  filter: python tools/query_dataset.py --dataset {name} --tag \"Good episode\"\n"
          f"  series: python tools/query_dataset.py --dataset {name} --entity left_arm/position")


if __name__ == "__main__":
    main()
