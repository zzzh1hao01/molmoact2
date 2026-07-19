"""YAM local data server: one long-lived process for the collection/curation loop.

Vendored from the so100-hackathon reference (``tools/apps/so100_server.py``) and slimmed
down: on the YAM the robot hardware is owned by the gello launcher
(``YAM/gello_software/experiments/launch_yaml_collect_data.py --rerun``), which logs
takes itself -- so this server has no arms/setup control, just the infrastructure trio:

* ``--grpc-port``    (default 9876):  a Rerun gRPC *proxy* server (its own process) --
  the launcher tees every take into it, and any viewer (`rerun --connect
  rerun+http://<host>:9876/proxy`) watches live. The proxy's memory limit drops the
  oldest data, so the stream flushes itself instead of growing forever.
* ``--catalog-port`` (default 51234): an in-process Rerun catalog (``rr.server.Server``).
  On startup every ``recordings/<dataset>/*.rrd`` on disk is registered into a catalog
  dataset named after its folder -- shutting the server down loses nothing. NOTE:
  ``Server(port=0)`` is broken on 0.34.1; the port must be fixed.
* ``--control-port`` (default 8001; 8000 is the DROID inference server convention):
  a small JSON API (CORS-enabled):

  - ``GET  /status``            -- ports, recordings dir, dataset summary
  - ``GET  /datasets``          -- catalog dataset names
  - ``GET  /episodes?dataset=X``-- the dataset's registered episodes (id, task, tag,
    viewer deep link) plus the id the next recording will get (``episode_NN``, max + 1)
  - ``POST /episode/update``    -- ``{"dataset": ..., "episode": ..., "task": ..., "tag": ...}``:
    rewrite a finished episode's properties via an ``edits`` catalog layer
  - ``POST /rescan``            -- register any ``.rrd`` files that appeared on disk
    since startup (e.g. recorded while this server was down)

Run it from the 0.34.1 tooling venv, e.g.::

    uv run --python 3.11 --with 'rerun-sdk[all]==0.34.1' python -m yam_rerun.server
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence  # noqa: F401 - Sequence is used in the cast below
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

import rerun as rr

from yam_rerun.takes import (
    DEFAULT_CATALOG_PORT,
    DEFAULT_CONTROL_PORT,
    DEFAULT_GRPC_PORT,
    DEFAULT_RECORDINGS_DIR,
    SEGMENT_TAGS,
    edits_path,
    next_episode,
    register_blueprint,
    register_edits,
    register_layer,
    register_rrd,
    sanitize_name,
    scan_blueprints,
    scan_edits,
    scan_layer_files,
    scan_recordings,
    write_edits,
)

# Low-latency micro-batcher (8 ms flush, == ChunkBatcherConfig.LOW_LATENCY) for every
# recording in this process and the proxy subprocess (inherits env).
os.environ.setdefault("RERUN_FLUSH_TICK_SECS", "0.008")


def require_port(port: int, what: str) -> None:
    """Fail fast (with a helpful hint) if a port is taken.

    SO_REUSEADDR matches how the real servers bind: without it, connections still in
    TIME_WAIT from the previous run would fail this check for ~30s after every restart,
    even though the port is actually available.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("localhost", port))
        except OSError:
            raise SystemExit(f"port {port} ({what}) is already in use -- is `python -m yam_rerun.server` already running in another terminal?") from None


def spawn_proxy(grpc_port: int) -> subprocess.Popen[bytes]:
    """A Rerun gRPC proxy server in its OWN process.

    ``rerun --serve-grpc`` is a *pure* proxy: unlike ``rr.serve_grpc()`` it needs no SDK
    recording, so it does not add an empty ghost recording to every connecting viewer.
    A tiny wrapper watches the parent pid and kills the proxy if the server dies without
    cleanup (SIGKILL, crash), so it can never orphan and squat the port.

    The wrapper runs the ``rerun`` *binary* directly: ``python -m rerun`` is a launcher
    that runs the binary as a grandchild, which SIGTERM would never reach -- the proxy
    would outlive its session and keep the port + buffered recordings alive.
    """
    import rerun_cli.__main__ as rerun_cli

    # Mirror rerun_cli.__main__'s binary resolution: on macOS the binary ships
    # inside an app bundle, elsewhere it sits next to the package.
    cli_dir = Path(rerun_cli.__file__).parent
    if binary := os.environ.get("RERUN_CLI_PATH"):
        pass
    else:
        bundled = cli_dir / "Rerun.app" / "Contents" / "MacOS" / "Rerun"
        if sys.platform == "darwin" and bundled.exists():
            binary = str(bundled)
        else:
            binary = rerun_cli.add_exe_suffix(str(cli_dir / "rerun"))
    code = "\n".join(
        (
            "import os, signal, subprocess, sys, time",
            f"proc = subprocess.Popen([{binary!r}, '--serve-grpc', '--port', '{grpc_port}'])",
            "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
            "parent = os.getppid()",
            "try:",
            "    while os.getppid() == parent and proc.poll() is None:",
            "        time.sleep(1.0)",
            "finally:",
            "    proc.terminate()",
        )
    )
    return subprocess.Popen([sys.executable, "-c", code])


def wait_for_port(port: int, timeout: float = 20.0) -> None:
    """Block until something accepts TCP connections on the port."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"proxy on port {port} did not come up within {timeout:.0f}s")


def list_episodes(client: rr.catalog.CatalogClient, recordings_dir: Path, dataset: str) -> dict[str, object]:
    """One entry per registered episode (id, current properties, viewer deep link), sorted,
    plus the id the NEXT recording in this dataset will get.

    Property columns are list-typed (one value per catalog layer, ``edits`` overlaying the
    base recording), so the first element is the current value.
    """

    def first(value: object) -> str:
        if isinstance(value, str):
            return value
        try:
            return str(value[0]) if len(value) else ""  # type: ignore[arg-type, index]
        except TypeError:
            return ""

    name = sanitize_name(dataset)
    episodes: list[dict[str, object]] = []
    dataset_url: str | None = None
    if name in set(client.dataset_names()):
        ds = client.get_dataset(name=name)
        table = ds.segment_table().to_pandas()
        for _, row in table.iterrows():
            segment_id = str(row["rerun_segment_id"])
            stem = segment_id.removeprefix(f"{name}-")
            episodes.append(
                {
                    "episode": first(row.get("property:RecordingInfo:name")) or stem,
                    "stem": stem,
                    "task": first(row.get("property:episode:task")),
                    "tag": first(row.get("property:episode:tag")),
                    "segment_id": segment_id,
                    "viewer_url": ds.segment_url(segment_id),
                }
            )

        def sort_key(entry: dict[str, object]) -> tuple[int, int, str]:
            # Episode number first, then collision suffix (episode_1-2 etc.).
            stem = str(entry["stem"])
            m = re.match(r"episode_(\d+)(?:-(\d+))?$", stem)
            if m is None:
                return (1 << 30, 0, stem)
            return (int(m.group(1)), int(m.group(2) or 1), stem)

        episodes.sort(key=sort_key)
        if episodes:
            # The viewer's URI parser accepts /entry/<id> (the dataset's catalog table
            # screen) but NOT a bare /dataset/<id> (only with a ?segment_id query).
            dataset_url = str(episodes[0]["viewer_url"]).split("?", 1)[0].replace("/dataset/", "/entry/")
    return {"dataset": name, "episodes": episodes, "next": next_episode(recordings_dir, name), "dataset_url": dataset_url}


class Catalog:
    """The startup scan + incremental rescan of ``recordings/`` into the catalog."""

    def __init__(self, catalog_uri: str, recordings_dir: Path) -> None:
        self.catalog_uri = catalog_uri
        self.recordings_dir = recordings_dir
        self._lock = threading.Lock()
        self._registered: set[str] = set()  # absolute .rrd paths already in the catalog

    def initial_datasets(self) -> dict[str, list[str]]:
        """The on-disk datasets, for ``rr.server.Server(datasets=...)`` (bulk-registered
        at server construction -- much faster than per-file registration)."""
        datasets = scan_recordings(self.recordings_dir)
        with self._lock:
            for files in datasets.values():
                self._registered.update(str(Path(f).resolve()) for f in files)
        return datasets

    def apply_edits_and_blueprints(self) -> None:
        """Re-apply saved metadata edits + default blueprints (call after Server is up)."""
        for name, edit_files in scan_edits(self.recordings_dir).items():
            register_edits(self.catalog_uri, name, edit_files)
            print(f"[scan]      dataset '{name}': {len(edit_files)} metadata edit(s) re-applied", flush=True)
        for name, metric_files in scan_layer_files(self.recordings_dir, "metrics").items():
            register_layer(self.catalog_uri, name, "metrics", metric_files)
            print(f"[scan]      dataset '{name}': {len(metric_files)} metric score(s) re-applied", flush=True)
        # Any eval* subdir is a policy-prediction layer (tools/eval_policy.py --layer):
        # "eval" for the default checkpoint, "eval1k" etc. for side-by-side comparisons.
        eval_layers = sorted({
            child.name
            for dataset_dir in self.recordings_dir.iterdir() if dataset_dir.is_dir()
            for child in dataset_dir.iterdir() if child.is_dir() and child.name.startswith("eval")
        }) if self.recordings_dir.is_dir() else []
        for layer in eval_layers:
            for name, eval_files in scan_layer_files(self.recordings_dir, layer).items():
                register_layer(self.catalog_uri, name, layer, eval_files)
                print(f"[scan]      dataset '{name}': {len(eval_files)} {layer} prediction file(s) re-applied", flush=True)
        for name, blueprint_file in scan_blueprints(self.recordings_dir).items():
            register_blueprint(self.catalog_uri, name, blueprint_file)
            print(f"[scan]      dataset '{name}': default blueprint re-applied", flush=True)

    def rescan(self) -> dict[str, object]:
        """Register any ``.rrd`` files that appeared since the last scan."""
        added: dict[str, list[str]] = {}
        for name, files in scan_recordings(self.recordings_dir).items():
            for file in files:
                resolved = str(Path(file).resolve())
                with self._lock:
                    if resolved in self._registered:
                        continue
                register_rrd(self.catalog_uri, name, Path(file))
                with self._lock:
                    self._registered.add(resolved)
                added.setdefault(name, []).append(Path(file).name)
                print(f"[rescan]    registered {file} in dataset '{name}'", flush=True)
        self.apply_edits_and_blueprints()
        return {"added": added}

    def note_registered(self, path: Path) -> None:
        with self._lock:
            self._registered.add(str(path.resolve()))

    def update_episode(self, *, dataset: str, episode: str, task: str, tag: str) -> dict[str, object]:
        """Rewrite a finished episode's properties via an ``edits`` layer."""
        name = sanitize_name(dataset)
        stem = sanitize_name(episode)
        if not (self.recordings_dir / name / f"{stem}.rrd").exists():
            raise RuntimeError(f"no episode '{stem}' in dataset '{name}'")
        path = edits_path(self.recordings_dir, name, stem)
        write_edits(path, recording_id=f"{name}-{stem}", task=task, tag=tag)
        register_edits(self.catalog_uri, name, [path])
        print(f"[catalog]   updated {name}/{stem} (task: {task!r}, tag: {tag!r})", flush=True)
        return {"dataset": name, "episode": stem, "task": task, "tag": tag}


def make_handler(
    catalog: Catalog,
    catalog_client_factory: Callable[[], rr.catalog.CatalogClient],
    status: dict[str, object],
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # silence request logging
            pass

        def _send_json(self, code: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length") or 0)
            if length == 0:
                return {}
            try:
                parsed = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}

        def do_OPTIONS(self) -> None:  # CORS preflight
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if path == "/status":
                    self._send_json(200, {**status, "datasets": scan_recordings(catalog.recordings_dir), "tags": list(SEGMENT_TAGS)})
                elif path == "/datasets":
                    names = list(catalog_client_factory().dataset_names())
                    self._send_json(200, {"datasets": sorted(names)})
                elif path == "/episodes":
                    dataset = (parse_qs(urlparse(self.path).query).get("dataset") or [""])[0].strip()
                    if not dataset:
                        self._send_json(400, {"error": "missing ?dataset=<name>"})
                        return
                    self._send_json(200, list_episodes(catalog_client_factory(), catalog.recordings_dir, dataset))
                else:
                    self._send_json(404, {"error": "not found"})
            except Exception as err:  # noqa: BLE001 - surface any failure to the caller
                self._send_json(500, {"error": f"{type(err).__name__}: {err}"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            body = self._body()
            try:
                if path == "/episode/update":
                    payload = catalog.update_episode(
                        dataset=str(body.get("dataset") or ""),
                        episode=str(body.get("episode") or ""),
                        task=str(body.get("task") or ""),
                        tag=str(body.get("tag") or ""),
                    )
                elif path == "/rescan":
                    payload = catalog.rescan()
                else:
                    self._send_json(404, {"error": "not found"})
                    return
                self._send_json(200, payload)
            except (RuntimeError, SystemExit) as err:
                self._send_json(200, {"error": str(err)})
            except Exception as err:  # noqa: BLE001
                self._send_json(500, {"error": f"{type(err).__name__}: {err}"})

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grpc-port", type=int, default=DEFAULT_GRPC_PORT, help="Rerun gRPC proxy port (live streams)")
    parser.add_argument("--catalog-port", type=int, default=DEFAULT_CATALOG_PORT, help="local Rerun catalog port (must be fixed; port 0 is broken)")
    parser.add_argument("--control-port", type=int, default=DEFAULT_CONTROL_PORT, help="JSON control API port")
    parser.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR, help="folder of <dataset>/<episode>.rrd files")
    args = parser.parse_args(argv)

    # SIGTERM (`pkill`, `kill`) must run the `finally` block below like Ctrl-C does --
    # otherwise the proxy child is orphaned on its port.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    for port, what in ((args.grpc_port, "gRPC proxy"), (args.catalog_port, "catalog"), (args.control_port, "control API")):
        require_port(port, what)

    # 1) gRPC proxy server -- what the collection hook tees live takes into.
    proxy_uri = f"rerun+http://localhost:{args.grpc_port}/proxy"
    proxy_proc = spawn_proxy(args.grpc_port)
    print(f"gRPC proxy server:  {proxy_uri}")

    # 2) In-process catalog server, pre-loaded with everything already on disk:
    #    each recordings/<dataset>/ folder becomes a catalog dataset. This is what makes
    #    the server safe to shut down -- the next start re-registers it all.
    catalog_uri = f"rerun+http://localhost:{args.catalog_port}"
    catalog = Catalog(catalog_uri, args.recordings_dir)
    datasets = catalog.initial_datasets()
    for name, files in datasets.items():
        print(f"[scan]      dataset '{name}': {len(files)} recording(s) from {args.recordings_dir / name}", flush=True)
    # dict is invariant in its value type, hence the cast to Server's accepted union.
    server_datasets = cast("dict[str, str | os.PathLike[str] | Sequence[str | os.PathLike[str]]] | None", datasets or None)
    catalog_server = rr.server.Server(port=args.catalog_port, datasets=server_datasets)
    print(f"Catalog server:     {catalog_uri}")
    catalog.apply_edits_and_blueprints()

    httpd: ThreadingHTTPServer | None = None
    try:
        # 3) Control server -- the JSON API curl (and later tools/ CLIs) talk to.
        status: dict[str, object] = {
            "proxy_uri": proxy_uri,
            "catalog_uri": catalog_uri,
            "recordings_dir": str(args.recordings_dir),
        }
        handler = make_handler(catalog, lambda: rr.catalog.CatalogClient(catalog_uri), status)
        httpd = ThreadingHTTPServer(("localhost", args.control_port), handler)
        print(f"Control API:        http://localhost:{args.control_port}")
        print()
        print("Leave this running. Record takes with `launch_yaml_collect_data.py --rerun`; `POST /rescan` picks up files recorded while this was down.")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        if httpd is not None:
            httpd.shutdown()
        catalog_server.shutdown()
        proxy_proc.terminate()
        try:
            proxy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()


if __name__ == "__main__":
    main()
