"""Modal serving app for fine-tuned MolmoAct2-BimanualYAM checkpoints.

Serves `scripts/serve_policy.py` (the lerobot-policy HTTP server, `yam` image
preset: [top_cam, left_cam, right_cam]) against a trainer checkpoint on the
`molmoact2-yam-fold` Modal Volume — the same volume `modal_train.py` writes to
(`/vol/checkpoints/<exp-name>/stepN`). The lerobot `MolmoAct2Policy` loads
olmo trainer checkpoints directly, so no HF conversion is needed; LoRA runs
must be merged first (see `::merge` below).

The wire protocol matches `examples/yam/molmoact_client.py`: POST `/act` with
json_numpy-encoded `top_cam`/`left_cam`/`right_cam`/`instruction`/`state`,
response `{"actions": ...}`. Point `eval.molmoact_server` in
`examples/yam/configs/yam_left.yaml` at the deployed URL (the client appends
`/act` itself). Note the client's `normalization_tag` payload key is ignored
by serve_policy.py — the tag comes from `--norm_tag`, baked in at deploy time
(default `yam_dual_molmoact2`, which the `yam_fold` mixture deliberately
reuses).

Workflow once fine-tuning has produced a checkpoint:

    # 1. See what's on the volume
    modal run experiments/modal_serve.py::ls
    modal run experiments/modal_serve.py::ls --path checkpoints/molmoact2-yam-fold-lora

    # 2. LoRA runs only: merge adapters + action expert into one checkpoint.
    #    --base-dir is the olmo-format base VLM checkpoint the run started
    #    from (find it under /vol/hf via ::ls; the run's config.yaml records
    #    the exact path). Writes <full-dir>-merged by default.
    modal run experiments/modal_serve.py::merge \
        --base-dir hf/molmoact2/checkpoints/<base> \
        --full-dir checkpoints/molmoact2-yam-fold-lora/step50000

    # 3. Deploy the server (checkpoint is baked in at deploy time; relative
    #    paths resolve against /vol/checkpoints)
    MODAL_SERVE_CHECKPOINT=molmoact2-yam-fold-lora/step50000-merged \
        modal deploy experiments/modal_serve.py
    # prints a persistent https://...modal.run URL

    # 4. Smoke-test with dummy frames before touching the robot.
    #    Keep MODAL_SERVE_CHECKPOINT set here too: every `modal run` of this
    #    file registers the app's web endpoint, and against a dev-mode URL
    #    (`modal serve` session) an unconfigured run steals the endpoint and
    #    its containers crash-loop on the missing checkpoint.
    MODAL_SERVE_CHECKPOINT=<...> \
        modal run experiments/modal_serve.py::smoke --url https://<...>.modal.run

Notes:
  - GPU defaults to H100 because the shared training image builds flash-attn
    for arch 9.0 only. Override with MODAL_SERVE_GPU_TYPE=<type> at deploy
    time — TODO: verify at first launch whether the serving path actually
    invokes flash-attn kernels (checkpoint config decides the attention
    backend); if it runs on sdpa, an L4/A10G is ~4x cheaper.
  - The container scales to zero after SCALEDOWN_WINDOW_S idle; the next
    request cold-starts (~minutes of model load). Before a robot session,
    warm it with `curl <url>/health` and wait for a 200.
  - Extra serve_policy.py flags (e.g. --disable_inference_cuda_graph,
    --n_action_steps 25) can be passed via MODAL_SERVE_EXTRA_ARGS at deploy
    time.
  - This module does no network/GPU work at import time.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

# Reuse the training app's image (fully cached after a training launch),
# volume, and path constants so serve sees exactly what train wrote.
# Locally modal_train.py is a sibling of this file; in the Modal runner this
# file lands at /root/modal_serve.py and modal_train.py is only there because
# the image below mounts it to /root/modal_train.py explicitly.
_HERE = Path(__file__).resolve().parent
_MODAL_TRAIN_PATH = _HERE / "modal_train.py"
if not _MODAL_TRAIN_PATH.is_file():
    raise ModuleNotFoundError(f"modal_train.py not found next to {__file__}")
sys.path.insert(0, str(_HERE))
from modal_train import (  # noqa: E402
    CHECKPOINT_ROOT,
    HF_SECRET_NAME,
    REMOTE_EXPERIMENTS_DIR,
    VOLUME_MOUNT,
    VOLUME_NAME,
    image as train_image,
    volume,
)

GPU_TYPE = os.environ.get("MODAL_SERVE_GPU_TYPE", "H100")

# Baked into the image env at deploy time (env layers are instant; the heavy
# layers stay cached). Inside the container the same os.environ.get calls
# read the baked values back, so local and remote stay consistent.
SERVE_CHECKPOINT = os.environ.get("MODAL_SERVE_CHECKPOINT", "")
SERVE_NORM_TAG = os.environ.get("MODAL_SERVE_NORM_TAG", "yam_dual_molmoact2")
SERVE_EXTRA_ARGS = os.environ.get("MODAL_SERVE_EXTRA_ARGS", "")

SERVE_PORT = 8000
STARTUP_TIMEOUT_S = 30 * 60  # checkpoint load from volume + warmup
SCALEDOWN_WINDOW_S = 15 * 60

app = modal.App("molmoact2-yam-serve")

image = train_image.env(
    {
        "MODAL_SERVE_CHECKPOINT": SERVE_CHECKPOINT,
        "MODAL_SERVE_NORM_TAG": SERVE_NORM_TAG,
        "MODAL_SERVE_EXTRA_ARGS": SERVE_EXTRA_ARGS,
    }
    # Runtime mount (instant, no layer rebuild) so the in-container import of
    # this module can resolve `import modal_train` as a /root sibling.
).add_local_file(str(_MODAL_TRAIN_PATH), "/root/modal_train.py")


def _resolve_checkpoint_dir(raw: str) -> str:
    """Resolve MODAL_SERVE_CHECKPOINT to an absolute dir on the volume.

    Relative paths resolve against CHECKPOINT_ROOT. Prefers a sibling
    `<dir>-unsharded` when present (serve_policy.py: "unsharded preferred").
    Runs inside the container, so listing errors show what actually exists.
    """
    if not raw:
        raise RuntimeError(
            "MODAL_SERVE_CHECKPOINT is not set. Redeploy with e.g.\n"
            "  MODAL_SERVE_CHECKPOINT=<exp-name>/stepN modal deploy "
            "experiments/modal_serve.py\n"
            f"Available under {CHECKPOINT_ROOT}: "
            + ", ".join(sorted(os.listdir(CHECKPOINT_ROOT)))
            if os.path.isdir(CHECKPOINT_ROOT)
            else "(no checkpoints directory on the volume yet)"
        )
    path = raw if raw.startswith("/") else f"{CHECKPOINT_ROOT}/{raw}"
    path = path.rstrip("/")
    unsharded = f"{path}-unsharded"
    if os.path.isdir(unsharded) and not path.endswith(("-unsharded", "-merged")):
        print(f"[modal_serve] using unsharded sibling: {unsharded}")
        path = unsharded
    if not os.path.isdir(path):
        parent = os.path.dirname(path)
        siblings = sorted(os.listdir(parent)) if os.path.isdir(parent) else []
        raise RuntimeError(
            f"Checkpoint dir not found on volume: {path}\n"
            f"Contents of {parent or '/'}: {siblings}"
        )
    # A raw LoRA step dir has adapter siblings; its own weights are the
    # un-merged base + action expert. Serving it silently drops the LoRA
    # deltas — insist on the merged artifact instead.
    if not path.endswith("-merged") and os.path.isdir(f"{path.removesuffix('-unsharded')}-lora-llm"):
        raise RuntimeError(
            f"{path} looks like a LoRA run (found -lora-llm sibling). "
            "Merge first:  modal run experiments/modal_serve.py::merge "
            f"--base-dir <base> --full-dir {raw.removesuffix('-unsharded')}  "
            "then deploy with the -merged dir."
        )
    return path


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    scaledown_window=SCALEDOWN_WINDOW_S,
)
@modal.concurrent(max_inputs=8)  # health checks alongside /act; serve_policy locks internally
@modal.web_server(SERVE_PORT, startup_timeout=STARTUP_TIMEOUT_S)
def serve() -> None:
    env = os.environ.copy()
    token = env.get("HF_ACCESS_TOKEN", "")
    if token and not env.get("HF_TOKEN"):
        env["HF_TOKEN"] = token
    # olmo/data/get_dataset.py wants this at import in some paths; nothing is
    # read from it when serving.
    env.setdefault("MOLMO_DATA_DIR", f"{VOLUME_MOUNT}/molmo_data")
    os.makedirs(env["MOLMO_DATA_DIR"], exist_ok=True)

    checkpoint = _resolve_checkpoint_dir(env.get("MODAL_SERVE_CHECKPOINT", ""))
    cmd = [
        sys.executable,
        "scripts/serve_policy.py",
        "--checkpoint", checkpoint,
        "--image_keys", "yam",
        "--norm_tag", env.get("MODAL_SERVE_NORM_TAG", "yam_dual_molmoact2"),
        "--host", "0.0.0.0",
        "--port", str(SERVE_PORT),
    ]
    cmd += shlex.split(env.get("MODAL_SERVE_EXTRA_ARGS", ""))
    print("[modal_serve] launching:\n  " + " ".join(cmd))
    subprocess.Popen(cmd, cwd=REMOTE_EXPERIMENTS_DIR, env=env)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    memory=65536,  # merge_lora builds the full model on CPU
    cpu=8,
    timeout=2 * 60 * 60,
)
def merge_remote(base_dir: str, full_dir: str, output_dir: str) -> str:
    def _abs(p: str) -> str:
        return p if p.startswith("/") else f"{VOLUME_MOUNT}/{p}"

    cmd = [
        sys.executable,
        "scripts/merge_lora.py",
        "--base_dir", _abs(base_dir),
        "--full_dir", _abs(full_dir),
        "--output_dir", _abs(output_dir),
    ]
    print("[modal_serve] merging:\n  " + " ".join(cmd))
    subprocess.run(cmd, cwd=REMOTE_EXPERIMENTS_DIR, check=True)
    volume.commit()
    return _abs(output_dir)


@app.local_entrypoint()
def merge(base_dir: str, full_dir: str, output_dir: str = ""):
    """Merge a LoRA run into a servable checkpoint (paths relative to /vol).

    TODO: verify --base-dir at first use — it must be the olmo-format base
    VLM checkpoint the training run started from (train_lerobot.py caches HF
    ids on the volume; the run's config.yaml records the resolved path).
    """
    # Default next to the LoRA step so _resolve_checkpoint_dir's -merged
    # preference and the deploy examples line up.
    output_dir = output_dir or f"{full_dir.rstrip('/')}-merged"
    if not full_dir.startswith("/") and not full_dir.startswith("checkpoints/"):
        full_dir = f"checkpoints/{full_dir}"
        output_dir = output_dir if output_dir.startswith(("/", "checkpoints/")) else f"checkpoints/{output_dir}"
    merged = merge_remote.remote(base_dir=base_dir, full_dir=full_dir, output_dir=output_dir)
    rel = merged.removeprefix(f"{VOLUME_MOUNT}/").removeprefix("checkpoints/")
    print(
        f"[modal_serve] merged checkpoint at {merged}\n"
        f"Deploy with:\n  MODAL_SERVE_CHECKPOINT={rel} "
        "modal deploy experiments/modal_serve.py"
    )


@app.local_entrypoint()
def ls(path: str = "checkpoints"):
    """List a directory on the '{VOLUME_NAME}' volume (default: checkpoints/)."""
    entries = volume.listdir(path)
    if not entries:
        print(f"(empty) {VOLUME_NAME}:/{path}")
        return
    for entry in sorted(entries, key=lambda e: e.path):
        print(entry.path)


@app.local_entrypoint()
def smoke(url: str):
    """POST dummy YAM frames to a deployed server and print the result.

    Uses plain JSON lists (serve_policy decodes those too), so it needs no
    robot-side deps. Expect the first call to take minutes on a cold start.
    """
    import json
    import time
    import urllib.request

    url = url.rstrip("/")
    with urllib.request.urlopen(f"{url}/health", timeout=STARTUP_TIMEOUT_S) as r:
        health = json.loads(r.read())
    print(f"[smoke] /health: {json.dumps(health, indent=2)}")

    frame = [[[0, 0, 0]] * 64] * 64  # 64x64x3 black; the processor resizes
    payload = {
        "top_cam": frame,
        "left_cam": frame,
        "right_cam": frame,
        "instruction": "smoke test",
        "state": [0.0] * 14,
    }
    req = urllib.request.Request(
        f"{url}/act",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=STARTUP_TIMEOUT_S) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    actions = out.get("actions")
    if isinstance(actions, dict) and "__ndarray__" in actions:  # json_numpy encoding
        shape, data = actions.get("shape"), actions
        print(f"[smoke] OK in {dt:.1f}s — actions ndarray shape={shape}")
    elif isinstance(actions, list):
        rows = len(actions)
        cols = len(actions[0]) if rows and isinstance(actions[0], list) else "?"
        print(f"[smoke] OK in {dt:.1f}s — actions {rows}x{cols}")
    else:
        print(f"[smoke] response in {dt:.1f}s: {str(out)[:500]}")
