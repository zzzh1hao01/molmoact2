# Serving quickstart — fine-tuned MolmoAct2 on the bimanual YAM via Modal

How to take a checkpoint produced by [`experiments/modal_train.py`](experiments/modal_train.py)
and serve it to the robot over HTTPS. The serving app is
[`experiments/modal_serve.py`](experiments/modal_serve.py); it runs
`experiments/scripts/serve_policy.py` (the lerobot `MolmoAct2Policy` server) on a
Modal GPU against the same Modal Volume the trainer writes checkpoints to, so
there is no download/upload/conversion step between training and serving.

```
robot workstation                        Modal (cloud GPU)
─────────────────                        ─────────────────
launch_yaml_eval_molmoact.py             modal_serve.py :: serve
  └─ molmoact_client.MolmoAct ──POST /act──▶ serve_policy.py
     (3 cams + 14-D state,                    └─ MolmoAct2Policy
      json_numpy over HTTPS)                     └─ /vol/checkpoints/<exp>/stepN
```

Verified end-to-end 2026-07-19: `/act` with the `yam` camera preset returns a
`(30, 14)` action chunk (~1.3 s warm) from
`molmoact2-yam-fold-lora-v2/step500-merged`.

## Prerequisites (one-time)

Same environment as training — if `modal run experiments/modal_train.py` works,
you have everything:

```bash
pip install modal && modal setup
modal secret create huggingface HF_ACCESS_TOKEN=hf_...
```

The image, volume (`molmoact2-yam-fold`), and secrets are shared with
`modal_train.py`; the first training launch pays the image build, serving
reuses it (deploys take seconds).

## 1. Find the checkpoint

```bash
modal run experiments/modal_serve.py::ls
modal run experiments/modal_serve.py::ls --path checkpoints/<exp-name>
```

A trained step looks like `checkpoints/<exp-name>/stepN`. What to serve:

- **LoRA runs** (the default full recipe): serve the **`stepN-merged`** dir.
  The training pipeline writes it alongside `stepN`; if only the raw `stepN` +
  `stepN-lora-llm`/`-lora-vision` adapter dirs exist, merge first:

  ```bash
  modal run experiments/modal_serve.py::merge \
      --base-dir <olmo-format base ckpt, see ::merge docstring> \
      --full-dir checkpoints/<exp-name>/stepN
  ```

  The server refuses to load a raw LoRA step dir (it would silently drop the
  adapters).
- **Action-expert-only / smoke runs**: serve `stepN` directly — sharded and
  unsharded trainer checkpoints both load as-is.

## 2. Deploy

```bash
MODAL_SERVE_CHECKPOINT=<exp-name>/stepN-merged modal deploy experiments/modal_serve.py
```

This prints a stable URL of the form
`https://<workspace>--molmoact2-yam-serve-serve.modal.run`. Redeploying with a
different checkpoint keeps the same URL, so the robot config never changes.

Deploy-time knobs (all env vars, baked into the deployment):

| Env var | Default | Notes |
| --- | --- | --- |
| `MODAL_SERVE_CHECKPOINT` | (required) | Relative to `/vol/checkpoints`, or absolute `/vol/...` path. |
| `MODAL_SERVE_NORM_TAG` | `yam_dual_molmoact2` | Must match the tag the mixture trained with (`yam_fold` reuses the base checkpoint's tag — leave the default). |
| `MODAL_SERVE_GPU_TYPE` | `H100` | The shared image builds flash-attn for Hopper only; try `L4` (~4× cheaper) and fall back if kernels are missing. |
| `MODAL_SERVE_EXTRA_ARGS` | empty | Extra `serve_policy.py` flags, e.g. `--n_action_steps 25 --disable_inference_cuda_graph`. |

## 3. Smoke-test (before touching the robot)

```bash
MODAL_SERVE_CHECKPOINT=<same as deploy> \
    modal run experiments/modal_serve.py::smoke --url https://<...>.modal.run
```

Expect `/health` JSON (check `default_norm_tag` and `n_action_steps`) and
`actions 30x14`. Keep `MODAL_SERVE_CHECKPOINT` set on every `modal run` of
this file — an unconfigured run registers a crash-looping copy of the web
endpoint, which matters if you are using `modal serve` dev mode.

## 4. Point the robot at it

In `examples/yam/configs/yam_left.yaml`:

```yaml
eval:
  mode: server
  molmoact_server: https://<workspace>--molmoact2-yam-serve-serve.modal.run
```

The client appends `/act` itself and speaks the matching `json_numpy` wire
format (`top_cam`/`left_cam`/`right_cam`, `instruction`, 14-D `state`). Note
the client's `normalization_tag` payload field is ignored by this server —
the tag is fixed at deploy time via `MODAL_SERVE_NORM_TAG`.

Then run a session per [`examples/yam/README.md`](examples/yam/README.md)
(camera server in one terminal, eval launcher in the other).

## Operational notes

- **Cold starts.** The container scales to zero after 15 min idle; the next
  request blocks ~2 min while the checkpoint loads. Warm it before a session:
  `curl https://<...>.modal.run/health` and wait for the 200.
- **Cost.** You pay only while a container is up: GPU-hours during sessions
  plus the 15-min idle window after the last request. Nothing accrues while
  scaled to zero.
- **Latency.** ~1–2 s per `/act` call warm; the launcher executes a 25-step
  chunk (~5 s at 5 Hz) per call, so inference overlaps execution.
- **Logs.** `modal app logs molmoact2-yam-serve` streams the server output,
  including the exact `serve_policy.py` command and checkpoint path it loaded.
- **Local fallback.** With a ≥16 GB NVIDIA GPU on the LAN you can skip Modal
  entirely: `eval.mode: local` in the YAM config, or run
  `examples/yam/host_server_yam.py` for HF-format checkpoints. The 8 GB cards
  don't fit the model.
