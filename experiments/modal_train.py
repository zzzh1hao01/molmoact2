"""Modal fine-tuning app for MolmoAct2-BimanualYAM on the ``yam_fold`` mixture.

Phase 4 of YAM/docs/so100_collection_finetune_plan.md: LoRA fine-tune
``allenai/MolmoAct2-BimanualYAM`` (LoRA adapters on the VLM path + fully
trained action expert) on the SO100-teleop laundry-folding dataset registered
as mixture ``yam_fold`` in ``launch_scripts/data_mixtures.py``.

One-time setup::

    pip install modal && modal setup
    modal secret create huggingface HF_ACCESS_TOKEN=hf_...   # private dataset pull
    modal secret create wandb WANDB_API_KEY=...              # dummy value is fine if you
                                                             # run without --wandb-entity
                                                             # (falls back to WANDB_MODE=offline)

First run — smoke test (Phase 3: 1 GPU, 20 steps, README "Smoke Test" recipe
with ``--packing=false --dynamic_seq_len=true``)::

    MODAL_TRAIN_GPUS=1 modal run experiments/modal_train.py --smoke

Full LoRA run (default H100:8)::

    modal run experiments/modal_train.py \
        --exp-name molmoact2-yam-fold-lora \
        --steps 50000 \
        --dataset-repo-id <your-hf-user>/hackathon \
        --wandb-entity <entity> --wandb-project <project>

Resume after a timeout/preemption (checkpoints persist on the Modal Volume)::

    modal run experiments/modal_train.py --exp-name molmoact2-yam-fold-lora \
        --resume-from /vol/checkpoints/molmoact2-yam-fold-lora/step10000

Notes:
  - GPU count/type are baked into the Modal function at import time; override
    with ``MODAL_TRAIN_GPUS=<n>`` (default 8) and ``MODAL_TRAIN_GPU_TYPE``
    (default H100) in the environment of ``modal run``.
  - ``--dataset-repo-id`` overrides the ``YAM_FOLD_REPO_ID`` placeholder in
    ``launch_scripts/data_mixtures.py`` via the env var of the same name.
  - Keep ``--global-batch-size 64`` and trade ``--device-batch-size`` against
    GPU count (the trainer gradient-accumulates the rest, e.g.
    64 = 8 GPUs x device_batch_size 2 x 4 accumulation steps).
  - This module does no network/GPU work at import time; everything runs
    inside ``modal run`` / the remote function.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Configuration (import-time constants only; no side effects).
# ---------------------------------------------------------------------------

GPU_TYPE = os.environ.get("MODAL_TRAIN_GPU_TYPE", "H100")
N_GPUS = int(os.environ.get("MODAL_TRAIN_GPUS", "8"))

HF_SECRET_NAME = "huggingface"  # must contain HF_ACCESS_TOKEN
WANDB_SECRET_NAME = "wandb"     # must contain WANDB_API_KEY

VOLUME_NAME = "molmoact2-yam-fold"
VOLUME_MOUNT = "/vol"
HF_HOME = f"{VOLUME_MOUNT}/hf"
LEROBOT_DATA_ROOT = f"{VOLUME_MOUNT}/lerobot_data"
CHECKPOINT_ROOT = f"{VOLUME_MOUNT}/checkpoints"

LOCAL_EXPERIMENTS_DIR = Path(__file__).resolve().parent  # this experiments/ dir
REMOTE_EXPERIMENTS_DIR = "/root/molmoact2/experiments"

DEFAULT_CHECKPOINT = "allenai/MolmoAct2-BimanualYAM"
MIXTURE = "yam_fold"

DEFAULT_STEPS = 50_000
SMOKE_STEPS = 20
VOLUME_COMMIT_INTERVAL_S = 15 * 60

app = modal.App("molmoact2-yam-fold-train")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Image: faithful translation of experiments/Dockerfile (build stage) into
# modal.Image calls — CUDA 12.8 devel base, cu128 torch 2.10 stack, then
# editable installs of experiments[all] + experiments/lerobot[async].
# ---------------------------------------------------------------------------

CUDA_BASE = "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"

image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .apt_install(
        "build-essential",
        "clang",  # Modal's add_python interpreter was built with clang; its
                  # sysconfig emits clang++ link commands (flash-attn link step)
        "cmake",
        "curl",
        "git",
        "pkg-config",
        "wget",
        "ffmpeg",           # video decode for LeRobot datasets (pyav/torchcodec)
        "libgl1",
        "libegl1",
        "libglib2.0-0",
        "libosmesa6",
        "libjpeg-dev",
        "libpng-dev",
        "libxml2-dev",
    )
    # Build prerequisites (mirrors the Dockerfile's pip/wheel/setuptools pins).
    .pip_install("wheel", "packaging", "setuptools<70.0.0", "ninja")
    # PyTorch core ecosystem, pinned as in experiments/Dockerfile.
    .pip_install(
        "torch==2.10.0",
        "torchao==0.16.0",
        "torchvision",
        "torchaudio",
        index_url=TORCH_INDEX_URL,
    )
    .pip_install("torchcodec==0.10.0")
    .run_commands(
        # flash-attn 2 source build (no prebuilt wheel for this stack). Slow on
        # the first image build (~30 min on Modal's builders), cached afterwards.
        "TORCH_CUDA_ARCH_LIST='9.0' FLASH_ATTN_CUDA_ARCHS='90' MAX_JOBS=8 "
        "pip install --no-build-isolation --no-cache-dir flash-attn==2.8.3",
        "pip install --no-build-isolation --no-cache-dir liger-kernel==0.6.4",
    )
    # NOTE: experiments/Dockerfile additionally installs grouped_gemm (MoE),
    # flash-attn 3 (Hopper source build), flash-attn-4, ring-flash-attn
    # (context parallelism), flash-linear-attention, causal-conv1d, vllm, and
    # molmo-utils[torchcodec]. They are omitted here to keep image builds
    # tractable; TODO: verify at the first real launch that the yam_fold LoRA
    # recipe imports none of them, and add any that turn out to be required.
    .add_local_dir(
        str(LOCAL_EXPERIMENTS_DIR),
        REMOTE_EXPERIMENTS_DIR,
        copy=True,
        ignore=[
            "**/.git",
            "**/__pycache__",
            "**/*.pyc",
            "**/*.egg-info",
            "**/.venv",
            "checkpoints/**",
            "wandb/**",
        ],
    )
    .run_commands(
        # Same editable installs as the README setup section (async extra only;
        # libero/hardware extras are not needed on Modal).
        f"pip install --no-cache-dir -e '{REMOTE_EXPERIMENTS_DIR}[all]'",
        f"pip install --no-cache-dir -e '{REMOTE_EXPERIMENTS_DIR}/lerobot[async]'",
    )
    .env(
        {
            "HF_HOME": HF_HOME,
            "LEROBOT_DATA_ROOT": LEROBOT_DATA_ROOT,
            "LEROBOT_VIDEO_BACKEND": "pyav",
            "PYTHONPATH": f"{REMOTE_EXPERIMENTS_DIR}:{REMOTE_EXPERIMENTS_DIR}/lerobot/src",
        }
    )
)

# ---------------------------------------------------------------------------
# Train command construction (pure; also used by the local entrypoint to echo
# the exact command).
# ---------------------------------------------------------------------------


def _build_train_command(
    *,
    nproc: int,
    checkpoint: str,
    exp_name: str,
    steps: int,
    device_batch_size: int,
    global_batch_size: int,
    save_interval: int,
    save_folder: str,
    wandb_entity: str,
    wandb_project: str,
    smoke: bool,
    resume_from: str,
) -> list[str]:
    """Build the torchrun command per the experiments/README.md recipes.

    ``smoke=True`` follows the "Smoke Test" recipe; otherwise the "LoRA
    Fine-Tuning" recipe (LoRA on the VLM path, fully trained action expert).
    All flag names are taken verbatim from README.md / train_lerobot.py.
    """
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={nproc}",
        "launch_scripts/train_lerobot.py",
        checkpoint,
        MIXTURE,
        f"--wandb.name={exp_name}",
        f"--wandb.entity={wandb_entity}",
        f"--wandb.project={wandb_project}",
        f"--max_duration={steps}",
        f"--save_folder={save_folder}",
        "--packing=false",
        "--dynamic_seq_len=true",
    ]
    if smoke:
        cmd += [
            "--device_batch_size=1",
            "--global_batch_size=1",
            "--num_workers=0",
            "--pin_memory=false",
            "--ft_vlm=false",
            "--ft_action_expert=true",
            "--ft_embedding=none",
        ]
    else:
        cmd += [
            f"--device_batch_size={device_batch_size}",
            f"--global_batch_size={global_batch_size}",
            "--num_workers=4",
            "--pin_memory=true",
            "--data.timeout=900",
            # README uses --save_interval=10000; default lowered to 1000 here so
            # a preempted/timed-out run loses at most ~1k steps (checkpoints go
            # to the persistent Volume).
            f"--save_interval={save_interval}",
            "--save_num_checkpoints_to_keep=20",
            "--ft_vlm=true",
            "--ft_action_expert=true",
            "--ft_embedding=lm_head",
            "--lora_enable=true",
            "--lora_rank=64",
            "--llm_learning_rate=5e-5",
            "--vit_learning_rate=5e-5",
            "--connector_learning_rate=5e-5",
            "--action_expert_learning_rate=5e-5",
        ]
    if resume_from:
        # OmegaConf dotlist override of TrainConfig.load_path (train_lerobot.py
        # sets load_path=None; extra --key=value args merge into the config).
        # The trainer config also sets allow_resume=True, so relaunching with
        # the same --exp-name / save_folder should pick up the latest
        # checkpoint automatically; --load_path is the explicit path.
        # TODO: verify on the first resume whether load_path expects the run's
        # save_folder or a specific stepN subdirectory.
        cmd.append(f"--load_path={resume_from}")
    return cmd


# ---------------------------------------------------------------------------
# Remote training function.
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    volumes={VOLUME_MOUNT: volume},
    secrets=[
        modal.Secret.from_name(HF_SECRET_NAME),
        modal.Secret.from_name(WANDB_SECRET_NAME),
    ],
    timeout=24 * 60 * 60,  # Modal's maximum; resume via --resume-from after that
)
def train(
    exp_name: str,
    steps: int,
    checkpoint: str,
    dataset_repo_id: str,
    device_batch_size: int,
    global_batch_size: int,
    save_interval: int,
    resume_from: str,
    wandb_entity: str,
    wandb_project: str,
    smoke: bool,
    nproc: int,
) -> str:
    env = os.environ.copy()

    # huggingface_hub reads HF_TOKEN; the README recipes export HF_ACCESS_TOKEN.
    token = env.get("HF_ACCESS_TOKEN", "")
    if token and not env.get("HF_TOKEN"):
        env["HF_TOKEN"] = token

    # Point the yam_fold mixture at the real dataset (see YAM_FOLD_REPO_ID in
    # launch_scripts/data_mixtures.py).
    if dataset_repo_id:
        env["YAM_FOLD_REPO_ID"] = dataset_repo_id

    # W&B config resolves entity/project from the dotlist overrides below; if
    # none are given, log offline so no valid WANDB_API_KEY is required.
    if not (wandb_entity and wandb_project):
        env["WANDB_MODE"] = "offline"
        wandb_entity = wandb_entity or "offline"
        wandb_project = wandb_project or "molmoact2-yam-fold"

    for path in (HF_HOME, LEROBOT_DATA_ROOT, CHECKPOINT_ROOT):
        os.makedirs(path, exist_ok=True)

    save_folder = f"{CHECKPOINT_ROOT}/{exp_name}"
    cmd = _build_train_command(
        nproc=nproc,
        checkpoint=checkpoint,
        exp_name=exp_name,
        steps=steps,
        device_batch_size=device_batch_size,
        global_batch_size=global_batch_size,
        save_interval=save_interval,
        save_folder=save_folder,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        smoke=smoke,
        resume_from=resume_from,
    )
    print(f"[modal_train] launching in {REMOTE_EXPERIMENTS_DIR}:\n  " + " ".join(cmd))

    proc = subprocess.Popen(cmd, cwd=REMOTE_EXPERIMENTS_DIR, env=env)
    last_commit = time.monotonic()
    try:
        while proc.poll() is None:
            time.sleep(30)
            # Periodically persist HF cache, dataset cache, and checkpoints so
            # a preemption or the 24 h timeout loses at most one interval.
            if time.monotonic() - last_commit >= VOLUME_COMMIT_INTERVAL_S:
                volume.commit()
                last_commit = time.monotonic()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
        volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"train_lerobot.py exited with code {proc.returncode}")
    print(f"[modal_train] done; checkpoints in {save_folder} on volume '{VOLUME_NAME}'")
    return save_folder


# ---------------------------------------------------------------------------
# Local entrypoint.
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    smoke: bool = False,
    exp_name: str = "",
    steps: int = 0,
    checkpoint: str = DEFAULT_CHECKPOINT,
    dataset_repo_id: str = "",
    device_batch_size: int = 2,
    global_batch_size: int = 64,
    save_interval: int = 1000,
    resume_from: str = "",
    wandb_entity: str = "",
    wandb_project: str = "",
    gpus: int = 0,
):
    """Launch fine-tuning (or a --smoke run) on Modal. See the module docstring."""
    nproc = 1 if smoke else N_GPUS
    if gpus:
        if not smoke and gpus != N_GPUS:
            raise SystemExit(
                f"--gpus={gpus} but this app was imported with {GPU_TYPE}:{N_GPUS}. "
                f"GPU count is baked in at import time; relaunch as "
                f"MODAL_TRAIN_GPUS={gpus} modal run experiments/modal_train.py ..."
            )
        nproc = min(gpus, N_GPUS)
    if smoke and N_GPUS > 1:
        print(
            f"[modal_train] note: smoke test uses 1 torchrun process but the app "
            f"reserves {GPU_TYPE}:{N_GPUS}; use MODAL_TRAIN_GPUS=1 to reserve one GPU."
        )
    if not exp_name:
        exp_name = "molmoact2-yam-fold-smoke" if smoke else "molmoact2-yam-fold-lora"
    if not steps:
        steps = SMOKE_STEPS if smoke else DEFAULT_STEPS

    # Echo the exact command that will run remotely.
    preview = _build_train_command(
        nproc=nproc,
        checkpoint=checkpoint,
        exp_name=exp_name,
        steps=steps,
        device_batch_size=device_batch_size,
        global_batch_size=global_batch_size,
        save_interval=save_interval,
        save_folder=f"{CHECKPOINT_ROOT}/{exp_name}",
        wandb_entity=wandb_entity or "offline",
        wandb_project=wandb_project or "molmoact2-yam-fold",
        smoke=smoke,
        resume_from=resume_from,
    )
    print(f"[modal_train] {GPU_TYPE}:{N_GPUS}, will run:\n  " + " ".join(preview))

    save_folder = train.remote(
        exp_name=exp_name,
        steps=steps,
        checkpoint=checkpoint,
        dataset_repo_id=dataset_repo_id,
        device_batch_size=device_batch_size,
        global_batch_size=global_batch_size,
        save_interval=save_interval,
        resume_from=resume_from,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        smoke=smoke,
        nproc=nproc,
    )
    print(f"[modal_train] finished; checkpoints at {save_folder} on volume '{VOLUME_NAME}'")
