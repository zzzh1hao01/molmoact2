# SO100 → YAM Data Collection Quickstart

From fresh laptop to uploaded LeRobot dataset. Assumes: a bimanual YAM rig with
two CAN buses + three RealSense cameras, two **pre-calibrated** SO100 leader
arms on USB serial, and Ubuntu with miniconda.

All collection code lives in the `YAM/` submodule; commands below are run from
inside it unless noted. Design/background: `YAM/docs/so100_collection_finetune_plan.md`.
Training-side (Modal) usage: `experiments/modal_train.py` docstring.

## 1. Clone + install (once)

```bash
git clone --recurse-submodules https://github.com/zzzh1hao01/molmoact2.git
cd molmoact2/YAM
./setup_so100.sh          # creates conda env ai2_yam, installs i2rt/gello/lerobot + feetech SDK
conda activate ai2_yam
```

(Collection only needs the `YAM/` submodule; the rest of the repo is the
training/inference side and can be ignored on the rig.)

## 2. Fill in the machine-specific config (once)

All placeholders live in `gello_software/configs/yam_left_so100.yaml` (and the
agent block of `yam_right_so100.yaml`):

| Field | Set to |
|---|---|
| `agent.port` (both yamls) | your SO100 serial devices — `ls /dev/serial/by-id/` |
| `agent.calibration_path` (both) | your existing SO100 calibration JSONs (lerobot-style per-motor files accepted) |
| `robot.channel` | verify against `ip link show \| grep can` (defaults: `can_leader_l` left, `can_follower_r` right) |
| `storage.base_dir` | a real directory on this machine for episode data |
| `storage.task_directory` | e.g. `fold_laundry` |
| `storage.language_instruction` | the actual task instruction, e.g. `"fold the towel in half"` |
| `storage.episodes` | episodes per session |
| `lerobot.hf_repo_id` | the team dataset repo, e.g. `<hf-user>/hackathon` |

Both calibration files referenced by `agent.calibration_path` must exist — the
arms are assumed pre-calibrated; `SO100LeaderAgent` also accepts lerobot-style
per-motor calibration JSONs.

Open decision points to settle on the rig (see the plan doc §1.2/§Open):
the SO100→YAM joint correspondence + pinned wrist joint (`agent` block knobs)
and the gripper mapping.

## 3. Credentials (once)

Get the shared `.env` from a teammate (never via git). For uploads the rig
needs a HuggingFace login:

```bash
hf auth login --token $HF_ACCESS_TOKEN
```

## 4. Per-boot startup (every power-cycle / replug of the arms)

```bash
sh i2rt/scripts/reset_all_can.sh                                                  # CAN up at 1 Mbit/s
python i2rt/i2rt/motor_config_tool/set_timeout.py --channel can_leader_l  --timeout   # 400ms watchdog
python i2rt/i2rt/motor_config_tool/set_timeout.py --channel can_follower_r --timeout
```

Skip this only for later sessions in the same boot. If the linear gripper
drifted on power-cycle, re-zero motor 7 (commands in `YAM/CLAUDE.md`).

## 5. Collect

```bash
cd gello_software
python experiments/launch_yaml_collect_data.py \
    --left_config_path  configs/yam_left_so100.yaml \
    --right_config_path configs/yam_right_so100.yaml
```

During collection, keyboard focus must be on the **color-pad window**:
`s` start episode, `a` save + end, `b` discard + end. Exit normally at the end
of the session — `ctrl+c` skips the convert/upload pipeline.

Before recording, double-check `storage.language_instruction` and
`storage.task_directory` in the yaml still match today's task — stale values
from a previous task silently mislabel the whole session.

## 6. Validate before collecting at scale (first session only)

1. Record 2–3 throwaway episodes.
2. Open-loop replay them — recorded actions drive the followers with no teleop:

   ```bash
   cd gello_software
   python experiments/launch_yaml_replay.py \
       --left_config_path  configs/yam_left_so100.yaml \
       --right_config_path configs/yam_right_so100.yaml
   ```

   The arms must reproduce the demo cleanly. Jitter or drift here means fix
   calibration/smoothing first; the model consumes 30-step action chunks
   open-loop, so replay quality is training quality.
3. Check one converted episode's `meta/info.json`:
   `robot_type: bi_yam_follower`, features `observation.images.{top,left,right}`,
   `(14,)` state/action, `fps: 30`.
4. Then set `lerobot.auto_upload: true` and collect for real. Vary garment
   placement/appearance across episodes; keep demos smooth and unhesitating.

## Post-session checklist

1. Confirm the session's episodes converted (and uploaded, if `auto_upload`).
2. Open-loop replay a random converted episode (command above).
3. Spot-check `meta/info.json` as in step 6.3.

## Troubleshooting

- **Both CAN buses report `loss communication` at once** — a long GIL-holding
  operation ran while motors were live; keep heavy init before motors
  energize (see `YAM/CLAUDE.md` on the 400 ms watchdog).
- **Gripper drifted after power-cycle** — re-zero motor 7 (commands in
  `YAM/CLAUDE.md` startup section).
- **`auto_upload` on but push failed** — check `hf auth whoami`; the token in
  the shared `.env` needs write access to `lerobot.hf_repo_id`.
- **Upload failed / skipped** — run the converter manually from the `YAM/` root:
  `python molmoact_to_lerobot_v30.py --config_path gello_software/configs/yam_left_so100.yaml`
  (individual CLI flags override the config values).
