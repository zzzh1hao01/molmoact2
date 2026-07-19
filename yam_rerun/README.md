# yam_rerun — Rerun tooling for the bimanual YAM

Rerun-centered data pipeline for the bimanual YAM (see `rerun-yam-port-plan.md`
at the repo root). Phase 0 lives here: the dedicated Rerun venv and the dual-arm
URDF spike. Offline API docs for the Rerun 0.34 APIs are in
`.agents/skills/rerun-*` (copied from the so100-hackathon reference).

## Environment: the `.venv-rerun` venv

All Rerun code runs on **rerun-sdk 0.34.1 + datafusion 53** in a dedicated venv
— NOT the repo's main `.venv` (whose torch/transformers pins must not move).
Exact creation commands:

```bash
uv venv /Users/zhihao/molmoact2/.venv-rerun --python 3.11
uv pip install -p /Users/zhihao/molmoact2/.venv-rerun \
    'rerun-sdk[all]==0.34.1' 'datafusion>=53,<54' numpy av
```

(`av` is only needed by `tools/import_lerobot.py` and `tools/view_episode.py`, which
decode mp4 camera streams; everything else runs without it. `view_episode.py
--detect` additionally needs `ultralytics` and the ultralytics CLIP fork:
`uv pip install -p .venv-rerun ultralytics 'git+https://github.com/ultralytics/CLIP.git'`
— run `uv pip install` from OUTSIDE the repo, or the repo's CUDA-only torch
index pins will fail the resolve on macOS.)

Sanity check (should print `0.34.1` and import all three submodules):

```bash
/Users/zhihao/molmoact2/.venv-rerun/bin/python -c \
    "import rerun as rr, rerun.urdf, rerun.server, rerun.catalog; print(rr.__version__)"
```

Run everything in this package with `/Users/zhihao/molmoact2/.venv-rerun/bin/python`.

## URDF spike

- `urdf_yam.py` — loads `YAM/i2rt/i2rt/robot_models/yam/yam.urdf` twice via
  `rr.urdf.UrdfTree.from_file_path(..., entity_path_prefix=..., frame_prefix=...)`
  as `left_arm` / `right_arm`, anchors the two root frames apart along world y
  (static `Transform3D` with `child_frame="<arm>/base_link"`), and applies a
  14-D state vector per timestep via `UrdfJoint.compute_transform`.
- `smoke.py` — animates both arms from a synthetic in-limits sine trajectory
  (default 10 s @ 30 Hz, `rec.set_time("time", timestamp=...)`), logging FK
  transforms plus `left_arm/position` / `right_arm/position` scalars (7-D each,
  the Phase-1 entity names).

```bash
# headless / CI: write a .rrd
/Users/zhihao/molmoact2/.venv-rerun/bin/python -m yam_rerun.smoke --save /tmp/yam_smoke.rrd
# interactive: open the native viewer
/Users/zhihao/molmoact2/.venv-rerun/bin/python -m yam_rerun.smoke --spawn
```

### 14-D state ordering (the contract everything else inherits)

Ground truth: `YAM/molmoact_to_lerobot_v30.py::STATE_DIM_NAMES` (the converter
behind the released `allenai/MolmoAct2-BimanualYAM` datasets,
`norm_tag="yam_dual_molmoact2"`):

| index | dim | URDF joint |
|---|---|---|
| 0–5 | `left_joint1..left_joint6` (rad) | `joint1..joint6` of the `left_arm` tree |
| 6 | `left_gripper` (normalized) | — (no gripper joint in `yam.urdf`) |
| 7–12 | `right_joint1..right_joint6` (rad) | `joint1..joint6` of the `right_arm` tree |
| 13 | `right_gripper` (normalized) | — |

Exposed as `urdf_yam.STATE_DIM_NAMES`, `JOINT_STATE_INDEX`, and
`GRIPPER_STATE_INDEX`. The gripper is a separate linear actuator (i2rt
`GripperType.LINEAR_4310`); FK carries it in the vector but skips it.

### Mesh URI gotcha (`package://assets/*.stl`)

`yam.urdf` references its meshes as `package://assets/*.stl`. Rerun 0.34.1
resolves `package://` URIs by searching `ROS_PACKAGE_PATH` /
`AMENT_PREFIX_PATH`; with neither set, `log_urdf_to_recording` raises
`Failed to resolve package URI`. Fix (implemented in
`urdf_yam._ensure_meshes_resolvable`, called automatically by
`YamArm.create`): append the URDF's own directory to `ROS_PACKAGE_PATH`
in-process, which makes the sibling `assets/` dir resolvable as the "package".
Resolution happens lazily at log time, so setting the env var from Python
before logging is sufficient — no patched copy of the URDF needed.

### Headless verification of a smoke .rrd

Load the file back through the 0.34.1 catalog API (fixed port — `port=0`
fails on 0.34.1):

```python
import os; os.environ.setdefault("RERUN_INSECURE_SKIP_HOST_CHECK", "1")
import rerun as rr
server = rr.server.Server(port=51999, datasets={"yam_smoke": ["/tmp/yam_smoke.rrd"]})
dataset = rr.catalog.CatalogClient("rerun+http://localhost:51999").get_dataset(name="yam_smoke")
df = dataset.filter_contents(["/left_arm/position"]).reader(index="time").to_pandas()
# columns are named "/path:Archetype:field"; 7-D scalar cells come back as
# lists -> np.stack(df["/left_arm/position:Scalars:scalars"]) has shape (300, 7)
```

Verified on this machine: 300 rows for both `<arm>/position:Scalars` and
`<arm>/joint_transforms:Transform3D`, and 14 `visual_geometries` asset paths
(7 links × 2 arms) — meshes really loaded.

## Importing + viewing existing episodes

Two `tools/` CLIs bring pre-existing demonstrations into the same viewer/catalog
pipeline (run both with the `.venv-rerun` interpreter):

- `tools/import_lerobot.py --source <v3 dataset root>` — inverse of
  `tools/export_lerobot.py`: reads a LeRobot v3 dataset directly (parquet + PyAV,
  no `lerobot` import), slices each episode out of the shared per-camera mp4s via
  the `from/to_timestamp` metadata, and writes standard takes to
  `recordings/<dataset>/episode_NNN.rrd` (Phase-1 entity contract + URDF FK), so
  imported episodes are indistinguishable from live-recorded ones. Registers to
  the catalog when the server is up; otherwise the startup rescan picks them up.
- `tools/view_episode.py <raw episode dir>` or `--dataset D --episode E` —
  visualise one episode as an animated dual-URDF + cameras + position/goal plots.
  Raw dirs are the workstation layout (`data.npz` with `state`/`action` (T,14) +
  `t`, one `<camera>.mp4` each for `top`/`wrist_1`/`wrist_2`); catalog episodes
  are fetched through the query API (with an ephemeral in-process
  `rr.server.Server` fallback when `yam_rerun/server.py` isn't running).
  Outputs: `--spawn` (native viewer, default), `--serve` (browser viewer),
  `--save out.rrd`. Optional `--detect "cup, bottle, ..."` runs open-vocabulary
  YOLO-World over the frames and logs `Boxes2D` overlays under
  `camera/<name>/detections` (image-space only — the episodes carry no
  depth/calibration).

### Left/right orientation (verified on real data)

Correlating per-arm joint speed with camera frame-diff energy on `data/ep01`
confirmed the naming contract end-to-end: state dims 0–6 ("left") drive
`wrist_1` (→ `camera/left`) and light up the **left half** of the top image, so
the top camera views from behind the arms (operator perspective) and nothing in
the data is mirrored. The arms reach toward +x with `left_arm` anchored at +y;
the blueprint pins the 3D view's default eye behind the arms
(`EyeControls3D`) to match that perspective, and `log_static` adds floating
`labels/left_arm` / `labels/right_arm` name tags so orientation stays
unambiguous from any orbit angle.
