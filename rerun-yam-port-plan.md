# Plan: Rerun-centered data pipeline for the bimanual YAM

Port of the so100-hackathon reference loop (collect → catalog → curate → export →
deploy) onto the YAM stack, targeting all three Rerun hackathon prizes. Reference
clone: `scratchpad/so100-hackathon` (rerun-sdk 0.34.1, datafusion 53).

**Where code lives:** new top-level dir `yam_rerun/` in this repo (molmoact2), with
`yam_rerun/takes.py`, `yam_rerun/server.py`, `yam_rerun/blueprint.py`,
`yam_rerun/urdf_yam.py`, plus `tools/` CLIs mirroring the reference layout.
Robot-side hooks patch into `YAM/gello_software/experiments/` launchers.

**Machine split:** collection/deploy code runs on the robot workstation (linux,
CAN, conda `ai2_yam`, py3.11); catalog/curation/export can run anywhere the
`recordings/` dir lives. Dev/smoke-testing on this mac with fake data.

---

## Phase 0 — Environment + URDF spike (half day)

1. Env — **two envs, one file format** (verified by the query-API research agent:
   a 0.26-written `.rrd` loads cleanly into a 0.34.1 catalog server):
   - Our lerobot 0.4.4 pins `rerun-sdk >=0.24,<0.27`; the currently installed
     0.26.2 is **broken** (missing `rerun_bindings`) and needs a force-reinstall.
   - Catalog/query/export tooling gets its own uv venv with
     `rerun-sdk[all]==0.34.1` + `datafusion>=53,<54` — the old `rr.dataframe`
     API was removed in 0.29 and the docs describe only the new
     `rr.server.Server` / `rr.catalog.CatalogClient` API (terminology:
     partition → segment).
   - **Decision (settled by the viewer research agent): all Rerun code runs on
     0.34.1.** Recording needs it too — `rr.urdf.UrdfTree` requires ≥0.29 and
     catalog registration ≥~0.32, so the 0.26-recording option is out. If pip
     conflicts inside `ai2_yam` bite, run the Rerun logging as a standalone
     bridge process in its own venv (our robot client is hand-rolled HTTP, not
     lerobot, so nothing forces them to share an interpreter).
   - Known traps (verified by execution): `Server(port=0)` fails — use a fixed
     port; columns are named `/path:Archetype:field`; 14-D scalar cells come
     back as lists needing `np.stack`; raw image buffers arrive flattened and
     need reshape via the `Image:format` column (moot if we log JPEG
     `EncodedImage`).
2. Copy Rerun's agent skills from the reference (`.agents/skills/rerun-*`) into
   this repo so future sessions have the API docs offline.
3. Spike `yam_rerun/urdf_yam.py`: load `YAM/i2rt/i2rt/robot_models/yam/yam.urdf`
   twice via `rr.urdf.UrdfTree.from_file_path(..., entity_path_prefix="left_arm",
   frame_prefix="left_arm/")`, anchor root frames, animate both arms from a synthetic
   14-D sine trajectory. Deliverable: `python -m yam_rerun.smoke` spawns a viewer
   with two moving YAMs. Resolve joint-name → 14-D index mapping here, once.
   Note: the YAM URDF references meshes as `package://assets/*.stl` — the
   importer may need those URIs rewritten to relative paths.

## Phase 1 — Collect: takes wired into the gello loop (1 day)

1. Vendor `takes.py` from the reference (near copy): take = RecordingStream with
   `set_sinks(GrpcSink, FileSink)`, properties via
   `send_property("episode", AnyValues(dataset=…, task=…, tag=…))`, `rrd optimize`
   on stop, catalog registration, edits layer, `next_episode` numbering.
   `APP_ID = "yam"`. Recordings land in `recordings/<dataset>/<episode>.rrd`.
2. Vendor the server trio (`yam_rerun/server.py`): gRPC proxy :9876, in-process
   catalog (`rr.server.Server`) :51234, minimal control API (:8001 — 8000 is
   taken by the DROID server convention). Startup rescan of `recordings/`.
3. Hook `launch_yaml_collect_data.py`: inside the existing collection loop, log
   per tick with `rec.set_time("time", timestamp=…)`:
   - `camera/top|left|right` → `rr.EncodedImage` (JPEG bytes, reuse the frames
     already fetched from the ZMQ camera server; do NOT re-encode raw)
   - `left_arm/position`, `right_arm/position` → `rr.Scalars` (7 each)
   - `left_arm/goal`, `right_arm/goal` → gello leader commands (the *action*)
   - URDF joint transforms via `UrdfJoint.compute_transform`
   - optional: i2rt motor telemetry (temp/current) for a diagnostics tab
   Take start/stop tied to the episode start/stop the launcher already has.
   Keep the existing JSON/h5 write path ON in parallel until Phase 3 parity.
4. `yam_rerun/blueprint.py`: 3D dual-URDF view + vertical stack of 3 cams on top;
   tabs below with left/right position-vs-goal TimeSeriesViews (sliding
   `VisibleTimeRange`, cursor-relative −10 s) + diagnostics grid. Registered as the
   dataset default blueprint on the catalog.

**Checkpoint:** record a real teleop episode, scrub it in the viewer, see it in
the catalog with task/tag columns.

## Phase 2 — Refine: queries + curation (1 day) → query-API prize

1. Port `query_dataset.py` (mostly copy; entity paths differ). List datasets,
   per-episode table (task/tag/duration/size), tag filter via DataFusion
   `col("property:episode:tag")[0] == lit(…)`, entity series → pandas.
2. New `tools/episode_metrics.py` — the query-prize centerpiece. Per episode, from
   `reader(index="time")` output compute: joint-space path length, jerk (3rd
   derivative RMS), idle fraction (‖Δq‖ < ε), gripper toggle count, goal-vs-position
   tracking error, duration; write scores back as an **edits-layer property**
   (auto-tag suggestions: "Needs review" for outliers). Curation becomes: run
   metrics → review flagged episodes in the viewer → retag.
3. Demo-vs-rollout comparison query: same metrics over the eval dataset
   (Phase 4) vs the teleop dataset, aligned with `using_index_values` +
   `fill_latest_at` resampling — one table showing where the policy diverges.

## Phase 3 — Export to LeRobot v3 (1 day)

1. `tools/export_lerobot.py`: per episode
   `dataset.filter_segments(id).filter_contents([…14 streams + 3 cams…])
   .reader(index="time", fill_latest_at=True).to_pandas()`; stack left+right into
   14-D `state`/`action`; drop incomplete leading rows; stage npy + JPEGs; spawn
   the writer inside the lerobot env (submodule) to produce v3 + video.
2. **No unit conversion** (unlike SO-100): YAM is radians end-to-end and
   normalization lives in the checkpoint's `norm_stats.json`
   (`norm_tag="yam_dual_molmoact2"`). The 14-D ordering and camera-key contract
   must exactly match `YAM/molmoact_to_lerobot_v30.py` — treat it as ground truth.
3. **Parity gate:** convert one episode through both paths (old h5→v3 converter vs
   catalog export) and diff state/action arrays + frame counts before trusting the
   new path. Then the JSON/h5 write path can be turned off.

## Phase 4 — Deploy: replay + rollouts into the catalog (1 day)

1. `tools/replay_episode.py`: query `<arm>/goal` trajectories from the catalog,
   drive both arms through `RobotEnv.step_command_only` with the existing
   interpolation/ramp; stream to the live proxy.
2. Patch `launch_yaml_eval_molmoact.py`: each rollout is a take (dataset
   `molmoact2_eval`, task = instruction, tag "Needs review"; the y/n label at
   episode end rewrites the tag). Log the **predicted action chunk** when it
   arrives (before execution) alongside executed goals — this is the eval-debug
   payoff. Rollouts land next to teleop data; good ones are exportable.
3. Replace/augment the cv2 3-pane view with the Rerun live proxy (a separate
   process — also removes the viewer-freezes-during-inference problem).

## Phase 5 — Prize polish (1–2 days) → viewer prize

1. **Live inference dashboard ("glass cockpit"):** FK the predicted 14-D chunk
   through the URDF and log it as ghost EE trajectories (`rr.LineStrips3D`) **at
   future timestamps**, then give the 3D view a `VisibleTimeRange` extending past
   the cursor (`TimeRangeBoundary.cursor_relative(seconds=+…)`) — predictions
   render ahead of execution, and scrubbing replays past predictions vs. what
   actually happened. Predicted-vs-executed overlay in the time-series tabs;
   `dt_ms` / chunk-latency series. Blueprint saved as the eval dataset default.
2. **MolmoAct2 reasoning X-ray:** `host_server_yam.py` already passes
   `enable_depth_reasoning=False` — the hook exists. Flip it on, verify what
   `predict_action` exposes on the GPU box, and log depth tokens as
   `rr.DepthImage`/TensorView and visual-trace waypoints as 2D overlays on the
   camera views — model internals no other team can show.
3. Browser demo for judges: `rr.serve_grpc()` + `rr.serve_web_viewer()`
   (`rr.serve_web` was removed by 0.34.1). Stretch: click-to-command via
   `rerun.notebook.Viewer.on_event` (callbacks exist only in notebook/JS/Rust
   embeds, not the native desktop viewer); Rust egui panel is explicitly
   unstable-API territory — demo-only if attempted.
4. Demo script + README section: the five-step loop reproduced on the YAM, one
   command per step.

---

## Prize map — which work buys which prize

**Shared foundation (prerequisite for all three, wins nothing by itself):**
Phase 0 (env + URDF spike) and Phase 1 steps 1–3 (takes, server trio, collection
hook). Build once, first.

**$1,000 — end-to-end port (judged on completeness + reproducibility):**
the *minimal* version of every stage — Phase 1 as-is; Phase 2 step 1 only
(`query_dataset` port: list/filter/retag); Phase 3 (export + parity gate);
Phase 4 steps 1–2 (replay-from-catalog, rollouts recorded as episodes);
Phase 5 step 4 (README/demo script: one command per step of the five-step loop).
No fancy metrics or dashboards required — a working, reproducible loop is the
whole rubric.

**$2,000 — query API (judged on queries doing real work):**
Phase 2 steps 2–3 (metrics scoreboard: jerk/idle/gripper-toggle/tracking-error
via DataFusion aggregates; auto-tag via edits layer; demo-vs-rollout comparison
on a shared resampled grid) plus the query-native exporter framing of Phase 3
(export set defined by `filter_segments(query predicate)`, alignment by
`reader(fill_latest_at)` — same module powers metrics and export). Deliverable
to show judges: `tools/episode_metrics.py` + a before/after curation story.

**$2,000 — viewer (judged on the viewer being central, not decorative):**
Phase 4 step 2's predicted-chunk logging (the data feed) + step 3 (Rerun
replaces the cv2 viewer) and all of Phase 5 steps 1–3: the glass cockpit
(future-timestamp ghost predictions + `cursor_relative` window), the MolmoAct2
depth/trace X-ray, per-dataset default blueprints, browser demo via
`serve_web_viewer`. Deliverable to show judges: a live rollout in the cockpit +
scrubbing a past rollout's prediction-vs-reality.

**Dual-counted work (do these regardless of prize focus):** Phase 3's exporter
(port completeness *and* query showcase) and Phase 4's rollout recording (port
loop-closer *and* the viewer dashboard's data source).

**If time runs short, cut in this order:** Rust/interactivity stretch goals →
X-ray (needs GPU-box verification) → demo-vs-rollout comparison → replay. The
port prize's minimal loop is the last thing to sacrifice — it's the only prize
that requires *everything* to work at least a little.

---

## Risks / open questions

- **rerun-sdk 0.34 catalog API is young** — pin exactly 0.34.1 + datafusion 53
  like the reference; their code encodes known footguns (property stamps must
  always send the full column set; blueprint files must not be rewritten while
  registered).
- **.rrd size**: 3 RealSense streams at 30 Hz — always log `EncodedImage` JPEG
  (reference does the same), use `ChunkBatcherConfig.LOW_LATENCY()` on takes,
  proxy memory limit for the live stream.
- **GIL vs 250 Hz CAN thread**: Rerun logging is cheap (native), but do take
  setup/optimize/registration *outside* motor-live sections, same rule as the
  model-load caveat in YAM/CLAUDE.md.
- **State ordering contract** (14-D order, gripper convention, camera names) —
  fixed once in Phase 0/3 against `molmoact_to_lerobot_v30.py` and
  `norm_stats.json`; everything else is mechanical.
- Hardware access: Phases 1/4 need the robot workstation; everything else can be
  developed on the mac against recorded .rrd files.
