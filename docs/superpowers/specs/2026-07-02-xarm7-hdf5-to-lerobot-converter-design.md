# xArm7 hdf5 → LeRobot Converter — Design

**Date:** 2026-07-02
**Goal:** Convert the raw xArm7 demo file into a LeRobot dataset so openpi's pi0.5 fine-tuning pipeline can consume it. Step 1 of the pi0.5 → xArm7 fine-tune + deploy project.

## Input data (verified by inspection)

`data/raw/put_the_bowl_on_plate_demo_15hz.hdf5` — format `libero_like_real_world`, env `RealWorldXArm`. 80 demos under `data/demo_0 … data/demo_79`, 23,015 frames total, 15 Hz, episode lengths 207–371. Per demo:

| hdf5 field | shape | semantics |
|---|---|---|
| `actions` | (T, 7) | delta-EE command: Δxyz (m, ±3mm/step), Δrpy (~zero in demos), gripper ±1 |
| `robot_states` | (T, 7) | 7 arm joint angles, **degrees** |
| `obs/gripper_states` | (T, 1) | gripper position ±1 |
| `obs/agentview_rgb_jpeg` | (T,) vlen | base camera, jpeg 256×256 |
| `obs/eye_in_hand_rgb_jpeg` | (T,) vlen | wrist camera, jpeg 256×256 |
| demo attr `task_description` | str | "Put the bowl on the plate" |

**Known data quirks (verified):**
- `obs/ee_pos`/`ee_ori`/`ee_states` are broken — they duplicate `actions[:, :6]` exactly (corr 1.0). Ignore.
- demo attr `success` is `False` on all 80 demos, but every demo ends with `rewards=1`/`dones=1` — flag is stale, data is good. Ignore.
- `states` field is all zeros. Ignore.
- Idle frames: 2.9% overall, no leading idle, ≤19 trailing frames. No filtering needed.

## Approach

Single-file script mirroring `examples/libero/convert_libero_data_to_lerobot.py` (approach A; generic converter and ALOHA-converter adaptation were considered and rejected — YAGNI / poorer structural match).

**File:** `examples/xarm7/convert_xarm_data_to_lerobot.py`, tyro CLI: `--data_path` (default `data/raw/put_the_bowl_on_plate_demo_15hz.hdf5`), `--push_to_hub` (default False; local-only per decision).

## Output schema

`LeRobotDataset.create(repo_id="ehayes/xarm7_put_bowl_on_plate", robot_type="xarm7", fps=15, features=...)`:

| feature | dtype/shape | source |
|---|---|---|
| `image` | image (256,256,3) | decoded `agentview_rgb_jpeg` |
| `wrist_image` | image (256,256,3) | decoded `eye_in_hand_rgb_jpeg` |
| `state` | float32 (8,) | `robot_states` converted deg→rad (7) ++ `gripper_states` (1) |
| `actions` | float32 (7,) | `actions` passthrough |

Key names deliberately match the LIBERO convention so the future `XArm7Inputs` policy transform can mirror `LiberoInputs`. Per-frame `task` string read from the demo's `task_description` attribute (keeps converter reusable for future multi-task data).

Output location: `$HF_LEROBOT_HOME/<repo_id>` (default `~/.cache/huggingface/lerobot/`). Existing output dir removed before conversion (same as LIBERO script).

## Frame loop

1. Open hdf5, iterate demos sorted **numerically** (`demo_0, demo_1, …` — lexicographic sort would give `demo_0, demo_1, demo_10, …`).
2. Per demo: read arrays once; per frame decode jpegs (PIL), assemble state (deg→rad via `np.deg2rad` on joints), `dataset.add_frame({image, wrist_image, state, actions, task})`.
3. `dataset.save_episode()` after each demo.

## Error handling

- Assert per demo: `actions.shape == (T, 7)`, `robot_states.shape == (T, 7)`, `gripper_states.shape == (T, 1)`, image arrays length T.
- Fail loudly (with demo name + frame index) on jpeg decode failure. No silent skips — every demo counts at N=80.

## Verification

After conversion:
1. Reload with `LeRobotDataset("ehayes/xarm7_put_bowl_on_plate")`.
2. Assert 80 episodes, 23,015 frames.
3. Spot-check one frame: state joint values within ±π-ish (radians, not the ±77 degree range), state dim 8, actions dim 7, images 256×256×3, task string correct.

## Out of scope (later steps)

`XArm7Inputs`/`Outputs` policy transforms, `LeRobotXArm7DataConfig` + `pi05_xarm7` TrainConfig, `compute_norm_stats`, training, deployment runtime.
