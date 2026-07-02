# xArm7 Runtime Client — Design

**Date:** 2026-07-02
**Status:** Approved (design), pending implementation plan
**Related:** pi0.5 xArm7 fine-tune project; reference rig facts in `new_cosmos_xarm_inference.py` (cosmos-policy, same physical rig).

## Purpose

Build the on-robot inference client that runs a fine-tuned pi0.5 policy on the UFactory xArm7 rig. This is the "robot PC" half of openpi's client↔server architecture: it reads cameras + arm state, sends observations to the openpi policy server over websocket, receives action chunks, and drives the arm. Target: run a real robot test of the `pi05_xarm7` fine-tune. **Task: "put the carrot in the bowl"** (50 demos — same task as the original cosmos rig).

**Dataset note / open item:** an earlier conversion produced `ehayes/xarm7_put_bowl_on_plate` (80 episodes). The fine-tune target is carrot-in-bowl (50 demos), so a new/replacement LeRobot dataset for that task is expected. The runtime client is task-agnostic (prompt is a CLI arg); the only requirement is that the `--prompt` default matches the instruction string the model was fine-tuned on.

This spec covers **only the runtime client**. The server-side policy transforms (`XArm7Inputs`/`XArm7Outputs`) and the `pi05_xarm7` `TrainConfig` are a separate, adjacent task; this client only depends on agreeing with them on observation key names and dimensions.

## Context: what already exists

- **Data → LeRobot:** done. `examples/xarm7/convert_xarm_data_to_lerobot.py` produced `ehayes/xarm7_put_bowl_on_plate` (80 episodes). Confirms shapes: `state` = 8-dim (7 joints deg→rad + gripper), `actions` = 7-dim delta-EE (Cartesian deltas), images 256×256×3 uint8.
- **Reference implementation:** `new_cosmos_xarm_inference.py` — a working inference loop for the *same physical rig* using cosmos-policy. It is the authoritative source for hardware facts (SDK calls, robot IP, camera device IDs, crops, control mode, safety). We reuse its hardware layer and replace its policy/comms layer.
- **openpi client library:** `packages/openpi-client` provides `WebsocketClientPolicy`, `ActionChunkBroker`, `image_tools`, and the `Runtime`/`Environment`/`Agent` abstractions.

## Architecture

**Chosen style: flat single-file loop** (mirrors the droid example `examples/droid/main.py` and the existing cosmos script), rather than the modular `Runtime`+`Environment`+`Agent` decomposition (aloha style). Rationale: it matches the reference script the user already understands, keeps the whole control loop readable on one screen, and reduces new abstractions for a user newer to the stack. Chunk handling is still delegated to the library's `ActionChunkBroker`.

Client↔server split:

```
 ROBOT PC (examples/xarm7/main.py)          GPU BOX (scripts/serve_policy.py)
 ┌──────────────────────────────┐         ┌──────────────────────────────┐
 │ read cameras + arm state     │  obs ──▶ │ pi05_xarm7 policy on GPU      │
 │ send obs, apply actions      │ ws:8000  │ runs XArm7 transforms +       │
 │ safety / homing / labeling   │ ◀── act  │ normalization, returns chunk  │
 └──────────────────────────────┘         └──────────────────────────────┘
```

### Files

- `examples/xarm7/main.py` — the runtime client (control loop, hardware I/O, safety, args). New.
- `examples/xarm7/requirements.txt` — client deps (`xArm-Python-SDK`, `opencv-python`, `openpi-client`, `tyro`, `numpy`). New.
- `examples/xarm7/README.md` — how to run (start server, then client). New.

The client is a leaf script; it imports `openpi_client` but not the training/model code.

### Control loop (single tick, ~5 Hz)

```
LOOP:
  if chunk empty: flush stale cam frames (grab ×5)   # V4L2 buffer workaround
  exo   = capture(cam_id=10) → crop[640,640,310,80]  → 256×256 RGB uint8
  wrist = capture(cam_id=4)  → crop[720,720,280,0]   → 256×256 RGB uint8
  state = [7 joint angles (rad), gripper]             # 8-dim  ← adds gripper vs cosmos
                                                       # gripper in SAME units the converter stored (raw obs/gripper_states),
                                                       # NOT cosmos's /850 — server norm-stats do the scaling
  obs = { <image keys>, "observation/state": state, "prompt": task }
  action = broker.infer(obs)      # 7-dim delta-EE; websocket re-query only when chunk empty
  execute_action_cartesian_position(arm, action)     # relative set_position + gripper
  sleep to maintain CONTROL_HZ
```

### Observation schema

Client sends raw, un-normalized observations; the server's `XArm7Inputs` transform does slot-mapping, masking, and normalization. Keys MUST match `XArm7Inputs` (finalized when that transform is written). Planned keys:

| Key | Content | Shape / dtype |
|---|---|---|
| `observation/image` | exocentric (fixed, scene) camera → server maps to `base_0_rgb` | 256×256×3 uint8 |
| `observation/wrist_image` | egocentric (gripper-mounted) camera → server maps to `left_wrist_0_rgb` | 256×256×3 uint8 |
| `observation/state` | 7 joints (rad) + gripper in raw `obs/gripper_states` units (match converter; NOT normalized — server norm-stats scale it) | (8,) float32 |
| `prompt` | task instruction string | e.g. "put the bowl on the plate" |

There is no right-wrist camera; the server transform zero-fills `right_wrist_0_rgb` and masks it False (UR5 pattern). The client sends only the two real cameras.

### Action execution

Server returns 7-dim **delta-EE** actions (dx,dy,dz meters; droll,dpitch,dyaw rad; gripper). Reuse cosmos's `execute_action_cartesian_position`:
- linear ×1000 → mm added to current pose; rotation ×180/π → deg;
- applied via `arm.set_position(..., relative=False, wait=False)` (accumulated onto current pose);
- gripper: >0 → close (0), <0 → open (850).

**Because the recorded actions are already deltas**, the server-side `XArm7` transforms must NOT apply `DeltaActions`/`AbsoluteActions` (those convert absolute datasets like UR5/LIBERO). Noted here as a cross-task constraint; enforced in the transform task.

Task-specific cosmos hacks to review/parameterize, not blindly copy: the hardcoded `speed=100`, and the `z <= 200 → clamp to ≥175mm` collision floor. The floor was tuned for this exact carrot-in-bowl task, so it is likely still applicable — but make it a configurable arg (`--z_floor`, default 175) rather than a magic number, so it can be disabled/tuned.

### Safety (reused from cosmos, unchanged)

- **Emergency stop:** Ctrl+C (SIGINT) → `arm.set_state(4)` + hard exit, no save.
- **Graceful end:** keypress `1` (success) / `0` (fail) → SIGUSR1 → halt + label + save.
- Terminal restored on exit; arm re-homed and gripper disabled in `finally`.
- Auto-home (`home_arm`) to the start pose between tests.

### Configuration (CLI args via tyro/argparse)

Hardware-specific values are args/defaults, never hardcoded logic, so the same file runs on any rig by changing flags:

| Arg | Default (this rig) | Notes |
|---|---|---|
| `robot_ip` | `192.168.1.198` | xArm controller |
| `exo_cam_id` | `10` | OpenCV device id (scene cam) |
| `wrist_cam_id` | `4` | OpenCV device id (gripper cam) |
| `remote_host` | `0.0.0.0` | policy server IP (set to GPU box) |
| `remote_port` | `8000` | openpi default |
| `prompt` | `"put the carrot in the bowl"` | task instruction (must match fine-tune language) |
| `action_horizon` / `open_loop_horizon` | TBD (e.g. 8–16) | chunk steps executed before re-query |
| `control_hz` | `5` | matches data-collection rate |
| `z_floor` | `175` | min end-effector height (mm); cosmos collision guard, tuned for carrot-in-bowl |

### Logging (v1 = Minimal, per decision)

- Save an mp4 of each rollout (primary/exo view).
- Record success/fail via the `1`/`0` keypress label; embed in the mp4 filename.
- **Dropped for v1:** HDF5 episode re-saving, per-frame obs dumping, future-image/value ensembles, T5 embedding cache, local model loading, dataset-stats loading.

## Reuse vs. change (summary)

| Component | Action |
|---|---|
| Camera init / warmup / crop / RGB convert | Reuse |
| `init_arm`, `get_proprio` (+ append gripper → 8-dim) | Reuse + extend |
| `execute_action_cartesian_position`, `home_arm` | Reuse (review z-floor/speed hacks) |
| SIGINT/SIGUSR1 safety, keypress listener, terminal restore | Reuse |
| HTTP `/act` + base64 jpeg comms | **Replace** with `WebsocketClientPolicy` + `ActionChunkBroker` |
| Proprio 7-dim | **Change** to 8-dim (add gripper) |
| Local model load, T5, dataset-stats, ensembles, HDF5 | **Delete** |

## Dependencies

- `xArm-Python-SDK` (`from xarm.wrapper import XArmAPI`)
- `opencv-python` (`cv2.VideoCapture`, crop/resize)
- `openpi-client` (websocket policy, chunk broker, image tools)
- `numpy`, `tyro`
- RealSense cameras are accessed as plain UVC devices via OpenCV — **`pyrealsense2` is NOT required** on this rig.

## Testing / verification

- **Cannot run on hardware in the dev environment** (no arm, no cameras, no SDKs). Real verification happens on the robot PC.
- **Unit-testable in isolation** (dev env): pure functions — `crop_and_resize` (fixed input → expected shape), observation-dict construction (fake frames + fake proprio → correct keys/shapes/dtypes), gripper normalization. Add `examples/xarm7/main_test.py` covering these.
- **On-robot bring-up sequence** (documented in README): (1) start policy server with the `pi05_xarm7` checkpoint; (2) run client with `--action_horizon 1` and arm powered but ready to E-stop, sanity-check a single delta; (3) increase horizon; (4) full rollout with success labeling.

## Open questions / assumptions

- **Obs key names** are provisional until `XArm7Inputs` is written; the two tasks must agree. Assumption: `observation/image`, `observation/wrist_image`, `observation/state`, `prompt`.
- **`action_horizon`** default not yet chosen; start conservative (1) for bring-up, then 8–16.
- **z-floor collision clamp** from cosmos: parameterize as `--z_floor` (default 175, the value tuned for carrot-in-bowl); confirm it still suits the current scene/table height.
- Whether to keep the mp4 rollout in v1 or defer — currently kept (Minimal option).
- **Gripper units:** confirm the raw range/scale of `obs/gripper_states` used by the converter and that `arm.get_gripper_position()` returns the same units; the client must reproduce the converter's units exactly (do NOT copy cosmos's `/850` normalization).

## Out of scope (separate tasks)

- `XArm7Inputs`/`XArm7Outputs` policy transforms + `pi05_xarm7` `TrainConfig` (server side).
- Any retraining / on-robot data collection pipeline.
