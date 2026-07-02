# xArm7 runtime inference client

On-robot client that runs a fine-tuned **pi0.5** policy (`pi05_xarm7`) on the
UFactory xArm7 rig. It reads the two cameras + arm state, streams observations to
an openpi policy server over a websocket, and executes the returned action chunks.

openpi splits inference into two processes:

```
 ROBOT PC (examples/xarm7/main.py)          GPU BOX (scripts/serve_policy.py)
 ┌──────────────────────────────┐         ┌──────────────────────────────┐
 │ read cameras + arm state     │  obs ──▶ │ pi05_xarm7 policy on GPU      │
 │ send obs, apply actions      │ ws:8000  │ runs XArm7 transforms +       │
 │ safety / homing / labeling   │ ◀── act  │ normalization, returns chunk  │
 └──────────────────────────────┘         └──────────────────────────────┘
```

- **State** is 8-dim: 7 joint angles (radians) + raw gripper position
  (`[0, 850]`, same units the data converter stored — the server's norm-stats do
  the scaling, so the client does **not** normalize).
- **Actions** are 7-dim Cartesian delta-EE (dx,dy,dz meters; droll,dpitch,dyaw
  rad; gripper).

## Install

On the robot PC (has the arm, cameras, and xArm SDK):

```bash
uv pip install -r examples/xarm7/requirements.txt
uv pip install -e packages/openpi-client
```

The RealSense cameras are used as plain UVC devices via OpenCV — `pyrealsense2`
is **not** required.

## Run

### 1. Start the policy server (GPU box)

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_xarm7 \
    --policy.dir=<path-to-checkpoint>
```

Serves on port 8000 by default.

### 2. Start the client (robot PC)

Point `--remote_host` at the GPU box (use `0.0.0.0` if same machine):

```bash
uv run examples/xarm7/main.py \
    --remote_host=<gpu-box-ip> \
    --prompt="put the carrot in the bowl"
```

During a rollout: press **`1`** for SUCCESS or **`0`** for FAIL to end the run —
the arm halts, re-homes, and an mp4 of the exo view is written to `rollouts/`
with the label in its filename. **Ctrl+C is an emergency stop**: the arm halts
and the process exits immediately with no save.

## Bring-up sequence (do this before trusting a rollout)

1. **Single-step dry run.** Power the arm, keep a hand on the E-stop, and run with
   `--action_horizon 1`. Verify a single small delta executes in the correct
   direction and that state/action shapes look right.
2. **Gripper-units check.** Print live `arm.get_gripper_position()` at fully open
   and fully closed; confirm the range matches the converter's
   `obs/gripper_states`. If they differ, fix `get_proprio` before continuing.
3. **Joint-units check.** Confirm `arm.get_joint_states()[1][0]` returns radians
   (the converter did deg→rad on stored degrees, so live state must be radians).
4. **Full rollout.** Raise `--action_horizon 8` and run a complete rollout; label
   with `1`/`0`; confirm the mp4 is saved.

## Known review items

- `execute_action_cartesian_position` applies only the **translation** deltas and
  drops the rotation deltas (`action[3:6]`), matching the reference rig's
  translation-only execution. Revisit if the fine-tune expects wrist rotation.
- `--z_floor` (default 175 mm) is a collision guard tuned for the carrot-in-bowl
  scene/table. Re-tune if the scene changes.

## Key config flags

| Flag | Default | Notes |
|---|---|---|
| `--robot_ip` | `192.168.1.198` | xArm controller |
| `--exo_cam_id` | `10` | OpenCV device id (scene cam) |
| `--wrist_cam_id` | `4` | OpenCV device id (gripper cam) |
| `--remote_host` | `0.0.0.0` | policy server IP |
| `--remote_port` | `8000` | openpi default |
| `--prompt` | `"put the carrot in the bowl"` | must match fine-tune language |
| `--action_horizon` | `8` | chunk steps before re-query (use 1 for bring-up) |
| `--control_hz` | `5` | matches data-collection rate |
| `--z_floor` | `175` | min end-effector height (mm) |

## Testing

Pure functions are unit-tested (no hardware needed):

```bash
uv run pytest examples/xarm7/main_test.py
uv run ruff check examples/xarm7
```
