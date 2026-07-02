"""On-robot inference client for the xArm7 rig running a fine-tuned pi0.5 policy.

This is the "robot PC" half of openpi's client/server split: it reads the two
cameras + arm state, streams observations to the policy server over a websocket,
receives action chunks, and drives the arm. The GPU-side policy server is
`scripts/serve_policy.py` (see README for the bring-up sequence).

Hardware behavior (SDK calls, crops, control rate, safety) is ported from the
old cosmos-policy inference script for the same physical rig; only the
policy/comms layer is swapped for openpi's WebsocketClientPolicy +
ActionChunkBroker. Actions are 7-dim Cartesian delta-EE; state is 8-dim
(7 joints in radians + raw gripper position).
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import select
import signal
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy
from PIL import Image
import tyro

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    # xArm SDK is only present on the robot PC. The pure functions (imaging,
    # observation building, state assembly) remain importable/testable without it.
    XArmAPI = None

# Collection-time crop settings (must match the data converter). Format is
# [width, height, x_offset, y_offset]; frames are cropped, resized to 480x480,
# then to the 256x256 the model was trained on.
WRISTCAM_CROP = [720, 720, 280, 0]
EXO_CROP = [640, 640, 310, 80]
CROP_TARGET_SIZE = (480, 480)
CAPTURE_FRAME_SIZE = (1280, 800)
WARMUP_FRAMES = 60

# Home pose (cartesian, degrees): x, y, z (mm), roll, pitch, yaw (deg).
STARTING_POSITION = (300, 0, 300, 180, 0, 0)
TCP_OFFSET = [0, 0, 0, 0, 0, 0]
GRIPPER_OPEN = 850
GRIPPER_CLOSED = 0


@dataclasses.dataclass
class Args:
    # Hardware.
    robot_ip: str = "192.168.1.198"
    exo_cam_id: int = 10  # OpenCV device id for the exocentric (scene) camera.
    wrist_cam_id: int = 4  # OpenCV device id for the wrist (gripper) camera.

    # Policy server.
    remote_host: str = "0.0.0.0"  # IP of the GPU box running serve_policy.py.
    remote_port: int = 8000  # openpi default server port.

    # Rollout.
    prompt: str = "put the carrot in the bowl"  # Must match the fine-tune language.
    # Number of actions executed from a predicted chunk before re-querying the
    # server. Use 1 for bring-up dry-runs, 8 for real rollouts.
    action_horizon: int = 8
    control_hz: float = 5.0  # Matches data-collection rate.
    z_floor: float = 175.0  # Min end-effector height (mm); collision guard.

    # Where to write the rollout mp4.
    out_dir: str = "rollouts"


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
def crop_and_resize(frame: np.ndarray, crop: list[int]) -> np.ndarray:
    w, h, x, y = crop
    cropped = frame[y : y + h, x : x + w]
    return cv2.resize(cropped, CROP_TARGET_SIZE, interpolation=cv2.INTER_LINEAR)


def init_cameras(exo_id: int, wrist_id: int) -> tuple[cv2.VideoCapture, cv2.VideoCapture]:
    exo_cap = cv2.VideoCapture(exo_id)
    wrist_cap = cv2.VideoCapture(wrist_id)
    for cap in (exo_cap, wrist_cap):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_FRAME_SIZE[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return exo_cap, wrist_cap


def warmup_cameras(exo_cap: cv2.VideoCapture, wrist_cap: cv2.VideoCapture, num_frames: int = WARMUP_FRAMES) -> None:
    for _ in range(num_frames):
        exo_cap.read()
        wrist_cap.read()


def flush_stale_frames(exo_cap: cv2.VideoCapture, wrist_cap: cv2.VideoCapture, num_frames: int = 5) -> None:
    """V4L2 ignores CAP_PROP_BUFFERSIZE=1. After a multi-second server call the
    buffer holds stale frames; drain them so the read that feeds the model is
    fresh."""
    for _ in range(num_frames):
        exo_cap.grab()
        wrist_cap.grab()


def read_observation_images(exo_cap: cv2.VideoCapture, wrist_cap: cv2.VideoCapture) -> tuple[np.ndarray, np.ndarray]:
    """Return (exo_rgb, wrist_rgb) as 256x256x3 uint8 RGB, matching the converter."""
    ret1, exo = exo_cap.read()
    ret2, wrist = wrist_cap.read()
    if not ret1 or not ret2:
        raise RuntimeError("Camera read failed")
    exo = crop_and_resize(exo, EXO_CROP)
    wrist = crop_and_resize(wrist, WRISTCAM_CROP)
    exo = cv2.cvtColor(exo, cv2.COLOR_BGR2RGB)
    wrist = cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)
    exo = np.asarray(Image.fromarray(exo).resize((256, 256), resample=Image.NEAREST), dtype=np.uint8)
    wrist = np.asarray(Image.fromarray(wrist).resize((256, 256), resample=Image.NEAREST), dtype=np.uint8)
    return exo, wrist


# ---------------------------------------------------------------------------
# Arm
# ---------------------------------------------------------------------------
def init_arm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip)
    arm.motion_enable(enable=True)
    arm.set_tcp_offset(TCP_OFFSET)
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_gripper_mode(0)
    arm.set_gripper_enable(enable=True)
    arm.set_gripper_speed(5000)
    _home_position(arm, wait=False)
    return arm


def _home_position(arm: XArmAPI, *, wait: bool) -> None:
    x, y, z, roll, pitch, yaw = STARTING_POSITION
    arm.set_position(
        x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, is_radian=False, wait=wait, relative=False, speed=100
    )
    arm.set_gripper_position(GRIPPER_OPEN, wait=wait)


def home_arm(arm: XArmAPI) -> None:
    """Return the arm to STARTING_POSITION and open the gripper."""
    arm.set_mode(0)
    arm.set_state(0)
    _home_position(arm, wait=True)


def get_proprio(arm: XArmAPI) -> np.ndarray:
    """8-dim state: 7 joint angles (radians) + raw gripper position.

    The gripper is sent in the SAME raw units the converter stored
    (obs/gripper_states, clamped [0, 850]) -- do NOT normalize by /850; the
    server's norm-stats handle scaling.
    """
    joints = arm.get_joint_states()[1][0]  # 7 values, radians.
    gripper = max(0, min(arm.get_gripper_position()[1], GRIPPER_OPEN))
    return np.array([*joints[:7], gripper], dtype=np.float32)


def execute_action_cartesian_position(arm: XArmAPI, action: np.ndarray, *, z_floor: float = 175.0) -> None:
    """Apply a 7-dim delta-EE action: linear deltas (m) accumulate onto the
    current cartesian pose; gripper: >0 close, <0 open.

    NOTE (ported from cosmos, bring-up review item): the rotation deltas
    (action[3:6]) are intentionally NOT applied here -- the reference rig ran
    translation-only execution. Revisit if the pi0.5 fine-tune expects rotation.
    """
    pose = arm.get_position(is_radian=False)[1]  # [x, y, z, roll, pitch, yaw] mm/deg.

    lin_mm = np.asarray(action[:3], dtype=float) * 1000.0
    pose[0] += lin_mm[0]
    pose[1] += lin_mm[1]
    pose[2] += lin_mm[2]

    # Collision guard: near the table, nudge down slightly but never below z_floor.
    if pose[2] <= 200:
        pose[2] = max(pose[2] - 5, z_floor)

    arm.set_position(
        x=pose[0], y=pose[1], z=pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5],
        relative=False, wait=False, speed=100,
    )

    gripper = float(action[6])
    if gripper > 0:
        arm.set_gripper_position(GRIPPER_CLOSED, wait=False)
    elif gripper < 0:
        arm.set_gripper_position(GRIPPER_OPEN, wait=False)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
def build_observation(exo_rgb: np.ndarray, wrist_rgb: np.ndarray, state: np.ndarray, prompt: str) -> dict:
    # TODO: reconcile keys with the XArm7Inputs transform (provisional names).
    return {
        "observation/image": exo_rgb,
        "observation/wrist_image": wrist_rgb,
        "observation/state": state,
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Safety / labeling
# ---------------------------------------------------------------------------
class TestEndKeyListener:
    """Background stdin watcher: '1' = success, '0' = fail. On either key it
    records the verdict and sends SIGUSR1 so the main process halts the arm and
    breaks the loop gracefully. SIGINT (Ctrl+C) is reserved for emergency stop.
    Restores terminal on exit; no-ops if stdin isn't a tty."""

    def __init__(self) -> None:
        self.result = {"decided": False, "success": False}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._old_settings = None

    def __enter__(self) -> TestEndKeyListener:
        if sys.stdin.isatty():
            try:
                self._fd = sys.stdin.fileno()
                self._old_settings = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
            except (termios.error, OSError):
                self._fd = None
                self._old_settings = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._fd is not None and self._old_settings is not None:
            with contextlib.suppress(termios.error, OSError):
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        return False

    def _loop(self) -> None:
        if self._fd is None:
            return
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([self._fd], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not r:
                continue
            try:
                ch = sys.stdin.read(1)
            except (OSError, ValueError):
                return
            if ch in ("0", "1"):
                self.result["decided"] = True
                self.result["success"] = ch == "1"
                os.kill(os.getpid(), signal.SIGUSR1)
                return


def write_rollout_mp4(frames: list[np.ndarray], out_dir: str, fps: float, *, success: bool) -> str | None:
    """Write accumulated exo RGB frames to an mp4, embedding the success label."""
    if not frames:
        return None
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"rollout_{timestamp}_success={success}.mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main(args: Args) -> None:
    policy = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    agent = action_chunk_broker.ActionChunkBroker(policy, action_horizon=args.action_horizon)

    arm = init_arm(args.robot_ip)
    exo_cap, wrist_cap = init_cameras(args.exo_cam_id, args.wrist_cam_id)
    warmup_cameras(exo_cap, wrist_cap)

    orig_term_settings = None
    if sys.stdin.isatty():
        try:
            orig_term_settings = termios.tcgetattr(sys.stdin.fileno())
        except (termios.error, OSError):
            orig_term_settings = None

    def _restore_terminal() -> None:
        if orig_term_settings is not None:
            with contextlib.suppress(termios.error, OSError):
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, orig_term_settings)

    # Ctrl+C = emergency stop: halt arm and hard-exit, no save. SIGUSR1 = normal
    # end-of-test from the keypress listener: halt arm, unwind loop to label/save.
    def _sigint_handler(signum, frame) -> None:
        with contextlib.suppress(Exception):
            arm.set_state(4)
        _restore_terminal()
        os._exit(1)

    def _sigusr1_handler(signum, frame) -> None:
        with contextlib.suppress(Exception):
            arm.set_state(4)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGUSR1, _sigusr1_handler)

    print("Running rollout. Press '1' for SUCCESS, '0' for FAIL. Ctrl+C is EMERGENCY STOP.")
    frames: list[np.ndarray] = []
    step = 0

    with TestEndKeyListener() as listener:
        try:
            while True:
                tick_start = time.time()

                # Flush stale camera frames only right before a fresh server query
                # (the broker re-queries every action_horizon steps).
                if step % args.action_horizon == 0:
                    flush_stale_frames(exo_cap, wrist_cap)

                exo, wrist = read_observation_images(exo_cap, wrist_cap)
                state = get_proprio(arm)
                obs = build_observation(exo, wrist, state, args.prompt)

                action = agent.infer(obs)["actions"]  # 7-dim delta-EE.
                execute_action_cartesian_position(arm, action, z_floor=args.z_floor)

                frames.append(exo)
                step += 1

                # Hold the control rate, subtracting the work already done this tick.
                remaining = (1.0 / args.control_hz) - (time.time() - tick_start)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            pass

    with contextlib.suppress(Exception):
        arm.set_state(4)

    success = listener.result["success"] if listener.result["decided"] else False
    path = write_rollout_mp4(frames, args.out_dir, args.control_hz, success=success)
    if path:
        print(f"Saved rollout ({step} steps, success={success}) to {path}")

    _restore_terminal()
    try:
        arm.set_state(4)
        arm.set_mode(0)
        arm.set_state(0)
        home_arm(arm)
        arm.set_gripper_enable(enable=False)
        arm.disconnect()
    except Exception as e:
        print(f"Cleanup error: {e}")
    exo_cap.release()
    wrist_cap.release()


if __name__ == "__main__":
    main(tyro.cli(Args))
