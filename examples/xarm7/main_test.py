"""Unit tests for the pure / mockable pieces of the xArm7 runtime client.

The control loop needs real hardware (arm + cameras + xArm SDK), so these tests
cover only the functions that can run in the dev environment: image
cropping/resizing, observation-dict construction, and 8-dim state assembly.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("xarm7_main", pathlib.Path(__file__).parent / "main.py")
main = importlib.util.module_from_spec(_spec)
# Register before exec: @dataclass + `from __future__ import annotations` resolves
# annotations via sys.modules[cls.__module__].
sys.modules["xarm7_main"] = main
_spec.loader.exec_module(main)


def test_crop_and_resize_returns_target_size():
    frame = np.zeros((800, 1280, 3), dtype=np.uint8)
    out = main.crop_and_resize(frame, main.EXO_CROP)
    assert out.shape == (main.CROP_TARGET_SIZE[1], main.CROP_TARGET_SIZE[0], 3)
    assert out.dtype == np.uint8


def test_build_observation_keys_shapes_dtypes():
    exo = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist = np.ones((256, 256, 3), dtype=np.uint8)
    state = np.arange(8, dtype=np.float32)
    obs = main.build_observation(exo, wrist, state, "put the carrot in the bowl")

    assert set(obs) == {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
    assert obs["observation/image"].shape == (256, 256, 3)
    assert obs["observation/image"].dtype == np.uint8
    assert obs["observation/wrist_image"].shape == (256, 256, 3)
    assert obs["observation/state"].shape == (8,)
    assert obs["observation/state"].dtype == np.float32
    assert obs["prompt"] == "put the carrot in the bowl"


class _FakeArm:
    """Minimal stand-in for XArmAPI covering only what get_proprio touches."""

    def __init__(self, joints_rad, gripper_raw):
        self._joints = joints_rad
        self._gripper = gripper_raw

    def get_joint_states(self):
        # xArm SDK returns (code, [positions, velocities, efforts]); positions
        # is a 7-vector in radians.
        return (0, [self._joints, None, None])

    def get_gripper_position(self):
        return (0, self._gripper)


def test_get_proprio_is_8dim_gripper_appended_raw():
    joints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    state = main.get_proprio(_FakeArm(joints, gripper_raw=850))

    assert state.shape == (8,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(state[:7], joints, rtol=1e-6)
    # Raw gripper, NOT normalized by /850.
    assert state[7] == pytest.approx(850.0)


def test_get_proprio_clamps_gripper_to_0_850():
    joints = [0.0] * 7
    assert main.get_proprio(_FakeArm(joints, gripper_raw=-20))[7] == pytest.approx(0.0)
    assert main.get_proprio(_FakeArm(joints, gripper_raw=999))[7] == pytest.approx(850.0)
