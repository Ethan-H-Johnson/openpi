"""Test for the xArm7 hdf5 -> LeRobot converter, using a tiny synthetic hdf5 fixture."""

import importlib.util
import io
import pathlib
import shutil

import h5py
import numpy as np
import PIL.Image
import pytest

_spec = importlib.util.spec_from_file_location(
    "convert_xarm_data_to_lerobot",
    pathlib.Path(__file__).parent / "convert_xarm_data_to_lerobot.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
convert = _mod.convert

TEST_REPO_ID = "local_test/xarm7_convert_test"


def _random_jpeg_bytes(rng: np.random.Generator) -> np.ndarray:
    img = PIL.Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


@pytest.fixture
def fixture_hdf5(tmp_path):
    """2 demos x 4 frames in the same layout as the real RealWorldXArm file."""
    rng = np.random.default_rng(0)
    path = tmp_path / "mini.hdf5"
    jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i in range(2):
            t = 4
            demo = data.create_group(f"demo_{i}")
            demo.attrs["task_description"] = "Put the bowl on the plate"
            demo.create_dataset("actions", data=rng.uniform(-0.003, 0.003, (t, 7)).astype(np.float32))
            demo.create_dataset("robot_states", data=rng.uniform(-90, 90, (t, 7)).astype(np.float32))
            obs = demo.create_group("obs")
            obs.create_dataset("gripper_states", data=rng.choice([-1.0, 1.0], (t, 1)).astype(np.float32))
            for cam in ("agentview_rgb_jpeg", "eye_in_hand_rgb_jpeg"):
                ds = obs.create_dataset(cam, (t,), dtype=jpeg_dtype)
                for j in range(t):
                    ds[j] = _random_jpeg_bytes(rng)
    return path


def test_convert(fixture_hdf5):
    output_path = convert(str(fixture_hdf5), repo_id=TEST_REPO_ID)
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(TEST_REPO_ID)
        assert ds.num_episodes == 2
        assert ds.num_frames == 8

        frame = ds[0]
        assert frame["state"].shape == (8,)
        assert frame["actions"].shape == (7,)
        # joints were degrees in [-90, 90] -> radians in [-pi/2, pi/2]
        assert abs(frame["state"][:7]).max() <= np.pi / 2 + 1e-5
        # gripper passthrough +-1
        assert frame["state"][7] in (-1.0, 1.0)
        assert frame["task"] == "Put the bowl on the plate"
        # image tensor shape is channel-first (lerobot convention on read)
        assert tuple(frame["image"].shape)[-2:] == (256, 256)
    finally:
        shutil.rmtree(output_path, ignore_errors=True)
