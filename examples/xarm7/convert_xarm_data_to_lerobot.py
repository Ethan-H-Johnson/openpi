"""
Convert raw xArm7 demos (RealWorldXArm 'libero_like_real_world' hdf5) to a LeRobot dataset
for pi0.5 fine-tuning.

Usage:
uv run examples/xarm7/convert_xarm_data_to_lerobot.py --data_path data/raw/put_the_bowl_on_plate_demo_15hz.hdf5

The resulting dataset is saved to the $HF_LEROBOT_HOME directory
(default: ~/.cache/huggingface/lerobot). Add --push_to_hub to also upload it.

Design notes (see docs/superpowers/specs/2026-07-02-xarm7-hdf5-to-lerobot-converter-design.md):
- state = 7 joint angles (deg -> rad) + 1 gripper position (8-dim)
- actions = 7-dim delta-EE command, passed through unchanged
- obs/ee_pos, ee_ori, ee_states are ignored: they duplicate the actions (recording bug)
- the per-demo `success` attr is stale (always False) and ignored; rewards/dones confirm success
"""

import io
import pathlib
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import PIL.Image
import tyro

REPO_NAME = "ehayes/xarm7_put_bowl_on_plate"


def _decode_jpeg(buf: np.ndarray, *, demo: str, frame: int, cam: str) -> np.ndarray:
    try:
        img = np.asarray(PIL.Image.open(io.BytesIO(buf.tobytes())))
    except Exception as e:
        raise RuntimeError(f"Failed to decode {cam} jpeg at {demo} frame {frame}") from e
    if img.shape != (256, 256, 3):
        raise ValueError(f"Unexpected image shape {img.shape} for {cam} at {demo} frame {frame}")
    return img


def convert(data_path: str, repo_id: str = REPO_NAME, *, push_to_hub: bool = False) -> pathlib.Path:
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="xarm7",
        fps=15,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    with h5py.File(data_path, "r") as f:
        data = f["data"]
        demo_names = sorted(data.keys(), key=lambda name: int(name.split("_")[1]))
        for demo_name in demo_names:
            demo = data[demo_name]
            actions = demo["actions"][:]
            joints_deg = demo["robot_states"][:]
            gripper = demo["obs/gripper_states"][:]
            agentview = demo["obs/agentview_rgb_jpeg"]
            wrist = demo["obs/eye_in_hand_rgb_jpeg"]
            task = demo.attrs["task_description"]

            num_frames = len(actions)
            if actions.shape != (num_frames, 7):
                raise ValueError(f"{demo_name}: bad actions shape {actions.shape}")
            if joints_deg.shape != (num_frames, 7):
                raise ValueError(f"{demo_name}: bad robot_states shape {joints_deg.shape}")
            if gripper.shape != (num_frames, 1):
                raise ValueError(f"{demo_name}: bad gripper_states shape {gripper.shape}")
            if len(agentview) != num_frames or len(wrist) != num_frames:
                raise ValueError(f"{demo_name}: image count mismatch")

            state = np.concatenate([np.deg2rad(joints_deg), gripper], axis=1).astype(np.float32)

            for i in range(num_frames):
                dataset.add_frame(
                    {
                        "image": _decode_jpeg(agentview[i], demo=demo_name, frame=i, cam="agentview"),
                        "wrist_image": _decode_jpeg(wrist[i], demo=demo_name, frame=i, cam="eye_in_hand"),
                        "state": state[i],
                        "actions": actions[i].astype(np.float32),
                        "task": task,
                    }
                )
            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["xarm7", "pi05"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )

    return output_path


if __name__ == "__main__":
    tyro.cli(convert)
