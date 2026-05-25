#!/usr/bin/env python3
"""
Visualize a LeRobot v2.1 parquet episode using Rerun.

Usage:
    conda run -n egoasis python visualize_lerobot_parquet.py /path/to/episode_000004.parquet
"""

import sys
import io
import numpy as np
import pandas as pd
from PIL import Image
import rerun as rr


def decode_image(img_cell) -> np.ndarray | None:
    """Decode image from LeRobot parquet cell (dict with 'bytes' key)."""
    if img_cell is None:
        return None
    raw = img_cell.get("bytes") if isinstance(img_cell, dict) else None
    if raw is None:
        return None
    return np.array(Image.open(io.BytesIO(raw)))


def main(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    n = len(df)
    fps = 15.0  # matches info.json

    rr.init("lerobot_episode", spawn=True)

    has_image     = "image"           in df.columns
    has_image2    = "image_secondary" in df.columns
    has_state     = "state"           in df.columns
    has_actions   = "actions"         in df.columns
    has_gw        = "franka_gripper_width" in df.columns
    has_gg        = "franka_gripper_grasp" in df.columns

    state_names  = ["eef_x","eef_y","eef_z","eef_ax","eef_ay","eef_az","gripper_0","gripper_1"]
    action_names = ["eef_x","eef_y","eef_z","eef_ax","eef_ay","eef_az","gripper"]

    for i, row in df.iterrows():
        t = float(row["timestamp"]) if "timestamp" in row else i / fps
        rr.set_time("time", timestamp=t)
        rr.set_time("frame", sequence=int(row.get("frame_index", i)))

        if has_image:
            img = decode_image(row["image"])
            if img is not None:
                rr.log("camera/image", rr.Image(img))

        if has_image2:
            img2 = decode_image(row["image_secondary"])
            if img2 is not None:
                rr.log("camera/image_secondary", rr.Image(img2))

        if has_state:
            state = np.asarray(row["state"], dtype=np.float32)
            for j, name in enumerate(state_names[:len(state)]):
                rr.log(f"state/{name}", rr.Scalars(float(state[j])))

        if has_actions:
            action = np.asarray(row["actions"], dtype=np.float32)
            for j, name in enumerate(action_names[:len(action)]):
                rr.log(f"action/{name}", rr.Scalars(float(action[j])))

        if has_gw:
            rr.log("state/gripper_width_raw", rr.Scalars(float(row["franka_gripper_width"])))
        if has_gg:
            rr.log("state/gripper_grasp", rr.Scalars(float(row["franka_gripper_grasp"])))

    print(f"Done — logged {n} frames from {parquet_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_lerobot_parquet.py /path/to/episode_XXXXXX.parquet")
        sys.exit(1)
    main(sys.argv[1])
