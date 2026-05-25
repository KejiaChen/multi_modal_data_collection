#!/usr/bin/env python3
"""
Visualize a LeRobot v2.1 dataset episode with Rerun.

Accepts a local root directory or a HuggingFace repo_id.

Usage:
    # local dataset, episode 0 (default)
    python rerun_lerobot.py --root-dir /path/to/dataset

    # local dataset, specific episode
    python rerun_lerobot.py --root-dir /path/to/dataset --episode 3

    # HuggingFace repo
    python rerun_lerobot.py --repo-id kejia98/pnp_microwave_v4 --episode 0

    # list available episodes
    python rerun_lerobot.py --root-dir /path/to/dataset --list
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import rerun as rr
from PIL import Image


# ── image decoding ────────────────────────────────────────────────────────────

def decode_image(cell) -> np.ndarray | None:
    if cell is None:
        return None
    raw = cell.get("bytes") if isinstance(cell, dict) else None
    if not raw:
        return None
    return np.array(Image.open(io.BytesIO(raw)))


# ── dataset resolution ────────────────────────────────────────────────────────

def resolve_root(repo_id: str | None, root_dir: str | None) -> str:
    """Return local root directory, downloading from HF if needed."""
    if root_dir:
        if not os.path.isdir(root_dir):
            sys.exit(f"Error: directory not found: {root_dir}")
        return root_dir

    # repo_id path — snapshot_download caches to ~/.cache/huggingface/hub
    from huggingface_hub import snapshot_download
    print(f"Downloading {repo_id} from HuggingFace Hub …")
    return snapshot_download(repo_id=repo_id, repo_type="dataset")


def load_info(root: str) -> dict:
    path = os.path.join(root, "meta", "info.json")
    if not os.path.exists(path):
        sys.exit(f"Error: meta/info.json not found in {root}")
    with open(path) as f:
        return json.load(f)


def parquet_path(root: str, episode_index: int, chunks_size: int, data_path_template: str) -> str:
    chunk = episode_index // chunks_size
    rel = data_path_template.format(episode_chunk=chunk, episode_index=episode_index)
    return os.path.join(root, rel)


def list_episodes(root: str, info: dict):
    total = info["total_episodes"]
    episodes_file = os.path.join(root, "meta", "episodes.jsonl")
    print(f"Dataset has {total} episode(s).")
    if os.path.exists(episodes_file):
        with open(episodes_file) as f:
            for line in f:
                ep = json.loads(line)
                print(f"  episode {ep['episode_index']:4d}  —  {ep.get('tasks', ['?'])}")
    else:
        for i in range(total):
            print(f"  episode {i:4d}")


# ── rerun logging ─────────────────────────────────────────────────────────────

STATE_NAMES  = ["eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az", "gripper_0", "gripper_1"]
ACTION_NAMES = ["eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az", "gripper"]


def log_episode(df: pd.DataFrame, episode_index: int):
    for i, row in df.iterrows():
        t = float(row.get("timestamp", i / 15.0))
        frame = int(row.get("frame_index", i))

        rr.set_time("time", timestamp=t)
        rr.set_time("frame", sequence=frame)

        img = decode_image(row.get("image"))
        if img is not None:
            rr.log("camera/image", rr.Image(img))

        img2 = decode_image(row.get("image_secondary"))
        if img2 is not None:
            rr.log("camera/image_secondary", rr.Image(img2))

        state = row.get("state")
        if state is not None:
            state = np.asarray(state, dtype=np.float32)
            for j, name in enumerate(STATE_NAMES[: len(state)]):
                rr.log(f"state/{name}", rr.Scalars(float(state[j])))

        action = row.get("actions")
        if action is not None:
            action = np.asarray(action, dtype=np.float32)
            for j, name in enumerate(ACTION_NAMES[: len(action)]):
                rr.log(f"action/{name}", rr.Scalars(float(action[j])))

        if "franka_gripper_width" in row and row["franka_gripper_width"] is not None:
            rr.log("state/gripper_width_raw", rr.Scalars(float(row["franka_gripper_width"])))
        if "franka_gripper_grasp" in row and row["franka_gripper_grasp"] is not None:
            rr.log("state/gripper_grasp", rr.Scalars(float(row["franka_gripper_grasp"])))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize a LeRobot dataset episode with Rerun.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--root-dir", metavar="DIR",  help="Local dataset root directory")
    src.add_argument("--repo-id",  metavar="REPO", help="HuggingFace dataset repo id (user/name)")
    parser.add_argument("--episode", type=int, default=0, metavar="N", help="Episode index (default: 0)")
    parser.add_argument("--list", action="store_true", help="List available episodes and exit")
    args = parser.parse_args()

    root = resolve_root(repo_id=args.repo_id, root_dir=args.root_dir)
    info = load_info(root)

    if args.list:
        list_episodes(root, info)
        return

    total = info["total_episodes"]
    if args.episode >= total:
        sys.exit(f"Error: episode {args.episode} out of range (dataset has {total} episodes: 0–{total-1})")

    pq = parquet_path(root, args.episode, info["chunks_size"], info["data_path"])
    if not os.path.exists(pq):
        sys.exit(f"Error: parquet not found: {pq}")

    df = pd.read_parquet(pq)
    print(f"Loaded episode {args.episode}: {len(df)} frames  ({info['fps']} fps)")

    rr.init(f"lerobot/{os.path.basename(root)}/episode_{args.episode:06d}", spawn=True)
    log_episode(df, args.episode)
    print("Done — Rerun viewer launched.")


if __name__ == "__main__":
    main()
