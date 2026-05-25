#!/usr/bin/env python3
"""Helpers for writing local LeRobot v2.1-compatible datasets."""

import json
import math
import os
from collections import OrderedDict

import cv2
import numpy as np


def _require_lerobot_export_deps():
    try:
        from datasets import Dataset, Features, Image, Sequence, Value  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot export requires the 'datasets' package and its parquet backend. "
            "Install them with: pip install datasets pyarrow"
        ) from exc


class LeRobotV21DatasetWriter:
    """Write episode-based parquet datasets with LeRobot v2.1 metadata."""

    IMAGE_KEY_MAP = OrderedDict(
        [
            ("rgb", "image"),
            ("rgb2", "image_secondary"),
            ("tactile", "tactile_image"),
            ("lucid_rgb", "lucid_image"),
        ]
    )

    PRIMARY_STATE_CANDIDATES = (
        ("pose", "observation.state"),
        ("ultimate_pose", "observation.state"),
    )

    EXTRA_VECTOR_KEY_MAP = OrderedDict(
        [
            ("pose", "observation.state.pose"),
            ("ultimate_pose", "observation.state.ultimate_pose"),
        ]
    )

    FRANKA_STATE_NAMES = [
        "eef_x",
        "eef_y",
        "eef_z",
        "eef_ax",
        "eef_ay",
        "eef_az",
        "gripper_0",
        "gripper_1",
    ]

    FRANKA_ACTION_NAMES = [
        "eef_x",
        "eef_y",
        "eef_z",
        "eef_ax",
        "eef_ay",
        "eef_az",
        "gripper",
    ]

    INT64_KEYS = {"frame_index", "episode_index", "index", "task_index"}

    EXCLUDED_EXPORT_KEYS = {
        "t_ref",
        "t",
        "rgb_t",
        "rgb_dt",
        "rgb2_t",
        "rgb2_dt",
        "tactile_t",
        "tactile_dt",
        "pose_t",
        "pose_dt",
        "ultimate_pose_t",
        "ultimate_pose_dt",
        "franka_ee_pose_t",
        "franka_ee_pose_dt",
        "franka_ee_pose_cmd_t",
        "franka_ee_pose_cmd_dt",
        "franka_gripper_width_t",
        "franka_gripper_width_dt",
        "franka_gripper_grasp_t",
        "franka_gripper_grasp_dt",
        "franka_gripper_state_t",
        "franka_gripper_state_dt",
        "franka_gripper_cmd_t",
        "franka_gripper_cmd_dt",
        "lucid_rgb_t",
        "lucid_rgb_dt",
    }

    def __init__(
        self,
        dataset_root,
        fps,
        robot_type="custom",
        default_task="default_task",
        chunk_size=1000,
    ):
        _require_lerobot_export_deps()
        self.dataset_root = os.path.abspath(dataset_root)
        self.meta_dir = os.path.join(self.dataset_root, "meta")
        self.data_dir = os.path.join(self.dataset_root, "data")
        self.fps = int(round(float(fps))) if fps else 1
        self.robot_type = str(robot_type)
        self.default_task = str(default_task)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    @staticmethod
    def _vector_feature_info_from_names(dtype, names):
        return {
            "dtype": dtype,
            "shape": [len(names)],
            "names": list(names),
        }

    @staticmethod
    def _is_homogeneous_transform(matrix):
        bottom = np.asarray(matrix[3, :], dtype=np.float64)
        return np.allclose(bottom, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-3)

    def _flat_transform_to_matrix(self, value):
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        if flat.size != 16:
            raise ValueError(f"Expected 16 values for a 4x4 transform, got {flat.size}")

        row_major = flat.reshape(4, 4)
        if self._is_homogeneous_transform(row_major):
            return row_major

        column_major = flat.reshape(4, 4, order="F")
        if self._is_homogeneous_transform(column_major):
            return column_major

        raise ValueError("Input does not look like a valid homogeneous transform matrix")

    @staticmethod
    def _quaternion_to_rotation_matrix(quaternion):
        quat = np.asarray(quaternion, dtype=np.float64).reshape(-1)
        if quat.size != 4:
            raise ValueError(f"Expected quaternion with 4 values, got {quat.size}")

        norm = np.linalg.norm(quat)
        if norm <= 0.0:
            raise ValueError("Quaternion norm must be positive")

        x, y, z, w = quat / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _coerce_gripper_values(value, dual_gripper=False):
        gripper = np.asarray(value, dtype=np.float32).reshape(-1)
        target_size = 2 if dual_gripper else 1

        if gripper.size == 0:
            return np.zeros(target_size, dtype=np.float32)
        if dual_gripper:
            if gripper.size == 1:
                return np.repeat(gripper, 2).astype(np.float32)
            return gripper[:2].astype(np.float32)
        return gripper[:1].astype(np.float32)

    def _robot_pose_to_pose_vector(self, value, gripper_value=None, dual_gripper=False):
        pose = np.asarray(value, dtype=np.float64).reshape(-1)
        if pose.size == 16:
            matrix = self._flat_transform_to_matrix(pose)
            position = matrix[:3, 3].astype(np.float32)
            rotation = matrix[:3, :3].astype(np.float64)
        elif pose.size == 7:
            position = pose[:3].astype(np.float32)
            rotation = self._quaternion_to_rotation_matrix(pose[3:])
        else:
            raise ValueError(
                f"Expected a 4x4 transform (16 values) or pose+quaternion (7 values), got {pose.size}"
            )

        rotvec, _ = cv2.Rodrigues(rotation)
        rotvec = rotvec.reshape(3).astype(np.float32)

        if gripper_value is None:
            gripper = np.zeros(2 if dual_gripper else 1, dtype=np.float32)
        else:
            gripper = self._coerce_gripper_values(gripper_value, dual_gripper=dual_gripper)
        return np.concatenate([position, rotvec, gripper], dtype=np.float32)

    @staticmethod
    def _is_string_array(arr):
        if arr.ndim != 1:
            return False
        if arr.dtype.kind in {"U", "S"}:
            return True
        if arr.dtype.kind != "O":
            return False
        return all(isinstance(v, (str, bytes, np.str_, np.bytes_)) for v in arr.tolist())

    def save_episode(self, arrays, episode_id, task_name=None):
        from datasets import Dataset, Features, Image, Sequence, Value

        episode_index = self._next_episode_index()
        task_name = str(task_name or self.default_task)
        task_index = self._ensure_task(task_name)
        rows, feature_specs, info_features = self._build_rows_and_features(
            arrays, episode_index, task_index
        )
        dataset = Dataset.from_list(rows, features=Features(feature_specs))

        episode_chunk = episode_index // self.chunk_size
        chunk_dir = os.path.join(self.data_dir, f"chunk-{episode_chunk:03d}")
        os.makedirs(chunk_dir, exist_ok=True)
        parquet_path = os.path.join(chunk_dir, f"episode_{episode_index:06d}.parquet")
        dataset.to_parquet(parquet_path)

        self._append_jsonl(
            os.path.join(self.meta_dir, "episodes.jsonl"),
            {
                "episode_index": episode_index,
                "tasks": [task_name],
                "length": len(rows),
                "episode_id": str(episode_id),
            },
        )
        self._append_jsonl(
            os.path.join(self.meta_dir, "episodes_stats.jsonl"),
            {
                "episode_index": episode_index,
                "stats": self._compute_episode_stats(arrays, rows),
            },
        )
        self._write_info_json(info_features)
        self._write_readme()

        return episode_index, parquet_path

    def _build_rows_and_features(self, arrays, episode_index, task_index):
        from datasets import Image, Sequence, Value

        n_frames = int(len(arrays.get("t_ref", [])))
        if n_frames <= 0:
            raise ValueError("Cannot export an empty episode to LeRobot format")

        rows = [{} for _ in range(n_frames)]
        feature_specs = OrderedDict()
        info_features = OrderedDict()

        image_columns = OrderedDict()
        for source_key, column_name in self.IMAGE_KEY_MAP.items():
            if source_key in arrays:
                image_columns[column_name] = arrays[source_key]
                first = np.asarray(arrays[source_key][0])
                info_features[column_name] = {
                    "dtype": "image",
                    "shape": [int(v) for v in first.shape],
                    "names": ["height", "width", "channel"],
                }
                feature_specs[column_name] = Image()

        franka_state_array = arrays.get("franka_ee_pose")
        franka_action_array = arrays.get("franka_ee_pose_cmd")
        # Prefer legacy dedicated gripper keys when present, but let the new
        # boolean grasp signal populate both state and action as a fallback.
        franka_gripper_state_array = arrays.get("franka_gripper_state")
        if franka_gripper_state_array is None:
            franka_gripper_state_array = arrays.get("franka_gripper_grasp")
        franka_gripper_cmd_array = arrays.get("franka_gripper_cmd")
        if franka_gripper_cmd_array is None:
            franka_gripper_cmd_array = arrays.get("franka_gripper_grasp")
        use_franka_state = franka_state_array is not None and len(franka_state_array) == n_frames
        use_franka_action = franka_action_array is not None and len(franka_action_array) == n_frames
        use_franka_gripper_state = (
            franka_gripper_state_array is not None and len(franka_gripper_state_array) == n_frames
        )
        use_franka_gripper_cmd = (
            franka_gripper_cmd_array is not None and len(franka_gripper_cmd_array) == n_frames
        )

        if use_franka_state:
            feature_specs["state"] = Sequence(Value("float32"), length=len(self.FRANKA_STATE_NAMES))
            info_features["state"] = self._vector_feature_info_from_names(
                "float32", self.FRANKA_STATE_NAMES
            )

        if use_franka_action:
            feature_specs["actions"] = Sequence(Value("float32"), length=len(self.FRANKA_ACTION_NAMES))
            info_features["actions"] = self._vector_feature_info_from_names(
                "float32", self.FRANKA_ACTION_NAMES
            )

        primary_state_key = None
        if not use_franka_state:
            for source_key, column_name in self.PRIMARY_STATE_CANDIDATES:
                if source_key in arrays:
                    primary_state_key = source_key
                    feature_specs[column_name] = Sequence(
                        Value("float32"), length=int(np.asarray(arrays[source_key][0]).size)
                    )
                    info_features[column_name] = self._vector_feature_info(arrays[source_key], "float32")
                    break

        for source_key, column_name in self.EXTRA_VECTOR_KEY_MAP.items():
            if source_key not in arrays:
                continue
            if source_key == primary_state_key:
                continue
            feature_specs[column_name] = Sequence(
                Value("float32"), length=int(np.asarray(arrays[source_key][0]).size)
            )
            info_features[column_name] = self._vector_feature_info(arrays[source_key], "float32")

        for key, value in arrays.items():
            if key in self.IMAGE_KEY_MAP:
                continue
            if key == primary_state_key:
                continue
            if key in self.EXTRA_VECTOR_KEY_MAP:
                continue
            if key in {
                "franka_ee_pose",
                "franka_ee_pose_cmd",
                "franka_gripper_state",
                "franka_gripper_cmd",
            }:
                continue
            if key in self.EXCLUDED_EXPORT_KEYS:
                continue
            arr = np.asarray(value)
            if arr.ndim == 1:
                if self._is_string_array(arr):
                    dtype = "string"
                else:
                    dtype = "int64" if key in self.INT64_KEYS else "float32"
                feature_specs[key] = Value(dtype)
                info_features[key] = {"dtype": dtype, "shape": [1], "names": None}
            elif arr.ndim == 2:
                feature_specs[key] = Sequence(Value("float32"), length=int(arr.shape[1]))
                info_features[key] = self._vector_feature_info(arr, "float32")

        timestamps = np.asarray(arrays.get("t_ref", []), dtype=np.float64)
        time_zero = float(timestamps[0])
        frame_index = np.arange(n_frames, dtype=np.int64)
        index_offset = self._next_global_index()

        feature_specs["timestamp"] = Value("float32")
        info_features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
        feature_specs["frame_index"] = Value("int64")
        info_features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
        feature_specs["episode_index"] = Value("int64")
        info_features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
        feature_specs["index"] = Value("int64")
        info_features["index"] = {"dtype": "int64", "shape": [1], "names": None}
        feature_specs["task_index"] = Value("int64")
        info_features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}

        for i in range(n_frames):
            row = rows[i]
            for column_name, image_array in image_columns.items():
                row[column_name] = np.asarray(image_array[i], dtype=np.uint8)

            if use_franka_state:
                row["state"] = self._robot_pose_to_pose_vector(
                    franka_state_array[i],
                    gripper_value=(
                        franka_gripper_state_array[i] if use_franka_gripper_state else None
                    ),
                    dual_gripper=True,
                ).tolist()
            elif primary_state_key is not None:
                row["observation.state"] = (
                    np.asarray(arrays[primary_state_key][i], dtype=np.float32).reshape(-1).tolist()
                )

            if use_franka_action:
                row["actions"] = self._robot_pose_to_pose_vector(
                    franka_action_array[i],
                    gripper_value=(
                        franka_gripper_cmd_array[i] if use_franka_gripper_cmd else None
                    ),
                    dual_gripper=False,
                ).tolist()

            for source_key, column_name in self.EXTRA_VECTOR_KEY_MAP.items():
                if source_key not in arrays or source_key == primary_state_key:
                    continue
                row[column_name] = (
                    np.asarray(arrays[source_key][i], dtype=np.float32).reshape(-1).tolist()
                )

            for key, value in arrays.items():
                if key in self.IMAGE_KEY_MAP:
                    continue
                if key == primary_state_key:
                    continue
                if key in self.EXTRA_VECTOR_KEY_MAP:
                    continue
                if key in {
                    "franka_ee_pose",
                    "franka_ee_pose_cmd",
                    "franka_gripper_state",
                    "franka_gripper_cmd",
                }:
                    continue
                if key in self.EXCLUDED_EXPORT_KEYS:
                    continue
                arr = np.asarray(value)
                if arr.ndim == 1:
                    if self._is_string_array(arr):
                        row[key] = str(arr[i])
                    elif key in self.INT64_KEYS:
                        row[key] = int(arr[i])
                    else:
                        row[key] = float(arr[i])
                elif arr.ndim == 2:
                    row[key] = np.asarray(arr[i], dtype=np.float32).reshape(-1).tolist()

            row["timestamp"] = float(timestamps[i] - time_zero)
            row["frame_index"] = int(frame_index[i])
            row["episode_index"] = int(episode_index)
            row["index"] = int(index_offset + i)
            row["task_index"] = int(task_index)

        return rows, feature_specs, info_features

    def _vector_feature_info(self, arr, dtype):
        first = np.asarray(arr[0]).reshape(-1)
        return {
            "dtype": dtype,
            "shape": [int(first.size)],
            "names": [f"dim_{i}" for i in range(int(first.size))],
        }

    def _write_info_json(self, current_features):
        info_path = os.path.join(self.meta_dir, "info.json")
        info = {}
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)

        episodes = self._load_jsonl(os.path.join(self.meta_dir, "episodes.jsonl"))
        tasks = self._load_jsonl(os.path.join(self.meta_dir, "tasks.jsonl"))
        total_frames = sum(int(ep["length"]) for ep in episodes)
        total_episodes = len(episodes)
        total_tasks = len(tasks)

        merged_features = OrderedDict(info.get("features", {}))
        merged_features.update(current_features)

        info.update(
            {
                "codebase_version": "v2.1",
                "robot_type": self.robot_type,
                "total_episodes": total_episodes,
                "total_frames": total_frames,
                "total_tasks": total_tasks,
                "total_videos": 0,
                "total_chunks": max(1, int(math.ceil(total_episodes / float(self.chunk_size)))),
                "chunks_size": self.chunk_size,
                "fps": self.fps,
                "splits": {"train": f"0:{total_episodes}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": merged_features,
            }
        )

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4)
            f.write("\n")

    def _write_readme(self):
        readme_path = os.path.join(self.dataset_root, "README.md")
        info_path = os.path.join(self.meta_dir, "info.json")
        if not os.path.exists(info_path):
            return
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        body = (
            "---\n"
            "license: apache-2.0\n"
            "task_categories:\n"
            "- robotics\n"
            "tags:\n"
            "- LeRobot\n"
            f"- {self.robot_type}\n"
            "configs:\n"
            "- config_name: default\n"
            "  data_files: data/*/*.parquet\n"
            "---\n\n"
            "This dataset was created using the local multimodal recorder exporter.\n\n"
            "## Dataset Structure\n\n"
            "`meta/info.json`:\n\n"
            "```json\n"
            f"{json.dumps(info, indent=4)}\n"
            "```\n"
        )
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(body)

    def _ensure_task(self, task_name):
        tasks_path = os.path.join(self.meta_dir, "tasks.jsonl")
        tasks = self._load_jsonl(tasks_path)
        for task in tasks:
            if task.get("task") == task_name:
                return int(task["task_index"])

        task_index = len(tasks)
        self._append_jsonl(tasks_path, {"task_index": task_index, "task": task_name})
        return task_index

    def _next_episode_index(self):
        return len(self._load_jsonl(os.path.join(self.meta_dir, "episodes.jsonl")))

    def _next_global_index(self):
        episodes = self._load_jsonl(os.path.join(self.meta_dir, "episodes.jsonl"))
        return sum(int(ep["length"]) for ep in episodes)

    def _compute_episode_stats(self, arrays, rows):
        stats = OrderedDict()

        for source_key, column_name in self.IMAGE_KEY_MAP.items():
            if source_key not in arrays:
                continue
            sample_count = min(100, len(arrays[source_key]))
            sample = np.stack(
                [np.asarray(arrays[source_key][i], dtype=np.float32) / 255.0 for i in range(sample_count)],
                axis=0,
            )
            channel_axes = (0, 1, 2)
            stats[column_name] = {
                "min": [[[float(v)]] for v in sample.min(axis=channel_axes)],
                "max": [[[float(v)]] for v in sample.max(axis=channel_axes)],
                "mean": [[[float(v)]] for v in sample.mean(axis=channel_axes)],
                "std": [[[float(v)]] for v in sample.std(axis=channel_axes)],
                "count": [sample_count],
            }

        image_feature_names = set(self.IMAGE_KEY_MAP.values())
        for key, value in rows[0].items():
            if key in image_feature_names:
                continue
            column_values = [row[key] for row in rows]
            first = column_values[0]
            if isinstance(first, list):
                try:
                    arr = np.asarray(column_values, dtype=np.float32)
                except (TypeError, ValueError):
                    continue
                stats[key] = {
                    "min": arr.min(axis=0).tolist(),
                    "max": arr.max(axis=0).tolist(),
                    "mean": arr.mean(axis=0).tolist(),
                    "std": arr.std(axis=0).tolist(),
                    "count": [int(arr.shape[0])],
                }
            else:
                arr = np.asarray(column_values)
                # String metadata columns are valid exports, but they do not
                # have meaningful numeric summary statistics.
                if arr.ndim == 1 and self._is_string_array(arr):
                    continue
                try:
                    numeric = arr.astype(np.float64)
                except (TypeError, ValueError):
                    continue
                value_type = int if key in self.INT64_KEYS else float
                stats[key] = {
                    "min": [value_type(numeric.min())],
                    "max": [value_type(numeric.max())],
                    "mean": [float(numeric.mean())],
                    "std": [float(numeric.std())],
                    "count": [int(arr.shape[0])],
                }

        return stats

    def _append_jsonl(self, path, payload):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload))
            f.write("\n")

    def _load_jsonl(self, path):
        if not os.path.exists(path):
            return []
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
