#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Multimodal Data Recorder (Humble) — with per-modality timestamps and dt.

Compared to the original multi_sensor_data_collection.py, this version:
- Uses the latest RealSense RGB frame as reference time t_ref.
- For each modality, stores:
    - its own timestamp (e.g., rgb_t, pose_t, tactile_t, ...)
    - its time offset to t_ref (e.g., rgb_dt = rgb_t - t_ref).
So you can later inspect synchronization accuracy between modalities.
"""

import os
import time
import uuid
import json
import threading
import queue
import zipfile
import numpy as np
import cv2
import collections
import traceback

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64, Float64MultiArray
from cv_bridge import CvBridge


# ========== Utility functions ==========

def t_now(node):
    """Return current ROS time in seconds."""
    return node.get_clock().now().nanoseconds / 1e9


def list_to_object_array(lst):
    arr = np.empty((len(lst),), dtype=object)
    arr[:] = lst
    return arr


# ========== General RecorderWorker ==========

class RecorderWorker:
    """Subscribe to a topic and buffer parsed data."""
    def __init__(
        self,
        node,
        topic,
        msg_type,
        parse_fn,
        name='worker',
        maxlen=1000,
        metadata_fn=None,
    ):
        self.node = node
        self.name = name
        self.topic = topic
        self.lock = threading.Lock()
        self.buf = collections.deque(maxlen=maxlen)
        self.parse_fn = parse_fn
        self.metadata_fn = metadata_fn or (lambda _msg: {})
        self.count = 0
        self.sub = node.create_subscription(msg_type, topic, self._cb, 10)
        node.get_logger().info(f"[{name}] Subscribed to {topic}")

    @staticmethod
    def _message_time(node, msg):
        if hasattr(msg, 'header'):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if t > 0.0:
                return t
        return t_now(node)

    def _cb(self, msg):
        try:
            t = self._message_time(self.node, msg)
            data = self.parse_fn(msg)
            metadata = self.metadata_fn(msg)
            with self.lock:
                self.buf.append((t, data, metadata))
                self.count += 1
        except Exception as e:
            self.node.get_logger().warn(f"[{self.name}] parse error: {e}")

    def nearest(self, t_star, max_slop):
        with self.lock:
            if not self.buf:
                return None
            best, best_dt = None, 1e9
            for (t, d, metadata) in self.buf:
                dt = abs(t - t_star)
                if dt < best_dt:
                    best, best_dt = (t, d, metadata), dt
            return best if best_dt <= max_slop else None


# ========== Dedicated Recorder ==========


def maybe_resize_image(image, width=0, height=0):
    """Resize an image if both target dimensions are positive."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return image

    interpolation = cv2.INTER_AREA
    if width > image.shape[1] or height > image.shape[0]:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


class RealSenseRGBRecorder(RecorderWorker):
    """RealSense RGB image recorder."""
    def __init__(self, node, topic='/camera_up/color/image_rect_raw', resize_width=0, resize_height=0):
        bridge = CvBridge()
        resize_width = int(resize_width)
        resize_height = int(resize_height)

        def parse_fn(msg: Image):
            img_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return maybe_resize_image(rgb, resize_width, resize_height)

        super().__init__(node, topic, Image, parse_fn, name='RealSenseRGB')
        
class TactileSensorRecorder(RecorderWorker):
    """Tactile sensor (GelSight) image recorder."""
    def __init__(self, node, topic='/gelsight/image_raw'):
        bridge = CvBridge()

        def parse_fn(msg: Image):
            img_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        super().__init__(node, topic, Image, parse_fn, name='TactileSensor')


class ViveTrackerRecorder(RecorderWorker):
    """Vive tracker pose recorder."""
    def __init__(self, node, topic='/vive_tracker/pose'):
        def parse_fn(msg: PoseStamped):
            p = msg.pose.position
            q = msg.pose.orientation
            return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

        def metadata_fn(msg: PoseStamped):
            return {"frame_id": str(msg.header.frame_id)}

        super().__init__(
            node, topic, PoseStamped, parse_fn, name='ViveTracker', metadata_fn=metadata_fn
        )

class ViveUltimateTrackerRecorder(RecorderWorker):
    def __init__(self, node, topic='/vive_ultimate_tracker/pose'):
        def parse_fn(msg: PoseStamped):
            p = msg.pose.position
            q = msg.pose.orientation
            return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

        def metadata_fn(msg: PoseStamped):
            return {"frame_id": str(msg.header.frame_id)}

        super().__init__(
            node,
            topic,
            PoseStamped,
            parse_fn,
            name='ViveUltimateTracker',
            metadata_fn=metadata_fn,
        )


class PoseStampedRecorder(RecorderWorker):
    """Recorder for PoseStamped topics stored as position + quaternion."""

    def __init__(self, node, topic, name):
        def parse_fn(msg: PoseStamped):
            p = msg.pose.position
            q = msg.pose.orientation
            return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

        def metadata_fn(msg: PoseStamped):
            return {"frame_id": str(msg.header.frame_id)}

        super().__init__(
            node, topic, PoseStamped, parse_fn, name=name, metadata_fn=metadata_fn
        )


class Float64ArrayRecorder(RecorderWorker):
    """Recorder for Float64MultiArray topics such as 4x4 robot transforms."""

    def __init__(self, node, topic, name, valid_sizes=None):
        valid_sizes = tuple(int(v) for v in valid_sizes) if valid_sizes else ()

        def parse_fn(msg: Float64MultiArray):
            arr = np.array(msg.data, dtype=np.float64)
            if valid_sizes and arr.size not in valid_sizes:
                raise ValueError(
                    f"expected {valid_sizes} values, received {arr.size} on {topic}"
                )
            return arr

        super().__init__(node, topic, Float64MultiArray, parse_fn, name=name)


class Float64Recorder(RecorderWorker):
    """Recorder for scalar Float64 topics."""

    def __init__(self, node, topic, name):
        def parse_fn(msg: Float64):
            return float(msg.data)

        super().__init__(node, topic, Float64, parse_fn, name=name)


class BoolRecorder(RecorderWorker):
    """Recorder for scalar Bool topics."""

    def __init__(self, node, topic, name):
        def parse_fn(msg: Bool):
            return bool(msg.data)

        super().__init__(node, topic, Bool, parse_fn, name=name)


class LucidRGBRecorder(RecorderWorker):
    """LUCID RGB image recorder (Bayer BGGR → RGB)."""

    def __init__(self, node, topic='/rgb_lucid'):
        bridge = CvBridge()

        def parse_fn(msg: Image):
            raw = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            rgb = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2RGB)
            return rgb

        super().__init__(node, topic, Image, parse_fn, name='LucidRGB')



# ========== Aggregator ==========

class Aggregator:
    """Sample all workers periodically and aggregate synchronized samples."""
    def __init__(self, node, workers, rate_hz=5.0, slop_sec=0.1, on_sample=None):
        self.node = node
        self.workers = workers
        self.rate_hz = float(rate_hz)
        self.slop = float(slop_sec)
        self.on_sample = on_sample
        self.timer = node.create_timer(1.0 / rate_hz, self._tick)
        self.active = False

    def start(self):
        self.active = True
        self.node.get_logger().info(f"Aggregator started ({self.rate_hz} Hz)")

    def stop(self):
        self.active = False
        self.node.get_logger().info("Aggregator stopped")

    @staticmethod
    def _assign_pick(sample, picks, pick_key, sample_key, sample_t_key, sample_frame_key=None):
        if pick_key not in picks:
            return
        t_value, data_value, metadata = picks[pick_key]
        sample[sample_key] = data_value
        sample[sample_t_key] = float(t_value)
        frame_id = str(metadata.get("frame_id", "")).strip()
        if sample_frame_key and frame_id:
            sample[sample_frame_key] = frame_id

    # using the latest RGB time to sample data from all workers
    def _tick(self):
        if not self.active:
            return
        ref_worker = self.workers.get("realsense_rgb")
        if ref_worker is None:
            ref_worker = next(iter(self.workers.values()), None)
        if not ref_worker or not ref_worker.buf:
            return

        # get latest time from reference worker
        with ref_worker.lock:
            t_star = ref_worker.buf[-1][0]

        picks = {}
        for name, worker in self.workers.items():
            item = worker.nearest(t_star, self.slop)
            if item:
                picks[name] = item

        if len(picks) < len(self.workers):
            return

        # keep old-style "t" (mean of all picked timestamps), but also add t_ref = RGB time
        sample = {
            "t_ref": float(t_star),
            "t": float(np.mean([p[0] for p in picks.values()])),
        }

        # For each modality, store both data and its own timestamp *_t
        self._assign_pick(sample, picks, "realsense_rgb", "rgb", "rgb_t")
        self._assign_pick(sample, picks, "realsense_rgb2", "rgb2", "rgb2_t")
        self._assign_pick(sample, picks, "vive_tracker", "pose", "pose_t", "pose_frame_id")
        self._assign_pick(
            sample,
            picks,
            "vive_ultimate",
            "ultimate_pose",
            "ultimate_pose_t",
            "ultimate_pose_frame_id",
        )
        self._assign_pick(sample, picks, "franka_ee_pose", "franka_ee_pose", "franka_ee_pose_t")
        self._assign_pick(
            sample,
            picks,
            "franka_ee_pose_cmd",
            "franka_ee_pose_cmd",
            "franka_ee_pose_cmd_t",
            "franka_ee_pose_cmd_frame_id",
        )
        self._assign_pick(
            sample,
            picks,
            "franka_gripper_width",
            "franka_gripper_width",
            "franka_gripper_width_t",
        )
        self._assign_pick(
            sample,
            picks,
            "franka_gripper_grasp",
            "franka_gripper_grasp",
            "franka_gripper_grasp_t",
        )
        self._assign_pick(sample, picks, "tactile_sensor", "tactile", "tactile_t")
        self._assign_pick(sample, picks, "lucid_rgb", "lucid_rgb", "lucid_rgb_t")

        if self.on_sample:
            self.on_sample(sample)


# ========== Asynchronous save thread ==========

class SaverWorker:
    def __init__(
        self,
        node,
        out_dir,
        save_format="npz",
        fps=5.0,
        robot_type="custom",
        task_name="default_task",
        chunk_size=1000,
    ):
        self.node = node
        self.out_dir = out_dir
        self.save_format = str(save_format)
        os.makedirs(out_dir, exist_ok=True)
        self.q = queue.Queue()
        self.lerobot_writer = None
        if self.save_format == "lerobot_v21":
            from .lerobot_dataset_export import LeRobotV21DatasetWriter

            self.lerobot_writer = LeRobotV21DatasetWriter(
                dataset_root=out_dir,
                fps=fps,
                robot_type=robot_type,
                default_task=task_name,
                chunk_size=chunk_size,
            )
        elif self.save_format != "npz":
            raise ValueError(f"Unsupported save_format: {self.save_format}")
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def save_async(self, payload):
        self.q.put(payload)

    def join(self):
        self.q.join()

    @staticmethod
    def _is_zipfile(path):
        try:
            return zipfile.is_zipfile(path)
        except Exception:
            return False

    def _loop(self):
        while True:
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                if not rclpy.ok():
                    break
                continue

            try:
                if self.save_format == "npz":
                    self._save_npz(payload)
                else:
                    self._save_lerobot(payload)
            except Exception as e:
                self.node.get_logger().error(f"Save error: {e}")
                self.node.get_logger().error(traceback.format_exc())
                self.node.get_logger().error(
                    f"Save payload keys: {sorted(payload.get('arrays', {}).keys())}"
                )
            finally:
                self.q.task_done()

    def _save_npz(self, payload):
        path = payload["path"]
        arrays = dict(payload["arrays"])
        meta = payload.get("meta", {})
        arrays["meta_json"] = np.array(json.dumps(meta), dtype=object)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        tmp_path = path + ".tmp.npz"
        with open(tmp_path, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

        if not self._is_zipfile(path):
            bad_path = path + ".bad"
            try:
                os.rename(path, bad_path)
            except Exception:
                pass
            raise IOError(f"Invalid NPZ file (moved to {bad_path})")

        self.node.get_logger().info(f"Saved episode to {path}")

    def _save_lerobot(self, payload):
        episode_index, parquet_path = self.lerobot_writer.save_episode(
            arrays=payload["arrays"],
            episode_id=payload["episode_id"],
            task_name=payload.get("task_name"),
        )
        self.node.get_logger().info(
            f"Saved episode {payload['episode_id']} to {parquet_path} "
            f"(LeRobot episode_index={episode_index})"
        )

# ========== Episode Recorder ==========

class EpisodeRecorder:
    def __init__(
        self,
        node,
        workers,
        out_dir='/tmp/data',
        rate_hz=5.0,
        slop_sec=0.1,
        save_format='npz',
        robot_type='custom',
        task_name='default_task',
        chunk_size=1000,
    ):
        self.node = node
        self.workers = workers
        self.save_format = str(save_format)
        self.task_name = str(task_name)
        self.saver = SaverWorker(
            node,
            out_dir,
            save_format=self.save_format,
            fps=rate_hz,
            robot_type=robot_type,
            task_name=self.task_name,
            chunk_size=chunk_size,
        )
        self.agg = Aggregator(node, workers, rate_hz, slop_sec, self._on_sample)
        self._reset_buffers()
        self.recording = False
        self.out_dir = out_dir
        self.gripper_log_period_sec = 0.5
        self._last_gripper_log_time = 0.0
        self._warned_pose_frame_changes = set()
        self._pose_frame_values = collections.defaultdict(set)

    def _reset_buffers(self):
        self.samples = []

    def _on_sample(self, sample):
        self.samples.append(sample)
        self._track_pose_frames(sample)
        self._maybe_log_gripper(sample)

    def _track_pose_frames(self, sample):
        frame_keys = (
            "pose_frame_id",
            "ultimate_pose_frame_id",
            "franka_ee_pose_cmd_frame_id",
        )
        for key in frame_keys:
            frame_id = str(sample.get(key, "")).strip()
            if not frame_id:
                continue
            seen = self._pose_frame_values[key]
            seen.add(frame_id)
            if len(seen) > 1 and key not in self._warned_pose_frame_changes:
                self._warned_pose_frame_changes.add(key)
                self.node.get_logger().warn(
                    f"[recording] {key} changed within one episode: {sorted(seen)}"
                )

    def _maybe_log_gripper(self, sample):
        width = sample.get("franka_gripper_width")
        grasp = sample.get("franka_gripper_grasp")
        if width is None and grasp is None:
            return

        now = time.monotonic()
        if now - self._last_gripper_log_time < self.gripper_log_period_sec:
            return
        self._last_gripper_log_time = now

        parts = []
        if width is not None:
            parts.append(f"width={float(width):.4f}")
        if grasp is not None:
            parts.append(f"grasp={bool(grasp)}")
        self.node.get_logger().info(f"[recording] gripper {' '.join(parts)}")

    def start(self):
        if self.recording:
            return False, "Already recording"
        self.recording = True
        self.episode_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self._reset_buffers()
        self._last_gripper_log_time = 0.0
        self._warned_pose_frame_changes.clear()
        self._pose_frame_values.clear()
        self.agg.start()
        return True, f"Started episode {self.episode_id}"

    def stop(self):
        if not self.recording:
            return False, "Not recording"
        self.recording = False
        self.agg.stop()
        os.makedirs(self.out_dir, exist_ok=True)
        path = None
        if self.save_format == 'npz':
            path = os.path.join(self.out_dir, f"episode_{self.episode_id}.npz")

        t_ref_list = []
        t_list = []

        rgb_list = []
        rgb_t_list = []
        rgb2_list = []
        rgb2_t_list = []
        tactile_list = []
        tactile_t_list = []
        pose_list = []
        pose_t_list = []
        pose_frame_id_list = []
        ultimate_pose_list = []
        ultimate_pose_t_list = []
        ultimate_pose_frame_id_list = []
        franka_ee_pose_list = []
        franka_ee_pose_t_list = []
        franka_ee_pose_cmd_list = []
        franka_ee_pose_cmd_t_list = []
        franka_ee_pose_cmd_frame_id_list = []
        franka_gripper_width_list = []
        franka_gripper_width_t_list = []
        franka_gripper_grasp_list = []
        franka_gripper_grasp_t_list = []
        lucid_rgb_list = []
        lucid_rgb_t_list = []

        for s in self.samples:
            t_ref_list.append(s.get("t_ref", np.nan))
            t_list.append(s.get("t", np.nan))

            if "rgb" in s:
                rgb_list.append(s["rgb"])
                rgb_t_list.append(s.get("rgb_t", np.nan))

            if "rgb2" in s:
                rgb2_list.append(s["rgb2"])
                rgb2_t_list.append(s.get("rgb2_t", np.nan))

            if "tactile" in s:
                tactile_list.append(s["tactile"])
                tactile_t_list.append(s.get("tactile_t", np.nan))

            if "pose" in s:
                pose_list.append(s["pose"])
                pose_t_list.append(s.get("pose_t", np.nan))
                pose_frame_id_list.append(str(s.get("pose_frame_id", "")))

            if "ultimate_pose" in s:
                ultimate_pose_list.append(s["ultimate_pose"])
                ultimate_pose_t_list.append(s.get("ultimate_pose_t", np.nan))
                ultimate_pose_frame_id_list.append(str(s.get("ultimate_pose_frame_id", "")))

            if "franka_ee_pose" in s:
                franka_ee_pose_list.append(s["franka_ee_pose"])
                franka_ee_pose_t_list.append(s.get("franka_ee_pose_t", np.nan))

            if "franka_ee_pose_cmd" in s:
                franka_ee_pose_cmd_list.append(s["franka_ee_pose_cmd"])
                franka_ee_pose_cmd_t_list.append(s.get("franka_ee_pose_cmd_t", np.nan))
                franka_ee_pose_cmd_frame_id_list.append(
                    str(s.get("franka_ee_pose_cmd_frame_id", ""))
                )

            if "franka_gripper_width" in s:
                franka_gripper_width_list.append(s["franka_gripper_width"])
                franka_gripper_width_t_list.append(s.get("franka_gripper_width_t", np.nan))

            if "franka_gripper_grasp" in s:
                franka_gripper_grasp_list.append(s["franka_gripper_grasp"])
                franka_gripper_grasp_t_list.append(s.get("franka_gripper_grasp_t", np.nan))

            if "lucid_rgb" in s:
                lucid_rgb_list.append(s["lucid_rgb"])
                lucid_rgb_t_list.append(s["lucid_rgb_t"])

        arrays = {
            "t_ref": np.array(t_ref_list, dtype=np.float64),
            "t": np.array(t_list, dtype=np.float64),
        }

        t_ref_arr = arrays["t_ref"]

        if rgb_list:
            arrays["rgb"] = np.array(rgb_list, dtype=object)
            if rgb_t_list:
                rgb_t_arr = np.array(rgb_t_list, dtype=np.float64)
                arrays["rgb_t"] = rgb_t_arr
                arrays["rgb_dt"] = rgb_t_arr - t_ref_arr

        if rgb2_list:
            arrays["rgb2"] = np.array(rgb2_list, dtype=object)
            if rgb2_t_list:
                rgb2_t_arr = np.array(rgb2_t_list, dtype=np.float64)
                arrays["rgb2_t"] = rgb2_t_arr
                arrays["rgb2_dt"] = rgb2_t_arr - t_ref_arr

        if tactile_list:
            arrays["tactile"] = np.array(tactile_list, dtype=object)
            if tactile_t_list:
                tactile_t_arr = np.array(tactile_t_list, dtype=np.float64)
                arrays["tactile_t"] = tactile_t_arr
                arrays["tactile_dt"] = tactile_t_arr - t_ref_arr

        if pose_list:
            arrays["pose"] = np.array(pose_list, dtype=np.float32)
            pose_t_arr = np.array(pose_t_list, dtype=np.float64)
            arrays["pose_t"] = pose_t_arr
            arrays["pose_dt"] = pose_t_arr - t_ref_arr
            arrays["pose_frame_id"] = np.array(pose_frame_id_list, dtype=np.str_)

        if ultimate_pose_list:
            arrays["ultimate_pose"] = np.array(ultimate_pose_list, dtype=np.float32)
            ultimate_pose_t_arr = np.array(ultimate_pose_t_list, dtype=np.float64)
            arrays["ultimate_pose_t"] = ultimate_pose_t_arr
            arrays["ultimate_pose_dt"] = ultimate_pose_t_arr - t_ref_arr
            arrays["ultimate_pose_frame_id"] = np.array(ultimate_pose_frame_id_list, dtype=np.str_)

        if franka_ee_pose_list:
            arrays["franka_ee_pose"] = np.array(franka_ee_pose_list, dtype=np.float64)
            franka_ee_pose_t_arr = np.array(franka_ee_pose_t_list, dtype=np.float64)
            arrays["franka_ee_pose_t"] = franka_ee_pose_t_arr
            arrays["franka_ee_pose_dt"] = franka_ee_pose_t_arr - t_ref_arr

        if franka_ee_pose_cmd_list:
            arrays["franka_ee_pose_cmd"] = np.array(franka_ee_pose_cmd_list, dtype=np.float64)
            franka_ee_pose_cmd_t_arr = np.array(franka_ee_pose_cmd_t_list, dtype=np.float64)
            arrays["franka_ee_pose_cmd_t"] = franka_ee_pose_cmd_t_arr
            arrays["franka_ee_pose_cmd_dt"] = franka_ee_pose_cmd_t_arr - t_ref_arr
            arrays["franka_ee_pose_cmd_frame_id"] = np.array(
                franka_ee_pose_cmd_frame_id_list, dtype=np.str_
            )

        if franka_gripper_width_list:
            arrays["franka_gripper_width"] = np.array(franka_gripper_width_list, dtype=np.float64)
            franka_gripper_width_t_arr = np.array(franka_gripper_width_t_list, dtype=np.float64)
            arrays["franka_gripper_width_t"] = franka_gripper_width_t_arr
            arrays["franka_gripper_width_dt"] = franka_gripper_width_t_arr - t_ref_arr

        if franka_gripper_grasp_list:
            arrays["franka_gripper_grasp"] = np.array(franka_gripper_grasp_list, dtype=np.bool_)
            franka_gripper_grasp_t_arr = np.array(franka_gripper_grasp_t_list, dtype=np.float64)
            arrays["franka_gripper_grasp_t"] = franka_gripper_grasp_t_arr
            arrays["franka_gripper_grasp_dt"] = franka_gripper_grasp_t_arr - t_ref_arr

        if lucid_rgb_list:
            arrays["lucid_rgb"] = np.array(lucid_rgb_list, dtype=object)
            lucid_t_arr = np.array(lucid_rgb_t_list, dtype=np.float64)
            arrays["lucid_rgb_t"] = lucid_t_arr
            arrays["lucid_rgb_dt"] = lucid_t_arr - t_ref_arr

        meta = {
            "episode_id": self.episode_id,
            "n_samples": len(self.samples),
            "keys": list(arrays.keys()),
            "save_format": self.save_format,
            "task_name": self.task_name,
            "pose_frame_ids": sorted(v for v in self._pose_frame_values["pose_frame_id"] if v),
            "ultimate_pose_frame_ids": sorted(
                v for v in self._pose_frame_values["ultimate_pose_frame_id"] if v
            ),
            "franka_ee_pose_cmd_frame_ids": sorted(
                v for v in self._pose_frame_values["franka_ee_pose_cmd_frame_id"] if v
            ),
        }

        payload = {
            "arrays": arrays,
            "meta": meta,
            "episode_id": self.episode_id,
            "task_name": self.task_name,
        }
        if path is not None:
            payload["path"] = path
            save_msg = f"Saved {len(self.samples)} samples to {path}"
        else:
            save_msg = f"Queued {len(self.samples)} samples for LeRobot export under {self.out_dir}"
        self.saver.save_async(payload)
        return True, save_msg

    def shutdown(self):
        self.agg.stop()
        self.saver.join()

# ========== Main Node ==========

class DataRecorderNode(Node):
    def __init__(self):
        
        super().__init__('multi_sensor_data_collection_with_timestamps')

        # declare parameters
        self.declare_parameter('out_dir', '/home/tailai.cheng/tailai_ws/src/multi_modal_data_collection/data')
        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('slop_sec', 0.10)
        self.declare_parameter('enable_rgb', True)
        self.declare_parameter('enable_rgb2', False)
        self.declare_parameter('enable_vive', False)
        self.declare_parameter('enable_vive_ultimate', False)
        self.declare_parameter('enable_tactile', False)
        self.declare_parameter('rgb_topic', '/camera_up/color/image_rect_raw')
        self.declare_parameter('rgb2_topic', '/camera_down/color/image_rect_raw')
        self.declare_parameter('rgb_resize_width', 0)
        self.declare_parameter('rgb_resize_height', 0)
        self.declare_parameter('rgb2_resize_width', 0)
        self.declare_parameter('rgb2_resize_height', 0)
        self.declare_parameter('vive_topic', '/vive_tracker/pose')
        self.declare_parameter('tactile_topic', '/gelsight/image_raw')
        self.declare_parameter('vive_ultimate_topic', '/vive_ultimate_tracker/pose')
        self.declare_parameter('enable_franka_ee_pose', False)
        self.declare_parameter('enable_franka_ee_pose_cmd', False)
        self.declare_parameter('enable_franka_gripper_width', False)
        self.declare_parameter('enable_franka_gripper_grasp', False)
        self.declare_parameter('franka_ee_pose_topic', '/frankaRight/ee_pose')
        self.declare_parameter('franka_ee_pose_cmd_topic', '/frankaRight/ee_pose_cmd')
        self.declare_parameter('franka_gripper_width_topic', '/frankaRight/gripper_width')
        self.declare_parameter('franka_gripper_grasp_topic', '/frankaRight/is_grasped')
        # ---- LUCID camera parameters ----
        self.declare_parameter('enable_lucid', False)
        self.declare_parameter('lucid_topic', '/rgb_lucid')
        self.declare_parameter('save_format', 'npz')
        self.declare_parameter('robot_type', 'custom')
        self.declare_parameter('task_name', 'default_task')
        self.declare_parameter('lerobot_chunk_size', 1000)

        


        # read parameters
        out_dir = self.get_parameter('out_dir').value
        rate_hz = self.get_parameter('rate_hz').value
        slop_sec = self.get_parameter('slop_sec').value
        enable_rgb = self.get_parameter('enable_rgb').value
        enable_rgb2 = self.get_parameter('enable_rgb2').value
        enable_vive = self.get_parameter('enable_vive').value
        enable_vive_ultimate = self.get_parameter('enable_vive_ultimate').value
        enable_tactile = self.get_parameter('enable_tactile').value

        rgb_topic = self.get_parameter('rgb_topic').value
        rgb2_topic = self.get_parameter('rgb2_topic').value
        rgb_resize_width = self.get_parameter('rgb_resize_width').value
        rgb_resize_height = self.get_parameter('rgb_resize_height').value
        rgb2_resize_width = self.get_parameter('rgb2_resize_width').value
        rgb2_resize_height = self.get_parameter('rgb2_resize_height').value
        vive_topic = self.get_parameter('vive_topic').value
        vive_ultimate_topic = self.get_parameter('vive_ultimate_topic').value
        tactile_topic = self.get_parameter('tactile_topic').value
        enable_franka_ee_pose = self.get_parameter('enable_franka_ee_pose').value
        enable_franka_ee_pose_cmd = self.get_parameter('enable_franka_ee_pose_cmd').value
        enable_franka_gripper_width = self.get_parameter('enable_franka_gripper_width').value
        enable_franka_gripper_grasp = self.get_parameter('enable_franka_gripper_grasp').value
        franka_ee_pose_topic = self.get_parameter('franka_ee_pose_topic').value
        franka_ee_pose_cmd_topic = self.get_parameter('franka_ee_pose_cmd_topic').value
        franka_gripper_width_topic = self.get_parameter('franka_gripper_width_topic').value
        franka_gripper_grasp_topic = self.get_parameter('franka_gripper_grasp_topic').value
        # LUCID camera topics
        enable_lucid = self.get_parameter('enable_lucid').value
        lucid_topic  = self.get_parameter('lucid_topic').value
        save_format = self.get_parameter('save_format').value
        robot_type = self.get_parameter('robot_type').value
        task_name = self.get_parameter('task_name').value
        lerobot_chunk_size = self.get_parameter('lerobot_chunk_size').value


        

        # --- create workers dynamically ---
        self.workers = {}

        if enable_rgb:
            self.workers["realsense_rgb"] = RealSenseRGBRecorder(
                self, rgb_topic, rgb_resize_width, rgb_resize_height
            )
        if enable_rgb2:
            self.workers["realsense_rgb2"] = RealSenseRGBRecorder(
                self, rgb2_topic, rgb2_resize_width, rgb2_resize_height
            )
        if enable_vive:
            self.workers["vive_tracker"] = ViveTrackerRecorder(self, vive_topic)
        if enable_vive_ultimate:
            self.workers["vive_ultimate"] = ViveUltimateTrackerRecorder(self, vive_ultimate_topic)
        if enable_tactile:
            self.workers["tactile_sensor"] = TactileSensorRecorder(self, tactile_topic)
        if enable_franka_ee_pose:
            self.workers["franka_ee_pose"] = Float64ArrayRecorder(
                self, franka_ee_pose_topic, "FrankaEEPose", valid_sizes=(7, 16)
            )
        if enable_franka_ee_pose_cmd:
            self.workers["franka_ee_pose_cmd"] = PoseStampedRecorder(
                self, franka_ee_pose_cmd_topic, "FrankaEEPoseCmd"
            )
        if enable_franka_gripper_width:
            self.workers["franka_gripper_width"] = Float64Recorder(
                self, franka_gripper_width_topic, "FrankaGripperWidth"
            )
        if enable_franka_gripper_grasp:
            self.workers["franka_gripper_grasp"] = BoolRecorder(
                self, franka_gripper_grasp_topic, "FrankaGripperGrasp"
            )
        # ---- LUCID camera workers ----
        if enable_lucid:
            self.workers["lucid_rgb"] = LucidRGBRecorder(self, lucid_topic)

    

        if not self.workers:
            self.get_logger().warn("⚠️ No sensor workers enabled! Nothing will be recorded.")

        # --- episode recorder ---
        self.episode = EpisodeRecorder(
            self,
            self.workers,
            out_dir,
            rate_hz,
            slop_sec,
            save_format=save_format,
            robot_type=robot_type,
            task_name=task_name,
            chunk_size=lerobot_chunk_size,
        )

        # --- services ---
        self.start_srv = self.create_service(Trigger, 'start_episode', self._srv_start)
        self.stop_srv = self.create_service(Trigger, 'stop_episode', self._srv_stop)

        active = ', '.join(self.workers.keys())
        self.get_logger().info(
            f"Recorder with timestamps ready. Active modalities: {active or 'None'}; "
            f"save_format={save_format}"
        )

    def _srv_start(self, request, response):
        ok, msg = self.episode.start()
        response.success = ok
        response.message = msg
        self.get_logger().info(msg)
        return response

    def _srv_stop(self, request, response):
        ok, msg = self.episode.stop()
        response.success = ok
        response.message = msg
        self.get_logger().info(msg)
        return response

    def destroy_node(self):
        self.episode.shutdown()
        super().destroy_node()


# ========== Main ==========

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down...")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except rclpy._rclpy_pybind11.RCLError:
            pass  # already shut down


if __name__ == '__main__':
    main()
