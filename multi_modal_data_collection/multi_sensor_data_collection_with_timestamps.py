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

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from manus_ros2_msgs.msg import ManusGlove
from sensor_msgs.msg import CompressedImage



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
    def __init__(self, node, topic, msg_type, parse_fn, name='worker', maxlen=1000):
        self.node = node
        self.name = name
        self.topic = topic
        self.lock = threading.Lock()
        self.buf = collections.deque(maxlen=maxlen)
        self.parse_fn = parse_fn
        self.count = 0
        self.sub = node.create_subscription(msg_type, topic, self._cb, 10)
        self.last_msg_time = None   # ROS time (sec)
        self.first_msg_time = None
        node.get_logger().info(f"[{name}] Subscribed to {topic}")

    def _cb(self, msg):
        try:
            if hasattr(msg, 'header'):
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            else:
                t = t_now(self.node)
            data = self.parse_fn(msg)
            with self.lock:
                self.buf.append((t, data))
                self.count += 1
                self.last_msg_time = t
                if self.first_msg_time is None:
                    self.first_msg_time = t
        except Exception as e:
            self.node.get_logger().warn(f"[{self.name}] parse error: {e}")

    def nearest(self, t_star, max_slop):
        with self.lock:
            if not self.buf:
                return None
            best, best_dt = None, 1e9
            for (t, d) in self.buf:
                dt = abs(t - t_star)
                if dt < best_dt:
                    best, best_dt = (t, d), dt
            return best if best_dt <= max_slop else None

    def seconds_since_last_msg(self, now):
        if self.last_msg_time is None:
            return None
        return now - self.last_msg_time


# ========== Dedicated Recorder ==========

# class RealSenseRGBRecorder(RecorderWorker):
#     """RealSense RGB image recorder."""
#     def __init__(self, node, topic='/camera_up/color/image_rect_raw'):
#         bridge = CvBridge()

#         def parse_fn(msg: Image):
#             img_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#             return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#         super().__init__(node, topic, Image, parse_fn, name='RealSenseRGB')
def encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    """
    Encode RGB uint8 image to JPEG bytes.
    """
    assert rgb.dtype == np.uint8
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()

class RealSenseRGBRecorder(RecorderWorker):
    """RealSense RGB image recorder (JPEG compressed)."""

    def __init__(self, node, topic='/camera_up/color/image_rect_raw', jpeg_quality=85):
        bridge = CvBridge()

        def parse_fn(msg: Image):
            img_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            jpeg = encode_jpeg(rgb, quality=jpeg_quality)
            return jpeg   # 👈 返回 bytes，而不是 numpy

        super().__init__(
            node,
            topic,
            Image,
            parse_fn,
            name='RealSenseRGB',
        )


        
# class TactileSensorRecorder(RecorderWorker):
#     """Tactile sensor (GelSight) image recorder."""
#     def __init__(self, node, topic='/gelsight/image_raw'):
#         bridge = CvBridge()

#         def parse_fn(msg: Image):
#             img_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#             return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#         super().__init__(node, topic, Image, parse_fn, name='TactileSensor')

class TactileSensorRecorder(RecorderWorker):
    """Tactile sensor (GelSight) recorder (JPEG compressed)."""

    def __init__(self, node, topic='/gelsight/image_raw', jpeg_quality=85):
        bridge = CvBridge()

        def parse_fn(msg: Image):
            img_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            jpeg = encode_jpeg(rgb, quality=jpeg_quality)
            return jpeg

        super().__init__(
            node,
            topic,
            Image,
            parse_fn,
            name='TactileSensor',
        )


class ViveTrackerRecorder(RecorderWorker):
    """Vive tracker pose recorder."""
    def __init__(self, node, topic='/vive_tracker/pose'):
        def parse_fn(msg: PoseStamped):
            p = msg.pose.position
            q = msg.pose.orientation
            return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

        super().__init__(node, topic, PoseStamped, parse_fn, name='ViveTracker')

class ViveUltimateTrackerRecorder(RecorderWorker):
    def __init__(self, node, topic='/vive_ultimate_tracker/pose'):
        def parse_fn(msg: PoseStamped):
            p = msg.pose.position
            q = msg.pose.orientation
            return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

        super().__init__(node, topic, PoseStamped, parse_fn, name='ViveUltimateTracker')

class ManusErgoRecorder(RecorderWorker):
    """Recorder for Manus glove ergonomics (20 joint values)."""

    def __init__(self, node, topic, name):
        def parse_fn(msg: ManusGlove):

            # 20 joint values of ergonomics
            ergo_vals = [e.value for e in msg.ergonomics]

            return np.array(ergo_vals, dtype=np.float32)

        super().__init__(node, topic, ManusGlove, parse_fn, name=name)

class ManusNodesRecorder(RecorderWorker):
    """Recorder for Manus glove raw node poses (25 × 7)."""

    def __init__(self, node, topic, name):
        def parse_fn(msg: ManusGlove):

            nodes = []
            # 25 nodes of the glove, each with position (x,y,z) and orientation (x,y,z,w)
            for n in msg.raw_nodes:
                p = n.pose.position
                q = n.pose.orientation
                nodes.append([p.x, p.y, p.z, q.x, q.y, q.z, q.w])

            return np.array(nodes, dtype=np.float32)

        super().__init__(node, topic, ManusGlove, parse_fn, name=name)

# class LucidRGBRecorder(RecorderWorker):
#     """LUCID RGB image recorder (Bayer BGGR → RGB)."""

#     def __init__(self, node, topic='/rgb_lucid'):
#         bridge = CvBridge()

#         def parse_fn(msg: Image):
#             raw = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#             rgb = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2RGB)
#             return rgb

#         super().__init__(node, topic, Image, parse_fn, name='LucidRGB')
class LucidRGBRecorder(RecorderWorker):
    """LUCID RGB recorder (Bayer BGGR → RGB → JPEG)."""

    def __init__(self, node, topic='/rgb_lucid', jpeg_quality=85):
        bridge = CvBridge()

        def parse_fn(msg: Image):
            raw = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            rgb = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2RGB)
            jpeg = encode_jpeg(rgb, quality=jpeg_quality)
            return jpeg

        super().__init__(
            node,
            topic,
            Image,
            parse_fn,
            name='LucidRGB',
        )

class ManusPalmPoseRecorder(RecorderWorker):
    """Recorder for Manus palm PoseStamped (x,y,z,qx,qy,qz,qw)."""
    def __init__(self, node, topic, name="ManusPalmPose"):
        def parse_fn(msg: PoseStamped):
            p = msg.pose.position
            q = msg.pose.orientation
            return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

        super().__init__(node, topic, PoseStamped, parse_fn, name=name)


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
        self._last_warn_time = 0.0
        self.warn_interval = 1.0  # seconds


    def start(self):
        self.active = True
        self.node.get_logger().info(f"Aggregator started ({self.rate_hz} Hz)")

    def stop(self):
        self.active = False
        self.node.get_logger().info("Aggregator stopped")

    # using the latest RGB time to sample data from all workers
    def _tick(self):
        now = t_now(self.node)
        if not self.active:
            return
        # ref_worker = self.workers.get("realsense_rgb")
        # not hard coded realsense_rgb 
        ref_worker = next(iter(self.workers.values()))
        if not ref_worker or not ref_worker.buf:
            return

        # get latest time from reference worker
        with ref_worker.lock:
            t_star = ref_worker.buf[-1][0]

        picks = {}
        missing = []
        for name, worker in self.workers.items():
            item = worker.nearest(t_star, self.slop)
            if item:
                picks[name] = item
            else:
                missing.append(name)

        # ---- NEW: log missing sensors ----
        if missing:
            if now - self._last_warn_time > self.warn_interval:
                self._last_warn_time = now

                msgs = []
                for name in missing:
                    w = self.workers[name]
                    if w.last_msg_time is None:
                        msgs.append(f"{name}: ❌ never received data")
                    else:
                        dt = now - w.last_msg_time
                        msgs.append(f"{name}: ⚠️ last msg {dt:.2f}s ago")

                self.node.get_logger().warn(
                    "[Aggregator] Waiting for sensors:\n  " +
                    "\n  ".join(msgs)
                )
            return

        if len(picks) < len(self.workers):
            return

        # keep old-style "t" (mean of all picked timestamps), but also add t_ref = RGB time
        sample = {
            "t_ref": float(t_star),
            "t": float(np.mean([p[0] for p in picks.values()])),
        }

        # For each modality, store both data and its own timestamp *_t
        if "realsense_rgb" in picks:
            t_rgb, data_rgb = picks["realsense_rgb"]
            sample["rgb"] = data_rgb
            sample["rgb_t"] = float(t_rgb)

        if "realsense_rgb2" in picks:
            t_rgb2, data_rgb2 = picks["realsense_rgb2"]
            sample["rgb2"] = data_rgb2
            sample["rgb2_t"] = float(t_rgb2)

        if "vive_tracker" in picks:
            t_pose, data_pose = picks["vive_tracker"]
            sample["pose"] = data_pose
            sample["pose_t"] = float(t_pose)

        if "vive_ultimate" in picks:
            t_u, data_u = picks["vive_ultimate"]
            sample["ultimate_pose"] = data_u
            sample["ultimate_pose_t"] = float(t_u)


        # if "tactile_sensor" in picks:
        #     t_tact, data_tact = picks["tactile_sensor"]
        #     sample["tactile"] = data_tact
        #     sample["tactile_t"] = float(t_tact)
        if "tactile_left" in picks:
            t, data = picks["tactile_left"]
            sample["tactile_left"] = data
            sample["tactile_left_t"] = float(t)

        if "tactile_right" in picks:
            t, data = picks["tactile_right"]
            sample["tactile_right"] = data
            sample["tactile_right_t"] = float(t)


        # Manus glove data
        if "manus_right_ergo" in picks:
            t_r, data_r = picks["manus_right_ergo"]
            sample["manus_right_ergo"] = data_r
            sample["manus_right_ergo_t"] = float(t_r)

        if "manus_right_nodes" in picks:
            t_rn, data_rn = picks["manus_right_nodes"]
            sample["manus_right_nodes"] = data_rn
            sample["manus_right_nodes_t"] = float(t_rn)

        if "manus_left_ergo" in picks:
            t_l, data_l = picks["manus_left_ergo"]
            sample["manus_left_ergo"] = data_l
            sample["manus_left_ergo_t"] = float(t_l)

        if "manus_left_nodes" in picks:
            t_ln, data_ln = picks["manus_left_nodes"]
            sample["manus_left_nodes"] = data_ln
            sample["manus_left_nodes_t"] = float(t_ln)

        # Manus palm pose topics
        if "manus_right_palm_pose" in picks:
            t_rp, data_rp = picks["manus_right_palm_pose"]
            sample["manus_right_palm_pose"] = data_rp
            sample["manus_right_palm_pose_t"] = float(t_rp)

        if "manus_left_palm_pose" in picks:
            t_lp, data_lp = picks["manus_left_palm_pose"]
            sample["manus_left_palm_pose"] = data_lp
            sample["manus_left_palm_pose_t"] = float(t_lp)

        # LUCID camera data
        if "lucid_rgb" in picks:
            t_l, data_l = picks["lucid_rgb"]
            sample["lucid_rgb"] = data_l
            sample["lucid_rgb_t"] = float(t_l)


        


        if self.on_sample:
            self.on_sample(sample)


# ========== Asynchronous save thread ==========

class SaverWorker:
    def __init__(self, node, out_dir):
        self.node = node
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.q = queue.Queue()
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
                self._save_npz(payload)
            except Exception as e:
                self.node.get_logger().error(f"Save error: {e}")
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


# ========== Episode Recorder ==========

class EpisodeRecorder:
    def __init__(self, node, workers, out_dir='/tmp/data', rate_hz=5.0, slop_sec=0.1):
        self.node = node
        self.workers = workers
        self.saver = SaverWorker(node, out_dir)
        self.agg = Aggregator(node, workers, rate_hz, slop_sec, self._on_sample)
        self._reset_buffers()
        self.recording = False
        self.out_dir = out_dir

    def _reset_buffers(self):
        self.samples = []

    def _on_sample(self, sample):
        self.samples.append(sample)

    def start(self):
        if self.recording:
            return False, "Already recording"
        self.recording = True
        self.episode_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self._reset_buffers()
        self.agg.start()
        return True, f"Started episode {self.episode_id}"

    def stop(self):
        if not self.recording:
            return False, "Not recording"
        self.recording = False
        self.agg.stop()
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"episode_{self.episode_id}.npz")

        # prepare lists
        t_ref_list = []
        t_list = []

        rgb_list = []
        rgb_t_list = []
        rgb2_list = []
        rgb2_t_list = []

        # tactile_list = []
        # tactile_t_list = []
        tactile_left_list = []
        tactile_left_t_list = []
        tactile_right_list = []
        tactile_right_t_list = []

        pose_list = []
        pose_t_list = []
        ultimate_pose_list = []
        ultimate_pose_t_list = []

        # Manus glove data
        manus_right_ergo_list = []
        manus_right_ergo_t_list = []
        manus_right_nodes_list = []
        manus_right_nodes_t_list = []

        manus_left_ergo_list = []
        manus_left_ergo_t_list = []
        manus_left_nodes_list = []
        manus_left_nodes_t_list = []

        manus_right_palm_pose_list = []
        manus_right_palm_pose_t_list = []
        manus_left_palm_pose_list = []
        manus_left_palm_pose_t_list = []

        # LUCID camera data
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

            # if "tactile" in s:
            #     tactile_list.append(s["tactile"])
            #     tactile_t_list.append(s.get("tactile_t", np.nan))
            if "tactile_left" in s:
                tactile_left_list.append(s["tactile_left"])
                tactile_left_t_list.append(s["tactile_left_t"])

            if "tactile_right" in s:
                tactile_right_list.append(s["tactile_right"])
                tactile_right_t_list.append(s["tactile_right_t"])

            if "pose" in s:
                pose_list.append(s["pose"])
                pose_t_list.append(s.get("pose_t", np.nan))

            if "ultimate_pose" in s:
                ultimate_pose_list.append(s["ultimate_pose"])
                ultimate_pose_t_list.append(s.get("ultimate_pose_t", np.nan))
            
            # Manus glove data
            if "manus_right_ergo" in s:
                manus_right_ergo_list.append(s["manus_right_ergo"])
                manus_right_ergo_t_list.append(s["manus_right_ergo_t"])

            if "manus_right_nodes" in s:
                manus_right_nodes_list.append(s["manus_right_nodes"])
                manus_right_nodes_t_list.append(s["manus_right_nodes_t"])

            if "manus_left_ergo" in s:
                manus_left_ergo_list.append(s["manus_left_ergo"])
                manus_left_ergo_t_list.append(s["manus_left_ergo_t"])

            if "manus_left_nodes" in s:
                manus_left_nodes_list.append(s["manus_left_nodes"])
                manus_left_nodes_t_list.append(s["manus_left_nodes_t"])

            if "manus_right_palm_pose" in s:
                manus_right_palm_pose_list.append(s["manus_right_palm_pose"])
                manus_right_palm_pose_t_list.append(s["manus_right_palm_pose_t"])

            if "manus_left_palm_pose" in s:
                manus_left_palm_pose_list.append(s["manus_left_palm_pose"])
                manus_left_palm_pose_t_list.append(s["manus_left_palm_pose_t"])

            # LUCID camera data
            if "lucid_rgb" in s:
                lucid_rgb_list.append(s["lucid_rgb"])
                lucid_rgb_t_list.append(s["lucid_rgb_t"])


        # assemble arrays
        arrays = {
            "t_ref": np.array(t_ref_list, dtype=np.float64),
            "t": np.array(t_list, dtype=np.float64),
        }

        # convenience handle
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

        # if tactile_list:
        #     arrays["tactile"] = np.array(tactile_list, dtype=object)
        #     if tactile_t_list:
        #         tactile_t_arr = np.array(tactile_t_list, dtype=np.float64)
        #         arrays["tactile_t"] = tactile_t_arr
        #         arrays["tactile_dt"] = tactile_t_arr - t_ref_arr
        if tactile_left_list:
            arrays["tactile_left"] = np.array(tactile_left_list, dtype=object)
            t_arr = np.array(tactile_left_t_list, dtype=np.float64)
            arrays["tactile_left_t"] = t_arr
            arrays["tactile_left_dt"] = t_arr - t_ref_arr

        if tactile_right_list:
            arrays["tactile_right"] = np.array(tactile_right_list, dtype=object)
            t_arr = np.array(tactile_right_t_list, dtype=np.float64)
            arrays["tactile_right_t"] = t_arr
            arrays["tactile_right_dt"] = t_arr - t_ref_arr

        if pose_list:
            arrays["pose"] = np.array(pose_list, dtype=np.float32)
            pose_t_arr = np.array(pose_t_list, dtype=np.float64)
            arrays["pose_t"] = pose_t_arr
            arrays["pose_dt"] = pose_t_arr - t_ref_arr

        if ultimate_pose_list:
            arrays["ultimate_pose"] = np.array(ultimate_pose_list, dtype=np.float32)
            ultimate_pose_t_arr = np.array(ultimate_pose_t_list, dtype=np.float64)
            arrays["ultimate_pose_t"] = ultimate_pose_t_arr
            arrays["ultimate_pose_dt"] = ultimate_pose_t_arr - t_ref_arr
        
        # Manus glove data
        if manus_right_ergo_list:
            arrays["manus_right_ergo"] = np.array(manus_right_ergo_list, dtype=object)
            t_arr = np.array(manus_right_ergo_t_list, dtype=np.float64)
            arrays["manus_right_ergo_t"] = t_arr
            arrays["manus_right_ergo_dt"] = t_arr - t_ref_arr

        if manus_right_nodes_list:
            arrays["manus_right_nodes"] = np.array(manus_right_nodes_list, dtype=object)
            t_arr = np.array(manus_right_nodes_t_list, dtype=np.float64)
            arrays["manus_right_nodes_t"] = t_arr
            arrays["manus_right_nodes_dt"] = t_arr - t_ref_arr

        if manus_left_ergo_list:
            arrays["manus_left_ergo"] = np.array(manus_left_ergo_list, dtype=object)
            t_arr = np.array(manus_left_ergo_t_list, dtype=np.float64)
            arrays["manus_left_ergo_t"] = t_arr
            arrays["manus_left_ergo_dt"] = t_arr - t_ref_arr

        if manus_left_nodes_list:
            arrays["manus_left_nodes"] = np.array(manus_left_nodes_list, dtype=object)
            t_arr = np.array(manus_left_nodes_t_list, dtype=np.float64)
            arrays["manus_left_nodes_t"] = t_arr
            arrays["manus_left_nodes_dt"] = t_arr - t_ref_arr
        
        if manus_right_palm_pose_list:
            arrays["manus_right_palm_pose"] = np.array(manus_right_palm_pose_list, dtype=np.float32)
            t_arr = np.array(manus_right_palm_pose_t_list, dtype=np.float64)
            arrays["manus_right_palm_pose_t"] = t_arr
            arrays["manus_right_palm_pose_dt"] = t_arr - t_ref_arr

        if manus_left_palm_pose_list:
            arrays["manus_left_palm_pose"] = np.array(manus_left_palm_pose_list, dtype=np.float32)
            t_arr = np.array(manus_left_palm_pose_t_list, dtype=np.float64)
            arrays["manus_left_palm_pose_t"] = t_arr
            arrays["manus_left_palm_pose_dt"] = t_arr - t_ref_arr


        # LUCID camera data
        if lucid_rgb_list:
            arrays["lucid_rgb"] = np.array(lucid_rgb_list, dtype=object)
            t_arr = np.array(lucid_rgb_t_list, dtype=np.float64)
            arrays["lucid_rgb_t"] = t_arr
            arrays["lucid_rgb_dt"] = t_arr - t_ref_arr





        meta = {
            "episode_id": self.episode_id,
            "n_samples": len(self.samples),
            "keys": list(arrays.keys())
        }

        # save asynchronously
        self.saver.save_async({"path": path, "arrays": arrays, "meta": meta})
        return True, f"Saved {len(self.samples)} samples to {path}"

    def shutdown(self):
        self.agg.stop()
        self.saver.join()


# ========== Main Node ==========

class DataRecorderNode(Node):
    def __init__(self):
        
        super().__init__('multi_sensor_data_collection_with_timestamps')

        # declare parameters
        self.declare_parameter('out_dir', '/home/agile/ros2_ws/src/multi_modal_data_collection/data')
        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('slop_sec', 0.10)

        # realsense cameras up and down
        self.declare_parameter('enable_rgb', True)
        self.declare_parameter('enable_rgb2', False)
        self.declare_parameter('rgb_topic', '/camera_up/color/image_rect_raw')
        self.declare_parameter('rgb2_topic', '/camera_down/color/image_rect_raw')

        # Vive tracker
        self.declare_parameter('enable_vive', False)
        self.declare_parameter('enable_vive_ultimate', True)
        self.declare_parameter('vive_topic', '/vive_tracker/pose')
        self.declare_parameter('vive_ultimate_topic', '/vive_ultimate_tracker/pose')

        # self.declare_parameter('enable_tactile', True)
        # tactile left/right
        self.declare_parameter('enable_tactile_left', True)
        self.declare_parameter('enable_tactile_right', False)

        # self.declare_parameter('tactile_topic', '/gelsight/image_raw')
        self.declare_parameter('tactile_left_topic', '/gelsight/left/image_raw')
        self.declare_parameter('tactile_right_topic', '/gelsight/right/image_raw')
        
        # ---- Manus glove parameters ----
        self.declare_parameter('enable_manus_right_ergo', True)
        self.declare_parameter('enable_manus_left_ergo', False)
        self.declare_parameter('enable_manus_right_nodes', True)
        self.declare_parameter('enable_manus_left_nodes', False)
        # self.declare_parameter('manus_right_topic', '/manus_glove_right_corrected')
        self.declare_parameter('manus_right_topic', '/manus_glove_0')
        # self.declare_parameter('manus_left_topic',  '/manus_glove_left_corrected')
        self.declare_parameter('manus_left_topic',  '/manus_glove_1')
        
        # ---- LUCID camera parameters ----
        self.declare_parameter('enable_lucid', False)
        self.declare_parameter('lucid_topic', '/rgb_lucid')

        # ---- Manus palm pose parameters ----
        self.declare_parameter('enable_manus_right_palm_pose', True)
        self.declare_parameter('enable_manus_left_palm_pose', False)
        self.declare_parameter('manus_right_palm_pose_topic', '/manus/right_palm/pose')
        self.declare_parameter('manus_left_palm_pose_topic',  '/manus/left_palm/pose')


        


        # read parameters
        out_dir = self.get_parameter('out_dir').value
        rate_hz = self.get_parameter('rate_hz').value
        slop_sec = self.get_parameter('slop_sec').value

        enable_rgb = self.get_parameter('enable_rgb').value
        enable_rgb2 = self.get_parameter('enable_rgb2').value
        rgb_topic = self.get_parameter('rgb_topic').value
        rgb2_topic = self.get_parameter('rgb2_topic').value

        enable_vive = self.get_parameter('enable_vive').value
        enable_vive_ultimate = self.get_parameter('enable_vive_ultimate').value
        vive_topic = self.get_parameter('vive_topic').value
        vive_ultimate_topic = self.get_parameter('vive_ultimate_topic').value

        # enable_tactile = self.get_parameter('enable_tactile').value
        # tactile_topic = self.get_parameter('tactile_topic').value
        enable_tactile_left  = self.get_parameter('enable_tactile_left').value
        enable_tactile_right = self.get_parameter('enable_tactile_right').value
        tactile_left_topic  = self.get_parameter('tactile_left_topic').value
        tactile_right_topic = self.get_parameter('tactile_right_topic').value
        

        # ---- Manus glove enable flags ----
        enable_manus_right_ergo  = self.get_parameter('enable_manus_right_ergo').value
        enable_manus_left_ergo   = self.get_parameter('enable_manus_left_ergo').value
        enable_manus_right_nodes = self.get_parameter('enable_manus_right_nodes').value
        enable_manus_left_nodes  = self.get_parameter('enable_manus_left_nodes').value

        enable_manus_right_palm_pose = self.get_parameter('enable_manus_right_palm_pose').value
        enable_manus_left_palm_pose  = self.get_parameter('enable_manus_left_palm_pose').value
        manus_right_palm_pose_topic  = self.get_parameter('manus_right_palm_pose_topic').value
        manus_left_palm_pose_topic   = self.get_parameter('manus_left_palm_pose_topic').value

        # Manus topics
        manus_right_topic = self.get_parameter('manus_right_topic').value
        manus_left_topic  = self.get_parameter('manus_left_topic').value

        # LUCID camera topics
        enable_lucid = self.get_parameter('enable_lucid').value
        lucid_topic  = self.get_parameter('lucid_topic').value

        

        # --- create workers dynamically ---
        self.workers = {}

        if enable_rgb:
            self.workers["realsense_rgb"] = RealSenseRGBRecorder(self, rgb_topic)
        if enable_rgb2:
            self.workers["realsense_rgb2"] = RealSenseRGBRecorder(self, rgb2_topic)
        if enable_vive:
            self.workers["vive_tracker"] = ViveTrackerRecorder(self, vive_topic)
        if enable_vive_ultimate:
            self.workers["vive_ultimate"] = ViveUltimateTrackerRecorder(self, vive_ultimate_topic)
        # if enable_tactile:
        #     self.workers["tactile_sensor"] = TactileSensorRecorder(self, tactile_topic)
        if enable_tactile_left:
            self.workers["tactile_left"] = TactileSensorRecorder(
                self, tactile_left_topic
            )

        if enable_tactile_right:
            self.workers["tactile_right"] = TactileSensorRecorder(
                self, tactile_right_topic
            )


        # ---- Manus glove workers ----
        if enable_manus_right_ergo:
            self.workers["manus_right_ergo"] = ManusErgoRecorder(
                self, manus_right_topic, "ManusRightErgo"
            )
        if enable_manus_right_nodes:
            self.workers["manus_right_nodes"] = ManusNodesRecorder(
                self, manus_right_topic, "ManusRightNodes"
            )
        if enable_manus_left_ergo:
            self.workers["manus_left_ergo"] = ManusErgoRecorder(
                self, manus_left_topic, "ManusLeftErgo"
            )
        if enable_manus_left_nodes:
            self.workers["manus_left_nodes"] = ManusNodesRecorder(
                self, manus_left_topic, "ManusLeftNodes"
            )

        # ---- Manus palm pose workers ----
        if enable_manus_right_palm_pose:
            self.workers["manus_right_palm_pose"] = ManusPalmPoseRecorder(
                self, manus_right_palm_pose_topic, "ManusRightPalmPose"
            )

        if enable_manus_left_palm_pose:
            self.workers["manus_left_palm_pose"] = ManusPalmPoseRecorder(
                self, manus_left_palm_pose_topic, "ManusLeftPalmPose"
            )

        # ---- LUCID camera workers ----
        if enable_lucid:
            self.workers["lucid_rgb"] = LucidRGBRecorder(self, lucid_topic)

    

        if not self.workers:
            self.get_logger().warn("⚠️ No sensor workers enabled! Nothing will be recorded.")

        # --- episode recorder ---
        self.episode = EpisodeRecorder(self, self.workers, out_dir, rate_hz, slop_sec)

        # --- services ---
        self.start_srv = self.create_service(Trigger, 'start_episode', self._srv_start)
        self.stop_srv = self.create_service(Trigger, 'stop_episode', self._srv_stop)

        active = ', '.join(self.workers.keys())
        self.get_logger().info(f"Recorder with timestamps ready. Active modalities: {active or 'None'}")

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
