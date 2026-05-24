#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Multimodal Data Recorder (Humble)
Record RealSense RGB and Vive Tracker pose only.
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


# ========== Dedicated Recorder ==========

class RealSenseRGBRecorder(RecorderWorker):
    """RealSense RGB image recorder."""
    def __init__(self, node, topic='/camera_up/color/image_rect_raw'):
        bridge = CvBridge()

        def parse_fn(msg: Image):
            img_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

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

        super().__init__(node, topic, PoseStamped, parse_fn, name='ViveTracker')

class ViveUltimateTrackerRecorder(RecorderWorker):
    """Vive ultimate tracker pose recorder."""
    def __init__(self, node, topic='/vive_ultimate_tracker/pose'):
        def parse_fn(msg: PoseStamped):
            p = msg.pose.position
            q = msg.pose.orientation
            return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

        super().__init__(node, topic, PoseStamped, parse_fn, name='ViveUltimateTracker')


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
        
    # # using the t_star time to sample data from all workers
    # def _tick(self):
    #     if not self.active:
    #         return
    #     t_star = t_now(self.node)
    #     picks = {}
    #     for name, worker in self.workers.items():
    #         item = worker.nearest(t_star, self.slop)
    #         if item:
    #             picks[name] = item
    #     if len(picks) < len(self.workers):
    #         return

    # using the latest RGB time to sample data from all workers
    def _tick(self):
        if not self.active:
            return
        ref_worker = self.workers.get("realsense_rgb")
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

        sample = {"t": float(np.mean([p[0] for p in picks.values()]))}
        if "realsense_rgb" in picks:
            sample["rgb"] = picks["realsense_rgb"][1]
        if "realsense_rgb2" in picks:
            sample["rgb2"] = picks["realsense_rgb2"][1]
        if "vive_tracker" in picks:
            sample["pose"] = picks["vive_tracker"][1]
        if "vive_ultimate_tracker" in picks:
            sample["pose_ultimate"] = picks["vive_ultimate_tracker"][1]
        if "tactile_sensor" in picks:
            sample["tactile"] = picks["tactile_sensor"][1]
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

    # def stop(self):
    #     if not self.recording:
    #         return False, "Not recording"
    #     self.recording = False
    #     self.agg.stop()
    #     os.makedirs(self.out_dir, exist_ok=True)
    #     path = os.path.join(self.out_dir, f"episode_{self.episode_id}.npz")
    #     arrays = {"samples": list_to_object_array(self.samples)}
    #     meta = {"episode_id": self.episode_id, "n_samples": len(self.samples)}
    #     self.saver.save_async({"path": path, "arrays": arrays, "meta": meta})
    #     return True, f"Saved {len(self.samples)} samples to {path}"
    def stop(self):
        if not self.recording:
            return False, "Not recording"
        self.recording = False
        self.agg.stop()
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"episode_{self.episode_id}.npz")

        # prepare arrays
        t_list = []
        rgb_list = []
        rgb2_list = []
        tactile_list = []
        pose_list = []
        pose_ultimate_list = []
        for s in self.samples:
            t_list.append(s.get("t", np.nan))
            if "rgb" in s:
                rgb_list.append(s["rgb"])
            if "rgb2" in s:
                rgb2_list.append(s["rgb2"])
            if "tactile" in s:
                tactile_list.append(s["tactile"])
            if "pose" in s:
                pose_list.append(s["pose"])
            if "pose_ultimate" in s:
                pose_ultimate_list.append(s["pose_ultimate"])
        # assemble arrays 
        arrays = {
            "t": np.array(t_list, dtype=np.float64),
            
        }
        if pose_list:
            arrays["pose"] = np.array(pose_list, dtype=np.float32)
        if pose_ultimate_list:
            arrays["pose_ultimate"] = np.array(pose_ultimate_list, dtype=np.float32)
        if rgb_list:
            arrays["rgb"] = np.array(rgb_list, dtype=object)
        if rgb2_list:
            arrays["rgb2"] = np.array(rgb2_list, dtype=object)
        if tactile_list:
            arrays["tactile"] = np.array(tactile_list, dtype=object)
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
        super().__init__('multi_sensor_data_collection')

        # declare parameters
        self.declare_parameter('out_dir', '/home/tailai.cheng/tailai_ws/src/multi_modal_data_collection/data')
        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('slop_sec', 0.10)
        self.declare_parameter('enable_rgb', True)
        self.declare_parameter('enable_rgb2', False)
        self.declare_parameter('enable_vive', True)
        self.declare_parameter('enable_tactile', True)
        self.declare_parameter('enable_vive_ultimate', True)
        self.declare_parameter('rgb_topic', '/camera_up/color/image_rect_raw')
        self.declare_parameter('rgb2_topic', '/camera_down/color/image_rect_raw')
        self.declare_parameter('vive_topic', '/vive_tracker/pose')
        self.declare_parameter('tactile_topic', '/gelsight/image_raw')
        self.declare_parameter('vive_ultimate_topic', '/vive_ultimate_tracker/pose')
        # read parameters
        out_dir = self.get_parameter('out_dir').value
        rate_hz = self.get_parameter('rate_hz').value
        slop_sec = self.get_parameter('slop_sec').value
        enable_rgb = self.get_parameter('enable_rgb').value
        enable_rgb2 = self.get_parameter('enable_rgb2').value
        enable_vive = self.get_parameter('enable_vive').value
        enable_tactile = self.get_parameter('enable_tactile').value
        rgb_topic = self.get_parameter('rgb_topic').value
        rgb2_topic = self.get_parameter('rgb2_topic').value
        vive_topic = self.get_parameter('vive_topic').value
        tactile_topic = self.get_parameter('tactile_topic').value
        enable_vive_ultimate = self.get_parameter('enable_vive_ultimate').value
        vive_ultimate_topic = self.get_parameter('vive_ultimate_topic').value
        # --- create workers dynamically ---
        self.workers = {}

        if enable_rgb:
            self.workers["realsense_rgb"] = RealSenseRGBRecorder(self, rgb_topic)
        if enable_rgb2:
            self.workers["realsense_rgb2"] = RealSenseRGBRecorder(self, rgb2_topic)
        if enable_vive:
            self.workers["vive_tracker"] = ViveTrackerRecorder(self, vive_topic)
        if enable_vive_ultimate:
            self.workers["vive_ultimate_tracker"] = ViveUltimateTrackerRecorder(self, vive_ultimate_topic)
        if enable_tactile:
            self.workers["tactile_sensor"] = TactileSensorRecorder(self, tactile_topic)

        if not self.workers:
            self.get_logger().warn("⚠️ No sensor workers enabled! Nothing will be recorded.")

        # --- episode recorder ---
        self.episode = EpisodeRecorder(self, self.workers, out_dir, rate_hz, slop_sec)

        # --- services ---
        self.start_srv = self.create_service(Trigger, 'start_episode', self._srv_start)
        self.stop_srv = self.create_service(Trigger, 'stop_episode', self._srv_stop)

        active = ', '.join(self.workers.keys())
        self.get_logger().info(f"Recorder ready. Active modalities: {active or 'None'}")


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
