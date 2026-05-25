# Pixi Workflow Notes

This package includes a local `pixi.toml`, but in this repository the recommended workflow is to use the workspace from:

- `/home/tp2/ws_humble_policy`

That is the main setup when working with:

- `multi_modal_data_collection`
- `realsense-ros`

## Environment

If needed, export the ROS discovery server before launching ROS nodes:

```bash
export ROS_DISCOVERY_SERVER=10.157.175.222:11811
```

## Recommended Setup

Use the workspace root rather than the package directory:

```bash
cd /home/tp2/ws_humble_policy
export PATH="$HOME/.pixi/bin:$PATH"
pixi install
pixi run build
pixi run ros-shell
```

## Launch RealSense

### Single camera

```bash
pixi run ros2 launch realsense2_camera rs_launch.py \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

### Two cameras

This example launches two color streams and disables depth on both cameras:

```bash
pixi run ros2 launch realsense2_camera rs_multi_camera_launch.py \
  enable_depth1:=false \
  enable_color1:=true \
  camera_name1:=cam1 \
  camera_namespace1:=cam1 \
  serial_no1:=_233522077069 \
  rgb_camera.color_profile1:=320x240x30 \
  enable_depth2:=false \
  enable_color2:=true \
  camera_name2:=cam2 \
  camera_namespace2:=cam2 \
  serial_no2:=_141722074396 \
  rgb_camera.color_profile2:=320x240x30
```

## Start Recording


Start the footswitch trigger:

```bash
pixi run ros2 run multi_modal_data_collection footswitch_trigger_node
```

Launch the recorder in a second terminal:
```bash
pixi run ros2 launch multi_modal_data_collection record_multimodal_with_timestamps.launch.py \
  rate_hz:=15.0 \
  slop_sec:=0.1 \
  enable_franka_ee_pose:=true \
  enable_franka_ee_pose_cmd:=true \
  enable_franka_gripper_width:=true \
  enable_franka_gripper_grasp:=true \
  enable_rgb:=true \
  enable_rgb2:=true \
  rgb_topic:=/cam1/cam1/color/image_raw \
  rgb2_topic:=/cam2/cam2/color/image_raw \
  rgb_resize_width:=320 \
  rgb2_resize_width:=320 \
  rgb_resize_height:=240 \
  rgb2_resize_height:=240 \
  out_dir:=/home/tp2/ws_humble_policy/output/pnp_microwave_v1 \
  save_format:=lerobot_v21 \
  task_name:=pnp_microwave \
  robot_type:=franka
```


## Visualize a Dataset

```bash
cd /home/tp2/ws_humble_policy/src/multi_modal_data_collection
pixi run python scripts/rerun_lerobot.py \
  --root-dir /home/tp2/ws_humble_policy/output/pnp_sponge_franka \
  --episode 0
```

## Upload a Dataset to Hugging Face

Using the helper script:

```bash
cd /home/tp2/ws_humble_policy/src/multi_modal_data_collection
pixi run upload-hf -- /home/tp2/ws_humble_policy/output/pnp_sponge_franka
```

Or with an explicit repo id:

```bash
cd /home/tp2/ws_humble_policy/src/multi_modal_data_collection
pixi run python scripts/upload_recordings_to_hf.py \
  /home/tp2/ws_humble_policy/output/pnp_sponge_franka \
  --repo-id kejia98/pnp_sponge_franka
```

## Footswitch Permissions

If the footswitch is not detected:

```bash
ls -la /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
groups tp2
sudo usermod -aG input tp2
newgrp input
```

## Package-Local Pixi

Use `src/multi_modal_data_collection/pixi.toml` only if you intentionally want to work with this package in isolation.

For the broader workspace workflow, use the workspace root:

- `/home/tp2/ws_humble_policy`
