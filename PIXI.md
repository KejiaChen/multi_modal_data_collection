# Package Pixi Note

`export ROS_DISCOVERY_SERVER=10.157.175.222:11811`

This package contains a local `pixi.toml`, but the recommended workflow for this repository is the workspace-level Pixi setup at:

- `/home/rsi/ws_humble/pixi.toml`

That is the canonical setup when working with:

- `multi_modal_data_collection`
- `realsense-ros`

## Recommended Workflow

Use the workspace root, not the package directory:

```bash
cd /home/rsi/ws_humble
export PATH="$HOME/.pixi/bin:$PATH"
pixi install
pixi run build
pixi run ros-shell
```

Daily run commands:

```bash
pixi run ros2 launch realsense2_camera rs_launch.py rgb_camera.color_profile:=640x480x30 depth_module.depth_profile:=640x480x30
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
  
pixi run ros2 launch multi_modal_data_collection record_multimodal_with_timestamps.launch.py \
  rate_hz:=15.0 \
  slop_sec:=0.1 \
  enable_franka_ee_pose:=true \
  enable_franka_ee_pose_cmd:=true \
  enable_franka_gripper_width:=true \
  enable_franka_gripper_grasp:=true \
  enable_rgb:=true \
  enable_rgb2:=true \
  rgb_topic:=/camera/camera/color/image_raw \
  rgb_resize_width:=320 \
  rgb2_resize_width:=320 \
  rgb_resize_height:=240 \
  rgb2_resize_height:=240 \
  out_dir:=/home/tp2/ws_humble_policy/output/pnp_microwave_v1 \
  save_format:=lerobot_v21 \
  task_name:=pnp_microwave \
  robot_type:=franka
pixi run ros2 run multi_modal_data_collection footswitch_trigger_node
```
  
## When to use this package-local manifest

Use `src/multi_modal_data_collection/pixi.toml` only if you intentionally want to work with this package in isolation from the rest of the workspace.

For the full install, build, and run workflow, see:

- `/home/rsi/ws_humble/PIXI.md`
