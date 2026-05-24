# Package Pixi Note

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
```

Daily run commands:

```bash
pixi run ros2 launch realsense2_camera rs_launch.py
pixi run ros2 launch multi_modal_data_collection record_multimodal_with_timestamps.launch.py \
  rate_hz:=15.0 \
  slop_sec:=0.1 \
  enable_franka_ee_pose:=true \
  enable_franka_ee_pose_cmd:=true \
  enable_rgb:=true \
  rgb_topic:=/camera/camera/color/image_raw \
  rgb_resize_width:=224 \
  rgb_resize_height:=224 \
  out_dir:=/home/rsi/ws_humble/src/output/pnp_microwave_v1 \
  save_format:=lerobot_v21 \
  task_name:=pnp_microwave \
  robot_type:=franka
pixi run ros2 run multi_modal_data_collection footswitch_trigger_node
```

## When to use this package-local manifest

Use `src/multi_modal_data_collection/pixi.toml` only if you intentionally want to work with this package in isolation from the rest of the workspace.

For the full install, build, and run workflow, see:

- `/home/rsi/ws_humble/PIXI.md`
