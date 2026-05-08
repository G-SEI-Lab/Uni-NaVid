# Real-time Uni-NaVid ROS runner

This directory contains the online runner for using Uni-NaVid on the robot dog.

## Prerequisites

Start these outside this node unless you pass `~camera_launch_cmd`:

```bash
# ZSL SDK server in Docker, then ROS bridge on Orin.
python3 tmp/elevator-robot/scripts/zsibot/client_in_orin.py

# RealSense ROS driver publishing:
# /camera_up/color/image_raw
```

## Run

```bash
python3 real_world_uninavid/realtime_uninavid_ros.py \
  _instruction:="move forward to the target and stop." \
  _model_path:=model_zoo/uninavid-7b-full-224-video-fps-1-grid-2
```

Optional camera launch example:

```bash
python3 real_world_uninavid/realtime_uninavid_ros.py \
  _camera_launch_cmd:="roslaunch realsense2_camera rs_camera.launch" \
  _instruction:="move forward to the target and stop."
```

The node publishes `/cmd_vel` and subscribes to:

- `/camera_up/color/image_raw`
- `/zsl/body_velocity`
- `/zsl/body_gyro`
- `/elevator/e_stop` when `~use_estop:=true`

## Key Parameters

- `~action_period_s` default `1.0`:
  one action cycle frequency, including motion + settle.
- `~action_motion_s` default `0.7`:
  used to derive default motion speed.
- `~post_action_settle_s` default `0.2`:
  stop and wait before requesting next inference frame.
- `~forward_distance_m` default `0.50`
- `~turn_angle_deg` default `30.0`
- `~loop_rate_hz` default `30.0`
- `~max_camera_age_s` default `1.0`
- `~camera_decode_max_hz` default `5.0`
- `~resize_before_model` default `false`
- `~model_input_size` default `224`
- `~inference_only` default `false`
- `~inference_only_period_s` default `1.0`
- `~max_runtime_s` default `0`
- `~max_inferences` default `0`
- `~cache_reset_interval` default `0`
- `~feat_cache_max_frames` default `64`
- `~long_feat_cache_max_tokens` default `256`
- `~empty_cuda_cache_every` default `0`
- `~memory_log_interval` default `1`
- `~debug_save_enabled` default `true`
- `~debug_dir` default `real_world_uninavid/debug`
- `~debug_keep_last_images` default `1000`
- `~debug_save_images` default `false`
- `~debug_save_raw_images` default `false`
- `~debug_image_interval` default `1`
- `~debug_image_max_count` default `16`
- `~debug_fsync_events` default `true`
- `~camera_launch_cmd` default empty
- `~shutdown_on_stop` default `true`

Preprocess behavior:

- Recommended (default): `~resize_before_model:=false`, keep raw frame input and let official `image_processor.preprocess` do resize + center crop.
- Optional compatibility fallback: set `~resize_before_model:=true` to force square resize before model input (deviates from official).

Default speeds are derived from the target duration:

- forward speed: `forward_distance_m / action_motion_s`
- turn speed: `turn_angle_deg / action_motion_s`

The node executes one atomic action at a time, but keeps a pending action queue
from Uni-NaVid's four-action prediction. After each completed action it requests
a new inference after stop-settle, and requires a newer frame than the pre-stop
frame to reduce blur. A leading `stop` prediction interrupts immediately.

Debug behavior:

- `events.jsonl` is written when `~debug_save_enabled:=true`; it includes each inference result and memory/cache snapshots.
- Input images are not written by default. Enable `~debug_save_images:=true` for short debug runs.
- Image writes use atomic JPEG replacement and are capped by `~debug_image_max_count` by default to reduce I/O pressure.
- Use `~debug_save_raw_images:=true` only if full-resolution raw frames are needed.

Crash isolation:

- Start `real_world_uninavid/jetson_debug_monitor.sh` before testing to capture `tegrastats`, kernel logs, memory, USB, and process snapshots.
- For valid `kernel_tail.log`, run monitor with root permission (or passwordless `sudo dmesg`): `sudo -E bash real_world_uninavid/jetson_debug_monitor.sh ...`.
- Use `~inference_only:=true` to run camera + model without publishing robot motion commands.
- Use `~camera_decode_max_hz` to avoid converting every camera frame in Python; one fresh frame per action cycle is enough for the current control loop.
