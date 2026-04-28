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

- `~forward_distance_m` default `0.50`
- `~turn_angle_deg` default `30.0`
- `~action_duration_s` default `0.7`
- `~loop_rate_hz` default `30.0`
- `~max_camera_age_s` default `1.0`
- `~camera_launch_cmd` default empty
- `~shutdown_on_stop` default `true`

Default speeds are derived from the target duration:

- forward speed: `forward_distance_m / action_duration_s`
- turn speed: `turn_angle_deg / action_duration_s`

The node executes one atomic action at a time, but keeps a pending action queue
from Uni-NaVid's four-action prediction. After each completed action it requests
a new inference on the latest camera frame. If the new inference returns while an
older action is still executing, only the pending queue is replaced; a leading
`stop` prediction interrupts immediately.
