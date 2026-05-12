# Real-time Uni-NaVid ROS 运行器

本目录包含用于在机器狗上运行 Uni-NaVid 的在线部署脚本。

## 前置条件

除非通过 `~camera_launch_cmd` 让本节点自动启动相机，否则需要在本节点外部先启动以下服务：

```bash
# ZSL SDK server in Docker, then ROS bridge on Orin.
python3 tmp/elevator-robot/scripts/zsibot/client_in_orin.py

# RealSense ROS driver publishing:
# /camera_up/color/image_raw
```

## 运行

串行运行器。该版本会等待每个动作后的推理完成，再开始下一个动作，适合测试和调试：

```bash
python3 real_world_uninavid/realtime_uninavid_ros.py \
  _instruction:="move forward to the target and stop." \
  _model_path:=model_zoo/uninavid-7b-full-224-video-fps-1-grid-2
```

流水线运行器。该版本会让动作执行和模型推理并行进行：

```bash
python3 real_world_uninavid/realtime_uninavid_ros_pipeline.py \
  _instruction:="move forward to the target and stop." \
  _model_path:=model_zoo/uninavid-7b-full-224-video-fps-1-grid-2
```

指令服务式流水线运行器。该版本启动后只加载模型和相机，不自动推理、不自动运动；收到 `/uninavid/instruction` 后才开始新任务。收到新指令时，会停止当前任务、清空动作队列，并在推理线程中重置模型在线缓存后执行新任务：

```bash
python3 real_world_uninavid/realtime_uninavid_ros_pipeline_instruction.py \
  _model_path:=model_zoo/uninavid-7b-full-224-video-fps-1-grid-2
```

发送新任务指令：

```bash
rostopic pub /uninavid/instruction std_msgs/String "move forward to the target and stop."
```

取消当前任务并等待下一条指令：

```bash
rostopic pub /uninavid/cancel std_msgs/Bool "data: true"
```

可选的相机自动启动示例：

```bash
python3 real_world_uninavid/realtime_uninavid_ros_pipeline.py \
  _camera_launch_cmd:="roslaunch realsense2_camera rs_camera.launch" \
  _instruction:="move forward to the target and stop."
```

## 离线官方推理验证

两个真机部署脚本都支持离线验证模式。传入 `~offline_eval_dir` 后，脚本会跳过相机订阅、运动控制和真机控制循环，改为按 `offline_eval_uninavid.py` 的方式读取文件夹图像、逐帧推理、绘制动作箭头并保存 GIF。

输入目录结构需要与官方离线评估一致：

- `path/to/test_case/images/*.jpg`
- `path/to/test_case/instruction.json`

串行脚本离线验证：

```bash
python3 real_world_uninavid/realtime_uninavid_ros.py \
  _offline_eval_dir:=path/to/test_case \
  _offline_eval_output_dir:=path/to/output \
  _model_path:=model_zoo/uninavid-7b-full-224-video-fps-1-grid-2
```

流水线脚本离线验证：

```bash
python3 real_world_uninavid/realtime_uninavid_ros_pipeline.py \
  _offline_eval_dir:=path/to/test_case \
  _offline_eval_output_dir:=path/to/output \
  _model_path:=model_zoo/uninavid-7b-full-224-video-fps-1-grid-2
```

离线验证输出：

- `result.gif`：与官方后处理一致的动作箭头动图。
- `result.jsonl`：每帧的 step、推理耗时、动作列表和轨迹结果，便于对比两份真机脚本和官方脚本输出。

节点发布 `/cmd_vel`，并订阅：

- `/camera_up/color/image_raw`
- `/zsl/body_velocity`
- `/zsl/body_gyro`
- 当 `~use_estop:=true` 时订阅 `/elevator/e_stop`

## 关键参数

- `~action_period_s` 默认 `1.0`：
  一个动作周期的总时长，包括运动和停稳。
- `~action_motion_s` 默认 `0.7`：
  用于推导默认运动速度。
- `~post_action_settle_s` 默认 `0.2`：
  动作后停止等待，再请求下一帧推理图像。
- `~forward_distance_m` 默认 `0.50`
- `~turn_angle_deg` 默认 `30.0`
- `~loop_rate_hz` 默认 `30.0`
- `~max_camera_age_s` 默认 `1.0`
- `~camera_decode_max_hz` 默认 `5.0`
- `~resize_before_model` 默认 `false`
- `~model_input_size` 默认 `224`
- `~inference_only` 默认 `false`
- `~inference_only_period_s` 默认 `1.0`
- `~max_runtime_s` 默认 `0`
- `~max_inferences` 默认 `0`
- `~cache_reset_interval` 默认 `0`
- `~feat_cache_max_frames` 默认 `64`
- `~long_feat_cache_max_tokens` 默认 `256`
- `~empty_cuda_cache_every` 默认 `0`
- `~memory_log_interval` 默认 `1`
- `~debug_save_enabled` 默认 `true`
- `~debug_dir` 默认 `real_world_uninavid/debug`
- `~debug_keep_last_images` 默认 `1000`
- `~debug_save_images` 默认 `false`
- `~debug_save_raw_images` 默认 `false`
- `~debug_image_interval` 默认 `1`
- `~debug_image_max_count` 默认 `16`
- `~debug_fsync_events` 默认 `true`
- `~camera_launch_cmd` 默认空
- `~shutdown_on_stop` 默认 `true`
- `~offline_eval_dir` 默认空：
  非空时进入离线官方推理验证模式。
- `~offline_eval_output_dir` 默认空：
  为空时输出到 `real_world_uninavid/offline_eval_output`。
- `~instruction_topic` 默认 `/uninavid/instruction`：
  仅用于 `realtime_uninavid_ros_pipeline_instruction.py`，用于接收新任务指令。
- `~cancel_topic` 默认 `/uninavid/cancel`：
  仅用于 `realtime_uninavid_ros_pipeline_instruction.py`，收到 `std_msgs/Bool` 的 `true` 后取消当前任务、停止并等待下一条指令。

`realtime_uninavid_ros_pipeline.py` 的流水线版本默认值差异：

- `~action_motion_s` 默认 `1.0`
- `~forward_distance_m` 默认 `0.25`
- `~turn_angle_deg` 默认 `30.0`
- `~camera_decode_max_hz` 默认 `5.0`
- `~min_inference_request_period_s` 默认 `0.0`
- `~still_frame_wait_timeout_s` 默认 `0.25`
- `~distance_tolerance_m`、`~distance_stop_lead_m`、`~turn_tolerance_deg` 和 `~turn_stop_lead_deg` 默认 `0.0`

流水线行为：

- 第一次推理用于初始化待执行动作队列。
- 每个动作按一个动作单元控制：前进 `0.25m` 用 `1.0s`，或旋转 `30deg` 用 `1.0s`，除非覆盖速度或动作单元参数。
- 每个动作完成后，节点会在 `~post_action_settle_s` 内短暂发布零速度，抓取一张新的静止帧，并把复制后的图像提交给推理线程。
- 如果队列中仍有待执行动作，机器人不会等待这次推理完成；静止帧捕获窗口结束后继续执行剩余动作。
- 每个推理请求都会记录已完成动作数，作为该帧对应的动作锚点。推理结果返回时，会丢弃已经完成或当前正在执行的动作槽对应的预测动作。
- 剩余预测动作会替换当前待执行队列。当前正在执行的动作不会被打断。
- 过时动作裁剪后，如果首个有效动作是 `stop`，空闲时会立即停止；运动中则排队到当前动作结束后停止。

预处理行为：

- 推荐方式（默认）：`~resize_before_model:=false`，保留原始相机帧输入，由官方 `image_processor.preprocess` 完成 resize 和 center crop。
- 可选兼容方案：设置 `~resize_before_model:=true`，在送入模型前强制做方形 resize（这会偏离官方预处理）。

默认速度由目标动作时长推导：

- 前进速度：`forward_distance_m / action_motion_s`
- 旋转速度：`turn_angle_deg / action_motion_s`

串行运行器每次只执行一个原子动作，但会保留 Uni-NaVid 四动作预测中的待执行动作队列。每个动作完成后，它会在 stop-settle 后请求一次新推理，并要求图像比停止前更新，以降低运动模糊影响。首个 `stop` 预测会立即中断执行。

调试行为：

- 当 `~debug_save_enabled:=true` 时写入 `events.jsonl`；其中包含每次推理结果以及内存/cache 快照。
- 默认不保存输入图像。短时间调试时可启用 `~debug_save_images:=true`。
- 图像写入使用原子 JPEG 替换；默认由 `~debug_image_max_count` 限制保存数量，以降低 I/O 压力。
- 只有需要完整分辨率原始帧时，才使用 `~debug_save_raw_images:=true`。

崩溃隔离：

- 测试前启动 `real_world_uninavid/jetson_debug_monitor.sh`，用于捕获 `tegrastats`、内核日志、内存、USB 和进程快照。
- 为了得到有效的 `kernel_tail.log`，请用 root 权限运行 monitor（或配置免密码 `sudo dmesg`）：`sudo -E bash real_world_uninavid/jetson_debug_monitor.sh ...`。
- 使用 `~inference_only:=true` 可以只运行相机和模型，不发布机器人运动控制指令。
- 使用 `~camera_decode_max_hz` 可以避免在 Python 中转换每一帧相机图像；当前控制循环中每个动作周期一张新帧即可。
