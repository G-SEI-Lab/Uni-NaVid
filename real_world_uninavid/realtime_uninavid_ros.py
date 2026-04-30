#!/usr/bin/env python3
"""Run Uni-NaVid online from a ROS camera topic and publish robot dog commands.

This node assumes the ZSL bridge is already running:

  RealSense ROS driver -> /camera_up/color/image_raw
  this node            -> /cmd_vel
  zsl client bridge    -> SDK move(vx, vy, yaw_rate)

The control loop intentionally runs independently from model inference. Uni-NaVid
predicts short action chunks, while this node executes one atomic action at a
time with body velocity / gyro feedback.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Vector3Stamped
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Bool, Float32, String


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_eval_uninavid import UniNaVid_Agent  # noqa: E402


RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0
ACTION_RE = re.compile(r"\b(forward|left|right|stop)\b", re.IGNORECASE)

DEFAULT_MODEL_PATH = "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"
DEFAULT_INSTRUCTION = "move forward to the chair and turn right, then move forward 10 step and stop."


@dataclass
class FrameSnapshot:
    image_bgr: object
    stamp: rospy.Time
    seq: int
    age_s: float


@dataclass
class InferenceResult:
    generation: int
    actions: List[str]
    raw_actions: str
    inference_s: float
    frame_seq: int
    frame_age_s: float
    error: Optional[str] = None


@dataclass
class RunningAction:
    name: str
    target: float
    speed: float
    direction: float
    start_ros: rospy.Time
    start_monotonic: float
    progress: float = 0.0
    feedback_ready: bool = False
    last_feedback_stamp: Optional[rospy.Time] = None
    open_loop: bool = False


def _resolve_repo_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _stamp_or_now(stamp: rospy.Time) -> rospy.Time:
    if stamp and stamp.to_sec() > 0.0:
        return stamp
    return rospy.Time.now()


def _parse_actions(raw_actions, max_actions: int) -> List[str]:
    if isinstance(raw_actions, (list, tuple)):
        text = " ".join(str(item) for item in raw_actions)
    else:
        text = str(raw_actions or "")

    actions: List[str] = []
    for match in ACTION_RE.findall(text.lower()):
        action = str(match).lower()
        actions.append(action)
        if action == "stop" or len(actions) >= max_actions:
            break
    return actions


class LatestImageBuffer:
    def __init__(self, topic: str) -> None:
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._image_bgr = None
        self._stamp = rospy.Time(0.0)
        self._seq = 0
        self._monotonic = 0.0
        self._sub = rospy.Subscriber(
            topic,
            RosImage,
            self._image_cb,
            queue_size=1,
            buff_size=2**24,
        )

    def _image_cb(self, msg: RosImage) -> None:
        try:
            image_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(2.0, "[uninavid] camera conversion failed: %s", exc)
            return

        with self._lock:
            self._seq += 1
            self._image_bgr = image_bgr.copy()
            self._stamp = _stamp_or_now(msg.header.stamp)
            self._monotonic = time.monotonic()

    def latest(self, max_age_s: float) -> Optional[FrameSnapshot]:
        with self._lock:
            if self._image_bgr is None:
                return None
            age_s = time.monotonic() - self._monotonic
            if max_age_s > 0.0 and age_s > max_age_s:
                return None
            return FrameSnapshot(
                image_bgr=self._image_bgr.copy(),
                stamp=self._stamp,
                seq=self._seq,
                age_s=age_s,
            )


class UniNaVidInferenceWorker:
    def __init__(
        self,
        agent: UniNaVid_Agent,
        frame_buffer: LatestImageBuffer,
        instruction: str,
        *,
        max_frame_age_s: float,
        max_actions: int,
    ) -> None:
        self._agent = agent
        self._frame_buffer = frame_buffer
        self._instruction = instruction
        self._max_frame_age_s = max_frame_age_s
        self._max_actions = max_actions

        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._busy = False
        self._last_request_reason = ""
        self._latest: Optional[InferenceResult] = None
        self._generation = 0
        self._thread = threading.Thread(target=self._run, name="uninavid_inference", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._request_event.set()
        self._thread.join(timeout=2.0)

    def request(self, reason: str) -> None:
        with self._lock:
            self._last_request_reason = reason
        self._request_event.set()

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def latest_after(self, generation: int) -> Optional[InferenceResult]:
        with self._lock:
            if self._latest is not None and self._latest.generation > generation:
                return self._latest
        return None

    def _set_busy(self, busy: bool) -> None:
        with self._lock:
            self._busy = busy

    def _store_result(self, result: InferenceResult) -> None:
        with self._lock:
            self._latest = result

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self._request_event.wait(timeout=0.1):
                continue
            if self._stop_event.is_set():
                break
            self._request_event.clear()

            frame = self._frame_buffer.latest(self._max_frame_age_s)
            if frame is None:
                rospy.logwarn_throttle(
                    2.0,
                    "[uninavid] no fresh camera frame available for inference",
                )
                continue

            with self._lock:
                reason = self._last_request_reason
                self._busy = True

            t0 = time.monotonic()
            rospy.loginfo(
                "[uninavid] inference start reason=%s frame_seq=%d age=%.3fs",
                reason,
                frame.seq,
                frame.age_s,
            )
            try:
                result_dict = self._agent.act(
                    {
                        "instruction": self._instruction,
                        "observations": frame.image_bgr,
                    }
                )
                raw_actions = " ".join(str(item) for item in result_dict.get("actions", []))
                actions = _parse_actions(raw_actions, self._max_actions)
                error = None
                if not actions:
                    error = f"no valid action parsed from: {raw_actions!r}"
            except Exception as exc:  # noqa: BLE001
                raw_actions = ""
                actions = []
                error = str(exc)
                rospy.logerr("[uninavid] inference failed: %s", exc)
            finally:
                dt = time.monotonic() - t0
                self._set_busy(False)

            with self._lock:
                self._generation += 1
                generation = self._generation

            result = InferenceResult(
                generation=generation,
                actions=actions,
                raw_actions=raw_actions,
                inference_s=dt,
                frame_seq=frame.seq,
                frame_age_s=frame.age_s,
                error=error,
            )
            self._store_result(result)

            if error:
                rospy.logwarn(
                    "[uninavid] inference result invalid gen=%d time=%.3fs error=%s",
                    generation,
                    dt,
                    error,
                )
            else:
                rospy.loginfo(
                    "[uninavid] inference done gen=%d time=%.3fs actions=%s",
                    generation,
                    dt,
                    " ".join(actions),
                )


class UniNaVidRealtimeNode:
    def __init__(self) -> None:
        self.model_path = _resolve_repo_path(str(rospy.get_param("~model_path", DEFAULT_MODEL_PATH)))
        self.instruction = str(rospy.get_param("~instruction", DEFAULT_INSTRUCTION))

        self.camera_topic = str(rospy.get_param("~camera_topic", "/camera_down/color/image_raw"))
        self.camera_launch_cmd = str(rospy.get_param("~camera_launch_cmd", "")).strip()
        self.camera_launch_startup_s = float(rospy.get_param("~camera_launch_startup_s", 2.0))
        self.cmd_vel_topic = str(rospy.get_param("~cmd_vel_topic", "/cmd_vel"))
        self.body_velocity_topic = str(rospy.get_param("~body_velocity_topic", "/zsl/body_velocity"))
        self.gyro_topic = str(rospy.get_param("~gyro_topic", "/zsl/body_gyro"))

        self.loop_rate_hz = float(rospy.get_param("~loop_rate_hz", 30.0))
        self.max_actions = int(rospy.get_param("~max_actions", 4))
        self.max_camera_age_s = float(rospy.get_param("~max_camera_age_s", 1.0))
        self.camera_wait_timeout_s = float(rospy.get_param("~camera_wait_timeout_s", 10.0))
        self.idle_reinfer_period_s = float(rospy.get_param("~idle_reinfer_period_s", 0.5))
        self.min_inference_request_period_s = float(
            rospy.get_param("~min_inference_request_period_s", 0.2)
        )

        self.action_duration_s = float(rospy.get_param("~action_duration_s", 0.7))
        self.forward_distance_m = float(rospy.get_param("~forward_distance_m", 0.50))
        self.forward_speed_mps = float(
            rospy.get_param(
                "~forward_speed_mps",
                self.forward_distance_m / max(self.action_duration_s, 1e-3),
            )
        )
        self.turn_angle_deg = float(rospy.get_param("~turn_angle_deg", 30.0))
        self.turn_yaw_rate_rps = float(
            rospy.get_param(
                "~turn_yaw_rate_rps",
                math.radians(self.turn_angle_deg) / max(self.action_duration_s, 1e-3),
            )
        )

        self.distance_tolerance_m = float(rospy.get_param("~distance_tolerance_m", 0.02))
        self.distance_stop_lead_m = float(rospy.get_param("~distance_stop_lead_m", 0.03))
        self.turn_tolerance_deg = float(rospy.get_param("~turn_tolerance_deg", 1.0))
        self.turn_stop_lead_deg = float(rospy.get_param("~turn_stop_lead_deg", 2.0))
        self.action_timeout_s = float(
            rospy.get_param("~action_timeout_s", max(1.4, self.action_duration_s * 2.0))
        )
        self.feedback_wait_timeout_s = float(rospy.get_param("~feedback_wait_timeout_s", 1.0))
        self.allow_open_loop_fallback = bool(rospy.get_param("~allow_open_loop_fallback", False))

        self.body_vel_deadband = float(rospy.get_param("~body_vel_deadband", 0.01))
        self.gyro_deadband = float(rospy.get_param("~gyro_deadband", 0.01))
        self.gyro_bias_calib_s = float(rospy.get_param("~gyro_bias_calib_s", 0.5))
        self.gyro_bias_sample_max_abs = float(rospy.get_param("~gyro_bias_sample_max_abs", 0.10))

        self.use_estop = bool(rospy.get_param("~use_estop", False))
        self.estop_topic = str(rospy.get_param("~e_stop_topic", "/elevator/e_stop"))
        self.use_speed_limit = bool(rospy.get_param("~use_speed_limit", False))
        self.speed_limit_topic = str(rospy.get_param("~speed_limit_topic", "/elevator/speed_limit"))
        self.shutdown_on_stop = bool(rospy.get_param("~shutdown_on_stop", True))
        self.stop_hold_s = float(rospy.get_param("~stop_hold_s", 0.3))

        self._lock = threading.Lock()
        self._pending_actions: Deque[str] = deque()
        self._current_action: Optional[RunningAction] = None
        self._last_inference_generation = 0
        self._last_inference_request_time = 0.0
        self._estop = False
        self._speed_limit: Optional[float] = None
        self._stop_requested = False
        self._stop_time: Optional[float] = None

        self._gyro_bias = 0.0
        self._gyro_bias_ready = False
        self._gyro_bias_samples: List[float] = []
        self._gyro_bias_deadline = rospy.Time.now() + rospy.Duration(max(self.gyro_bias_calib_s, 0.0))
        self._camera_proc: Optional[subprocess.Popen] = None

        self._start_camera_if_configured()

        self._image_buffer = LatestImageBuffer(self.camera_topic)
        self._cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=20)
        self._state_pub = rospy.Publisher("~state", String, queue_size=10, latch=True)

        rospy.Subscriber(self.body_velocity_topic, Vector3Stamped, self._body_velocity_cb, queue_size=100)
        rospy.Subscriber(self.gyro_topic, Vector3Stamped, self._gyro_cb, queue_size=100)
        if self.use_estop:
            rospy.Subscriber(self.estop_topic, Bool, self._estop_cb, queue_size=10)
        if self.use_speed_limit:
            rospy.Subscriber(self.speed_limit_topic, Float32, self._speed_limit_cb, queue_size=10)

        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "[uninavid] loading model=%s instruction=%r",
            self.model_path,
            self.instruction,
        )
        self._agent = UniNaVid_Agent(self.model_path)
        self._agent.reset()
        self._worker = UniNaVidInferenceWorker(
            self._agent,
            self._image_buffer,
            self.instruction,
            max_frame_age_s=self.max_camera_age_s,
            max_actions=self.max_actions,
        )
        self._worker.start()

        rospy.loginfo(
            "[uninavid] ready camera=%s cmd=%s body_vel=%s gyro=%s "
            "forward=%.3fm@%.3fm/s turn=%.1fdeg@%.3frad/s",
            self.camera_topic,
            self.cmd_vel_topic,
            self.body_velocity_topic,
            self.gyro_topic,
            self.forward_distance_m,
            self.forward_speed_mps,
            self.turn_angle_deg,
            self.turn_yaw_rate_rps,
        )
        self._publish_state("ready")

    def _start_camera_if_configured(self) -> None:
        if not self.camera_launch_cmd:
            return
        rospy.loginfo("[uninavid] starting camera command: %s", self.camera_launch_cmd)
        self._camera_proc = subprocess.Popen(
            self.camera_launch_cmd,
            shell=True,
            executable="/bin/bash",
        )
        if self.camera_launch_startup_s > 0.0:
            rospy.sleep(self.camera_launch_startup_s)

    # ------------------------------------------------------------------ ROS callbacks
    def _body_velocity_cb(self, msg: Vector3Stamped) -> None:
        stamp = _stamp_or_now(msg.header.stamp)
        vx = float(msg.vector.x)
        if abs(vx) < self.body_vel_deadband:
            vx = 0.0

        with self._lock:
            action = self._current_action
            if action is None or action.name != "forward":
                return

            if action.last_feedback_stamp is not None:
                dt = (stamp - action.last_feedback_stamp).to_sec()
                if dt > 0.0:
                    action.progress += max(vx, 0.0) * dt
            action.last_feedback_stamp = stamp
            action.feedback_ready = True

    def _gyro_cb(self, msg: Vector3Stamped) -> None:
        stamp = _stamp_or_now(msg.header.stamp)
        gz = float(msg.vector.z)

        with self._lock:
            self._update_gyro_bias_locked(stamp, gz)
            if not self._gyro_bias_ready:
                return

            action = self._current_action
            if action is None or action.name not in {"left", "right"}:
                return

            if action.last_feedback_stamp is not None:
                dt = (stamp - action.last_feedback_stamp).to_sec()
                if dt > 0.0:
                    gz_unbiased = gz - self._gyro_bias
                    if abs(gz_unbiased) < self.gyro_deadband:
                        gz_unbiased = 0.0
                    directed_rate = action.direction * gz_unbiased
                    action.progress += max(directed_rate, 0.0) * dt * RAD2DEG
            action.last_feedback_stamp = stamp
            action.feedback_ready = True

    def _estop_cb(self, msg: Bool) -> None:
        with self._lock:
            self._estop = bool(msg.data)

    def _speed_limit_cb(self, msg: Float32) -> None:
        value = float(msg.data)
        with self._lock:
            self._speed_limit = value if value > 0.0 else None

    # ------------------------------------------------------------------ inference / control
    def _request_inference(self, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_inference_request_time < self.min_inference_request_period_s:
            return
        self._last_inference_request_time = now
        self._worker.request(reason)

    def _consume_inference(self) -> None:
        result = self._worker.latest_after(self._last_inference_generation)
        if result is None:
            return

        self._last_inference_generation = result.generation
        if result.error:
            rospy.logwarn("[uninavid] ignoring invalid inference result: %s", result.error)
            with self._lock:
                if self._current_action is None:
                    self._pending_actions.clear()
            return

        with self._lock:
            if result.actions and result.actions[0] == "stop":
                self._pending_actions.clear()
                self._current_action = None
                self._request_stop_locked("model predicted stop")
            else:
                self._pending_actions = deque(result.actions)

        self._publish_state(
            "inference_result",
            generation=result.generation,
            actions=result.actions,
            inference_s=round(result.inference_s, 3),
            frame_seq=result.frame_seq,
        )

    def _start_action_locked(self, name: str, now_ros: rospy.Time) -> None:
        if name == "forward":
            self._current_action = RunningAction(
                name=name,
                target=self.forward_distance_m,
                speed=self.forward_speed_mps,
                direction=1.0,
                start_ros=now_ros,
                start_monotonic=time.monotonic(),
            )
        elif name in {"left", "right"}:
            self._current_action = RunningAction(
                name=name,
                target=self.turn_angle_deg,
                speed=self.turn_yaw_rate_rps,
                direction=1.0 if name == "left" else -1.0,
                start_ros=now_ros,
                start_monotonic=time.monotonic(),
            )
        else:
            raise ValueError(f"unsupported action: {name}")

        rospy.loginfo("[uninavid] action start: %s", name)
        self._publish_state("action_start", action=name)

    def _step_control(self) -> Twist:
        cmd = Twist()
        request_reason: Optional[str] = None
        publish_state: Optional[Tuple[str, dict]] = None
        immediate_return = False

        with self._lock:
            now_ros = rospy.Time.now()
            now_mono = time.monotonic()

            if self._stop_requested:
                return cmd

            if self.use_estop and self._estop:
                return cmd

            if self._current_action is None:
                if self._pending_actions:
                    next_action = self._pending_actions.popleft()
                    if next_action == "stop":
                        self._request_stop_locked("stop action reached")
                        return cmd
                    self._start_action_locked(next_action, now_ros)
                else:
                    if (
                        not self._worker.is_busy()
                        and now_mono - self._last_inference_request_time >= self.idle_reinfer_period_s
                    ):
                        request_reason = "idle"
                    immediate_return = True

            action = self._current_action
            if not immediate_return and action is not None:
                elapsed = now_mono - action.start_monotonic
                if not action.feedback_ready:
                    if elapsed > self.feedback_wait_timeout_s:
                        if self.allow_open_loop_fallback:
                            action.feedback_ready = True
                            action.open_loop = True
                            rospy.logwarn(
                                "[uninavid] feedback timeout for %s, using open-loop fallback",
                                action.name,
                            )
                        else:
                            rospy.logerr(
                                "[uninavid] no feedback for %s within %.2fs, stopping",
                                action.name,
                                self.feedback_wait_timeout_s,
                            )
                            self._pending_actions.clear()
                            self._current_action = None
                            self._request_stop_locked("feedback timeout")
                            immediate_return = True
                    else:
                        immediate_return = True

                if not immediate_return and action.open_loop:
                    ratio = min(max(elapsed / max(self.action_duration_s, 1e-3), 0.0), 1.0)
                    action.progress = max(action.progress, action.target * ratio)

                if not immediate_return and self._action_done_locked(action):
                    completed_name = action.name
                    completed_progress = action.progress
                    self._current_action = None
                    request_reason = f"action_complete:{completed_name}"
                    publish_state = (
                        "action_done",
                        {
                            "action": completed_name,
                            "progress": round(completed_progress, 3),
                        },
                    )
                    rospy.loginfo(
                        "[uninavid] action done: %s progress=%.3f",
                        completed_name,
                        completed_progress,
                    )
                    immediate_return = True

                if not immediate_return and elapsed > self.action_timeout_s:
                    rospy.logwarn(
                        "[uninavid] action timeout: %s progress=%.3f target=%.3f",
                        action.name,
                        action.progress,
                        action.target,
                    )
                    self._current_action = None
                    self._pending_actions.clear()
                    request_reason = f"action_timeout:{action.name}"
                    immediate_return = True

                if not immediate_return:
                    if action.name == "forward":
                        cmd.linear.x = self._limited_forward_speed_locked(action.speed)
                    elif action.name in {"left", "right"}:
                        cmd.angular.z = action.direction * action.speed

            elif action is None:
                immediate_return = True

        if request_reason:
            self._request_inference(request_reason)
        if publish_state:
            state, payload = publish_state
            self._publish_state(state, **payload)
        return cmd

    def _action_done_locked(self, action: RunningAction) -> bool:
        if action.name == "forward":
            threshold = max(action.target - self.distance_stop_lead_m, 0.0)
            return action.progress + self.distance_tolerance_m >= threshold
        if action.name in {"left", "right"}:
            threshold = max(action.target - self.turn_stop_lead_deg, 0.0)
            return action.progress + self.turn_tolerance_deg >= threshold
        return False

    def _limited_forward_speed_locked(self, speed: float) -> float:
        value = max(float(speed), 0.0)
        if self.use_speed_limit and self._speed_limit is not None:
            value = min(value, self._speed_limit)
        return value

    def _request_stop_locked(self, reason: str) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        self._stop_time = time.monotonic()
        rospy.loginfo("[uninavid] stop requested: %s", reason)
        self._publish_state("stop", reason=reason)

    def _update_gyro_bias_locked(self, stamp: rospy.Time, gz: float) -> None:
        if self._gyro_bias_ready:
            return
        if self.gyro_bias_calib_s <= 0.0 or stamp >= self._gyro_bias_deadline:
            if self._gyro_bias_samples:
                self._gyro_bias = float(sum(self._gyro_bias_samples) / len(self._gyro_bias_samples))
            self._gyro_bias_ready = True
            rospy.loginfo(
                "[uninavid] gyro bias calibrated: %.6f rad/s from %d samples",
                self._gyro_bias,
                len(self._gyro_bias_samples),
            )
            return
        if abs(gz) <= self.gyro_bias_sample_max_abs:
            self._gyro_bias_samples.append(gz)

    # ------------------------------------------------------------------ lifecycle
    def _wait_for_camera(self) -> None:
        if self.camera_wait_timeout_s <= 0.0:
            return

        deadline = time.monotonic() + self.camera_wait_timeout_s
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self._image_buffer.latest(self.max_camera_age_s) is not None:
                rospy.loginfo("[uninavid] camera frame received")
                return
            self._cmd_pub.publish(Twist())
            rate.sleep()
        rospy.logwarn(
            "[uninavid] no camera frame within %.1fs, inference will retry",
            self.camera_wait_timeout_s,
        )

    def _publish_state(self, state: str, **payload) -> None:
        message = {"state": state, "stamp": rospy.Time.now().to_sec()}
        message.update(payload)
        self._state_pub.publish(String(data=json.dumps(message, ensure_ascii=True)))

    def _on_shutdown(self) -> None:
        try:
            self._worker.stop()
        except Exception:
            pass
        if self._camera_proc is not None and self._camera_proc.poll() is None:
            self._camera_proc.terminate()
            try:
                self._camera_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._camera_proc.kill()
        zero = Twist()
        for _ in range(3):
            self._cmd_pub.publish(zero)
            rospy.sleep(0.02)

    def run(self) -> int:
        self._wait_for_camera()
        self._request_inference("startup")

        rate = rospy.Rate(max(self.loop_rate_hz, 1.0))
        while not rospy.is_shutdown():
            self._consume_inference()
            cmd = self._step_control()
            self._cmd_pub.publish(cmd)

            with self._lock:
                should_shutdown = (
                    self._stop_requested
                    and self.shutdown_on_stop
                    and self._stop_time is not None
                    and time.monotonic() - self._stop_time >= self.stop_hold_s
                )
            if should_shutdown:
                rospy.signal_shutdown("Uni-NaVid stop action reached")
                break

            rate.sleep()
        self._cmd_pub.publish(Twist())
        return 0


def main() -> None:
    rospy.init_node("uninavid_realtime_controller")
    node = UniNaVidRealtimeNode()
    sys.exit(node.run())


if __name__ == "__main__":
    main()
