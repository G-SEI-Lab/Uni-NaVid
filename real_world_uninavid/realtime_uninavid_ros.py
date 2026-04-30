#!/usr/bin/env python3
"""Real-time Uni-NaVid ROS runner for robot dog control."""

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
from typing import Deque, List, Optional

import cv2  # type: ignore
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Vector3Stamped
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Bool, Float32, String


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_eval_uninavid import UniNaVid_Agent  # noqa: E402


RAD2DEG = 180.0 / math.pi
ACTION_RE = re.compile(r"\b(forward|left|right|stop)\b", re.IGNORECASE)

DEFAULT_MODEL_PATH = "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"
DEFAULT_INSTRUCTION = "move forward to the chair in the center of frame and stop."


@dataclass
class FrameSnapshot:
    image_bgr: object
    stamp: rospy.Time
    seq: int
    age_s: float


@dataclass
class InferenceRequest:
    reason: str
    min_seq: int


@dataclass
class InferenceResult:
    generation: int
    actions: List[str]
    raw_actions: str
    inference_s: float
    frame_seq: int
    frame_age_s: float
    request_reason: str
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
    """Only keep one latest frame to avoid memory buildup."""

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
            self._image_bgr = image_bgr
            self._stamp = _stamp_or_now(msg.header.stamp)
            self._monotonic = time.monotonic()

    def latest(self, max_age_s: float, min_seq: int = 0) -> Optional[FrameSnapshot]:
        with self._lock:
            if self._image_bgr is None:
                return None
            if self._seq <= min_seq:
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

    def latest_seq(self) -> int:
        with self._lock:
            return self._seq


class DebugRecorder:
    def __init__(
        self,
        enabled: bool,
        root_dir: str,
        keep_last_images: int,
        save_raw_images: bool,
    ) -> None:
        self.enabled = enabled
        self.keep_last_images = max(keep_last_images, 0)
        self.save_raw_images = save_raw_images
        self._lock = threading.Lock()
        self.root = Path(root_dir).expanduser()
        self.events_file = self.root / "events.jsonl"
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def record_inference(
        self,
        generation: int,
        request: InferenceRequest,
        frame: FrameSnapshot,
        raw_image,
        model_input_image,
        result: InferenceResult,
        instruction: str,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            raw_path = self.root / f"infer_{generation:06d}_raw.jpg"
            input_path = self.root / f"infer_{generation:06d}_input.jpg"
            if self.save_raw_images:
                cv2.imwrite(str(raw_path), raw_image)
            cv2.imwrite(str(input_path), model_input_image)
            if self.keep_last_images > 0:
                old_generation = generation - self.keep_last_images
                if old_generation > 0:
                    old_raw = self.root / f"infer_{old_generation:06d}_raw.jpg"
                    old_input = self.root / f"infer_{old_generation:06d}_input.jpg"
                    try:
                        if self.save_raw_images:
                            old_raw.unlink(missing_ok=True)
                        old_input.unlink(missing_ok=True)
                    except Exception:
                        pass
            event = {
                "type": "inference",
                "generation": generation,
                "request_reason": request.reason,
                "request_min_seq": request.min_seq,
                "frame_seq": frame.seq,
                "frame_age_s": frame.age_s,
                "instruction": instruction,
                "raw_actions": result.raw_actions,
                "actions": result.actions,
                "inference_s": result.inference_s,
                "error": result.error,
                "raw_image": raw_path.name if self.save_raw_images else None,
                "model_input_image": input_path.name,
                "ts": time.time(),
            }
            with self.events_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def record_event(self, event: dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            with self.events_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")


class UniNaVidInferenceWorker:
    def __init__(
        self,
        agent: UniNaVid_Agent,
        frame_buffer: LatestImageBuffer,
        instruction: str,
        debug: DebugRecorder,
        *,
        max_frame_age_s: float,
        max_actions: int,
        request_frame_wait_timeout_s: float,
        resize_before_model: bool,
        model_input_size: int,
        cache_reset_interval: int,
        empty_cuda_cache_every: int,
        feat_cache_max_frames: int,
        long_feat_cache_max_tokens: int,
        memory_log_interval: int,
    ) -> None:
        self._agent = agent
        self._frame_buffer = frame_buffer
        self._instruction = instruction
        self._debug = debug
        self._max_frame_age_s = max_frame_age_s
        self._max_actions = max_actions
        self._request_frame_wait_timeout_s = request_frame_wait_timeout_s
        self._resize_before_model = resize_before_model
        self._model_input_size = max(model_input_size, 1)
        self._cache_reset_interval = max(cache_reset_interval, 0)
        self._empty_cuda_cache_every = max(empty_cuda_cache_every, 0)
        self._feat_cache_max_frames = max(feat_cache_max_frames, 0)
        self._long_feat_cache_max_tokens = max(long_feat_cache_max_tokens, 0)
        self._memory_log_interval = max(memory_log_interval, 0)
        self._official_short_side, self._official_crop = self._resolve_official_image_size()

        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._busy = False
        self._latest: Optional[InferenceResult] = None
        self._generation = 0
        self._inference_count = 0
        self._pending_request: Optional[InferenceRequest] = None
        self._thread = threading.Thread(target=self._run, name="uninavid_inference", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._request_event.set()
        self._thread.join(timeout=2.0)

    def request(self, reason: str, min_seq: int) -> None:
        with self._lock:
            self._pending_request = InferenceRequest(reason=reason, min_seq=min_seq)
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

    def _next_request(self) -> Optional[InferenceRequest]:
        with self._lock:
            request = self._pending_request
            self._pending_request = None
        return request

    def _store_result(self, result: InferenceResult) -> None:
        with self._lock:
            self._latest = result

    def _wait_frame(self, request: InferenceRequest) -> Optional[FrameSnapshot]:
        deadline = time.monotonic() + max(self._request_frame_wait_timeout_s, 0.0)
        while not self._stop_event.is_set():
            frame = self._frame_buffer.latest(self._max_frame_age_s, min_seq=request.min_seq)
            if frame is not None:
                return frame
            if self._request_frame_wait_timeout_s > 0.0 and time.monotonic() >= deadline:
                return None
            time.sleep(0.01)
        return None

    def _prepare_model_input(self, image_bgr):
        # Debug preview only. Real inference input should follow official pipeline.
        if not self._resize_before_model:
            return self._official_preview(image_bgr)
        return cv2.resize(
            image_bgr,
            (self._model_input_size, self._model_input_size),
            interpolation=cv2.INTER_AREA,
        )

    def _resolve_official_image_size(self) -> tuple[int, int]:
        processor = self._agent.image_processor
        size_raw = getattr(processor, "size", 224)
        crop_raw = getattr(processor, "crop_size", 224)

        def to_int(value, default: int) -> int:
            if isinstance(value, int):
                return max(value, 1)
            if isinstance(value, (list, tuple)) and value:
                return max(int(value[0]), 1)
            if isinstance(value, dict):
                for key in ("shortest_edge", "height", "width"):
                    if key in value:
                        return max(int(value[key]), 1)
            try:
                return max(int(value), 1)
            except Exception:
                return default

        short_side = to_int(size_raw, 224)
        crop_side = to_int(crop_raw, short_side)
        return short_side, crop_side

    def _official_preview(self, image_bgr):
        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return image_bgr
        short = min(h, w)
        scale = float(self._official_short_side) / float(short)
        nh = max(int(round(h * scale)), 1)
        nw = max(int(round(w * scale)), 1)
        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

        crop = self._official_crop
        top = max((nh - crop) // 2, 0)
        left = max((nw - crop) // 2, 0)
        bottom = min(top + crop, nh)
        right = min(left + crop, nw)
        preview = resized[top:bottom, left:right]
        return preview

    def _maybe_reset_model_cache(self) -> None:
        if self._cache_reset_interval <= 0:
            return
        if self._inference_count > 0 and self._inference_count % self._cache_reset_interval == 0:
            rospy.logwarn(
                "[uninavid] periodic agent.reset() to bound online cache (count=%d)",
                self._inference_count,
            )
            self._agent.reset()

    def _maybe_trim_feat_cache(self) -> None:
        if self._feat_cache_max_frames <= 0:
            return
        try:
            core = self._agent.model.get_model()
            feat_cache = getattr(core, "feat_cache", None)
            if feat_cache is None:
                return
            if feat_cache.shape[0] > self._feat_cache_max_frames:
                core.feat_cache = feat_cache[-self._feat_cache_max_frames :, :, :].detach().clone()
                rospy.logwarn_throttle(
                    10.0,
                    "[uninavid] trimmed feat_cache to last %d frames",
                    self._feat_cache_max_frames,
                )
            long_feat_cache = getattr(core, "long_feat_cache", None)
            if (
                self._long_feat_cache_max_tokens > 0
                and long_feat_cache is not None
                and long_feat_cache.shape[0] > self._long_feat_cache_max_tokens
            ):
                core.long_feat_cache = long_feat_cache[-self._long_feat_cache_max_tokens :, :].detach().clone()
                rospy.logwarn_throttle(
                    10.0,
                    "[uninavid] trimmed long_feat_cache to last %d tokens",
                    self._long_feat_cache_max_tokens,
                )
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(5.0, "[uninavid] feat_cache trim failed: %s", exc)

    def _maybe_empty_cuda_cache(self) -> None:
        if self._empty_cuda_cache_every <= 0:
            return
        if self._inference_count > 0 and self._inference_count % self._empty_cuda_cache_every == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    @staticmethod
    def _rss_mb() -> Optional[float]:
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = float(line.split()[1])
                        return kb / 1024.0
        except Exception:
            return None
        return None

    def _log_memory_snapshot(self, frame_shape) -> None:
        if self._memory_log_interval <= 0:
            return
        if self._inference_count == 0 or self._inference_count % self._memory_log_interval != 0:
            return

        rss = self._rss_mb()
        cuda_alloc = cuda_reserved = None
        try:
            cuda_alloc = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            cuda_reserved = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        except Exception:
            pass

        feat_frames = long_tokens = None
        try:
            core = self._agent.model.get_model()
            feat_cache = getattr(core, "feat_cache", None)
            long_feat_cache = getattr(core, "long_feat_cache", None)
            feat_frames = int(feat_cache.shape[0]) if feat_cache is not None else 0
            long_tokens = int(long_feat_cache.shape[0]) if long_feat_cache is not None else 0
        except Exception:
            pass

        rospy.loginfo(
            "[uninavid] mem count=%d rss=%sMB cuda_alloc=%sMB cuda_reserved=%sMB "
            "feat_frames=%s long_tokens=%s frame_shape=%s",
            self._inference_count,
            "%.1f" % rss if rss is not None else "n/a",
            "%.1f" % cuda_alloc if cuda_alloc is not None else "n/a",
            "%.1f" % cuda_reserved if cuda_reserved is not None else "n/a",
            str(feat_frames) if feat_frames is not None else "n/a",
            str(long_tokens) if long_tokens is not None else "n/a",
            str(tuple(frame_shape)) if frame_shape is not None else "n/a",
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self._request_event.wait(timeout=0.1):
                continue
            if self._stop_event.is_set():
                break
            self._request_event.clear()

            request = self._next_request()
            if request is None:
                continue

            frame = self._wait_frame(request)
            if frame is None:
                rospy.logwarn(
                    "[uninavid] no suitable frame for inference request reason=%s min_seq=%d",
                    request.reason,
                    request.min_seq,
                )
                continue

            self._maybe_reset_model_cache()
            model_input_image = self._prepare_model_input(frame.image_bgr)
            inference_image = (
                model_input_image if self._resize_before_model else frame.image_bgr
            )

            self._set_busy(True)
            t0 = time.monotonic()
            rospy.loginfo(
                "[uninavid] inference start reason=%s frame_seq=%d age=%.3fs",
                request.reason,
                frame.seq,
                frame.age_s,
            )
            try:
                result_dict = self._agent.act(
                    {
                        "instruction": self._instruction,
                        "observations": inference_image,
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

            self._inference_count += 1
            self._generation += 1
            generation = self._generation

            result = InferenceResult(
                generation=generation,
                actions=actions,
                raw_actions=raw_actions,
                inference_s=dt,
                frame_seq=frame.seq,
                frame_age_s=frame.age_s,
                request_reason=request.reason,
                error=error,
            )
            self._store_result(result)
            self._maybe_trim_feat_cache()
            self._maybe_empty_cuda_cache()
            self._log_memory_snapshot(getattr(frame.image_bgr, "shape", None))

            self._debug.record_inference(
                generation=generation,
                request=request,
                frame=frame,
                raw_image=frame.image_bgr,
                model_input_image=model_input_image,
                result=result,
                instruction=self._instruction,
            )

            if error:
                rospy.logwarn(
                    "[uninavid] inference invalid gen=%d time=%.3fs error=%s",
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
        self.idle_reinfer_period_s = float(rospy.get_param("~idle_reinfer_period_s", 1.0))
        self.min_inference_request_period_s = float(
            rospy.get_param("~min_inference_request_period_s", 0.3)
        )
        self.request_frame_wait_timeout_s = float(
            rospy.get_param("~request_frame_wait_timeout_s", 2.0)
        )

        # Motion cadence:
        # action_period_s is the required end-to-end action frequency.
        # action_motion_s is used only to derive default speeds.
        self.action_period_s = float(rospy.get_param("~action_period_s", 1.0))
        self.action_motion_s = float(rospy.get_param("~action_motion_s", 0.7))
        self.post_action_settle_s = float(rospy.get_param("~post_action_settle_s", 0.2))
        self.forward_distance_m = float(rospy.get_param("~forward_distance_m", 0.50))
        self.turn_angle_deg = float(rospy.get_param("~turn_angle_deg", 30.0))
        self.forward_speed_mps = float(
            rospy.get_param(
                "~forward_speed_mps",
                self.forward_distance_m / max(self.action_motion_s, 1e-3),
            )
        )
        self.turn_yaw_rate_rps = float(
            rospy.get_param(
                "~turn_yaw_rate_rps",
                math.radians(self.turn_angle_deg) / max(self.action_motion_s, 1e-3),
            )
        )

        self.distance_tolerance_m = float(rospy.get_param("~distance_tolerance_m", 0.02))
        self.distance_stop_lead_m = float(rospy.get_param("~distance_stop_lead_m", 0.03))
        self.turn_tolerance_deg = float(rospy.get_param("~turn_tolerance_deg", 1.0))
        self.turn_stop_lead_deg = float(rospy.get_param("~turn_stop_lead_deg", 2.0))
        self.action_timeout_s = float(
            rospy.get_param("~action_timeout_s", max(1.6, self.action_period_s * 1.8))
        )
        self.feedback_wait_timeout_s = float(rospy.get_param("~feedback_wait_timeout_s", 1.0))
        self.allow_open_loop_fallback = bool(rospy.get_param("~allow_open_loop_fallback", False))
        self.turn_progress_use_abs_gyro = bool(rospy.get_param("~turn_progress_use_abs_gyro", True))
        self.max_consecutive_turn_actions = int(rospy.get_param("~max_consecutive_turn_actions", 3))

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

        self.resize_before_model = bool(rospy.get_param("~resize_before_model", False))
        self.model_input_size = int(rospy.get_param("~model_input_size", 224))

        self.cache_reset_interval = int(rospy.get_param("~cache_reset_interval", 0))
        self.empty_cuda_cache_every = int(rospy.get_param("~empty_cuda_cache_every", 0))
        self.feat_cache_max_frames = int(rospy.get_param("~feat_cache_max_frames", 64))
        self.long_feat_cache_max_tokens = int(rospy.get_param("~long_feat_cache_max_tokens", 256))
        self.memory_log_interval = int(rospy.get_param("~memory_log_interval", 1))

        self.debug_save_enabled = bool(rospy.get_param("~debug_save_enabled", True))
        default_debug_dir = str(REPO_ROOT / "real_world_uninavid" / "debug")
        self.debug_dir = str(rospy.get_param("~debug_dir", default_debug_dir))
        self.debug_keep_last_images = int(rospy.get_param("~debug_keep_last_images", 1000))
        self.debug_save_raw_images = bool(rospy.get_param("~debug_save_raw_images", False))

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

        self._settle_until = 0.0
        self._awaiting_post_action_inference = False
        self._post_action_inference_requested = False
        self._post_action_min_seq = 0
        self._next_action_not_before = 0.0
        self._consecutive_turn_actions = 0

        self._debug = DebugRecorder(
            self.debug_save_enabled,
            self.debug_dir,
            self.debug_keep_last_images,
            self.debug_save_raw_images,
        )
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
            self._debug,
            max_frame_age_s=self.max_camera_age_s,
            max_actions=self.max_actions,
            request_frame_wait_timeout_s=self.request_frame_wait_timeout_s,
            resize_before_model=self.resize_before_model,
            model_input_size=self.model_input_size,
            cache_reset_interval=self.cache_reset_interval,
            empty_cuda_cache_every=self.empty_cuda_cache_every,
            feat_cache_max_frames=self.feat_cache_max_frames,
            long_feat_cache_max_tokens=self.long_feat_cache_max_tokens,
            memory_log_interval=self.memory_log_interval,
        )
        self._worker.start()

        rospy.loginfo(
            "[uninavid] ready camera=%s cmd=%s body_vel=%s gyro=%s "
            "period=%.2fs motion=%.2fs settle=%.2fs "
            "forward=%.3fm@%.3fm/s turn=%.1fdeg@%.3frad/s resize=%s:%d debug=%s "
            "cache_reset_interval=%d feat_cache_max_frames=%d long_feat_cache_max_tokens=%d",
            self.camera_topic,
            self.cmd_vel_topic,
            self.body_velocity_topic,
            self.gyro_topic,
            self.action_period_s,
            self.action_motion_s,
            self.post_action_settle_s,
            self.forward_distance_m,
            self.forward_speed_mps,
            self.turn_angle_deg,
            self.turn_yaw_rate_rps,
            str(self.resize_before_model),
            self.model_input_size,
            str(self.debug_save_enabled),
            self.cache_reset_interval,
            self.feat_cache_max_frames,
            self.long_feat_cache_max_tokens,
        )
        if not self.resize_before_model:
            rospy.loginfo(
                "[uninavid] inference input uses raw camera frame; official image_processor handles resize+center-crop"
            )
        else:
            rospy.logwarn(
                "[uninavid] resize_before_model=true: this deviates from official offline_eval preprocessing"
            )
        self._publish_state("ready")
        if self.debug_save_enabled:
            rospy.loginfo("[uninavid] debug outputs: %s", self.debug_dir)

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
                    if self.turn_progress_use_abs_gyro:
                        delta_deg = abs(gz_unbiased) * dt * RAD2DEG
                    else:
                        directed = action.direction * gz_unbiased
                        delta_deg = max(directed, 0.0) * dt * RAD2DEG
                    action.progress += delta_deg
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
    def _request_inference(self, reason: str, min_seq: int) -> None:
        now = time.monotonic()
        if now - self._last_inference_request_time < self.min_inference_request_period_s:
            return
        self._last_inference_request_time = now
        self._worker.request(reason=reason, min_seq=min_seq)

    def _consume_inference(self) -> None:
        result = self._worker.latest_after(self._last_inference_generation)
        if result is None:
            return

        self._last_inference_generation = result.generation
        if result.error:
            rospy.logwarn("[uninavid] ignoring invalid inference result: %s", result.error)
            with self._lock:
                self._awaiting_post_action_inference = False
                self._post_action_inference_requested = False
                if self._current_action is None:
                    self._pending_actions.clear()
            return

        with self._lock:
            self._awaiting_post_action_inference = False
            self._post_action_inference_requested = False
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
            reason=result.request_reason,
        )

    def _start_action_locked(self, name: str, now_ros: rospy.Time, now_mono: float) -> None:
        if name in {"left", "right"}:
            if self._consecutive_turn_actions >= self.max_consecutive_turn_actions:
                self._request_stop_locked(
                    f"safety stop: consecutive turn actions >= {self.max_consecutive_turn_actions}"
                )
                return
            self._consecutive_turn_actions += 1
        else:
            self._consecutive_turn_actions = 0

        if name == "forward":
            action = RunningAction(
                name=name,
                target=self.forward_distance_m,
                speed=self.forward_speed_mps,
                direction=1.0,
                start_ros=now_ros,
                start_monotonic=now_mono,
            )
        elif name in {"left", "right"}:
            action = RunningAction(
                name=name,
                target=self.turn_angle_deg,
                speed=self.turn_yaw_rate_rps,
                direction=1.0 if name == "left" else -1.0,
                start_ros=now_ros,
                start_monotonic=now_mono,
            )
        else:
            raise ValueError(f"unsupported action: {name}")

        self._current_action = action
        rospy.loginfo("[uninavid] action start: %s", name)
        self._publish_state("action_start", action=name)
        self._debug.record_event({"type": "action_start", "action": name, "ts": time.time()})

    def _finish_action_locked(self, now_mono: float, action: RunningAction, reason: str) -> None:
        elapsed = max(now_mono - action.start_monotonic, 0.0)
        settle_floor = action.start_monotonic + self.action_period_s
        settle_extra = now_mono + self.post_action_settle_s
        self._settle_until = max(settle_floor, settle_extra)
        self._next_action_not_before = self._settle_until
        self._post_action_min_seq = self._image_buffer.latest_seq()
        self._awaiting_post_action_inference = True
        self._post_action_inference_requested = False
        self._pending_actions.clear()
        self._current_action = None
        rospy.loginfo(
            "[uninavid] action done=%s reason=%s elapsed=%.3fs settle_until=%.3fs",
            action.name,
            reason,
            elapsed,
            self._settle_until - now_mono,
        )
        self._publish_state(
            "action_done",
            action=action.name,
            progress=round(action.progress, 3),
            elapsed_s=round(elapsed, 3),
            reason=reason,
        )
        self._debug.record_event(
            {
                "type": "action_done",
                "action": action.name,
                "reason": reason,
                "progress": action.progress,
                "elapsed_s": elapsed,
                "ts": time.time(),
            }
        )

    def _step_control(self) -> Twist:
        cmd = Twist()
        with self._lock:
            now_ros = rospy.Time.now()
            now_mono = time.monotonic()

            if self._stop_requested:
                return cmd
            if self.use_estop and self._estop:
                return cmd

            action = self._current_action
            if action is not None:
                elapsed = now_mono - action.start_monotonic

                if not action.feedback_ready:
                    if elapsed > self.feedback_wait_timeout_s:
                        if self.allow_open_loop_fallback:
                            action.feedback_ready = True
                            action.open_loop = True
                            rospy.logwarn(
                                "[uninavid] feedback timeout for %s, use open-loop fallback",
                                action.name,
                            )
                        else:
                            self._pending_actions.clear()
                            self._current_action = None
                            self._request_stop_locked("feedback timeout")
                            return cmd
                    else:
                        return cmd

                if action.open_loop:
                    ratio = min(max(elapsed / max(self.action_motion_s, 1e-3), 0.0), 1.0)
                    action.progress = max(action.progress, action.target * ratio)

                if self._action_done_locked(action):
                    self._finish_action_locked(now_mono, action, reason="target_reached")
                    return cmd

                if elapsed > self.action_timeout_s:
                    self._finish_action_locked(now_mono, action, reason="timeout")
                    return cmd

                if action.name == "forward":
                    cmd.linear.x = self._limited_forward_speed_locked(action.speed)
                else:
                    cmd.angular.z = action.direction * action.speed
                return cmd

            # No running action. Keep zero command during settle or while waiting for fresh post-action inference.
            if now_mono < self._settle_until:
                return cmd

            if self._awaiting_post_action_inference:
                if not self._post_action_inference_requested and not self._worker.is_busy():
                    self._request_inference(
                        reason="post_action_settle",
                        min_seq=self._post_action_min_seq,
                    )
                    self._post_action_inference_requested = True
                return cmd

            if self._pending_actions and now_mono >= self._next_action_not_before:
                next_action = self._pending_actions.popleft()
                if next_action == "stop":
                    self._request_stop_locked("stop action reached")
                    return cmd
                self._start_action_locked(next_action, now_ros, now_mono)
                return cmd

            # Idle: periodically infer on latest fresh frame.
            if (
                not self._worker.is_busy()
                and now_mono - self._last_inference_request_time >= self.idle_reinfer_period_s
            ):
                self._request_inference(
                    reason="idle",
                    min_seq=self._image_buffer.latest_seq(),
                )
            return cmd

    def _action_done_locked(self, action: RunningAction) -> bool:
        if action.name == "forward":
            threshold = max(action.target - self.distance_stop_lead_m, 0.0)
            return action.progress + self.distance_tolerance_m >= threshold
        threshold = max(action.target - self.turn_stop_lead_deg, 0.0)
        return action.progress + self.turn_tolerance_deg >= threshold

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
        self._debug.record_event({"type": "stop", "reason": reason, "ts": time.time()})

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
        self._request_inference(reason="startup", min_seq=self._image_buffer.latest_seq())

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
