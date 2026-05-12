#!/usr/bin/env python3
"""Instruction-driven pipelined Uni-NaVid ROS runner for robot dog control."""

from __future__ import annotations

import json
import math
import os
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
import imageio
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Vector3Stamped
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Bool, Float32, String


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_eval_uninavid import (  # noqa: E402
    UniNaVid_Agent,
    draw_traj_arrows_fpv,
    get_sorted_images,
    get_traj_data,
)


RAD2DEG = 180.0 / math.pi
ACTION_RE = re.compile(r"\b(forward|left|right|stop)\b", re.IGNORECASE)

DEFAULT_MODEL_PATH = "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"
DEFAULT_INSTRUCTION = "Turn back to find the traffic cone, then move to it and stop."

try:
    cv2.setNumThreads(1)
except Exception:
    pass


@dataclass
class FrameSnapshot:
    image_bgr: object
    stamp: rospy.Time
    seq: int
    age_s: float
    captured_monotonic: float


@dataclass
class InferenceRequest:
    reason: str
    min_seq: int
    instruction: str
    task_id: int
    frame: Optional[FrameSnapshot] = None
    action_anchor: int = 0
    request_monotonic: float = 0.0
    reset_agent: bool = False


@dataclass
class InferenceResult:
    generation: int
    task_id: int
    actions: List[str]
    raw_actions: str
    inference_s: float
    frame_seq: int
    frame_age_s: float
    request_reason: str
    action_anchor: int
    request_monotonic: float
    result_monotonic: float
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


def _resolve_output_dir(value: str) -> Path:
    if not value.strip():
        return REPO_ROOT / "real_world_uninavid" / "offline_eval_output"
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _run_offline_eval_mode(model_path: str, recording_dir: str, output_dir: str) -> int:
    recording_path = Path(recording_dir).expanduser()
    if not recording_path.is_absolute():
        recording_path = (REPO_ROOT / recording_path).resolve()
    output_path = _resolve_output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    agent = UniNaVid_Agent(model_path)
    agent.reset()

    images = get_sorted_images(str(recording_path))
    instruction = get_traj_data(str(recording_path))
    print(f"Total {len(images)} images")
    if not images:
        raise RuntimeError(f"no .jpg images found under {recording_path / 'images'}")

    result_vis_list = []
    result_jsonl = output_path / "result.jsonl"
    with result_jsonl.open("w", encoding="utf-8") as f:
        for step_count, image in enumerate(images, start=1):
            t_s = time.time()
            result = agent.act({"instruction": instruction, "observations": image})
            inference_s = time.time() - t_s

            actions = list(result.get("actions", []))
            print("step", step_count, "inference time", inference_s)

            vis = draw_traj_arrows_fpv(image, actions, arrow_len=20)
            result_vis_list.append(vis)
            f.write(
                json.dumps(
                    {
                        "step": step_count,
                        "inference_s": inference_s,
                        "actions": actions,
                        "raw_actions": " ".join(str(action) for action in actions),
                        "path": result.get("path"),
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    gif_path = output_path / "result.gif"
    imageio.mimsave(str(gif_path), result_vis_list)
    print(f"Saved {gif_path}")
    print(f"Saved {result_jsonl}")
    return 0


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

    def __init__(self, topic: str, decode_max_hz: float = 0.0) -> None:
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._image_bgr = None
        self._stamp = rospy.Time(0.0)
        self._seq = 0
        self._monotonic = 0.0
        self._decode_min_period_s = 1.0 / decode_max_hz if decode_max_hz > 0.0 else 0.0
        self._last_decode_monotonic = 0.0
        self._sub = rospy.Subscriber(
            topic,
            RosImage,
            self._image_cb,
            queue_size=1,
            buff_size=2**24,
        )

    def _image_cb(self, msg: RosImage) -> None:
        now = time.monotonic()
        if (
            self._decode_min_period_s > 0.0
            and now - self._last_decode_monotonic < self._decode_min_period_s
        ):
            return
        self._last_decode_monotonic = now

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
                captured_monotonic=self._monotonic,
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
        save_images: bool,
        save_raw_images: bool,
        image_interval: int,
        image_max_count: int,
        fsync_events: bool,
    ) -> None:
        self.enabled = enabled
        self.keep_last_images = max(keep_last_images, 0)
        self.save_images = save_images
        self.save_raw_images = save_raw_images
        self.image_interval = max(image_interval, 1)
        self.image_max_count = max(image_max_count, 0)
        self.fsync_events = fsync_events
        self._lock = threading.Lock()
        self.root = Path(root_dir).expanduser()
        self.events_file = self.root / "events.jsonl"
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_jpeg_atomic(path: Path, image, quality: int = 90) -> bool:
        if image is None or not hasattr(image, "shape"):
            return False
        try:
            if len(image.shape) < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
                return False
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
            )
            if not ok:
                return False
            tmp_path = path.with_name(f"{path.name}.tmp")
            tmp_path.write_bytes(encoded.tobytes())
            tmp_path.replace(path)
            return True
        except Exception:
            try:
                path.with_name(f"{path.name}.tmp").unlink(missing_ok=True)
            except Exception:
                pass
            return False

    def _write_event_locked(self, event: dict) -> None:
        with self.events_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            if self.fsync_events:
                f.flush()
                os.fsync(f.fileno())

    def record_inference(
        self,
        generation: int,
        request: InferenceRequest,
        frame: FrameSnapshot,
        raw_image,
        model_input_image,
        result: InferenceResult,
        instruction: str,
        memory: Optional[dict] = None,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            raw_path = self.root / f"infer_{generation:06d}_raw.jpg"
            input_path = self.root / f"infer_{generation:06d}_input.jpg"
            raw_name = None
            input_name = None
            image_write_ok = None
            should_save_images = (
                self.save_images
                and generation % self.image_interval == 0
                and (self.image_max_count <= 0 or generation <= self.image_max_count)
            )
            if should_save_images:
                image_write_ok = True
                if self.save_raw_images:
                    if self._write_jpeg_atomic(raw_path, raw_image):
                        raw_name = raw_path.name
                    else:
                        image_write_ok = False
                if self._write_jpeg_atomic(input_path, model_input_image):
                    input_name = input_path.name
                else:
                    image_write_ok = False
            if should_save_images and self.keep_last_images > 0:
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
                "request_task_id": request.task_id,
                "request_action_anchor": request.action_anchor,
                "request_monotonic": request.request_monotonic,
                "frame_seq": frame.seq,
                "frame_age_s": frame.age_s,
                "frame_captured_monotonic": frame.captured_monotonic,
                "instruction": instruction,
                "task_id": result.task_id,
                "raw_actions": result.raw_actions,
                "actions": result.actions,
                "inference_s": result.inference_s,
                "request_to_result_s": result.result_monotonic - result.request_monotonic,
                "result_action_anchor": result.action_anchor,
                "result_monotonic": result.result_monotonic,
                "error": result.error,
                "raw_image": raw_name,
                "model_input_image": input_name,
                "image_write_ok": image_write_ok,
                "memory": memory,
                "ts": time.time(),
            }
            self._write_event_locked(event)

    def record_event(self, event: dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._write_event_locked(event)

    def record_config(self, config: dict) -> None:
        if not self.enabled:
            return
        self.record_event({"type": "config", "config": config, "ts": time.time()})


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

    def request(
        self,
        reason: str,
        min_seq: int,
        *,
        instruction: str,
        task_id: int,
        frame: Optional[FrameSnapshot] = None,
        action_anchor: int = 0,
        reset_agent: bool = False,
    ) -> None:
        with self._lock:
            self._pending_request = InferenceRequest(
                reason=reason,
                min_seq=min_seq,
                instruction=instruction,
                task_id=max(int(task_id), 0),
                frame=frame,
                action_anchor=max(int(action_anchor), 0),
                request_monotonic=time.monotonic(),
                reset_agent=reset_agent,
            )
        self._request_event.set()

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def has_pending(self) -> bool:
        with self._lock:
            return self._pending_request is not None

    def has_work(self) -> bool:
        with self._lock:
            return self._busy or self._pending_request is not None

    def cancel_pending(self, task_id: Optional[int] = None) -> bool:
        with self._lock:
            if self._pending_request is None:
                return False
            if task_id is not None and self._pending_request.task_id != task_id:
                return False
            self._pending_request = None
            return True

    def latest_after(self, generation: int) -> Optional[InferenceResult]:
        with self._lock:
            if self._latest is not None and self._latest.generation > generation:
                return self._latest
        return None

    def latest_generation(self) -> int:
        with self._lock:
            return self._generation

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
        if request.frame is not None:
            return request.frame

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

    def _memory_snapshot(self, frame_shape) -> dict:
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

        return {
            "count": self._inference_count,
            "rss_mb": round(rss, 1) if rss is not None else None,
            "cuda_alloc_mb": round(cuda_alloc, 1) if cuda_alloc is not None else None,
            "cuda_reserved_mb": round(cuda_reserved, 1) if cuda_reserved is not None else None,
            "feat_frames": feat_frames,
            "long_tokens": long_tokens,
            "frame_shape": list(frame_shape) if frame_shape is not None else None,
        }

    def _log_memory_snapshot(self, snapshot: dict) -> None:
        if self._memory_log_interval <= 0:
            return
        if self._inference_count == 0 or self._inference_count % self._memory_log_interval != 0:
            return

        rospy.loginfo(
            "[uninavid] mem count=%d rss=%sMB cuda_alloc=%sMB cuda_reserved=%sMB "
            "feat_frames=%s long_tokens=%s frame_shape=%s",
            int(snapshot.get("count", 0)),
            str(snapshot.get("rss_mb")) if snapshot.get("rss_mb") is not None else "n/a",
            str(snapshot.get("cuda_alloc_mb")) if snapshot.get("cuda_alloc_mb") is not None else "n/a",
            str(snapshot.get("cuda_reserved_mb")) if snapshot.get("cuda_reserved_mb") is not None else "n/a",
            str(snapshot.get("feat_frames")) if snapshot.get("feat_frames") is not None else "n/a",
            str(snapshot.get("long_tokens")) if snapshot.get("long_tokens") is not None else "n/a",
            str(tuple(snapshot["frame_shape"])) if snapshot.get("frame_shape") is not None else "n/a",
        )

    def _record_inference_start(self, request: InferenceRequest, frame: FrameSnapshot) -> None:
        self._debug.record_event(
            {
                "type": "inference_start",
                "request_reason": request.reason,
                "request_min_seq": request.min_seq,
                "request_task_id": request.task_id,
                "request_action_anchor": request.action_anchor,
                "request_monotonic": request.request_monotonic,
                "request_wait_s": time.monotonic() - request.request_monotonic,
                "next_generation": self._generation + 1,
                "frame_seq": frame.seq,
                "frame_age_s": frame.age_s,
                "frame_captured_monotonic": frame.captured_monotonic,
                "frame_shape": list(frame.image_bgr.shape),
                "memory": self._memory_snapshot(getattr(frame.image_bgr, "shape", None)),
                "ts": time.time(),
            }
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

            if request.reset_agent:
                rospy.loginfo(
                    "[uninavid] reset online inference cache for task_id=%d",
                    request.task_id,
                )
                self._agent.reset()

            frame = self._wait_frame(request)
            if frame is None:
                rospy.logwarn(
                    "[uninavid] no suitable frame for inference request reason=%s task_id=%d min_seq=%d",
                    request.reason,
                    request.task_id,
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
                "[uninavid] inference start reason=%s task_id=%d frame_seq=%d age=%.3fs anchor=%d",
                request.reason,
                request.task_id,
                frame.seq,
                frame.age_s,
                request.action_anchor,
            )
            self._record_inference_start(request, frame)
            try:
                result_dict = self._agent.act(
                    {
                        "instruction": request.instruction,
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
                task_id=request.task_id,
                actions=actions,
                raw_actions=raw_actions,
                inference_s=dt,
                frame_seq=frame.seq,
                frame_age_s=frame.age_s,
                request_reason=request.reason,
                action_anchor=request.action_anchor,
                request_monotonic=request.request_monotonic,
                result_monotonic=time.monotonic(),
                error=error,
            )
            self._store_result(result)
            self._maybe_trim_feat_cache()
            self._maybe_empty_cuda_cache()
            memory_snapshot = self._memory_snapshot(getattr(frame.image_bgr, "shape", None))
            self._log_memory_snapshot(memory_snapshot)

            self._debug.record_inference(
                generation=generation,
                request=request,
                frame=frame,
                raw_image=frame.image_bgr,
                model_input_image=model_input_image,
                result=result,
                instruction=request.instruction,
                memory=memory_snapshot,
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
                    "[uninavid] inference done gen=%d task_id=%d time=%.3fs actions=%s",
                    generation,
                    request.task_id,
                    dt,
                    " ".join(actions),
                )


class UniNaVidInstructionPipelineNode:
    def __init__(self) -> None:
        self.model_path = _resolve_repo_path(str(rospy.get_param("~model_path", DEFAULT_MODEL_PATH)))
        self.instruction = ""
        self.instruction_topic = str(rospy.get_param("~instruction_topic", "/uninavid/instruction"))
        self.cancel_topic = str(rospy.get_param("~cancel_topic", "/uninavid/cancel"))

        self.camera_topic = str(rospy.get_param("~camera_topic", "/camera_down/color/image_raw"))
        self.camera_decode_max_hz = float(rospy.get_param("~camera_decode_max_hz", 5.0))
        self.camera_launch_cmd = str(rospy.get_param("~camera_launch_cmd", "")).strip()
        self.camera_launch_startup_s = float(rospy.get_param("~camera_launch_startup_s", 2.0))
        self.cmd_vel_topic = str(rospy.get_param("~cmd_vel_topic", "/cmd_vel"))
        self.body_velocity_topic = str(rospy.get_param("~body_velocity_topic", "/zsl/body_velocity"))
        self.gyro_topic = str(rospy.get_param("~gyro_topic", "/zsl/body_gyro"))

        self.loop_rate_hz = float(rospy.get_param("~loop_rate_hz", 30.0))
        self.max_actions = int(rospy.get_param("~max_actions", 4))
        self.max_camera_age_s = float(rospy.get_param("~max_camera_age_s", 1.0))
        self.camera_wait_timeout_s = float(rospy.get_param("~camera_wait_timeout_s", 10.0))
        self.idle_reinfer_period_s = float(rospy.get_param("~idle_reinfer_period_s", 1.8))
        self.min_inference_request_period_s = float(
            rospy.get_param("~min_inference_request_period_s", 0.0)
        )
        self.request_frame_wait_timeout_s = float(
            rospy.get_param("~request_frame_wait_timeout_s", 2.0)
        )
        self.still_frame_wait_timeout_s = float(
            rospy.get_param("~still_frame_wait_timeout_s", 0.25)
        )

        # Motion cadence:
        # action_period_s is the required end-to-end action frequency.
        # action_motion_s is used only to derive default speeds.
        self.action_period_s = float(rospy.get_param("~action_period_s", 1.0))
        self.action_motion_s = float(rospy.get_param("~action_motion_s", 1.0))
        self.post_action_settle_s = float(rospy.get_param("~post_action_settle_s", 0.2))
        self.forward_distance_m = float(rospy.get_param("~forward_distance_m", 0.25))
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

        self.distance_tolerance_m = float(rospy.get_param("~distance_tolerance_m", 0.0))
        self.distance_stop_lead_m = float(rospy.get_param("~distance_stop_lead_m", 0.0))
        self.turn_tolerance_deg = float(rospy.get_param("~turn_tolerance_deg", 0.0))
        self.turn_stop_lead_deg = float(rospy.get_param("~turn_stop_lead_deg", 0.0))
        self.action_timeout_s = float(
            rospy.get_param("~action_timeout_s", max(1.6, self.action_period_s * 1.8))
        )
        self.feedback_wait_timeout_s = float(rospy.get_param("~feedback_wait_timeout_s", 1.0))
        self.allow_open_loop_fallback = bool(rospy.get_param("~allow_open_loop_fallback", False))
        self.turn_progress_use_abs_gyro = bool(rospy.get_param("~turn_progress_use_abs_gyro", True))

        self.body_vel_deadband = float(rospy.get_param("~body_vel_deadband", 0.01))
        self.gyro_deadband = float(rospy.get_param("~gyro_deadband", 0.01))
        self.gyro_bias_calib_s = float(rospy.get_param("~gyro_bias_calib_s", 0.5))
        self.gyro_bias_sample_max_abs = float(rospy.get_param("~gyro_bias_sample_max_abs", 0.10))

        self.use_estop = bool(rospy.get_param("~use_estop", False))
        self.estop_topic = str(rospy.get_param("~e_stop_topic", "/elevator/e_stop"))
        self.use_speed_limit = bool(rospy.get_param("~use_speed_limit", False))
        self.speed_limit_topic = str(rospy.get_param("~speed_limit_topic", "/elevator/speed_limit"))
        self.shutdown_on_stop = bool(rospy.get_param("~shutdown_on_stop", False))
        self.stop_hold_s = float(rospy.get_param("~stop_hold_s", 0.3))
        self.inference_only = bool(rospy.get_param("~inference_only", False))
        self.inference_only_period_s = float(rospy.get_param("~inference_only_period_s", 1.0))
        self.max_runtime_s = float(rospy.get_param("~max_runtime_s", 0.0))
        self.max_inferences = int(rospy.get_param("~max_inferences", 0))

        self.resize_before_model = bool(rospy.get_param("~resize_before_model", False))
        self.model_input_size = int(rospy.get_param("~model_input_size", 224))

        self.cache_reset_interval = int(rospy.get_param("~cache_reset_interval", 0))
        self.empty_cuda_cache_every = int(rospy.get_param("~empty_cuda_cache_every", 0))
        self.feat_cache_max_frames = int(rospy.get_param("~feat_cache_max_frames", 0))
        self.long_feat_cache_max_tokens = int(rospy.get_param("~long_feat_cache_max_tokens", 0))
        self.memory_log_interval = int(rospy.get_param("~memory_log_interval", 1))

        self.debug_save_enabled = bool(rospy.get_param("~debug_save_enabled", True))
        default_debug_dir = str(REPO_ROOT / "real_world_uninavid" / "debug")
        self.debug_dir = str(rospy.get_param("~debug_dir", default_debug_dir))
        self.debug_keep_last_images = int(rospy.get_param("~debug_keep_last_images", 1000))
        self.debug_save_images = bool(rospy.get_param("~debug_save_images", False))
        self.debug_save_raw_images = bool(rospy.get_param("~debug_save_raw_images", False))
        self.debug_image_interval = int(rospy.get_param("~debug_image_interval", 1))
        self.debug_image_max_count = int(rospy.get_param("~debug_image_max_count", 16))
        self.debug_fsync_events = bool(rospy.get_param("~debug_fsync_events", True))

        self._lock = threading.Lock()
        self._pending_actions: Deque[str] = deque()
        self._current_action: Optional[RunningAction] = None
        self._task_id = 0
        self._task_active = False
        self._task_start_monotonic: Optional[float] = None
        self._completed_action_count = 0
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
        self._settle_needs_inference = False
        self._settle_capture_min_seq = 0
        self._settle_frame_deadline = 0.0
        self._next_action_not_before = 0.0
        self._turn_run_abs_deg = 0.0
        self._total_abs_turn_deg = 0.0

        self._debug = DebugRecorder(
            self.debug_save_enabled,
            self.debug_dir,
            self.debug_keep_last_images,
            self.debug_save_images,
            self.debug_save_raw_images,
            self.debug_image_interval,
            self.debug_image_max_count,
            self.debug_fsync_events,
        )
        self._start_camera_if_configured()

        self._debug.record_config(
            {
                "model_path": self.model_path,
                "instruction_topic": self.instruction_topic,
                "cancel_topic": self.cancel_topic,
                "camera_topic": self.camera_topic,
                "camera_decode_max_hz": self.camera_decode_max_hz,
                "cmd_vel_topic": self.cmd_vel_topic,
                "body_velocity_topic": self.body_velocity_topic,
                "gyro_topic": self.gyro_topic,
                "inference_only": self.inference_only,
                "inference_only_period_s": self.inference_only_period_s,
                "max_runtime_s": self.max_runtime_s,
                "max_inferences": self.max_inferences,
                "still_frame_wait_timeout_s": self.still_frame_wait_timeout_s,
                "action_period_s": self.action_period_s,
                "action_motion_s": self.action_motion_s,
                "forward_distance_m": self.forward_distance_m,
                "turn_angle_deg": self.turn_angle_deg,
                "resize_before_model": self.resize_before_model,
                "cache_reset_interval": self.cache_reset_interval,
                "feat_cache_max_frames": self.feat_cache_max_frames,
                "long_feat_cache_max_tokens": self.long_feat_cache_max_tokens,
                "debug_save_images": self.debug_save_images,
                "debug_save_raw_images": self.debug_save_raw_images,
                "instruction_driven": True,
            }
        )

        self._image_buffer = LatestImageBuffer(self.camera_topic, self.camera_decode_max_hz)
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
            "[uninavid] loading model=%s; waiting for instructions on %s",
            self.model_path,
            self.instruction_topic,
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
        rospy.Subscriber(self.instruction_topic, String, self._instruction_cb, queue_size=10)
        rospy.Subscriber(self.cancel_topic, Bool, self._cancel_cb, queue_size=10)

        rospy.loginfo(
            "[uninavid] ready camera=%s decode_max_hz=%.2f cmd=%s body_vel=%s gyro=%s "
            "pipeline=true instruction_topic=%s cancel_topic=%s period=%.2fs motion=%.2fs settle=%.2fs still_wait=%.2fs "
            "forward=%.3fm@%.3fm/s turn=%.1fdeg@%.3frad/s resize=%s:%d debug=%s inference_only=%s "
            "cache_reset_interval=%d feat_cache_max_frames=%d long_feat_cache_max_tokens=%d",
            self.camera_topic,
            self.camera_decode_max_hz,
            self.cmd_vel_topic,
            self.body_velocity_topic,
            self.gyro_topic,
            self.instruction_topic,
            self.cancel_topic,
            self.action_period_s,
            self.action_motion_s,
            self.post_action_settle_s,
            self.still_frame_wait_timeout_s,
            self.forward_distance_m,
            self.forward_speed_mps,
            self.turn_angle_deg,
            self.turn_yaw_rate_rps,
            str(self.resize_before_model),
            self.model_input_size,
            str(self.debug_save_enabled),
            str(self.inference_only),
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
        self._publish_state(
            "waiting_for_instruction",
            instruction_topic=self.instruction_topic,
            cancel_topic=self.cancel_topic,
        )
        if self.debug_save_enabled:
            rospy.loginfo(
                "[uninavid] debug outputs: %s images=%s raw_images=%s image_interval=%d "
                "image_max_count=%d fsync_events=%s",
                self.debug_dir,
                str(self.debug_save_images),
                str(self.debug_save_raw_images),
                self.debug_image_interval,
                self.debug_image_max_count,
                str(self.debug_fsync_events),
            )

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

    def _instruction_cb(self, msg: String) -> None:
        instruction = str(msg.data or "").strip()
        if not instruction:
            rospy.logwarn("[uninavid] ignoring empty instruction")
            return
        self._start_new_task(instruction)

    def _cancel_cb(self, msg: Bool) -> None:
        if not bool(msg.data):
            return
        self._cancel_current_task("cancel signal received")

    def _publish_zero_command(self, repeats: int = 3) -> None:
        zero = Twist()
        for _ in range(max(repeats, 1)):
            self._cmd_pub.publish(zero)
            rospy.sleep(0.02)

    def _reset_runtime_state_locked(self) -> None:
        self._pending_actions.clear()
        self._current_action = None
        self._completed_action_count = 0
        self._last_inference_generation = self._worker.latest_generation()
        self._last_inference_request_time = 0.0
        self._stop_requested = False
        self._stop_time = None
        self._settle_until = 0.0
        self._settle_needs_inference = False
        self._settle_capture_min_seq = 0
        self._settle_frame_deadline = 0.0
        self._next_action_not_before = 0.0
        self._turn_run_abs_deg = 0.0
        self._total_abs_turn_deg = 0.0

    def _cancel_current_task(self, reason: str) -> None:
        self._publish_zero_command()
        with self._lock:
            was_active = self._task_active or self._current_action is not None or bool(self._pending_actions)
            task_id = self._task_id
            self._task_active = False
            self.instruction = ""
            self._reset_runtime_state_locked()
            self._task_start_monotonic = None

        canceled_pending = self._worker.cancel_pending(task_id)
        rospy.loginfo(
            "[uninavid] task cancel requested active=%s task_id=%d pending_request_canceled=%s reason=%s",
            str(was_active),
            task_id,
            str(canceled_pending),
            reason,
        )
        self._publish_state(
            "task_canceled",
            task_id=task_id,
            was_active=was_active,
            pending_request_canceled=canceled_pending,
            reason=reason,
        )
        self._publish_state(
            "waiting_for_instruction",
            instruction_topic=self.instruction_topic,
            cancel_topic=self.cancel_topic,
        )
        self._debug.record_event(
            {
                "type": "task_canceled",
                "task_id": task_id,
                "was_active": was_active,
                "pending_request_canceled": canceled_pending,
                "reason": reason,
                "ts": time.time(),
            }
        )

    def _start_new_task(self, instruction: str) -> None:
        self._publish_zero_command()
        with self._lock:
            self._task_id += 1
            task_id = self._task_id
            self.instruction = instruction
            self._task_active = True
            self._task_start_monotonic = time.monotonic()
            self._reset_runtime_state_locked()
            min_seq = self._image_buffer.latest_seq()

        rospy.loginfo("[uninavid] new task_id=%d instruction=%r", task_id, instruction)
        self._publish_state("task_received", task_id=task_id, instruction=instruction)
        self._debug.record_event(
            {
                "type": "task_received",
                "task_id": task_id,
                "instruction": instruction,
                "request_min_seq": min_seq,
                "worker_busy": self._worker.is_busy(),
                "worker_pending": self._worker.has_pending(),
                "ts": time.time(),
            }
        )

        requested = self._request_inference(
            reason="new_instruction",
            min_seq=min_seq,
            action_anchor=0,
            task_id=task_id,
            instruction=instruction,
            reset_agent=True,
            capture_frame=False,
        )
        self._publish_state(
            "task_started" if requested else "task_waiting_for_frame",
            task_id=task_id,
            instruction=instruction,
            request_min_seq=min_seq,
        )

    # ------------------------------------------------------------------ inference / control
    def _request_inference(
        self,
        reason: str,
        min_seq: int = 0,
        *,
        frame: Optional[FrameSnapshot] = None,
        action_anchor: Optional[int] = None,
        task_id: Optional[int] = None,
        instruction: Optional[str] = None,
        reset_agent: bool = False,
        capture_frame: bool = True,
    ) -> bool:
        if not self._task_active:
            return False
        request_task_id = self._task_id if task_id is None else int(task_id)
        if request_task_id != self._task_id:
            return False
        request_instruction = self.instruction if instruction is None else instruction
        if not request_instruction:
            return False

        now = time.monotonic()
        if now - self._last_inference_request_time < self.min_inference_request_period_s:
            return False
        if capture_frame and frame is None:
            frame = self._image_buffer.latest(self.max_camera_age_s, min_seq=min_seq)
        if capture_frame and frame is None:
            return False

        anchor = self._completed_action_count if action_anchor is None else int(action_anchor)
        self._last_inference_request_time = now
        self._worker.request(
            reason=reason,
            min_seq=min_seq,
            instruction=request_instruction,
            task_id=request_task_id,
            frame=frame,
            action_anchor=anchor,
            reset_agent=reset_agent,
        )
        self._debug.record_event(
            {
                "type": "inference_request",
                "reason": reason,
                "task_id": request_task_id,
                "frame_seq": frame.seq if frame is not None else None,
                "frame_age_s": frame.age_s if frame is not None else None,
                "action_anchor": anchor,
                "completed_action_count": self._completed_action_count,
                "reset_agent": reset_agent,
                "capture_frame": capture_frame,
                "worker_busy": self._worker.is_busy(),
                "worker_pending": self._worker.has_pending(),
                "ts": time.time(),
            }
        )
        return True

    def _trim_stale_actions_locked(self, result: InferenceResult) -> tuple[List[str], int]:
        first_result_action_index = result.action_anchor + 1
        if self._current_action is None:
            first_replaceable_action_index = self._completed_action_count + 1
        else:
            first_replaceable_action_index = self._completed_action_count + 2

        skip = max(first_replaceable_action_index - first_result_action_index, 0)
        if skip >= len(result.actions):
            return [], skip
        return result.actions[skip:], skip

    def _consume_inference(self) -> None:
        result = self._worker.latest_after(self._last_inference_generation)
        if result is None:
            return

        self._last_inference_generation = result.generation
        with self._lock:
            result_is_current_task = self._task_active and result.task_id == self._task_id
            current_task_id = self._task_id
            current_task_active = self._task_active
        if not result_is_current_task:
            rospy.loginfo(
                "[uninavid] ignoring stale task inference gen=%d result_task_id=%d current_task_id=%d active=%s",
                result.generation,
                result.task_id,
                current_task_id,
                str(current_task_active),
            )
            self._publish_state(
                "stale_task_inference_ignored",
                generation=result.generation,
                result_task_id=result.task_id,
                current_task_id=current_task_id,
            )
            return

        if result.error:
            rospy.logwarn("[uninavid] ignoring invalid inference result: %s", result.error)
            return

        with self._lock:
            adjusted_actions, skipped_actions = self._trim_stale_actions_locked(result)
            completed_action_count = self._completed_action_count
            current_action = self._current_action.name if self._current_action is not None else None
            if self.inference_only:
                self._pending_actions.clear()
                self._current_action = None
            elif not adjusted_actions:
                rospy.logwarn(
                    "[uninavid] stale inference ignored gen=%d anchor=%d completed=%d current=%s skipped=%d actions=%s",
                    result.generation,
                    result.action_anchor,
                    self._completed_action_count,
                    current_action,
                    skipped_actions,
                    " ".join(result.actions),
                )
            elif adjusted_actions[0] == "stop":
                self._pending_actions.clear()
                if self._current_action is None:
                    self._request_stop_locked("model predicted stop")
                else:
                    self._pending_actions.append("stop")
            else:
                self._pending_actions = deque(adjusted_actions)

        self._publish_state(
            "inference_result",
            generation=result.generation,
            task_id=result.task_id,
            actions=result.actions,
            adjusted_actions=adjusted_actions,
            skipped_actions=skipped_actions,
            inference_s=round(result.inference_s, 3),
            request_to_result_s=round(result.result_monotonic - result.request_monotonic, 3),
            frame_seq=result.frame_seq,
            action_anchor=result.action_anchor,
            completed_action_count=completed_action_count,
            current_action=current_action,
            reason=result.request_reason,
        )
        if self.max_inferences > 0 and result.generation >= self.max_inferences:
            with self._lock:
                self._request_stop_locked(f"max_inferences reached: {self.max_inferences}")

    def _step_inference_only(self) -> None:
        if not self._task_active:
            return
        now_mono = time.monotonic()
        if self._worker.has_work():
            return
        if now_mono - self._last_inference_request_time < self.inference_only_period_s:
            return
        self._request_inference(
            reason="inference_only",
            min_seq=0,
            action_anchor=self._completed_action_count,
        )

    def _start_action_locked(self, name: str, now_ros: rospy.Time, now_mono: float) -> None:
        action_index = self._completed_action_count + 1
        if name not in {"left", "right"}:
            self._turn_run_abs_deg = 0.0

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
        rospy.loginfo("[uninavid] action start #%d: %s", action_index, name)
        self._publish_state("action_start", action=name, action_index=action_index)
        self._debug.record_event(
            {
                "type": "action_start",
                "action": name,
                "action_index": action_index,
                "ts": time.time(),
            }
        )

    def _finish_action_locked(self, now_mono: float, action: RunningAction, reason: str) -> None:
        elapsed = max(now_mono - action.start_monotonic, 0.0)
        self._completed_action_count += 1
        completed_action_count = self._completed_action_count
        if action.name in {"left", "right"}:
            turn_delta = action.progress if action.progress > 0.0 else action.target
            self._turn_run_abs_deg += turn_delta
            self._total_abs_turn_deg += turn_delta
        settle_floor = action.start_monotonic + self.action_period_s
        settle_extra = now_mono + self.post_action_settle_s
        self._settle_until = max(settle_floor, settle_extra)
        self._next_action_not_before = self._settle_until
        self._settle_capture_min_seq = self._image_buffer.latest_seq()
        self._settle_needs_inference = True
        self._settle_frame_deadline = self._settle_until + max(self.still_frame_wait_timeout_s, 0.0)
        self._current_action = None
        rospy.loginfo(
            "[uninavid] action done #%d=%s reason=%s elapsed=%.3fs settle_until=%.3fs pending=%d",
            completed_action_count,
            action.name,
            reason,
            elapsed,
            self._settle_until - now_mono,
            len(self._pending_actions),
        )
        self._publish_state(
            "action_done",
            action=action.name,
            action_index=completed_action_count,
            progress=round(action.progress, 3),
            elapsed_s=round(elapsed, 3),
            reason=reason,
            turn_run_deg=round(self._turn_run_abs_deg, 1),
            total_turn_deg=round(self._total_abs_turn_deg, 1),
            pending_actions=list(self._pending_actions),
        )
        self._debug.record_event(
            {
                "type": "action_done",
                "action": action.name,
                "action_index": completed_action_count,
                "reason": reason,
                "progress": action.progress,
                "elapsed_s": elapsed,
                "turn_run_deg": self._turn_run_abs_deg,
                "total_turn_deg": self._total_abs_turn_deg,
                "pending_actions": list(self._pending_actions),
                "ts": time.time(),
            }
        )

    def _try_request_settle_inference_locked(self, now_mono: float) -> bool:
        if not self._settle_needs_inference:
            return True

        frame = self._image_buffer.latest(
            self.max_camera_age_s,
            min_seq=self._settle_capture_min_seq,
        )
        if frame is not None:
            anchor = self._completed_action_count
            requested = self._request_inference(
                reason="post_action_still",
                min_seq=self._settle_capture_min_seq,
                frame=frame,
                action_anchor=anchor,
            )
            self._settle_needs_inference = False
            self._debug.record_event(
                {
                    "type": "still_frame_capture",
                    "requested": requested,
                    "frame_seq": frame.seq,
                    "frame_age_s": frame.age_s,
                    "action_anchor": anchor,
                    "completed_action_count": self._completed_action_count,
                    "pending_actions": list(self._pending_actions),
                    "worker_busy": self._worker.is_busy(),
                    "worker_pending": self._worker.has_pending(),
                    "ts": time.time(),
                }
            )
            return True

        if now_mono < self._settle_frame_deadline:
            return False

        if not self._pending_actions:
            return False

        self._settle_needs_inference = False
        rospy.logwarn(
            "[uninavid] no fresh still frame after action #%d; continue with %d pending actions",
            self._completed_action_count,
            len(self._pending_actions),
        )
        self._debug.record_event(
            {
                "type": "still_frame_skipped",
                "reason": "timeout",
                "action_anchor": self._completed_action_count,
                "pending_actions": list(self._pending_actions),
                "ts": time.time(),
            }
        )
        return True

    def _step_control(self) -> Twist:
        cmd = Twist()
        with self._lock:
            now_ros = rospy.Time.now()
            now_mono = time.monotonic()

            if not self._task_active:
                return cmd
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

            # No running action. Briefly hold zero command so the camera can capture a still frame.
            if now_mono < self._settle_until:
                return cmd

            if self._pending_actions and self._pending_actions[0] == "stop":
                self._pending_actions.popleft()
                self._settle_needs_inference = False
                self._request_stop_locked("stop action reached")
                return cmd

            if not self._try_request_settle_inference_locked(now_mono):
                return cmd

            if self._pending_actions and now_mono >= self._next_action_not_before:
                next_action = self._pending_actions.popleft()
                if next_action == "stop":
                    self._request_stop_locked("stop action reached")
                    return cmd
                self._start_action_locked(next_action, now_ros, now_mono)
                return cmd

            # Idle: periodically infer on the latest still frame when no queued action is available.
            if (
                not self._worker.has_work()
                and now_mono - self._last_inference_request_time >= self.idle_reinfer_period_s
            ):
                self._request_inference(
                    reason="idle",
                    min_seq=0,
                    action_anchor=self._completed_action_count,
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
        self._task_active = False
        self._stop_time = time.monotonic()
        rospy.loginfo("[uninavid] stop requested: %s", reason)
        self._publish_state("stop", task_id=self._task_id, reason=reason)
        self._debug.record_event(
            {
                "type": "stop",
                "task_id": self._task_id,
                "reason": reason,
                "ts": time.time(),
            }
        )

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
        self._publish_state(
            "waiting_for_instruction",
            instruction_topic=self.instruction_topic,
            cancel_topic=self.cancel_topic,
        )

        rate = rospy.Rate(max(self.loop_rate_hz, 1.0))
        while not rospy.is_shutdown():
            self._consume_inference()
            with self._lock:
                should_publish_cmd = self._task_active

            if should_publish_cmd:
                if self.inference_only:
                    self._step_inference_only()
                    cmd = Twist()
                else:
                    cmd = self._step_control()
                self._cmd_pub.publish(cmd)

            with self._lock:
                if (
                    self._task_active
                    and not self._stop_requested
                    and self.max_runtime_s > 0.0
                    and self._task_start_monotonic is not None
                    and time.monotonic() - self._task_start_monotonic >= self.max_runtime_s
                ):
                    self._request_stop_locked(f"max_runtime_s reached: {self.max_runtime_s:.1f}")
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
    rospy.init_node("uninavid_instruction_pipeline_controller")
    offline_eval_dir = str(rospy.get_param("~offline_eval_dir", "")).strip()
    if offline_eval_dir:
        model_path = _resolve_repo_path(str(rospy.get_param("~model_path", DEFAULT_MODEL_PATH)))
        output_dir = str(rospy.get_param("~offline_eval_output_dir", "")).strip()
        sys.exit(_run_offline_eval_mode(model_path, offline_eval_dir, output_dir))

    node = UniNaVidInstructionPipelineNode()
    sys.exit(node.run())


if __name__ == "__main__":
    main()
