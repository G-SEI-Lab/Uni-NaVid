#!/usr/bin/env python3
"""Instruction-driven Uni-NaVid ROS runner with lidar tunnel realignment."""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2

try:
    import open3d as o3d  # type: ignore
except Exception:  # pragma: no cover - robot runtime dependency
    o3d = None

import realtime_uninavid_ros_pipeline_instruction as base


@dataclass
class TunnelDirectionSnapshot:
    alignment: Optional[float]
    in_tunnel: bool
    stamp_monotonic: float
    point_count: int
    mode: str
    reason: str
    direction_xyz: Optional[List[float]]


class TunnelDirectionEstimator:
    """Estimate tunnel axis from PointCloud2 and expose robot/tunnel alignment."""

    def __init__(
        self,
        *,
        topic: str,
        process_max_hz: float,
        min_points: int,
        max_points: int,
        voxel_size: float,
        normal_radius: float,
        normal_max_nn: int,
        max_vertical_abs_z: float,
        unknown_after_misses: int,
        use_open3d: bool,
    ) -> None:
        self.topic = topic
        self.process_max_hz = max(process_max_hz, 0.0)
        self.min_points = max(int(min_points), 1)
        self.max_points = max(int(max_points), 0)
        self.voxel_size = max(float(voxel_size), 1e-3)
        self.normal_radius = max(float(normal_radius), 1e-3)
        self.normal_max_nn = max(int(normal_max_nn), 1)
        self.max_vertical_abs_z = max(float(max_vertical_abs_z), 0.0)
        self.unknown_after_misses = max(int(unknown_after_misses), 0)
        self.use_open3d = bool(use_open3d)

        self._lock = threading.Lock()
        self._snapshot = TunnelDirectionSnapshot(
            alignment=None,
            in_tunnel=False,
            stamp_monotonic=0.0,
            point_count=0,
            mode="none",
            reason="not_started",
            direction_xyz=None,
        )
        self._last_process_monotonic = 0.0
        self._miss_count = 0
        self._subscriber = rospy.Subscriber(topic, PointCloud2, self._callback, queue_size=1)

        if self.use_open3d and o3d is None:
            rospy.logwarn(
                "[uninavid_tunnel] open3d unavailable; falling back to numpy PCA"
            )
        rospy.loginfo(
            "[uninavid_tunnel] lidar estimator subscribed topic=%s max_hz=%.2f min_points=%d",
            topic,
            self.process_max_hz,
            self.min_points,
        )

    def snapshot(self) -> TunnelDirectionSnapshot:
        with self._lock:
            return TunnelDirectionSnapshot(
                alignment=self._snapshot.alignment,
                in_tunnel=self._snapshot.in_tunnel,
                stamp_monotonic=self._snapshot.stamp_monotonic,
                point_count=self._snapshot.point_count,
                mode=self._snapshot.mode,
                reason=self._snapshot.reason,
                direction_xyz=list(self._snapshot.direction_xyz)
                if self._snapshot.direction_xyz is not None
                else None,
            )

    def _callback(self, msg: PointCloud2) -> None:
        now = time.monotonic()
        if self.process_max_hz > 0.0:
            min_period = 1.0 / self.process_max_hz
            if now - self._last_process_monotonic < min_period:
                return
        self._last_process_monotonic = now

        try:
            points = np.array(
                [
                    [p[0], p[1], p[2]]
                    for p in pc2.read_points(
                        msg, field_names=("x", "y", "z"), skip_nans=True
                    )
                ],
                dtype=np.float64,
            )
        except Exception as exc:  # noqa: BLE001
            self._store_miss(now, 0, "read_points_failed:%s" % exc)
            return

        if points.ndim != 2 or points.shape[1] != 3:
            self._store_miss(now, 0, "bad_point_shape")
            return

        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if self.max_points > 0 and points.shape[0] > self.max_points:
            stride = int(math.ceil(points.shape[0] / float(self.max_points)))
            points = points[::stride]

        point_count = int(points.shape[0])
        if point_count < self.min_points:
            self._store_miss(now, point_count, "too_few_points")
            return

        try:
            direction, mode = self._estimate_direction(points)
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(
                2.0,
                "[uninavid_tunnel] tunnel direction estimate failed: %s",
                exc,
            )
            self._store_miss(now, point_count, "estimate_failed:%s" % exc)
            return

        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-6:
            self._store_miss(now, point_count, "bad_direction_norm")
            return
        direction = direction / norm

        if abs(float(direction[2])) > self.max_vertical_abs_z:
            self._store_miss(now, point_count, "vertical_axis")
            return

        x = float(direction[0])
        y = float(direction[1])
        alignment = abs(x) if x * y > 0.0 else -abs(x)
        alignment = max(min(float(alignment), 1.0), -1.0)

        with self._lock:
            self._miss_count = 0
            self._snapshot = TunnelDirectionSnapshot(
                alignment=alignment,
                in_tunnel=True,
                stamp_monotonic=now,
                point_count=point_count,
                mode=mode,
                reason="ok",
                direction_xyz=[
                    round(float(direction[0]), 6),
                    round(float(direction[1]), 6),
                    round(float(direction[2]), 6),
                ],
            )

    def _estimate_direction(self, points: np.ndarray) -> Tuple[np.ndarray, str]:
        if self.use_open3d and o3d is not None:
            try:
                return self._estimate_direction_open3d(points)
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn_throttle(
                    5.0,
                    "[uninavid_tunnel] open3d estimate failed; using numpy fallback: %s",
                    exc,
                )
        return self._estimate_direction_numpy(points)

    def _estimate_direction_open3d(self, points: np.ndarray) -> Tuple[np.ndarray, str]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_voxel = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        pcd_voxel.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn,
            )
        )
        normals = np.asarray(pcd_voxel.normals, dtype=np.float64)
        if normals.ndim != 2 or normals.shape[0] < 10 or normals.shape[1] != 3:
            raise ValueError("too few normals after voxel filtering")

        hessian = np.dot(normals.T, normals)
        eigvals, eigvecs = np.linalg.eigh(hessian)
        direction = eigvecs[:, int(np.argmin(np.abs(eigvals)))]
        return direction, "open3d_normals"

    @staticmethod
    def _estimate_direction_numpy(points: np.ndarray) -> Tuple[np.ndarray, str]:
        centered = points - np.mean(points, axis=0)
        covariance = np.dot(centered.T, centered) / max(points.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        direction = eigvecs[:, int(np.argmax(eigvals))]
        return direction, "numpy_pca"

    def _store_miss(self, now: float, point_count: int, reason: str) -> None:
        with self._lock:
            self._miss_count += 1
            if (
                self._snapshot.alignment is not None
                and self._miss_count <= self.unknown_after_misses
            ):
                self._snapshot = TunnelDirectionSnapshot(
                    alignment=self._snapshot.alignment,
                    in_tunnel=self._snapshot.in_tunnel,
                    stamp_monotonic=now,
                    point_count=point_count,
                    mode=self._snapshot.mode,
                    reason="held_last:%s" % reason,
                    direction_xyz=self._snapshot.direction_xyz,
                )
                return

            self._snapshot = TunnelDirectionSnapshot(
                alignment=None,
                in_tunnel=False,
                stamp_monotonic=now,
                point_count=point_count,
                mode="unknown",
                reason=reason,
                direction_xyz=None,
            )


class UniNaVidTunnelInstructionPipelineNode(base.UniNaVidInstructionPipelineNode):
    """Uni-NaVid instruction pipeline with lidar-assisted tunnel realignment."""

    def __init__(self) -> None:
        self._tunnel_estimator: Optional[TunnelDirectionEstimator] = None
        self._tunnel_realigning = False
        self._tunnel_realign_started_at = 0.0
        self._tunnel_realign_task_id = 0
        self._tunnel_last_turn_sign = 1.0
        self._tunnel_last_log_monotonic = 0.0

        super().__init__()

        self.use_tunnel_lidar = bool(rospy.get_param("~use_tunnel_lidar", True))
        self.lidar_topic = str(rospy.get_param("~lidar_topic", "/livox/lidar"))
        self.tunnel_allow_alignment = float(
            rospy.get_param("~tunnel_allow_alignment", 0.7)
        )
        self.tunnel_target_alignment = float(
            rospy.get_param("~tunnel_target_alignment", 0.95)
        )
        self.tunnel_turn_rate_rps = float(rospy.get_param("~tunnel_turn_rate_rps", 0.5))
        self.tunnel_freshness_s = float(rospy.get_param("~tunnel_freshness_s", 1.0))
        self.tunnel_process_max_hz = float(
            rospy.get_param("~tunnel_process_max_hz", 5.0)
        )
        self.tunnel_min_points = int(rospy.get_param("~tunnel_min_points", 100))
        self.tunnel_max_points = int(rospy.get_param("~tunnel_max_points", 30000))
        self.tunnel_voxel_size = float(rospy.get_param("~tunnel_voxel_size", 0.3))
        self.tunnel_normal_radius = float(rospy.get_param("~tunnel_normal_radius", 1.0))
        self.tunnel_normal_max_nn = int(rospy.get_param("~tunnel_normal_max_nn", 8))
        self.tunnel_max_vertical_abs_z = float(
            rospy.get_param("~tunnel_max_vertical_abs_z", 0.1)
        )
        self.tunnel_unknown_after_misses = int(
            rospy.get_param("~tunnel_unknown_after_misses", 10)
        )
        self.tunnel_use_open3d = bool(rospy.get_param("~tunnel_use_open3d", True))

        if self.use_tunnel_lidar:
            self._tunnel_estimator = TunnelDirectionEstimator(
                topic=self.lidar_topic,
                process_max_hz=self.tunnel_process_max_hz,
                min_points=self.tunnel_min_points,
                max_points=self.tunnel_max_points,
                voxel_size=self.tunnel_voxel_size,
                normal_radius=self.tunnel_normal_radius,
                normal_max_nn=self.tunnel_normal_max_nn,
                max_vertical_abs_z=self.tunnel_max_vertical_abs_z,
                unknown_after_misses=self.tunnel_unknown_after_misses,
                use_open3d=self.tunnel_use_open3d,
            )

        self._debug.record_event(
            {
                "type": "tunnel_config",
                "enabled": self.use_tunnel_lidar,
                "lidar_topic": self.lidar_topic,
                "allow_alignment": self.tunnel_allow_alignment,
                "target_alignment": self.tunnel_target_alignment,
                "turn_rate_rps": self.tunnel_turn_rate_rps,
                "freshness_s": self.tunnel_freshness_s,
                "assist_during_forward": False,
                "ts": time.time(),
            }
        )
        rospy.loginfo(
            "[uninavid_tunnel] enabled=%s lidar=%s allow=%.2f target=%.2f turn_rate=%.3f assist_forward=false",
            str(self.use_tunnel_lidar),
            self.lidar_topic,
            self.tunnel_allow_alignment,
            self.tunnel_target_alignment,
            self.tunnel_turn_rate_rps,
        )

    def _reset_tunnel_realign_state(self) -> None:
        self._tunnel_realigning = False
        self._tunnel_realign_started_at = 0.0
        self._tunnel_realign_task_id = 0

    def _start_new_task(self, instruction: str) -> None:
        self._reset_tunnel_realign_state()
        super()._start_new_task(instruction)

    def _cancel_current_task(self, reason: str) -> None:
        self._reset_tunnel_realign_state()
        super()._cancel_current_task(reason)

    def _step_control(self) -> Twist:
        if not self.use_tunnel_lidar or self._tunnel_estimator is None:
            return super()._step_control()

        snapshot = self._tunnel_estimator.snapshot()
        now_mono = time.monotonic()

        with self._lock:
            task_active = self._task_active
            stop_requested = self._stop_requested
            estop_active = self.use_estop and self._estop

        if not task_active or stop_requested or estop_active:
            self._tunnel_realigning = False
            return super()._step_control()

        if not self._snapshot_is_fresh_tunnel(snapshot, now_mono):
            if self._tunnel_realigning:
                self._publish_tunnel_throttled(
                    "tunnel_realign_lidar_lost",
                    snapshot=snapshot,
                    alignment_abs=None,
                )
                return Twist()
            return super()._step_control()

        alignment = float(snapshot.alignment or 0.0)
        alignment_abs = abs(alignment)

        if self._tunnel_realigning or alignment_abs < self.tunnel_allow_alignment:
            if alignment_abs >= self.tunnel_target_alignment:
                return self._finish_tunnel_realign_and_restart(snapshot)
            return self._tunnel_realign_command(snapshot, now_mono)

        return super()._step_control()

    def _snapshot_is_fresh_tunnel(
        self, snapshot: TunnelDirectionSnapshot, now_mono: float
    ) -> bool:
        if not snapshot.in_tunnel or snapshot.alignment is None:
            return False
        if self.tunnel_freshness_s <= 0.0:
            return True
        return now_mono - snapshot.stamp_monotonic <= self.tunnel_freshness_s

    def _tunnel_realign_command(
        self, snapshot: TunnelDirectionSnapshot, now_mono: float
    ) -> Twist:
        alignment = float(snapshot.alignment or 0.0)
        if abs(alignment) > 1e-6:
            self._tunnel_last_turn_sign = 1.0 if alignment > 0.0 else -1.0

        started = False
        with self._lock:
            if not self._task_active or self._stop_requested:
                return Twist()

            if not self._tunnel_realigning:
                started = True
                self._tunnel_realigning = True
                self._tunnel_realign_started_at = now_mono
                self._tunnel_realign_task_id = self._task_id

            self._pending_actions.clear()
            self._current_action = None
            self._settle_until = 0.0
            self._settle_needs_inference = False
            self._settle_capture_min_seq = 0
            self._settle_frame_deadline = 0.0
            self._next_action_not_before = 0.0

        self._worker.cancel_pending(self._tunnel_realign_task_id)

        if started:
            self._publish_state(
                "tunnel_realign_start",
                task_id=self._tunnel_realign_task_id,
                alignment=round(alignment, 3),
                target_alignment=self.tunnel_target_alignment,
                mode=snapshot.mode,
                reason=snapshot.reason,
            )
            self._debug.record_event(
                {
                    "type": "tunnel_realign_start",
                    "task_id": self._tunnel_realign_task_id,
                    "alignment": alignment,
                    "target_alignment": self.tunnel_target_alignment,
                    "mode": snapshot.mode,
                    "reason": snapshot.reason,
                    "direction_xyz": snapshot.direction_xyz,
                    "ts": time.time(),
                }
            )
        else:
            self._publish_tunnel_throttled(
                "tunnel_realigning",
                snapshot=snapshot,
                alignment_abs=abs(alignment),
            )

        cmd = Twist()
        cmd.angular.z = self._tunnel_last_turn_sign * abs(self.tunnel_turn_rate_rps)
        return cmd

    def _finish_tunnel_realign_and_restart(
        self, snapshot: TunnelDirectionSnapshot
    ) -> Twist:
        with self._lock:
            task_id = self._task_id
            instruction = self.instruction
            can_restart = self._task_active and not self._stop_requested and bool(instruction)
            self._pending_actions.clear()
            self._current_action = None
            self._settle_until = 0.0
            self._settle_needs_inference = False
            self._next_action_not_before = 0.0
            self._tunnel_realigning = False

        alignment = float(snapshot.alignment or 0.0)
        elapsed = max(time.monotonic() - self._tunnel_realign_started_at, 0.0)
        self._publish_state(
            "tunnel_realign_done",
            task_id=task_id,
            alignment=round(alignment, 3),
            elapsed_s=round(elapsed, 3),
            restart_same_instruction=can_restart,
            mode=snapshot.mode,
        )
        self._debug.record_event(
            {
                "type": "tunnel_realign_done",
                "task_id": task_id,
                "alignment": alignment,
                "elapsed_s": elapsed,
                "restart_same_instruction": can_restart,
                "mode": snapshot.mode,
                "direction_xyz": snapshot.direction_xyz,
                "ts": time.time(),
            }
        )

        if can_restart:
            rospy.loginfo(
                "[uninavid_tunnel] tunnel aligned; restarting task_id=%d with same instruction",
                task_id,
            )
            self._start_new_task(instruction)

        return Twist()

    def _publish_tunnel_throttled(
        self,
        state: str,
        *,
        snapshot: TunnelDirectionSnapshot,
        alignment_abs: Optional[float],
    ) -> None:
        now = time.monotonic()
        if now - self._tunnel_last_log_monotonic < 1.0:
            return
        self._tunnel_last_log_monotonic = now
        payload = {
            "alignment": round(float(snapshot.alignment), 3)
            if snapshot.alignment is not None
            else None,
            "alignment_abs": round(alignment_abs, 3)
            if alignment_abs is not None
            else None,
            "allow_alignment": self.tunnel_allow_alignment,
            "target_alignment": self.tunnel_target_alignment,
            "mode": snapshot.mode,
            "reason": snapshot.reason,
            "point_count": snapshot.point_count,
        }
        self._publish_state(state, **payload)


def main() -> None:
    rospy.init_node("uninavid_instruction_pipeline_tunnel_controller")
    offline_eval_dir = str(rospy.get_param("~offline_eval_dir", "")).strip()
    if offline_eval_dir:
        model_path = base._resolve_repo_path(
            str(rospy.get_param("~model_path", base.DEFAULT_MODEL_PATH))
        )
        output_dir = str(rospy.get_param("~offline_eval_output_dir", "")).strip()
        sys.exit(base._run_offline_eval_mode(model_path, offline_eval_dir, output_dir))

    node = UniNaVidTunnelInstructionPipelineNode()
    sys.exit(node.run())


if __name__ == "__main__":
    main()
