"""Analytic short-horizon shared control; no trained policy or grasp detector.

Human motion provides progress and reversal. A constant-velocity lookahead
predicts transverse misses; calibrated path geometry supplies a bounded visual
servo residual. The controller cannot choose objects, actions, or contact state.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
import numpy as np
from scipy.spatial.transform import Rotation


def _limit(value, maximum):
    norm = float(np.linalg.norm(value))
    return value if norm <= maximum else value * maximum / norm


@dataclass(frozen=True)
class PredictiveConfig:
    horizon_s: float = .12
    max_prediction_speed_m_s: float = .4
    max_correction_m: float = .055
    max_orientation_correction_rad: float = .20
    activation_radius_m: float = .22
    min_confidence: float = .55
    max_position_sigma_m: float = .025
    max_orientation_sigma_rad: float = .35
    moving_threshold_m_s: float = .002
    # Whole-hand stillness is decided above. An already moving hand can have
    # a much slower axial component during a diagonal or transverse gesture;
    # it must not repeatedly trigger a full stop as sampling latency varies.
    progress_zero_velocity_m_s: float = 1e-6
    progress_zero_angular_velocity_rad_s: float = 1e-6
    motion_expiry_s: float = .08
    motion_window_s: float = .08

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid predictive configuration: {name}")
        if self.min_confidence > 1 or self.horizon_s > .5 or self.motion_window_s > .12:
            raise ValueError("Invalid predictive confidence or horizon")


class PredictiveServo:
    def __init__(self, config=None):
        self.config = config or PredictiveConfig()
        self.reset()

    def reset(self):
        self.previous_position = None
        self.previous_sample_time = None
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.previous_rotation = None
        self._translation_motion_samples = deque()
        self._angular_motion_samples = deque()
        self.motion_speed = 0.
        self.angular_motion_speed = 0.
        self.diagnostics = {"model": "analytic_constant_velocity_visual_servo", "horizon_s": self.config.horizon_s}

    def prime(self, position, sample_time, quaternion=None):
        self.reset()
        self.previous_position = np.asarray(position, dtype=float).copy()
        self.previous_sample_time = float(sample_time)
        self.previous_rotation = Rotation.from_quat(quaternion) if quaternion is not None else None

    def update_motion(self, raw, sample_time, quaternion=None):
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (3,) or not np.isfinite(raw).all() or not math.isfinite(sample_time):
            raise ValueError("Prediction needs a finite operator pose and time")
        rotation = Rotation.from_quat(quaternion) if quaternion is not None else None
        if self.previous_sample_time is None:
            self.prime(raw, sample_time, quaternion)
            return
        delta_t = sample_time - self.previous_sample_time
        if delta_t <= 0:
            return
        # Do not preserve velocity through packet gaps or input-session changes.
        if delta_t > .3:
            self.velocity = np.zeros(3)
            self.angular_velocity = np.zeros(3)
            self._translation_motion_samples.clear()
            self._angular_motion_samples.clear()
            self.motion_speed = self.angular_motion_speed = 0.
        else:
            displacement = raw - self.previous_position
            angular_displacement = ((rotation * self.previous_rotation.inv()).as_rotvec()
                                    if rotation is not None and self.previous_rotation is not None else np.zeros(3))
            self.velocity = _limit(displacement / delta_t, self.config.max_prediction_speed_m_s)
            self.angular_velocity = angular_displacement / delta_t
            self.motion_speed = self._motion_speed(self._translation_motion_samples, np.linalg.norm(displacement), sample_time)
            self.angular_motion_speed = self._motion_speed(self._angular_motion_samples, np.linalg.norm(angular_displacement), sample_time)
        self.previous_position = raw.copy()
        self.previous_sample_time = float(sample_time)
        self.previous_rotation = rotation

    def _motion_speed(self, samples, distance, now):
        # Arrival jitter changes a single finite-difference speed even for a
        # steadily moving hand. Use a short interval-weighted history only for
        # the moving/stationary decision; direction, correction prediction and
        # execution progress still use the newest sample's actual velocity.
        # An explicitly unchanged pose clears history immediately, preventing
        # a filtered tail after an operator stop. Reversal is never smoothed.
        if distance <= 1e-9:
            samples.clear()
            return 0.
        samples.append((self.previous_sample_time, now, float(distance)))
        while len(samples)>1 and samples[0][1]<now-self.config.motion_window_s:
            samples.popleft()
        start=max(samples[0][0],now-self.config.motion_window_s)
        distance=sum(row[2]*(row[1]-max(row[0],start))/(row[1]-row[0]) for row in samples)
        return distance/(now-start)

    def evaluate(self, raw, sample_time, now, action, phase, target, quaternion=None):
        self.update_motion(raw, sample_time, quaternion)
        raw = np.asarray(raw, dtype=float)
        confidence = float(target.get("pose_confidence", target.get("confidence", 0.)))
        sigma = float(target.get("position_sigma_m", .01))
        orientation_sigma = float(target.get("orientation_sigma_rad", .1))
        result = {"correction": np.zeros(3), "orientation_correction": np.zeros(3),
                  "hold": False, "active": False, "reason": "predictive_aligned"}
        if not all(math.isfinite(v) for v in (confidence, sigma, orientation_sigma)) or not 0 <= confidence <= 1 or sigma < 0 or orientation_sigma < 0:
            result.update(hold=True, reason="perception_uncertain")
            return result
        if confidence < self.config.min_confidence or sigma > self.config.max_position_sigma_m or (
                quaternion is not None and "grasp_quaternion_xyzw" in action and orientation_sigma > self.config.max_orientation_sigma_rad):
            result.update(hold=True, reason="perception_uncertain")
            return result
        moving = (now - sample_time <= self.config.motion_expiry_s and
                  (self.motion_speed >= self.config.moving_threshold_m_s or
                   self.angular_motion_speed >= .015))
        predicted = raw + self.velocity * self.config.horizon_s if moving else raw.copy()
        grasp = np.asarray(action["grasp_position_world"], dtype=float)
        axis = np.asarray(action["approach_axis_world"], dtype=float)
        if phase == "execute" and action["kind"] == "linear":
            axis = np.asarray(action["motion_axis_world"], dtype=float)
        projector = np.eye(3) - np.outer(axis, axis)
        progress_axis = axis
        progress_velocity = float(self.velocity @ axis)
        if phase == "execute" and action["kind"] == "axial_rotation":
            axis = np.asarray(action["motion_axis_world"], dtype=float)
            pivot = np.asarray(action["pivot_world"], dtype=float)
            residual = pivot - predicted
            distance = float(np.linalg.norm(pivot - raw))
            progress_velocity = 0.
            progress_axis = np.zeros(3)
        elif phase == "execute" and action["kind"] == "revolute":
            axis = np.asarray(action["motion_axis_world"], dtype=float)
            pivot = np.asarray(action["pivot_world"], dtype=float)
            origin_radius = grasp - pivot
            height = float(origin_radius @ axis)
            origin_radius -= height * axis
            radius = float(np.linalg.norm(origin_radius))
            radial = raw - pivot - axis * float((raw - pivot) @ axis)
            if np.linalg.norm(radial) < .005:
                result.update(hold=True, reason="revolute_singularity")
                return result
            tangent = np.cross(axis, radial / np.linalg.norm(radial))
            progress_axis = tangent
            projector = np.eye(3) - np.outer(tangent, tangent)
            predicted_radial = predicted - pivot - axis * float((predicted - pivot) @ axis)
            if np.linalg.norm(predicted_radial) < .005:
                result.update(hold=True, reason="revolute_singularity")
                return result
            on_path = pivot + height * axis + radius * predicted_radial / np.linalg.norm(predicted_radial)
            residual = projector @ (on_path - predicted)
            distance = float(np.hypot(np.linalg.norm(radial) - radius, float((raw - pivot) @ axis) - height))
            progress_velocity = float(self.velocity @ tangent)
        else:
            residual = projector @ (grasp - predicted)
            distance = float(np.linalg.norm(grasp - raw)) if phase == "approach" else float(np.linalg.norm(projector @ (grasp - raw)))
        self.diagnostics = {"model": "analytic_constant_velocity_visual_servo", "horizon_s": self.config.horizon_s,
                            "predicted_operator_position": predicted.tolist(), "operator_velocity_m_s": self.velocity.tolist(),
                            "operator_progress_velocity_m_s": progress_velocity, "moving": bool(moving),
                            "motion_speed_m_s": self.motion_speed, "angular_motion_speed_rad_s": self.angular_motion_speed,
                            "motion_window_s": self.config.motion_window_s,
                            "progress_axis_world": progress_axis.tolist(),
                            "phase": phase, "position_sigma_m": sigma,
                            "calibration_id": action["calibration_id"]}
        if phase == "execute" and action["kind"] == "axial_rotation":
            self.diagnostics.update(angular_progress_axis_world=axis.tolist(),
                                    operator_angular_progress_rad_s=float(self.angular_velocity @ axis))
            if quaternion is None:
                result.update(hold=True, reason="pose_control_required_for_rotation")
                return result
        if not moving:
            result.update(hold=True, reason="operator_stationary")
            return result
        if distance > self.config.activation_radius_m:
            result["reason"] = "outside_assist_region"
            return result
        result["correction"] = _limit(residual, self.config.max_correction_m)
        if quaternion is not None and "grasp_quaternion_xyzw" in action:
            rotation_error = (Rotation.from_quat(action["grasp_quaternion_xyzw"]) * Rotation.from_quat(quaternion).inv()).as_rotvec()
            if phase == "execute" and action["kind"] in ("revolute", "axial_rotation"):
                rotation_error -= axis * float(rotation_error @ axis)
            result["orientation_correction"] = _limit(rotation_error, self.config.max_orientation_correction_rad)
        result["active"] = bool(np.linalg.norm(result["correction"]) > 1e-5 or np.linalg.norm(result["orientation_correction"]) > 1e-5)
        result["reason"] = "predictive_alignment" if result["active"] else "predictive_aligned"
        return result
