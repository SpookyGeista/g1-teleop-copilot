"""Explicit, calibrated action capabilities and the operator's action latch.

An image label or bounding box is never a mechanical model. Capabilities are
accepted only from a declared fixture/CAD/operator calibration. Geometry is in
the robot world frame, lengths in metres, quaternions XYZW.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np


def _vector(value, length, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (length,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain {length} finite numbers")
    return a


def _axis(value, name):
    a = _vector(value, 3, name)
    if np.linalg.norm(a) < 1e-6:
        raise ValueError(f"{name} must have nonzero length")
    return (a / np.linalg.norm(a)).tolist()


def calibrated_actions(target):
    """Validate declared capabilities; return no guessed fallback actions."""
    calibration = (target or {}).get("task_calibration")
    if calibration is None:
        return []
    if not isinstance(calibration, dict) or calibration.get("source") not in (
            "configured_fixture", "cad_registration", "operator_calibration"):
        raise ValueError("Action capabilities require an explicit calibration source")
    if not isinstance(calibration.get("id"), str) or not calibration["id"]:
        raise ValueError("Action calibration requires a stable id")
    entries = calibration.get("actions", [])
    if not isinstance(entries, list) or len(entries) > 32:
        raise ValueError("Invalid calibrated action list")
    result, ids = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid action capability")
        a = deepcopy(entry)
        if not isinstance(a.get("id"), str) or not a["id"] or a["id"] in ids:
            raise ValueError("Calibrated actions require unique string ids")
        ids.add(a["id"])
        if a.get("kind") not in ("linear", "revolute", "axial_rotation", "grasp"):
            raise ValueError("Unknown action mechanism")
        a["label"] = str(a.get("label", a["id"]))[:100]
        a["grasp_position_world"] = _vector(a["grasp_position_world"], 3, "grasp position").tolist()
        a["approach_axis_world"] = _axis(a["approach_axis_world"], "approach axis")
        if a["kind"] in ("linear", "revolute", "axial_rotation"):
            a["motion_axis_world"] = _axis(a["motion_axis_world"], "motion axis")
        if a["kind"] == "axial_rotation":
            a["pivot_world"] = _vector(a["pivot_world"], 3, "pivot").tolist()
        if a["kind"] == "revolute":
            a["pivot_world"] = _vector(a["pivot_world"], 3, "pivot").tolist()
            axis = np.asarray(a["motion_axis_world"])
            radius = np.asarray(a["grasp_position_world"]) - np.asarray(a["pivot_world"])
            radius -= axis * float(radius @ axis)
            if np.linalg.norm(radius) < .005:
                raise ValueError("Revolute action requires a calibrated radius of at least 5 mm")
        if "grasp_quaternion_xyzw" in a:
            q = _vector(a["grasp_quaternion_xyzw"], 4, "grasp quaternion")
            if not .5 <= np.linalg.norm(q) <= 1.5:
                raise ValueError("Invalid grasp quaternion")
            a["grasp_quaternion_xyzw"] = (q / np.linalg.norm(q)).tolist()
        a["calibration_id"] = calibration["id"]
        a["calibration_source"] = calibration["source"]
        if "execution_ready" in a and not isinstance(a["execution_ready"],bool):
            raise ValueError("execution_ready must be a JSON boolean")
        if "execution_travel_limit_m" in a:
            limit=float(a["execution_travel_limit_m"])
            if not np.isfinite(limit) or not 0<limit<=.2 or a["kind"]!="linear":
                raise ValueError("Bounded execution requires a linear action and a 0..20 cm limit")
            a["execution_travel_limit_m"]=limit
        result.append(a)
    return result


class ActionSelection:
    """Object and action identity stay fixed until explicit operator changes."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.target_id = None
        self.id = None
        self.confirmed = False
        self.phase = "approach"
        self.engagement_status = "not_confirmed"
        self.available = []
        self.calibration_id = None
        self.valid = False
        self.reason = "target_confirmation_required"
        self._geometry = None
        self._mechanism = None
        self._axis = None
        self._pivot = None
        self.execution_origin_world = None
        self.execution_axis_world = None
        self.execution_travel_limit_m = None

    def update(self, target, object_confirmed):
        if not object_confirmed or target is None:
            self.valid = False
            self.available = []
            self.reason = "target_confirmation_required"
            return
        entries = calibrated_actions(target)
        if self.target_id != target["id"]:
            self.reset()
            self.target_id = target["id"]
        self.available = entries
        if not self.confirmed:
            if self.id not in {a["id"] for a in entries}:
                self.id = entries[0]["id"] if entries else None
            self.valid = bool(entries)
            self.reason = "action_confirmation_required" if entries else "calibrated_action_unavailable"
            return
        current = next((a for a in entries if a["id"] == self.id), None)
        self.valid = current is not None and current["calibration_id"] == self.calibration_id and current["kind"] == self._mechanism
        self.reason = "ready" if self.valid else "action_calibration_lost"
        if (self.valid and self.execution_origin_world is not None and
                current.get("execution_travel_limit_m")!=self.execution_travel_limit_m):
            self.valid=False
            self.reason="execution_limit_changed"
        if self.valid and self.phase=="execute" and current.get("execution_ready") is False:
            self.valid=False
            self.reason="execution_calibration_lost"
        if self.valid and self._geometry is not None:
            shift = np.linalg.norm(np.asarray(current["grasp_position_world"]) - self._geometry)
            if shift > .08:
                self.valid = False
                self.reason = "action_geometry_jump"
            current_axis=np.asarray(current.get("motion_axis_world",current["approach_axis_world"]))
            if self._axis is not None and float(current_axis@self._axis)<np.cos(.5):
                self.valid=False
                self.reason="action_geometry_jump"
            if self._pivot is not None and np.linalg.norm(np.asarray(current["pivot_world"])-self._pivot)>.08:
                self.valid=False
                self.reason="action_geometry_jump"
        if self.valid:
            self._geometry = np.asarray(current["grasp_position_world"])
            self._axis=np.asarray(current.get("motion_axis_world",current["approach_axis_world"]))
            self._pivot=np.asarray(current["pivot_world"]) if "pivot_world" in current else None

    def select(self, action_id):
        if action_id not in {a["id"] for a in self.available}:
            raise ValueError("Action is not calibrated for the confirmed object")
        self.id = action_id
        self.confirmed = False
        self.phase = "approach"
        self.engagement_status = "not_confirmed"
        self.reason = "action_confirmation_required"
        self.execution_origin_world = None
        self.execution_axis_world = None
        self.execution_travel_limit_m = None

    def navigate(self, direction):
        if direction not in ("up", "down", "left", "right"):
            raise ValueError("Unknown direction")
        if self.confirmed or not self.available:
            return False
        index = next((i for i, a in enumerate(self.available) if a["id"] == self.id), 0)
        delta = -1 if direction in ("up", "left") else 1
        self.select(self.available[(index + delta) % len(self.available)]["id"])
        return True

    def confirm(self):
        current = self.current()
        if current is None or not self.valid:
            raise ValueError("A calibrated action for the visible locked target is required")
        self.confirmed = True
        self.calibration_id = current["calibration_id"]
        self._geometry = np.asarray(current["grasp_position_world"])
        self._mechanism = current["kind"]
        self._axis = np.asarray(current.get("motion_axis_world",current["approach_axis_world"]))
        self._pivot = np.asarray(current["pivot_world"]) if "pivot_world" in current else None
        self.reason = "ready"

    def set_phase(self, phase, engagement_confirmed=False, execution_origin_world=None):
        if not self.confirmed or not self.valid:
            raise ValueError("Confirm a valid calibrated action first")
        if phase not in ("approach", "execute"):
            raise ValueError("Unknown action phase")
        if phase == "execute" and engagement_confirmed is not True:
            raise ValueError("Execution requires explicit operator engagement confirmation")
        if phase == "execute" and self.current().get("execution_ready") is False:
            raise ValueError("This action provides surface approach only; a calibrated grasp pose is required for execution")
        # Validate before changing state: a malformed request cannot enter the
        # execute phase or erase the preceding motion budget.
        origin=None
        if phase=="execute" and execution_origin_world is not None:
            origin=_vector(execution_origin_world,3,"measured execution origin")
        if phase!=self.phase or phase=="approach":
            self.execution_origin_world=None
            self.execution_axis_world=None
            self.execution_travel_limit_m=None
        self.phase = phase
        self.engagement_status = "operator_confirmed_not_sensor_verified" if phase == "execute" else "not_confirmed"
        if origin is not None:self.latch_execution_origin(origin)

    def latch_execution_origin(self, measured_position):
        action=self.current()
        if (self.phase=="execute" and action is not None and "execution_travel_limit_m" in action
                and self.execution_origin_world is None):
            self.execution_origin_world=_vector(measured_position,3,"measured execution origin").tolist()
            self.execution_axis_world=list(action["motion_axis_world"])
            self.execution_travel_limit_m=action["execution_travel_limit_m"]

    def current(self):
        return next((a for a in self.available if a["id"] == self.id), None)

    def snapshot(self, required=False, rearm_required=False):
        return {"required": required, "target_id": self.target_id, "id": self.id,
                "confirmed": self.confirmed, "phase": self.phase,
                "engagement_status": self.engagement_status, "valid": self.valid,
                "reason": self.reason, "rearm_required": rearm_required,
                "execution_origin_world":self.execution_origin_world,
                "execution_axis_world":self.execution_axis_world,
                "execution_travel_limit_m":self.execution_travel_limit_m,
                "available": deepcopy(self.available)}
