"""Stateful, bounded wrist-orientation assistance, independent of hardware.

Selection and confirmation are separate events. A confirmed pose is copied and
never follows candidate ranking or re-generated poses. An exact same candidate
may be revalidated by a newer observation only when the upstream explicitly
declares the target stationary and repeats every required validity check.

Only the *assistance offset* is rate and increment limited. Each proposal is
bounded by both elapsed-time rate and a one-degree increment; slow callers do
not catch up with a large offset jump. An actuating consumer must commit the
offset only after accepting the command (SimControlSession does this).
Raw human motion remains manual;
this is not a bound on robot speed, IK, collision or physical safety. Fingers
are absent from this API. Button release fades the offset back to zero at the
same bounded rate, reported as ``returning_to_manual`` in the UI snapshot.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .types import (CopilotConfig, CopilotRequest, CopilotResult, GraspCandidate,
                    OperatorSample, PreviewSelection, ProposalBatch)


def rigid(value, name="transform") -> np.ndarray:
    pose = np.array(value, dtype=float, copy=True)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.all(np.abs(pose[3] - [0, 0, 0, 1]) <= 1e-8)
            or not np.all(np.abs(pose[:3, :3].T @ pose[:3, :3] - np.eye(3)) <= 1e-6)
            or abs(np.linalg.det(pose[:3, :3]) - 1) > 1e-6):
        raise ValueError(f"{name} must be a finite proper rigid 4x4 transform")
    return pose


def _limited_rotation(matrix, limit):
    vector = Rotation.from_matrix(matrix).as_rotvec()
    norm = float(np.linalg.norm(vector))
    if norm > limit:
        vector *= limit / norm
    return Rotation.from_rotvec(vector).as_matrix()


@dataclass
class _Choice:
    target_id: str
    observation_id: str
    candidate_id: str
    calibration_id: str
    pose: np.ndarray
    source: str
    model_source_id: str | None = None
    model_source_capture_timestamp: float | None = None
    model_source_clock_id: str | None = None


class OrientationCopilot:
    def __init__(self, config: CopilotConfig | None = None):
        self.config = config or CopilotConfig()
        self._selected: _Choice | None = None
        self._confirmed: _Choice | None = None
        self._offset = np.eye(3)
        self._last_now: float | None = None
        self._last_clock: str | None = None
        self._calibration_id: str | None = None
        self._previous_raw_rotation: np.ndarray | None = None
        self._opposition = deque()
        self._reason = "manual"
        self._preview_pending = False

    def _clear_choice(self, reason):
        self._selected = None
        self._confirmed = None
        self._preview_pending = False
        self._opposition.clear()
        self._reason = reason

    def _freshness(self, timestamp, clock_id, now, now_clock_id, max_age):
        if timestamp is None:
            return "acquisition_time_unknown"
        if not clock_id or clock_id != now_clock_id:
            return "clock_mismatch"
        if not isinstance(timestamp, (int, float, np.number)) or not np.isfinite(timestamp):
            return "acquisition_time_invalid"
        if now - timestamp < -self.config.future_tolerance_s:
            return "acquisition_time_in_future"
        if now - timestamp > max_age:
            return "stale"
        return None

    def _batch_issue(self, batch, now, clock_id, calibration_id):
        if batch is None:
            return "no_target_proposals"
        if not isinstance(batch, ProposalBatch):
            return "authorized_proposal_batch_required"
        if (batch.valid is not True or not batch.target_id or not batch.observation_id):
            return "target_invalid"
        if batch.calibration_id != calibration_id:
            return "target_calibration_mismatch"
        if batch.motion_static is not True:
            return "target_not_stationary"
        issue = self._freshness(batch.capture_timestamp, batch.clock_id, now,
                                clock_id, self.config.target_max_age_s)
        if issue:
            return "target_" + issue
        if (batch.model_source_id is not None or batch.model_source_capture_timestamp is not None
                or batch.model_source_clock_id is not None):
            stamp = batch.model_source_capture_timestamp
            source_clock = batch.model_source_clock_id or batch.clock_id
            try:
                valid_stamp = (not isinstance(stamp, (bool, np.bool_)) and
                    isinstance(stamp, (int, float, np.number)) and np.isfinite(float(stamp)) and
                    stamp >= 0 and (source_clock != batch.clock_id or stamp <= batch.capture_timestamp))
            except (ValueError, TypeError, OverflowError):
                valid_stamp = False
            if (not isinstance(batch.model_source_id, str) or not 0 < len(batch.model_source_id) <= 512 or
                    not valid_stamp or (batch.model_source_clock_id is not None and
                    (not isinstance(batch.model_source_clock_id,str) or not 0 < len(batch.model_source_clock_id) <= 256))):
                return "target_model_source_binding_invalid"
        identifiers = [candidate.candidate_id for candidate in batch.candidates]
        if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
            return "candidate_identity_invalid"
        return None

    @staticmethod
    def _candidate_issue(candidate: GraspCandidate, require_validation=True):
        if not candidate.source:
            return "candidate_source_unknown"
        try:
            rigid(candidate.base_ee)
        except (TypeError, ValueError):
            return "candidate_transform_invalid"
        if (not isinstance(candidate.model_score, (int, float, np.number))
                or not np.isfinite(candidate.model_score)):
            return "candidate_score_invalid"
        if not require_validation:
            return None
        if candidate.geometry_valid is not True:
            return "candidate_geometry_unvalidated"
        if candidate.ik_valid is not True:
            return "candidate_ik_unvalidated"
        if candidate.collision_valid is not True:
            return "candidate_collision_unvalidated"
        return None

    def _choice_issue(self, choice, batch, *, selected=False, require_validation=True):
        if batch.target_id != choice.target_id:
            return "target_changed"
        if (choice.model_source_id != batch.model_source_id or
                choice.model_source_capture_timestamp != batch.model_source_capture_timestamp or
                choice.model_source_clock_id != batch.model_source_clock_id):
            return "candidate_model_source_binding_changed"
        candidate = next((item for item in batch.candidates
                          if item.candidate_id == choice.candidate_id), None)
        if candidate is None:
            return "confirmed_candidate_lost" if not selected else "selected_candidate_lost"
        changed_validation = selected and batch.observation_id != choice.observation_id
        issue = self._candidate_issue(candidate, require_validation=require_validation or changed_validation)
        if issue:
            return issue
        # No pose updates under an old ID: frozen means frozen. Small tolerance
        # only permits floating-point transport roundoff, not measured motion.
        if (candidate.source != choice.source
                or not np.allclose(candidate.base_ee, choice.pose, atol=1e-8, rtol=0)):
            return "candidate_pose_or_source_changed"
        if changed_validation and not choice.model_source_id:
            return "selection_observation_changed"
        return None

    def _snapshot(self, state, reason, hold, request, operator, batch, now):
        choice = self._confirmed or self._selected
        return {
            "state": state, "reason": reason, "input_hold_required": bool(hold),
            "assist_held": bool(request.assist_held),
            "confirmed": self._confirmed is not None,
            "preview_selection_pending": self._preview_pending,
            "selected_candidate_id": self._selected.candidate_id if self._selected else None,
            "confirmed_candidate_id": self._confirmed.candidate_id if self._confirmed else None,
            "target_id": choice.target_id if choice else None,
            "confirmed_observation_id": self._confirmed.observation_id if self._confirmed else None,
            "validation_observation_id": batch.observation_id if batch and choice else None,
            "model_source_id": choice.model_source_id if choice else None,
            "model_source_capture_timestamp": choice.model_source_capture_timestamp if choice else None,
            "model_source_clock_id": choice.model_source_clock_id if choice else None,
            "calibration_id": self._calibration_id,
            "orientation_offset_rad": float(Rotation.from_matrix(self._offset).magnitude()),
            "max_orientation_correction_rad": float(self.config.max_orientation_correction_rad),
            "offset_rate_rad_s": float(self.config.offset_rate_rad_s),
            "max_offset_increment_rad": float(self.config.max_offset_increment_rad),
            "returning_to_manual": state == "releasing",
            "translation_assistance": False, "finger_assistance": False,
            "robot_command_sent": False,
            "input_age_s": (float(now - operator.acquisition_timestamp)
                            if isinstance(operator.acquisition_timestamp, (int, float, np.number))
                            and np.isfinite(operator.acquisition_timestamp) else None),
            "target_age_s": (float(now - batch.capture_timestamp)
                             if batch and isinstance(batch.capture_timestamp, (int, float, np.number))
                             and np.isfinite(batch.capture_timestamp) else None),
        }

    def select_preview(self, preview, operator, *, now, clock_id, calibration_id):
        """Stage only a frozen choice; caller must finish its own input gates.

        This separate API cannot confirm, clutch, calculate an offset or issue
        a robot command. The normal step still requires a real ProposalBatch.
        """
        if type(preview) is not PreviewSelection:
            raise ValueError('preview_selection_type_required')
        if (type(preview.target_revision) is not int or preview.target_revision < 1
                or type(preview.ui_revision) is not int or preview.ui_revision < 1
                or not isinstance(preview.target_signature,str) or not 0 < len(preview.target_signature) <= 65536
                or not isinstance(preview.ui_signature,str) or len(preview.ui_signature) != 64
                or any(c not in '0123456789abcdef' for c in preview.ui_signature)
                or not isinstance(preview.ui_client_id,str) or not 0 < len(preview.ui_client_id) <= 256):
            raise ValueError('preview_selection_binding_invalid')
        for value in (preview.target_id, preview.observation_id, preview.candidate_id,
                      preview.source, preview.model_source_id, preview.model_source_clock_id):
            if not isinstance(value, str) or not 0 < len(value) <= 512:
                raise ValueError('preview_identity_invalid')
        if preview.clock_id != clock_id or preview.calibration_id != calibration_id:
            raise ValueError('preview_clock_or_calibration_changed')
        if (type(preview.capture_timestamp) not in (int,float)
                or type(preview.model_score) not in (int,float)):
            raise ValueError('preview_numeric_provenance_invalid')
        issue = self._freshness(preview.capture_timestamp, preview.clock_id, now, clock_id, .25)
        if issue:
            raise ValueError('preview_' + issue)
        stamp = preview.model_source_capture_timestamp
        if (isinstance(stamp, (bool, np.bool_)) or not isinstance(stamp, (int, float))
                or not np.isfinite(stamp) or stamp < 0
                or preview.model_source_clock_id != clock_id or stamp > preview.capture_timestamp):
            raise ValueError('preview_model_source_binding_invalid')
        candidate = GraspCandidate(preview.candidate_id, rigid(preview.base_ee),
            preview.model_score, source=preview.source)
        issue = self._candidate_issue(candidate, require_validation=False)
        if issue:
            raise ValueError(issue)
        self._clear_choice('preview_selected_awaiting_endpoint')
        self._offset = np.eye(3)
        self._selected = _Choice(preview.target_id, preview.observation_id,
            preview.candidate_id, calibration_id, candidate.base_ee.copy(),
            preview.source + '|' + preview.model_source_id, preview.model_source_id,
            stamp, preview.model_source_clock_id)
        self._preview_pending = True
        self._last_now, self._last_clock, self._calibration_id = now, clock_id, calibration_id
        snapshot = self._snapshot('preview_selected', self._reason, True, CopilotRequest(),
            operator, None, now)
        snapshot.update(preview_observation_id=preview.observation_id,
                        preview_capture_timestamp=preview.capture_timestamp)
        return snapshot

    def step(self, operator: OperatorSample, proposals: ProposalBatch | None, *,
             now: float, clock_id: str, calibration_id: str,
             request: CopilotRequest | None = None) -> CopilotResult:
        request = request or CopilotRequest()
        left = rigid(operator.left_base_ee, "left_base_ee")
        right = rigid(operator.right_base_ee, "right_base_ee")
        if not np.isfinite(now) or not isinstance(clock_id, str) or not clock_id:
            raise ValueError("now and a nonempty clock_id must be explicit")
        if not isinstance(calibration_id, str):
            raise ValueError("calibration_id must be a string")

        # Establish a clock once. Bad intervals require caller HOLD and erase
        # permission; the next valid interval may return to manual, not assist.
        dt = 0.0 if self._last_now is None else float(now - self._last_now)
        timing_issue = None
        if self._last_now is not None:
            if clock_id != self._last_clock:
                timing_issue = "step_clock_changed"
            elif dt <= 0 or dt > self.config.max_step_s:
                timing_issue = "step_interval_invalid"
        self._last_now, self._last_clock = float(now), clock_id

        input_issue = ("input_tracking_invalid" if operator.tracking_valid is not True else None)
        freshness_issue = self._freshness(operator.acquisition_timestamp, operator.clock_id,
                                          now, clock_id, self.config.input_max_age_s)
        if not input_issue and freshness_issue:
            input_issue = "input_" + freshness_issue
        hold_reason = timing_issue or input_issue
        if self._calibration_id != calibration_id:
            self._clear_choice("calibration_changed" if self._calibration_id is not None
                               else "manual")
            self._calibration_id = calibration_id
        if hold_reason:
            self._clear_choice(hold_reason)
            self._previous_raw_rotation = None
            return CopilotResult(left, right, None, True, False, hold_reason,
                                 self._snapshot("hold", hold_reason, True, request,
                                                operator, proposals, now))

        issue = ("calibration_unavailable" if not calibration_id else
                 self._batch_issue(proposals, now, clock_id, calibration_id))
        if issue:
            self._clear_choice(issue)
        else:
            choice = self._confirmed or self._selected
            if choice:
                invalid = self._choice_issue(choice, proposals, selected=self._confirmed is None,
                                             require_validation=self._confirmed is not None or self._preview_pending)
                if invalid:
                    self._clear_choice(invalid)
                else:
                    if self._preview_pending:
                        self._reason = 'selected'
                    self._preview_pending = False

        if request.release:
            self._clear_choice("explicit_release")
        elif not issue:
            if request.select_candidate_id is not None:
                self._clear_choice("candidate_not_found")
                candidate = next((item for item in proposals.candidates
                                  if item.candidate_id == request.select_candidate_id), None)
                if candidate:
                    # Selection is a preview, so upstream IK/collision flags
                    # may still be false. Confirmation enforces all checks.
                    try:
                        pose = rigid(candidate.base_ee)
                    except (ValueError, TypeError):
                        self._reason = "candidate_transform_invalid"
                    else:
                        self._selected = _Choice(proposals.target_id, proposals.observation_id,
                                                 candidate.candidate_id, calibration_id,
                                                 pose, candidate.source, proposals.model_source_id,
                                                 proposals.model_source_capture_timestamp,proposals.model_source_clock_id)
                        self._reason = "selected"
            if request.confirm:
                if request.select_candidate_id is not None:
                    self._reason = "separate_confirmation_required"
                elif self._selected is None:
                    self._reason = "selection_required"
                else:
                    invalid = self._choice_issue(self._selected, proposals, selected=True)
                    if invalid:
                        self._reason = invalid
                    else:
                        s = self._selected
                        self._confirmed = _Choice(s.target_id, proposals.observation_id, s.candidate_id,
                                                  s.calibration_id, s.pose.copy(), s.source,
                                                  s.model_source_id, s.model_source_capture_timestamp,s.model_source_clock_id)
                        self._reason = "confirmed"

        active = self._confirmed is not None and request.assist_held and not request.release
        if self._confirmed is not None and not request.assist_held and not request.confirm:
            self._reason = "assist_not_held"
        raw_rotation = right[:3, :3]
        if active and self._previous_raw_rotation is not None and dt > 0:
            offset_vec = Rotation.from_matrix(self._offset).as_rotvec()
            offset_angle = float(np.linalg.norm(offset_vec))
            if offset_angle > np.deg2rad(1.0):
                manual_delta = Rotation.from_matrix(
                    raw_rotation @ self._previous_raw_rotation.T).as_rotvec()
                opposition = -float(np.dot(manual_delta, offset_vec / offset_angle))
                signed = opposition if abs(opposition) / dt >= self.config.counterrotation_min_speed_rad_s else 0.0
                self._opposition.append((float(now), signed))
                while self._opposition and self._opposition[0][0] < now - self.config.counterrotation_window_s:
                    self._opposition.popleft()
                if sum(value for _, value in self._opposition) >= self.config.counterrotation_threshold_rad:
                    self._clear_choice("manual_counterrotation")
                    active = False
        elif not active:
            self._opposition.clear()
        self._previous_raw_rotation = raw_rotation.copy()

        preview = None
        choice = self._confirmed or self._selected
        if choice is not None:
            preview = right.copy()
            preview[:3, :3] = choice.pose[:3, :3]

        desired_offset = np.eye(3)
        if active:
            desired_offset = _limited_rotation(
                self._confirmed.pose[:3, :3] @ raw_rotation.T,
                self.config.max_orientation_correction_rad)
        change = _limited_rotation(desired_offset @ self._offset.T,
                                   min(self.config.offset_rate_rad_s * dt,
                                       self.config.max_offset_increment_rad))
        self._offset = _limited_rotation(change @ self._offset,
                                         self.config.max_orientation_correction_rad)
        corrected = right.copy()
        corrected[:3, :3] = self._offset @ raw_rotation
        applied = float(Rotation.from_matrix(self._offset).magnitude()) > 1e-9
        if active:
            state, reason = "assisting", "assist_held"
        elif applied:
            state, reason = "releasing", self._reason
        else:
            state = "confirmed" if self._confirmed else "selected" if self._selected else "manual"
            reason = self._reason
        return CopilotResult(left, corrected, preview, False, applied, reason,
                             self._snapshot(state, reason, False, request, operator, proposals, now))
