"""Bounded end-to-end acceptance using observed targets and synthetic VR input.

The driver consumes the same perception/session API as a headset. It never
reads simulator object poses, masks, joint metadata, or hidden target identities
to choose or move toward an object. Mechanism telemetry may be reported after
execution but is not an input to the trajectory or controller.
"""
from __future__ import annotations

import numpy as np

from .actions import calibrated_actions


class StackAcceptanceDriver:
    MAX_WALL_SECONDS = 180.
    MAX_SIM_SECONDS = 90.
    TRIAL_DURATION = 12.
    RETURN_DURATION = 8.
    STANDOFF_M = .10
    COMPARISON_APPROACH_M = .02
    OFFSET_M = .025
    XR_TO_ROBOT = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])

    def __init__(self, session, preferred_action="pull", preferred_label="handle", require_cad_pose=False):
        self.session = session
        self.preferred_action = str(preferred_action)
        self.preferred_label = str(preferred_label) if preferred_label is not None else None
        self.require_cad_pose = bool(require_cad_pose)
        if not self.preferred_action:
            raise ValueError("Acceptance requires an explicit preferred action")
        self.phase = "await_perception"
        self.phase_start = None
        self.phase_wall = session.clock()
        self.wall_start = session.clock()
        self.start_sim = None
        self.done = False
        self.seq = 0
        self.target_id = None
        self.action_id = None
        self.target_description = None
        self.goal = None
        self.normal = None
        self.offset = None
        self.start_pose = None
        self.phase_pose = None
        self.results = {}
        self.events = []
        self.samples = []
        self.mechanism_telemetry = None
        self.invalid_states = 0
        self.max_joint_violation = 0.
        self.contact_stop_steps = 0
        self.max_correction_m = 0.
        self.max_rotation_correction_rad = 0.
        self.max_target_drift_m = 0.
        self._fault_seen = False
        self._fault_correct = True
        self._dropout_frames = 0
        self._reverse_seen = False
        self._reverse_correct = True
        self._fresh_clutch_seen = False
        self._candidate_id = None
        self._candidate_frames = 0
        self._candidate_last_frame = -1
        self._candidate_last_time = None
        self._candidate_center = None
        self._candidate_calibration = None

    def _event(self, name, sim_time, **values):
        self.events.append({"event": name, "sim_time": float(sim_time), "wall_time": self.session.clock(), **values})

    def _set_phase(self, name, sim_time, state):
        self.phase = name
        self.phase_start = float(sim_time)
        self.phase_wall = self.session.clock()
        self.phase_pose = np.asarray(state["ee_position"], dtype=float)
        self.samples = []
        self._fault_seen = False
        self._fault_correct = True
        self._event("stack_acceptance_phase", sim_time, phase=name)

    def _input(self, desired=None, clutch=False):
        self.seq += 1
        if desired is None or self.session.hand_origin is None or self.session.robot_origin is None:
            hand = np.zeros(3)
        else:
            hand = self.XR_TO_ROBOT.T @ (np.asarray(desired) - self.session.robot_origin) / self.session.config.hand_scale
            hand += self.session.hand_origin
        self.session.command({"type": "input", "sample": {
            "source": "synthetic_stack_acceptance", "session_id": "stack_acceptance_v1", "seq": self.seq,
            "right": {"valid": True, "position": hand.tolist(), "quaternion_xyzw": [0., 0., 0., 1.]},
            "clutch": bool(clutch), "squeeze": 1. if clutch else 0., "trigger": 0.,
            "orientation_mode": "position", "axes": [0., 0.], "primary": False, "secondary": False,
            "frame_id": self.session.frame_id}})

    def _arm(self, state):
        self.session.robot = dict(state)
        self._input(clutch=False)
        self.session.command({"type": "calibrate"})
        self._input(clutch=False)
        self._input(clutch=True)

    def _find_goal(self, state):
        candidates = []
        ee = np.asarray(state["ee_position"], dtype=float)
        for target in self.session.selection.targets.values():
            if self.require_cad_pose:
                calibration=target.get("task_calibration",{})
                if (target.get("pose_valid") is not True
                        or target.get("pose_source") not in ("foundationpose","foundationpose_rgbd_tracking")
                        or calibration.get("source")!="cad_registration"
                        or not calibration.get("has_calibrated_grasp_template",False)):
                    continue
            if self.preferred_label is not None and target.get("semantic_label", target.get("label")) != self.preferred_label:
                continue
            actions = calibrated_actions(target)
            preferred = next((a for a in actions if a["id"] == self.preferred_action), None)
            if preferred is None:
                continue
            # The calibrated TCP and its approach axis define the trial. A
            # visible handle edge can have a very different local depth normal.
            normal = np.asarray(preferred["approach_axis_world"], dtype=float)
            normal /= np.linalg.norm(normal)
            goal = np.asarray(preferred["grasp_position_world"], dtype=float) + self.STANDOFF_M * normal
            lo, hi = np.asarray(self.session.config.workspace_min), np.asarray(self.session.config.workspace_max)
            if np.any(goal < lo + .015) or np.any(goal > hi - .015) or np.linalg.norm(goal - ee) > .45:
                continue
            score = float(np.linalg.norm(goal - ee))
            candidates.append((score, target, preferred, normal, goal))
        return min(candidates, key=lambda row: row[0]) if candidates else None

    def _stable_goal(self, state):
        """Require three distinct, fresh acquisition frames before locking.

        Physics ticks and repeated reads of one camera frame do not count as
        extra observations. Neither ID replacement nor a missing frame silently
        changes the chosen object to a different class or action.
        """
        frame_id = self.session.frame_id
        if frame_id <= self._candidate_last_frame:
            return None
        self._candidate_last_frame = frame_id
        if self.session.clock() - self.session.frame_time >= .2:
            self._candidate_id = None
            self._candidate_frames = 0
            self._candidate_last_time = None
            return None
        # Acquisition freshness is evaluated once per delivered frame. Slow
        # camera delivery must not let repeated physics ticks invalidate a
        # previously fresh observation. A real measurement gap still breaks
        # the consecutive-observation requirement when delivery resumes.
        if (self._candidate_last_time is not None
                and self.session.frame_time - self._candidate_last_time > .5):
            self._candidate_id = None
            self._candidate_frames = 0
        self._candidate_last_time = self.session.frame_time
        found = self._find_goal(state)
        if found is None:
            self._candidate_id = None
            self._candidate_frames = 0
            self._candidate_center = None
            self._candidate_calibration = None
            return None
        _, target, action, _, _ = found
        center = np.asarray(target["center_world"], dtype=float)
        same = (target["id"] == self._candidate_id and action["calibration_id"] == self._candidate_calibration
                and self._candidate_center is not None and np.linalg.norm(center - self._candidate_center) <= .03)
        self._candidate_frames = self._candidate_frames + 1 if same else 1
        self._candidate_id = target["id"]
        self._candidate_center = center
        self._candidate_calibration = action["calibration_id"]
        return found if self._candidate_frames >= 3 else None

    def _select_action(self):
        self.session.command({"type": "release"})
        self.session.command({"type": "select", "id": self.target_id})
        self.session.command({"type": "confirm"})
        self.session.command({"type": "action_select", "action_id": self.action_id})
        self.session.command({"type": "action_confirm"})

    def _begin_trial(self, name, state, sim_time):
        if self.target_id not in self.session.selection.targets:
            self._finish("Previously selected measured target is unavailable")
            return
        self.session.command({"type": "assist_enable", "enabled": name == "assisted_approach"})
        self.session.command({"type": "eye_enable", "enabled": False})
        self._select_action()
        self._arm(state)
        self._set_phase(name, sim_time, state)

    def _complete_trial(self):
        final = [s for s in self.samples if s["elapsed_s"] >= self.TRIAL_DURATION - 2.]
        lateral = [s["lateral_error_m"] for s in final]
        reasons = sorted({s["reason"] for s in self.samples})
        error = float(np.mean(lateral)) if lateral else None
        allowed = ({"manual_control"} if self.phase == "baseline_approach" else
                   {"operator_stationary", "outside_assist_region", "predictive_alignment", "predictive_aligned"})
        invalid_samples = [s for s in self.samples if s["reason"] not in allowed or
                           (self.phase == "baseline_approach" and s.get("hold", False))]
        control_valid = bool(self.samples) and not invalid_samples
        self.results[self.phase] = {"passed": bool(control_valid and error is not None and
            (.012 < error < .05 if self.phase == "baseline_approach" else error < .016)),
            "mean_lateral_error_m": error, "all_samples": len(self.samples), "comparison_samples": len(final),
            "control_valid": control_valid, "invalid_control_samples": len(invalid_samples),
            "invalid_control_reasons": sorted({s["reason"] for s in invalid_samples}),
            "observed_reasons": reasons, "intentional_operator_offset_m": self.OFFSET_M,
            "standoff_m": self.STANDOFF_M, "comparison_standoff_range_m": [self.STANDOFF_M,self.STANDOFF_M+self.COMPARISON_APPROACH_M],
            "comparison_motion": "identical straight operator approach, no oscillation",
            "phase": "approach", "grip_command": 0.}

    def _finish(self, error=None):
        self.done = True
        if error:
            self.results["driver"] = {"passed": False, "reason": str(error)}
        self._input(clutch=False)
        self.session.command({"type": "assist_enable", "enabled": False})

    def filter_targets(self, targets):
        if self.phase == "target_lost":
            self._dropout_frames += 1
            return [t for t in targets if t["id"] != self.target_id]
        return targets

    def update(self, robot_state, sim_time):
        if self.done:
            return
        now = self.session.clock()
        if self.start_sim is None:
            self.start_sim = self.phase_start = float(sim_time)
        if now - self.wall_start > self.MAX_WALL_SECONDS or float(sim_time) - self.start_sim > self.MAX_SIM_SECONDS:
            self._finish("Bounded stack acceptance deadline exceeded")
            return
        if not self.session.config.require_action:
            self._finish("Stack acceptance requires the production action gate")
            return
        elapsed = float(sim_time) - self.phase_start
        if self.phase == "await_perception":
            self._input(clutch=False)
            found = self._stable_goal(robot_state)
            if found is None:
                if now - self.phase_wall > 45.:
                    self._finish(f"No stable fresh observed {self.preferred_label or 'target'} with calibrated {self.preferred_action} action")
                return
            _, target, action, self.normal, self.goal = found
            self.target_id, self.action_id = target["id"], action["id"]
            self.start_pose = np.asarray(robot_state["ee_position"], dtype=float)
            tangent = np.array([0., 1., 0.])
            tangent -= self.normal * float(tangent @ self.normal)
            if np.linalg.norm(tangent) < .1:
                tangent = np.cross(self.normal, [0., 0., 1.])
            self.offset = tangent / np.linalg.norm(tangent) * self.OFFSET_M
            self.target_description = {"id": self.target_id, "action_id": self.action_id,
                "semantic_label": target.get("semantic_label", target.get("label")),
                "confidence": target.get("confidence"), "pose_confidence": target.get("pose_confidence"),
                "initial_measured_center_world": list(target["center_world"]),
                "task_calibration_source": action["calibration_source"],
                "calibration_id": action["calibration_id"],
                "selection_rule": "explicit requested semantic class and calibrated action; no fallback",
                "requested_semantic_label": self.preferred_label, "requested_action": self.preferred_action,
                "fresh_observations_before_lock": self._candidate_frames,
                "max_start_frame_age_s": .2}
            self._event("observed_goal_selected", sim_time, **self.target_description, approach_world=self.goal.tolist())
            self._begin_trial("baseline_approach", robot_state, sim_time)
            return
        if self.phase in ("baseline_approach", "assisted_approach"):
            if self.session.selection.mode == "lost":
                self._complete_trial()
                self._finish("Locked measured target lost during comparison; fail-closed hold is preserved")
                return
            fraction = min(1., max(0., elapsed / 8.))
            # Both conditions reach 12 cm standoff after eight seconds, then
            # make the same straight 20 mm approach over four seconds. Quality
            # is compared during deliberate continuous motion; strict stopping
            # and reversal remain independent tests later in this driver.
            final_fraction=min(1.,max(0.,(elapsed-8.)/4.))
            approach=self.goal+self.offset+self.normal*self.COMPARISON_APPROACH_M*(1.-final_fraction)
            desired=(1.-fraction)*self.start_pose+fraction*approach
            self._input(desired, clutch=True)
            if elapsed >= self.TRIAL_DURATION:
                previous = self.phase
                self._complete_trial()
                if previous == "baseline_approach":
                    self.session.command({"type": "assist_enable", "enabled": False})
                    self.session.command({"type": "release"})
                    self._arm(robot_state)
                    self._set_phase("return_to_common_start", sim_time, robot_state)
                else:
                    self._arm(robot_state)
                    self._set_phase("stationary", sim_time, robot_state)
            return
        if self.phase == "return_to_common_start":
            self._input(self.start_pose, clutch=True)
            if elapsed >= self.RETURN_DURATION:
                error = float(np.linalg.norm(np.asarray(robot_state["ee_position"]) - self.start_pose))
                self.results["comparable_trial_start"] = {"passed": error < .02, "measured_start_error_m": error}
                self._begin_trial("assisted_approach", robot_state, sim_time)
            return
        if self.phase == "stationary":
            self._input(self.phase_pose, clutch=True)
            if elapsed >= .5:
                self.results["stationary_hold"] = {"passed": self._fault_seen and self._fault_correct}
                self._arm(robot_state)
                self._set_phase("reversal", sim_time, robot_state)
            return
        if self.phase == "reversal":
            distance = .008 * min(elapsed / .35, 1.) if elapsed <= .35 else .008 * max(0., 1. - (elapsed - .35) / .35)
            self._input(self.phase_pose - self.normal * distance, clutch=True)
            if elapsed >= .7:
                self.results["operator_reversal"] = {"passed": self._reverse_seen and self._reverse_correct,
                    "contract": "No queued progress against the latest operator direction"}
                self.session.command({"type": "action_confirm"})
                self._set_phase("confirmation_gate", sim_time, robot_state)
            return
        if self.phase == "confirmation_gate":
            self._input(self.phase_pose - self.normal * .02, clutch=True)
            if elapsed >= .3:
                self.results["confirmation_requires_new_clutch"] = {"passed": self._fault_seen and self._fault_correct}
                self._set_phase("release", sim_time, robot_state)
            return
        if self.phase == "release":
            self._input(self.phase_pose - self.normal * .02, clutch=False)
            if elapsed >= .3:
                self.results["deadman_release_hold"] = {"passed": self._fault_seen and self._fault_correct}
                self._set_phase("fresh_clutch", sim_time, robot_state)
            return
        if self.phase == "fresh_clutch":
            # The first fresh press anchors the measured EE; the next samples
            # describe a small, newly commanded retreat from that anchor.
            self._input(self.phase_pose + self.normal * min(elapsed, .5) * .01, clutch=True)
            if elapsed >= .5:
                self.results["fresh_clutch_rearms"] = {"passed": self._fresh_clutch_seen}
                self._set_phase("target_lost", sim_time, robot_state)
                self._event("detector_dropout_injected", sim_time, target_id=self.target_id)
            return
        if self.phase == "target_lost":
            self._input(self.phase_pose, clutch=True)
            if elapsed >= 1.5 and (self._dropout_frames > 0 or now - self.phase_wall > 5.):
                self.results["lost_target_hold"] = {"passed": self._fault_seen and self._fault_correct and self._dropout_frames > 0,
                    "withheld_detection_frames": self._dropout_frames, "fault_injection": "measurement boundary dropout"}
                self.session.command({"type": "stop"})
                self._set_phase("stop", sim_time, robot_state)
            return
        if self.phase == "stop":
            self._input(self.phase_pose - self.normal * .02, clutch=True)
            if elapsed >= .3:
                self.results["operator_stop_hold"] = {"passed": self._fault_seen and self._fault_correct}
                self._finish()

    def observe(self, robot_state, reference, sim_time):
        if self.done:
            return
        self.invalid_states += int(not robot_state.get("valid", False))
        self.max_joint_violation = max(self.max_joint_violation, float(robot_state.get("joint_limit_violation", 0.)))
        self.contact_stop_steps += int(robot_state.get("collision_stop", False))
        self.max_correction_m = max(self.max_correction_m, float(self.session.control.get("correction_m", 0.)))
        self.max_rotation_correction_rad = max(self.max_rotation_correction_rad, float(self.session.control.get("orientation_correction_rad", 0.)))
        self.mechanism_telemetry = robot_state.get("mechanisms", robot_state.get("interaction_state", self.mechanism_telemetry))
        reason = self.session.control.get("reason", "")
        elapsed = float(sim_time) - self.phase_start
        target = self.session.selection.targets.get(self.target_id)
        if target is not None and self.target_description is not None:
            drift = np.linalg.norm(np.asarray(target["center_world"]) - self.target_description["initial_measured_center_world"])
            self.max_target_drift_m = max(self.max_target_drift_m, float(drift))
        if self.phase in ("baseline_approach", "assisted_approach"):
            delta = np.asarray(robot_state["ee_position"]) - self.goal
            lateral = delta - self.normal * float(delta @ self.normal)
            self.samples.append({"elapsed_s": elapsed, "lateral_error_m": float(np.linalg.norm(lateral)),
                                 "reason": reason, "hold": bool(reference.get("hold", False))})
        expected = {"stationary": {"operator_stationary"}, "confirmation_gate": {"fresh_clutch_required"},
                    "release": {"fresh_clutch_required", "deadman_released"}, "target_lost": {"target_lost"},
                    "stop": {"stopped"}}.get(self.phase, set())
        if reason in expected:
            self._fault_seen = True
            self._fault_correct &= bool(reference.get("hold") and np.linalg.norm(
                np.asarray(reference["ee_target_world"]) - self.session.robot["ee_position"]) < 1e-8)
        if self.phase == "reversal" and .38 < elapsed < .68:
            diagnostics = self.session.control.get("prediction", {})
            velocity = np.asarray(diagnostics.get("operator_velocity_m_s", [0., 0., 0.]))
            if float(velocity @ self.normal) > .001:
                self._reverse_seen = True
                delta = np.asarray(reference["ee_target_world"]) - self.session.robot["ee_position"]
                self._reverse_correct &= float(delta @ self.normal) >= -1e-8
        if self.phase == "fresh_clutch" and not reference.get("hold") and not self.session.rearm_required:
            self._fresh_clutch_seen = True

    def report(self):
        checks = dict(self.results)
        baseline = checks.get("baseline_approach", {}).get("mean_lateral_error_m")
        assisted = checks.get("assisted_approach", {}).get("mean_lateral_error_m")
        improvement = 1. - assisted / baseline if baseline and assisted is not None else None
        comparison_valid = all(checks.get(name, {}).get("control_valid", False)
                               for name in ("baseline_approach", "assisted_approach"))
        checks["relative_alignment_improvement"] = {"passed": comparison_valid and improvement is not None and improvement > .4,
            "comparison_control_valid": comparison_valid,
            "fractional_lateral_error_reduction": improvement}
        checks["measured_control_limits"] = {"passed": self.invalid_states == 0 and self.max_joint_violation <= .035 and
            self.max_correction_m <= self.session.config.max_correction + 1e-8 and
            self.max_rotation_correction_rad <= self.session.config.max_orientation_correction + 1e-8,
            "invalid_states": self.invalid_states, "max_joint_violation_rad": self.max_joint_violation,
            "contact_stop_steps": self.contact_stop_steps, "max_correction_m": self.max_correction_m,
            "max_orientation_correction_rad": self.max_rotation_correction_rad}
        return {"schema_version": 1, "complete": self.done,
            "passed": bool(self.done and checks and all(c.get("passed", False) for c in checks.values())),
            "phase": self.phase, "checks": checks, "events": list(self.events),
            "target": self.target_description, "goal_approach_world": None if self.goal is None else self.goal.tolist(),
            "requested_task": {"semantic_label": self.preferred_label, "action_id": self.preferred_action,
                               "fallback_allowed": False, "required_fresh_camera_frames": 3,
                               "measured_cad_pose_required": self.require_cad_pose},
            "max_observed_target_drift_m": self.max_target_drift_m,
            "wall_duration_s": self.session.clock() - self.wall_start,
            "mechanism_telemetry_observation_only": self.mechanism_telemetry,
            "perception_diagnostics": dict(self.session.perception_diagnostics),
            "input_provenance": "synthetic controller through production API; observed perception targets only",
            "limits": ["No live headset or human usability validation.",
                "Compared approach alignment at 10 cm standoff; no grasp/contact/opening success claim.",
                "Action execution and axial rotation are tested separately in deterministic controller tests.",
                "Mechanics are explicitly declared fixture capabilities, not inferred from image semantics.",
                "Target loss is a disclosed detector-output dropout injection.",
                "A learned perception claim requires diagnostics from an actually loaded vision checkpoint."]}
