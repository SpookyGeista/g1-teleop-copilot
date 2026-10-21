"""Persistent CPU guard for an explicitly observed, static simulation scene.

It returns an expiring proposed q14 command; it never publishes or executes it.
The caller must check the lease at consumption and retain a separate actuator
hold/stop mechanism. Scope is numerical original-URDF prechecking, not hardware
safety certification, dynamic collision avoidance, or a grasp controller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .arm_collision import CollisionConfig, OriginalArmCollision, _provenance
from .controller import rigid
from .full_ik import FullStateIKSolver
from .kinematics import (OriginalIKValidator, ARM_JOINT_NAMES, sha256,
    ALREADY_AT_TARGET_POSITION_TOLERANCE_M, ALREADY_AT_TARGET_ORIENTATION_TOLERANCE_RAD)

JOINT_REFERENCE_SCHEMA = 'punktnull.sim_clutch_joint_reference.v1'


@dataclass(frozen=True)
class MotionGuardConfig:
    max_operator_age_s: float = .15
    max_robot_state_age_s: float = .15
    max_scene_age_s: float = .15
    future_tolerance_s: float = .01
    max_joint_command_change_rad: float = .15

    def __post_init__(self):
        values = [self.max_operator_age_s, self.max_robot_state_age_s, self.max_scene_age_s,
                  self.future_tolerance_s, self.max_joint_command_change_rad]
        if (not np.isfinite(values).all() or not all(0 < value <= .5 for value in values[:3])
                or not 0 <= self.future_tolerance_s <= .02
                or not 0 < self.max_joint_command_change_rad <= .3):
            raise ValueError("Invalid motion guard configuration")


class ActualMotionGuard:
    """Reusable original IK + full q43 static-scene short-path collision gate."""

    def __init__(self, repository, config: MotionGuardConfig | None = None,
                 collision_config: CollisionConfig | None = None):
        self.config = config or MotionGuardConfig()
        self.ik = OriginalIKValidator(repository)
        self.collision = OriginalArmCollision(repository, collision_config)
        self.full_ik = FullStateIKSolver(self.collision.model)
        self.sequence = 0
        self.manifest = {"guard_sha256": sha256(__file__), "configuration": asdict(self.config),
                         "ik": self.ik.manifest, "collision": self.collision.manifest,
                         "full_state_ik": self.full_ik.manifest,
                         "kinematic_order": ["verified_clutch_reference", "strict_measured_full_q43_noop",
                            "bounded_full_q43_DLS_current_seed", "original_IK_fallback_seed_then_same_strict_full_q43_DLS"],
                         "robot_command_sent": False, "scope": "static_simulation_numerical_precheck"}

    def _already_at_full_state_target(self, left, right, full_q):
        """Preserve measured arms only when BOTH full-q43 EE poses agree.

        The reduced IK fixes waist joints at zero, unlike a measured robot.
        This check uses the separate original full-URDF model, including the
        measured waist. It does not skip the later body-FK or collision gate.
        """
        started = time.perf_counter()
        model, data, pin = self.collision.model, self.collision.data, self.collision.pin
        pin.framesForwardKinematics(model, data, full_q)
        tcp = np.eye(4); tcp[0, 3] = .05
        residuals = {}
        for side, target in (('left', left), ('right', right)):
            fk = data.oMf[model.getFrameId(side+'_wrist_yaw_link')].homogeneous @ tcp
            residuals[side+'_position_error_m'] = float(np.linalg.norm(fk[:3,3]-target[:3,3]))
            residuals[side+'_orientation_error_rad'] = float(Rotation.from_matrix(fk[:3,:3] @ target[:3,:3].T).magnitude())
        position_limit = min(ALREADY_AT_TARGET_POSITION_TOLERANCE_M, self.ik.config.max_position_error_m)
        rotation_limit = min(ALREADY_AT_TARGET_ORIENTATION_TOLERANCE_RAD, self.ik.config.max_orientation_error_rad)
        if (not np.isfinite(list(residuals.values())).all()
                or max(residuals[s+'_position_error_m'] for s in ('left','right')) > position_limit
                or max(residuals[s+'_orientation_error_rad'] for s in ('left','right')) > rotation_limit):
            return None
        current = full_q[self.collision.arm_indices].tolist()
        return dict(ik_valid=True, collision_valid=False, reason='already_at_target_full_q43',
            scope='independent_original_full_q43_current_pose_FK', validation_method='independent_current_full_q43_FK',
            solver_invoked=False, q_solution=current, q_initial=list(current),
            q_initial_43=full_q.tolist(), limits_valid=True,
            left_target_base_ee=left.tolist(), right_target_base_ee=right.tolist(),
            iterations=[], numerical_convergence_steps=0,
            final=dict(iteration=0,solver_success=None,solver_status='not_run_already_at_full_q43_target',
                limits_valid=True,**residuals),
            elapsed_s=float(time.perf_counter()-started),
            joint_state_freshness_validated=False, trajectory_validated=False)

    def _check_joint_reference(self, reference, left, right, full_q, *, now, clock_id):
        """Test an old clutch posture as a hypothesis against CURRENT full q43.

        Its original timestamp is provenance, not a measurement freshness lease.
        All non-arm joints remain current. A matching reference only avoids the
        optimizer; the caller still checks present state, limits, motion and
        collisions. Malformed or nonmatching optional hints use normal IK.
        """
        evidence = {'supplied': reference is not None, 'used': False,
                    'reason': 'not_supplied', 'motion_authorized': False}
        if reference is None:
            return None, evidence
        try:
            _provenance(reference, 'joint_reference')
            if reference.get('schema') != JOINT_REFERENCE_SCHEMA:
                raise ValueError('joint_reference_schema_invalid')
            if reference['clock_id'] != clock_id:
                raise ValueError('joint_reference_clock_mismatch')
            if reference['acquisition_timestamp'] > now + self.config.future_tolerance_s:
                raise ValueError('joint_reference_acquisition_in_future')
            if reference.get('joint_names') != list(ARM_JOINT_NAMES):
                raise ValueError('joint_reference_joint_order_invalid')
            values = reference.get('q14')
            if (not isinstance(values, list) or len(values) != 14 or
                    any(isinstance(x, bool) or not isinstance(x, (int, float))
                        or not np.isfinite(x) for x in values)):
                raise ValueError('joint_reference_q14_invalid')
            named = {name: float(full_q[self.collision.indices[name]]) for name in self.collision.names}
            named.update(zip(ARM_JOINT_NAMES, values))
            trial = self.collision._q(named)  # Full original limits still apply.
            ik = self._already_at_full_state_target(left, right, trial)
            evidence.update(source=reference['source'], clock_id=reference['clock_id'],
                            acquisition_timestamp=reference['acquisition_timestamp'],
                            calibration_id=reference.get('calibration_id'),
                            reference_age_s=float(now-reference['acquisition_timestamp']))
            if ik is None:
                evidence['reason'] = 'joint_reference_does_not_reproduce_requested_EEs'
                return None, evidence
            # The hypothetical solution is never relabelled as measured state.
            ik.update(reason='clutch_joint_reference_matches_full_q43_target',
                      scope='independent_original_full_q43_joint_reference_FK',
                      validation_method='independent_joint_reference_full_q43_FK',
                      q_initial=full_q[self.collision.arm_indices].tolist(),
                      q_initial_43=full_q.tolist(), q_solution_43=trial.tolist(),
                      joint_reference_provenance={key: reference[key] for key in
                          ('schema', 'source', 'acquisition_timestamp', 'clock_id')})
            ik['final']['solver_status'] = 'not_run_verified_clutch_joint_reference'
            evidence.update(used=True, reason=ik['reason'])
            return ik, evidence
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            evidence['reason'] = str(exc)
            return None, evidence

    def _quantized_command_fk(self, ik, left, right, full_q):
        """Revalidate the exact HG Float32 arm values, without transport access.

        Quantization is a new numerical candidate, never a granted tolerance.
        The current non-arm state is retained bit for bit; the ordinary current
        body-FK, complete-scene, path and lease gates still run afterwards.
        """
        evidence = dict(valid=False, command_format='unitree_float32',
            encoding='IEEE754_binary32_little_endian_arm_order',
            joint_names=list(ARM_JOINT_NAMES), motion_authorized=False,
            transport_serialized=False, crc_validated=False,
            reason='unvalidated', fixed_nonarm_joints_unchanged=False)
        updated = deepcopy(ik)
        try:
            raw = np.asarray(ik['q_solution'], dtype=float)
            if raw.shape != (14,) or not np.isfinite(raw).all():
                raise ValueError('unitree_float32_q14_invalid')
            with np.errstate(over='ignore', invalid='ignore'):
                packed = raw.astype('<f4')
            command = packed.astype(np.float64)
            if not np.isfinite(command).all():
                raise ValueError('unitree_float32_q14_nonfinite_after_quantization')
            full = full_q.copy()
            full[self.collision.arm_indices] = command
            nonarm = np.ones(len(full), dtype=bool)
            nonarm[self.collision.arm_indices] = False
            model, pin = self.collision.model, self.collision.pin
            limits_valid = bool(np.all(full >= model.lowerPositionLimit)
                                and np.all(full <= model.upperPositionLimit))
            evidence.update(q14_before_quantization=raw.tolist(),
                q14_after_quantization=command.tolist(),
                q14_float32_le_hex=packed.tobytes().hex(),
                q14_float32_le_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),
                maximum_quantization_error_rad=float(np.max(abs(command-raw))),
                limits_valid=limits_valid,
                fixed_nonarm_joints_unchanged=bool(np.array_equal(full[nonarm],full_q[nonarm])))
            updated.update(q_solution=command.tolist(), q_solution_43=full.tolist(),
                q_initial=full_q[self.collision.arm_indices].tolist(), q_initial_43=full_q.tolist(),
                limits_valid=limits_valid, command_format='unitree_float32',
                prequantization_final=deepcopy(ik.get('final')),
                prequantization_validation_method=ik.get('validation_method'),
                prequantization_scope=ik.get('scope'),
                scope='independent_original_full_q43_quantized_arm_command_FK',
                validation_method='independent_unitree_float32_full_q43_FK',
                max_joint_delta_rad=float(np.max(abs(command-full_q[self.collision.arm_indices]))),
                fixed_nonarm_joints_unchanged=evidence['fixed_nonarm_joints_unchanged'])
            if not limits_valid:
                raise ValueError('unitree_float32_full_q43_outside_hard_joint_limits')
            # Use separate FK data rather than accepting any optimizer residual.
            data = model.createData()
            pin.framesForwardKinematics(model, data, full)
            tcp = np.eye(4); tcp[0,3] = .05
            residuals = {}
            for side, target in (('left',left),('right',right)):
                fk = data.oMf[model.getFrameId(side+'_wrist_yaw_link')].homogeneous @ tcp
                residuals[side+'_position_error_m'] = float(np.linalg.norm(fk[:3,3]-target[:3,3]))
                residuals[side+'_orientation_error_rad'] = float(
                    Rotation.from_matrix(fk[:3,:3]@target[:3,:3].T).magnitude())
            position_limit = min(.0005, self.full_ik.config.position_tolerance_m)
            orientation_limit = min(float(np.deg2rad(.5)), self.full_ik.config.orientation_tolerance_rad)
            evidence.update(residuals=residuals, position_tolerance_m=position_limit,
                            orientation_tolerance_rad=orientation_limit)
            updated['final'] = dict(solver_success=None,
                solver_status='not_run_independent_quantized_command_FK',
                limits_valid=limits_valid, **residuals)
            if (not np.isfinite(list(residuals.values())).all()
                    or any(residuals[s+'_position_error_m'] > position_limit or
                           residuals[s+'_orientation_error_rad'] > orientation_limit
                           for s in ('left','right'))):
                raise ValueError('unitree_float32_strict_full_q43_FK_rejected')
            evidence.update(valid=True, reason='unitree_float32_strict_full_q43_FK_valid')
            updated.update(ik_valid=True, reason=evidence['reason'])
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            evidence['reason'] = str(exc)
            updated.update(ik_valid=False, reason=evidence['reason'])
        return updated, evidence

    def validate(self, left_base_ee, right_base_ee, robot_state, scene, *,
                 acquisition_timestamp, clock_id, now, tracking_valid=False,
                 joint_reference=None, command_format='float64'):
        """Return allowed/q14_command or HOLD, with a named-clock lease.

        `now`, acquisition times and consumption time must use `clock_id`.
        A state is an actual complete q43/base/body-pose snapshot; source metadata
        may not be refreshed to make old measurements appear current.
        `unitree_float32` additionally validates the exact quantized arm values;
        it does not serialize a LowCmd, compute its CRC or open any transport.
        """
        started = time.monotonic()
        self.sequence += 1
        result = {"allowed": False, "hold_required": True, "q14_command": None,
                  "reason": "unvalidated", "sequence": self.sequence,
                  "clock_id": clock_id, "valid_until": None,
                  "robot_command_sent": False, "scope": "static_simulation_numerical_precheck",
                  "fingers_commanded": False, "requires_unchanged_scene_base_and_fingers": True}
        try:
            if command_format not in ('float64', 'unitree_float32'):
                raise ValueError('unsupported_command_format')
            if tracking_valid is not True:
                raise ValueError("operator_tracking_invalid_or_unspecified")
            if isinstance(now, bool) or not isinstance(now, (int, float)) or not np.isfinite(now):
                raise ValueError("now_must_be_finite_in_named_clock")
            operator = {"source": "caller_operator_acquisition", "acquisition_timestamp": acquisition_timestamp,
                        "clock_id": clock_id}
            _provenance(operator, "operator")
            deadlines = []
            for name, state, age in (("operator", operator, self.config.max_operator_age_s),
                                     ("robot_state", robot_state, self.config.max_robot_state_age_s),
                                     ("scene", scene, self.config.max_scene_age_s)):
                _provenance(state, name)
                if state["clock_id"] != clock_id:
                    raise ValueError(name + "_clock_mismatch")
                timestamp = state["acquisition_timestamp"]
                if timestamp > now + self.config.future_tolerance_s:
                    raise ValueError(name + "_acquisition_in_future")
                if now - timestamp > age:
                    raise ValueError(name + "_stale")
                deadlines.append(timestamp + age)
            deadline = float(min(deadlines))
            result["valid_until"] = deadline
            left, right = rigid(left_base_ee, "left_base_ee"), rigid(right_base_ee, "right_base_ee")
            # Require all state before expensive IK; no default OPEN fingers.
            full_q = self.collision._q(robot_state.get("joint_positions"))
            current = full_q[self.collision.arm_indices]
            ik, reference_evidence = self._check_joint_reference(
                joint_reference, left, right, full_q, now=now, clock_id=clock_id)
            result['joint_reference_validation'] = reference_evidence
            if ik is None:
                ik = self._already_at_full_state_target(left, right, full_q)
            if ik is None:
                ik = self.full_ik.solve(left, right, full_q)
                result['direct_full_state_ik_attempt'] = {
                    key:ik.get(key) for key in ('ik_valid','reason','elapsed_s','numerical_convergence_steps')}
                if ik['ik_valid'] is not True:
                    seed = self.ik.validate_pose(right, left, current)
                    result['original_ik_fallback_seed'] = seed
                    # Original regularized IK never authorizes a command here.
                    # Only a second strict full-state solve can pass it onward.
                    ik = self.full_ik.solve(left, right, full_q, seed_q14=seed.get('q_solution'))
                    ik['seed_source'] = 'original_reduced_IK_numerical_seed_only'
            result["ik_validation"] = ik
            if ik["ik_valid"] is not True:
                raise ValueError("command_kinematic_validation_rejected:" + ik["reason"])
            if command_format == 'unitree_float32':
                ik, wire = self._quantized_command_fk(ik, left, right, full_q)
                result.update(command_format=command_format, wire_format_validation=wire,
                              ik_validation=ik)
                if wire['valid'] is not True:
                    raise ValueError(wire['reason'])
            command = np.asarray(ik["q_solution"])
            max_change = float(np.max(abs(command - current)))
            result["max_joint_command_change_rad"] = max_change
            if max_change > self.config.max_joint_command_change_rad:
                raise ValueError("command_joint_step_exceeds_bound")
            candidate = {"candidate_id": f"command-{self.sequence}", "base_ee": right.tolist(),
                         "ik_valid": True, "collision_valid": False, "kinematic_validation": ik}
            checked = self.collision.validate_request({"batch": {"candidates": [candidate]},
                                                       "robot_state": robot_state, "scene": scene})
            motion = checked["candidates"][0]["arm_collision_validation"]
            result["collision_validation"] = motion
            result["current_state_precheck"] = checked["arm_collision_validation"].get("current_state_precheck")
            result["urdf_vs_observed_frames"] = checked["arm_collision_validation"].get("urdf_vs_observed_frames")
            result["snapshot_sha256"] = checked["arm_collision_validation"]["snapshot_sha256"]
            if motion["collision_valid"] is not True:
                raise ValueError("command_collision_rejected:" + motion["reason"])
            if now + (time.monotonic() - started) >= deadline:
                raise ValueError("command_lease_expired_during_validation")
            result.update(allowed=True, hold_required=False, q14_command=command.tolist(),
                          reason="full_q43_FK_and_conservative_short_joint_path_valid")
        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
            result["reason"] = str(exc)
        result["elapsed_s"] = float(time.monotonic() - started)
        result["validated_at"] = float(now + result["elapsed_s"]) if isinstance(now, (int, float)) and np.isfinite(now) else None
        return result


def command_is_current(result, *, now, clock_id):
    """Mandatory consumer-side lease check; no side effects or implicit refresh."""
    return bool(isinstance(result, dict) and result.get("allowed") is True
                and result.get("hold_required") is False and clock_id == result.get("clock_id")
                and isinstance(now, (int, float)) and not isinstance(now, bool) and np.isfinite(now)
                and isinstance(result.get("validated_at"), (int, float))
                and isinstance(result.get("valid_until"), (int, float))
                and result["validated_at"] <= now < result["valid_until"])
