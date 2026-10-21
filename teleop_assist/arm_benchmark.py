"""Deterministic synthetic operator and honest measured-state arm metrics.

Geometry in this benchmark is an explicitly supplied ideal line in free space.
It does not exercise perception, networking, a headset, grasping or force control.
AssistSession uses an injected simulation clock so different solver runtimes see
the same input times; wall durations are recorded separately by the runner.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .core import AssistConfig, AssistSession, XR_TO_ROBOT


SCHEMA = 'punktnull.arm_benchmark.v1'
SCENARIO_VERSION = 'reference_pose_cartesian_paths_v1'
INPUT_HZ = 50.
SCENARIOS = ('pose_loop', 'misalignment')


@dataclass(frozen=True)
class PathPoint:
    phase: str
    raw_offset: np.ndarray
    intended_offset: np.ndarray
    rotation_vector: np.ndarray


def _ease(value):
    u = float(np.clip(value, 0., 1.))
    return u*u*u*(10.+u*(-15.+6.*u))


def scenario_duration(name):
    if name == 'pose_loop':
        return 8.5
    if name == 'misalignment':
        return 6.2
    raise ValueError('Unknown benchmark scenario')


def scenario_point(name, elapsed):
    """C2 smooth move/hold/reverse paths, independent of any measured result."""
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Scenario time must be finite and nonnegative')
    zero = np.zeros(3)
    if name == 'pose_loop':
        translation = np.array([.04, -.02, .025])
        rotation = np.array([0., .12, .08])
        if elapsed < .4:
            return PathPoint('start_hold', zero, zero, zero)
        if elapsed < 2.4:
            offset = translation*_ease((elapsed-.4)/2.)
            return PathPoint('translation', offset, offset, zero)
        if elapsed < 3.2:
            return PathPoint('translation_hold', translation, translation, zero)
        if elapsed < 4.7:
            return PathPoint('rotation', translation, translation, rotation*_ease((elapsed-3.2)/1.5))
        if elapsed < 5.5:
            return PathPoint('rotation_hold', translation, translation, rotation)
        if elapsed < 7.5:
            scale = 1.-_ease((elapsed-5.5)/2.)
            return PathPoint('reverse', translation*scale, translation*scale, rotation*scale)
        return PathPoint('end_hold', zero, zero, zero)
    if name == 'misalignment':
        if elapsed < .4:
            scale, phase = 0., 'start_hold'
        elif elapsed < 2.4:
            scale, phase = _ease((elapsed-.4)/2.), 'translation'
        elif elapsed < 3.2:
            scale, phase = 1., 'translation_hold'
        elif elapsed < 5.2:
            scale, phase = 1.-_ease((elapsed-3.2)/2.), 'reverse'
        else:
            scale, phase = 0., 'end_hold'
        intended = np.array([.06*scale, 0., 0.])
        return PathPoint(phase, intended+np.array([0., .025*scale, 0.]), intended, zero)
    raise ValueError('Unknown benchmark scenario')


def scenario_contract():
    return {'version': SCENARIO_VERSION, 'input_hz': INPUT_HZ,
            'start_pose': 'reference', 'geometry_source': 'synthetic_known_line_not_perception',
            'pose_loop': {'duration_s': 8.5, 'translation_m': [.04, -.02, .025],
                          'rotation_vector_world_rad': [0., .12, .08],
                          'assistance': 'off_in_both_modes'},
            'misalignment': {'duration_s': 6.2, 'forward_m': .06,
                             'raw_transverse_error_m': .025, 'axis_world': [1., 0., 0.],
                             'action': 'configured_linear_approach',
                             'assistance': 'production_predictive_servo_when_geometric'},
            'clock': 'AssistSession receives simulation time; wall time is separate',
            'reset': 'new backend per solver/mode; physical return between scenarios; no live state teleport'}


class SyntheticArmOperator:
    """The normal calibration, target/action and XR input API, without a server."""
    def __init__(self, assistance='manual', event_sink=None):
        if assistance not in ('manual', 'geometric'):
            raise ValueError('Unknown assistance mode')
        self.assistance = assistance
        self.now = 1000.
        self.session = AssistSession(AssistConfig(require_action=True), clock=lambda: self.now,
                                     event_sink=event_sink)
        self.seq = 0
        self.frame_id = 0
        self.scenario = None

    def begin(self, scenario, state, sim_elapsed):
        scenario_duration(scenario)
        self.scenario = scenario
        self.now = 1000.+float(sim_elapsed)
        self.origin = np.asarray(state['ee_position'], float).copy()
        self.orientation = Rotation.from_quat(state['ee_quaternion'])
        self.session.robot = dict(state)
        self.session.command({'type': 'release'})
        self.session.command({'type': 'assist_enable', 'enabled':
                              self.assistance == 'geometric' and scenario == 'misalignment'})
        self._frame()
        self.session.command({'type': 'select', 'id': 'synthetic_benchmark_line'})
        self.session.command({'type': 'confirm'})
        self.session.command({'type': 'action_select', 'action_id': 'synthetic_approach'})
        self.session.command({'type': 'action_confirm'})
        initial = scenario_point(scenario, 0.)
        self._input(initial, clutch=False)
        self.session.command({'type': 'calibrate'})
        # Calibration invalidates the old clutch edge. Production code requires
        # a new released sample before the next explicit active sample.
        self._input(initial, clutch=False)
        self._input(initial, clutch=True)

    def _frame(self):
        self.frame_id += 1
        goal = self.origin+np.array([.06, 0., 0.])
        target = {'id': 'synthetic_benchmark_line', 'label': 'synthetic_free_space_line',
                  'bbox_px': [10., 10., 90., 90.], 'center_world': goal.tolist(),
                  'normal_world': [-1., 0., 0.], 'confidence': 1., 'position_sigma_m': 0.,
                  'orientation_sigma_rad': 0., 'stale': False,
                  'source': 'explicit_synthetic_benchmark_geometry',
                  'task_calibration': {'source': 'configured_fixture',
                      'id': 'synthetic_arm_benchmark_fixture_v1', 'actions': [{
                          'id': 'synthetic_approach', 'kind': 'linear',
                          'label': 'Synthetic approach along known free-space line',
                          'grasp_position_world': goal.tolist(),
                          'approach_axis_world': [1., 0., 0.],
                          'motion_axis_world': [1., 0., 0.],
                          'grasp_quaternion_xyzw': self.orientation.as_quat().tolist(),
                          'execution_ready': False}]}}
        self.session.update_frame([target], self.frame_id, self.now,
                                  {'width': 100, 'height': 100},
                                  {'synthetic_geometry': True, 'perception_evaluated': False})

    def _input(self, point, clutch):
        self.seq += 1
        basis = Rotation.from_matrix(XR_TO_ROBOT)
        hand_rotation = basis.inv()*Rotation.from_rotvec(point.rotation_vector)*basis
        packet = {'source': 'synthetic_arm_benchmark', 'session_id': 'benchmark_'+self.scenario,
                  'seq': self.seq, 'orientation_mode': 'pose',
                  'right': {'valid': True, 'position': (XR_TO_ROBOT.T@point.raw_offset).tolist(),
                            'quaternion_xyzw': hand_rotation.as_quat().tolist()},
                  'xr_to_world_rotation': XR_TO_ROBOT.tolist(), 'clutch': bool(clutch),
                  'squeeze': float(clutch), 'trigger': 0., 'axes': [0., 0.],
                  'primary': False, 'secondary': False, 'frame_id': self.frame_id}
        if not self.session.command({'type': 'input', 'sample': packet})['accepted']:
            raise RuntimeError('Production session rejected a synthetic benchmark packet')
        return packet

    def reference(self, elapsed, sim_elapsed, state, dt, *, send_packet):
        self.now = 1000.+float(sim_elapsed)
        point = scenario_point(self.scenario, elapsed)
        if send_packet:
            self._frame()
            self._input(point, clutch=True)
        reference = self.session.step(state, dt, now=self.now)
        quaternion = (Rotation.from_rotvec(point.rotation_vector)*self.orientation).as_quat()
        expected = {'phase': point.phase, 'intended_position': (self.origin+point.intended_offset).tolist(),
                    'raw_position': (self.origin+point.raw_offset).tolist(),
                    'intended_quaternion': quaternion.tolist(), 'scenario_origin': self.origin.tolist(),
                    'initial_quaternion': self.orientation.as_quat().tolist(),
                    'session_reason': self.session.control['reason'],
                    'assist_active': self.session.control['active']}
        return reference, expected


def _rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, float)**2)))


def trial_metrics(samples, scenario, assistance):
    """Never score stationary, invalid or interrupted trajectories as success."""
    if not samples:
        return {'passed': False, 'failures': ['no_measured_samples'], 'samples': 0}
    try:
        positions = np.array([s['ee_position'] for s in samples], float)
        intended = np.array([s['intended_position'] for s in samples], float)
        raw = np.array([s['raw_position'] for s in samples], float)
        command = np.array([s['command_position'] for s in samples], float)
        velocity = np.array([s['arm_velocities'] for s in samples], float)
        joints = np.array([s['arm_joints'] for s in samples], float)
        times = np.array([s['scenario_time'] for s in samples], float)
        quaternions = np.array([s['ee_quaternion'] for s in samples], float)
        intended_quaternions = np.array([s['intended_quaternion'] for s in samples], float)
        contacts = np.array([s['contact_force_max'] for s in samples], float)
        if (positions.shape != (len(samples), 3) or velocity.shape != (len(samples), 7)
                or joints.shape != velocity.shape or len(samples) < 2
                or any(not np.isfinite(v).all() for v in
                       (positions, intended, raw, command, velocity, joints, times,
                        quaternions, intended_quaternions, contacts))
                or np.any(np.diff(times) <= 0)):
            raise ValueError('Invalid measured array shape, finite values or time order')
        angular = (Rotation.from_quat(quaternions)*Rotation.from_quat(intended_quaternions).inv()).magnitude()
    except (KeyError, ValueError, TypeError) as error:
        return {'passed': False, 'failures': ['invalid_measured_telemetry'], 'error': str(error),
                'samples': len(samples)}
    error = np.linalg.norm(positions-intended, axis=1)
    command_error = np.linalg.norm(positions-command, axis=1)
    dt = np.diff(times)
    acceleration = np.diff(velocity, axis=0)/dt[:, None]
    jerk = np.diff(acceleration, axis=0)/((dt[1:]+dt[:-1])*.5)[:, None]
    origin = np.asarray(samples[0]['scenario_origin'])
    peak_progress = float(np.max(positions[:, 0]-origin[0]))
    expected_progress = .04 if scenario == 'pose_loop' else .06
    orientation_from_start = (Rotation.from_quat(quaternions)*
                              Rotation.from_quat(samples[0]['initial_quaternion']).inv()).magnitude()
    hold_indices = [i for i, s in enumerate(samples) if s['phase'] == 'translation_hold']
    hold_tail = hold_indices[-max(1, len(hold_indices)//2):]
    lateral = np.linalg.norm((positions-origin)[:, 1:], axis=1)
    solver_rows = [s['arm_solver'] for s in samples if isinstance(s.get('arm_solver'), dict)]
    unique_solves = {}
    for row in solver_rows:
        if (row.get('status') in ('solved', 'rejected') and row.get('solve_ms') is not None
                and row.get('request_id') is not None):
            # Held/reused results are not new solves. A later hold carrying an
            # old request id must not replace the original measured solve cost.
            unique_solves.setdefault(row['request_id'], row)
    solve_ms = [float(s['solve_ms']) for s in unique_solves.values()
                if math.isfinite(float(s['solve_ms'])) and float(s['solve_ms']) >= 0]
    solver_failures = [s for s in solver_rows if s.get('status') in
                       ('error', 'failed', 'invalid_solution', 'timeout', 'stale', 'rejected')]
    result_age_ms = [float(s['result_age_s'])*1000 for s in unique_solves.values()
                     if s.get('result_age_s') is not None and math.isfinite(float(s['result_age_s']))]
    failures = []
    if times[-1] < scenario_duration(scenario)-.011:
        failures.append('scenario_incomplete')
    if not all(s['valid'] for s in samples):
        failures.append('invalid_robot_state')
    if contacts.max() > 18.:
        failures.append('contact_guard_triggered')
    if any(float(s.get('joint_limit_violation', 0.)) > .035 for s in samples):
        failures.append('joint_limit_violation')
    if peak_progress < expected_progress*.7:
        failures.append('insufficient_measured_forward_progress')
    if error[-1] > .030:
        failures.append('return_error_over_30mm')
    if _rms(command_error) > .040:
        failures.append('command_tracking_rmse_over_40mm')
    if solver_failures:
        failures.append('solver_failure')
    if scenario == 'pose_loop' and float(orientation_from_start.max()) < .070:
        failures.append('insufficient_measured_rotation')
    if scenario == 'misalignment' and assistance == 'geometric' and (
            not hold_tail or float(np.mean(lateral[hold_tail])) > .015):
        failures.append('assisted_lateral_error_over_15mm')
    result = {'passed': not failures, 'failures': failures, 'samples': len(samples),
              'completed_sim_seconds': float(times[-1]),
              'intended_position_rmse_m': _rms(error),
              'raw_operator_position_rmse_m': _rms(np.linalg.norm(positions-raw, axis=1)),
              'command_tracking_rmse_m': _rms(command_error),
              'final_intended_error_m': float(error[-1]),
              'peak_forward_progress_m': peak_progress, 'required_forward_progress_m': expected_progress*.7,
              'intended_orientation_rmse_rad': _rms(angular),
              'final_orientation_error_rad': float(angular[-1]),
              'peak_measured_rotation_rad': float(orientation_from_start.max()),
              'arm_joint_velocity_rms_rad_s': _rms(velocity),
              'arm_joint_velocity_max_rad_s': float(np.max(np.abs(velocity))),
              'arm_joint_acceleration_rms_rad_s2': _rms(acceleration),
              'arm_joint_jerk_rms_rad_s3': _rms(jerk) if len(jerk) else None,
              'max_measured_joint_step_rad': float(np.max(np.abs(np.diff(joints, axis=0)))),
              'contact_force_max_N': float(contacts.max()),
              'invalid_states': sum(not s['valid'] for s in samples),
              'session_reasons': sorted({s['session_reason'] for s in samples}),
              'assist_active_samples': sum(s['assist_active'] for s in samples),
              'solver_failures': len(solver_failures), 'unique_solver_calls': len(solve_ms),
              'solver_status_sample_counts': {name: sum(str(s.get('status')) == name for s in solver_rows)
                                              for name in sorted({str(s.get('status')) for s in solver_rows})},
              'solver_time_mean_ms': float(np.mean(solve_ms)) if solve_ms else None,
              'solver_time_p95_ms': float(np.percentile(solve_ms, 95)) if solve_ms else None,
              'solver_time_max_ms': float(np.max(solve_ms)) if solve_ms else None,
              'solver_input_to_result_p95_ms': float(np.percentile(result_age_ms, 95)) if result_age_ms else None,
              'solver_input_to_result_max_ms': float(np.max(result_age_ms)) if result_age_ms else None}
    if scenario == 'misalignment':
        result['outbound_hold_lateral_error_m'] = float(np.mean(lateral[hold_tail])) if hold_tail else None
    return result


def compare_trials(manual, geometric):
    """An apparent reduction is not accepted when either physical trial failed."""
    failures = []
    if not manual.get('passed') or not geometric.get('passed'):
        failures.append('physical_trial_failed')
    raw = manual.get('outbound_hold_lateral_error_m')
    corrected = geometric.get('outbound_hold_lateral_error_m')
    if any(v is None or not math.isfinite(v) or v < 0 for v in (raw, corrected)):
        return {'passed': False, 'failures': failures+['missing_lateral_metric'], 'reduction_fraction': None}
    if raw < .010:
        failures.append('manual_error_too_small_to_evaluate')
    reduction = (raw-corrected)/raw if raw > 0 else None
    if reduction is None or reduction < .30:
        failures.append('lateral_reduction_under_30_percent')
    return {'passed': not failures, 'failures': failures,
            'manual_lateral_error_m': raw, 'geometric_lateral_error_m': corrected,
            'reduction_fraction': reduction,
            'claim': 'synthetic known-geometry alignment only; no human, perception or contact-task result'}
