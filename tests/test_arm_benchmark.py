"""Reject fake improvements, failed physical trials and nondeterministic inputs."""
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.arm_benchmark import (SyntheticArmOperator, compare_trials,
    scenario_duration, scenario_point, trial_metrics)


def samples(scenario='misalignment', assistance='manual', stationary=False):
    origin = np.array([.37, -.15, .94])
    rows = []
    for index, elapsed in enumerate(np.arange(0., scenario_duration(scenario), .01)):
        point = scenario_point(scenario, float(elapsed))
        raw = origin+point.raw_offset
        intended = origin+point.intended_offset
        position = origin if stationary else (raw if assistance == 'manual' else intended)
        quaternion = Rotation.from_rotvec(point.rotation_vector).as_quat().tolist()
        rows.append({'scenario_time': float(elapsed), 'phase': point.phase,
                     'scenario_origin': origin.tolist(), 'initial_quaternion': [0, 0, 0, 1],
                     'intended_position': intended.tolist(), 'raw_position': raw.tolist(),
                     'command_position': position.tolist(), 'ee_position': position.tolist(),
                     'ee_quaternion': [0, 0, 0, 1] if stationary else quaternion,
                     'intended_quaternion': quaternion, 'arm_velocities': [0.]*7,
                     'arm_joints': [0.]*7, 'contact_force_max': 0., 'valid': True,
                     'session_reason': 'predictive_alignment' if assistance == 'geometric' else 'manual_control',
                     'assist_active': assistance == 'geometric',
                     'arm_solver': {'status': 'solved', 'request_id': index//4, 'solve_ms': 2.}})
    return rows


def test_actual_intended_error_is_separate_from_command_tracking():
    manual = trial_metrics(samples(), 'misalignment', 'manual')
    assisted = trial_metrics(samples(assistance='geometric'), 'misalignment', 'geometric')
    assert manual['passed'] and assisted['passed']
    assert manual['command_tracking_rmse_m'] < 1e-10
    assert manual['intended_position_rmse_m'] > .010
    assert manual['outbound_hold_lateral_error_m'] == pytest.approx(.025)
    assert assisted['outbound_hold_lateral_error_m'] == pytest.approx(0.)
    assert compare_trials(manual, assisted)['passed']
    # Solver time is counted once per solve id, not once per physics sample.
    assert assisted['unique_solver_calls'] == 155


def test_hold_cannot_overwrite_cost_of_a_real_solver_call():
    rows = samples()
    rows[1]['arm_solver'] = {'status': 'hold', 'request_id': 0, 'solve_ms': 0.}
    rows[2]['arm_solver'] = {'status': 'held', 'request_id': 0, 'solve_ms': 0.}
    metrics = trial_metrics(rows, 'misalignment', 'manual')
    assert metrics['unique_solver_calls'] == 155
    assert metrics['solver_time_mean_ms'] == pytest.approx(2.)
    assert metrics['solver_time_p95_ms'] == pytest.approx(2.)


def test_motionless_robot_cannot_win_alignment_comparison():
    manual = trial_metrics(samples(), 'misalignment', 'manual')
    motionless = trial_metrics(samples(assistance='geometric', stationary=True), 'misalignment', 'geometric')
    assert motionless['outbound_hold_lateral_error_m'] == 0.
    assert 'insufficient_measured_forward_progress' in motionless['failures']
    comparison = compare_trials(manual, motionless)
    assert comparison['reduction_fraction'] == 1.
    assert not comparison['passed']
    assert 'physical_trial_failed' in comparison['failures']


def test_short_invalid_contact_trial_is_retained_as_failure():
    rows = samples()[:100]
    rows[25]['valid'] = False
    rows[42]['contact_force_max'] = 19.
    rows[77]['arm_solver']['status'] = 'failed'
    metrics = trial_metrics(rows, 'misalignment', 'manual')
    assert not metrics['passed']
    assert {'scenario_incomplete', 'invalid_robot_state', 'contact_guard_triggered',
            'solver_failure'} <= set(metrics['failures'])
    assert metrics['contact_force_max_N'] == 19.


def test_orientation_failure_is_not_hidden_by_perfect_translation():
    rows = samples('pose_loop')
    for row in rows:
        row['ee_quaternion'] = [0., 0., 0., 1.]
    metrics = trial_metrics(rows, 'pose_loop', 'manual')
    assert metrics['intended_position_rmse_m'] < 1e-10
    assert not metrics['passed']
    assert 'insufficient_measured_rotation' in metrics['failures']


@pytest.mark.parametrize('corrupt', ['nan_position', 'time_reversal', 'zero_quaternion'])
def test_invalid_telemetry_never_becomes_zero_error(corrupt):
    rows = samples()
    if corrupt == 'nan_position':
        rows[10]['ee_position'][0] = float('nan')
    elif corrupt == 'time_reversal':
        rows[10]['scenario_time'] = rows[9]['scenario_time']
    else:
        rows[10]['ee_quaternion'] = [0, 0, 0, 0]
    assert trial_metrics(rows, 'misalignment', 'manual')['failures'] == ['invalid_measured_telemetry']


def test_path_returns_to_origin_and_has_explicit_hold_reverse_rotation():
    for name in ('pose_loop', 'misalignment'):
        np.testing.assert_array_equal(scenario_point(name, 0).raw_offset, np.zeros(3))
        end = scenario_point(name, scenario_duration(name))
        np.testing.assert_array_equal(end.raw_offset, np.zeros(3))
        np.testing.assert_array_equal(end.rotation_vector, np.zeros(3))
        assert end.phase == 'end_hold'
    assert scenario_point('pose_loop', 4.).phase == 'rotation'
    assert scenario_point('pose_loop', 6.).phase == 'reverse'
    assert scenario_point('misalignment', 3.).phase == 'translation_hold'


def test_production_action_latch_and_stale_input_gate_use_same_synthetic_times():
    state = {'valid': True, 'ee_position': [.37, -.15, .94],
             'ee_quaternion': [0., 0., 0., 1.]}
    first = SyntheticArmOperator('geometric')
    second = SyntheticArmOperator('geometric')
    for operator in (first, second):
        operator.begin('misalignment', copy.deepcopy(state), 0.)
        assert operator.session.action.confirmed
        assert not operator.session.rearm_required
        for i in range(1, 61):
            reference, expected = operator.reference(i*.02, i*.02, state, .02, send_packet=True)
        assert expected['session_reason'] == 'predictive_alignment'
        assert reference['correction'][1] < 0.
    np.testing.assert_allclose(first.session.latest_reference['ee_target_world'],
                               second.session.latest_reference['ee_target_world'])
    first.now += .31
    reference = first.session.step(state, .005)
    assert reference['hold']
    assert first.session.control['reason'] == 'input_stale'


def test_comparison_does_not_claim_gain_when_baseline_error_is_too_small():
    report = compare_trials({'passed': True, 'outbound_hold_lateral_error_m': .001},
                            {'passed': True, 'outbound_hold_lateral_error_m': 0.})
    assert not report['passed']
    assert 'manual_error_too_small_to_evaluate' in report['failures']
