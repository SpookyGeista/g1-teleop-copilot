from dataclasses import replace
import numpy as np
from teleop_assist.copilot.observations import GeometryObservation
from teleop_assist.copilot.stationarity import ObservedStationarity


def obs(index=0, t=10., displacement=0., **changes):
    T = np.eye(4); T[0, 3] = displacement
    base = GeometryObservation('object', f'frame:{index}', t, 'clock', 'cal', 'measured',
        np.ones((24, 24)), np.ones((24, 24), bool),
        np.array([[300., 0, 12], [0, 300, 12], [0, 0, 1]]), T, np.eye(4))
    return replace(base, **changes)


def test_independent_frames_required_and_source_time_preserved():
    gate = ObservedStationarity(obs())
    for _ in range(8):
        assert not gate.observe(obs(), now=10.1, clock_id='clock')['motion_static']
    assert not gate.observe(obs(1, 10.05), now=10.06, clock_id='clock')['motion_static']
    report = gate.observe(obs(2, 10.11), now=10.12, clock_id='clock')
    assert report['motion_static']
    assert report['model_source_capture_timestamp'] == 10.
    assert report['validation_capture_timestamp'] == 10.11


def test_object_moves_then_returns_does_not_revalidate_old_selection():
    gate = ObservedStationarity(obs())
    changed = gate.observe(obs(1, 10.1, displacement=.03), now=10.1, clock_id='clock')
    assert changed['reason'] == 'observed_target_surface_changed'
    assert not gate.observe(obs(2, 10.2), now=10.2, clock_id='clock')['motion_static']


def test_first_surface_failure_keeps_measured_error_and_original_source_binding():
    gate = ObservedStationarity(obs())
    changed = gate.observe(obs(1,10.1,displacement=.03),now=10.1,clock_id='clock')
    assert changed['centroid_change_m'] > changed['centroid_limit_m']
    assert changed['surface_p95_m'] > changed['surface_p95_limit_m']
    assert changed['model_source_capture_timestamp'] == 10.
    assert changed['validation_observation_id'] == 'frame:1'
    rejected = gate.observe(obs(2,10.2),now=10.2,clock_id='clock')
    assert rejected == changed  # Later return must not replace the failure evidence.


def test_loss_staleness_clock_calibration_sequence_and_base():
    for current, now, clock in [
        (obs(1, 10.1), 11., 'clock'), (obs(1, 10.1), 10.1, 'other'),
        (obs(1, 10.1, calibration_id='new'), 10.1, 'clock'),
        (obs(1, 9.), 10., 'clock'), (obs(1, 10.4), 10.4, 'clock'),
        (obs(1, 10.), 10., 'clock')]:
        assert not ObservedStationarity(obs()).observe(current, now=now, clock_id=clock)['motion_static']
    T = np.eye(4); T[0, 3] = .002
    assert ObservedStationarity(obs()).observe(obs(1, 10.1, T_world_base=T), now=10.1, clock_id='clock')['reason'] == 'robot_base_moved'


def test_rotation_can_be_rejected_when_center_does_not_move():
    anchor = obs()
    depth = anchor.depth.copy(); depth[::2, :] += .04
    current = obs(1, 10.1, depth=depth)
    assert not ObservedStationarity(anchor).observe(current, now=10.1, clock_id='clock')['motion_static']


def test_nonconsecutive_frame_reuse_is_not_independent_evidence():
    gate = ObservedStationarity(obs())
    gate.observe(obs(1, 10.1), now=10.1, clock_id='clock')
    result = gate.observe(obs(0, 10.2), now=10.2, clock_id='clock')
    assert result['reason'] == 'observation_id_reused'


def test_explicit_recovery_needs_new_stable_window_and_keeps_model_anchor():
    gate=ObservedStationarity(obs())
    gate.invalidate('selected_object_not_observed')
    assert not gate.recover_current(obs(1,11.,displacement=.03),now=11.,clock_id='clock')
    assert not gate.recover_current(obs(1,10.1),now=11.,clock_id='clock')
    assert gate.recover_current(obs(1,11.),now=11.,clock_id='clock')
    assert not gate.observe(obs(1,11.),now=11.,clock_id='clock')['motion_static']
    assert not gate.observe(obs(2,11.06),now=11.06,clock_id='clock')['motion_static']
    report=gate.observe(obs(3,11.12),now=11.12,clock_id='clock')
    assert report['motion_static'] and report['model_source_capture_timestamp']==10.
    gate.invalidate('observed_target_surface_changed')
    assert not gate.recover_current(obs(4,11.2),now=11.2,clock_id='clock')
