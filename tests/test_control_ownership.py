from copy import deepcopy

import pytest

from teleop_assist.core import AssistConfig, AssistSession
from teleop_assist.control_ownership import BrowserControlLease, ControlConflict


@pytest.fixture
def lease():
    now = [10.]
    session = AssistSession(AssistConfig(require_action=True), clock=lambda: now[0])
    session.step({'valid': True, 'ee_position': [.4, -.25, 1.], 'ee_quaternion': [0, 0, 0, 1]}, .005)
    return BrowserControlLease(session), now


def claim(lease, client):
    return lease.dispatch({'type': 'control_claim', 'client_id': client, 'deadman_released': True})


def sample(client, seq, clutch=False):
    return {'type': 'input', 'client_id': client, 'session_id': client, 'seq': seq,
            'source': 'webxr_controller', 'right': {'valid': True, 'position': [0, 0, 0]},
            'clutch': clutch, 'squeeze': float(clutch), 'trigger': 0.}


def test_observer_cannot_mutate_calibration_selection_settings_or_input(lease):
    control, _ = lease
    claim(control, 'headset')
    control.dispatch(sample('headset', 1))
    control.dispatch({'type': 'calibrate', 'client_id': 'headset'})
    initial = deepcopy(control.session.input)
    origin = control.session.hand_origin.copy()
    for command in [sample('pc', 1), {'type': 'assist_enable', 'enabled': True},
                    {'type': 'calibrate'}, {'type': 'release'}, {'type': 'control_release'}]:
        with pytest.raises(ControlConflict):
            control.dispatch(dict(command, client_id='pc'))
    assert control.session.input == initial
    assert (control.session.hand_origin == origin).all()
    assert not control.session.config.assist_enabled
    assert control.owner == 'headset'


def test_explicit_takeover_rejects_fresh_held_deadman_then_requires_new_release(lease):
    control, _ = lease
    claim(control, 'headset')
    control.dispatch(sample('headset', 1))
    control.dispatch(sample('headset', 2, True))
    assert not control.session.rearm_required
    with pytest.raises(ControlConflict, match='holding'):
        claim(control, 'pc')
    control.dispatch(sample('headset', 3))
    claim(control, 'pc')
    assert control.session.input is None and control.session.hand_origin is None
    assert control.session.rearm_required
    control.dispatch(sample('pc', 1, True))
    assert control.session.rearm_required  # no inherited released edge
    control.dispatch(sample('pc', 2))
    control.dispatch(sample('pc', 3, True))
    assert not control.session.rearm_required
    assert control.session.input_session == 'pc'


def test_timeout_clears_old_held_input_and_does_not_auto_claim(lease):
    control, now = lease
    claim(control, 'headset')
    control.dispatch(sample('headset', 1, True))
    now[0] += 1.01
    assert control.snapshot()['owner_client_id'] is None
    assert control.session.input is None
    with pytest.raises(ControlConflict):
        control.dispatch(sample('headset', 2, True))
    claim(control, 'pc')
    assert control.owner == 'pc'


def test_stale_deadman_cannot_block_explicit_takeover_until_lease_timeout(lease):
    control, now = lease
    claim(control, 'headset')
    control.dispatch(sample('headset', 1, True))
    now[0] += .31  # production input watchdog already requires a hold
    claim(control, 'pc')
    assert control.owner == 'pc'
    assert control.session.input is None and control.session.rearm_required


def test_observer_stop_still_available_and_owner_release_cancels_input(lease):
    control, _ = lease
    claim(control, 'headset')
    control.dispatch(sample('headset', 1, True))
    control.dispatch({'type': 'stop', 'client_id': 'observer'})
    assert control.session.stopped
    assert control.owner == 'headset'
    control.dispatch({'type': 'control_release', 'client_id': 'headset'})
    assert control.owner is None and control.session.input is None


def test_unclaimed_or_anonymous_http_commands_require_claim(lease):
    control, _ = lease
    with pytest.raises(ControlConflict):
        control.dispatch(sample('browser', 1))
    with pytest.raises(ControlConflict):
        control.dispatch({'type': 'calibrate'})
    with pytest.raises(ControlConflict):
        control.dispatch({'type': 'control_claim', 'client_id': 'browser'})


def test_duplicate_input_cannot_renew_lease(lease):
    control, now = lease
    claim(control, 'headset')
    control.dispatch(sample('headset', 1))
    now[0] += .8
    assert control.dispatch(sample('headset', 1)) == {'accepted': False}
    now[0] += .21
    assert control.snapshot()['owner_client_id'] is None


def test_internal_driver_api_remains_independent(lease):
    control, _ = lease
    assert control.session.command(sample('internal_driver', 1))['accepted']
    assert control.owner is None


def test_external_eye_sensor_does_not_take_arm_ownership(lease):
    control, _ = lease
    eye = {'type': 'gaze', 'source': 'hardware_adapter', 'session_id': 'sensor', 'seq': 1,
           'valid': True, 'confidence': .9, 'calibrated': True, 'uv': [.4, .5]}
    assert control.dispatch(eye)['accepted']
    assert control.owner is None
    with pytest.raises(ControlConflict):
        control.dispatch({'type': 'gaze', 'sample': dict(eye, source='webxr_gaze_projection')})


def test_owner_heartbeat_keeps_calibration_session_without_forging_fresh_motion(lease):
    control, now = lease
    claim(control, 'headset')
    control.dispatch(sample('headset', 1, True))
    original_time = control.session.input_time
    now[0] += .8
    assert control.dispatch({'type': 'control_heartbeat', 'client_id': 'headset'})['accepted']
    assert control.session.input_time == original_time
    now[0] += .3
    assert control.snapshot()['owner_client_id'] == 'headset'
    with pytest.raises(ControlConflict):
        control.dispatch({'type': 'control_heartbeat', 'client_id': 'pc'})
    claim(control, 'pc')  # stale motion input cannot block an explicit handover
    assert control.session.input is None


def test_replay_refuses_claim_release_heartbeat_and_stop_without_mutation():
    from tools.replay_assist import ReplaySession
    session = ReplaySession()
    control = BrowserControlLease(session)
    before = session.snapshot()
    for kind in ('control_claim', 'control_release', 'control_heartbeat', 'stop'):
        assert not control.dispatch({'type': kind, 'client_id': 'browser', 'deadman_released': True})['accepted']
    assert control.owner is None and session.input is None
    assert session.stopped == before['stopped']
    assert control.snapshot()['read_only']
