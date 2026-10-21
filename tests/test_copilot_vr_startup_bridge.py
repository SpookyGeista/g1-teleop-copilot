"""Exercise the real VR bridge and core together, including the first action."""
from dataclasses import replace

import pytest

from teleop_assist.copilot.mailbox import read_json
from test_copilot_sim_control import Fixture
from test_copilot_vr_buttons import CALIBRATE, Source, bridge, status
from test_copilot_vr_reconnect import Paired


def test_first_real_selection_is_not_swallowed_by_core_startup_baseline(tmp_path):
    source = Source(tmp_path/'operator.json')
    app = bridge(source, tmp_path)
    f = Fixture()
    f.session.clock = lambda: source.now
    previous = {}

    def tick(buttons=()):
        nonlocal previous
        operator = source.sample(buttons)
        current = status(source, **previous)
        action = app.consume_vr(operator, runtime_status=current, now=source.now)
        f.now = source.now
        state, scene = f.state(), f.scene()
        state['clock_id'] = scene['clock_id'] = source.clock_id
        proposals = f.proposal()
        proposals = replace(proposals, clock_id=source.clock_id,
                            candidates=tuple(replace(c, candidate_id='a') for c in proposals.candidates))
        ui = read_json(app.sim_control.path)
        result = f.session.step(operator, ui, state, scene, proposals,
            measured_left_base_ee=f.left, measured_right_base_ee=f.right, now=source.now,
            clock_id=source.clock_id, calibration_id='cal')
        previous = result['snapshot']
        return action, ui, result

    action, ui, result = tick()
    assert action['reason'] == 'vr_neutral_core_baseline'
    assert app.sim_ownership.client_id is None
    assert not any(ui[k] for k in ('clutch','confirm','release','assist_held','select_candidate_id'))
    assert not result['allowed'] and not result['snapshot']['clutched']
    baseline_revision = result['snapshot']['last_ui_revision']
    action, ui, result = tick(['left.bButton'])
    assert ui['revision'] > baseline_revision and ui['select_candidate_id'] == 'a'
    assert result['snapshot']['selected_candidate_id'] == 'a'
    assert not result['allowed'] and not result['snapshot']['confirmed']
    tick()
    _, _, result = tick(CALIBRATE)
    assert result['snapshot']['clutched'] and not result['allowed']
    assert not f.worker.calls  # Selection and clutch do not issue a motion command.


@pytest.mark.parametrize('constraint', ['held_at_boot', 'desktop_owner', 'clutched'])
def test_neutral_priming_cannot_override_existing_control_or_treat_pressed_button_as_neutral(tmp_path, constraint):
    source = Source(tmp_path/'operator.json')
    app = bridge(source, tmp_path)
    buttons = CALIBRATE if constraint == 'held_at_boot' else []
    snapshot = {}
    if constraint == 'desktop_owner':
        app.sim_ownership.require('desktop', acquire=True)
    if constraint == 'clutched':
        snapshot['clutched'] = True
    before = read_json(app.sim_control.path)
    operator = source.sample(buttons)
    result = app.consume_vr(operator, runtime_status=status(source, **snapshot), now=source.now)
    assert not result['written']
    assert read_json(app.sim_control.path) == before


def test_old_ui_ack_alone_does_not_prove_current_source_epoch_is_primed(tmp_path):
    # Regression_003's former "already_primed" fixture supplied only revision
    # 10. That can belong to a stale browser epoch: the core acknowledges UI
    # before checking a changed source. A fresh neutral baseline must cross
    # both gates without taking ownership or issuing a control action.
    source = Source(tmp_path/'operator.json')
    app = bridge(source, tmp_path)
    app.sim_control.revision=9
    before = app.sim_control.write('idle',client_id='prior_source_owner')
    assert before['revision']==10
    operator = source.sample()
    result = app.consume_vr(operator, runtime_status=status(source,last_ui_revision=10),now=source.now)
    assert result['written'] and result['reason']=='vr_neutral_core_baseline'
    ui = read_json(app.sim_control.path)
    assert ui['revision']>before['revision']
    assert not any(ui[k] for k in ('clutch','confirm','release','assist_held','select_candidate_id'))
    assert app.sim_ownership.client_id is None
    assert ui['capture_timestamp']==operator['vr_buttons']['acquisition_timestamp']
    assert ui['expires_at']==operator['vr_buttons']['expires_at']
    assert ui['source_event_sequence']==operator['vr_buttons']['event_sequence']


def test_real_current_epoch_ack_prevents_repriming_or_overwriting_baseline(tmp_path):
    paired = Paired(tmp_path)
    paired.start()
    paired.synchronize()
    before = read_json(paired.dashboard.sim_control.path)
    assert paired.core._source_epoch==paired.dashboard.vr_buttons.epoch
    assert paired.runtime['snapshot']['last_ui_revision']==before['revision']
    assert 'operator_source_sequence' in paired.runtime
    assert paired.dashboard._vr_reconnect_epoch is None
    for _ in range(3):
        result,packet = paired.tick()
        assert not result['written']
        assert not paired.consume(packet)['written']  # Identical event cannot replay.
        assert read_json(paired.dashboard.sim_control.path)==before
        assert paired.dashboard.sim_ownership.client_id is None
        assert not paired.core.clutched and not paired.runtime['allowed']
    assert not paired.worker.calls


def test_pressed_boot_requires_real_release_before_neutral_baseline_can_prime(tmp_path):
    source = Source(tmp_path/'operator.json')
    app = bridge(source, tmp_path)
    for buttons in (CALIBRATE, CALIBRATE):
        packet = source.sample(buttons)
        assert not app.consume_vr(packet, runtime_status=status(source), now=source.now)['written']
    packet = source.sample()
    result = app.consume_vr(packet, runtime_status=status(source), now=source.now)
    assert result['reason'] == 'vr_neutral_core_baseline'
    ui = read_json(app.sim_control.path)
    assert ui['source_event_sequence'] == packet['vr_buttons']['event_sequence']
    assert ui['capture_timestamp'] == packet['vr_buttons']['acquisition_timestamp']
    assert ui['expires_at'] == packet['vr_buttons']['expires_at']
    assert not ui['clutch'] and not ui['assist_held']
