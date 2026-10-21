"""Non-motion reconnect handshake with no authorized live proposal batch."""
from copy import deepcopy
from dataclasses import replace

import pytest

from test_copilot_vr_reconnect import Paired, pending_pair
from test_copilot_sim_control import Fixture
from teleop_assist.copilot.dashboard import _neutral_ack_matches
from teleop_assist.copilot.mailbox import read_json


class LiveMissingBatch(Paired):
    def core_step(self, packet):
        f = self.fixture
        f.now = self.source.now
        state, scene = f.state(), f.scene()
        state['clock_id'] = scene['clock_id'] = self.source.clock_id
        batch = replace(f.proposal(), target_id='', candidates=(), valid=False,
                        clock_id=self.source.clock_id)
        self.runtime = self.core.step(packet, read_json(self.dashboard.sim_control.path),
            state, scene, batch, measured_left_base_ee=f.left, measured_right_base_ee=f.right,
            now=self.source.now, clock_id=self.source.clock_id, calibration_id='cal')
        self.runtime['candidates'] = [{'candidate_id': 'a'}, {'candidate_id': 'b'}]
        return self.runtime


def reconnect(tmp_path):
    p = Paired(tmp_path)
    p.start()  # Valid fixed batch only for the pre-reconnect historical session.
    p.__class__ = LiveMissingBatch
    p.source.now += .16
    return p


def await_ack(p):
    for _ in range(10):
        result, packet = p.tick()
        if p.runtime.get('neutral_source_ui_ack'):
            return packet, deepcopy(p.runtime)
    raise AssertionError(p.runtime)


def test_missing_live_batch_no_longer_circularly_blocks_explicit_new_select(tmp_path):
    p = reconnect(tmp_path)
    packet, status = await_ack(p)
    ack = status['neutral_source_ui_ack']
    baseline = deepcopy(p.dashboard._vr_reconnect_baseline)
    assert status['reason'] == 'proposal_target_invalid'
    assert 'operator_source_sequence' not in status
    assert ack['motion_authorized'] is False
    assert ack['source_event_clock_id'] == 'browser:new'
    assert ack['ui_revision'] == status['snapshot']['last_ui_revision'] == baseline['revision']
    assert not status['allowed'] and not status['command_pending'] and not p.worker.calls
    assert p.dashboard.sim_ownership.client_id is None
    result, selected = p.tick(['left.bButton'], core=False)
    assert result['written'] and p.dashboard._vr_reconnect_epoch is None
    ui = read_json(p.dashboard.sim_control.path)
    assert ui['select_candidate_id'] == 'a' and not ui['clutch'] and not ui['confirm']
    assert ui['capture_timestamp'] == selected['vr_buttons']['acquisition_timestamp']
    assert ui['expires_at'] == selected['vr_buttons']['expires_at']
    assert ui['revision'] > baseline['revision']
    assert not p.core.clutched and not p.worker.calls


def test_deferred_edge_can_follow_neutral_ack_with_newer_pressed_source_without_retiming(tmp_path):
    p, _ = pending_pair(tmp_path)
    p.__class__ = LiveMissingBatch
    result, pressed = p.tick(['left.bButton'], core=False)
    assert not result['written']
    original = deepcopy(pressed['vr_buttons'])
    for _ in range(2):
        p.core_step(pressed)
        if p.runtime.get('neutral_source_ui_ack'):
            break
        p.source.now += .01
    ack = p.runtime['neutral_source_ui_ack']
    assert ack['ui_source_event_sequence'] < ack['source_sequence']
    assert ack['ui_capture_timestamp'] < ack['operator_acquisition_timestamp']
    result, _ = p.tick(['left.bButton'], core=False)
    assert result['written']
    ui = read_json(p.dashboard.sim_control.path)
    for target, source in [('capture_timestamp','acquisition_timestamp'),
                           ('expires_at','expires_at'), ('source_event_sequence','event_sequence')]:
        assert ui[target] == original[source]
    assert ui['select_candidate_id'] == 'a' and not p.worker.calls


@pytest.mark.parametrize('mutation', [
    ('ui_revision', -1), ('ui_client_id', 'other'), ('ui_signature', 'changed'),
    ('ui_capture_timestamp', 0.), ('ui_source_event_sequence', -1), ('ui_expires_at', 1e30),
    ('source_instance_id', 'other'), ('source_event_clock_id', 'browser:old'),
    ('source_sequence', -1), ('source_sequence', True), ('operator_acquisition_timestamp', 0.),
    ('clock_id', 'wrong'), ('calibration_id', ''), ('motion_authorized', True),
    ('checked_at', float('nan')), ('checked_at', 1e30), ('valid_until', 1e30),
    ('schema', 'other')])
def test_ack_cannot_close_a_different_or_malformed_baseline(tmp_path, mutation):
    p = reconnect(tmp_path)
    _, status = await_ack(p)
    baseline = p.dashboard._vr_reconnect_baseline
    epoch = p.dashboard._vr_reconnect_epoch
    assert _neutral_ack_matches(status, baseline, epoch, p.source.now)
    status['neutral_source_ui_ack'][mutation[0]] = mutation[1]
    assert not _neutral_ack_matches(status, baseline, epoch, p.source.now)


def test_expired_ack_and_original_deferred_edge_never_renew_from_held_samples(tmp_path):
    p, _ = pending_pair(tmp_path)
    p.__class__ = LiveMissingBatch
    _, pressed = p.tick(['left.bButton'], core=False)
    for _ in range(2):
        p.core_step(pressed)
        if p.runtime.get('neutral_source_ui_ack'):
            break
        p.source.now += .01
    acked = deepcopy(p.runtime)
    p.source.now = pressed['expires_at'] + .001
    before = read_json(p.dashboard.sim_control.path)
    result, _ = p.tick(['left.bButton'], core=False)
    assert not result['written']
    assert read_json(p.dashboard.sim_control.path) == before
    assert not p.dashboard.vr_buttons.armed
    assert p.dashboard.sim_ownership.client_id is None
    assert not _neutral_ack_matches(acked, p.dashboard._vr_reconnect_baseline,
        p.dashboard._vr_reconnect_epoch, p.source.now)


def core_ack_fixture():
    f = Fixture()
    f.tick(release=True)
    batch = replace(f.proposal(), target_id='', candidates=(), valid=False)
    return f, batch


@pytest.mark.parametrize('invalid', ['operator_stale','operator_epoch','ui_stale','ui_owner',
    'state_stale','scene_stale','scene_incomplete','measured_EE'])
def test_invalid_inputs_cannot_mint_ack_before_proposal_gate(invalid):
    f, batch = core_ack_fixture()
    f.now += .02
    packet = f.packet()
    state, scene = f.state(), f.scene()
    ui = dict(revision=f.revision+1,capture_timestamp=f.now,clock_id='mono',client_id='test_ui')
    if invalid == 'operator_stale': packet['acquisition_timestamp'] -= 1.
    elif invalid == 'operator_epoch': packet['source_event_clock_id'] = 'new'
    elif invalid == 'ui_stale': ui['capture_timestamp'] -= 1.
    elif invalid == 'ui_owner': ui['client_id'] = 'other'
    elif invalid == 'state_stale': state['acquisition_timestamp'] -= 1.
    elif invalid == 'scene_stale': scene['acquisition_timestamp'] -= 1.
    elif invalid == 'scene_incomplete': scene['geometry_complete'] = False
    elif invalid == 'measured_EE': state['observed_body_world_poses']['left_wrist_yaw_link'][0][3] += .01
    out = f.tick(packet=packet,ui=ui,state=state,scene=scene,batch=batch)
    assert not out['allowed'] and 'neutral_source_ui_ack' not in out


@pytest.mark.parametrize('actions', [dict(clutch=True),dict(confirm=True),dict(assist_held=True),
    dict(release=True),dict(select_candidate_id='grasp')])
def test_one_shot_or_held_intent_never_mints_neutral_ack(actions):
    f, batch = core_ack_fixture()
    out = f.tick(batch=batch, **actions)
    assert 'neutral_source_ui_ack' not in out and not f.worker.calls


@pytest.mark.parametrize('limit', ['ui','operator','state','scene'])
@pytest.mark.parametrize('delta', [0., .001])
def test_actual_end_clock_rejects_equality_or_overrun_of_any_original_deadline(limit, delta):
    f, batch = core_ack_fixture()
    f.now += .16 if limit == 'operator' else .02
    f.clock = f.now
    f.sequence += 1
    packet=f.packet(); state=f.state(); scene=f.scene()
    ui=dict(revision=f.revision+1,capture_timestamp=f.now,expires_at=f.now+.1,clock_id='mono',client_id='test_ui')
    if limit == 'ui': ui['expires_at']=f.now+.01
    elif limit == 'operator':
        stamp=f.now-.14
        packet['acquisition_timestamp']=stamp
        packet['envelope']['acquisition_timestamp']=stamp
        packet['envelope']['received_at_host_monotonic']=stamp
        packet['expires_at']=stamp+.15
    elif limit == 'state': state['acquisition_timestamp']=f.now-f.session.config.state_max_age_s+.01
    elif limit == 'scene': scene['acquisition_timestamp']=f.now-f.session.config.scene_max_age_s+.01
    deadline=min(ui['expires_at'],packet['expires_at'],
        state['acquisition_timestamp']+f.session.config.state_max_age_s,
        scene['acquisition_timestamp']+f.session.config.scene_max_age_s)
    values=iter([f.now, deadline+delta])
    f.session.clock=lambda:next(values)
    out=f.session.step(packet,ui,state,scene,batch,measured_left_base_ee=f.left,
        measured_right_base_ee=f.right,now=f.now,clock_id='mono',calibration_id='cal')
    assert out['reason']=='proposal_target_invalid'
    assert 'neutral_source_ui_ack' not in out and not f.worker.calls


def test_ack_result_private_copy_and_hold_reason_are_unchanged():
    f, batch = core_ack_fixture()
    out=f.tick(batch=batch)
    assert out['reason']=='proposal_target_invalid' and not out['allowed']
    assert out['neutral_source_ui_ack']['checked_at']==f.clock
    out['neutral_source_ui_ack']['source_sequence']=-999
    assert f.session.status['neutral_source_ui_ack']['source_sequence']==f.sequence
    f.session.hold('explicit_reset',now=f.now,clock_id='mono')
    assert 'neutral_source_ui_ack' not in f.session.status


@pytest.mark.parametrize('choice', ['selected','confirmed','preview_pending','clutched'])
def test_existing_choice_or_clutch_never_gets_neutral_baseline_ack(choice):
    f=Fixture();f.tick(release=True)
    if choice=='clutched':f.tick(clutch=True)
    else:
        f.tick(select_candidate_id='grasp')
        if choice=='confirmed':f.tick(confirm=True)
        if choice=='preview_pending':f.session.controller._preview_pending=True
    out=f.tick(batch=replace(f.proposal(),target_id='',valid=False,candidates=()))
    assert 'neutral_source_ui_ack' not in out


def test_valid_selected_flow_is_not_replaced_by_neutral_ack():
    f=Fixture();f.tick(release=True)
    selected=f.tick(select_candidate_id='grasp')
    out=f.tick()
    assert out['snapshot']['selected_candidate_id']==selected['snapshot']['selected_candidate_id']
    assert 'neutral_source_ui_ack' not in out


@pytest.mark.parametrize('finished', [None, float('nan'), float('inf'), True, 0.])
def test_invalid_or_regressed_end_clock_leaves_plain_hold(finished):
    f, batch = core_ack_fixture()
    values=iter([f.clock+.02,finished])
    f.session.clock=lambda:next(values)
    out=f.tick(batch=batch)
    assert out['reason']=='proposal_target_invalid'
    assert 'neutral_source_ui_ack' not in out


def test_end_clock_failure_does_not_replace_hold_with_exception():
    f, batch = core_ack_fixture()
    calls=[]
    def clock():
        calls.append(True)
        if len(calls)==1:return f.clock
        raise OSError('Synthetic clock failure')
    f.session.clock=clock
    out=f.tick(batch=batch)
    assert out['reason']=='proposal_target_invalid' and 'neutral_source_ui_ack' not in out


@pytest.mark.parametrize('snapshot', [None, [], True])
def test_malformed_status_cannot_ack(tmp_path, snapshot):
    p=reconnect(tmp_path)
    _, status=await_ack(p)
    status['snapshot']=snapshot
    assert not _neutral_ack_matches(status,p.dashboard._vr_reconnect_baseline,
        p.dashboard._vr_reconnect_epoch,p.source.now)


def test_desktop_ownership_is_not_acquired_or_overwritten_by_neutral_handshake(tmp_path):
    p=reconnect(tmp_path)
    p.dashboard.sim_ownership.require('desktop',acquire=True)
    before=read_json(p.dashboard.sim_control.path)
    for _ in range(5):p.tick(core=False)
    assert p.dashboard.sim_ownership.client_id=='desktop'
    assert read_json(p.dashboard.sim_control.path)==before
