"""Bounded mapped clock evidence, not headset/transport or robot validation."""
from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.copilot.mailbox import ControlWriter, read_json
from teleop_assist.copilot.sim_control import SimUIIntent
from teleop_assist.copilot.vr_buttons import (
    VR_TIME_FIELDS, controller_event_time_issue, source_payload)
from test_copilot_sim_control import Fixture
from test_copilot_vr_buttons import CALIBRATE, START, Source, bridge, status


def sample(**changes):
    result = dict(acquisition_timestamp=10.0028, expires_at=10.15, clock_id='mono',
        event_sequence=7, clock_uncertainty_s=.00955, received_at_host_monotonic=10.)
    result.update(changes)
    return result


def ui_packet(**changes):
    data = sample()
    result = dict(revision=1, capture_timestamp=data.pop('acquisition_timestamp'),
        source_kind='webxr_controller_event', source_event_sequence=data.pop('event_sequence'),
        client_id='xr:test', **data)
    result.update(changes)
    return result


class MappedSource(Source):
    """Actual telemetry/publisher, with explicitly synthetic mapped timestamps."""
    ahead = .0028
    bound = .00955
    browser_clock = 'browser:mapped-fixture'

    def sample(self, pressed=(), *, receipt_lag=0.):
        self.now += .02
        self.sequence += 2
        pose = np.eye(4).flatten(order='F').tolist()
        meta = dict(schema='punktnull.xr_frame.v1', sequence=self.sequence,
            sample_kind='simulation_pose_sample', sample_time_s=self.now-9.,
            sample_clock_id=self.browser_clock, host_sample_time_s=self.now+self.ahead,
            host_clock_id=self.clock_id, clock_uncertainty_s=self.bound, clock_sync_age_s=.1,
            sensor_capture_time_s=None, latency_unknown=False,
            left_tracking_valid=True, right_tracking_valid=True)
        self.shared.record('head', dict(camera={'matrix':pose},
            _copilot=dict(meta, sequence=self.sequence-1)), now=self.now-receipt_lag)
        states = {side:{name:side+'.'+name in pressed for name in ('aButton','bButton','squeeze')}
                  for side in ('left','right')}
        for state in states.values():
            state['_copilot_gamepad'] = dict(mapping='xr-standard', button_count=6,
                                             required_buttons_available=True)
        left = getattr(self, 'left_pose', np.eye(4))
        right = getattr(self, 'right_pose', np.eye(4))
        self.shared.record('controller', dict(left=left.flatten(order='F').tolist(),
            right=right.flatten(order='F').tolist(),leftState=states['left'],
            rightState=states['right'],_copilot=meta), now=self.now-receipt_lag)
        self.tele_data = SimpleNamespace(left_wrist_pose=left,right_wrist_pose=right,
                                         copilot_envelope=self.shared.envelope())
        return self.publisher.publish(self.tele_data)


def check(data, now=10.):
    return controller_event_time_issue(data['acquisition_timestamp'], data['expires_at'], now=now,
        clock_uncertainty_s=data['clock_uncertainty_s'],
        received_at_host_monotonic=data['received_at_host_monotonic'])


@pytest.mark.parametrize('ahead,bound', [(0.,0.),(.0028,.00955),(.01,.02),(.02,.02)])
def test_future_cap_is_explicit_and_never_bigger_than_ten_ms(ahead,bound):
    issue = check(sample(acquisition_timestamp=10.+ahead,clock_uncertainty_s=bound))
    assert (issue is None) == (ahead <= .01)


@pytest.mark.parametrize('changes', [
    {'acquisition_timestamp':math.nextafter(10.+.00955,math.inf)},
    {'acquisition_timestamp':math.nextafter(10.+.01,math.inf),'clock_uncertainty_s':.02},
    {'received_at_host_monotonic':math.nextafter(10.,math.inf)},
    {'expires_at':math.nextafter(10.15,math.inf)},
    {'acquisition_timestamp':9.99,'expires_at':math.nextafter(10.14,math.inf)},
    {'expires_at':10.0028}, {'expires_at':10.001},
    {'clock_uncertainty_s':0.}, {'clock_uncertainty_s':-.001},
    {'clock_uncertainty_s':math.nextafter(.02,math.inf)},
])
def test_clock_and_expiry_boundaries_reject(changes):
    assert check(sample(**changes)) is not None


@pytest.mark.parametrize('field', ['acquisition_timestamp','expires_at',*VR_TIME_FIELDS])
@pytest.mark.parametrize('value', [True,None,float('nan'),float('inf'),-float('inf'),10**500])
def test_nonfinite_boolean_or_unknown_values_never_get_tolerance(field,value):
    assert check(sample(**{field:value})) is not None


def test_expiry_is_strict_even_when_capture_time_is_mapped_forward():
    data=sample(expires_at=10.0029)
    assert check(data,now=data['expires_at']) is not None


def test_recorded_capture_publication_offset_with_synthetic_valid_arm_provenance():
    # Run004 records +2.785 ms capture vs publication, but only aggregate
    # head/arm receipt+bound. The own-arm receipt+bound below are an explicit
    # synthetic assumption matching that aggregate, NOT an exact run replay.
    data=sample(acquisition_timestamp=2229753.364943822,
        received_at_host_monotonic=2229753.360157521,
        clock_uncertainty_s=.009549999952313868, expires_at=2229753.510157521)
    assert check(data,now=2229753.362158569) is None
    assert data['acquisition_timestamp'] == 2229753.364943822


@pytest.mark.parametrize('changes', [
    {'clock_uncertainty_s':.021}, {'clock_uncertainty_s':None},
    {'clock_uncertainty_s':True}, {'received_at_host_monotonic':10.01},
    {'acquisition_timestamp':10.010001}, {'expires_at':10.150001},
    {'event_sequence':True},
])
def test_writer_rejects_bad_provenance_without_mutating_mailbox(tmp_path,changes):
    writer=ControlWriter(tmp_path,clock=lambda:10.,clock_id='mono')
    before=read_json(writer.path)
    with pytest.raises(ValueError):writer.write('hold',source_sample=sample(**changes))
    assert read_json(writer.path)==before


@pytest.mark.parametrize('absent', [VR_TIME_FIELDS[:1],VR_TIME_FIELDS[1:],VR_TIME_FIELDS])
def test_legacy_and_partial_writer_samples_never_gain_future_tolerance(tmp_path,absent):
    writer=ControlWriter(tmp_path,clock=lambda:10.,clock_id='mono')
    data=sample()
    for name in absent:data.pop(name)
    with pytest.raises(ValueError):writer.write('hold',source_sample=data)
    if len(absent)==2:
        data.update(acquisition_timestamp=10.)
        assert writer.write('hold',source_sample=data)['capture_timestamp']==10.


def test_writer_roundtrip_preserves_all_source_times_and_first_arrival(tmp_path):
    writer=ControlWriter(tmp_path,clock=lambda:10.,clock_id='mono')
    original=sample()
    packet=writer.write('hold',client_id='xr:test',source_sample=original)
    parsed=SimUIIntent.from_packet(read_json(writer.path))
    assert parsed.capture_timestamp==original['acquisition_timestamp']
    assert parsed.expires_at==original['expires_at']
    assert parsed.received_at_host_monotonic==original['received_at_host_monotonic']
    assert parsed.clock_uncertainty_s==original['clock_uncertainty_s']
    assert parsed.source_event_sequence==original['event_sequence']
    assert parsed.source_kind=='webxr_controller_event'
    assert packet==read_json(writer.path)
    validated,_,deadline=Fixture().session._ui(parsed,10.,'mono')
    assert deadline==original['expires_at'] and validated==parsed


@pytest.mark.parametrize('changes', [
    {'clock_uncertainty_s':None}, {'received_at_host_monotonic':None},
    {'clock_uncertainty_s':None,'received_at_host_monotonic':None},
    {'clock_uncertainty_s':True}, {'received_at_host_monotonic':float('nan')},
    {'source_kind':'desktop'}, {'source_event_sequence':True},
    {'capture_timestamp':10.010001}, {'received_at_host_monotonic':10.001},
    {'expires_at':10.150001},
])
def test_core_rejects_malformed_or_unbounded_source_contract(changes):
    with pytest.raises(ValueError):Fixture().session._ui(ui_packet(**changes),10.,'mono')


@pytest.mark.parametrize('name', VR_TIME_FIELDS)
def test_partial_typed_intent_rejects_without_losing_its_explicit_provenance(name):
    value=SimUIIntent.from_packet(ui_packet())
    value=replace(value,**{name:None})
    with pytest.raises(ValueError):Fixture().session._ui(value,10.,'mono')


def test_core_desktop_and_old_webxr_kind_remain_strict():
    for kind in (None,'webxr_controller_event'):
        packet=ui_packet(source_kind=kind)
        for name in VR_TIME_FIELDS:packet.pop(name)
        with pytest.raises(ValueError,match='ui_stale_or_future'):
            Fixture().session._ui(packet,10.,'mono')
    packet.update(capture_timestamp=10.)
    assert Fixture().session._ui(packet,10.,'mono')[0].capture_timestamp==10.


@pytest.mark.parametrize('change', ['bound','receipt','remove_pair','sequence'])
def test_same_revision_cannot_change_or_remove_time_provenance(change):
    f=Fixture();packet=ui_packet()
    f.session._ui(packet,10.,'mono')
    if change=='bound':packet['clock_uncertainty_s']=.009
    elif change=='receipt':packet['received_at_host_monotonic']-=.0001
    elif change=='sequence':packet['source_event_sequence']+=1
    else:
        for name in VR_TIME_FIELDS:packet.pop(name)
    with pytest.raises(ValueError,match='ui_revision_reused_with_changed_content'):
        f.session._ui(packet,10.01,'mono')


def test_rejected_future_click_cannot_become_an_edge_when_clock_catches_up():
    f=Fixture();f.session._ui(ui_packet(capture_timestamp=10.),10.,'mono')
    packet=ui_packet(revision=2,capture_timestamp=10.011,clock_uncertainty_s=.02,confirm=True)
    with pytest.raises(ValueError):f.session._ui(packet,10.,'mono')
    _,edge,_=f.session._ui(packet,10.012,'mono')
    assert not edge


def test_parser_rejects_mismatched_pose_bound_arrival_or_sequence(tmp_path):
    for changed in ('bound','arrival_before_envelope','arrival_after_publication','sequence','partial'):
        source=MappedSource(tmp_path/'operator.json');source.sample()
        packet=source.sample(CALIBRATE)
        buttons=packet['vr_buttons']
        if changed=='bound':buttons['clock_uncertainty_s']=.01
        elif changed=='arrival_before_envelope':buttons['received_at_host_monotonic']-=.001
        elif changed=='arrival_after_publication':buttons['received_at_host_monotonic']+=.001
        elif changed=='sequence':buttons['event_sequence']+=1
        else:buttons.pop('clock_uncertainty_s')
        assert source.parser.step(packet,now=source.now,clock_id=source.clock_id).action=='release'


def test_source_payload_refuses_stamp_after_first_arrival_plus_bound(tmp_path):
    source=MappedSource(tmp_path/'operator.json');packet=source.sample()
    arms=source.shared.read()['arms']
    arms['host_sample_time_s']=math.nextafter(arms['received_at_host_monotonic']+
                                             arms['clock_uncertainty_s'],math.inf)
    assert source_payload(arms,packet)['valid'] is False


def test_new_publication_never_retimes_old_arrival_or_resurrects_lease(tmp_path):
    source=MappedSource(tmp_path/'operator.json');first=source.sample()
    source.now+=.16
    duplicate=source.publisher.publish(source.tele_data)
    assert duplicate['published_at']>first['published_at']
    assert duplicate['vr_buttons']['received_at_host_monotonic']==first['vr_buttons']['received_at_host_monotonic']
    assert source.parser.step(duplicate,now=source.now,clock_id=source.clock_id).action=='release'


@pytest.mark.parametrize('move_while_stopped', [False, True])
def test_real_source_bridge_core_future_neutral_and_actions_have_exact_time_provenance(tmp_path, move_while_stopped):
    source=MappedSource(tmp_path/'operator.json');app=bridge(source,tmp_path);f=Fixture()
    f.session.clock=lambda:source.now
    original_fake_call=f.worker.call
    f.worker.call=lambda *args,**kwargs:dict(original_fake_call(*args,**kwargs),clock_id=source.clock_id)
    snapshot={}
    def tick(buttons=()):
        nonlocal snapshot
        operator=source.sample(buttons)
        result=app.consume_vr(operator,runtime_status=status(source,**snapshot),now=source.now)
        f.now=source.now
        state,scene=f.state(),f.scene()
        state['clock_id']=scene['clock_id']=source.clock_id
        proposal=replace(f.proposal(),clock_id=source.clock_id,
            candidates=tuple(replace(c,candidate_id='a') for c in f.proposal().candidates))
        ui=read_json(app.sim_control.path)
        outcome=f.session.step(operator,ui,state,scene,proposal,
            measured_left_base_ee=f.left,measured_right_base_ee=f.right,
            now=source.now,clock_id=source.clock_id,calibration_id='cal')
        snapshot=outcome['snapshot']
        assert ui['capture_timestamp']==operator['vr_buttons']['acquisition_timestamp']
        assert ui['expires_at']==operator['vr_buttons']['expires_at']
        for name in VR_TIME_FIELDS:assert ui[name]==operator['vr_buttons'][name]
        assert ui['source_event_sequence']==operator['vr_buttons']['event_sequence']
        return result,ui,outcome
    first,ui,outcome=tick()
    assert first['reason']=='vr_neutral_core_baseline'
    assert not ui['release'] and app.sim_ownership.client_id is None
    assert not outcome['snapshot']['clutched']
    _,ui,outcome=tick(['left.bButton'])
    assert outcome['snapshot']['selected_candidate_id']=='a'
    tick()
    _,ui,outcome=tick(CALIBRATE)
    assert outcome['snapshot']['clutched'] and not f.worker.calls
    # A neutral event after calibration remains stopped until the explicit AX edge.
    _,ui,outcome=tick()
    assert not ui['release'] and outcome['snapshot']['clutched'] and not outcome['allowed']
    assert outcome['snapshot']['teleop_start_required'] and not f.worker.calls
    if move_while_stopped:
        # Both hands can be raised and rotated after ABXY. AX must map these
        # new controller origins to the measured robot wrists, without a jump.
        source.left_pose = np.eye(4); source.right_pose = np.eye(4)
        source.left_pose[:3, 3] = [.2, .1, .3]
        source.right_pose[:3, 3] = [.25, -.1, .35]
        source.left_pose[:3, :3] = Rotation.from_euler('xyz', [30, -20, 15], degrees=True).as_matrix()
        source.right_pose[:3, :3] = Rotation.from_euler('xyz', [-10, 35, -25], degrees=True).as_matrix()
    _,ui,outcome=tick(START)
    assert ui['start'] and outcome['snapshot']['teleop_active'] and not outcome['allowed']
    _,ui,outcome=tick()
    assert outcome['allowed'] and len(f.worker.calls)==1
    np.testing.assert_allclose(f.worker.calls[-1]['left_base_ee'], f.left, atol=1e-14)
    np.testing.assert_allclose(f.worker.calls[-1]['right_base_ee'], f.right, atol=1e-14)
    _,ui,outcome=tick(START)
    assert ui['start'] and not outcome['snapshot']['teleop_active'] and not outcome['allowed']
    _,ui,outcome=tick()
    assert outcome['snapshot']['teleop_start_required'] and not outcome['allowed']
    assert len(f.worker.calls)==1
    if move_while_stopped:
        source.left_pose[:3, 3] += [.1, .2, -.1]
        source.right_pose[:3, 3] += [-.1, .1, .2]
        _,_,outcome=tick(START)
        assert not outcome['allowed'] and outcome['snapshot']['teleop_active']
        assert len(f.worker.calls)==1
        _,_,outcome=tick()
        assert outcome['allowed'] and len(f.worker.calls)==2
        np.testing.assert_allclose(f.worker.calls[-1]['left_base_ee'], f.left, atol=1e-14)
        np.testing.assert_allclose(f.worker.calls[-1]['right_base_ee'], f.right, atol=1e-14)
        reference=f.worker.calls[-1]['joint_reference']
        assert reference['acquisition_timestamp']==pytest.approx(source.now-.02)
        # Independent motion after start is retained on both sides.
        source.left_pose[0, 3] += .02
        source.right_pose[2, 3] += .02
        _,_,outcome=tick()
        assert outcome['allowed'] and len(f.worker.calls)==3
        assert not np.allclose(f.worker.calls[-1]['left_base_ee'], f.left)
        assert not np.allclose(f.worker.calls[-1]['right_base_ee'], f.right)
        f.worker.reject=True
        _,_,outcome=tick()
        assert not outcome['allowed'] and not outcome['snapshot']['clutched']
        return
    _,ui,outcome=tick(CALIBRATE)
    assert ui['release'] and not outcome['snapshot']['clutched'] and len(f.worker.calls)==1


def test_future_grip_release_disarm_and_reconnect_are_immediate_not_deferred(tmp_path):
    source=MappedSource(tmp_path/'operator.json')
    def step(buttons=(),snapshot=None):
        packet=source.sample(buttons)
        return source.parser.step(packet,now=source.now,clock_id=source.clock_id,snapshot=snapshot)
    step()
    confirmed=dict(clutched=True,confirmed=True,confirmed_candidate_id='a')
    step(snapshot=confirmed)
    assert step(['right.squeeze'],confirmed).action=='hold'
    assert step([],confirmed).action=='idle'
    assert step(CALIBRATE,confirmed).action=='release'
    step();source.publisher.source_instance_id='replacement'
    assert step(CALIBRATE,confirmed).reason=='vr_source_reconnected'
    assert step(CALIBRATE,confirmed).action is None
    step();assert step(CALIBRATE,{}).action=='clutch'


def test_future_pressed_at_boot_never_creates_a_clutch_edge(tmp_path):
    source=MappedSource(tmp_path/'operator.json')
    for _ in range(2):
        packet=source.sample(CALIBRATE)
        assert source.parser.step(packet,now=source.now,clock_id=source.clock_id).action is None
    packet=source.sample()
    source.parser.step(packet,now=source.now,clock_id=source.clock_id)
    packet=source.sample(CALIBRATE)
    assert source.parser.step(packet,now=source.now,clock_id=source.clock_id).action=='clutch'


def test_live_feed_accepts_bounded_future_selection_once_without_copying_ui_permissions(tmp_path):
    from test_copilot_sim_target_feed import FeedFixture
    f=FeedFixture(tmp_path);now=f.base.now
    packet=ui_packet(capture_timestamp=now+.0028, received_at_host_monotonic=now,
        expires_at=now+.15, select_candidate_id='grasp1')
    f.intent=packet
    result=f.prepare()
    assert result['batch'] is not None and len(f.calls)==1
    f.feed.revoke_validation()
    assert f.prepare()['batch'] is None
    changed=deepcopy(packet);changed['clock_uncertainty_s']-=.001
    with pytest.raises(ValueError,match='sim_selection_revision_reused_with_changed_content'):
        f.feed._intent(changed,now,'mono',f.selection)


@pytest.mark.parametrize('kind', ['raw_legacy','typed_legacy','partial','null_pair','unknown_source'])
def test_live_feed_cannot_gain_future_tolerance_from_missing_or_partial_provenance(tmp_path,kind):
    from test_copilot_sim_target_feed import FeedFixture
    f=FeedFixture(tmp_path);now=f.base.now
    packet=ui_packet(capture_timestamp=now+.0028, received_at_host_monotonic=now,
        expires_at=now+.15, select_candidate_id='grasp1')
    if kind in ('raw_legacy','typed_legacy'):
        for name in VR_TIME_FIELDS:packet.pop(name)
        if kind=='typed_legacy':packet=SimUIIntent.from_packet(packet)
    elif kind=='partial':packet.pop('clock_uncertainty_s')
    elif kind=='null_pair':
        for name in VR_TIME_FIELDS:packet[name]=None
    else:packet['source_kind']='desktop'
    with pytest.raises(ValueError):f.feed._intent(packet,now,'mono',f.selection)


def test_live_feed_original_object_selection_order_still_applies(tmp_path):
    from test_copilot_sim_target_feed import FeedFixture
    f=FeedFixture(tmp_path);now=f.base.now
    packet=ui_packet(capture_timestamp=now+.0028, received_at_host_monotonic=now,
        expires_at=now+.15, select_candidate_id='grasp1')
    selection=dict(f.selection,capture_timestamp=now+.005)
    with pytest.raises(ValueError,match='before_object_selection'):
        f.feed._intent(packet,now,'mono',selection)


@pytest.mark.parametrize('ordering', ['arrival_before_choice','capture_interval_overlaps_choice'])
def test_clock_uncertainty_cannot_reorder_a_button_across_object_choice(tmp_path,ordering):
    from test_copilot_sim_target_feed import FeedFixture
    f=FeedFixture(tmp_path);now=f.base.now
    packet=ui_packet(capture_timestamp=now+.0028,received_at_host_monotonic=now,
        expires_at=now+.15,select_candidate_id='grasp1')
    selection=dict(f.selection,capture_timestamp=now+.001 if ordering=='arrival_before_choice' else now-.001)
    with pytest.raises(ValueError,match='before_object_selection_with_clock_uncertainty'):
        f.feed._intent(packet,now,'mono',selection)
    # The rejected revision is spent even after time advances; recovery needs
    # a new deliberate event, not a replay of the old click.
    assert f.feed._intent(packet,now+.02,'mono',selection) is None
