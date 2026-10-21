"""Paired real parser/dashboard/core regression; no browser, DDS or physics.

Reproduces run_003 stale acknowledged UI302 across a browser epoch restart.
The original recorded failure remains untouched; this is a synthetic contract
regression using real OperatorPublisher, VRButtonParser and SimControlSession.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from test_copilot_vr_buttons import CALIBRATE, Source, bridge
from test_copilot_sim_control import Fixture
from teleop_assist.copilot.mailbox import read_json
from teleop_assist.copilot.sim_control import SimControlSession
from teleop_assist.copilot.types import GraspCandidate, ProposalBatch


class BrowserSource(Source):
    browser_clock='browser:old'

    def sample(self, pressed=(), *, capability=True):
        self.now+=.01;self.sequence+=2
        pose=np.eye(4).flatten(order='F').tolist()
        meta=dict(schema='punktnull.xr_frame.v1',sequence=self.sequence,sample_kind='webxr_pose_read',
            sample_time_s=self.now-9.,sample_clock_id=self.browser_clock,host_sample_time_s=self.now,
            host_clock_id=self.clock_id,clock_uncertainty_s=.003,clock_sync_age_s=.1,
            sensor_capture_time_s=None,latency_unknown=False,left_tracking_valid=True,right_tracking_valid=True)
        self.shared.record('head',dict(camera={'matrix':pose},_copilot=dict(meta,sequence=self.sequence-1,
            sample_kind='camera_pose_callback')),now=self.now)
        states={side:{name:side+'.'+name in pressed for name in ('aButton','bButton','squeeze')}
                for side in ('left','right')}
        for state in states.values():
            state['_copilot_gamepad']=dict(mapping='xr-standard',button_count=6,required_buttons_available=capability)
        self.shared.record('controller',dict(left=pose,right=pose,leftState=states['left'],
            rightState=states['right'],_copilot=meta),now=self.now)
        self.tele_data=SimpleNamespace(left_wrist_pose=np.eye(4),right_wrist_pose=np.eye(4),
                                      copilot_envelope=self.shared.envelope())
        return self.publisher.publish(self.tele_data)


class NeverWorker:
    def __init__(self):self.calls=[]
    def call(self,*args,**kwargs):
        self.calls.append((args,kwargs))
        raise RuntimeError('No motion authorized by a neutral baseline or UI action')


class Paired:
    def __init__(self,tmp_path):
        self.source=BrowserSource(tmp_path/'operator.json')
        self.dashboard=bridge(self.source,tmp_path)
        self.fixture=Fixture()
        self.worker=NeverWorker()
        self.core=SimControlSession(self.worker,clock=lambda:self.source.now)
        self.runtime=dict(schema='punktnull.sim_runtime.v1',mode='isolated_PhysX',
            clock_id=self.source.clock_id,timestamp=self.source.now,
            candidates=[{'candidate_id':'a'},{'candidate_id':'b'}],snapshot={})

    def consume(self,packet):
        return self.dashboard.consume_vr(packet,runtime_status=self.runtime,now=self.source.now)

    def core_step(self,packet):
        f=self.fixture;f.now=self.source.now
        state=f.state();scene=f.scene()
        state['clock_id']=scene['clock_id']=self.source.clock_id
        batch=replace(f.proposal(),clock_id=self.source.clock_id,
            candidates=tuple(replace(f.proposal().candidates[0],candidate_id=x) for x in ('a','b')))
        self.runtime=self.core.step(packet,read_json(self.dashboard.sim_control.path),state,scene,batch,
            measured_left_base_ee=f.left,measured_right_base_ee=f.right,
            now=self.source.now,clock_id=self.source.clock_id,calibration_id='cal')
        self.runtime['candidates']=[{'candidate_id':'a'},{'candidate_id':'b'}]
        return self.runtime

    def tick(self,buttons=(),*,core=True):
        packet=self.source.sample(buttons)
        result=self.consume(packet)
        if core:self.core_step(packet)
        return result,packet

    def start(self):
        self.tick()  # Real startup neutral gives the deliberately ignored first UI revision.
        result,_=self.tick(['left.bButton'])
        assert result['written'] and self.runtime['snapshot']['selected_candidate_id']=='a'
        self.tick()
        result,_=self.tick(CALIBRATE)
        assert result['written'] and self.runtime['snapshot']['clutched']
        # Explicit release gives an acknowledged old epoch and free ownership.
        self.tick()
        self.tick(CALIBRATE)
        assert not self.core.clutched
        self.dashboard.sim_control.revision=301
        self.dashboard.sim_control.write('release',client_id=self.dashboard.vr_buttons.client_id)
        self.tick()
        assert self.runtime['snapshot']['last_ui_revision']==302
        self.source.browser_clock='browser:new'
        self.source.sequence=0

    def synchronize(self):
        results=[]
        for _ in range(8):
            result,_=self.tick()
            results.append(result)
            if (getattr(self.dashboard,'_vr_reconnect_epoch',None) is None
                    and self.dashboard.vr_buttons.epoch[1]=='browser:new'):
                return results
        raise AssertionError((results,self.runtime))


@pytest.mark.parametrize('new_publisher',[False,True])
def test_actual_pair_new_epoch_neutral_crosses_old_acked_UI_before_first_select(tmp_path,new_publisher):
    p=Paired(tmp_path);p.start()
    if new_publisher:p.source.publisher.source_instance_id='publisher:restarted'
    # Mirrors the stale acknowledged UI baseline in run_003, not a fresh core.
    p.source.now+=.16
    results=p.synchronize()
    assert any(r.get('reason')=='vr_neutral_core_baseline' for r in results)
    assert p.core._source_epoch[1]=='browser:new'
    assert p.dashboard.sim_ownership.client_id is None
    assert not p.core.clutched and not p.worker.calls
    result,packet=p.tick(['left.bButton'])
    written=read_json(p.dashboard.sim_control.path)
    assert result['written'] and written['select_candidate_id']=='a'
    assert written['capture_timestamp']==packet['vr_buttons']['acquisition_timestamp']
    assert written['expires_at']==packet['vr_buttons']['expires_at']
    assert p.runtime['snapshot']['selected_candidate_id']=='a'
    assert not p.core.clutched and not p.worker.calls


def test_reconnect_neutral_revokes_old_clutch_without_new_ownership(tmp_path):
    p=Paired(tmp_path);p.tick();p.tick(CALIBRATE)
    assert p.core.clutched
    p.source.browser_clock='browser:new'
    results=p.synchronize()
    assert not p.core.clutched and p.dashboard.sim_ownership.client_id is None
    assert not p.worker.calls
    last=read_json(p.dashboard.sim_control.path)
    assert not any(last[k] for k in ('clutch','confirm','select_candidate_id','assist_held','release'))


def test_desktop_owner_mailbox_not_replaced_by_reconnect_baseline(tmp_path):
    p=Paired(tmp_path);p.start()
    p.dashboard.sim_ownership.require('desktop',acquire=True)
    before=read_json(p.dashboard.sim_control.path)
    for _ in range(5):p.tick(core=False)
    assert p.dashboard.sim_ownership.client_id=='desktop'
    assert read_json(p.dashboard.sim_control.path)==before


def pending_pair(tmp_path):
    p=Paired(tmp_path);p.start()
    p.source.now+=.16
    for _ in range(5):
        result,packet=p.tick(core=False)
        if result.get('reason')=='vr_neutral_core_baseline':return p,packet
    raise AssertionError('No fresh reconnect baseline')


def test_waiting_edge_keeps_original_event_deadline_when_core_ack_arrives(tmp_path):
    p,neutral=pending_pair(tmp_path)
    result,pressed=p.tick(['left.bButton'],core=False)
    assert not result['written']
    original=pressed['vr_buttons']
    # Process the neutral mailbox against a new source before releasing the
    # first still-current edge. No UI action has reached the core yet.
    p.core_step(pressed)
    if 'operator_source_sequence' not in p.runtime:
        p.source.now+=.01;p.core_step(pressed)
    result,held=p.tick(['left.bButton'])
    assert result['written']
    written=read_json(p.dashboard.sim_control.path)
    assert written['select_candidate_id']=='a'
    assert written['capture_timestamp']==original['acquisition_timestamp']
    assert written['expires_at']==original['expires_at']
    assert written['source_event_sequence']==original['event_sequence']
    assert written['capture_timestamp']<held['vr_buttons']['acquisition_timestamp']
    assert p.runtime['snapshot']['selected_candidate_id']=='a'


def test_late_held_samples_cannot_renew_edge_waiting_for_neutral_ack(tmp_path):
    p,neutral=pending_pair(tmp_path)
    p.tick(['left.bButton'],core=False)
    for _ in range(18):p.tick(['left.bButton'],core=False)
    # Recover with genuinely fresh all-up samples; expired press cannot replay.
    p.synchronize()
    assert p.runtime['snapshot'].get('selected_candidate_id') is None
    assert p.dashboard.sim_ownership.client_id is None
    result,_=p.tick(['left.bButton'])
    assert result['written'] and p.runtime['snapshot']['selected_candidate_id']=='a'


def test_delayed_successful_core_ACK_never_relicenses_expired_held_edge(tmp_path):
    p,neutral=pending_pair(tmp_path)
    pending_status=deepcopy(p.runtime)
    _,pressed=p.tick(['left.bButton'],core=False)
    p.core_step(pressed)
    if 'operator_source_sequence' not in p.runtime:
        p.source.now+=.01;p.core_step(pressed)
    successful_ack=deepcopy(p.runtime)
    assert 'operator_source_sequence' in successful_ack
    p.runtime=pending_status  # Delayed observation of an actual paired-core ACK.
    for _ in range(18):p.tick(['left.bButton'],core=False)
    p.runtime=successful_ack
    result,held=p.tick(['left.bButton'],core=False)
    assert not result['written']
    assert p.dashboard.sim_ownership.client_id is None
    assert not p.dashboard.vr_buttons.armed
    assert p.runtime['snapshot'].get('selected_candidate_id') is None


def test_new_epoch_with_held_buttons_requires_actual_neutral_before_actions(tmp_path):
    p=Paired(tmp_path);p.start()
    for _ in range(8):
        result,_=p.tick([*CALIBRATE,'right.squeeze'])
        assert not p.core.clutched
    assert not p.worker.calls
    p.synchronize()
    result,_=p.tick(CALIBRATE)
    assert result['written'] and p.core.clutched


def test_three_reads_per_owner_tick_do_not_outrun_neutral_core_ack(tmp_path):
    p=Paired(tmp_path);p.start()
    p.source.now+=.16
    for _ in range(8):
        cached_status=deepcopy(p.runtime)
        first=p.source.sample()
        p.consume(first)  # Owner-loop read.
        registered=getattr(p.dashboard,'_vr_reconnect_baseline',None)
        before=read_json(p.dashboard.sim_control.path)
        second=p.source.sample()
        p.consume(second)  # Fresh post-endpoint read, same published core status.
        if registered is not None and cached_status['snapshot'].get('last_ui_revision',0)<registered['revision']:
            assert read_json(p.dashboard.sim_control.path)==before
        core_status=deepcopy(p.core_step(second))
        p.runtime=cached_status
        third=p.source.sample()
        p.consume(third)  # Final handoff read before owner publishes core result.
        p.runtime=core_status
    assert p.dashboard._vr_reconnect_epoch is None
    assert p.core._source_epoch[1]=='browser:new'
    assert not p.core.clutched and not p.worker.calls
    result,_=p.tick(['left.bButton'])
    assert result['written'] and p.runtime['snapshot']['selected_candidate_id']=='a'


def test_unacked_neutral_lease_is_not_renewed_until_expiry_and_observed_core_hold(tmp_path):
    p,_=pending_pair(tmp_path)
    before=read_json(p.dashboard.sim_control.path)
    for _ in range(18):
        result,_=p.tick(core=False)
        assert not result['written']
        assert read_json(p.dashboard.sim_control.path)==before
    assert p.source.now>before['expires_at']
    # An actual new core hold observed after the original lease expired permits
    # a new source-bound neutral attempt, never implicit clutch or ownership.
    p.runtime=p.core.hold('external_hold',now=p.source.now,clock_id=p.source.clock_id)
    p.runtime['candidates']=[{'candidate_id':'a'},{'candidate_id':'b'}]
    result,packet=p.tick(core=False)
    assert result['written'] and result['reason']=='vr_neutral_core_baseline'
    current=read_json(p.dashboard.sim_control.path)
    assert current['revision']>before['revision']
    assert current['capture_timestamp']==packet['vr_buttons']['acquisition_timestamp']
    assert current['expires_at']==packet['vr_buttons']['expires_at']
    assert not any(current[k] for k in ('clutch','confirm','select_candidate_id','assist_held','release'))
    assert not p.core.clutched and p.dashboard.sim_ownership.client_id is None
