from copy import deepcopy
from types import SimpleNamespace
import threading
import numpy as np
import pytest

from teleop_assist.copilot.vr_buttons import VRButtonParser
from teleop_assist.copilot.mailbox import ControlWriter, host_clock_id, read_json
from teleop_assist.copilot.dashboard import Dashboard, OperatorOwnership
from tools.copilot.instrument_xr import SharedXRTelemetry
from tools.copilot.xr_source import OperatorPublisher, main

CALIBRATE = ('left.aButton', 'left.bButton', 'right.aButton', 'right.bButton')
START = ('left.aButton', 'right.aButton')


class Source:
    def __init__(self, path, enabled=True):
        self.now=10.;self.sequence=0;self.clock_id=host_clock_id()
        self.publisher=OperatorPublisher(path,clock=lambda:self.now,vr_buttons=enabled)
        self.shared=SharedXRTelemetry()
        self.parser=VRButtonParser()

    def sample(self, pressed=(), *, kind='controller', capability=True, triggers=(0.,0.)):
        self.now+=.01;self.sequence+=2
        pose=np.eye(4).flatten(order='F').tolist()
        meta=dict(schema='punktnull.xr_frame.v1',sequence=self.sequence,sample_kind='webxr_pose_read',
            sample_time_s=self.now-9.,sample_clock_id='browser:test',host_sample_time_s=self.now,
            host_clock_id=self.clock_id,clock_uncertainty_s=.003,clock_sync_age_s=.1,
            sensor_capture_time_s=None,latency_unknown=False,left_tracking_valid=True,right_tracking_valid=True)
        self.shared.record('head',{'camera':{'matrix':pose},'_copilot':dict(meta,sequence=self.sequence-1,
            sample_kind='camera_pose_callback')},now=self.now)
        states={side:{name:side+'.'+name in pressed for name in ('aButton','bButton','squeeze')}
                for side in ('left','right')}
        for state in states.values():
            state['_copilot_gamepad']=dict(mapping='xr-standard',button_count=6,required_buttons_available=capability)
        for side,value in zip(('left','right'),triggers):states[side]['triggerValue']=value
        self.shared.record(kind,dict(left=pose*(25 if kind=='hand' else 1),right=pose*(25 if kind=='hand' else 1),
            leftState=states['left'],rightState=states['right'],_copilot=meta),now=self.now)
        self.tele_data=SimpleNamespace(left_wrist_pose=np.eye(4),right_wrist_pose=np.eye(4),copilot_envelope=self.shared.envelope())
        return self.publisher.publish(self.tele_data)

    def step(self, pressed=(), snapshot=None, **kwargs):
        return self.parser.step(self.sample(pressed,**kwargs),now=self.now,clock_id=self.clock_id,
                                candidate_ids=['a','b'],snapshot=snapshot)


@pytest.fixture
def source(tmp_path): return Source(tmp_path/'operator.json')


def test_mapping_edges_once_and_actual_event_time(source):
    assert source.step().action is None
    clutch=source.step(CALIBRATE)
    assert clutch.action=='clutch' and clutch.acquisition_timestamp==source.now
    assert clutch.expires_at==pytest.approx(source.now+.15)
    assert source.step(CALIBRATE).action is None
    source.step()
    assert source.step(START,snapshot={'clutched':True}).action=='start'
    assert source.step(START,snapshot={'clutched':True}).action=='idle'
    source.step()
    selected=source.step(['left.bButton'])
    assert selected.action=='select' and selected.candidate_id=='a'
    source.step()
    assert source.step(['left.bButton'],snapshot={'selected_candidate_id':'a'}).candidate_id=='b'
    source.step()
    assert source.step(['right.bButton']).action=='confirm'
    source.step()
    assert source.step(CALIBRATE,snapshot={'clutched':True}).action=='release'


def test_releasing_calibration_buttons_never_starts_teleoperation(source):
    source.step()
    assert source.step(CALIBRATE).action == 'clutch'
    assert source.step(START, snapshot={'clutched':True}).action == 'idle'
    source.step(snapshot={'clutched':True})
    assert source.step(START, snapshot={'clutched':True}).action == 'start'


def test_pressed_at_boot_and_reconnect_are_baseline_not_commands(source):
    assert source.step(START).action is None
    assert source.step(CALIBRATE).action is None
    assert source.step().action is None
    assert source.step(CALIBRATE).action=='clutch'
    source.publisher.source_instance_id='new-process'
    intent=source.step(CALIBRATE)
    assert intent.action=='release' and intent.reason=='vr_source_reconnected'
    assert source.step(CALIBRATE).action is None
    source.step()
    assert source.step(CALIBRATE).action=='clutch'


def test_confirmation_needs_squeeze_release_before_hold(source):
    source.step()
    confirmed=dict(clutched=True,confirmed=True,confirmed_candidate_id='a')
    assert source.step(['right.squeeze'],snapshot=confirmed).action=='idle'
    assert source.step([],snapshot=confirmed).action=='idle'
    assert source.step(['right.squeeze'],snapshot=confirmed).action=='hold'
    assert source.step([],snapshot=confirmed).action=='idle'
    # Different confirmed pose must not inherit an already pressed squeeze.
    source.step(['right.squeeze'],snapshot=confirmed)
    assert source.step(['right.squeeze'],snapshot=dict(confirmed,confirmed_candidate_id='b')).action=='idle'


def test_source_replication_and_new_head_never_renew_controller_lease(source):
    source.step();packet=source.sample(CALIBRATE)
    assert source.parser.step(packet,now=source.now,clock_id=source.clock_id).action=='clutch'
    source.now+=.02
    replicated=source.publisher.publish(source.tele_data)
    assert replicated['vr_buttons']==packet['vr_buttons']
    assert source.parser.step(replicated,now=source.now,clock_id=source.clock_id).action is None
    # A new head callback changes the normalized source sequence, not buttons.
    metadata=source.shared.read()['head'];metadata.update(sequence=99,host_sample_time_s=source.now)
    source.shared.record('head',dict(camera={'matrix':np.eye(4).flatten(order='F').tolist()},_copilot=metadata),now=source.now)
    source.tele_data.copilot_envelope=source.shared.envelope()
    new_head=source.publisher.publish(source.tele_data)
    assert new_head['vr_buttons']==packet['vr_buttons']
    assert source.parser.step(new_head,now=source.now,clock_id=source.clock_id).action is None
    source.now+=.16
    assert source.parser.step(new_head,now=source.now,clock_id=source.clock_id).action=='release'


@pytest.mark.parametrize('broken',['capability','hand','clock','sequence','epoch','time','expiry','states','reused','disabled','malformed'])
def test_unknown_invalid_or_replayed_metadata_rejects_and_requires_neutral(source,broken):
    source.step();packet=source.sample(CALIBRATE)
    source.parser.step(packet,now=source.now,clock_id=source.clock_id)
    packet=source.sample(['right.bButton'],kind='hand' if broken=='hand' else 'controller',capability=broken!='capability')
    payload=packet['vr_buttons']
    if broken=='clock':payload['clock_id']='other'
    if broken=='sequence':packet['source_event_sequences']=[]
    if broken=='epoch':payload['source_instance_id']='other'
    if broken=='time':payload['acquisition_timestamp']=10**500
    if broken=='expiry':payload['expires_at']=source.now+10
    if broken=='states':payload['states']['left']['aButton']=1
    if broken=='reused':
        payload['event_sequence']-=2;packet['source_event_sequences']['arms']-=2
    if broken=='disabled':packet.pop('vr_buttons')
    if broken=='malformed':payload['unexpected']=float('nan')
    rejected=source.parser.step(packet,now=source.now,clock_id=source.clock_id)
    assert rejected.action=='release'
    assert source.step(CALIBRATE).action is None
    source.step()
    assert source.step(CALIBRATE).action=='clutch'


def test_conflicting_edges_release_and_disarm_has_priority(source):
    source.step()
    assert source.step(['left.bButton','right.bButton']).action=='release'
    source.step()
    assert source.step([*CALIBRATE,'right.squeeze']).action=='release'
    source.step()
    assert source.step(START).reason=='vr_calibration_required_before_ax'


def test_default_publisher_does_not_enable_buttons_and_cli_rejects_handtracking(tmp_path):
    source=Source(tmp_path/'operator.json',enabled=False)
    assert 'vr_buttons' not in source.sample()
    with pytest.raises(SystemExit):main(['--sim','--check','--vr-buttons'])


def test_source_writer_keeps_capture_expiry_and_rejects_stale(tmp_path):
    writer=ControlWriter(tmp_path,clock=lambda:10.1,clock_id='test')
    sample=dict(acquisition_timestamp=10.,clock_id='test',expires_at=10.15,event_sequence=4)
    packet=writer.write('hold',client_id='xr:abc',source_sample=sample)
    assert packet['capture_timestamp']==10. and packet['expires_at']==10.15
    before=packet['revision']
    writer.clock=lambda:10.16
    with pytest.raises(ValueError):writer.write('hold',source_sample=sample)
    assert read_json(writer.path)['revision']==before


def bridge(source,tmp_path):
    # Deliberately no socket, browser, Vuer constructor or simulator.
    dashboard=Dashboard.__new__(Dashboard)
    dashboard.lock=threading.RLock();dashboard.sim_control=ControlWriter(tmp_path,clock=lambda:source.now)
    dashboard.sim_ownership=OperatorOwnership(dashboard.sim_control,clock=lambda:source.now)
    dashboard.vr_buttons=VRButtonParser();dashboard._vr_pending_revision=None
    return dashboard


def status(source,**snapshot):
    return dict(schema='punktnull.sim_runtime.v1',mode='isolated_PhysX',clock_id=source.clock_id,
                timestamp=source.now,candidates=[{'candidate_id':'a'},{'candidate_id':'b'}],snapshot=snapshot)


def test_bridge_shared_desktop_ownership_and_ack_prevent_takeover_or_overwrite(source,tmp_path):
    app=bridge(source,tmp_path)
    def tick(buttons=(),**snapshot):
        packet=source.sample(buttons)
        return app.consume_vr(packet,runtime_status=status(source,**snapshot),now=source.now)
    tick();app.sim_ownership.require('desktop',acquire=True)
    assert tick(CALIBRATE)['reason']=='vr_desktop_ownership_conflict'
    assert app.sim_ownership.client_id=='desktop'
    app.sim_ownership.release('desktop')
    assert not tick(CALIBRATE)['written']
    tick()
    clutch=tick(CALIBRATE);revision=clutch['revision']
    assert tick([],clutched=True)['reason']=='vr_waiting_for_core_ack'
    assert read_json(app.sim_control.path)['revision']==revision
    heartbeat=tick([],clutched=True,last_ui_revision=revision)
    assert heartbeat['written']
    packet=read_json(app.sim_control.path)
    assert packet['capture_timestamp']==source.now and packet['expires_at']==pytest.approx(source.now+.15)
    chosen=tick(['left.bButton'],clutched=True,last_ui_revision=packet['revision'])
    assert chosen['written']
    # Disarm may cancel while select has not yet been acknowledged.
    assert tick(CALIBRATE,clutched=True)['written']
    assert read_json(app.sim_control.path)['release'] and app.sim_ownership.client_id is None


def test_preclutch_choice_keeps_fresh_idle_lease_without_implicit_motion_or_confirm(source,tmp_path):
    app=bridge(source,tmp_path)
    def tick(buttons=(),**snapshot):
        packet=source.sample(buttons)
        return app.consume_vr(packet,runtime_status=status(source,**snapshot),now=source.now)
    tick()
    selection=tick(['left.bButton']);assert selection['written']
    assert read_json(app.sim_control.path)['select_candidate_id']=='a'
    tick([],last_ui_revision=selection['revision'])
    for _ in range(40):
        previous=read_json(app.sim_control.path)
        assert tick(['right.squeeze'],last_ui_revision=previous['revision'])['written']
        packet=read_json(app.sim_control.path)
        assert not any(packet[key] for key in ('clutch','assist_held','confirm','release'))
    tick()
    assert tick(CALIBRATE,last_ui_revision=packet['revision'])['written']
    assert read_json(app.sim_control.path)['clutch']


def test_scripted_runtime_is_readonly_and_lost_input_releases_only_vr_owner(source,tmp_path):
    app=bridge(source,tmp_path);packet=source.sample()
    before=read_json(app.sim_control.path)
    assert not app.consume_vr(packet,runtime_status=dict(status(source),scripted_operator=True),now=source.now)['written']
    assert read_json(app.sim_control.path)==before
    app.consume_vr(packet,runtime_status=status(source),now=source.now)
    packet=source.sample(CALIBRATE)
    app.consume_vr(packet,runtime_status=status(source),now=source.now)
    assert app.sim_ownership.client_id.startswith('xr:')
    assert app.consume_vr({},runtime_status=status(source),now=source.now)['written']
    assert read_json(app.sim_control.path)['release']
    app.sim_ownership.require('desktop',acquire=True)
    before=read_json(app.sim_control.path)
    assert not app.consume_vr({},runtime_status=status(source),now=source.now)['written']
    assert read_json(app.sim_control.path)==before
