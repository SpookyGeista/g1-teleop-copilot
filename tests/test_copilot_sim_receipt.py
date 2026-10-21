"""Immutable consumed-command evidence; fake worker, no physics or transport."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import numpy as np
import pytest

from teleop_assist.copilot.sim_control import SimCommandReceipt
from test_copilot_sim_control import Fixture, Worker, race_ui, recheck


def digest(state,scene):
    return hashlib.sha256(json.dumps({'robot_state':state,'scene':scene},
        sort_keys=True,allow_nan=False).encode()).hexdigest()


class BoundWorker(Worker):
    """Adds honest fake-input binding, not real robot/FK validation."""
    command_format='unitree_float32'
    def call(self,operation,**payload):
        result=super().call(operation,**payload)
        result['q14_command']=np.asarray(result['q14_command'],dtype=np.float32).astype(float).tolist()
        result['command_format']=self.command_format
        result['snapshot_sha256']=digest(payload['robot_state'],payload['scene'])
        self.last_result=result
        return result


def fixture(*, assisting=False):
    f=Fixture();f.worker=BoundWorker(f);f.session.worker=f.worker
    offer=f.start()
    if assisting:
        f.tick(select_candidate_id='grasp');f.tick(confirm=True)
        offer=f.tick(assist_held=True)
        assert offer['assistance_applied']
    return f,offer


def receipt(f,offer,**changes):
    args=dict(now=f.now,clock_id='mono',measured_state=f.last['state'],scene=f.last['scene'])
    args.update(changes)
    return f.session.consume_receipt(offer,**args)


def test_receipt_exports_private_guard_values_and_distinct_exact_snapshot_bindings():
    f,offer=fixture();original_state=deepcopy(f.last['state']);original_scene=deepcopy(f.last['scene'])
    current_state=deepcopy(original_state);current_scene=deepcopy(original_scene)
    current_state['acquisition_timestamp']+=.005;current_scene['acquisition_timestamp']+=.005
    r=receipt(f,offer,now=f.now+.005,measured_state=current_state,scene=current_scene)
    assert isinstance(r,SimCommandReceipt) and isinstance(r.q14,tuple)
    assert r.q14==tuple(f.worker.last_result['q14_command'])
    assert r.command_format=='unitree_float32' and r.guard_sequence==f.worker.last_result['sequence']
    assert r.valid_until==offer['expires_at'] and r.clock_id=='mono' and r.token==offer['command_token']
    assert r.guard_snapshot_sha256==digest(original_state,original_scene)
    assert r.consumed_snapshot_sha256==digest(current_state,current_scene)
    assert r.guard_snapshot_sha256!=r.consumed_snapshot_sha256
    assert json.loads(r.guard_json)==f.worker.last_result
    assert json.loads(r.guarded_state_json)==original_state
    assert json.loads(r.guarded_scene_json)==original_scene
    assert json.loads(r.measured_state_json)==current_state and json.loads(r.scene_json)==current_scene
    assert r.scope=='isolated_PhysX_consumed_arm_command'
    assert f.session.status['command_consumed_once']


def test_rechecked_shorter_private_deadline_is_exported_without_rewriting_guard_time():
    f,offer=fixture();old_deadline=offer['expires_at'];short=f.now+.025
    assert recheck(f,ui=race_ui(f,expires_at=short))
    assert offer['expires_at']==old_deadline  # Caller still has the older copy.
    r=receipt(f,offer,now=f.now+.006)
    assert r.valid_until==short<old_deadline
    assert json.loads(r.guard_json)['valid_until']==f.worker.last_result['valid_until']>short
    assert json.loads(r.guarded_state_json)['acquisition_timestamp']==f.last['state']['acquisition_timestamp']


def test_public_offer_cannot_override_private_command_deadline_clock_or_guard():
    f,offer=fixture();short=f.now+.025
    assert recheck(f,ui=race_ui(f,expires_at=short))
    expected=tuple(f.worker.last_result['q14_command'])
    offer.update(q14_command=[99.]*14,expires_at=f.now+999,clock_id='forged',
        guard={'sequence':999,'command_format':'hardware'},snapshot_sha256='f'*64)
    r=receipt(f,offer,now=f.now+.006)
    assert r.q14==expected and r.valid_until==short and r.clock_id=='mono'
    assert r.guard_sequence==f.worker.last_result['sequence']!=999
    assert r.guard_snapshot_sha256!='f'*64


def test_public_deadline_extension_cannot_resurrect_privately_expired_command():
    f,offer=fixture();short=f.now+.025
    assert recheck(f,ui=race_ui(f,expires_at=short))
    offer['expires_at']=f.now+999
    assert receipt(f,offer,now=short) is None
    assert f.session.status['handoff_rejection_reason']=='command_expired_before_sim_handoff'
    assert not f.session.clutched


@pytest.mark.parametrize('first,second',[('receipt','receipt'),('receipt','array'),('array','receipt')])
def test_receipt_and_legacy_array_share_one_private_consumption(first,second):
    f,offer=fixture()
    call=lambda kind:receipt(f,offer) if kind=='receipt' else f.consume(offer)
    assert call(first) is not None
    assert call(second) is None and not f.session.clutched


def test_receipt_and_legacy_consume_commit_the_same_checked_assistance_offset():
    a,offer_a=fixture(assisting=True);b,offer_b=fixture(assisting=True)
    expected=a.session._pending['assist_offset'].copy()
    assert not np.array_equal(expected,a.session._committed_offset)
    r=receipt(a,offer_a);q=b.consume(offer_b)
    np.testing.assert_array_equal(r.q14,q)
    np.testing.assert_array_equal(a.session._committed_offset,expected)
    np.testing.assert_array_equal(a.session._committed_offset,b.session._committed_offset)


def test_receipt_contains_no_mutable_reference_to_worker_input_status_or_parsed_copy():
    f,offer=fixture();expected=tuple(f.worker.last_result['q14_command'])
    r=receipt(f,offer);before=(r.guard_json,r.guarded_state_json,r.measured_state_json,r.scene_json)
    with pytest.raises(FrozenInstanceError):r.valid_until=999.
    with pytest.raises(TypeError):r.q14[0]=999.
    parsed=json.loads(r.guard_json);parsed['q14_command'][0]=999.
    f.worker.last_result['q14_command'][0]=999.
    f.last['state']['joint_positions']['left_elbow_joint']=999.
    f.last['scene']['obstacles'].clear();offer['guard']['sequence']=999
    f.session.status['expires_at']=999.
    assert r.q14==expected and before==(r.guard_json,r.guarded_state_json,r.measured_state_json,r.scene_json)
    assert json.loads(r.guard_json)['q14_command'][0]!=999.


@pytest.mark.parametrize('fault',['finger','body','base','scene','wrong_clock','expired','token','superseded'])
def test_receipt_retains_current_state_scene_token_and_lease_gates(fault):
    f,offer=fixture();args={}
    if fault in ('finger','body','base'):
        state=deepcopy(f.last['state'])
        if fault=='finger':state['joint_positions']['right_hand_thumb_0_joint']+=.01
        elif fault=='body':state['observed_body_world_poses']['right_wrist_yaw_link'][0][3]+=.01
        else:state['base_in_world'][0][3]+=.01
        args['measured_state']=state
    elif fault=='scene':
        scene=deepcopy(f.last['scene']);scene['obstacles'][0]['size_xyz'][0]+=.01;args['scene']=scene
    elif fault=='wrong_clock':args['clock_id']='other'
    elif fault=='expired':args['now']=offer['expires_at']
    elif fault=='token':offer['command_token']='unknown'
    else:f.tick()
    assert receipt(f,offer,**args) is None
    assert not f.session.clutched


def test_receipt_encoding_delay_is_checked_before_return_and_offset_commit(monkeypatch):
    f,offer=fixture(assisting=True);original=f.session._command_receipt
    def delayed(*args):
        value=original(*args);f.clock+=.151;return value
    monkeypatch.setattr(f.session,'_command_receipt',delayed)
    assert receipt(f,offer) is None
    assert f.session.status['handoff_rejection_reason']=='command_expired_during_sim_handoff_checks'
    np.testing.assert_array_equal(f.session._committed_offset,np.eye(3))
    assert f.session._pending is None and not f.session.clutched


def test_worker_snapshot_digest_must_really_bind_original_guard_inputs():
    f,offer=fixture();f.session._pending['guard']['snapshot_sha256']='f'*64
    assert receipt(f,offer) is None
    assert f.session.status['handoff_rejection_reason']=='receipt_original_guard_snapshot_binding_invalid'


def test_default_float64_is_recorded_honestly_without_implicit_quantization():
    f=Fixture();f.worker=BoundWorker(f);f.worker.command_format='float64';f.session.worker=f.worker
    offer=f.start();r=receipt(f,offer)
    assert r.command_format=='float64'
    assert json.loads(r.guard_json)['command_format']=='float64'


def test_consumed_receipt_is_only_evidence_and_does_not_rearm_session():
    f,offer=fixture();r=receipt(f,offer)
    f.session.hold('downstream_refusal',now=f.now,clock_id='mono')
    assert r.token==offer['command_token'] and r.valid_until>f.now
    assert not f.session.clutched and not f.tick()['allowed']
    assert receipt(f,offer) is None


def recheck_receipt(f,r,*,operator=None,ui=None,now=None,clock_id='mono'):
    return f.session.recheck_receipt_inputs(r,f.last['packet'] if operator is None else operator,
        f.last['ui'] if ui is None else ui,now=f.now+.005 if now is None else now,clock_id=clock_id)


def test_postencoding_recheck_uses_inflight_context_without_new_consume_or_IK():
    f,offer=fixture(assisting=True);r=receipt(f,offer);calls=len(f.worker.calls)
    offset=f.session._committed_offset.copy();revision=f.session._ui_revision
    assert f.session._pending is None
    updated=recheck_receipt(f,r)
    assert isinstance(updated,SimCommandReceipt) and updated is not r and updated==r
    assert f.session.clutched and f.session._pending is None
    assert len(f.worker.calls)==calls and f.session._ui_revision==revision
    np.testing.assert_array_equal(f.session._committed_offset,offset)
    assert recheck_receipt(f,updated,now=f.now+.006)==updated


def test_postencoding_recheck_can_only_shorten_private_deadline():
    f,offer=fixture();r=receipt(f,offer);short=f.now+.025
    shortened=recheck_receipt(f,r,ui=race_ui(f,expires_at=short))
    assert shortened.valid_until==short<r.valid_until
    heartbeat=race_ui(f,expires_at=f.now+.15);heartbeat['revision']+=1
    later=recheck_receipt(f,shortened,ui=heartbeat,now=f.now+.006)
    assert later.valid_until==short and later.guard_json==r.guard_json
    # The old longer value can no longer match the private inflight evidence.
    assert recheck_receipt(f,r,ui=heartbeat,now=f.now+.007) is None
    assert not f.session.clutched


@pytest.mark.parametrize('field,value',[
    ('q14',tuple([99.]*14)),('valid_until',999.),('valid_until',0.),('clock_id','forged'),
    ('token','forged'),('guard_sequence',999),('command_format','hardware'),
    ('guard_snapshot_sha256','f'*64),('consumed_snapshot_sha256','f'*64),
    ('guard_json','{}'),('guarded_state_json','{}'),('guarded_scene_json','{}'),
    ('measured_state_json','{}'),('scene_json','{}'),('scope','hardware')])
def test_replaced_receipt_cannot_override_any_private_evidence(field,value):
    f,offer=fixture();r=receipt(f,offer)
    altered=replace(r,**{field:value})
    assert recheck_receipt(f,altered) is None
    assert f.session.status['reason']=='receipt_missing_changed_or_superseded'
    assert f.session._inflight_receipt is None


def test_private_binding_does_not_alias_even_deliberately_mutated_frozen_value():
    f,offer=fixture();r=receipt(f,offer)
    # Python reflection is outside the value-object contract, but must not
    # alter the independently retained private binding by shared identity.
    object.__setattr__(r,'valid_until',999.)
    assert recheck_receipt(f,r) is None
    assert not f.session.clutched


@pytest.mark.parametrize('action',['release','clutch','owner'])
def test_new_UI_stop_action_or_owner_revokes_consumed_command(action):
    f,offer=fixture(assisting=True);r=receipt(f,offer);ui=race_ui(f)
    if action=='select':ui['select_candidate_id']='grasp'
    elif action=='assist_release':ui['assist_held']=False
    elif action=='owner':ui['client_id']='different_owner'
    else:ui[action]=True
    assert recheck_receipt(f,r,ui=ui) is None
    assert not f.session.clutched and f.session._inflight_receipt is None
    assert not f.tick()['allowed']


@pytest.mark.parametrize('fault',['tracking','source_epoch','event_epoch','sequence','stale','future','clock'])
def test_new_operator_faults_revoke_consumed_command(fault):
    f,offer=fixture();r=receipt(f,offer);packet=deepcopy(f.last['packet'])
    if fault=='tracking':packet['tracking_valid']=False
    elif fault=='source_epoch':packet['source_instance_id']='restarted'
    elif fault=='event_epoch':packet['source_event_clock_id']='restarted'
    elif fault=='sequence':
        packet['source_sequence']-=1;packet['envelope']['sequence']-=1;packet['publish_sequence']-=1
    elif fault in ('stale','future'):
        offset=-1. if fault=='stale' else 1.
        packet['acquisition_timestamp']+=offset;packet['envelope']['acquisition_timestamp']+=offset
    else:packet['clock_id']='other'
    assert recheck_receipt(f,r,operator=packet) is None
    assert not f.session.clutched and f.session._inflight_receipt is None


def test_new_valid_operator_sample_does_not_retimestamp_or_replace_consumed_target():
    f,offer=fixture();r=receipt(f,offer);packet=deepcopy(f.last['packet']);stamp=f.now+.005
    packet.update(source_sequence=packet['source_sequence']+1,publish_sequence=packet['publish_sequence']+1,
        acquisition_timestamp=stamp,published_at=stamp,expires_at=stamp+.15)
    packet['envelope'].update(sequence=packet['source_sequence'],acquisition_timestamp=stamp,received_at_host_monotonic=stamp)
    packet['right_base_ee'][0][3]+=.01
    updated=recheck_receipt(f,r,operator=packet)
    assert updated is not None and updated.q14==r.q14 and updated.guard_json==r.guard_json
    assert updated.valid_until==r.valid_until
    assert json.loads(updated.guarded_state_json)['acquisition_timestamp']<stamp


@pytest.mark.parametrize('change',['step','hold','other_session'])
def test_receipt_cannot_survive_new_control_cycle_or_move_to_another_session(change):
    f,offer=fixture();r=receipt(f,offer)
    if change=='step':
        assert f.tick()['allowed'] and f.session._inflight_receipt is None
    elif change=='hold':f.session.hold('downstream_refused',now=f.now,clock_id='mono')
    else:f,_=fixture()
    assert recheck_receipt(f,r) is None


def test_legacy_array_consumption_does_not_install_inflight_receipt_authority():
    f,offer=fixture();assert f.consume(offer) is not None
    assert f.session._inflight_receipt is None


@pytest.mark.parametrize('kind',['elapsed','clock_regression','revoked'])
def test_expiry_or_revocation_inside_receipt_recheck_cannot_restore_authority(monkeypatch,kind):
    f,offer=fixture(assisting=True);r=receipt(f,offer);original=f.session._operator_sequence
    def delayed(*args):
        value=original(*args)
        if kind=='elapsed':f.clock+=.151
        elif kind=='clock_regression':f.clock-=.01
        else:f.session.hold('simulated_concurrent_release',now=f.now,clock_id='mono')
        return value
    monkeypatch.setattr(f.session,'_operator_sequence',delayed)
    assert recheck_receipt(f,r) is None
    assert not f.session.clutched and f.session._inflight_receipt is None
    if kind=='elapsed':assert f.session.status['reason']=='receipt_source_lease_expired_during_recheck'
    if kind=='revoked':assert f.session.status['reason']=='receipt_revoked_during_recheck'
