"""Pinned original ctor + real CDR + private session, with labelled fake PhysX.

Tests run in audit-guarded children: no DDS participant/socket, original module,
torch/Isaac initialization, logging process or actual robot is permitted.
The fake guard and articulation prove integration contracts, not motion safety.
"""
from copy import deepcopy
from dataclasses import replace, asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_copilot_unitree_codec import child

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/home/loya2/xr_teleoperate/teleop/robot_control/robot_arm.py')
ASSET = Path('/home/loya2/unitree_sim_isaaclab/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd')


class Tensor(np.ndarray):
    """Minimal CPU-only Articulation tensor fixture, not imported torch."""
    def clone(self):return self.copy().view(Tensor)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return np.asarray(self)
    def new_tensor(self,value):return np.asarray(value,dtype=self.dtype).view(Tensor)


def assert_equal(a,b):
    assert np.array_equal(a,b), (np.asarray(a), np.asarray(b))


def tensor(value):return np.asarray(value,dtype=np.float32).view(Tensor)


def rig(codec, *, assisting=False):
    from teleop_assist.copilot.original_sim_snapshot import OriginalSimSnapshotAdapter,G1_JOINT_NAMES,G1_BODY_NAMES
    from teleop_assist.copilot.original_xr_sender import OriginalSimReceiptSink,guarded_original_controller_class
    from test_copilot_sim_control import Fixture
    from test_copilot_sim_receipt import BoundWorker
    f=Fixture();f.worker=BoundWorker(f);f.session.worker=f.worker
    f.generation=0;f.target_calls=[];f.fail_write=False
    names=sorted(G1_JOINT_NAMES);bodies=sorted(G1_BODY_NAMES)
    body=np.zeros((1,55,7));body[0,:,3]=1
    offset=np.eye(4);offset[0,3]=-.05
    for side in ('left','right'):
        pose=getattr(f,side)@offset;idx=bodies.index(side+'_wrist_yaw_link')
        body[0,idx,:3]=pose[:3,3]
        quaternion=Rotation.from_matrix(pose[:3,:3]).as_quat()
        body[0,idx,3:]=quaternion[[3,0,1,2]]
    q=[.007*(i+1) if 'hand_' in name or 'hip_' in name else 0. for i,name in enumerate(names)]
    data=SimpleNamespace(joint_names=names,body_names=bodies,joint_pos=tensor([q]),
        joint_vel=tensor(np.zeros((1,43))),root_pose_w=tensor([[0,0,0,1,0,0,0]]),
        body_link_pose_w=tensor(body))
    def install(value):
        f.target_calls.append(np.asarray(value).copy())
        if f.fail_write:raise RuntimeError('test_fixture_installation_unknown')
    robot=SimpleNamespace(data=data,joint_names=names,body_names=bodies,
        cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path=str(ASSET))),set_joint_position_target=install)
    sim_scene={'robot':robot,'obstacle':np.eye(4)}
    sim_scene['obstacle'][:3,3]=[.5,0,.8]
    def scene_reader(scene,context):
        return dict(source='current_simulator_complete_collision_scene',geometry_complete=True,static=True,
            scene_id='TEST_FIXTURE_complete_known_table',obstacles=[dict(id='table',type='box',
                size_xyz=[.5,.8,.04],world_pose=scene['obstacle'].tolist())],
            allowed_static_contacts=[],**asdict(context))
    source=OriginalSimSnapshotAdapter(sim_scene,read_generation=lambda:f.generation,clock_id='mono',
        source_instance_id='TEST_FIXTURE_PhysX_owner',calibration_id='cal',
        robot_asset_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        collision_scene_reader=scene_reader,simulation_mode=True,clock=lambda:f.clock)
    f.source=source;f.robot=robot;f.sim_scene=sim_scene
    def step(**actions):
        f.now+=.02;f.clock=f.now;f.sequence+=1;f.revision+=1;f.generation+=1
        sample=source.capture();f.left=sample.left_base_ee;f.right=sample.right_base_ee
        packet=f.packet();ui=dict(revision=f.revision,capture_timestamp=f.now,clock_id='mono',client_id='test_ui',**actions)
        f.last=dict(packet=deepcopy(packet),ui=deepcopy(ui),state=sample.robot_state,scene=sample.scene)
        return f.session.step(packet,ui,sample.robot_state,sample.scene,f.proposal(),
            measured_left_base_ee=f.left,measured_right_base_ee=f.right,
            now=f.now,clock_id='mono',calibration_id='cal')
    f.step=step
    assert not step(release=True)['allowed']
    assert step(clutch=True)['snapshot']['clutched']
    offer=step()
    if assisting:
        step(select_candidate_id='grasp');step(confirm=True);offer=step(assist_held=True)
        assert offer['assistance_applied']
    f.receipt=f.session.consume_receipt(offer,now=f.now,clock_id='mono',
        measured_state=f.last['state'],scene=f.last['scene'])
    assert f.receipt is not None
    f.sink=OriginalSimReceiptSink(f.session,source,simulation_mode=True,codec=codec,clock=lambda:f.clock)
    cls=guarded_original_controller_class(ORIGINAL,f.sink)
    f.controller=cls(simulation_mode=True)
    f.read_inputs=lambda:(f.last['packet'],f.last['ui'])
    f.target_check=lambda:dict(accepted=True,source='TEST_FIXTURE_only_no_geometry_claim')
    f.publish=lambda value=None:f.controller.publish_receipt(f.receipt if value is None else value,
        read_inputs=f.read_inputs,recheck_target=lambda:f.target_check())
    return f


def performed_case(name, codec):
    from teleop_assist.copilot.original_xr_sender import (
        OriginalSimReceiptSink,guarded_original_controller_class,NONARM_POLICY)
    from test_copilot_sim_control import race_ui
    f=rig(codec,assisting=True)
    original=codec.encode_arm_targets
    if name=='constructor':
        assert not f.target_calls and f.controller.lowcmd_publisher.write_attempts==0
        assert not f.controller.publish_thread.is_alive() and not f.controller.subscribe_thread.is_alive()
        assert f.controller.publish_thread.started
        f.controller.q_target[:]=99
        f.controller.tauff_target[:]=88
        for motor in f.controller.msg.motor_cmd:
            motor.q=77;motor.tau=66;motor.kp=1000
        f.controller._ctrl_motor_state()
        assert not f.target_calls
        result=f.publish();assert result['applied'],result
        actual=f.target_calls[-1][0];measured=f.robot.data.joint_pos[0]
        mask=np.ones(43,dtype=bool);mask[list(f.sink.arm_ids)]=False
        assert_equal(actual[list(f.sink.arm_ids)],f.receipt.q14)
        assert_equal(actual[mask],measured[mask])
        assert f.controller.lowcmd_publisher.write_attempts==0
        assert result['cdr_bytes']==1004 and not result['DDS_transport_started']
        assert not result['hardware_authorized']
        assert f.sink.expire_before_physics_step()
        assert f.sink.last_physics_interlock['accepted']
        assert f.sink.last_physics_interlock['checked_at']<f.sink.last_physics_interlock['valid_until']
        assert f.sink.last_physics_interlock['receipt_token']==f.receipt.token
        return
    if name in ('raw_setter','clip','home','close','cancel'):
        if name=='raw_setter':
            try:f.controller.ctrl_dual_arm(np.ones(14),np.ones(14))
            except ValueError:pass
            else:raise AssertionError('raw setter allowed')
        if name=='clip':
            try:f.controller.clip_arm_q_target(np.ones(14),20)
            except ValueError:pass
            else:raise AssertionError('downstream clip allowed')
        if name=='home':f.controller.ctrl_dual_arm_go_home()
        if name=='cancel':f.controller.cancel()
        if name=='close':
            f.controller.close();count=len(f.target_calls);f.controller.close()
            assert len(f.target_calls)==count
        assert not f.publish()['applied']
        assert not f.session.clutched
        for target in f.target_calls:assert_equal(target,f.robot.data.joint_pos)
        assert f.controller.lowcmd_publisher.write_attempts==0
        return
    if name in ('short_lease','short_lease_admitted'):
        short=f.now+(.025 if name=='short_lease' else .09)
        def encode(q):
            f.clock=f.now+.006;f.last['ui']=race_ui(f,expires_at=short,assist_held=True)
            return original(q)
        codec.encode_arm_targets=encode
        result=f.publish()
        if name=='short_lease':
            # The original 19-ms remaining lease is now refused BEFORE the
            # target setter. Valid intent survives only as measured HOLD.
            assert not result['applied'] and not result['write_attempted'],result
            assert result['reason']=='execution_budget_replan_measured_hold'
            assert result['execution_budget']['receipt_valid_until']==short<f.receipt.valid_until
            assert not result['execution_budget']['admitted'] and f.session.clutched
            assert f.sink._last_deadline is None and f.sink._last_receipt is None
            assert len(f.target_calls)==1
            assert_equal(f.target_calls[-1],f.robot.data.joint_pos)
            return
        assert result['applied'],result
        assert result['applied_valid_until']==short<f.receipt.valid_until
        f.clock=short
        assert not f.sink.expire_before_physics_step()
        assert not f.sink.last_physics_interlock['accepted']
        assert f.sink.last_physics_interlock['valid_until']==short
        assert_equal(f.target_calls[-1],f.robot.data.joint_pos)
        return
    if name=='final_deadline':
        # Preserve the original final-write expiry race: first pass admission
        # with 84 ms, then reach the unchanged private deadline at write.
        short=f.now+.09;f.last['ui']=race_ui(f,expires_at=short,assist_held=True)
        f.clock=f.now+.006
        times=[]
        def clock():
            times.append(1)
            if len(times)==2:f.clock=short
            return f.clock
        f.sink.clock=clock
        result=f.publish();assert not result['applied'] and not result['write_attempted'],result
        assert result['reason']=='private_receipt_expired_at_original_sim_write',result
        return
    if name=='watchdog_release':
        assert f.publish()['applied']
        f.last['ui']=race_ui(f,release=True)
        f.clock=f.now+.006
        assert not f.sink.expire_before_physics_step()
        assert_equal(f.target_calls[-1],f.robot.data.joint_pos)
        return
    if name=='idle_preserves_clutch_waiting':
        f.step(release=True)
        pending=f.step(clutch=True)
        assert pending['snapshot']['clutched'] and not pending['allowed']
        before=deepcopy(f.session.status)
        f.controller.idle_hold()
        assert not f.sink.expire_before_physics_step()
        assert f.session.status==before and f.session.clutched
        assert f.sink.last_physics_interlock['reason']=='idle_measured_hold_without_session_revocation'
        assert f.sink._last_receipt is None and f.sink._last_deadline is None
        for target in f.target_calls:assert_equal(target,f.robot.data.joint_pos)
        return
    if name in ('watchdog_never_renews','watchdog_target','watchdog_session_cancel','watchdog_new_step'):
        assert f.publish()['applied']
        if name=='watchdog_never_renews':
            for number in range(1,4):
                f.clock=f.now+.01*number
                f.last['ui']=dict(f.last['ui'],revision=f.revision+number,capture_timestamp=f.clock)
                assert f.sink.expire_before_physics_step()
                assert f.sink.last_physics_interlock['valid_until']==f.receipt.valid_until
            f.clock=f.receipt.valid_until
        if name=='watchdog_target':f.target_check=lambda:dict(accepted=False)
        if name=='watchdog_session_cancel':f.session.hold('explicit_test_cancel',now=f.clock,clock_id='mono')
        if name=='watchdog_new_step':f.step(assist_held=True)
        assert not f.sink.expire_before_physics_step()
        assert_equal(f.target_calls[-1],f.robot.data.joint_pos)
        assert not f.session.clutched
        return
    if name in ('finger_during_tensor','generation_during_tensor'):
        construct=Tensor.new_tensor
        def mutate(self,values):
            if name=='finger_during_tensor':
                f.robot.data.joint_pos[0,f.robot.joint_names.index('right_hand_index_0_joint')]+=.01
            else:f.generation+=1
            return construct(self,values)
        Tensor.new_tensor=mutate
        result=f.publish()
        assert not result['applied'] and not result['write_attempted'],result
        assert_equal(f.target_calls[-1],f.robot.data.joint_pos)
        assert not f.session.clutched
        return
    if name=='forged':
        result=f.publish(replace(f.receipt,q14=tuple([0.]*14)))
    elif name=='replay':
        assert f.publish()['applied'];result=f.publish()
    elif name=='mode1':
        codec.manifest['motor_mode']=1;result=f.publish()
    elif name=='write_failure':
        f.fail_write=True;result=f.publish()
        assert result['delivery_unknown'] and result['write_attempted']
    else:
        def encode(q):
            f.clock=f.now+.006
            if name=='release':f.last['ui']=race_ui(f,release=True)
            if name=='new_owner':f.last['ui']=race_ui(f,client_id='other')
            if name=='source_epoch':f.last['packet']['source_instance_id']='other'
            if name=='assist_button':f.last['ui']=race_ui(f,assist_held=False)
            if name=='expired':f.clock=f.receipt.valid_until
            if name=='finger':f.robot.data.joint_pos[0,f.robot.joint_names.index('right_hand_index_0_joint')]+=.01
            if name=='body':f.robot.data.body_link_pose_w[0,0,0]+=.01
            if name=='scene':f.sim_scene['obstacle'][0,3]+=.01
            if name=='generation':f.generation+=1
            if name=='cancel_during_codec':f.session.hold('test_cancel',now=f.clock,clock_id='mono')
            if name=='target':f.target_check=lambda:dict(accepted=False)
            payload=original(q)
            if name=='crc':payload=payload[:-1]+bytes([payload[-1]^1])
            return payload
        codec.encode_arm_targets=encode
        # This target closure observes replacements occurring during encoding.
        result=f.controller.publish_receipt(f.receipt,read_inputs=f.read_inputs,
            recheck_target=lambda:f.target_check())
    assert not result['applied'],result
    if name!='write_failure':
        assert not result['write_attempted']
        assert_equal(f.target_calls[-1],f.robot.data.joint_pos)
    if name=='assist_button':
        assert f.session.clutched and result['receipt_replan'] is True
    else:
        assert not f.session.clutched


@pytest.mark.parametrize('name',[
    'constructor','raw_setter','clip','home','close','cancel','short_lease','short_lease_admitted','final_deadline',
    'watchdog_release','idle_preserves_clutch_waiting','forged','replay','mode1','write_failure','release','new_owner',
    'source_epoch','assist_button','expired','finger','body','scene','generation',
    'cancel_during_codec','target','crc','watchdog_never_renews','watchdog_target',
    'watchdog_session_cancel','watchdog_new_step','finger_during_tensor','generation_during_tensor'])
def test_original_constructor_receipt_and_real_CDR_contracts(name):
    child("sys.path.insert(0,'tests')\nfrom test_copilot_original_xr_sender import performed_case\n"
          f"performed_case({name!r},codec)")


def performed_ui_replan_case(action, substep, codec):
    """substep=None is before target write; 0..4 is each actual interlock slot."""
    from test_copilot_receipt_replan import changed_ui
    f=rig(codec,assisting=True)
    prior=f.session._inflight_receipt['prior_committed_offset'].copy()
    offered=f.session._committed_offset.copy()
    revision=f.session._ui_revision
    if substep is None:
        f.last['ui']=changed_ui(f,action);f.clock=f.now+.006
        result=f.publish()
        assert not result['applied'] and not result['write_attempted'] and result['receipt_replan'],result
        assert len(f.target_calls)==1 and f.session.clutched
        assert_equal(f.session._committed_offset,prior)
    else:
        assert f.publish()['applied']
        assert f.session._inflight_receipt['installed']
        original_deadline=f.receipt.valid_until
        for index in range(5):
            f.clock=f.now+.006+index*.005
            if index==substep:f.last['ui']=changed_ui(f,action)
            accepted=f.sink.expire_before_physics_step()
            assert accepted is (index<substep),(index,substep,f.sink.last_physics_interlock)
            if index==substep:
                assert f.sink.last_physics_interlock['receipt_replan']
                assert f.sink.last_physics_interlock['valid_until']<=original_deadline
            assert f.session.clutched
        assert_equal(f.session._committed_offset,offered)
        assert len(f.target_calls)==1+(5-substep)
    assert_equal(f.target_calls[-1],f.robot.data.joint_pos)
    assert f.session._ui_revision==revision  # No hidden edge ACK.
    assert f.session._inflight_receipt is None and f.session._receipt_replan is None
    assert f.sink._last_receipt is None and f.sink._last_deadline is None
    # No q14 survives for the rest of this physics window.
    assert not f.sink.expire_before_physics_step() and f.session.clutched
    assert_equal(f.target_calls[-1],f.robot.data.joint_pos)
    assert not f.publish()['applied'] and not f.session.clutched  # Replay is a fault.


def test_valid_UI_replans_before_write_and_each_of_five_PhysX_interlocks():
    # One existing audit-guarded CPU worker: no SDK channel/socket/Isaac/GPU.
    child("sys.path.insert(0,'tests')\nfrom test_copilot_original_xr_sender import performed_ui_replan_case\n"
          "for action in ('clutch','select','confirm','assist_release'):\n"
          " for substep in (None,0,1,2,3,4):\n"
          "  performed_ui_replan_case(action,substep,codec)")


def test_modes_and_nonarm_policy_reject_before_owning_any_robot():
    from teleop_assist.copilot.original_xr_sender import OriginalSimReceiptSink
    for changes in [dict(simulation_mode=False),dict(simulation_mode='sim'),
                    dict(simulation_mode=True,motion_mode=True),dict(simulation_mode=True,nonarm_policy='mode1')]:
        with pytest.raises(ValueError):OriginalSimReceiptSink(None,None,**changes)


def test_class_modes_source_pin_and_exclusive_private_session():
    child(r'''
sys.path.insert(0,'tests')
from test_copilot_original_xr_sender import rig,ORIGINAL
from teleop_assist.copilot.original_xr_sender import OriginalSimReceiptSink,guarded_original_controller_class
import tempfile
f=rig(codec)
for kwargs in ({},{'simulation_mode':False},{'simulation_mode':True,'motion_mode':True},{'simulation_mode':1}):
 assert rejected(lambda:type(f.controller)(**kwargs))
assert not f.target_calls
assert rejected(lambda:OriginalSimReceiptSink(f.session,f.source,simulation_mode=True,codec=codec))
with tempfile.TemporaryDirectory() as d:
 p=__import__('pathlib').Path(d)/'robot_arm.py';p.write_bytes(ORIGINAL.read_bytes()+b'\n')
 assert rejected(lambda:guarded_original_controller_class(p,f.sink))
try:f.controller.lowcmd_publisher.Write(f.controller.msg)
except RuntimeError as error:assert str(error)=='original_unchecked_LowCmd_publication_forbidden'
else:raise AssertionError('legacy publication allowed')
assert f.controller.lowcmd_publisher.write_attempts==1 and not f.target_calls
try:type(f.controller).__bases__[0]._ctrl_motor_state(f.controller)
except RuntimeError as error:assert str(error)=='original_unchecked_message_CRC_forbidden'
else:raise AssertionError('original repeated loop bypass allowed')
assert f.controller.lowcmd_publisher.write_attempts==1 and not f.target_calls
assert f.controller.original_manifest['publisher_override_installed_before_constructor']
assert not f.controller.original_manifest['complete_XR_entry_integration_verified']
''')


def test_snapshot_thread_and_articulation_ownership_also_gate_hold_paths():
    child(r'''
sys.path.insert(0,'tests')
from test_copilot_original_xr_sender import rig
from teleop_assist.copilot.original_xr_sender import OriginalSimReceiptSink
from teleop_assist.copilot.sim_control import SimControlSession
import threading
f=rig(codec);errors=[]
other=SimControlSession(f.worker,clock=lambda:f.clock)
def foreign_thread():
 try:
  sink=OriginalSimReceiptSink(other,f.source,simulation_mode=True,codec=codec,clock=lambda:f.clock)
  sink.idle_hold()
 except ValueError as error:errors.append(str(error))
t=threading.Thread(target=foreign_thread);t.start();t.join(2)
assert not t.is_alive() and errors==['single_simulator_owner_thread_required']
assert not f.target_calls
f.sim_scene['robot']=object()
assert rejected(f.sink.idle_hold)=='simulator_articulation_instance_changed'
assert not f.target_calls
''')
