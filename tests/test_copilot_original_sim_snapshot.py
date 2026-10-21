"""Pure CPU mock-scene contract tests; these are NOT original Isaac integration."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import hashlib
import json
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.copilot.original_sim_snapshot import (
    OriginalSimSnapshotAdapter, G1_JOINT_NAMES, G1_BODY_NAMES, ARM_JOINT_NAMES)
from teleop_assist.copilot.xr_send import snapshot_digest


@pytest.fixture
def setup(tmp_path):
    """Actual-property-shaped doubles with deliberately distinct named fingers."""
    asset=tmp_path/'mock_asset.usd';asset.write_bytes(b'mock asset, not actual USD physics')
    joints=sorted(G1_JOINT_NAMES,reverse=True);bodies=sorted(G1_BODY_NAMES)
    poses=np.tile(np.array([0.,0.,.75,1.,0.,0.,0.]),(1,55,1))
    data=NS(joint_names=joints,body_names=bodies,
        joint_pos=np.arange(43,dtype=float)[None,:]/100,
        joint_vel=np.zeros((1,43)),root_pose_w=poses[:,0,:].copy(),body_link_pose_w=poses)
    robot=NS(data=data,cfg=NS(spawn=NS(usd_path=str(asset))))
    owner=NS(scene={'robot':robot},clock=10.,generation=42,reader_calls=0,change=None,
        geometry=dict(scene_id='mock_known_geometry',geometry_complete=True,static=True,
            source='current_simulator_complete_collision_scene',
            obstacles=[dict(id='target',type='cylinder',radius=.02,length=.08,world_pose=np.eye(4).tolist()),
                       dict(id='ground',type='plane',normal=[0,0,1],offset=0.)],
            observed_target_velocity=[0.]*6,
            allowed_static_contacts=[dict(link='left_ankle_roll_link',obstacle='ground',max_penetration_m=.002)]))
    def scene_reader(scene,context):
        assert scene is owner.scene
        owner.reader_calls+=1
        value=deepcopy(owner.geometry);value.update(asdict(context))
        if owner.change:owner.change(value,context)
        return value
    owner.options=dict(read_generation=lambda:owner.generation,clock_id='test_monotonic',
        source_instance_id='test_sim_instance',calibration_id='test_calibration',
        robot_asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        collision_scene_reader=scene_reader,simulation_mode=True,clock=lambda:owner.clock)
    owner.adapter=OriginalSimSnapshotAdapter(owner.scene,**owner.options)
    owner.data=data
    return owner


def test_complete_named_state_keeps_finger_mapping_and_matches_existing_digest(setup):
    f=setup;s=f.adapter.capture();state=s.robot_state
    assert len(state['joint_positions'])==43 and len(state['observed_body_world_poses'])==55
    for name in G1_JOINT_NAMES:
        assert state['joint_positions'][name]==f.data.joint_pos[0,f.data.joint_names.index(name)]
    assert state['joint_positions']['right_hand_middle_0_joint']!=state['joint_positions']['right_hand_index_0_joint']
    assert s.snapshot_sha256==snapshot_digest(state,s.scene)
    assert state['acquisition_timestamp']==s.scene['acquisition_timestamp']==10.
    assert state['physics_generation']==s.scene['physics_generation']==42
    assert state['robot_asset_sha256']==f.options['robot_asset_sha256']
    assert len(ARM_JOINT_NAMES)==14


def test_body_normalization_is_exactly_robot_request_convention(setup):
    f=setup
    q=Rotation.from_euler('xyz',[.2,-.3,.4]).as_quat()[[3,0,1,2]].astype(np.float32)
    index=f.data.body_names.index('right_wrist_yaw_link')
    f.data.body_link_pose_w[0,index,3:]=q
    s=f.adapter.capture()
    expected=Rotation.from_quat(q[[1,2,3,0]].astype(float)).as_matrix()
    np.testing.assert_array_equal(np.array(s.robot_state['observed_body_world_poses']['right_wrist_yaw_link'])[:3,:3],expected)
    tcp=np.eye(4);tcp[0,3]=.05
    np.testing.assert_array_equal(s.right_base_ee,np.linalg.inv(s.robot_state['base_in_world'])@
        np.asarray(s.robot_state['observed_body_world_poses']['right_wrist_yaw_link'])@tcp)


def test_same_generation_capture_and_reread_never_refresh_acquisition(setup):
    f=setup;first=f.adapter.capture();f.clock+=.05
    assert f.adapter.read_same()==first
    assert f.adapter.capture()==first
    assert first.acquisition_timestamp==10. and first.capture_finished_at==10.
    assert f.reader_calls==3
    f.generation+=1;second=f.adapter.capture()
    assert second.acquisition_timestamp==10.05 and second.sample_sequence==2
    assert second.physics_generation==43 and second.snapshot_sha256!=first.snapshot_sha256


def test_owner_assertion_permits_advanced_generation_without_acquisition_or_retiming(setup):
    f=setup
    assert f.adapter.assert_owner()==42 and f.reader_calls==0
    first=f.adapter.capture()
    before=(f.adapter._active,f.adapter._last_time,f.adapter._sequence,f.reader_calls)
    f.clock+=1.
    f.generation+=1
    assert f.adapter.assert_owner()==43
    assert (f.adapter._active,f.adapter._last_time,f.adapter._sequence,f.reader_calls)==before
    assert first.acquisition_timestamp==10.
    with pytest.raises(ValueError):f.adapter.read_same()


def test_owner_assertion_rejects_foreign_thread_before_any_sample_read(setup):
    f=setup;errors=[]
    def foreign_owner():
        try:f.adapter.assert_owner()
        except ValueError as error:errors.append(str(error))
    thread=threading.Thread(target=foreign_owner);thread.start();thread.join(timeout=2)
    assert not thread.is_alive() and errors==['single_simulator_owner_thread_required']
    assert f.reader_calls==0 and f.adapter._active is None and f.adapter._last_time is None


def test_owner_assertion_rejects_replaced_articulation_without_acquisition(setup):
    f=setup;f.scene['robot']=deepcopy(f.scene['robot'])
    with pytest.raises(ValueError,match='simulator_articulation_instance_changed'):
        f.adapter.assert_owner()
    assert f.reader_calls==0 and f.adapter._active is None


def test_caller_mutation_and_even_frozen_dataclass_override_cannot_change_private_sample(setup):
    f=setup;s=f.adapter.capture();digest=s.snapshot_sha256
    state=s.robot_state;state['joint_positions'][next(iter(G1_JOINT_NAMES))]=99
    scene=s.scene;scene['obstacles'][0]['radius']=99
    with pytest.raises(FrozenInstanceError):s.acquisition_timestamp=99
    object.__setattr__(s,'robot_state_json','{}')
    assert f.adapter.read_same().snapshot_sha256==digest
    assert len(f.adapter.read_same().robot_state['joint_positions'])==43


@pytest.mark.parametrize('kind',['finger','arm','dq','root','body','object_pose','object_velocity','size','contact','body_order','joint_order','asset_path','robot_instance','data_instance'])
def test_current_state_or_scene_change_revokes_private_transaction(setup,kind,tmp_path):
    f=setup;f.adapter.capture()
    if kind=='finger':f.data.joint_pos[0,f.data.joint_names.index('right_hand_index_0_joint')]+=.001
    if kind=='arm':f.data.joint_pos[0,f.data.joint_names.index('left_elbow_joint')]+=.001
    if kind=='dq':f.data.joint_vel[0,0]=.001
    if kind=='root':
        f.data.root_pose_w[0,0]=.001
        f.data.body_link_pose_w[0,f.data.body_names.index('pelvis'),0]=.001
    if kind=='body':f.data.body_link_pose_w[0,1,0]=.001
    if kind=='object_pose':f.geometry['obstacles'][0]['world_pose'][0][3]+=.001
    if kind=='object_velocity':f.geometry['observed_target_velocity'][0]=.001
    if kind=='size':f.geometry['obstacles'][0]['radius']+=.001
    if kind=='contact':f.geometry['allowed_static_contacts'][0]['max_penetration_m']=.001
    if kind=='body_order':f.data.body_names.reverse();f.data.body_link_pose_w=f.data.body_link_pose_w[:,::-1,:]
    if kind=='joint_order':f.data.joint_names.reverse();f.data.joint_pos=f.data.joint_pos[:,::-1];f.data.joint_vel=f.data.joint_vel[:,::-1]
    if kind=='asset_path':
        other=tmp_path/'other.usd';other.write_bytes(b'different');f.scene['robot'].cfg.spawn.usd_path=str(other)
    if kind=='robot_instance':f.scene['robot']=deepcopy(f.scene['robot'])
    if kind=='data_instance':f.scene['robot'].data=deepcopy(f.scene['robot'].data)
    with pytest.raises(ValueError):f.adapter.read_same()
    with pytest.raises(ValueError,match='no_active'):f.adapter.read_same()


@pytest.mark.parametrize('kind',['missing_joint','duplicate_joint','unknown_joint','missing_body','duplicate_body','unknown_body','nan_q','bool_q','wrong_batch','nan_dq','zero_quaternion','unnormalized_quaternion','bad_root','root_pelvis_mismatch'])
def test_incomplete_or_invalid_actual_properties_are_not_authoritative(setup,kind):
    f=setup;d=f.data
    if kind=='missing_joint':d.joint_names.pop()
    if kind=='duplicate_joint':d.joint_names[0]=d.joint_names[1]
    if kind=='unknown_joint':d.joint_names[0]='invented_joint'
    if kind=='missing_body':d.body_names.pop()
    if kind=='duplicate_body':d.body_names[0]=d.body_names[1]
    if kind=='unknown_body':d.body_names[0]='invented_body'
    if kind=='nan_q':d.joint_pos[0,0]=float('nan')
    if kind=='bool_q':d.joint_pos=np.zeros((1,43),dtype=bool)
    if kind=='wrong_batch':d.joint_pos=np.zeros((2,43))
    if kind=='nan_dq':d.joint_vel[0,0]=float('nan')
    if kind=='zero_quaternion':d.body_link_pose_w[0,0,3:]=0
    if kind=='unnormalized_quaternion':d.body_link_pose_w[0,0,3:]*=2
    if kind=='bad_root':d.root_pose_w=np.eye(4)[None,:,:]
    if kind=='root_pelvis_mismatch':d.root_pose_w[0,0]=.01
    with pytest.raises(ValueError):f.adapter.capture()


@pytest.mark.parametrize('kind',['clock','capture_time','source','generation','calibration','boolean_generation','boolean_time','incomplete','dynamic','unknown_source','empty','duplicate_obstacle','bad_pose','bad_radius','bad_type','bad_plane','unknown_contact','nonplane_contact','excess_penetration','nonfinite','missing_velocity','moving_target','rotating_target'])
def test_scene_reader_cannot_supply_wrong_context_or_incomplete_geometry(setup,kind):
    f=setup
    def corrupt(s,c):
        if kind=='clock':s['clock_id']='other'
        if kind=='capture_time':s['acquisition_timestamp']-=.001
        if kind=='source':s['source_instance_id']='old_instance'
        if kind=='generation':s['physics_generation']-=1
        if kind=='calibration':s['calibration_id']='other'
        if kind=='boolean_generation':s['physics_generation']=True
        if kind=='boolean_time':s['acquisition_timestamp']=True
        if kind=='incomplete':s['geometry_complete']=False
        if kind=='dynamic':s['static']=False
        if kind=='unknown_source':s['source']='cached_DDS_packet'
        if kind=='empty':s['obstacles']=[]
        if kind=='duplicate_obstacle':s['obstacles'].append(deepcopy(s['obstacles'][0]))
        if kind=='bad_pose':s['obstacles'][0]['world_pose'][0][0]=2
        if kind=='bad_radius':s['obstacles'][0]['radius']=-1
        if kind=='bad_type':s['obstacles'][0]['type']='unknown_mesh'
        if kind=='bad_plane':s['obstacles'][1]['normal']=[0,1,0]
        if kind=='unknown_contact':s['allowed_static_contacts'][0]['link']='right_hand_palm_link'
        if kind=='nonplane_contact':s['allowed_static_contacts'][0]['obstacle']='target'
        if kind=='excess_penetration':s['allowed_static_contacts'][0]['max_penetration_m']=.1
        if kind=='nonfinite':s['extra_value']=float('nan')
        if kind=='missing_velocity':s.pop('observed_target_velocity')
        if kind=='moving_target':s['observed_target_velocity'][0]=.011
        if kind=='rotating_target':s['observed_target_velocity'][3]=.021
    f.change=corrupt
    with pytest.raises(ValueError):f.adapter.capture()


@pytest.mark.parametrize('phase',['capture','reread'])
@pytest.mark.parametrize('kind',['generation','body_mutation','clock_stall','clock_regression'])
def test_mid_read_races_or_elapsed_work_revoke_and_do_not_retime(setup,phase,kind):
    f=setup
    if phase=='reread':f.adapter.capture()
    def disturb(scene,context):
        if kind=='generation':f.generation+=1
        if kind=='body_mutation':f.data.joint_pos[0,0]+=.001
        if kind=='clock_stall':f.clock+=.15
        if kind=='clock_regression':f.clock-=.01
    f.change=disturb
    with pytest.raises(ValueError):getattr(f.adapter,'capture' if phase=='capture' else 'read_same')()
    f.change=None
    with pytest.raises(ValueError,match='no_active'):f.adapter.read_same()
    if kind not in ('generation','clock_regression'):
        with pytest.raises(ValueError,match='already_captured'):f.adapter.capture()


def test_expired_sample_cannot_be_refreshed_by_republication_or_invalidate(setup):
    f=setup;f.adapter.capture();f.clock+=.151
    with pytest.raises(ValueError,match='expired'):f.adapter.capture()
    f.adapter.invalidate()
    with pytest.raises(ValueError,match='already_captured'):f.adapter.capture()
    f.generation+=1
    assert f.adapter.capture().acquisition_timestamp==f.clock


@pytest.mark.parametrize('value',[True,-1,1.2,None,'42'])
def test_invalid_generations_rejected(setup,value):
    f=setup;f.generation=value
    with pytest.raises(ValueError,match='generation_invalid'):f.adapter.capture()


def test_generation_regression_requires_new_source_instance(setup):
    f=setup;f.adapter.capture();f.generation=0
    with pytest.raises(ValueError,match='regressed'):f.adapter.capture()


def test_other_thread_cannot_capture_or_recheck_simulation(setup):
    f=setup;f.adapter.capture();errors=[]
    def attempt():
        try:f.adapter.read_same()
        except ValueError as e:errors.append(str(e))
    t=threading.Thread(target=attempt);t.start();t.join(timeout=2)
    assert not t.is_alive() and errors==['single_simulator_owner_thread_required']


@pytest.mark.parametrize('change',[dict(simulation_mode=False),dict(simulation_mode=1),dict(clock_id=''),
    dict(source_instance_id=None),dict(calibration_id=''),dict(robot_asset_sha256='b'*64),
    dict(robot_asset_sha256='nohash'),dict(collision_scene_reader=None),dict(read_generation=None)])
def test_constructor_requires_bound_local_asset_and_explicit_owned_readers(setup,change):
    f=setup
    with pytest.raises(ValueError):OriginalSimSnapshotAdapter(f.scene,**dict(f.options,**change))


def test_configured_asset_url_never_triggers_remote_loading(setup):
    f=setup;f.scene['robot'].cfg.spawn.usd_path='https://example.invalid/robot.usd'
    with pytest.raises(ValueError,match='local_configured'):OriginalSimSnapshotAdapter(f.scene,**f.options)
