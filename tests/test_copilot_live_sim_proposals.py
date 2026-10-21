"""Evidence-binding tests with explicit fake worker output; no physics claims."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from teleop_assist.copilot.live_sim_proposals import LiveSimProposals
from teleop_assist.copilot.sim_control import G1_JOINT_NAMES


def pose(x=0., y=0., z=0.):
    result=np.eye(4);result[:3,3]=[x,y,z]
    return result.tolist()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


class Fixture:
    def __init__(self):
        self.gate=LiveSimProposals(descriptor_sha256='a'*64)
        self.now=10.12;self.index=2
        self.preview=dict(target_id='learned_1',observation_id='cal:0',capture_timestamp=self.now,
            model_source_capture_timestamp=10.,model_source_observation_id='cal:0',
            validation_observation_id='cal:2',clock_id='mono',calibration_id='cal',descriptor_sha256='a'*64,
            motion_static=True,valid=False,validation_reason='preview_only_actual_command_path_not_validated',
            observed_centroid_world=[.4,-.18,.9],candidates=[dict(candidate_id='grasp1',
                base_ee=pose(.4,-.18,.9),model_score=.9,geometry_valid=True,ik_valid=True,collision_valid=False,
                source='explicit_fake_model_fixture',observed_hand_collision={'pregrasp_collision_free':True})])
        for i,t in enumerate((10.,10.06,10.12)):
            assert self.gate.record_frame(self.frame(i,t),self.scene(t))['recorded']

    def frame(self,index=None,stamp=None):
        return dict(frame_id=self.index if index is None else index,timestamp=self.now if stamp is None else stamp,
            clock_id='mono',calibration_id='cal',T_world_base=np.eye(4))

    def scene(self,stamp=None):
        return dict(source='actual_simulator_fixture_contract',scene_id='known_sim_scene',
            acquisition_timestamp=self.now if stamp is None else stamp,clock_id='mono',
            geometry_complete=True,static=True,observed_target_velocity=[0.]*6,obstacles=[
                dict(id='table',type='box',size_xyz=[.65,.8,.04],world_pose=pose(.48,0,.84)),
                dict(id='target',type='cylinder',radius=.02,length=.08,world_pose=pose(.4,-.18,.9))])

    def state(self):
        return dict(source='actual_simulator_fixture_contract',complete=True,acquisition_timestamp=self.now,
            clock_id='mono',base_in_world=pose(),joint_positions={name:0. for name in G1_JOINT_NAMES},
            observed_body_world_poses={'left_wrist_yaw_link':pose(.25,.2,.9),'right_wrist_yaw_link':pose(.35,-.18,.9)})

    def inspect(self,**kwargs):
        return self.gate.inspect(kwargs.pop('preview',self.preview),selected_candidate_id=kwargs.pop('selected','grasp1'),
            confirmed_candidate_id=kwargs.pop('confirmed',None),now=kwargs.pop('now',self.now),
            clock_id=kwargs.pop('clock','mono'),calibration_id=kwargs.pop('calibration','cal'),**kwargs)

    def endpoint(self):
        state,scene=self.state(),self.scene()
        final=dict(solver_success=True,limits_valid=True)
        for side in ('left','right'):
            final[side+'_position_error_m']=.001;final[side+'_orientation_error_rad']=.01
        return dict(accepted=True,scope='sim_candidate_endpoint_only',motion_authorized=False,
            ik=dict(ik_valid=True,right_target_base_ee=deepcopy(self.preview['candidates'][0]['base_ee']),
                left_target_base_ee=pose(.3,.2,.9),q_initial=[0.]*14,q_solution=[.01]*14,final=final),
            endpoint=dict(collision_valid=True),state_evidence=dict(scope='sim_candidate_endpoint_only',reason='evaluated',
                path_check_requested=False,live_execution_authorized=False,current_state_precheck={'collision_valid':True},
                urdf_vs_observed_frames={'valid':True},robot_state=state,scene=scene,
                snapshot_sha256=digest({'robot_state':state,'scene':scene})))

    def authorize(self,prepared,endpoint=None,**kwargs):
        return self.gate.authorize(prepared,self.endpoint() if endpoint is None else endpoint,
            robot_state=kwargs.pop('state',self.state()),scene=kwargs.pop('scene',self.scene()),
            now=kwargs.pop('now',self.now),clock_id=kwargs.pop('clock','mono'),**kwargs)

    def advance(self,dt=.06):
        self.now+=dt;self.index+=1
        return self.gate.record_frame(self.frame(),self.scene())

    def renew_preview(self):
        self.preview.update(capture_timestamp=self.now,validation_observation_id=f'cal:{self.index}')


def complete_sim_fixture():
    f=Fixture();original_scene=f.scene
    f.scene=lambda stamp=None:dict(original_scene(stamp),source='current_simulator_complete_collision_scene')
    f.gate=LiveSimProposals(descriptor_sha256='a'*64,policy='complete_static_sim_orientation')
    for i,t in enumerate((10.,10.06,10.12)):
        assert f.gate.record_frame(f.frame(i,t),f.scene(t))['recorded']
    f.preview['candidates'][0]['observed_hand_collision']=dict(status='observed_collision_rejected',
        hand_collision_free=False,pregrasp_collision_free=False,collision_valid=False,
        scene_coverage_unknown=True,object_point_count=100,scene_point_count=800,
        final=dict(forbidden_object_points=7,scene_collision_points=23,allowed_target_pad_contacts=2))
    return f


def test_complete_sim_orientation_policy_only_changes_preview_eligibility():
    f=complete_sim_fixture();original=deepcopy(f.preview)
    prepared=f.inspect();assert prepared.ready
    assert not f.authorize(prepared,endpoint={'accepted':True}).accepted
    prepared=f.inspect();result=f.authorize(prepared)
    assert result.accepted and result.batch.candidates[0].collision_valid
    assert not result.evidence['motion_authorized'] and not result.evidence['hardware_authorized']
    assert result.evidence['requires_actual_command_guard']
    assert result.evidence['preview_collision_policy']=='complete_static_sim_orientation'
    assert f.preview==original  # No observed flag was changed to manufacture clearance.


def cached_fk_endpoint(f):
    endpoint=f.endpoint();ik=endpoint['ik']
    ik.update(reason='cached_endpoint_FK',validation_method='independent_cached_endpoint_full_q43_FK',
        scope='numerical_candidate_endpoint_full_q43_FK_only',solver_invoked=False,
        original_solver_invoked=False,cache_hit=True,cache_revalidated_this_call=True,
        cache_authority=False,motion_authorized=False,fixed_nonarm_joints_unchanged=True,limits_valid=True)
    ik['final'].update(solver_status='validated_cached_endpoint_FK',solver_success=None)
    return endpoint


def test_fresh_endpoint_fk_hypothesis_still_needs_full_current_evidence():
    f=Fixture();endpoint=cached_fk_endpoint(f);prepared=f.inspect()
    result=f.authorize(prepared,endpoint)
    assert result.accepted and not result.evidence['motion_authorized']
    endpoint['state_evidence']['snapshot_sha256']='wrong'
    assert not f.authorize(f.inspect(),endpoint).accepted


@pytest.mark.parametrize('key,value',[
    ('cache_revalidated_this_call',False),('cache_authority',True),('motion_authorized',True),
    ('cache_hit',False),('fixed_nonarm_joints_unchanged',False),('solver_invoked',True),
    ('validation_method','old_cached_evidence'),('scope','actual_command')])
def test_cache_flags_cannot_replace_fresh_full_fk_contract(key,value):
    f=Fixture();endpoint=cached_fk_endpoint(f);endpoint['ik'][key]=value
    assert not f.authorize(f.inspect(),endpoint).accepted


@pytest.mark.parametrize('fault',['fk','collision','stale_state'])
def test_cached_endpoint_does_not_bypass_existing_residual_collision_or_freshness_limits(fault):
    f=Fixture();endpoint=cached_fk_endpoint(f);kwargs={}
    if fault=='fk':endpoint['ik']['final']['right_position_error_m']=.011
    elif fault=='collision':endpoint['endpoint']['collision_valid']=False
    else:kwargs['now']=f.now+.151
    assert not f.authorize(f.inspect(),endpoint,**kwargs).accepted


@pytest.mark.parametrize('fault',['source','incomplete','not_static','target_missing','different_clock'])
def test_complete_sim_policy_rejects_untrusted_or_incomplete_recorded_scene(fault):
    f=complete_sim_fixture();f.now+=.03;f.index+=1;scene=f.scene()
    if fault=='source':scene['source']='real_camera_unknown_geometry'
    elif fault=='incomplete':scene['geometry_complete']=False
    elif fault=='not_static':scene['static']=False
    elif fault=='target_missing':scene['obstacles']=scene['obstacles'][:1]
    else:scene['clock_id']='another_clock'
    assert not f.gate.record_frame(f.frame(),scene)['recorded']
    f.renew_preview();assert not f.inspect().ready


@pytest.mark.parametrize('fault',['source','incomplete','not_static','new_hidden_obstacle',
    'current_collision','endpoint_collision','body_fk','snapshot_hash','state_joint_changed'])
def test_complete_sim_policy_still_requires_exact_current_full_scene_endpoint_evidence(fault):
    f=complete_sim_fixture();prepared=f.inspect();endpoint=f.endpoint();scene=f.scene()
    if fault=='source':scene['source']='real_camera_unknown_geometry'
    elif fault=='incomplete':scene['geometry_complete']=False
    elif fault=='not_static':scene['static']=False
    elif fault=='new_hidden_obstacle':
        scene['obstacles'].append(dict(id='hidden_from_camera',type='box',size_xyz=[.2,.2,.2],world_pose=pose(.4,-.18,.9)))
    elif fault=='current_collision':endpoint['state_evidence']['current_state_precheck']['collision_valid']=False
    elif fault=='endpoint_collision':endpoint['endpoint']['collision_valid']=False
    elif fault=='body_fk':endpoint['state_evidence']['urdf_vs_observed_frames']['valid']=False
    elif fault=='snapshot_hash':endpoint['state_evidence']['snapshot_sha256']='wrong'
    else:endpoint['state_evidence']['robot_state']['joint_positions']['right_hand_thumb_0_joint']=.01
    if fault in ('source','incomplete','not_static','new_hidden_obstacle'):
        # Even a result claiming acceptance for this newly matching snapshot
        # cannot override the gate's independently recorded scene contract.
        endpoint['state_evidence']['scene']=deepcopy(scene)
        endpoint['state_evidence']['snapshot_sha256']=digest({'robot_state':f.state(),'scene':scene})
    result=f.authorize(prepared,endpoint,scene=scene)
    assert not result.accepted and result.batch is None


@pytest.mark.parametrize('policy',[None,'ignore_collisions','real_robot'])
def test_collision_policy_requires_an_explicit_known_mode(policy):
    with pytest.raises(ValueError):LiveSimProposals(descriptor_sha256='a'*64,policy=policy)


def test_only_selected_candidate_becomes_typed_valid_after_separate_endpoint_evidence():
    f=Fixture();preview=deepcopy(f.preview)
    second=deepcopy(f.preview['candidates'][0]);second['candidate_id']='unselected';f.preview['candidates'].append(second)
    prepared=f.inspect();assert prepared.ready
    assert not f.authorize(prepared,endpoint={'accepted':True}).accepted
    prepared=f.inspect();result=f.authorize(prepared)
    assert result.accepted and len(result.batch.candidates)==1 and result.batch.valid
    assert result.batch.candidates[0].candidate_id=='grasp1' and result.batch.candidates[0].collision_valid
    assert f.preview['valid'] is False and not f.preview['candidates'][0]['collision_valid']
    assert result.batch.observation_id=='cal:2' and result.batch.capture_timestamp==10.12
    assert result.batch.model_source_id==prepared.model_source_id
    assert result.batch.model_source_capture_timestamp==10.
    assert result.batch.model_source_clock_id=='mono'
    assert result.evidence['motion_authorized'] is False and result.evidence['requires_actual_command_guard']
    assert f.preview['candidates'][0]==preview['candidates'][0]


def test_fresh_validation_never_pretends_the_frozen_model_reran():
    f=Fixture();first=f.inspect();assert f.authorize(first).accepted
    for _ in range(20):assert f.advance(.10)['recorded']
    f.renew_preview();renewed=f.inspect(confirmed='grasp1');result=f.authorize(renewed)
    assert renewed.ready and result.accepted
    assert result.batch.model_source_id==first.model_source_id
    assert result.batch.model_source_capture_timestamp==10.
    assert result.batch.capture_timestamp==f.now and result.batch.observation_id==f'cal:{f.index}'
    assert result.evidence['model_source_observation_id']=='cal:0'


def test_frozen_anchor_survives_bounded_history_pruning_without_retiming():
    f=Fixture();first=f.inspect();assert f.authorize(first).accepted
    for _ in range(1030):assert f.advance(.03)['recorded']
    assert len(f.gate.frames)==1024 and 'cal:0' not in f.gate.frames
    f.renew_preview();checked=f.inspect(confirmed='grasp1');result=f.authorize(checked)
    assert checked.ready and result.accepted
    assert result.batch.model_source_id==first.model_source_id
    assert result.batch.model_source_capture_timestamp==10.
    assert result.batch.capture_timestamp==f.now
    assert result.evidence['model_source_observation_id']=='cal:0'


@pytest.mark.parametrize('fault',['motion','gap','invalid_record'])
def test_failure_remains_latched_after_its_frame_is_pruned(fault):
    f=Fixture();assert f.inspect().ready
    f.now+=.30 if fault=='gap' else .03;f.index+=1
    frame,scene=f.frame(),f.scene()
    if fault=='motion':scene['obstacles'][1]['world_pose'][0][3]+=.01
    elif fault=='invalid_record':scene['observed_target_velocity'][0]=.1
    f.gate.record_frame(frame,scene)
    for _ in range(1030):f.advance(.03)
    f.renew_preview()
    assert 'cal:3' not in f.gate.frames
    assert not f.inspect().ready


def test_pruned_unselected_anchor_cannot_be_invented_at_first_selection():
    f=Fixture()
    for _ in range(1030):f.advance(.03)
    f.renew_preview()
    assert f.inspect().reason=='model_source_frame_was_not_recorded_by_this_simulator'


def test_frozen_validation_cannot_regress_before_original_stationarity_proof():
    f=Fixture();assert f.inspect().ready
    f.preview.update(validation_observation_id='cal:1',capture_timestamp=10.06)
    assert f.inspect().reason=='validation_predates_bound_stationarity_evidence'


def test_frozen_inspection_does_not_rescan_previously_checked_history(monkeypatch):
    from teleop_assist.copilot import live_sim_proposals as module
    f=Fixture();assert f.inspect().ready
    for _ in range(1030):f.advance(.03)
    f.renew_preview();calls=[];original=module._same_geometry
    def compare(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(module,'_same_geometry',compare)
    assert f.inspect().ready and calls==[]
    f.advance(.03)
    assert len(calls)==1  # Each new actual frame still checks the exact anchor.
    f.renew_preview();assert f.inspect().ready and len(calls)==1


def test_late_first_binding_checks_all_1024_frames_without_scalar_geometry_loop(monkeypatch):
    from teleop_assist.copilot import live_sim_proposals as module
    f=Fixture()
    while len(f.gate.frames)<1024:f.advance(.03)
    f.renew_preview()
    def redundant_scalar_loop(*_args,**_kwargs):
        raise AssertionError('Initial history must use acquisition-validated vector profiles')
    monkeypatch.setattr(module,'_same_geometry',redundant_scalar_loop)
    monkeypatch.setattr(module,'_pose_close',redundant_scalar_loop)
    checked=f.inspect()
    assert checked.ready and f.gate._frozen['anchor']['timestamp']==10.
    assert checked.validation_capture_timestamp==f.now


@pytest.mark.parametrize('fault',['object_position','object_rotation','base_position','base_rotation',
    'slow_cumulative_drift','object_parameter','allowed_contacts','scene_identity','gap','invalid_velocity'])
def test_first_binding_detects_historical_fault_even_after_return_to_original_scene(fault):
    from scipy.spatial.transform import Rotation
    f=Fixture()
    for i in range(3,1024):
        f.index=i;f.now+=.30 if fault=='gap' and i==500 else .03
        frame,scene=f.frame(),f.scene()
        if fault=='slow_cumulative_drift':
            # Every adjacent displacement is < 4 micrometres; only an exact
            # original-anchor comparison reveals the accumulated 1.5 mm.
            scene['obstacles'][1]['world_pose'][0][3]+=.0015*max(0.,1-abs(i-500)/500)
        if i==500:
            if fault=='object_position':scene['obstacles'][1]['world_pose'][0][3]+=.00101
            elif fault=='object_rotation':
                scene['obstacles'][1]['world_pose']=np.asarray(scene['obstacles'][1]['world_pose'])
                scene['obstacles'][1]['world_pose'][:3,:3]=Rotation.from_rotvec([0,0,.00501]).as_matrix()
                scene['obstacles'][1]['world_pose']=scene['obstacles'][1]['world_pose'].tolist()
            elif fault=='base_position':frame['T_world_base'][0,3]+=.00101
            elif fault=='base_rotation':frame['T_world_base'][:3,:3]=Rotation.from_rotvec([0,0,.00501]).as_matrix()
            elif fault=='object_parameter':scene['obstacles'][1]['radius']+=1e-9
            elif fault=='allowed_contacts':scene['allowed_static_contacts']=[['changed','contact']]
            elif fault=='scene_identity':scene['scene_id']='changed_scene'
            elif fault=='invalid_velocity':scene['observed_target_velocity'][0]=.01001
        f.gate.record_frame(frame,scene)
    f.renew_preview()
    assert not f.inspect().ready


def test_vector_geometry_profiles_match_scalar_reference_for_all_transforms_and_parameters():
    from scipy.spatial.transform import Rotation
    from teleop_assist.copilot.live_sim_proposals import (
        _recorded_geometry_profile, _recorded_geometry_matches, _same_geometry, _pose_close)
    f=Fixture();anchor=deepcopy(f.gate.frames['cal:0'])
    anchor['geometry']['obstacles']['floor']=dict(id='floor',type='plane',normal=[0,0,1],offset=0)
    profile=_recorded_geometry_profile(anchor)
    rng=np.random.default_rng(71)
    records=[]
    for _ in range(120):
        record=deepcopy(anchor)
        for identity in ('table','target','base'):
            transform=np.asarray(record['base'] if identity=='base' else record['geometry']['obstacles'][identity]['world_pose'])
            transform[:3,3]+=rng.normal(size=3)*.00035
            transform[:3,:3]=Rotation.from_rotvec(rng.normal(size=3)*.002).as_matrix()
            if identity=='base':record['base']=transform.tolist()
            else:record['geometry']['obstacles'][identity]['world_pose']=transform.tolist()
        records.append(record)
    for fault in ('obstacle_order','numeric_representation','parameter','contact','plane_pose','plane_parameter','new_obstacle'):
        record=deepcopy(anchor);geometry=record['geometry']
        if fault=='obstacle_order':geometry['obstacles']=dict(reversed(list(geometry['obstacles'].items())))
        elif fault=='numeric_representation':geometry['obstacles']['floor']['normal']=[0.,0.,1.]
        elif fault=='parameter':geometry['obstacles']['target']['length']+=1e-9
        elif fault=='contact':geometry['allowed_static_contacts']=[['left_foot','floor']]
        elif fault=='plane_pose':geometry['obstacles']['floor']['world_pose']=pose()
        elif fault=='plane_parameter':geometry['obstacles']['floor']['offset']+=1e-9
        else:geometry['obstacles']['additional']=dict(id='additional',type='sphere',radius=.1,world_pose=pose())
        records.append(record)
    for record in records:
        expected=_same_geometry(anchor['geometry'],record['geometry']) and _pose_close(anchor['base'],record['base'])
        assert _recorded_geometry_matches(profile,[_recorded_geometry_profile(record)])==expected


def test_geometry_profile_cache_is_bounded_and_cleared_on_source_context_change():
    f=Fixture()
    for _ in range(1030):f.advance(.03)
    assert len(f.gate._geometry_profiles)==1024
    assert list(f.gate.frames)==list(f.gate._geometry_profiles)
    f.now+=.03;f.index+=1;frame,scene=f.frame(),f.scene()
    frame.update(clock_id='another_boot',calibration_id='other_cal');scene['clock_id']='another_boot'
    assert f.gate.record_frame(frame,scene)['recorded']
    assert list(f.gate._geometry_profiles)==[f'other_cal:{f.index}']


def test_ranking_changes_never_replace_candidate_pose_score_or_binding():
    f=Fixture();old=f.inspect();f.preview['candidates'][0]['model_score']=.1
    other=deepcopy(f.preview['candidates'][0]);other.update(candidate_id='better_score',model_score=.99)
    f.preview['candidates'].insert(0,other)
    refreshed=f.inspect(confirmed='grasp1')
    assert refreshed.ready and refreshed.candidate==old.candidate and refreshed.model_source_id==old.model_source_id


@pytest.mark.parametrize('fault',['target','model_timestamp','model_observation','pose','candidate_source','candidate_missing','candidate_switch'])
def test_frozen_identity_mutation_requires_deliberate_reset(fault):
    f=Fixture();assert f.inspect().ready;before=deepcopy(f.preview);kwargs={}
    if fault=='target':f.preview['target_id']='different_object'
    elif fault=='model_timestamp':f.preview['model_source_capture_timestamp']+=.001
    elif fault=='model_observation':f.preview.update(model_source_observation_id='cal:1',observation_id='cal:1')
    elif fault=='pose':f.preview['candidates'][0]['base_ee'][0][3]+=.001
    elif fault=='candidate_source':f.preview['candidates'][0]['source']='another_model'
    elif fault=='candidate_missing':f.preview['candidates']=[]
    else:
        other=deepcopy(f.preview['candidates'][0]);other['candidate_id']='other';f.preview['candidates'].append(other)
        kwargs={'selected':'other','confirmed':'grasp1'}
    assert not f.inspect(**kwargs).ready
    f.preview=before
    assert not f.inspect().ready  # Restoring the old JSON is not a re-arm.
    f.gate.reset();assert f.inspect().ready


@pytest.mark.parametrize('fault',['unknown_frame','renewed_same_id','preview_valid','stationarity_flag','descriptor','clock',
    'calibration','stale','empty','unobserved_model','duplicate_candidate','centroid_wrong_object','hand_invalid'])
def test_untrusted_preview_cannot_invent_freshness_geometry_or_stationarity(fault):
    f=Fixture();kwargs={}
    if fault=='unknown_frame':f.preview['validation_observation_id']='cal:999'
    elif fault=='renewed_same_id':f.preview['capture_timestamp']+=.001
    elif fault=='preview_valid':f.preview['valid']=True
    elif fault=='stationarity_flag':f.preview['motion_static']=False
    elif fault=='descriptor':f.preview['descriptor_sha256']='b'*64
    elif fault=='clock':f.preview['clock_id']='other_clock'
    elif fault=='calibration':f.preview['calibration_id']='other_calibration'
    elif fault=='stale':kwargs['now']=f.now+.251
    elif fault=='empty':f.preview['candidates']=[]
    elif fault=='unobserved_model':f.preview.update(model_source_observation_id='cal:999',observation_id='cal:999')
    elif fault=='duplicate_candidate':f.preview['candidates'].append(deepcopy(f.preview['candidates'][0]))
    elif fault=='centroid_wrong_object':f.preview['observed_centroid_world']=[1.,1.,1.]
    else:f.preview['candidates'][0]['observed_hand_collision']['pregrasp_collision_free']=False
    assert not f.inspect(**kwargs).ready


def test_stationarity_requires_real_independent_frame_history_not_three_json_reads():
    f=Fixture();f.gate=LiveSimProposals(descriptor_sha256='a'*64)
    f.gate.record_frame(f.frame(0,10.),f.scene(10.))
    for _ in range(3):assert not f.gate.record_frame(f.frame(0,10.),f.scene(10.))['recorded']
    f.preview.update(capture_timestamp=10.,validation_observation_id='cal:0')
    assert not f.inspect(now=10.).ready


@pytest.mark.parametrize('fault',['object','table','base','gap','velocity','same_frame_changed'])
def test_actual_sim_movement_latches_failure_even_if_preview_flags_remain_true(fault):
    f=Fixture();assert f.inspect().ready;f.now+=.30 if fault=='gap' else .06;f.index+=1
    frame,scene=f.frame(),f.scene()
    if fault=='object':scene['obstacles'][1]['world_pose'][0][3]+=.005
    elif fault=='table':scene['obstacles'][0]['world_pose'][2][3]+=.005
    elif fault=='base':frame['T_world_base'][0][3]+=.005
    elif fault=='velocity':scene['observed_target_velocity'][0]=.1
    elif fault=='same_frame_changed':
        frame=f.frame(2,10.12);scene=f.scene(10.12);scene['obstacles'][1]['world_pose'][0][3]+=.005
    f.gate.record_frame(frame,scene);f.renew_preview()
    assert not f.inspect().ready
    f.advance();f.renew_preview();assert not f.inspect().ready


def test_movement_after_preview_capture_but_before_first_binding_is_not_ignored():
    f=Fixture();f.now+=.01;f.index+=1;scene=f.scene();scene['obstacles'][1]['world_pose'][0][3]+=.02
    assert f.gate.record_frame(f.frame(),scene)['recorded']
    # JSON still refers to the previous still-fresh validation frame.
    assert not f.inspect().ready


@pytest.mark.parametrize('fault',['accepted','scope','current_collision','body_fk','endpoint_collision','right_target','left_target',
    'ik_initial','solver','limits','fk_error','hash','different_snapshot','stale_state','moving_now'])
def test_endpoint_evidence_is_bound_to_exact_candidate_state_scene_and_independent_checks(fault):
    f=Fixture();prepared=f.inspect();endpoint=f.endpoint();kwargs={}
    if fault=='accepted':endpoint['accepted']=False
    elif fault=='scope':endpoint['scope']='hardware'
    elif fault=='current_collision':endpoint['state_evidence']['current_state_precheck']['collision_valid']=False
    elif fault=='body_fk':endpoint['state_evidence']['urdf_vs_observed_frames']['valid']=False
    elif fault=='endpoint_collision':endpoint['endpoint']['collision_valid']=False
    elif fault=='right_target':endpoint['ik']['right_target_base_ee'][0][3]+=.01
    elif fault=='left_target':endpoint['ik']['left_target_base_ee'][1][3]+=.01
    elif fault=='ik_initial':endpoint['ik']['q_initial'][0]=.2
    elif fault=='solver':endpoint['ik']['final']['solver_success']=False
    elif fault=='limits':endpoint['ik']['final']['limits_valid']=False
    elif fault=='fk_error':endpoint['ik']['final']['right_position_error_m']=.011
    elif fault=='hash':endpoint['state_evidence']['snapshot_sha256']='forged'
    elif fault=='different_snapshot':endpoint['state_evidence']['robot_state']['joint_positions']['right_hand_thumb_0_joint']=.01
    elif fault=='stale_state':kwargs['now']=f.now+.151
    else:
        scene=f.scene();scene['observed_target_velocity'][0]=.1;kwargs['scene']=scene
    result=f.authorize(prepared,endpoint,**kwargs)
    assert not result.accepted and result.batch is None


def test_prepared_candidate_cannot_be_mutated_or_reused_after_reset():
    f=Fixture();prepared=f.inspect();prepared.candidate['base_ee'][0][3]+=.001
    assert not f.authorize(prepared).accepted
    prepared=f.inspect();f.gate.reset();assert not f.authorize(prepared).accepted


def test_recording_rejects_scene_captured_after_unrelated_physics_time():
    f=Fixture();frame=f.frame(3,10.18);scene=f.scene(10.21)
    result=f.gate.record_frame(frame,scene)
    assert not result['recorded'] and result['reason']=='frame_scene_source_snapshot_mismatch'


def test_rejected_motion_observation_before_inference_finishes_remains_in_stationarity_audit():
    f=Fixture();f.now+=.03;f.index+=1
    scene=f.scene();scene['observed_target_velocity'][0]=.2
    assert not f.gate.record_frame(f.frame(),scene)['recorded']
    f.advance();f.renew_preview()
    assert f.inspect().reason=='model_source_predates_invalid_sim_observation'
    f.gate.reset();assert not f.inspect().ready  # Reset cannot manufacture a new inference anchor.
    anchor_id,anchor_stamp=f'cal:{f.index}',f.now
    f.advance();f.advance();f.renew_preview()
    f.preview.update(observation_id=anchor_id,model_source_observation_id=anchor_id,
        model_source_capture_timestamp=anchor_stamp)
    assert f.inspect().ready


@pytest.mark.parametrize('fault',[None,'command_changed','residual_too_large','method_unknown'])
def test_independent_fk_noop_endpoint_needs_no_invented_solver_success(fault):
    f=Fixture();prepared=f.inspect();endpoint=f.endpoint();ik=endpoint['ik']
    ik.update(reason='already_at_target',solver_invoked=False,validation_method='independent_current_pose_FK',q_solution=[0.]*14)
    ik['final'].update(solver_success=None,solver_status='not_run_already_at_target')
    for side in ('left','right'):
        ik['final'][side+'_position_error_m']=0.
        ik['final'][side+'_orientation_error_rad']=0.
    if fault=='command_changed':ik['q_solution'][0]=.001
    elif fault=='residual_too_large':ik['final']['right_position_error_m']=.001
    elif fault=='method_unknown':ik['validation_method']='assumed_held'
    assert f.authorize(prepared,endpoint).accepted is (fault is None)
