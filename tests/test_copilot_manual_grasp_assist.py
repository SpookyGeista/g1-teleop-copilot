from copy import deepcopy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from teleop_assist.copilot.manual_grasp_assist import ManualGraspAssist,wrist_support


def fixture():
    pose=np.eye(4);pose[2,3]=1.
    goal=pose.copy();goal[0,3]=.025;goal[:3,:3]=Rotation.from_euler('z',15,degrees=True).as_matrix()
    candidate=dict(candidate_id='learned:1',base_ee=goal.tolist(),geometry_valid=True,ik_valid=True,
        recorded_sim_endpoint_clear=True,source='GraspGenX_pretrained_current_Dex3')
    batch=dict(capture_timestamp=10.,clock_id='test',motion_static=True,processed_frame=dict(rejected=False),
        candidates=[candidate],observed_centroid_world=[0.,0.,1.])
    frame=dict(camera_source='right_wrist',clock_id='test',timestamp=10.,frame_id=1,
        depth=np.ones((40,40)),intrinsics=np.array([[500,0,20],[0,500,20],[0,0,1]]),T_world_camera=np.eye(4))
    return pose,goal,batch,frame


def ready(**kwargs):
    a=ManualGraspAssist(simulation_mode=True,**kwargs);pose,goal,batch,frame=fixture()
    a.update(batch,dict(revision=1,select_candidate_id='learned:1'),frame,now=10.,clock_id='test',active=True)
    a.update(batch,dict(revision=2,confirm=True),frame,now=10.,clock_id='test',active=True)
    return a,pose,goal,batch,frame


def test_both_views_required_and_metric_depth_checks():
    _,_,batch,frame=fixture()
    assert wrist_support(frame,batch['observed_centroid_world'],now=10.,clock_id='test')['accepted']
    for change in (dict(timestamp=9.),dict(clock_id='other'),dict(depth=np.ones((40,40))*1.2),
                   dict(camera_source='head'),dict(depth=np.full((40,40),np.nan))):
        assert not wrist_support(dict(frame,**change),batch['observed_centroid_world'],now=10.,clock_id='test')['accepted']


def test_learned_alignment_is_bounded_right_only_and_releases_without_jump():
    a,pose,goal,batch,frame=ready();previous=pose.copy()
    for i in range(100):
        now=10.+i*.02;batch['capture_timestamp']=frame['timestamp']=now
        a.update(batch,dict(revision=3+i,assist_held=True),frame,now=now,clock_id='test',active=True)
        left,right=a.apply([pose,pose],now=now)
        np.testing.assert_array_equal(left,pose)
        assert np.linalg.norm(right[:3,3]-pose[:3,3])<=.030001
        assert np.linalg.norm(right[:3,3]-previous[:3,3])<=.000601
        assert (Rotation.from_matrix(right[:3,:3])*Rotation.from_matrix(previous[:3,:3]).inv()).magnitude()<=.007001
        previous=right
    assert np.linalg.norm(right[:3,3]-goal[:3,3])<1e-6
    assert (Rotation.from_matrix(right[:3,:3])*Rotation.from_matrix(goal[:3,:3]).inv()).magnitude()<1e-6
    a.update(batch,dict(revision=104,assist_held=False),frame,now=now,clock_id='test',active=True)
    np.testing.assert_allclose(a.apply([pose,pose],now=now+.02)[1],previous,atol=1e-8)
    assert not a.applied
    a.update(batch,dict(revision=105,start=True),frame,now=now,clock_id='test',active=False)
    np.testing.assert_array_equal(a.apply([pose,pose],now=now+.04)[1],pose)


@pytest.mark.parametrize('condition',['no_confirmation','stale_head','stale_wrist','foreign_source','changed_candidate','far_hand','moving_object'])
def test_no_new_correction_with_invalid_or_unconfirmed_evidence(condition):
    a,pose,goal,batch,frame=ready()
    if condition=='no_confirmation':a.confirmed=None
    if condition=='stale_head':batch['capture_timestamp']=9.
    if condition=='stale_wrist':frame['timestamp']=9.
    if condition=='foreign_source':batch['candidates'][0]['source']='invented_pose'
    if condition=='changed_candidate':batch['candidates'][0]['base_ee'][0][3]=.04
    if condition=='far_hand':pose[0,3]=-.2
    if condition=='moving_object':batch['motion_static']=False
    a.update(batch,dict(revision=3,assist_held=True),frame,now=10.,clock_id='test',active=True)
    a.apply([pose,pose],now=10.)
    np.testing.assert_allclose(a.apply([pose,pose],now=10.02)[1],pose,atol=1e-8)
    assert not a.applied


def test_refuses_hardware_mode():
    with pytest.raises(ValueError,match='simulation-only'):ManualGraspAssist(simulation_mode=False)


@pytest.mark.parametrize('confirmed',[False,True])
def test_calibrate_after_choosing_grasp_preserves_choice_without_starting_assistance(confirmed):
    a,pose,goal,batch,frame=ready()
    if not confirmed:a.confirmed=None
    choice,confirmation=deepcopy(a.selected),deepcopy(a.confirmed)
    a.translation[:]=.01;a.rotation=Rotation.from_euler('z',10,degrees=True)
    a.update(batch,dict(revision=3,clutch=True),frame,now=10.,clock_id='test',active=False)
    assert a.selected==choice and a.confirmed==confirmation
    np.testing.assert_array_equal(a.translation,np.zeros(3))
    assert a.rotation.magnitude()==0 and not a.applied
    for revision,ui in [(4,{}),(5,dict(start=True)),(6,{})]:
        a.update(batch,dict(revision=revision,**ui),frame,now=10.,clock_id='test',active=True)
        np.testing.assert_array_equal(a.apply([pose,pose],now=10.+revision*.01)[1],pose)
        assert a.selected==choice and a.confirmed==confirmation
    a.update(batch,dict(revision=7,release=True),frame,now=10.,clock_id='test',active=False)
    assert a.selected is None and a.confirmed is None


@pytest.mark.parametrize('invalid',['stale','changed_pose'])
def test_retained_confirmation_after_calibration_cannot_bypass_current_evidence(invalid):
    a,pose,_,batch,frame=ready()
    if invalid=='stale':batch['capture_timestamp']=9.
    else:batch['candidates'][0]['base_ee'][0][3]+=.01
    a.update(batch,dict(revision=3,clutch=True),frame,now=10.,clock_id='test',active=False)
    assert a.confirmed is not None
    a.update(batch,dict(revision=4,start=True),frame,now=10.,clock_id='test',active=False)
    a.update(batch,dict(revision=5,assist_held=True),frame,now=10.,clock_id='test',active=True)
    for now in (10.,10.02):
        np.testing.assert_array_equal(a.apply([pose,pose],now=now)[1],pose)
        assert not a.applied


@pytest.mark.parametrize('distance',[.20,.30,.39])
def test_early_guidance_works_outside_old_capture_radius_and_remains_bounded(distance):
    a,pose,goal,batch,frame=ready(guidance_profile='early_guidance')
    pose[:3,3]=goal[:3,3]-[distance,0,0]
    pose[:3,:3]=Rotation.from_euler('z',-60,degrees=True).as_matrix()
    previous=pose.copy()
    for i in range(160):
        now=10.+i*.02;batch['capture_timestamp']=frame['timestamp']=now
        a.update(batch,dict(revision=3+i,assist_held=True),frame,now=now,clock_id='test',active=True)
        left,right=a.apply([pose,pose],now=now)
        np.testing.assert_array_equal(left,pose)
        assert np.linalg.norm(right[:3,3]-previous[:3,3])<=.001001
        assert np.linalg.norm(right[:3,3]-pose[:3,3])<=.100001
        assert Rotation.from_matrix(right[:3,:3]@previous[:3,:3].T).magnitude()<=.010001
        assert Rotation.from_matrix(right[:3,:3]@pose[:3,:3].T).magnitude()<=np.deg2rad(45)+1e-6
        previous=right
    assert a.applied and np.linalg.norm(right[:3,3]-goal[:3,3])<distance
    assert a.gain>0
    a.update(batch,dict(revision=200,assist_held=False),frame,now=now,clock_id='test',active=True)
    np.testing.assert_allclose(a.apply([pose,pose],now=now+.02)[1],previous,atol=1e-8)
    assert not a.applied


@pytest.mark.parametrize('distance,correction',[(.30,True),(.14,False),(.41,False)])
def test_head_can_guide_early_but_wrist_is_needed_near_grasp(distance,correction):
    a,pose,goal,batch,frame=ready(guidance_profile='early_guidance')
    pose[:3,3]=goal[:3,3]-[distance,0,0]
    frame['depth'][:]=np.nan
    for i in range(5):
        now=10.+i*.02;batch['capture_timestamp']=frame['timestamp']=now
        a.update(batch,dict(revision=3+i,assist_held=True),frame,now=now,clock_id='test',active=True)
        right=a.apply([pose,pose],now=now)[1]
    assert a.applied is correction
    if correction:
        assert a.depth_mode=='head_approach' and a.translation[0]>0
        assert np.linalg.norm(goal[:3,3]-right[:3,3])>=.15-1e-8
    else:np.testing.assert_array_equal(right,pose)


@pytest.mark.parametrize('fault',['stale_wrist','disagreement','stale_head','no_hold','no_confirmation'])
def test_early_guidance_still_requires_current_nonconflicting_evidence_and_explicit_hold(fault):
    a,pose,goal,batch,frame=ready(guidance_profile='early_guidance')
    pose[:3,3]=goal[:3,3]-[.30,0,0]
    if fault=='stale_wrist':frame['timestamp']=9.
    if fault=='stale_head':batch['capture_timestamp']=9.
    if fault=='no_confirmation':a.confirmed=None
    a.update(batch,dict(revision=3,assist_held=fault!='no_hold'),frame,now=10.,clock_id='test',active=True)
    if fault=='disagreement':a.wrist.update(accepted=False,reason='wrist_head_disagreement')
    for now in (10.,10.02):
        np.testing.assert_array_equal(a.apply([pose,pose],now=now)[1],pose)
        assert not a.applied
