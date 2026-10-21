"""Opt-in stationary pregrasp surface owner; synthetic CPU evidence, no motion."""
from copy import deepcopy
from types import SimpleNamespace
import json
import time

import cv2
import numpy as np
import pytest

from teleop_assist import surface_tracking, visual_tracker
from teleop_assist.learned_perception import mask_to_target
from teleop_assist.perception_runtime import AsyncPerception
from teleop_assist.visual_tracker import RGBDMaskTracker, VisualTrackerConfig
from teleop_assist.copilot.service import freeze_selected_failure_inputs, write_selected_feature_failure
from test_visual_tracker_surface_integration import scene, next_frame, NoWorker, neighbour_scene
from test_visual_tracker import cad_evidence, frames


@pytest.fixture
def clock(monkeypatch):
    # Explicit recorded evaluation clock; original acquisition values unchanged.
    clock=SimpleNamespace(now=10.)
    native=surface_tracking.measure
    monkeypatch.setattr(surface_tracking,'measure',lambda *a,**kw:native(*a,**kw,now=clock.now))
    monkeypatch.setattr(visual_tracker,'time',SimpleNamespace(
        monotonic=lambda:clock.now,perf_counter=time.perf_counter))
    return clock


def make_tracker(frame,target,mask,*,enabled=True,config=None,selected='selected'):
    tracker=RGBDMaskTracker(config,selected_surface_primary=enabled)
    tracker.set_surface_target(selected,camera_role='head')
    tracker.set_diagnostic_target(target['id'])
    tracker.initialize(frame,[target],{target['id']:mask})
    return tracker


def model_result(targets,masks):
    return dict(targets=deepcopy(targets),_masks=deepcopy(masks),diagnostics={})


def initialized_runtime(frame,targets,masks,*,enabled=True):
    runtime=AsyncPerception(NoWorker(),semantic_interval=float('inf'),selected_surface_primary=enabled)
    runtime.selected_target_id='selected'
    runtime._initialize(frame,model_result(targets,masks))
    runtime.last_submit=float('inf')
    return runtime


def target_for(frame,mask,identity='selected',semantic=None):
    target,reason=mask_to_target(mask,frame['depth'],frame['intrinsics'],frame['T_world_camera'],
        label='handle',detection_score=.6,mask_score=.9)
    assert reason is None
    target.update(id=identity,frame_id=frame['frame_id'],timestamp=frame['timestamp'])
    if semantic is not None:target['semantic_keyframe_timestamp']=semantic
    return target


@pytest.mark.parametrize('bad',[None,1,0,'true',np.bool_(True)])
def test_mode_requires_explicit_boolean(bad):
    with pytest.raises(ValueError):RGBDMaskTracker(selected_surface_primary=bad)
    with pytest.raises(ValueError):AsyncPerception(NoWorker(),selected_surface_primary=bad)


def test_primary_measures_without_calling_gftt(clock,monkeypatch):
    old,target,mask=scene();tracker=make_tracker(old,target,mask)
    monkeypatch.setattr(cv2,'goodFeaturesToTrack',lambda *a,**kw:pytest.fail('No GFTT in selected primary'))
    current=next_frame(old);clock.now=current['timestamp']
    result=tracker.track(current);measured,=result['targets']
    assert measured['perception_method']=='measured_rgbd_surface_reprojection'
    assert measured['semantic_keyframe_timestamp']==10.
    assert measured['position_sigma_m']>=target['position_sigma_m']
    assert measured['pose_confidence']<=target['pose_confidence']
    assert 'optical_failure_trigger' not in measured['tracking_quality']
    assert tracker.last_feature_failure is None
    json.dumps(result)


@pytest.mark.parametrize('bad',['depth_missing','depth_shift','rgb','epoch','calibration','clock'])
def test_primary_failure_has_no_lk_rescue(clock,monkeypatch,bad):
    old,target,mask=scene();tracker=make_tracker(old,target,mask)
    current=next_frame(old);clock.now=current['timestamp']
    if bad=='depth_missing':current['depth'][mask]=np.nan
    if bad=='depth_shift':current['depth'][mask]+=.020
    if bad=='rgb':current['rgb'][mask]=0
    if bad in ('epoch','calibration','clock'):
        current[dict(epoch='source_instance_id',calibration='calibration_id',clock='clock_id')[bad]]='changed'
    monkeypatch.setattr(cv2,'goodFeaturesToTrack',lambda *a,**kw:pytest.fail('A failed surface cannot invoke LK'))
    assert tracker.track(current)['targets']==[]
    assert tracker.last_masks=={}
    assert tracker.last_feature_failure['failure_stage']=='surface_measurement'
    assert tracker.last_feature_failure['reason']=='selected_surface_measurement_failed'
    assert tracker.track(next_frame(current))['targets']==[]


@pytest.mark.parametrize('enabled',[False,True])
def test_original_confirmed_smaller_mask_counterexample(clock,enabled):
    old,_,mask=scene();old['intrinsics'][0,0]=old['intrinsics'][1,1]=1000.
    inner=np.zeros_like(mask);inner[36:64,46:74]=True
    runtime=initialized_runtime(old,[target_for(old,mask)],{'selected':mask},enabled=enabled)
    try:
        runtime.confirmed_head_tracker.initialize(old,[target_for(old,inner)],{'selected':inner})
        current=next_frame(old);current['depth'][mask&~inner]=.7
        for index in (2,3,4):
            current.update(frame_id=index,timestamp=10.+.1*(index-1));clock.now=current['timestamp']
            result=runtime.process({'frames':{'head':current}},'selected')
            assert result['diagnostics']['multiview']['confirmed_head']['detected_count']==1
            assert bool(result['targets']) is (not enabled)
            assert ('selected' in runtime.tracker.last_masks) is (not enabled)
            assert ('selected' in runtime.observations) is (not enabled)
    finally:runtime.close()


@pytest.mark.parametrize('enabled',[False,True])
def test_wrist_success_cannot_restore_primary_owner(clock,enabled):
    old,target,mask=frames();old.update(clock_id='offline',calibration_id='head',T_world_base=np.eye(4))
    clock.now=old['timestamp'];identity=target['id']
    runtime=AsyncPerception(NoWorker(),semantic_interval=float('inf'),selected_surface_primary=enabled)
    runtime.selected_target_id=identity
    try:
        result=model_result([],{});result['_camera_observations']={'right_wrist':dict(
            frame=old,targets=[target],masks={identity:mask},head_targets={identity:deepcopy(target)})}
        runtime._initialize(old,result);runtime.last_submit=float('inf')
        current=next_frame(old);clock.now=current['timestamp']
        result=runtime.process({'frames':{'head':current,'right_wrist':current}},identity)
        assert result['diagnostics']['multiview']['views']['right_wrist']['detected_count']==1
        assert bool(result['targets']) is (not enabled)
        assert (identity in runtime.observations) is (not enabled)
        assert identity not in runtime.tracker.last_masks
    finally:runtime.close()


@pytest.mark.parametrize('enabled',[False,True])
def test_current_empty_semantic_is_not_retained(clock,enabled):
    old,target,mask=scene();runtime=initialized_runtime(old,[target],{'selected':mask},enabled=enabled)
    try:
        current=next_frame(old);clock.now=current['timestamp']
        runtime._initialize(current,model_result([],{}),{'frames':{'head':current}})
        result=runtime.process({'frames':{'head':current}},'selected')
        assert bool(result['targets']) is (not enabled)
        assert runtime.last_model_diagnostics['direct_current_frame_rebase']['head']['adopted']
        if enabled:assert runtime.observations=={} and runtime.tracker.last_masks=={}
    finally:runtime.close()


def delayed_scene():
    old,target,mask=scene();other=np.zeros_like(mask);other[30:70,85:110]=True
    old['rgb'][other]=np.random.default_rng(413).integers(20,220,(int(other.sum()),3),dtype=np.uint8)
    old['depth'][other]=1.
    return old,target,mask,other


@pytest.mark.parametrize('case',['valid','expired','depth_failed'])
def test_delayed_unselected_only_success_retains_owner_but_remeasures(clock,case):
    old,target,mask,other=delayed_scene()
    if case=='expired':target['semantic_keyframe_timestamp']=9.15
    runtime=initialized_runtime(old,[target],{'selected':mask});owner=runtime.tracker
    try:
        source=next_frame(old);current=next_frame(source);clock.now=current['timestamp']
        if case=='depth_failed':current['depth'][mask]=np.nan
        semantic=model_result([target_for(source,other,'other')],{'other':other})
        runtime._initialize(source,semantic,{'frames':{'head':current}})
        rebase=runtime.last_model_diagnostics['direct_current_frame_rebase']['head']
        assert rebase['detected_count']==1 and not rebase['adopted']
        assert runtime.tracker is owner
        result=runtime.process({'frames':{'head':current}},'selected')
        if case=='valid':
            target,=result['targets'];assert target['id']=='selected'
            assert target['semantic_keyframe_timestamp']==10. and target['timestamp']==current['timestamp']
        else:
            assert result['targets']==[] and runtime.observations=={} and runtime.tracker.last_masks=={}
    finally:runtime.close()


def test_delayed_unselected_neighbour_remains_negative_evidence(clock):
    old,targets,masks=neighbour_scene(True);selected,neighbour=targets
    runtime=initialized_runtime(old,[selected],{'selected':masks['selected']})
    try:
        source=next_frame(old);current=next_frame(source);clock.now=current['timestamp']
        neighbour=dict(neighbour,frame_id=source['frame_id'],timestamp=source['timestamp'])
        runtime._initialize(source,model_result([neighbour],{'neighbour':masks['neighbour']}),{'frames':{'head':current}})
        result=runtime.process({'frames':{'head':current}},'selected')
        assert result['targets']==[] and runtime.observations=={} and runtime.tracker.last_masks=={}
        assert 'neighbour' in result['diagnostics']['lost_tracks']['selected']
    finally:runtime.close()


@pytest.mark.parametrize('enabled',[False,True])
def test_all_new_trackers_inherit_mode(clock,enabled):
    old,target,mask=scene();runtime=initialized_runtime(old,[target],{'selected':mask},enabled=enabled)
    try:
        source=next_frame(old);clock.now=source['timestamp']
        target=dict(target,frame_id=source['frame_id'],timestamp=source['timestamp'])
        result=model_result([target],{'selected':mask})
        result['_camera_observations']={'right_wrist':dict(frame=source,targets=[target],
            masks={'selected':mask},head_targets={'selected':target})}
        runtime._initialize(source,result)
        assert all(t.selected_surface_primary is enabled for t in (
            runtime.tracker,runtime.confirmed_head_tracker,*runtime.wrist_trackers.values()))
    finally:runtime.close()


@pytest.mark.parametrize('semantic,frame_time,check_time,accepted',[
    (9.,10.,10.,True),(9.,10.,np.nextafter(10.,np.inf),False),
    (10.,10.9,11.01,False),(8.9,10.,10.,False)])
def test_same_frame_cad_cannot_extend_capture_or_real_evaluation_ttl(clock,semantic,frame_time,check_time,accepted):
    old,target,mask=scene();old['timestamp']=frame_time;target.update(timestamp=frame_time,semantic_keyframe_timestamp=semantic)
    tracker=make_tracker(old,target,mask);assert tracker.confirm_identity(cad_evidence(target,frame_time,1))
    clock.now=check_time
    assert bool(tracker.track(old)['targets']) is accepted
    if not accepted:assert tracker.last_masks=={}


def test_default_cad_semantics_remain_unchanged(clock):
    old,target,mask=scene();target['semantic_keyframe_timestamp']=8.9
    tracker=make_tracker(old,target,mask,enabled=False)
    assert tracker.confirm_identity(cad_evidence(target,10.,1));clock.now=11.01
    assert tracker.track(old)['targets']


@pytest.mark.parametrize('kind',['motion','normal'])
def test_primary_keeps_common_geometry_gates(clock,kind):
    old,target,mask=scene()
    cfg=VisualTrackerConfig(world_motion_slack_m=0,max_world_speed_m_s=0) if kind=='motion' else VisualTrackerConfig(max_normal_angle_deg=.01)
    tracker=make_tracker(old,target,mask,config=cfg);current=next_frame(old);clock.now=current['timestamp']
    if kind=='motion':current['depth'][mask]+=.001
    else:current['depth'][30:70,40:80]+=np.linspace(-.004,.004,40)[None,:]
    result=tracker.track(current);assert result['targets']==[]
    assert 'inconsistent_' in result['diagnostics']['lost_tracks']['selected']


def test_exact_surface_failure_capture_has_no_invented_gftt(clock,tmp_path):
    old,target,mask=scene();tracker=make_tracker(old,target,mask)
    expected=deepcopy(old);current=next_frame(old);current['depth'][mask]=np.nan;clock.now=current['timestamp']
    assert tracker.track(current)['targets']==[]
    capture=tracker.last_feature_failure
    assert capture['capture_status']=='captured' and capture['failure_stage']=='surface_measurement'
    assert set(capture['arrays'])=={'previous_rgb','previous_depth','previous_mask'}
    assert not {'point_count','feature_parameters','corner_points','flow_inputs'}&set(capture)
    assert capture['previous_target']['id']=='selected'
    frozen,arrays,metadata,_=freeze_selected_failure_inputs(current,capture)
    old['rgb'][:]=0;old['depth'][:]=0;mask[:]=False;capture['arrays']['previous_rgb'][:]=0
    directory=tmp_path/'replay';write_selected_feature_failure(directory,frozen,arrays,metadata,dict(motion_authorized=False))
    with np.load(directory/'tracking_inputs.npz',allow_pickle=False) as data:
        assert np.array_equal(data['previous_rgb'],expected['rgb'])
        assert np.array_equal(data['previous_depth'],expected['depth'])
        assert np.array_equal(data['previous_mask'],expected['depth']==1.)
        assert json.loads(str(data['metadata_json']))['surface_previous_frame']['timestamp']==10.
    with np.load(directory/'frame.npz',allow_pickle=False) as data:
        assert np.array_equal(data['depth'],current['depth'],equal_nan=True)
    with pytest.raises(FileExistsError):write_selected_feature_failure(directory,frozen,arrays,metadata,{})


def test_capture_failure_does_not_change_surface_rejection(clock,monkeypatch):
    old,target,mask=scene();tracker=make_tracker(old,target,mask)
    def fail(*a,**k):raise MemoryError('diagnostic-only')
    monkeypatch.setattr(tracker,'_capture_surface_failure',fail)
    current=next_frame(old);current['depth'][mask]=np.nan;clock.now=current['timestamp']
    assert tracker.track(current)['targets']==[]
    assert tracker.last_feature_failure['capture_status']=='capture_failed'


def test_selected_diagnostic_id_cannot_enable_primary(clock,monkeypatch):
    old,target,mask=scene();tracker=make_tracker(old,target,mask,selected=None)
    calls=[];native=cv2.goodFeaturesToTrack
    monkeypatch.setattr(cv2,'goodFeaturesToTrack',lambda *a,**kw:(calls.append(1),native(*a,**kw))[1])
    current=next_frame(old);clock.now=current['timestamp']
    assert tracker.track(current)['targets']==[] and calls
    assert tracker.last_feature_failure['reason']=='insufficient_visual_features'


def test_delayed_selected_failure_and_other_success_cannot_replace_valid_owner(clock):
    old,target,mask,other=delayed_scene();runtime=initialized_runtime(old,[target],{'selected':mask})
    owner=runtime.tracker
    try:
        source=next_frame(old);source['depth'][mask]+=.030
        current=next_frame(old,frame_id=3,timestamp=10.2);clock.now=current['timestamp']
        result=model_result([target_for(source,mask),target_for(source,other,'other')],
                            {'selected':mask,'other':other})
        from concurrent.futures import Future
        runtime.future=Future();runtime.future.set_result(result);runtime.source={'frames':{'head':source}}
        observed=runtime.process({'frames':{'head':current}},'selected')
        assert runtime.tracker is owner and [t['id'] for t in observed['targets']]==['selected']
        assert observed['targets'][0]['semantic_keyframe_timestamp']==10.
        capture=runtime.selected_feature_failure
        assert capture['phase']=='semantic_keyframe_rebase' and capture['tracker_stage']=='head_rebase'
        assert capture['failure_stage']=='surface_measurement'
        assert capture['previous_frame']['timestamp']==source['timestamp']
        assert capture['selected_target_observed_after_processing'] is True
        assert np.array_equal(capture['arrays']['previous_depth'],source['depth'])
    finally:runtime.close()


def test_primary_capture_budget_is_still_eight_mib(clock):
    old,target,mask=scene();tracker=make_tracker(old,target,mask)
    tracker._surface_frame['rgb']=np.zeros((1100,1100,3),np.uint8)
    tracker._surface_frame['depth']=np.ones((1100,1100),np.float32)
    tracker._capture_surface_failure(next_frame(old),target,np.ones((1100,1100),bool),'test',{})
    capture=tracker.last_feature_failure
    assert capture['capture_status']=='size_or_dtype_limit'
    assert capture['max_array_bytes']==8*1024*1024 and 'arrays' not in capture


@pytest.mark.parametrize('recording_failure',[False,True])
def test_surface_capture_reaches_service_once_without_control_authority(clock,tmp_path,monkeypatch,recording_failure):
    from teleop_assist.copilot import service
    from teleop_assist.copilot.mailbox import atomic_json,read_json
    from test_copilot_service import Engine
    old,target,mask=scene();runtime=initialized_runtime(old,[target],{'selected':mask})
    engine=Engine();engine.finish.set()
    session=service.ProposalSession(tmp_path,engine,clock=lambda:clock.now,clock_id=old['clock_id'])
    try:
        atomic_json(tmp_path/'target_selection.json',dict(target_id='selected',revision=1))
        session.update(old,[target],{'selected':mask})
        if session.future is not None:session.future.result(2)
        current=next_frame(old);current['depth'][mask]=np.nan;clock.now=current['timestamp']
        observed=runtime.process({'frames':{'head':current}},'selected')
        capture=runtime.selected_feature_failure
        assert capture['tracker_stage']=='head' and capture['phase']=='current_tracking'
        if recording_failure:monkeypatch.setattr(service,'SELECTED_REPLAY_MAX_BYTES',32)
        session.update(current,observed['targets'],runtime.tracker.last_masks,
            tracking_failure=runtime.selected_tracking_failure,feature_failure=capture)
        if session.selected_feature_audit is not None:session.selected_feature_audit.result(2)
        session.publish();first=deepcopy(session.first_selected_feature_failure)
        assert first['reason']=='selected_surface_measurement_failed'
        assert first['replay_status']==('failed' if recording_failure else 'complete')
        assert first['motion_authorized'] is False
        assert session.error=='selected_object_not_observed'
        assert read_json(tmp_path/'latest_proposals.json')['valid'] is False
        monkeypatch.setattr(service,'SELECTED_REPLAY_MAX_BYTES',16*1024*1024)
        session._capture_selected_feature_failure(capture)
        assert session.first_selected_feature_failure==first
        if not recording_failure:
            with np.load(tmp_path/'first_selected_feature_failure/tracking_inputs.npz',allow_pickle=False) as data:
                assert set(data.files)=={'metadata_json','previous_rgb','previous_depth','previous_mask'}
                assert json.loads(str(data['metadata_json']))['failure_stage']=='surface_measurement'
    finally:session.close();runtime.close()
