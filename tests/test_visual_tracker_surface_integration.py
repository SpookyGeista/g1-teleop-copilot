"""Smooth selected surfaces, actual RGBD correspondence and unchanged loss gates."""
from copy import deepcopy
from dataclasses import replace
import json

import cv2
import numpy as np
import pytest

from teleop_assist import surface_tracking
from teleop_assist.learned_perception import mask_to_target
from teleop_assist.visual_tracker import RGBDMaskTracker, VisualTrackerConfig
from teleop_assist.perception_runtime import AsyncPerception
from teleop_assist.copilot.service import freeze_selected_failure_inputs, write_selected_feature_failure
from test_visual_tracker import cad_evidence


@pytest.fixture(autouse=True)
def explicit_recorded_evaluation_clock(monkeypatch):
    # Only this offline test's evaluation clock follows the recorded image.
    # Native acquisition fields are unchanged. Production uses host monotonic.
    measure=surface_tracking.measure
    def offline(*args,**kwargs):
        return measure(*args,**kwargs,now=args[3]['timestamp'])
    monkeypatch.setattr(surface_tracking,'measure',offline)


def scene():
    rgb=np.full((100,120,3),95,np.uint8)
    mask=np.zeros((100,120),bool);mask[30:70,40:80]=True
    depth=np.full(mask.shape,1.3,np.float32);depth[mask]=1.
    frame=dict(rgb=rgb,depth=depth,intrinsics=np.array([[200.,0,60.],[0,200.,50.],[0,0,1.]]),
        T_world_camera=np.eye(4),T_world_base=np.eye(4),clock_id='offline_camera_clock',
        calibration_id='measured_camera_mount',frame_id=1,timestamp=10.)
    target,reason=mask_to_target(mask,depth,frame['intrinsics'],frame['T_world_camera'],
        label='handle',detection_score=.6,mask_score=.9)
    assert reason is None
    target.update(id='selected',frame_id=1,timestamp=10.)
    return frame,target,mask


def setup_tracker(*,selected='selected',enabled=True,config=None):
    frame,target,mask=scene();tracker=RGBDMaskTracker(config)
    if enabled:tracker.set_surface_target(selected,camera_role='primary')
    tracker.set_diagnostic_target(target['id'])
    tracker.initialize(frame,[target],{target['id']:mask})
    return tracker,frame,target,mask


def next_frame(frame,**changes):
    return {**deepcopy(frame),'frame_id':frame['frame_id']+1,'timestamp':frame['timestamp']+.1,**changes}


def test_smooth_selected_surface_is_measured_without_invented_lk_evidence(tmp_path):
    tracker,old,target,mask=setup_tracker()
    assert cv2.goodFeaturesToTrack(cv2.cvtColor(old['rgb'],cv2.COLOR_RGB2GRAY),
        maxCorners=40,qualityLevel=.01,minDistance=3,mask=mask.astype(np.uint8)*255,blockSize=3) is None
    current=next_frame(old);result=tracker.track(current)
    measured,=result['targets']
    assert measured['perception_method']=='measured_rgbd_surface_reprojection'
    assert measured['frame_id']==2 and measured['timestamp']==10.1
    assert measured['semantic_keyframe_timestamp']==10.
    assert measured['tracking_quality']['method']=='measured_rgbd_surface_reprojection'
    assert 'flow_residual_px' not in measured['tracking_quality']
    assert 'inlier_count' not in measured['tracking_quality']
    assert measured['position_sigma_m']>=target['position_sigma_m']
    assert measured['pose_confidence']<=target['pose_confidence']
    assert np.array_equal(tracker.last_masks['selected'],mask)
    capture=tracker.last_feature_failure
    assert capture['point_count']==0 and capture['surface_fallback']['accepted_measurement'] is True
    assert np.array_equal(capture['arrays']['previous_rgb'],old['rgb'])
    assert np.array_equal(capture['arrays']['previous_depth'],old['depth'])
    assert capture['surface_previous_frame']['camera_id_source']=='tracker_owner_channel'
    frozen,arrays,metadata,_=freeze_selected_failure_inputs(current,capture)
    out=tmp_path/'replay'
    write_selected_feature_failure(out,frozen,arrays,metadata,{'motion_authorized':False})
    with np.load(out/'tracking_inputs.npz',allow_pickle=False) as recorded:
        assert np.array_equal(recorded['previous_depth'],old['depth'])
        assert json.loads(str(recorded['metadata_json']))['surface_previous_frame']['timestamp']==10.
    json.dumps(result)  # No RGB/depth arrays enter public perception JSON.


@pytest.mark.parametrize('selected,enabled',[(None,True),('different',True),('selected',False)])
def test_no_implicit_surface_measurement_permission(selected,enabled):
    tracker,old,_,_=setup_tracker(selected=selected,enabled=enabled)
    assert tracker.track(next_frame(old))['targets']==[]


@pytest.mark.parametrize('field,value',[('clock_id','different'),('calibration_id','different'),
    ('source_instance_id','new_producer')])
def test_source_change_loses_track_and_later_good_image_cannot_revive(field,value):
    tracker,old,_,_=setup_tracker()
    rejected=next_frame(old,**{field:value})
    assert tracker.track(rejected)['targets']==[]
    assert tracker.track(next_frame(old,frame_id=3,timestamp=10.2))['targets']==[]


def test_capture_is_owned_and_missing_source_metadata_does_not_enable_fallback():
    tracker,old,_,mask=setup_tracker();current=next_frame(old)
    old['depth'][:]=.2;old['rgb'][:]=0;old['T_world_camera'][0,3]=99.
    assert tracker.track(current)['targets']
    assert np.array_equal(tracker.last_masks['selected'],mask)
    tracker,old,_,_=setup_tracker()
    current=next_frame(old);current.pop('clock_id')
    assert tracker.track(current)['targets']==[]


@pytest.mark.parametrize('kind',['missing_depth','far_occluder','changed_rgb'])
def test_current_surface_evidence_must_be_measured(kind):
    tracker,old,_,mask=setup_tracker();current=next_frame(old)
    if kind=='missing_depth':current['depth'][mask]=np.nan
    if kind=='far_occluder':current['depth'][mask]=.7
    if kind=='changed_rgb':current['rgb'][mask]=0
    assert tracker.track(current)['targets']==[]


def test_current_cad_does_not_extend_the_surface_fallback_semantic_ttl():
    old,target,mask=scene();target['semantic_keyframe_timestamp']=9.1
    tracker=RGBDMaskTracker();tracker.set_surface_target('selected',camera_role='primary')
    tracker.initialize(old,[target],{'selected':mask})
    assert tracker.confirm_identity(cad_evidence(target,10.,1))
    result=tracker.track(next_frame(old,timestamp=10.2))
    assert result['targets']==[]
    assert 'original_semantic_evidence_expired' in result['diagnostics']['lost_tracks']['selected']


def test_surface_fallback_cannot_bypass_stricter_configured_world_motion_gate():
    config=VisualTrackerConfig(world_motion_slack_m=0.,max_world_speed_m_s=0.)
    tracker,old,_,mask=setup_tracker(config=config);current=next_frame(old)
    current['depth'][mask]+=.001
    result=tracker.track(current)
    assert result['targets']==[]
    assert result['diagnostics']['lost_tracks']['selected']=='inconsistent_measured_world_motion'


def test_surface_fallback_cannot_bypass_stricter_configured_normal_gate():
    tracker,old,_,mask=setup_tracker(config=VisualTrackerConfig(max_normal_angle_deg=.01))
    current=next_frame(old)
    current['depth'][30:70,40:80]+=np.linspace(-.004,.004,40)[None,:]
    result=tracker.track(current)
    assert result['targets']==[]
    assert result['diagnostics']['lost_tracks']['selected']=='inconsistent_surface_normal'


def test_current_geometry_config_is_applied():
    tracker,old,_,mask=setup_tracker()
    tracker.geometry_config=replace(tracker.geometry_config,min_mask_pixels=int(mask.sum())+1)
    assert tracker.track(next_frame(old))['targets']==[]


def neighbour_scene(textured_neighbour,fx=1000.):
    old,_,_=scene();old['intrinsics'][0,0]=old['intrinsics'][1,1]=fx
    selected=np.zeros(old['depth'].shape,bool);selected[30:70,40:60]=True
    neighbour=np.zeros_like(selected);neighbour[30:70,62:72]=True
    old['depth'][:]=1.3;old['depth'][selected|neighbour]=1.
    if textured_neighbour:
        old['rgb'][neighbour]=np.random.default_rng(12).integers(20,220,(int(neighbour.sum()),3),dtype=np.uint8)
    targets=[]
    for identity,mask in [('selected',selected),('neighbour',neighbour)]:
        target,reason=mask_to_target(mask,old['depth'],old['intrinsics'],old['T_world_camera'],
            label='handle',detection_score=.6,mask_score=.9)
        assert reason is None
        targets.append(dict(target,id=identity,frame_id=1,timestamp=10.))
    return old,targets,dict(selected=selected,neighbour=neighbour)


@pytest.mark.parametrize('textured_neighbour',[False,True])
def test_nearby_competing_instance_cannot_disappear_from_correspondence_checks(textured_neighbour):
    old,targets,masks=neighbour_scene(textured_neighbour)
    tracker=RGBDMaskTracker();tracker.set_surface_target('selected',camera_role='primary')
    tracker.initialize(old,targets,masks)
    result=tracker.track(next_frame(old))
    assert all(target['id']!='selected' for target in result['targets'])
    expected='ambiguous_current_neighbour' if textured_neighbour else 'unresolved_previous_neighbour'
    assert result['diagnostics']['lost_tracks']['selected']==expected


class NoWorker:
    def process_frame(self,*args,**kwargs):raise AssertionError('No model inference in offline test')


@pytest.mark.parametrize('textured_neighbour',[False,True])
@pytest.mark.parametrize('only_confirmed',[False,True])
def test_confirmed_merge_cannot_bypass_known_or_unresolved_neighbour(textured_neighbour,only_confirmed):
    old,targets,masks=neighbour_scene(textured_neighbour)
    runtime=AsyncPerception(NoWorker(),semantic_interval=float('inf'))
    try:
        runtime.selected_target_id='selected'
        runtime._initialize(old,dict(targets=targets,_masks=masks,diagnostics={}))
        if only_confirmed:
            runtime.tracker.initialize(old,[targets[1]],{'neighbour':masks['neighbour']})
        runtime.last_submit=float('inf')
        for index in (2,3,4):
            current=next_frame(old,frame_id=index,timestamp=10.+.1*(index-1))
            result=runtime.process({'frames':{'head':current}},selected_target_id='selected')
            assert all(target['id']!='selected' for target in result['targets'])
            assert 'selected' not in runtime.tracker.last_masks
            assert 'selected' not in runtime.observations
            assert 'neighbour' in result['diagnostics']['lost_tracks']['selected']
    finally:runtime.close()


def test_confirmed_merge_allows_a_measured_surface_when_known_neighbour_is_far():
    old,targets,masks=neighbour_scene(True,fx=200.)
    runtime=AsyncPerception(NoWorker(),semantic_interval=float('inf'))
    try:
        runtime.selected_target_id='selected'
        runtime._initialize(old,dict(targets=targets,_masks=masks,diagnostics={}))
        runtime.tracker.initialize(old,[targets[1]],{'neighbour':masks['neighbour']})
        runtime.last_submit=float('inf')
        result=runtime.process({'frames':{'head':next_frame(old)}},selected_target_id='selected')
        assert any(target['id']=='selected' for target in result['targets'])
        assert 'selected' in runtime.tracker.last_masks
        assert result['diagnostics']['multiview']['selected_surface_neighbour_gate']['reason']=='known_neighbours_clear'
    finally:runtime.close()


def test_delayed_semantic_rebase_uses_current_surface_and_preserves_source_time():
    old,target,mask=scene();current=next_frame(old)
    runtime=AsyncPerception(NoWorker(),semantic_interval=float('inf'))
    try:
        runtime.selected_target_id='selected'
        initialized=runtime._initialize(old,dict(targets=[target],_masks={'selected':mask},diagnostics={}),
            current_bundle={'frames':{'head':current}})
        assert {'head','confirmed_head'}<=set(initialized)
        target,=runtime.tracker.track(current)['targets']
        assert target['timestamp']==current['timestamp']
        assert target['semantic_keyframe_timestamp']==old['timestamp']
        assert target['perception_method']=='measured_rgbd_surface_reprojection'
        assert runtime.selected_feature_failure['phase']=='semantic_keyframe_rebase'
    finally:runtime.close()


def test_confirmed_surface_measurement_is_exported_with_actual_current_mask():
    old,target,mask=scene();current=next_frame(old)
    runtime=AsyncPerception(NoWorker(),semantic_interval=float('inf'))
    try:
        runtime.selected_target_id='selected'
        runtime._initialize(old,dict(targets=[target],_masks={'selected':mask},diagnostics={}))
        runtime.tracker.initialize(old,[],{})
        runtime.last_submit=float('inf')
        result=runtime.process({'frames':{'head':current}},selected_target_id='selected')
        assert len(result['targets'])==1
        assert result['targets'][0]['timestamp']==current['timestamp']
        assert np.array_equal(runtime.tracker.last_masks['selected'],mask)
        assert runtime.observations['selected']['frame']['frame_id']==current['frame_id']
        assert runtime.selected_tracking_failure is None
    finally:runtime.close()
