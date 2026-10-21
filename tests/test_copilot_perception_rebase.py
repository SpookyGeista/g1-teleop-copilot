"""Real CPU LK/RGB-D controls for delayed semantic replacement; no model/GPU."""
import copy
from concurrent.futures import Future
from dataclasses import replace

import cv2
import numpy as np
import pytest

from teleop_assist.learned_perception import mask_to_target
from teleop_assist.perception_runtime import AsyncPerception


def image(timestamp=1., frame_id=10, *, shift=0, camera_x=0.):
    rgb=np.zeros((100,120,3),np.uint8)
    rgb[30:70,40:80]=np.random.default_rng(41).integers(20,220,(40,40,3),dtype=np.uint8)
    mask=np.zeros((100,120),bool);mask[30:70,40:80]=True
    if shift:
        transform=np.float32([[1,0,shift],[0,1,0]])
        rgb=cv2.warpAffine(rgb,transform,(120,100))
        mask=cv2.warpAffine(mask.astype(np.uint8),transform,(120,100)).astype(bool)
    camera=np.eye(4);camera[0,3]=camera_x
    frame={'rgb':rgb,'depth':np.ones((100,120),np.float32),
        'intrinsics':{'fx':200.,'fy':200.,'cx':60.,'cy':50.},'T_world_camera':camera,
        'timestamp':timestamp,'frame_id':frame_id}
    target,reason=mask_to_target(mask,frame['depth'],frame['intrinsics'],camera,
        label='red cylinder',detection_score=.8,mask_score=.95)
    assert reason is None
    target.update(id='learned_001',frame_id=frame_id,timestamp=timestamp)
    return frame,target,mask


def semantic(target,mask):
    return {'targets':[copy.deepcopy(target)],'_masks':{target['id']:mask.copy()},'diagnostics':{}}


class NoModel:
    def process_frame(self,*args):
        raise AssertionError('No learned model call in this component test')


@pytest.fixture
def runtime():
    value=AsyncPerception(NoModel(),semantic_interval=float('inf'))
    value.last_submit=0.
    try:yield value
    finally:value.close()


def complete_delayed(runtime,source,target,mask,current):
    runtime.source={'frames':{'head':source}}
    runtime.future=Future();runtime.future.set_result(semantic(target,mask))
    return runtime.process({'frames':{'head':current}})


def test_delayed_generic_rebase_preserves_current_measured_track_without_confirmation(runtime):
    recent,target,mask=image(1.9,19)
    runtime._initialize(recent,semantic(target,mask));previous=runtime.tracker
    delayed,proposal,mask=image(1.2,12)
    current,_,_=image(2.,20)
    result=complete_delayed(runtime,delayed,proposal,mask,current)
    assert runtime.selected_target_id is None
    assert runtime.tracker is previous
    assert [t['id'] for t in result['targets']]==['learned_001']
    assert result['targets'][0]['frame_id']==20
    assert result['targets'][0]['timestamp']==2.
    assert result['targets'][0]['semantic_keyframe_timestamp']==1.9
    diagnostic=result['diagnostics']['semantic_keyframe']['direct_current_frame_rebase']['head']
    assert diagnostic['adopted'] is False
    assert diagnostic['reason']=='expired_or_nonadvancing_tracking_frame'
    assert diagnostic['active_semantic_keyframe_timestamp']==1.9


@pytest.mark.parametrize('loss',['texture','missing_depth','depth_occluder','camera_jump'])
def test_rejected_candidate_cannot_reuse_old_target_without_current_support(runtime,loss):
    recent,target,mask=image(1.9,19)
    runtime._initialize(recent,semantic(target,mask));previous=runtime.tracker
    delayed,proposal,mask=image(1.2,12)
    current,_,_=image(2.,20)
    if loss=='texture':current['rgb']=np.zeros_like(current['rgb'])
    elif loss=='missing_depth':current['depth'][:]=0.
    elif loss=='camera_jump':current['T_world_camera'][0,3]=2.
    else:current['depth'][mask]=.7
    result=complete_delayed(runtime,delayed,proposal,mask,current)
    assert runtime.tracker is previous
    assert result['targets']==[]
    assert result['diagnostics']['lost_tracks']
    assert runtime.tracker.last_masks=={}


def test_rejected_rebase_does_not_refresh_original_semantic_expiry(runtime):
    source,target,mask=image(1.,10)
    runtime._initialize(source,semantic(target,mask))
    for stamp,index in ((1.4,14),(1.8,18)):
        current,_,_=image(stamp,index)
        assert runtime.process({'frames':{'head':current}})['targets']
    delayed,proposal,mask=image(1.2,12)
    current,_,_=image(1.9,19)
    result=complete_delayed(runtime,delayed,proposal,mask,current)
    assert result['targets'][0]['semantic_keyframe_timestamp']==1.
    expired,_,_=image(2.01,21)
    result=runtime.process({'frames':{'head':expired}})
    assert result['targets']==[]
    assert result['diagnostics']['lost_tracks']['learned_001']=='expired_semantic_and_cad_identity_evidence'


def test_accepted_rebase_uses_current_rgbd_and_original_semantic_time(runtime):
    source,target,mask=image(1.,10)
    runtime._initialize(source,semantic(target,mask));previous=runtime.tracker
    delayed,proposal,mask=image(1.2,12)
    # A 4-pixel image motion and matching measured camera translation leave
    # the observed world surface stationary; real LK and RGB-D must agree.
    current,_,_=image(1.4,14,shift=4,camera_x=-.02)
    result=complete_delayed(runtime,delayed,proposal,mask,current)
    assert runtime.tracker is not previous
    assert len(result['targets'])==1
    observed=result['targets'][0]
    assert observed['id']=='learned_001' and observed['frame_id']==14
    assert observed['timestamp']==1.4 and observed['semantic_keyframe_timestamp']==1.2
    assert np.linalg.norm(np.asarray(observed['center_world'])-proposal['center_world'])<.008
    assert result['diagnostics']['semantic_keyframe']['direct_current_frame_rebase']['head']['adopted']


def test_candidate_inherits_flow_and_geometry_configuration(runtime):
    runtime.tracker.config=replace(runtime.tracker.config,max_frame_gap_s=.17,max_flow_points=17)
    runtime.tracker.geometry_config=replace(runtime.tracker.geometry_config,max_depth_m=.9)
    source,target,mask=image(1.,10)
    runtime._initialize(source,semantic(target,mask))
    flow_config=runtime.tracker.config;geometry_config=runtime.tracker.geometry_config
    delayed,proposal,mask=image(1.1,11)
    current,_,_=image(1.2,12)
    result=complete_delayed(runtime,delayed,proposal,mask,current)
    # Candidate and retained tracker both reject actual 1-m depth against .9.
    assert result['targets']==[]
    assert runtime.tracker.config is flow_config
    assert runtime.tracker.geometry_config is geometry_config
    assert runtime.last_model_diagnostics['direct_current_frame_rebase']['head']['adopted'] is False


def test_successful_candidate_preserves_both_configurations(runtime):
    runtime.tracker.config=replace(runtime.tracker.config,max_flow_points=17)
    runtime.tracker.geometry_config=replace(runtime.tracker.geometry_config,max_mask_fraction=.8)
    config=runtime.tracker.config;geometry=runtime.tracker.geometry_config
    source,target,mask=image(1.,10)
    runtime._initialize(source,semantic(target,mask))
    delayed,proposal,mask=image(1.1,11)
    current,_,_=image(1.2,12)
    assert complete_delayed(runtime,delayed,proposal,mask,current)['targets']
    assert runtime.tracker.config is config and runtime.tracker.geometry_config is geometry


def test_initial_and_same_current_frame_empty_semantics_clear_generic_chooser(runtime):
    frame,target,mask=image()
    runtime._initialize(frame,{'targets':[],'_masks':{},'diagnostics':{}})
    assert runtime.initialized and runtime.tracker.track(frame)['targets']==[]
    assert runtime.last_model_diagnostics['direct_current_frame_rebase']['head']['adoption_reason']=='current_empty_semantic_keyframe'
    runtime._initialize(frame,semantic(target,mask))
    current,_,_=image(1.1,11)
    runtime._initialize(current,{'targets':[],'_masks':{},'diagnostics':{}})
    assert runtime.tracker.track(current)['targets']==[]


@pytest.mark.parametrize('source_time,source_frame',[(1.2,12),(1.9,19)])
def test_empty_delayed_result_retains_track_only_until_current_measurement(runtime,source_time,source_frame):
    frame,target,mask=image(1.9,19)
    runtime._initialize(frame,semantic(target,mask));previous=runtime.tracker
    delayed,_,_=image(source_time,source_frame)
    current,_,_=image(2.,20)
    runtime._initialize(delayed,{'targets':[],'_masks':{},'diagnostics':{}},current_bundle={'frames':{'head':current}})
    assert runtime.tracker is previous
    assert runtime.tracker.track(current)['targets'][0]['semantic_keyframe_timestamp']==1.9


def test_new_semantic_identity_is_not_relabelled_as_selected_identity(runtime):
    source,target,mask=image(1.,10)
    runtime.selected_target_id='selected_missing'
    runtime._initialize(source,semantic(target,mask))
    current,_,_=image(1.1,11)
    result=runtime.process({'frames':{'head':current}},selected_target_id='selected_missing')
    assert runtime.selected_target_id=='selected_missing'
    assert [t['id'] for t in result['targets']]==['learned_001']
    assert 'selected_missing' not in runtime.observations


def test_direct_future_semantic_candidate_does_not_replace_current_tracker(runtime):
    recent,target,mask=image(1.9,19)
    runtime._initialize(recent,semantic(target,mask));previous=runtime.tracker
    future,proposal,mask=image(2.1,21)
    current,_,_=image(2.,20)
    runtime._initialize(future,semantic(proposal,mask),current_bundle={'frames':{'head':current}})
    assert runtime.tracker is previous
    assert runtime.tracker.track(current)['targets'][0]['semantic_keyframe_timestamp']==1.9
