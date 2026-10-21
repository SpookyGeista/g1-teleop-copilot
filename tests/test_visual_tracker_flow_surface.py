"""Independent RGBD measurement on selected optical failures; unchanged later gates."""
from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from teleop_assist import surface_tracking
from teleop_assist.visual_tracker import RGBDMaskTracker, VisualTrackerConfig
from teleop_assist.copilot.service import freeze_selected_failure_inputs, write_selected_feature_failure
from test_visual_tracker_surface_integration import setup_tracker, next_frame, neighbour_scene


TRIGGERS=('forward_flow_failed','backward_flow_failed','inconsistent_optical_flow')
POINTS=np.array([[[45.,35.]],[[65.,35.]],[[45.,60.]],[[65.,60.]]],np.float32)


@pytest.fixture(autouse=True)
def recorded_evaluation_clock(monkeypatch):
    original=surface_tracking.measure
    def offline(*args,**kwargs):return original(*args,**kwargs,now=args[3]['timestamp'])
    monkeypatch.setattr(surface_tracking,'measure',offline)


def optical(monkeypatch,trigger,*,filter_failure='forward_status'):
    """Inject native-return shapes to reach one exact branch, not a historical replay."""
    monkeypatch.setattr(cv2,'goodFeaturesToTrack',lambda *a,**k:POINTS.copy())
    calls=[]
    def lk(a,b,points,*args,**kwargs):
        forward=len(calls)%2==0;calls.append(forward)
        if trigger=='forward_flow_failed':return None,None,None
        if trigger=='backward_flow_failed' and not forward:return None,None,None
        out=points.copy();status=np.ones((len(points),1),np.uint8);error=np.zeros((len(points),1),np.float32)
        if trigger=='inconsistent_optical_flow':
            if filter_failure=='forward_status' and forward:status[:]=0
            if filter_failure=='backward_status' and not forward:status[:]=0
            if filter_failure=='forward_backward' and not forward:out[:,:,0]+=2
            if filter_failure=='photometric' and forward:error[:]=40
            if filter_failure=='nonfinite_photometric' and forward:error[:]=np.nan
        return out,status,error
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',lk)
    monkeypatch.setattr(cv2,'estimateAffinePartial2D',lambda *a,**k:(np.array([[1.,0,0],[0,1.,0]]),np.ones((4,1),np.uint8)))
    return calls


@pytest.mark.parametrize('trigger',TRIGGERS)
def test_selected_native_flow_failure_measures_current_depth_and_retains_exact_trigger(monkeypatch,trigger,tmp_path):
    optical(monkeypatch,trigger)
    tracker,old,target,mask=setup_tracker();current=next_frame(old)
    result=tracker.track(current);measured,=result['targets']
    assert measured['perception_method']=='measured_rgbd_surface_reprojection'
    quality=measured['tracking_quality']
    assert quality['optical_failure_trigger']==trigger
    assert 'inlier_count' not in quality and 'flow_residual_px' not in quality
    assert measured['timestamp']==10.1 and measured['semantic_keyframe_timestamp']==10.
    assert measured['position_sigma_m']>=target['position_sigma_m']
    assert measured['pose_confidence']<=target['pose_confidence']
    capture=tracker.last_feature_failure
    assert capture['reason']==capture['tracking_failure_trigger']==trigger
    assert capture['failure_stage']=='optical_correspondence'
    assert capture['schema']=='punktnull.selected_feature_failure.v1'
    assert capture['surface_fallback']['accepted_measurement'] is True
    assert set(capture['arrays'])=={'previous_gray','previous_feature_mask','corner_points','previous_rgb','previous_depth'}
    np.testing.assert_array_equal(capture['arrays']['corner_points'],POINTS)
    np.testing.assert_array_equal(capture['arrays']['previous_rgb'],old['rgb'])
    np.testing.assert_array_equal(capture['arrays']['previous_depth'],old['depth'])
    np.testing.assert_array_equal(capture['arrays']['previous_feature_mask'],mask.astype(np.uint8)*255)
    frozen,arrays,metadata,_=freeze_selected_failure_inputs(current,capture)
    destination=tmp_path/'exact_capture'
    write_selected_feature_failure(destination,frozen,arrays,metadata,{'reason':trigger,'motion_authorized':False})
    with np.load(destination/'tracking_inputs.npz',allow_pickle=False) as archived:
        restored=json.loads(str(archived['metadata_json']))
        assert restored['reason']==trigger and restored['optical_flow']==capture['optical_flow']
        np.testing.assert_array_equal(archived['previous_depth'],old['depth'])
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('failure',['forward_status','backward_status','forward_backward','photometric','nonfinite_photometric'])
def test_exact_bounded_flow_statistics_identify_native_filter_failure(monkeypatch,failure):
    optical(monkeypatch,'inconsistent_optical_flow',filter_failure=failure)
    tracker,old,_,_=setup_tracker();current=next_frame(old)
    assert tracker.track(current)['targets']
    flow=tracker.last_feature_failure['optical_flow']
    assert flow['usable']['positive_count']==0
    assert flow['forward_points']['values']==POINTS.reshape(-1).astype(float).tolist()
    if failure=='forward_status':assert flow['forward_status']['positive_count']==0
    if failure=='backward_status':assert flow['backward_status']['positive_count']==0
    if failure=='forward_backward':assert flow['forward_backward_error_px']['median']==2.
    if failure=='photometric':assert flow['photometric_error']['median']==40.
    if failure=='nonfinite_photometric':
        assert flow['photometric_error']['nonfinite_count']==4
        assert flow['photometric_error']['values']==[None]*4
    freeze_selected_failure_inputs(current,tracker.last_feature_failure)


@pytest.mark.parametrize('trigger',TRIGGERS)
@pytest.mark.parametrize('kind',['depth','colour','semantic','clock','calibration','frame_gap'])
def test_new_triggers_do_not_bypass_current_measurement_or_source_leases(monkeypatch,trigger,kind):
    optical(monkeypatch,trigger)
    tracker,old,_,mask=setup_tracker();current=next_frame(old)
    if kind=='depth':current['depth'][mask]=.7
    if kind=='colour':current['rgb'][mask]=0
    if kind=='semantic':
        tracker._tracks['selected']['semantic_keyframe_timestamp']=9.
        tracker._tracks['selected']['target']['semantic_keyframe_timestamp']=9.
    if kind=='clock':current['clock_id']='wrong'
    if kind=='calibration':current['calibration_id']='wrong'
    if kind=='frame_gap':current['timestamp']=10.7
    assert tracker.track(current)['targets']==[]
    assert tracker.track(next_frame(old,frame_id=3,timestamp=10.8))['targets']==[]


@pytest.mark.parametrize('trigger',TRIGGERS)
@pytest.mark.parametrize('selected,enabled',[(None,True),('different',True),('selected',False)])
def test_only_explicit_selected_surface_can_measure(monkeypatch,trigger,selected,enabled):
    optical(monkeypatch,trigger)
    tracker,old,_,_=setup_tracker(selected=selected,enabled=enabled)
    def forbidden(*a,**k):raise AssertionError('Unselected measurement attempted')
    monkeypatch.setattr(surface_tracking,'measure',forbidden)
    result=tracker.track(next_frame(old))
    assert result['targets']==[] and result['diagnostics']['lost_tracks']['selected']==trigger


@pytest.mark.parametrize('trigger',TRIGGERS)
def test_new_trigger_does_not_erase_unresolved_close_neighbour(monkeypatch,trigger):
    optical(monkeypatch,trigger)
    old,targets,masks=neighbour_scene(False)
    tracker=RGBDMaskTracker();tracker.set_surface_target('selected',camera_role='head')
    tracker.initialize(old,targets,masks)
    result=tracker.track(next_frame(old))
    assert all(t['id']!='selected' for t in result['targets'])
    assert result['diagnostics']['lost_tracks']['selected']=='unresolved_previous_neighbour'


@pytest.mark.parametrize('trigger',TRIGGERS)
@pytest.mark.parametrize('gate',['world','normal'])
def test_surface_output_must_still_pass_existing_strict_geometry_gate(monkeypatch,trigger,gate):
    optical(monkeypatch,trigger)
    config=(VisualTrackerConfig(world_motion_slack_m=0.,max_world_speed_m_s=0.) if gate=='world'
            else VisualTrackerConfig(max_normal_angle_deg=.01))
    tracker,old,_,mask=setup_tracker(config=config);current=next_frame(old)
    if gate=='world':current['depth'][mask]+=.001
    else:current['depth'][30:70,40:80]+=np.linspace(-.004,.004,40)[None,:]
    result=tracker.track(current)
    assert result['targets']==[]
    assert result['diagnostics']['lost_tracks']['selected']==('inconsistent_measured_world_motion' if gate=='world' else 'inconsistent_surface_normal')
    assert tracker.last_feature_failure['surface_fallback']['accepted_measurement'] is False


@pytest.mark.parametrize('failure',['ransac','view','depth','geometry'])
def test_later_native_gates_cannot_invoke_surface_fallback(monkeypatch,failure):
    optical(monkeypatch,None)
    config=VisualTrackerConfig(world_motion_slack_m=0.,max_world_speed_m_s=0.)
    tracker,old,_,mask=setup_tracker(config=config);current=next_frame(old)
    def forbidden(*a,**k):raise AssertionError('Later failure incorrectly invoked fallback')
    monkeypatch.setattr(tracker,'_optical_surface_fallback',forbidden)
    if failure=='ransac':monkeypatch.setattr(cv2,'estimateAffinePartial2D',lambda *a,**k:(None,None))
    if failure=='view':monkeypatch.setattr(cv2,'estimateAffinePartial2D',lambda *a,**k:(np.array([[2.,0,0],[0,2.,0]]),np.ones((4,1),np.uint8)))
    if failure=='depth':current['depth'][mask]=np.nan
    if failure=='geometry':current['depth'][mask]+=.001
    result=tracker.track(current)
    assert result['targets']==[]
    expected={'ransac':'insufficient_ransac_consensus','view':'excessive_view_change','depth':'insufficient_mask_depth','geometry':'inconsistent_measured_world_motion'}
    assert result['diagnostics']['lost_tracks']['selected']==expected[failure]
    assert tracker.last_feature_failure is None


def test_diagnostic_failure_does_not_authorize_or_block_independent_measurement(monkeypatch):
    optical(monkeypatch,'forward_flow_failed')
    tracker,old,_,_=setup_tracker()
    def failing(*a,**k):raise MemoryError('bounded diagnostic fixture')
    monkeypatch.setattr(tracker,'_capture_feature_failure',failing)
    assert tracker.track(next_frame(old))['targets']
    assert tracker.last_feature_failure['capture_status']=='capture_failed'


def test_optical_diagnostic_vector_limit_does_not_allocate_extra_arrays(monkeypatch):
    optical(monkeypatch,'forward_flow_failed')
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',lambda *a,**k:(None,None,np.zeros(81,np.float32)))
    tracker,old,_,_=setup_tracker()
    assert tracker.track(next_frame(old))['targets']
    capture=tracker.last_feature_failure
    assert capture['capture_status']=='capture_failed' and 'arrays' not in capture
    assert capture['surface_fallback']['accepted_measurement'] is True
