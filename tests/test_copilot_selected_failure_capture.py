"""Bounded diagnostic data copies; no model inference, browser, GPU or motion."""
from copy import deepcopy
import hashlib
import json
import threading
import time

import cv2
import numpy as np
import pytest

from teleop_assist.visual_tracker import RGBDMaskTracker
from teleop_assist.perception_runtime import AsyncPerception
from teleop_assist.copilot.mailbox import atomic_json,read_json
from teleop_assist.copilot.service import ProposalSession,freeze_selected_failure_inputs
from test_visual_tracker import frames
from test_copilot_service import Engine,frame,target,masks


def feature_failure(monkeypatch, *, points=None, enabled=True):
    old,t,mask=frames()
    tracker=RGBDMaskTracker()
    tracker.set_diagnostic_target(t['id'] if enabled else None)
    tracker.initialize(old,[t],{t['id']:mask})
    observed={}
    def features(gray,**kwargs):
        observed['gray']=gray.copy();observed['mask']=kwargs['mask'].copy()
        observed['kwargs']={key:value for key,value in kwargs.items() if key!='mask'}
        return points
    monkeypatch.setattr(cv2,'goodFeaturesToTrack',features)
    current={**old,'rgb':np.full_like(old['rgb'],231),'timestamp':1.1,'frame_id':2}
    result=tracker.track(current)
    return tracker,old,current,t,mask,observed,result


@pytest.mark.parametrize('points',[None,np.array([[[12.25,13.5]],[[33.75,41.25]]],np.float32)])
def test_exact_native_feature_inputs_are_copied_before_tracker_advances(monkeypatch,points):
    tracker,old,current,t,mask,observed,result=feature_failure(monkeypatch,points=points)
    capture=tracker.last_feature_failure
    assert result['targets']==[] and result['diagnostics']['lost_tracks'][t['id']]=='insufficient_visual_features'
    assert capture['points_were_none']==(points is None)
    assert capture['point_count']==(0 if points is None else 2)
    assert np.array_equal(capture['arrays']['previous_gray'],observed['gray'])
    assert np.array_equal(capture['arrays']['previous_feature_mask'],observed['mask'])
    assert capture['previous_frame']['frame_id']==1 and capture['current_frame_id']==2
    assert capture['previous_frame']['timestamp']==1. and capture['current_timestamp']==1.1
    original_target=deepcopy(t)
    t['center_world'][0]=55.
    np.testing.assert_array_equal(capture['previous_target']['center_world'],original_target['center_world'])
    assert capture['library_versions']==dict(opencv=cv2.__version__,numpy=np.__version__)
    assert capture['tracker_config']['min_flow_points']==3
    assert capture['geometry_config']['min_mask_pixels']==24
    if points is not None:
        expected=points.copy();points[:]=999.
        assert np.array_equal(capture['arrays']['corner_points'],expected)
    old['T_world_camera'][0,3]=8.;mask[:]=False;old['rgb'][:]=0
    assert capture['previous_frame']['T_world_camera'][0,3]==0.
    assert np.array_equal(capture['arrays']['previous_gray'],observed['gray'])
    assert np.array_equal(capture['arrays']['previous_feature_mask'],observed['mask'])
    assert np.all(tracker._gray==231)  # Failure archive really contains older gray.
    json.dumps(result,default=lambda v:v.tolist())  # No arrays inserted in diagnostics.
    assert not any(key in result['diagnostics'] for key in ('arrays','previous_gray','previous_feature_mask'))


def test_native_untextured_previous_image_records_none_without_lowering_thresholds():
    old,t,mask=frames();old['rgb'][:]=95
    tracker=RGBDMaskTracker();tracker.set_diagnostic_target(t['id'])
    tracker.initialize(old,[t],{t['id']:mask})
    result=tracker.track({**old,'timestamp':1.1,'frame_id':2})
    assert result['diagnostics']['lost_tracks'][t['id']]=='insufficient_visual_features'
    assert tracker.last_feature_failure['points_were_none'] is True
    assert tracker.last_feature_failure['feature_parameters']['min_flow_points']==3


def test_unselected_failure_does_not_copy_images(monkeypatch):
    tracker,*_,result=feature_failure(monkeypatch,enabled=False)
    assert tracker.last_feature_failure is None and result['targets']==[]


def test_same_frame_preserves_actual_capture_but_new_frame_or_selection_clears_it(monkeypatch):
    tracker,_,current,*_=feature_failure(monkeypatch)
    original=tracker.last_feature_failure
    tracker.track(current)
    assert tracker.last_feature_failure is original
    tracker.track({**current,'timestamp':1.2,'frame_id':3})
    assert tracker.last_feature_failure is None
    tracker.last_feature_failure=original;tracker.set_diagnostic_target('another')
    assert tracker.last_feature_failure is None


def test_tracker_oversize_capture_does_not_change_loss_decision():
    tracker=RGBDMaskTracker();tracker.set_diagnostic_target('object')
    tracker._gray=np.zeros((2049,2049),np.uint8)
    tracker._frame=dict(frame_id=1,timestamp=1.,intrinsics=np.eye(3),T_world_camera=np.eye(4))
    tracker._capture_feature_failure(dict(frame_id=2,timestamp=1.1),{'id':'object'},
                                     np.zeros_like(tracker._gray),None)
    assert tracker.last_feature_failure['capture_status']=='size_or_dtype_limit'
    assert 'arrays' not in tracker.last_feature_failure


def test_oversize_previous_target_metadata_is_not_copied(monkeypatch):
    old,t,mask=frames();old['rgb'][:]=80;t['unexpected_large_metadata']='x'*(64*1024)
    tracker=RGBDMaskTracker();tracker.set_diagnostic_target(t['id']);tracker.initialize(old,[t],{t['id']:mask})
    result=tracker.track({**old,'timestamp':1.1,'frame_id':2})
    assert result['diagnostics']['lost_tracks'][t['id']]=='insufficient_visual_features'
    assert tracker.last_feature_failure['capture_status']=='capture_failed'
    assert 'previous_target' not in tracker.last_feature_failure and 'arrays' not in tracker.last_feature_failure


def test_perception_keeps_failure_arrays_private_and_only_for_final_selected_loss():
    class NoModel:
        def process_frame(self,*args):raise AssertionError('No inference in this test')
    old,t,mask=frames();old['rgb'][:]=80
    runtime=AsyncPerception(NoModel(),semantic_interval=float('inf'))
    try:
        runtime.selected_target_id=t['id']
        runtime._initialize(old,dict(targets=[t],_masks={t['id']:mask},diagnostics={}))
        runtime.last_submit=time.monotonic()  # inf-inf would otherwise schedule the test stub.
        result=runtime.process({'frames':{'head':{**old,'timestamp':1.1,'frame_id':2}}},t['id'])
        capture=runtime.selected_tracking_failure
        assert capture['target_id']==t['id'] and capture['current_frame_id']==2
        assert capture['camera_role']=='head' and capture['point_count']==0
        assert result['targets']==[]
        json.dumps(result)  # Private arrays are not in the result/RPC JSON.
        result=runtime.process({'frames':{'head':{**old,'timestamp':1.2,'frame_id':3}}},'different')
        assert runtime.selected_tracking_failure is None
    finally:runtime.close()


def diagnostic_capture(current,*,target_id='object'):
    return dict(schema='punktnull.selected_feature_failure.v1',target_id=target_id,
        current_frame_id=current['frame_id'],current_timestamp=current['timestamp'],
        reason='insufficient_visual_features',capture_status='captured',points_were_none=False,point_count=2,
        previous_frame=dict(frame_id=current['frame_id']-1,timestamp=current['timestamp']-.05,
                            T_world_camera=np.eye(4)),
        arrays=dict(previous_gray=np.full((24,24),47,np.uint8),
            previous_feature_mask=np.full((24,24),255,np.uint8),
            corner_points=np.array([[[4.5,8.25]],[[12.75,19.5]]],np.float32)))


def test_warmup_then_selected_loss_keeps_both_archives_exact_once_and_off_thread(tmp_path,monkeypatch):
    from teleop_assist.copilot import service as module
    started,finish=threading.Event(),threading.Event()
    writer=module.write_selected_target_failure;threads=[]
    def blocked(*args):
        threads.append(threading.current_thread().name)
        started.set();assert finish.wait(3);return writer(*args)
    monkeypatch.setattr(module,'write_selected_target_failure',blocked)
    clock=[10.];engine=Engine();engine.finish.set()
    session=ProposalSession(tmp_path,engine,clock=lambda:clock[0],clock_id='clock')
    try:
        session.update(frame(1,9.),[],{})  # Unselected stale startup frame.
        session.failure_audit.result(2)
        warmup=(tmp_path/'first_failure/frame.npz').read_bytes()
        assert session.first_selected_target_failure is None
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        session.update(frame(2,10.),target(2,10.),masks())
        if session.future is not None:session.future.result(2)
        session.poll()
        current=frame(3,10.05);current['rgb'][:]=73;clock[0]=10.05
        capture=diagnostic_capture(current);original=deepcopy(capture)
        session.update(current,[],{},tracking_failure=capture)
        assert started.wait(1) and not session.selected_failure_audit.done()
        assert session.error=='selected_object_not_observed'
        assert not read_json(tmp_path/'latest_proposals.json')['valid']
        assert not read_json(tmp_path/'latest_proposals.json')['motion_static']
        small=read_json(tmp_path/'ui_state.json')['first_selected_target_failure']
        assert small['replay_status']=='pending' and 'arrays' not in small
        current['rgb'][:]=199;current['depth'][:]=3.
        for value in capture['arrays'].values():value[:]=0
        capture['previous_frame']['T_world_camera'][0,3]=8.
        finish.set();session.selected_failure_audit.result(2);session.publish()
        directory=tmp_path/'first_selected_target_failure'
        with np.load(directory/'frame.npz',allow_pickle=False) as saved:
            assert int(saved['frame_id'])==3 and float(saved['timestamp'])==10.05
            assert np.all(saved['rgb']==73) and np.all(saved['depth']==1.)
        with np.load(directory/'tracking_inputs.npz',allow_pickle=False) as saved:
            for key,value in original['arrays'].items():assert np.array_equal(saved[key],value)
            metadata=json.loads(str(saved['metadata_json'].item()))
            assert metadata['previous_frame']['T_world_camera'][0][3]==0.
            assert metadata['point_count']==2 and not metadata['points_were_none']
        completed=read_json(tmp_path/'ui_state.json')['first_selected_target_failure']
        assert completed['replay_status']=='complete' and completed['selected_target_id']=='object'
        for name,info in completed['files'].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==info['sha256']
        first=(directory/'evidence.json').read_bytes()
        atomic_json(tmp_path/'target_selection.json',{'target_id':'other','revision':2})
        session.update(frame(4,10.06),[],{})
        assert session.first_selected_target_failure['selected_target_id']=='object'
        assert (directory/'evidence.json').read_bytes()==first
        assert (tmp_path/'first_failure/frame.npz').read_bytes()==warmup
        assert len(threads)==1 and threads[0].startswith('copilot-failure-audit')
    finally:finish.set();session.close()


@pytest.mark.parametrize('mismatch',['target','frame','timestamp'])
def test_changed_selection_or_frame_cannot_mislabel_old_tracking_inputs(tmp_path,mismatch):
    engine=Engine();engine.finish.set();session=ProposalSession(tmp_path,engine,clock=lambda:10.,clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        current=frame();capture=diagnostic_capture(current)
        if mismatch=='target':capture['target_id']='previous_object'
        elif mismatch=='frame':capture['current_frame_id']=4
        else:capture['current_timestamp']=9.9
        session.update(current,[],{},tracking_failure=capture)
        session.selected_failure_audit.result(2);session.publish()
        with np.load(tmp_path/'first_selected_target_failure/tracking_inputs.npz',allow_pickle=False) as saved:
            assert saved.files==['metadata_json'] and json.loads(str(saved['metadata_json'].item())) is None
        assert session.first_selected_target_failure['tracking_inputs_status']=='unavailable_for_this_target_and_frame'
    finally:session.close()


def test_existing_archive_never_overwritten_and_recording_error_does_not_change_control(tmp_path):
    directory=tmp_path/'first_selected_target_failure';directory.mkdir();(directory/'keep.txt').write_text('old')
    engine=Engine();engine.finish.set();session=ProposalSession(tmp_path,engine,clock=lambda:10.,clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        session.update(frame(),[],{})
        with pytest.raises(FileExistsError):session.selected_failure_audit.result(2)
        session.publish()
        assert session.error=='selected_object_not_observed'
        assert session.first_selected_target_failure['replay_status']=='failed'
        assert session.first_selected_target_failure['replay_error']=='FileExistsError'
        assert (directory/'keep.txt').read_text()=='old' and len(list(directory.iterdir()))==1
    finally:session.close()


def test_selected_camera_error_does_not_consume_actual_observation_loss_slot(tmp_path):
    engine=Engine();engine.finish.set();clock=[10.]
    session=ProposalSession(tmp_path,engine,clock=lambda:clock[0],clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        session.update(frame(1,9.),[],{})
        assert session.error=='camera_stream_expired'  # Existing poll() adds this after the initial stale-frame gate.
        assert session.first_selected_target_failure is None and session.selected_failure_audit is None
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':2})
        session.update(frame(2,10.),target(2,10.),masks())
        if session.future is not None:session.future.result(2)
        session.poll();clock[0]=10.05
        current=frame(3,10.05)
        session.update(current,[],{},tracking_failure=diagnostic_capture(current))
        session.selected_failure_audit.result(2);session.publish()
        assert session.first_selected_target_failure['frame_id']==3
        assert session.first_selected_target_failure['selection_revision']==2
        assert session.first_selected_target_failure['replay_status']=='complete'
    finally:session.close()


def test_latched_camera_error_without_reselection_cannot_block_current_loss_capture(tmp_path):
    engine=Engine();engine.finish.set();clock=[10.]
    session=ProposalSession(tmp_path,engine,clock=lambda:clock[0],clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        session.update(frame(1,9.),[],{})
        assert session.error and session.first_selected_target_failure is None
        first_reason=session.first_failure['reason'];first_frame=session.first_failure['frame_id']
        current=frame(2,10.);session.update(current,[],{},tracking_failure=diagnostic_capture(current))
        session.selected_failure_audit.result(2);session.publish()
        assert session.first_selected_target_failure['frame_id']==2
        assert session.first_selected_target_failure['selection_revision']==1
        assert session.first_selected_target_failure['tracking_inputs_status']=='captured'
        assert session.first_failure['frame_id']==first_frame and session.first_failure['reason']==first_reason
        assert session.error=='selected_object_not_observed' and session.batch is None and session.future is None
        assert not read_json(tmp_path/'latest_proposals.json')['valid']
    finally:session.close()


def test_over_budget_replay_is_refused_without_second_attempt_or_permission_change(tmp_path,monkeypatch):
    from teleop_assist.copilot import service as module
    monkeypatch.setattr(module,'SELECTED_REPLAY_MAX_BYTES',100)
    engine=Engine();engine.finish.set();session=ProposalSession(tmp_path,engine,clock=lambda:10.,clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        session.update(frame(),[],{})
        assert session.selected_failure_audit is None
        assert session.first_selected_target_failure['replay_status']=='failed'
        assert not (tmp_path/'first_selected_target_failure').exists()
        monkeypatch.setattr(module,'SELECTED_REPLAY_MAX_BYTES',16*1024*1024)
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':2})
        session.update(frame(1,10.01),[],{})
        assert session.selected_failure_audit is None
        assert session.error=='selected_object_not_observed'
    finally:session.close()


@pytest.mark.parametrize('bad', [np.array([object()],object),{'opaque':object()}])
def test_pickle_or_unserializable_metadata_is_rejected(bad):
    value=frame();value['extra']=bad
    with pytest.raises((TypeError,ValueError)):freeze_selected_failure_inputs(value,None)


def test_dict_calibration_is_preserved_as_explicit_json_not_pickled():
    value=frame();value['intrinsics']={'fx':300.,'fy':300.,'cx':12.,'cy':12.}
    copied,_,_,_=freeze_selected_failure_inputs(value,None)
    assert json.loads(str(copied['_nonarray_metadata_json'].item()))['intrinsics']==value['intrinsics']
    assert not any(array.dtype.hasobject for array in copied.values())


def test_failed_diagnostic_allocation_never_changes_native_tracking_loss(monkeypatch):
    old,t,mask=frames();old['rgb'][:]=80
    tracker=RGBDMaskTracker();tracker.set_diagnostic_target(t['id']);tracker.initialize(old,[t],{t['id']:mask})
    def broken(*args):raise MemoryError('synthetic diagnostic failure')
    monkeypatch.setattr(tracker,'_capture_feature_failure',broken)
    result=tracker.track({**old,'frame_id':2,'timestamp':1.1})
    assert result['diagnostics']['lost_tracks'][t['id']]=='insufficient_visual_features'
    assert result['targets']==[] and tracker.last_feature_failure['capture_status']=='capture_failed'


def test_close_waits_for_owned_selected_audit_and_publishes_completion(tmp_path,monkeypatch):
    from teleop_assist.copilot import service as module
    started,finish,closed=threading.Event(),threading.Event(),threading.Event()
    writer=module.write_selected_target_failure
    def blocked(*args):
        started.set();assert finish.wait(3);return writer(*args)
    monkeypatch.setattr(module,'write_selected_target_failure',blocked)
    engine=Engine();engine.finish.set();session=ProposalSession(tmp_path,engine,clock=lambda:10.,clock_id='clock')
    atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
    session.update(frame(),[],{})
    assert started.wait(1)
    thread=threading.Thread(target=lambda:(session.close(),closed.set()),daemon=True)
    thread.start()
    try:
        assert not closed.wait(.03)
        finish.set();thread.join(2)
        assert closed.is_set()
        assert read_json(tmp_path/'ui_state.json')['first_selected_target_failure']['replay_status']=='complete'
    finally:finish.set();thread.join(3)
