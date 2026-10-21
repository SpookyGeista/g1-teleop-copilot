"""First selected GFTT evidence, including discarded semantic rebases; CPU only."""
from concurrent.futures import Future
from copy import deepcopy
import hashlib
import json
import threading

import cv2
import numpy as np
import pytest

from teleop_assist.perception_runtime import AsyncPerception
from teleop_assist.copilot.mailbox import atomic_json, read_json
from teleop_assist.copilot.service import ProposalSession
from test_copilot_perception_rebase import NoModel, image, semantic
from test_copilot_selected_failure_capture import diagnostic_capture
from test_copilot_service import Engine, frame, target, masks


def feature_capture(current, **changes):
    return dict(diagnostic_capture(current), phase='semantic_keyframe_rebase',
                tracker_stage='head_rebase', **changes)


def test_real_blank_rebase_is_archived_before_discard_while_old_target_still_observed():
    runtime=AsyncPerception(NoModel(),semantic_interval=float('inf'));runtime.last_submit=0.
    try:
        recent,t,mask=image(1.9,19);runtime.selected_target_id=t['id']
        runtime._initialize(recent,semantic(t,mask))
        previous,confirmed=runtime.tracker,runtime.confirmed_head_tracker
        delayed,proposal,_=image(1.8,18);delayed['rgb'][:]=81
        current,_,_=image(2.,20)
        runtime.source={'frames':{'head':delayed}}
        runtime.future=Future();runtime.future.set_result(semantic(proposal,mask))
        result=runtime.process({'frames':{'head':current}},t['id'])
        capture=runtime.selected_feature_failure
        assert runtime.tracker is previous and runtime.confirmed_head_tracker is confirmed
        assert [v['id'] for v in result['targets']]==[t['id']]
        assert result['targets'][0]['semantic_keyframe_timestamp']==1.9
        assert runtime.selected_tracking_failure is None  # Existing actual-loss diagnostic stays separate.
        assert capture['phase']=='semantic_keyframe_rebase' and capture['tracker_stage']=='head_rebase'
        assert capture['previous_frame']['frame_id']==18 and capture['previous_frame']['timestamp']==1.8
        assert capture['current_frame_id']==20 and capture['current_timestamp']==2.
        assert capture['previous_target']['id']==t['id']
        assert capture['selected_target_observed_after_processing'] is True
        assert capture['points_were_none'] and capture['point_count']==0
        assert np.all(capture['arrays']['previous_gray']==81)
        assert np.array_equal(capture['arrays']['previous_feature_mask'],mask.astype(np.uint8)*255)
        assert capture['arrays']['corner_points'].shape==(0,1,2)
        delayed['rgb'][:]=0;mask[:]=False
        assert np.all(capture['arrays']['previous_gray']==81)
        json.dumps(result)  # No arrays/extra permissions placed in result JSON.
        later,_,_=image(2.05,21)
        assert runtime.process({'frames':{'head':later}},t['id'])['targets']
        assert runtime.selected_feature_failure is None  # Never label old evidence as a later frame.
    finally:runtime.close()


@pytest.mark.parametrize('failed_call,stage',[(1,'head_rebase'),(2,'confirmed_head_rebase')])
def test_each_candidate_is_observed_before_overwrite(monkeypatch,failed_call,stage):
    runtime=AsyncPerception(NoModel(),semantic_interval=float('inf'));runtime.last_submit=0.
    real=cv2.goodFeaturesToTrack;calls=[]
    try:
        recent,t,mask=image(1.9,19);runtime.selected_target_id=t['id']
        runtime._initialize(recent,semantic(t,mask))
        delayed,proposal,_=image(1.8,18);current,_,_=image(2.,20)
        def features(gray,**kwargs):
            calls.append((gray.copy(),kwargs['mask'].copy()))
            return None if len(calls)==failed_call else real(gray,**kwargs)
        monkeypatch.setattr(cv2,'goodFeaturesToTrack',features)
        runtime.source={'frames':{'head':delayed}}
        runtime.future=Future();runtime.future.set_result(semantic(proposal,mask))
        result=runtime.process({'frames':{'head':current}},t['id'])
        capture=runtime.selected_feature_failure
        assert capture['tracker_stage']==stage and capture['phase']=='semantic_keyframe_rebase'
        assert np.array_equal(capture['arrays']['previous_gray'],calls[failed_call-1][0])
        assert np.array_equal(capture['arrays']['previous_feature_mask'],calls[failed_call-1][1])
        assert capture['selected_target_observed_after_processing'] and result['targets']
    finally:runtime.close()


def test_current_tracking_feature_failure_separate_phase_and_selection_reset():
    runtime=AsyncPerception(NoModel(),semantic_interval=float('inf'));runtime.last_submit=0.
    try:
        old,t,mask=image();old['rgb'][:]=81;runtime.selected_target_id=t['id']
        runtime._initialize(old,semantic(t,mask));current={**old,'frame_id':11,'timestamp':1.1}
        assert runtime.process({'frames':{'head':current}},t['id'])['targets']==[]
        capture=runtime.selected_feature_failure
        assert capture['phase']=='current_tracking' and capture['tracker_stage']=='head'
        assert capture['selected_target_observed_after_processing'] is False
        assert runtime.selected_tracking_failure is not None
        runtime.process({'frames':{'head':{**current,'frame_id':12,'timestamp':1.2}}},'another')
        assert runtime.selected_feature_failure is None
    finally:runtime.close()


def selected_session(directory):
    engine=Engine();engine.finish.set();clock=[10.]
    session=ProposalSession(directory,engine,clock=lambda:clock[0],clock_id='clock')
    atomic_json(directory/'target_selection.json',{'target_id':'object','revision':1})
    session.update(frame(),target(),masks())
    if session.future is not None:session.future.result(2)
    session.poll()
    return session,clock


@pytest.mark.parametrize('reason',[
    'forward_flow_failed','backward_flow_failed','inconsistent_optical_flow'])
def test_correspondence_trigger_retained_and_archived_without_relabelling(tmp_path,reason):
    """Transport preserves actual failure cause; it grants no target authority."""
    runtime=AsyncPerception(NoModel(),semantic_interval=float('inf'))
    session,clock=selected_session(tmp_path)
    try:
        current=frame();capture=feature_capture(current)
        capture.update(reason=reason,tracking_failure_trigger=reason,
                       flow_filter_counts={'input_points':4,'usable_points':2})
        runtime.selected_target_id='object'
        runtime.tracker.last_feature_failure=capture
        runtime._retain_feature_failure(runtime.tracker,current,
            phase='current_tracking',tracker_stage='head')
        retained=runtime.selected_feature_failure
        assert retained['reason']==reason and retained['phase']=='current_tracking'
        before=read_json(tmp_path/'latest_proposals.json')
        session.update(current,target(),masks(),feature_failure=retained)
        session.selected_feature_audit.result(2);session.publish()
        evidence=read_json(tmp_path/'first_selected_feature_failure/evidence.json')
        assert evidence['reason']==reason and evidence['motion_authorized'] is False
        with np.load(tmp_path/'first_selected_feature_failure/tracking_inputs.npz',
                     allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata_json'].item()))
        assert metadata['reason']==reason
        assert metadata['tracking_failure_trigger']==reason
        assert metadata['flow_filter_counts']==capture['flow_filter_counts']
        assert session.error is None and read_json(tmp_path/'latest_proposals.json')==before
    finally:session.close();runtime.close()


@pytest.mark.parametrize('reason',[
    'insufficient_ransac_consensus','inconsistent_measured_world_motion',
    'inconsistent_surface_normal','identity_expired'])
def test_non_correspondence_failure_does_not_use_transport_slot(tmp_path,reason):
    runtime=AsyncPerception(NoModel(),semantic_interval=float('inf'))
    session,clock=selected_session(tmp_path)
    try:
        current=frame();capture=feature_capture(current);capture['reason']=reason
        runtime.selected_target_id='object';runtime.tracker.last_feature_failure=capture
        runtime._retain_feature_failure(runtime.tracker,current,
            phase='current_tracking',tracker_stage='head')
        assert runtime.selected_feature_failure is None
        session.update(current,target(),masks(),feature_failure=capture)
        assert session.first_selected_feature_failure is None
        assert session.selected_feature_audit is None and session.error is None
    finally:session.close();runtime.close()


def test_discarded_rebase_evidence_written_once_without_invalidating_visible_target(tmp_path,monkeypatch):
    from teleop_assist.copilot import service as module
    writer=module.write_selected_feature_failure;started,finish=threading.Event(),threading.Event();threads=[]
    def blocked(*args):
        threads.append(threading.current_thread().name);started.set()
        assert finish.wait(3);return writer(*args)
    monkeypatch.setattr(module,'write_selected_feature_failure',blocked)
    session,clock=selected_session(tmp_path)
    try:
        candidate=deepcopy(session.batch['candidates']);clock[0]=10.07
        session.update(frame(1,10.07),target(1,10.07),masks())
        clock[0]=10.14
        current=frame(2,10.14);current['rgb'][:]=75;capture=feature_capture(current)
        frozen=deepcopy(capture)
        session.update(current,target(2,10.14),masks(),feature_failure=capture)
        assert started.wait(1)
        assert session.error is None and session.first_selected_target_failure is None
        assert session.batch['candidates']==candidate
        before=read_json(tmp_path/'latest_proposals.json')
        assert before['motion_static'] and before['valid'] is False
        small=read_json(tmp_path/'ui_state.json')['first_selected_feature_failure']
        assert small['replay_status']=='pending' and small['selected_target_observed_after_processing']
        assert 'arrays' not in small and 'previous_gray' not in json.dumps(small)
        current['rgb'][:]=201;capture['arrays']['previous_gray'][:]=0
        capture['previous_frame']['T_world_camera'][0,3]=9.
        finish.set();session.selected_feature_audit.result(2);session.publish()
        directory=tmp_path/'first_selected_feature_failure'
        with np.load(directory/'tracking_inputs.npz',allow_pickle=False) as saved:
            for key,arr in frozen['arrays'].items():np.testing.assert_array_equal(saved[key],arr)
            metadata=json.loads(str(saved['metadata_json'].item()))
            assert metadata['previous_frame']['T_world_camera'][0][3]==0
            assert metadata['phase']=='semantic_keyframe_rebase'
        with np.load(directory/'frame.npz',allow_pickle=False) as saved:
            assert int(saved['frame_id'])==2 and float(saved['timestamp'])==10.14
            assert np.all(saved['rgb']==75)
        evidence=read_json(directory/'evidence.json')
        assert evidence['scope']=='first_selected_feature_failure_exact_local_replay_only'
        assert evidence['motion_authorized'] is False
        assert read_json(tmp_path/'latest_proposals.json')==before
        data=(directory/'evidence.json').read_bytes()
        clock[0]=10.2;current=frame(3,10.2)
        session.update(current,[],{},tracking_failure=diagnostic_capture(current),feature_failure=feature_capture(current))
        session.selected_failure_audit.result(2)
        assert session.first_selected_target_failure['frame_id']==3
        atomic_json(tmp_path/'target_selection.json',{'target_id':'different','revision':2})
        session.update(current,[],{},feature_failure=feature_capture(current,target_id='different'))
        assert session.first_selected_feature_failure['selected_target_id']=='object'
        assert (directory/'evidence.json').read_bytes()==data
        assert len(threads)==1 and threads[0].startswith('copilot-failure-audit')
    finally:finish.set();session.close()


@pytest.mark.parametrize('changed',['target','frame','timestamp','reason','phase','stage'])
def test_stale_wrong_selection_or_unbound_capture_does_not_consume_slot(tmp_path,changed):
    session,clock=selected_session(tmp_path)
    try:
        current=frame();capture=feature_capture(current)
        key,value={'target':('target_id','other'),'frame':('current_frame_id',99),
            'timestamp':('current_timestamp',9.),'reason':('reason','identity_expired'),
            'phase':('phase','unknown'),'stage':('tracker_stage','unknown')}[changed]
        capture[key]=value
        session.update(current,target(),masks(),feature_failure=capture)
        assert session.first_selected_feature_failure is None and session.selected_feature_audit is None
        assert session.error is None
    finally:session.close()


def test_warmup_and_latched_camera_error_do_not_use_feature_slot(tmp_path):
    engine=Engine();engine.finish.set();session=ProposalSession(tmp_path,engine,clock=lambda:10.,clock_id='clock')
    try:
        session.update(frame(0,9.),[],{},feature_failure=feature_capture(frame(0,9.)))
        session.failure_audit.result(2);warmup=(tmp_path/'first_failure/frame.npz').read_bytes()
        assert session.first_selected_feature_failure is None
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        session.update(frame(1,9.),[],{})
        old_error=session.error
        current=frame(2,10.)
        session.update(current,target(2,10.),masks(),feature_failure=feature_capture(current))
        session.selected_feature_audit.result(2)
        assert session.first_selected_feature_failure['frame_id']==2
        assert session.error==old_error and session.batch is None
        assert session.first_selected_target_failure is None
        assert (tmp_path/'first_failure/frame.npz').read_bytes()==warmup
    finally:session.close()


@pytest.mark.parametrize('failure',['budget','metadata','copy','submit','existing','write'])
def test_diagnostic_errors_never_retry_or_change_target_permissions(tmp_path,monkeypatch,failure):
    from teleop_assist.copilot import service as module
    session,clock=selected_session(tmp_path)
    original_budget=module.SELECTED_REPLAY_MAX_BYTES;capture=feature_capture(frame())
    original_freeze=module.freeze_selected_failure_inputs;original_submit=session.audit_pool.submit
    original_writer=module.write_selected_feature_failure
    def broken(*args,**kwargs):raise OSError('diagnostic only')
    try:
        before=read_json(tmp_path/'latest_proposals.json')
        directory=tmp_path/'first_selected_feature_failure'
        if failure=='budget':monkeypatch.setattr(module,'SELECTED_REPLAY_MAX_BYTES',10)
        elif failure=='metadata':capture['oversize']='x'*(64*1024)
        elif failure=='copy':monkeypatch.setattr(module,'freeze_selected_failure_inputs',broken)
        elif failure=='submit':monkeypatch.setattr(session.audit_pool,'submit',broken)
        elif failure=='write':monkeypatch.setattr(module,'write_selected_feature_failure',broken)
        else:directory.mkdir();(directory/'keep').write_text('original')
        session.update(frame(),target(),masks(),feature_failure=capture)
        if session.selected_feature_audit is not None:
            with pytest.raises((OSError,FileExistsError)):session.selected_feature_audit.result(2)
        session.publish()
        evidence=deepcopy(session.first_selected_feature_failure)
        assert evidence['replay_status']=='failed'
        assert session.error is None and read_json(tmp_path/'latest_proposals.json')==before
        monkeypatch.setattr(module,'SELECTED_REPLAY_MAX_BYTES',original_budget)
        monkeypatch.setattr(module,'freeze_selected_failure_inputs',original_freeze)
        monkeypatch.setattr(session.audit_pool,'submit',original_submit)
        monkeypatch.setattr(module,'write_selected_feature_failure',original_writer)
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':2})
        session.update(frame(),target(),masks(),feature_failure=feature_capture(frame()))
        assert session.first_selected_feature_failure==evidence
        if failure=='existing':assert (directory/'keep').read_text()=='original' and len(list(directory.iterdir()))==1
    finally:
        monkeypatch.setattr(session.audit_pool,'submit',original_submit);session.close()


def test_close_waits_for_feature_writer_and_publishes_completion(tmp_path,monkeypatch):
    from teleop_assist.copilot import service as module
    started,finish,closed=threading.Event(),threading.Event(),threading.Event()
    writer=module.write_selected_feature_failure
    def blocked(*args):started.set();assert finish.wait(3);return writer(*args)
    monkeypatch.setattr(module,'write_selected_feature_failure',blocked)
    session,clock=selected_session(tmp_path)
    session.update(frame(),target(),masks(),feature_failure=feature_capture(frame()))
    assert started.wait(1)
    thread=threading.Thread(target=lambda:(session.close(),closed.set()),daemon=True);thread.start()
    try:
        assert not closed.wait(.03);finish.set();thread.join(2)
        assert closed.is_set()
        published=read_json(tmp_path/'ui_state.json')['first_selected_feature_failure']
        assert published['replay_status']=='complete'
        for name,info in published['files'].items():
            assert hashlib.sha256((tmp_path/'first_selected_feature_failure'/name).read_bytes()).hexdigest()==info['sha256']
    finally:finish.set();thread.join(3)
