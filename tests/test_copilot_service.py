from copy import deepcopy
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from teleop_assist.copilot.mailbox import atomic_json, read_json
from teleop_assist.copilot.service import ProposalSession,FrameTimingTrace


class Engine:
    def __init__(self):
        self.started = threading.Event(); self.finish = threading.Event(); self.closed = False
    def infer(self, obs, frame):
        self.started.set(); self.finish.wait(2)
        return dict(target_id=obs.target_id, observation_id=obs.observation_id,
            capture_timestamp=obs.capture_timestamp, clock_id=obs.clock_id,
            calibration_id=obs.calibration_id, candidates=[dict(candidate_id=obs.observation_id+':0',
                base_ee=np.eye(4).tolist(), world_canonical=np.eye(4).tolist(), model_score=.9,
                geometry_valid=True, ik_valid=True, collision_valid=False,
                observed_hand_collision={'pregrasp_collision_free': True})])
    def close(self): self.closed = True


def frame(i=0, stamp=10.):
    return dict(frame_id=i, timestamp=stamp, clock_id='clock', calibration_id='cal',
        rgb=np.zeros((24, 24, 3), np.uint8), depth=np.ones((24, 24)),
        intrinsics=np.array([[300., 0, 12], [0, 300, 12], [0, 0, 1]]),
        T_world_camera=np.eye(4), T_world_base=np.eye(4))


def target(i=0, stamp=10.): return [{'id': 'object', 'label': 'Zylinder', 'bbox_px': [0, 0, 24, 24], 'frame_id':i, 'timestamp':stamp}]
def masks(): return {'object': np.ones((24, 24), bool)}


def test_late_result_cannot_replace_new_selection(tmp_path):
    clock = [10.]; engine = Engine()
    service = ProposalSession(tmp_path, engine, clock=lambda: clock[0], clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json', {'target_id': 'object', 'revision': 1})
        service.update(frame(), target(), masks())
        assert engine.started.wait(1)
        atomic_json(tmp_path/'target_selection.json', {'target_id': None, 'revision': 2})
        service.poll()
        engine.finish.set(); service.future.result(2); service.poll()
        assert read_json(tmp_path/'latest_proposals.json')['candidates'] == []
        assert service.batch is None
    finally:
        engine.finish.set(); service.close()


def test_manual_sim_opt_in_recovers_observation_gap_without_retimestamping_model(tmp_path):
    clock=[10.];engine=Engine();engine.finish.set()
    service=ProposalSession(tmp_path,engine,clock=lambda:clock[0],clock_id='clock',recover_transient_tracking=True)
    try:
        atomic_json(tmp_path/'target_selection.json',dict(target_id='object',revision=1))
        service.update(frame(),target(),masks())
        if service.future is not None:service.future.result(2)
        service.poll()
        source=service.batch['model_source_capture_timestamp']
        clock[0]=10.1;service.update(frame(1,10.1),[],{})
        assert not read_json(tmp_path/'latest_proposals.json')['motion_static']
        for i in range(3):
            clock[0]=11.+i*.07
            service.update(frame(i+2,clock[0]),target(i+2,clock[0]),masks())
            assert read_json(tmp_path/'latest_proposals.json')['motion_static']==(i==2)
        assert service.batch['model_source_capture_timestamp']==source==10.
        assert service.first_failure['reason']=='selected_object_not_observed'
    finally:service.close()


def test_frozen_pose_renews_only_with_independent_measurement_and_never_authorizes_motion(tmp_path):
    clock = [10.]; engine = Engine(); engine.finish.set()
    service = ProposalSession(tmp_path, engine, clock=lambda: clock[0], clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json', {'target_id': 'object', 'revision': 1})
        service.update(frame(), target(), masks())
        if service.future is not None: service.future.result(2)
        service.poll()
        frozen = deepcopy(service.batch['candidates'][0])
        for i in (1, 2):
            clock[0] = 10.+i*.07
            service.update(frame(i, clock[0]), target(i, clock[0]), masks())
        batch = read_json(tmp_path/'latest_proposals.json')
        assert batch['motion_static'] and batch['valid'] is False
        assert batch['capture_timestamp'] == clock[0]
        assert batch['model_source_capture_timestamp'] == 10.
        assert batch['candidates'][0]['base_ee'] == frozen['base_ee']
        clock[0] = 11.; service.poll()
        failure = read_json(tmp_path/'ui_state.json')['first_failure']
        assert failure['reason'] == 'camera_stream_expired'
        assert abs(failure['camera_age_s']-.86) < 1e-9
        batch = read_json(tmp_path/'latest_proposals.json')
        assert not batch['motion_static'] and not batch['valid']
        service.update(frame(3, 11.), target(3, 11.), masks())
        assert service.error  # lost evidence requires deliberate reselection
        later=read_json(tmp_path/'ui_state.json')['first_failure']
        assert {key:value for key,value in later.items() if key!='replay_status'} == {
            key:value for key,value in failure.items() if key!='replay_status'}
        atomic_json(tmp_path/'target_selection.json', {'target_id': 'object', 'revision': 2})
        service.update(frame(4, 11.01), target(4, 11.01), masks())
        assert service.error is None
        assert service.first_failure is None
    finally:
        service.close()
    assert engine.closed


def test_stream_expiry_while_inference_pending_discards_result(tmp_path):
    clock = [10.]; engine = Engine()
    service = ProposalSession(tmp_path, engine, clock=lambda: clock[0], clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json', {'target_id': 'object', 'revision': 1})
        service.update(frame(), target(), masks())
        assert engine.started.wait(1)
        clock[0] = 11.; service.poll()
        engine.finish.set(); service.future.result(2); service.poll()
        assert service.batch is None
        assert read_json(tmp_path/'latest_proposals.json')['valid'] is False
    finally:
        engine.finish.set(); service.close()


def test_new_selection_failure_never_claims_the_previous_failures_replay(tmp_path):
    engine=Engine();engine.finish.set();clock=[10.]
    service=ProposalSession(tmp_path,engine,clock=lambda:clock[0],clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        service.update(frame(1,10.),[],{})
        service.failure_audit.result(2);service.publish()
        first=read_json(tmp_path/'ui_state.json')['first_failure']
        before=(tmp_path/'first_failure/evidence.json').read_bytes()
        assert first['frame_id']==1 and first['replay_status']=='complete'
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':2})
        clock[0]=10.1;service.update(frame(2,10.1),[],{})
        second=read_json(tmp_path/'ui_state.json')['first_failure']
        assert second['frame_id']==2
        assert second['replay_status']=='not_captured_previous_failure_retained'
        assert 'replay_directory' not in second
        assert (tmp_path/'first_failure/evidence.json').read_bytes()==before
        assert read_json(tmp_path/'latest_proposals.json')['valid'] is False
    finally:service.close()


def test_startup_expiry_without_selection_recovers_only_on_fresh_camera_evidence(tmp_path):
    clock = [11.]; engine = Engine()
    service = ProposalSession(tmp_path, engine, clock=lambda: clock[0], clock_id='clock')
    try:
        service.update(frame(0, 10.), target(), masks())
        assert service.error and service.selected is None
        service.poll()
        assert service.error  # receipt/polling cannot manufacture fresh evidence
        service.update(frame(1, 11.), target(1, 11.), masks())
        assert service.error is None and service.status == 'choose_object'
        assert service.future is None
    finally:
        engine.finish.set(); service.close()


def test_compact_batch_retains_full_validation_counts_and_original_ids(tmp_path):
    class ManyEngine(Engine):
        def infer(self, obs, current):
            result = super().infer(obs, current)
            base = result['candidates'][0]
            result['candidates'] = [dict(deepcopy(base), candidate_id=f'original-{i}',
                model_score=1.-i*.01, ik_valid=i%2==0,
                observed_hand_collision={'pregrasp_collision_free':i==8}) for i in range(10)]
            return result
    engine = ManyEngine(); engine.finish.set()
    service = ProposalSession(tmp_path, engine, clock=lambda:10., clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json', {'target_id':'object','revision':1})
        service.update(frame(), target(), masks())
        if service.future is not None: service.future.result(2)
        service.poll()
        batch = read_json(tmp_path/'latest_proposals.json')
        assert len(batch['candidates']) == 8
        assert batch['candidates'][0]['candidate_id'] == 'original-8'
        assert batch['candidate_counts'] == {'total':10,'ik_valid':5,'ik_and_pregrasp_clear':1}
        assert len(read_json(tmp_path/'proposal_audit.json')['candidates']) == 10
        assert not batch['valid']
    finally:
        service.close()


def test_display_frame_binds_exact_image_and_projection_and_survives_recording_close(tmp_path):
    clock=[10.];engine=Engine();engine.finish.set()
    service=ProposalSession(tmp_path,engine,clock=lambda:clock[0],clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        service.update(frame(),target(),masks())
        if service.future is not None:service.future.result(2)
        service.poll()
        batch=read_json(tmp_path/'preview_proposals.json');ui=read_json(tmp_path/'ui_state.json')
        assert batch['preview_frame']==ui['display_frame']
        assert ui['display_frame']['jpeg_sha256']==hashlib.sha256((tmp_path/'frame.jpg').read_bytes()).hexdigest()
        first=ui['display_frame']
        clock[0]=10.07;service.update(frame(1,10.07),target(1,10.07),masks())
        ui=read_json(tmp_path/'ui_state.json')
        assert ui['display_frame']['jpeg_sha256']==first['jpeg_sha256']
        assert ui['display_frame']['key']!=first['key']  # identical RGB is not the same observation
        assert ui['camera_current']
    finally:service.close()
    batch=read_json(tmp_path/'preview_proposals.json');ui=read_json(tmp_path/'ui_state.json')
    assert batch['preview_frame']==ui['display_frame']
    assert 'preview_uv' in batch['candidates'][0]
    assert not batch['preview_current'] and not ui['camera_current']
    assert batch['model_source_capture_timestamp']==10. and batch['valid'] is False


def test_foreign_or_untimed_target_cannot_supply_selected_model_geometry(tmp_path):
    engine=Engine();service=ProposalSession(tmp_path,engine,clock=lambda:10.,clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        service.update(frame(),target(99,9.),masks())
        assert service.error=='selected_object_not_observed'
        assert service.future is None and not engine.started.is_set()
        assert read_json(tmp_path/'ui_state.json')['objects']==[]
    finally:engine.finish.set();service.close()


def test_first_failed_geometry_is_copied_once_and_compressed_off_update_thread(tmp_path,monkeypatch):
    from teleop_assist.copilot import service as module
    from teleop_assist.copilot.observations import GeometryObservation
    started=threading.Event();finish=threading.Event();original=module.write_failed_observation
    def slow(*args):
        started.set();assert finish.wait(2);return original(*args)
    monkeypatch.setattr(module,'write_failed_observation',slow)
    clock=[10.];engine=Engine();engine.finish.set()
    service=ProposalSession(tmp_path,engine,clock=lambda:clock[0],clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        service.update(frame(),target(),masks())
        if service.future is not None:service.future.result(2)
        service.poll()
        failed=frame(1,10.07);failed['depth'][:]=1.1;failed['rgb'][:]=73
        clock[0]=10.07;service.update(failed,target(1,10.07),masks())
        assert service.error and started.wait(1)
        assert not service.failure_audit.done()  # update returned while compressor is blocked
        failed['depth'][:]=3.;failed['rgb'][:]=211
        finish.set();service.failure_audit.result(2);service.publish()
        saved=GeometryObservation.from_npz(tmp_path/'first_failure/observation.npz')
        assert saved.capture_timestamp==10.07 and np.allclose(saved.depth,1.1)
        with np.load(tmp_path/'first_failure/frame.npz',allow_pickle=False) as data:
            assert int(data['frame_id'])==1 and np.all(data['rgb']==73)
        evidence=read_json(tmp_path/'first_failure/evidence.json')
        assert evidence['scope']=='failed_observed_geometry_replay_only' and not evidence['motion_authorized']
        assert read_json(tmp_path/'ui_state.json')['first_failure']['replay_status']=='complete'
    finally:finish.set();service.close()


def test_identical_poll_skips_publication_but_still_expires_and_consumes_selection(tmp_path,monkeypatch):
    from teleop_assist.copilot import service as module
    clock=[10.];engine=Engine();engine.finish.set()
    session=ProposalSession(tmp_path,engine,clock=lambda:clock[0],clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        session.update(frame(),target(),masks())
        if session.future is not None:session.future.result(2)
        session.poll();before=(tmp_path/'latest_proposals.json').read_bytes()
        original=module.atomic_json;writes=[]
        def write(path,value):writes.append(path.name);return original(path,value)
        monkeypatch.setattr(module,'atomic_json',write)
        clock[0]+=.10;session.poll();session.poll()
        assert writes==[] and (tmp_path/'latest_proposals.json').read_bytes()==before
        clock[0]+=.151;session.poll()
        assert session.error=='camera_stream_expired' and 'latest_proposals.json' in writes
        assert not read_json(tmp_path/'latest_proposals.json')['preview_current']
        original(tmp_path/'target_selection.json',{'target_id':None,'revision':2})
        session.poll()
        assert session.selected is None and read_json(tmp_path/'latest_proposals.json')['candidates']==[]
    finally:session.close()


def test_frame_timing_trace_stays_bounded_and_writes_original_source_times(tmp_path):
    trace=FrameTimingTrace(max_records=3,max_bytes=200)
    original={'frame_id':1,'capture_timestamp':12.,'observed_at':12.1}
    trace.append(original);original['capture_timestamp']=999.
    for index in range(10):trace.append({'frame_id':index+2,'capture_timestamp':12.+index*.1})
    with ThreadPoolExecutor(max_workers=1) as pool:
        trace.publish(pool,tmp_path,force=True);trace.future.result(2)
    data=(tmp_path/'frame_timing.jsonl').read_bytes()
    import json
    records=[json.loads(line) for line in data.splitlines()]
    metadata=read_json(tmp_path/'frame_timing_meta.json')
    assert len(data)<=200 and len(records)<=3 and metadata['dropped']>0
    assert records[0]['capture_timestamp']==12. and metadata['bytes']==len(data)


def test_selected_frame_trace_is_retained_on_close(tmp_path):
    engine=Engine();engine.finish.set()
    session=ProposalSession(tmp_path,engine,clock=lambda:10.,clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        session.frame_diagnostics=dict(received_age_s=.015,perception_s=.02,post_perception_age_s=.035)
        session.update(frame(),target(),masks())
    finally:session.close()
    import json
    records=[json.loads(line) for line in (tmp_path/'frame_timing.jsonl').read_text().splitlines()]
    assert len(records)==1 and records[0]['capture_timestamp']==10.
    assert records[0]['perception_s']==.02 and records[0]['received_age_s']==.015
    assert records[0]['target_id']=='object' and records[0]['update_work_s']>=0


def test_lost_target_keeps_actual_failed_image_and_tracker_reason_without_inventing_mask(tmp_path):
    engine=Engine();engine.finish.set()
    clock=[10.]
    session=ProposalSession(tmp_path,engine,clock=lambda:clock[0],clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        session.update(frame(),target(),masks())
        if session.future is not None:session.future.result(2)
        session.poll()
        failed=frame(1,10.07);failed['rgb'][:]=47;clock[0]=10.07
        session.frame_diagnostics={'perception_diagnostics':{'lost_tracks':{'object':'insufficient_visual_features'}}}
        session.update(failed,[],{})
        critical=read_json(tmp_path/'latest_proposals.json')
        assert critical['processed_frame']==dict(frame_id=1,capture_timestamp=10.07,clock_id='clock',
            calibration_id=failed['calibration_id'],rejected=True,reason='selected_object_not_observed')
        assert critical['capture_timestamp']==10. and not critical['motion_static']
        failed['rgb'][:]=99
        session.failure_audit.result(2);session.publish()
        with np.load(tmp_path/'first_failure/frame.npz',allow_pickle=False) as data:
            assert np.all(data['rgb']==47) and int(data['frame_id'])==1
        assert not (tmp_path/'first_failure/observation.npz').exists()
        evidence=read_json(tmp_path/'first_failure/evidence.json')
        assert evidence['frame_timing']['perception_diagnostics']['lost_tracks']['object']=='insufficient_visual_features'
        assert evidence['scope']=='failed_camera_frame_without_target_observation' and not evidence['motion_authorized']
    finally:session.close()


def test_explicit_same_frame_update_can_change_object_chooser_without_retiming(tmp_path):
    engine=Engine();session=ProposalSession(tmp_path,engine,clock=lambda:10.,clock_id='clock')
    try:
        session.update(frame(),target(),masks());first=read_json(tmp_path/'ui_state.json')
        targets=target();targets[0]['label']='Neuer Modellhinweis'
        session.update(frame(),targets,masks());second=read_json(tmp_path/'ui_state.json')
        assert second['objects'][0]['label']=='Neuer Modellhinweis'
        assert first['display_frame']==second['display_frame']
        assert session.future is None
    finally:engine.finish.set();session.close()


def test_trace_write_failure_is_reported_without_breaking_service_close(tmp_path,monkeypatch):
    engine=Engine();engine.finish.set()
    session=ProposalSession(tmp_path,engine,clock=lambda:10.,clock_id='clock')
    def fail(*args):raise OSError('simulated unavailable audit storage')
    monkeypatch.setattr(session.frame_trace,'_write',fail)
    atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
    session.update(frame(),target(),masks())
    session.close()
    trace=read_json(tmp_path/'ui_state.json')['timing_trace']
    assert trace['status']=='failed' and trace['write_error']=='OSError'
    assert trace['records']==1 and trace['motion_authorized'] is False and engine.closed


def test_fresh_geometry_is_published_before_jpeg_or_ghost_work_without_retiming(tmp_path,monkeypatch):
    clock=[10.];engine=Engine();engine.finish.set()
    session=ProposalSession(tmp_path,engine,clock=lambda:clock[0],clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        session.update(frame(),target(),masks())
        if session.future is not None:session.future.result(2)
        session.poll()
        clock[0]=10.07;session.update(frame(1,10.07),target(1,10.07),masks())
        original=session._publish_image;prior_ui=read_json(tmp_path/'preview_proposals.json')
        seen=[]
        def image_stage():
            critical=read_json(tmp_path/'latest_proposals.json')
            seen.append(('jpeg',critical))
            assert critical['capture_timestamp']==10.14
            assert critical['validation_observation_id']=='cal:2'
            assert critical['motion_static'] is True and critical['valid'] is False
            assert critical['model_source_capture_timestamp']==10.
            assert critical['preview_current'] is False and 'preview_frame' not in critical
            assert all('preview_hand_polygons' not in c and 'preview_uv' not in c for c in critical['candidates'])
            assert read_json(tmp_path/'preview_proposals.json')==prior_ui
            clock[0]+=.02  # diagnostic display work never renews source time
            original()
        class Preview:
            def project(self,pose,current):
                critical=read_json(tmp_path/'latest_proposals.json');seen.append(('ghost',critical))
                assert critical['capture_timestamp']==10.14
                return [{'link':'fixture','uv':[[1,1],[2,1],[1,2]]}]
        monkeypatch.setattr(session,'_publish_image',image_stage);engine.preview=Preview()
        clock[0]=10.14;session.update(frame(2,10.14),target(2,10.14),masks())
        assert [stage for stage,_ in seen]==['jpeg','ghost']
        critical=read_json(tmp_path/'latest_proposals.json');ui=read_json(tmp_path/'preview_proposals.json')
        assert critical['candidates'][0]['base_ee']==ui['candidates'][0]['base_ee']
        assert critical['capture_timestamp']==ui['capture_timestamp']==10.14
        assert ui['preview_frame']==read_json(tmp_path/'ui_state.json')['display_frame']
        assert ui['preview_frame']['capture_timestamp']==10.14
        assert 'preview_hand_polygons' in ui['candidates'][0]
        assert ui['valid'] is False and ui['preview_current'] is True
        without_visuals={key:value for key,value in ui.items() if key!='preview_frame'}
        without_visuals['preview_current']=False
        without_visuals['candidates']=[{key:value for key,value in item.items()
            if key not in ('preview_uv','preview_hand_polygons')} for item in ui['candidates']]
        assert without_visuals==critical  # no geometry/identity/permission metadata omitted
        assert session.publication_timing['critical_published_at']==10.14
        monkeypatch.setattr(session,'_publish_image',original)
    finally:session.close()


def test_revocation_reaches_critical_mailbox_even_if_image_publication_fails(tmp_path,monkeypatch):
    import pytest
    engine=Engine();engine.finish.set();clock=[10.]
    session=ProposalSession(tmp_path,engine,clock=lambda:clock[0],clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json',{'target_id':'object','revision':1})
        session.update(frame(),target(),masks())
        if session.future is not None:session.future.result(2)
        session.poll();original=session._publish_image
        def fail_image():raise OSError('display storage failed')
        monkeypatch.setattr(session,'_publish_image',fail_image)
        clock[0]=10.07
        with pytest.raises(OSError):session.update(frame(1,10.07),[],{})
        critical=read_json(tmp_path/'latest_proposals.json')
        assert critical['motion_static'] is False and critical['valid'] is False
        assert critical['validation_reason']=='selected_object_not_observed'
        monkeypatch.setattr(session,'_publish_image',original)
    finally:session.close()
