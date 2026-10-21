"""Live-feed glue with explicit fake endpoint evidence; no physics claims."""
from copy import deepcopy
from dataclasses import replace
import json
import time

import numpy as np
import pytest

from teleop_assist.copilot.mailbox import atomic_json
from teleop_assist.copilot.sim_control import SimUIIntent
from teleop_assist.copilot.sim_target_feed import SimTargetFeed
from test_copilot_live_sim_proposals import Fixture, complete_sim_fixture, pose, digest


class FeedFixture:
    def __init__(self, directory):
        self.base=Fixture();self.directory=directory;self.calls=[];self.delay=0.;self.failure=None
        self.feed=SimTargetFeed(directory,self,'a'*64,clock=lambda:self.base.now)
        for i,t in enumerate((10.,10.06,10.12)):
            self.feed.record_frame(self.base.frame(i,t),self.base.scene(t))
        self.selection=dict(target_id='learned_1',revision=1,capture_timestamp=9.9,clock_id='mono')
        self.intent=SimUIIntent(1,self.base.now,'mono','explicit_test_operator')
        self.session_status={}
        self.write()

    def write(self):
        atomic_json(self.directory/'target_selection.json',self.selection)
        atomic_json(self.directory/'latest_proposals.json',self.base.preview)

    def call(self, operation, **request):
        self.calls.append((operation,deepcopy(request)))
        if self.failure:raise RuntimeError(self.failure)
        endpoint=self.base.endpoint()
        endpoint['ik']['right_target_base_ee']=deepcopy(request['base_ee'])
        endpoint['ik']['left_target_base_ee']=deepcopy(request['left_base_ee'])
        endpoint['state_evidence'].update(robot_state=deepcopy(request['robot_state']),scene=deepcopy(request['scene']),
            snapshot_sha256=digest({'robot_state':request['robot_state'],'scene':request['scene']}))
        self.base.now+=self.delay
        return endpoint

    def select(self, candidate='grasp1', revision=None):
        self.intent=SimUIIntent(revision or self.intent.revision+1,self.base.now,'mono','explicit_test_operator',
            select_candidate_id=candidate)

    def prepare(self, **changes):
        """Drive selection then endpoint for the existing endpoint-only tests.

        The fake ACK below is explicitly a fixture; actual Feed→Session commit
        and denial are covered in test_copilot_preview_selection.py.
        """
        args=dict(intent=self.intent,session_status=self.session_status,robot_state=self.base.state(),scene=self.base.scene(),
            left_base_ee=pose(.3,.2,.9),now=self.base.now,clock_id='mono',calibration_id='cal')
        args.update(changes)
        result=self.feed.prepare(**args)
        preview=result.get('preview_selection')
        if preview is not None:
            self.session_status=dict(allowed=False,snapshot=dict(clutched=False,
                selected_candidate_id=preview.candidate_id,preview_selection_pending=True,
                preview_selection=self.feed._selection_identity(preview)))
            assert self.feed.finish_preview_selection(self.session_status)
            args['session_status']=self.session_status
            next_result=self.feed.prepare(**args)
            next_result['reset_control'] |= result['reset_control']
            return next_result
        return result


def test_ui_preview_never_auto_selects_or_copies_permissions(tmp_path):
    f=FeedFixture(tmp_path);result=f.prepare()
    assert result['batch'] is None and result['goal'] is None and f.calls==[]
    assert result['reset_control'] and result['reason']=='choose_grasp_explicitly'
    assert result['candidates'][0]['candidate_id']=='grasp1'
    assert result['candidates'][0]['preview_only'] is True
    assert not {'collision_valid','ik_valid','geometry_valid','valid','motion_static'} & result['candidates'][0].keys()
    assert not f.prepare()['reset_control']


def test_scheduler_timeout_revokes_without_replaying_old_selection(tmp_path):
    f=FeedFixture(tmp_path);f.select('grasp1');assert f.prepare()['batch'] is not None
    f.feed.revoke_validation()
    assert f.feed.active_candidate_id is None and f.feed._authorized_binding is None
    assert f.prepare()['batch'] is None  # Same still-readable click is consumed.
    f.select('grasp1');assert f.prepare()['batch'] is not None


def current_frame_fixture(tmp_path,monkeypatch,*,fault=None,delay=.08):
    f=FeedFixture(tmp_path);f.base.now=10.30;f.delay=delay;f.select('grasp1')
    frame=f.base.frame(3,10.30);f.feed.record_frame(frame,f.base.scene())
    def acknowledge(directory,captured,**kwargs):
        assert captured is frame
        f.base.preview.update(capture_timestamp=10.30,validation_observation_id='cal:3')
        result=dict(ready=True,reason='exact_current_frame_processed',capture_timestamp=10.30,
            expected_validation_observation_id='cal:3',clock_id='mono',calibration_id='cal',
            motion_authorized=False,deadline_renewed=False)
        if fault=='target_revision':f.selection['revision']+=1
        elif fault=='pose':f.base.preview['candidates'][0]['base_ee'][0][3]+=.01
        elif fault=='model_source':f.base.preview.update(model_source_observation_id='cal:1',observation_id='cal:1',model_source_capture_timestamp=10.06)
        elif fault=='stationarity':f.base.preview['motion_static']=False
        elif fault=='frame_changed':f.base.preview['capture_timestamp']=10.31
        elif fault=='timeout':result.update(ready=False,reason='current_frame_preview_timeout')
        f.write();return result
    monkeypatch.setattr('teleop_assist.copilot.sim_target_feed.await_preview_frame',acknowledge)
    return f,frame


def test_two_phase_uses_genuinely_new_observation_before_creating_batch(tmp_path,monkeypatch):
    f,frame=current_frame_fixture(tmp_path,monkeypatch)
    result=f.prepare(current_frame=frame)
    assert result['batch'] is not None
    assert f.base.now-10.12>.25  # Previous observation expired during endpoint work.
    assert result['batch'].capture_timestamp==10.30
    assert result['batch'].model_source_capture_timestamp==10.
    assert result['batch'].observation_id=='cal:3'
    assert result['endpoint_evidence']['state_evidence']['robot_state']['acquisition_timestamp']==10.30
    assert result['preview_evidence']['preview_capture_timestamp']==10.30
    assert not result['frame_barrier']['motion_authorized']


@pytest.mark.parametrize('fault',['target_revision','pose','model_source','stationarity','frame_changed','timeout'])
def test_two_phase_rejects_changes_during_parallel_endpoint_work(tmp_path,monkeypatch,fault):
    f,frame=current_frame_fixture(tmp_path,monkeypatch,fault=fault)
    result=f.prepare(current_frame=frame)
    assert result['batch'] is None and result['reset_control']
    assert f.feed.active_candidate_id is None


def test_new_camera_observation_cannot_renew_expired_robot_evidence(tmp_path,monkeypatch):
    f,frame=current_frame_fixture(tmp_path,monkeypatch,delay=.16)
    result=f.prepare(current_frame=frame)
    assert result['batch'] is None and 'evidence_clock_stale_or_future' in result['reason']
    assert result['endpoint_evidence']['state_evidence']['robot_state']['acquisition_timestamp']==10.30


def test_deliberate_selection_binds_only_chosen_pose_and_preserves_source_time(tmp_path):
    f=FeedFixture(tmp_path)
    other=deepcopy(f.base.preview['candidates'][0]);other.update(candidate_id='higher_score',model_score=.99)
    f.base.preview['candidates'].insert(0,other);f.write();f.select('grasp1')
    result=f.prepare(session_status={'snapshot':{'confirmed_candidate_id':'previous_grasp'}})
    assert result['reset_control'] and result['batch'] is not None
    assert [x.candidate_id for x in result['batch'].candidates]==['grasp1']
    assert result['batch'].capture_timestamp==10.12
    assert result['batch'].model_source_capture_timestamp==10.
    assert f.calls[0][0]=='validate_sim_candidate'
    np.testing.assert_array_equal(result['goal'],f.base.preview['candidates'][1]['base_ee'])
    assert result['authorization_evidence']['motion_authorized'] is False
    assert 'confirmed' not in result
    assert not f.prepare()['reset_control']


def test_new_object_revision_clears_choice_and_old_grasp_click_cannot_reactivate(tmp_path):
    f=FeedFixture(tmp_path);f.select();assert f.prepare()['batch'] is not None
    f.selection.update(revision=2,capture_timestamp=f.base.now);f.write()
    result=f.prepare()
    assert result['reset_control'] and result['batch'] is None and f.feed.active_candidate_id is None
    assert len(f.calls)==1
    f.select();assert f.prepare()['batch'] is not None


def test_new_deliberate_same_grasp_revision_resets_and_does_not_confirm(tmp_path):
    f=FeedFixture(tmp_path);f.select();assert f.prepare()['batch']
    f.select();result=f.prepare(session_status={'snapshot':{'confirmed_candidate_id':'grasp1'}})
    assert result['reset_control'] and result['batch'] is not None
    assert 'confirmed' not in result


@pytest.mark.parametrize('which',['latest_proposals.json','target_selection.json'])
@pytest.mark.parametrize('failure',['missing','malformed'])
def test_file_failure_is_closed_and_repair_does_not_replay_selection(tmp_path,which,failure):
    f=FeedFixture(tmp_path);f.select();path=tmp_path/which
    if failure=='missing':path.unlink()
    else:path.write_text('{broken')
    result=f.prepare();assert result['batch'] is None and result['reset_control'] and not f.calls
    f.write();assert f.prepare()['batch'] is None and not f.calls
    f.select();assert f.prepare()['batch'] is not None


@pytest.mark.parametrize('fault',['stale','future','clock','owner','confirm','expiry','unoffered','reused','regressed'])
def test_bad_selection_never_calls_worker(tmp_path,fault):
    f=FeedFixture(tmp_path);f.prepare();f.select()
    if fault=='stale':f.intent=replace(f.intent,capture_timestamp=f.base.now-.3)
    elif fault=='future':f.intent=replace(f.intent,capture_timestamp=f.base.now+.001)
    elif fault=='clock':f.intent=replace(f.intent,clock_id='other')
    elif fault=='owner':f.intent=replace(f.intent,client_id=None)
    elif fault=='confirm':f.intent=replace(f.intent,confirm=True)
    elif fault=='expiry':f.intent=replace(f.intent,expires_at=f.base.now)
    elif fault=='unoffered':f.intent=replace(f.intent,select_candidate_id='invented')
    elif fault=='reused':f.intent=replace(f.intent,revision=1)
    else:f.intent=replace(f.intent,revision=0)
    result=f.prepare();assert result['batch'] is None and result['reset_control'] and not f.calls


def test_slow_endpoint_uses_actual_post_worker_clock_and_keeps_old_capture_time(tmp_path):
    f=FeedFixture(tmp_path);f.select();f.delay=.16
    result=f.prepare()
    assert result['batch'] is None and result['goal'] is None and result['reset_control']
    assert 'stale' in result['reason']
    assert result['endpoint_evidence']['state_evidence']['robot_state']['acquisition_timestamp']==10.12


def test_fast_endpoint_does_not_retime_inference_or_frame(tmp_path):
    f=FeedFixture(tmp_path);f.select();f.delay=.04
    result=f.prepare();assert result['batch'] is not None
    assert result['batch'].capture_timestamp==10.12 and result['batch'].model_source_capture_timestamp==10.
    assert f.base.now==pytest.approx(10.16)


def test_worker_exception_requires_new_deliberate_choice(tmp_path):
    f=FeedFixture(tmp_path);f.select();f.failure='test_worker_failed'
    result=f.prepare();assert result['batch'] is None and result['reset_control']
    f.failure=None;assert f.prepare()['batch'] is None and len(f.calls)==1
    f.select();assert f.prepare()['batch'] is not None


def test_json_execution_flags_or_candidate_mutation_never_become_authority(tmp_path):
    f=FeedFixture(tmp_path);f.select();assert f.prepare()['batch'] is not None
    f.base.preview['candidates'][0]['base_ee'][0][3]+=.001;f.write()
    assert f.prepare()['batch'] is None
    f.select();f.base.preview['valid']=True;f.write()
    assert f.prepare()['batch'] is None


def test_ui_list_is_bounded_and_duplicate_ids_fail_closed(tmp_path):
    f=FeedFixture(tmp_path);one=f.base.preview['candidates'][0]
    f.base.preview['candidates']=[dict(deepcopy(one),candidate_id=f'id{i}') for i in range(64)]
    f.write();result=f.prepare();assert len(result['candidates'])==32 and result['batch'] is None
    f.base.preview['candidates'].append(deepcopy(f.base.preview['candidates'][0]));f.write()
    assert f.prepare()['candidates']==[]


def test_wrong_measured_left_pose_cannot_authorize_endpoint(tmp_path):
    f=FeedFixture(tmp_path);f.select()
    result=f.prepare(left_base_ee=pose(.31,.2,.9))
    assert result['batch'] is None and 'measured_left' in result['reason']


def test_recorded_scene_motion_still_invalidates_bound_selection(tmp_path):
    f=FeedFixture(tmp_path);f.select();assert f.prepare()['batch'] is not None
    f.base.now+=.06;f.base.index+=1
    scene=f.base.scene();scene['obstacles'][1]['world_pose'][0][3]+=.01
    f.feed.record_frame(f.base.frame(),scene)
    assert f.prepare()['batch'] is None


def test_reused_object_revision_with_changed_content_cannot_keep_authority(tmp_path):
    f=FeedFixture(tmp_path);f.select();assert f.prepare()['batch'] is not None
    f.selection['target_id']='another_object';f.write()
    result=f.prepare();assert result['batch'] is None and result['reset_control']


def test_unbound_worker_acceptance_flag_does_not_form_a_batch(tmp_path):
    f=FeedFixture(tmp_path);f.select()
    f.feed.worker=type('ForgedWorker',(),{'call':lambda *_args,**_kwargs:{'accepted':True}})()
    result=f.prepare();assert result['batch'] is None and result['reset_control']
    assert result['endpoint_evidence']=={'accepted':True}


def test_regressing_injected_clock_is_rejected(tmp_path):
    f=FeedFixture(tmp_path);f.select();f.delay=-.001
    result=f.prepare();assert result['batch'] is None and 'clock_regressed' in result['reason']


def test_malformed_recursive_intent_does_not_escape_fail_closed(tmp_path):
    f=FeedFixture(tmp_path);packet={'revision':2};packet['recursive']=packet
    result=f.prepare(intent=packet)
    assert result['batch'] is None and result['reset_control']


def recheck(f, **changes):
    args=dict(session_status={'snapshot':{'confirmed_candidate_id':'grasp1'}},
        now=f.base.now,clock_id='mono',calibration_id='cal')
    args.update(changes)
    return f.feed.recheck(**args)


def test_handoff_recheck_never_calls_worker_or_renews_existing_batch(tmp_path):
    f=FeedFixture(tmp_path);f.select();prepared=f.prepare();batch=prepared['batch']
    assert batch is not None
    f.base.now+=.06;f.base.index+=1
    f.feed.record_frame(f.base.frame(),f.base.scene());f.base.renew_preview();f.write()
    result=recheck(f)
    assert result['accepted'] and not result['motion_authorized'] and not result['deadline_renewed']
    assert len(f.calls)==1 and batch.capture_timestamp==10.12
    assert batch.model_source_capture_timestamp==10. and batch.observation_id=='cal:2'
    assert f.base.preview['capture_timestamp']==10.18


@pytest.mark.parametrize('fault',['selection_revision','selection_content','selection_missing',
    'selection_malformed','preview_missing','preview_invalid','preview_stale','preview_pose',
    'preview_source','preview_model_timestamp','preview_clock','preview_validation_id','confirmed_id'])
def test_handoff_recheck_revokes_changed_or_unavailable_sources(tmp_path,fault):
    f=FeedFixture(tmp_path);f.select();assert f.prepare()['batch'] is not None
    kwargs={}
    if fault=='selection_revision':f.selection['revision']+=1;f.write()
    elif fault=='selection_content':f.selection['target_id']='another_object';f.write()
    elif fault=='selection_missing':(tmp_path/'target_selection.json').unlink()
    elif fault=='selection_malformed':(tmp_path/'target_selection.json').write_text('{broken')
    elif fault=='preview_missing':(tmp_path/'latest_proposals.json').unlink()
    elif fault=='preview_invalid':f.base.preview['valid']=True;f.write()
    elif fault=='preview_stale':f.base.now+=.251
    elif fault=='preview_pose':f.base.preview['candidates'][0]['base_ee'][0][3]+=.001;f.write()
    elif fault=='preview_source':f.base.preview['candidates'][0]['source']='different_model';f.write()
    elif fault=='preview_model_timestamp':f.base.preview['model_source_capture_timestamp']+=.001;f.write()
    elif fault=='preview_clock':f.base.preview['clock_id']='other';f.write()
    elif fault=='preview_validation_id':f.base.preview['validation_observation_id']='cal:999';f.write()
    else:kwargs['session_status']={'snapshot':{'confirmed_candidate_id':'different_grasp'}}
    result=recheck(f,**kwargs)
    assert not result['accepted'] and not result['motion_authorized'] and not result['deadline_renewed']
    assert f.feed.active_candidate_id is None and f.feed.gate._frozen is None
    assert len(f.calls)==1
    f.write();assert not recheck(f)['accepted']  # Repaired files do not select again.


def test_handoff_recheck_without_successful_prepare_has_no_authority(tmp_path):
    f=FeedFixture(tmp_path);assert not recheck(f)['accepted'] and not f.calls


def test_object_change_while_endpoint_worker_runs_is_revoked_before_handoff(tmp_path):
    f=FeedFixture(tmp_path);f.select();original=f.feed.worker.call
    class ChangingWorker:
        def call(self, *args, **kwargs):
            endpoint=original(*args,**kwargs)
            f.selection['revision']+=1;f.write()
            return endpoint
    f.feed.worker=ChangingWorker()
    assert f.prepare()['batch'] is not None
    assert not recheck(f)['accepted']


def test_object_change_during_final_preview_check_is_revoked(tmp_path,monkeypatch):
    f=FeedFixture(tmp_path);f.select();assert f.prepare()['batch'] is not None
    original=f.feed.gate.recheck
    def inspect(*args,**kwargs):
        checked=original(*args,**kwargs)
        f.selection['revision']+=1;f.write()
        return checked
    monkeypatch.setattr(f.feed.gate,'recheck',inspect)
    result=recheck(f)
    assert not result['accepted'] and 'during_handoff_recheck' in result['reason']


@pytest.mark.parametrize('delay',[.251,-.001])
def test_handoff_recheck_accounts_for_time_spent_checking(tmp_path,monkeypatch,delay):
    f=FeedFixture(tmp_path);f.select();assert f.prepare()['batch'] is not None
    original=f.feed.gate.recheck
    def inspect(*args,**kwargs):
        checked=original(*args,**kwargs);f.base.now+=delay
        return checked
    monkeypatch.setattr(f.feed.gate,'recheck',inspect)
    assert not recheck(f)['accepted']


def test_late_first_selection_with_1024_frames_fits_real_clock_state_lease(tmp_path):
    f=FeedFixture(tmp_path)
    while len(f.feed.gate.frames)<1024:
        f.base.now+=.03;f.base.index+=1
        f.feed.record_frame(f.base.frame(),f.base.scene())
    f.base.renew_preview();f.write();f.select()
    started=time.perf_counter();source_now=f.base.now
    f.feed.clock=lambda:source_now+time.perf_counter()-started
    result=f.prepare()
    assert result['batch'] is not None, result['reason']
    assert result['batch'].capture_timestamp==source_now
    assert result['batch'].model_source_capture_timestamp==10.
    assert f.feed.clock()-source_now<.15


def test_complete_sim_policy_preserves_negative_observed_preview_while_independently_checking_endpoint(tmp_path):
    f=FeedFixture(tmp_path);f.base=complete_sim_fixture()
    f.feed=SimTargetFeed(tmp_path,f,'a'*64,clock=lambda:f.base.now,policy='complete_static_sim_orientation')
    for i,t in enumerate((10.,10.06,10.12)):
        assert f.feed.record_frame(f.base.frame(i,t),f.base.scene(t))['recorded']
    f.write();f.select();before=deepcopy(f.base.preview)
    result=f.prepare();assert result['batch'] is not None
    row=result['candidates'][0]
    assert row['preview_collision_policy']=='complete_static_sim_orientation' and row['preview_only']
    assert not {'collision_valid','geometry_valid','ik_valid','valid'} & row.keys()
    observed=row['observed_hand_collision']
    assert observed['status']=='observed_collision_rejected'
    assert observed['pregrasp_collision_free'] is False and observed['hand_collision_free'] is False
    assert observed['final']==dict(forbidden_object_points=7,scene_collision_points=23,allowed_target_pad_contacts=2)
    assert observed['object_point_count']==100 and observed['scene_point_count']==800
    assert 'collision_valid' not in observed and f.base.preview==before
    assert recheck(f)['accepted'] and len(f.calls)==1


def test_default_policy_keeps_observed_pregrasp_requirement(tmp_path):
    f=FeedFixture(tmp_path);f.base.preview['candidates'][0]['observed_hand_collision']['pregrasp_collision_free']=False
    f.write();f.select();result=f.prepare()
    assert result['candidates']==[] and result['batch'] is None and f.calls==[]


def test_complete_sim_opt_in_with_real_or_unknown_frame_scene_does_not_call_worker(tmp_path):
    f=FeedFixture(tmp_path)
    f.feed=SimTargetFeed(tmp_path,f,'a'*64,clock=lambda:f.base.now,policy='complete_static_sim_orientation')
    for i,t in enumerate((10.,10.06,10.12)):
        assert not f.feed.record_frame(f.base.frame(i,t),f.base.scene(t))['recorded']
    f.select();result=f.prepare()
    assert result['batch'] is None and not f.calls


@pytest.mark.parametrize('fault,expected',[('stale','preview_age_exceeds_250ms'),
    ('future','preview_capture_timestamp_future'),('clock','preview_clock_mismatch'),
    ('static','motion_static_unavailable'),('service','service_validation_reason'),
    ('capture_unknown','preview_capture_timestamp_unknown')])
def test_recheck_rejection_reports_exact_failed_preview_condition_without_rearming(tmp_path,fault,expected):
    f=FeedFixture(tmp_path);f.select();prepared=f.prepare();original_stamp=prepared['batch'].capture_timestamp
    if fault=='stale':f.base.now+=.251
    elif fault=='future':f.base.preview['capture_timestamp']=f.base.now+.011
    elif fault=='clock':f.base.preview['clock_id']='wrong'
    elif fault=='static':f.base.preview['motion_static']=False
    elif fault=='service':f.base.preview['validation_reason']='camera_stream_expired'
    else:f.base.preview['capture_timestamp']=None
    f.write();result=recheck(f);evidence=result['rejection_evidence']
    assert not result['accepted'] and evidence['phase']=='recheck_preview'
    assert expected in evidence['failed_preview_checks']
    assert evidence['validation_observation_id']=='cal:2' and evidence['model_source_observation_id']=='cal:0'
    assert evidence['selected_candidate_id']=='grasp1' and evidence['model_source_id']
    assert evidence['check_now']==f.base.now and evidence['entered_at']==f.base.now
    if fault=='stale':
        assert evidence['preview_age_s']==pytest.approx(.251)
        assert evidence['preview_capture_timestamp']==10.12
    assert len(json.dumps(evidence,allow_nan=False))<4096
    assert prepared['batch'].capture_timestamp==original_stamp and f.feed.active_candidate_id is None
    assert not recheck(f)['accepted']  # Same old click does not revive after any diagnostic path.


def test_prepare_endpoint_and_recheck_each_report_age_without_renewal(tmp_path):
    f=FeedFixture(tmp_path);f.select();f.delay=.04;prepared=f.prepare()
    evidence=prepared['preview_evidence']
    assert evidence['preview_capture_timestamp']==10.12
    assert evidence['preview_age_s']==pytest.approx(.04)
    assert evidence['endpoint_elapsed_s']==pytest.approx(.04)
    assert evidence['model_source_id']==prepared['batch'].model_source_id
    result=recheck(f);assert result['accepted'] and result['rejection_evidence'] is None
    assert result['preview_evidence']['preview_age_s']==pytest.approx(.04)
    assert result['preview_evidence']['failed_preview_checks']==[]
    assert prepared['batch'].capture_timestamp==10.12


def test_during_check_expiry_records_final_age_and_phase(tmp_path,monkeypatch):
    f=FeedFixture(tmp_path);f.select();assert f.prepare()['batch'] is not None
    original=f.feed.gate.recheck
    def inspect(*args,**kwargs):
        result=original(*args,**kwargs);f.base.now+=.251;return result
    monkeypatch.setattr(f.feed.gate,'recheck',inspect)
    evidence=recheck(f)['rejection_evidence']
    assert evidence['phase']=='recheck_finished'
    assert evidence['preview_age_s']==pytest.approx(.251)
    assert evidence['checked_at']==f.base.now and evidence['elapsed_s']==pytest.approx(.251)
    assert evidence['reason']=='preview_expired_during_handoff_recheck'


def test_rejection_diagnostics_stay_bounded_and_json_finite_for_malformed_input(tmp_path):
    f=FeedFixture(tmp_path);f.select()
    f.base.preview.update(validation_reason='x'*20000,capture_timestamp=None,clock_id=['wrong'])
    f.write();result=f.prepare()
    evidence=result['rejection_evidence'];assert len(evidence['validation_reason'])==256
    assert evidence['preview_capture_timestamp'] is None and evidence['preview_clock_id'] is None
    assert len(json.dumps(evidence,allow_nan=False))<4096
    result=f.prepare(now=10**10000)
    assert result['rejection_evidence']['check_now'] is None and result['batch'] is None
