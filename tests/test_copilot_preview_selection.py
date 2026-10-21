"""Early live choice is state only. CPU fixtures do not prove robot motion."""
from copy import deepcopy
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from teleop_assist.copilot.sim_control import SimUIIntent
from teleop_assist.copilot.sim_actor import SimReceiptActor
from teleop_assist.copilot.types import PreviewSelection, ProposalBatch, GraspCandidate
from test_copilot_sim_control import Fixture
from test_copilot_sim_target_feed import FeedFixture, pose


def signature(ui):
    return hashlib.sha256(json.dumps(vars(SimUIIntent.from_packet(ui)),
        sort_keys=True,allow_nan=False).encode()).hexdigest()


def preview_for(f, ui, *, stamp=None):
    return PreviewSelection('target',1,'object_signature','preview_observation',
        f.now if stamp is None else stamp,'mono','cal','grasp',
        tuple(tuple(float(x) for x in row) for row in f.proposal().candidates[0].base_ee),
        .9,'fake_test_only','model1',9.,'mono',ui['revision'],signature(ui),ui['client_id'])


def select(f, **changes):
    f.now += .02;f.clock=f.now;f.sequence+=1;f.revision+=1
    ui=dict(revision=f.revision,capture_timestamp=f.now,expires_at=f.now+.15,
        clock_id='mono',client_id='test_ui',select_candidate_id='grasp')
    ui.update(changes)
    preview=preview_for(f,ui)
    return preview,ui


def commit(f, preview, ui, *, packet=None,state=None,scene=None):
    return f.session.step(f.packet() if packet is None else packet,ui,
        f.state() if state is None else state,f.scene() if scene is None else scene,preview,
        measured_left_base_ee=f.left,measured_right_base_ee=f.right,
        now=f.now,clock_id='mono',calibration_id='cal')


def authorized(f, preview):
    return ProposalBatch(preview.target_id,'fresh_endpoint',f.now,'mono','cal',(
        GraspCandidate(preview.candidate_id,np.array(preview.base_ee),.9,True,True,True,
                       preview.source+'|'+preview.model_source_id),),True,True,
        preview.model_source_id,preview.model_source_capture_timestamp,preview.model_source_clock_id)


def test_early_commit_precedes_later_44ms_endpoint_33ms_barrier_without_replay():
    f=Fixture();f.tick();p,u=select(f)
    # This click is already 75ms old, mirroring an arrival-dependent path.
    u.update(capture_timestamp=f.now-.075,expires_at=f.now+.075)
    p=replace(p,ui_signature=signature(u))
    result=commit(f,p,u)
    assert result['snapshot']['selected_candidate_id']=='grasp'
    assert result['snapshot']['preview_selection_pending']
    assert not result['snapshot']['clutch_available']
    assert not result['allowed'] and result['q14_command'] is None
    assert not f.worker.calls and f.session.consume_receipt(result,now=f.now,clock_id='mono',
        measured_state=f.state(),scene=f.scene()) is None
    # consume(None authority) deliberately resets the session; use a fresh
    # fixture for the separate successful next-tick continuation.
    f=Fixture();f.tick();p,u=select(f)
    u.update(capture_timestamp=f.now-.075,expires_at=f.now+.075);p=replace(p,ui_signature=signature(u))
    result=commit(f,p,u);f.now+=.044+.033;f.clock=f.now
    assert f.now>u['expires_at']
    result=f.tick(batch=authorized(f,p),delta=.001)
    assert result['snapshot']['selected_candidate_id']=='grasp'
    assert not result['snapshot']['preview_selection_pending']
    assert result['snapshot']['clutch_available']
    assert not result['snapshot']['confirmed'] and not result['snapshot']['clutched']
    assert not result['allowed'] and not f.worker.calls


@pytest.mark.parametrize('offset,allowed',[(0.,False),(1e-9,False),(-1e-9,True)])
def test_actual_end_clock_boundary_consumes_expired_revision_without_revival(offset,allowed):
    f=Fixture();f.tick();p,u=select(f)
    initial=f.now;values=iter([initial,u['expires_at']+offset])
    f.session.clock=lambda:next(values)
    result=commit(f,p,u)
    assert (result['snapshot'].get('selected_candidate_id')=='grasp') is allowed
    assert result['snapshot']['last_ui_revision']==u['revision']
    assert not result['allowed'] and not f.worker.calls
    if allowed:
        assert result['snapshot']['preview_selection']['committed_at']==u['expires_at']+offset
    else:
        f.session.clock=lambda:f.clock
        retry=commit(f,p,u)
        assert retry['snapshot'].get('selected_candidate_id') is None


@pytest.mark.parametrize('action',['confirm','clutch','assist_held','release'])
def test_preview_never_accepts_other_action_or_combination(action):
    f=Fixture();f.tick();p,u=select(f)
    u[action]=True
    result=commit(f,p,u)
    assert not result['allowed'] and not result['snapshot']['confirmed']
    assert result['snapshot'].get('selected_candidate_id') is None and not f.worker.calls


@pytest.mark.parametrize('change',['pose','source','model','target','calibration','owner','epoch','release'])
def test_changed_binding_after_commit_revokes_without_rearm(change):
    f=Fixture();f.tick();p,u=select(f);assert commit(f,p,u)['snapshot']['selected_candidate_id']=='grasp'
    batch=authorized(f,p);options={}
    if change=='pose':
        changed=batch.candidates[0].base_ee.copy();changed[0,3]+=.001
        batch=replace(batch,candidates=(replace(batch.candidates[0],base_ee=changed),))
    if change=='source':batch=replace(batch,candidates=(replace(batch.candidates[0],source='changed'),))
    if change=='model':batch=replace(batch,model_source_id='changed')
    if change=='target':batch=replace(batch,target_id='changed')
    if change=='calibration':options['calibration_id']='changed'
    if change=='owner':options['ui']=dict(u,revision=u['revision']+1,select_candidate_id=None,client_id='other')
    if change=='epoch':options['packet']=dict(f.packet(),source_instance_id='new')
    if change=='release':options['release']=True
    result=f.tick(batch=batch,**options)
    assert not result['allowed'] and result['snapshot'].get('selected_candidate_id') is None
    assert not f.worker.calls
    result=f.tick(batch=authorized(f,p))
    assert result['snapshot'].get('selected_candidate_id') is None


def raw_prepare(f, **changes):
    args=dict(intent=f.intent,session_status={},robot_state=f.base.state(),scene=f.base.scene(),
        left_base_ee=pose(.3,.2,.9),now=f.base.now,clock_id='mono',calibration_id='cal')
    args.update(changes)
    return f.feed.prepare(**args)


def test_feed_returns_separate_immutable_preview_without_endpoint_or_barrier(tmp_path,monkeypatch):
    f=FeedFixture(tmp_path);f.select()
    monkeypatch.setattr('teleop_assist.copilot.sim_target_feed.await_preview_frame',
        lambda *a,**kw:pytest.fail('A preview tick cannot await the endpoint camera'))
    result=raw_prepare(f,current_frame={'ignored':'preview_only'})
    p=result['preview_selection']
    assert type(p) is PreviewSelection and not isinstance(p,ProposalBatch)
    assert result['batch'] is None and result['endpoint_evidence'] is None
    assert result['frame_barrier'] is None and not f.calls
    saved=p.base_ee;f.base.preview['candidates'][0]['base_ee'][0][3]+=.01
    assert p.base_ee==saved and isinstance(p.base_ee,tuple)
    assert not hasattr(p,'valid') and not hasattr(p,'candidates')


def test_failed_commit_discards_feed_only_staging_and_neutral_cannot_revive(tmp_path):
    f=FeedFixture(tmp_path);f.select();result=raw_prepare(f)
    assert result['preview_selection'] is not None
    assert not f.feed.finish_preview_selection({'allowed':False,'snapshot':{'last_ui_revision':f.intent.revision}})
    assert f.feed.active_candidate_id is None
    assert raw_prepare(f)['batch'] is None and not f.calls


def test_missing_commit_ack_never_runs_endpoint_even_with_same_click(tmp_path):
    f=FeedFixture(tmp_path);f.select();assert raw_prepare(f)['preview_selection'] is not None
    rejected=raw_prepare(f)
    assert rejected['batch'] is None and 'not_committed' in rejected['reason']
    assert not f.calls


@pytest.mark.parametrize('change',['target_revision','candidate_pose','candidate_source','model_source','clock','calibration'])
def test_actual_target_reread_before_commit_rejects_changed_binding_without_endpoint(tmp_path,change):
    f=FeedFixture(tmp_path);f.select();p=raw_prepare(f)['preview_selection']
    if change=='target_revision':f.selection['revision']+=1
    if change=='candidate_pose':f.base.preview['candidates'][0]['base_ee'][0][3]+=.001
    if change=='candidate_source':f.base.preview['candidates'][0]['source']='other_model'
    if change=='model_source':f.base.preview['model_source_capture_timestamp']+=.001
    if change=='clock':f.base.preview['clock_id']='other_clock'
    if change=='calibration':f.base.preview['calibration_id']='other_calibration'
    f.write()
    checked=f.feed.recheck_preview_selection(p,now=f.base.now,clock_id='mono',calibration_id='cal')
    assert not checked['accepted'] and not checked['motion_authorized']
    assert f.feed.active_candidate_id is None and not f.calls
    assert raw_prepare(f)['preview_selection'] is None


def test_preview_last_target_read_is_inert_and_does_not_refresh_capture(tmp_path):
    f=FeedFixture(tmp_path);f.select();p=raw_prepare(f)['preview_selection'];before=deepcopy(p)
    checked=f.feed.recheck_preview_selection(p,now=f.base.now,clock_id='mono',calibration_id='cal')
    assert checked['accepted'] and not checked['motion_authorized']
    assert p==before and not f.calls


def test_preview_or_its_json_cannot_be_consumed_or_become_a_normal_authorized_batch():
    f=Fixture();f.tick();p,u=select(f)
    with pytest.raises(TypeError):replace(p,valid=True)
    with pytest.raises(TypeError):ProposalBatch(**vars(p))
    result=commit(f,dict(vars(p)),u)
    assert result['reason']=='authorized_proposal_batch_required'
    assert result['snapshot'].get('selected_candidate_id') is None and not f.worker.calls
    from test_copilot_sim_actor import Bench
    bench=Bench(f);actor=SimReceiptActor(f.session,bench,clock=lambda:f.clock)
    denied=actor.apply(p,read_inputs=lambda:pytest.fail('Preview cannot reach actor inputs'))
    assert not denied['applied'] and bench.commands==[None]


@pytest.mark.parametrize('change',['different_ui_content','source_epoch','source_payload'])
def test_last_actual_input_sample_revokes_staged_selection(change):
    f=Fixture();f.tick();p,u=select(f);packet=f.packet()
    if change=='different_ui_content':u['capture_timestamp']-=.0001
    if change=='source_epoch':packet['source_instance_id']='new_epoch'
    if change=='source_payload':
        # Reuse the established previous sequence with changed coordinates.
        packet=deepcopy(f.last['packet']);packet['left_base_ee'][0][3]+=.001
    result=commit(f,p,u,packet=packet)
    assert result['snapshot'].get('selected_candidate_id') is None
    assert not result['allowed'] and not f.worker.calls


def test_reselection_replaces_clutch_and_confirmation_and_burns_old_offer():
    f=Fixture();old=f.start();assert old['allowed']
    p,u=select(f);result=commit(f,p,u)
    assert result['snapshot']['selected_candidate_id']=='grasp'
    assert not result['snapshot']['confirmed'] and not result['snapshot']['clutched']
    assert not result['snapshot']['clutch_available'] and not result['allowed']
    assert f.consume(old) is None


def test_actual_feed_and_session_two_ticks_keep_choice_after_original_click_expires(tmp_path):
    feed=FeedFixture(tmp_path);core=Fixture()
    core.left=np.asarray(pose(.3,.2,.9));core.right=np.asarray(pose(.4,-.18,.9))
    core.now=10.10;core.tick(delta=.01)  # Consume genuine session startup baseline.
    core.now=core.clock=feed.base.now
    core.sequence+=1;core.revision+=1
    # Genuine original click is 75ms old, not renewed by either feed or core.
    ui=dict(revision=core.revision,capture_timestamp=core.now-.075,expires_at=core.now+.075,
        clock_id='mono',client_id='test_ui',select_candidate_id='grasp1')
    feed.intent=SimUIIntent.from_packet(ui)
    prepared=raw_prepare(feed,session_status=core.session.status)
    preview=prepared['preview_selection']
    assert preview is not None and prepared['batch'] is None and not feed.calls
    committed=commit(core,preview,ui,state=feed.base.state(),scene=feed.base.scene())
    assert committed['snapshot']['selected_candidate_id']=='grasp1'
    assert feed.feed.finish_preview_selection(committed)
    assert not committed['allowed'] and not core.worker.calls and not feed.calls
    assert not committed['snapshot']['clutch_available']
    # A fresh NEUTRAL next tick uses the already committed choice. The real
    # Feed endpoint evidence fixture accounts for 44+33ms elapsed work.
    feed.base.now+=.02;feed.delay=.044+.033
    feed.intent=SimUIIntent(core.revision+1,feed.base.now,'mono','test_ui')
    ready=raw_prepare(feed,session_status=core.session.status)
    assert ready['batch'] is not None and len(feed.calls)==1
    assert feed.base.now>ui['expires_at']
    core.now=core.clock=feed.base.now;core.sequence+=1;core.revision+=1
    result=commit(core,ready['batch'],feed.intent,state=feed.base.state(),scene=feed.base.scene())
    assert result['snapshot']['selected_candidate_id']=='grasp1'
    assert not result['snapshot']['preview_selection_pending']
    assert result['snapshot']['clutch_available']
    assert not result['snapshot']['confirmed'] and not result['snapshot']['clutched']
    assert not result['allowed'] and not core.worker.calls
    assert result['snapshot']['preview_selection']['ui_revision']==ui['revision']


def test_preview_needs_later_real_batch_then_separate_clutch_confirmation_and_guard():
    f=Fixture();f.tick();p,u=select(f);assert commit(f,p,u)['snapshot']['selected_candidate_id']=='grasp'
    # Flags left unvalidated cannot make the pending choice ready even when
    # a caller wraps the numeric preview in a normal-looking batch.
    invalid=replace(authorized(f,p),candidates=(replace(authorized(f,p).candidates[0],collision_valid=False),))
    denied=f.tick(batch=invalid,clutch=True)
    assert not denied['snapshot']['clutched'] and not denied['snapshot']['confirmed']
    assert denied['snapshot'].get('selected_candidate_id') is None and not f.worker.calls
    # A deliberate NEW selection, genuine validated batch, NEW clutch and
    # confirmation are all required. The only worker call is the later guard.
    p,u=select(f);assert commit(f,p,u)['snapshot']['selected_candidate_id']=='grasp'
    ready=f.tick(batch=authorized(f,p))
    assert ready['snapshot']['clutch_available'] and not f.worker.calls
    clutched=f.tick(batch=authorized(f,p),clutch=True)
    assert clutched['snapshot']['clutched'] and not clutched['snapshot']['confirmed']
    assert not clutched['allowed'] and not f.worker.calls
    confirmed=f.tick(batch=authorized(f,p),confirm=True)
    assert confirmed['snapshot']['confirmed'] and not confirmed['allowed'] and not f.worker.calls
    guarded=f.tick(batch=authorized(f,p),assist_held=True)
    assert guarded['allowed'] and len(f.worker.calls)==1


@pytest.mark.parametrize('which',['operator','state','scene','preview'])
def test_every_original_source_deadline_is_rechecked_at_actual_commit_end(which):
    f=Fixture()
    if which=='operator':f.now=9.8
    f.tick()
    if which=='operator':f.now=9.98
    p,u=select(f)
    packet,state,scene=f.packet(),f.state(),f.scene()
    if which=='operator':
        f.now-=.149;packet=f.packet();f.now+=.149
    if which=='state':state['acquisition_timestamp']=f.now-.149
    if which=='scene':scene['acquisition_timestamp']=f.now-.149
    if which=='preview':p=replace(p,capture_timestamp=f.now-.249)
    end=f.now+.0011;values=iter([f.now,end]);f.session.clock=lambda:next(values)
    result=commit(f,p,u,packet=packet,state=state,scene=scene)
    assert result['reason']=='preview_selection_lease_expired_during_transition'
    assert result['snapshot'].get('selected_candidate_id') is None and not result['allowed']
    assert not f.worker.calls


def test_late_release_or_changed_selection_cannot_hide_behind_staged_choice():
    for replacement in ({'release':True,'select_candidate_id':None},
                        {'select_candidate_id':'different'}):
        f=Fixture();f.tick();p,u=select(f)
        fresh=dict(u,revision=u['revision']+1,**replacement)
        result=commit(f,p,fresh)
        assert result['snapshot'].get('selected_candidate_id') is None
        assert not result['allowed'] and not f.worker.calls


def test_runtime_keeps_one_real_step_trace_and_same_five_hold_substeps():
    # Static orchestration regression only: actual GPU execution remains a
    # separate acceptance. Feed/Core behavior above is executed, not mocked.
    source=Path(__file__).parents[1]/'tools/copilot/sim_runtime.py'
    tree=ast.parse(source.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in ast.walk(main) if isinstance(n,ast.While))
    calls=[n for n in ast.walk(loop) if isinstance(n,ast.Call)]
    session_steps=[n for n in calls if ast.unparse(n.func)=='session.step']
    traces=[n for n in calls if ast.unparse(n.func)=='input_trace.capture'
            and any(k.arg=='phase' and isinstance(k.value,ast.Constant)
                    and k.value.value=='session_step' for k in n.keywords)]
    physics=[n for n in calls if ast.unparse(n.func)=='bench.step']
    assert len(session_steps)==len(traces)==len(physics)==1
    assert ast.literal_eval(physics[0].args[0])==5
    assert not any(isinstance(n,ast.Continue) for n in ast.walk(loop))
    selection_read=next(n for n in ast.walk(loop) if isinstance(n,ast.IfExp)
        and ast.unparse(n.test)=='preview_selection is not None'
        and isinstance(n.body,ast.Name) and n.body.id=='latest_step_intent')
    assert selection_read.lineno<session_steps[0].lineno<physics[0].lineno


def test_actual_runtime_expression_never_reports_old_guard_as_selection_tick_work():
    tree=ast.parse((Path(__file__).parents[1]/'tools/copilot/sim_runtime.py').read_text())
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='guard_worker_timing' for t in n.targets))
    code=compile(ast.Module(body=[assignment],type_ignores=[]),'<actual_runtime_assignment>','exec')
    previous={'operation':'validate','request_id':19,'client_wall_s':.04}
    namespace={'worker':SimpleNamespace(last_call_timing=previous),'preview_selection':object()}
    exec(code,namespace)
    assert namespace['guard_worker_timing'] is None
    namespace['preview_selection']=None;exec(code,namespace)
    assert namespace['guard_worker_timing'] is previous


@pytest.mark.parametrize('accepted',[False,True])
def test_actual_scripted_trial_bookkeeping_waits_for_endpoint_following_preview(accepted,tmp_path):
    tree=ast.parse((Path(__file__).parents[1]/'tools/copilot/sim_runtime.py').read_text())
    # Execute the real two fixture-only branches, isolating disk serialization
    # from the state machine; no application or physics runtime is imported.
    blocks=[n for n in ast.walk(tree) if isinstance(n,ast.If)
        and isinstance(n.test,ast.BoolOp)
        and 'args.scripted_test' in ast.unparse(n.test)
        and 'not scenario_started' in ast.unparse(n.test)
        and (any(isinstance(t,ast.Name) and t.id=='script_pending_candidate'
                 for a in n.body if isinstance(a,ast.Assign) for t in a.targets)
             or 'script_pending_candidate is not None' in ast.unparse(n.test))]
    assert len(blocks)==2
    code=compile(ast.Module(body=sorted(blocks,key=lambda n:n.lineno),type_ignores=[]),'<actual_script_trial>','exec')
    ns=dict(args=SimpleNamespace(scripted_test=True,output=tmp_path),scenario_started=False,
        intent={'select_candidate_id':'grasp'},script_pending_candidate=None,script_candidate='grasp',
        preview_selection=object(),script_selection_trials=[],live_reason='preview',
        bundle={'batch':None,'endpoint_evidence':None},state={},scene={},poses={'left':np.eye(4)},
        write_audit=lambda *a:None)
    exec(code,ns)
    assert ns['script_pending_candidate']=='grasp' and ns['script_selection_trials']==[]
    ns.update(intent={},preview_selection=None,live_reason='endpoint',
              bundle={'batch':object() if accepted else None,'endpoint_evidence':{}})
    exec(code,ns)
    assert ns['script_pending_candidate'] is None
    assert ns['script_selection_trials'][0]['candidate_id']=='grasp'
    assert ns['script_selection_trials'][0]['accepted'] is accepted
    assert ns['script_candidate']==('grasp' if accepted else None)
