"""Pre-install capacity refusals are not motion permission or renewed leases.

Pure core tests and audit-guarded actual codec/original-class tests use a fake
worker and articulation. No DDS participant, Isaac, GPU or robot is started.
"""
import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from teleop_assist.copilot.sim_control import SimCommandReceipt, SimReceiptReplan
from test_copilot_sim_receipt import fixture, receipt
from test_copilot_sim_control import race_ui
from test_copilot_unitree_codec import child


RESERVE = .060


def test_postinstallation_watchdog_source_is_byte_unchanged():
    path=Path(__file__).resolve().parents[1]/'teleop_assist/copilot/original_xr_sender.py'
    source=path.read_text()
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='OriginalSimReceiptSink')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='expire_before_physics_step')
    assert hashlib.sha256(ast.get_source_segment(source,method).encode()).hexdigest()==(
        '7ed55c1dea92ccc9e9107dd310cddde6eee5a5a402833dc3aad9e273a6f40857')


def prepared():
    f, offer = fixture(assisting=True)
    previous = f.session._committed_offset.copy()
    return f, receipt(f, offer), previous


def check(f, r, *, reserve=RESERVE, require_uninstalled=True, packet=None, ui=None, now=None):
    return f.session.recheck_receipt_inputs(r,
        f.last['packet'] if packet is None else packet,
        f.last['ui'] if ui is None else ui,
        now=f.clock if now is None else now, clock_id='mono',
        require_uninstalled=require_uninstalled, execution_reserve_s=reserve)


def deferred():
    f, r, previous = prepared()
    f.clock = r.valid_until-.05
    decision = check(f, r)
    assert type(decision) is SimReceiptReplan
    return f, r, decision, previous


@pytest.mark.parametrize('direction,admitted', [(-1,True),(0,True),(1,False)])
def test_exact_final_admission_cutoff_and_neighbouring_floats(direction, admitted):
    f, r, previous = prepared()
    cutoff = r.valid_until-RESERVE
    f.clock = cutoff if direction==0 else float(np.nextafter(cutoff, np.inf if direction>0 else -np.inf))
    result = check(f, r)
    budget = f.session.status['execution_budget']
    assert budget['checked_at']==f.clock and budget['latest_admission_at']==cutoff
    assert budget['receipt_valid_until']==r.valid_until and budget['admitted'] is admitted
    assert budget['motion_authorized'] is False and not budget['worst_case_execution_time_guaranteed']
    if admitted:
        assert type(result) is SimCommandReceipt and result.valid_until==r.valid_until
    else:
        assert type(result) is SimReceiptReplan
        np.testing.assert_array_equal(f.session._committed_offset, previous)


def test_capacity_decision_is_bound_burned_unacknowledged_and_preserves_valid_choice():
    f, r, decision, previous = deferred()
    revision = f.session._ui_revision
    assert f.session.status['reason']=='receipt_execution_budget_replan'
    assert f.session.status['receipt_replan_kind']=='execution_budget'
    assert f.session._receipt_replan['receipt'] is r
    assert f.session._receipt_replan['deadline']==r.valid_until
    assert f.session._pending is None and f.session._inflight_receipt is None
    assert f.session.status['snapshot']['confirmed'] and f.session.clutched
    assert f.session.consume_receipt_replan(decision, r, now=f.clock, clock_id='mono')
    assert f.session.clutched and f.session._ui_revision==revision
    np.testing.assert_array_equal(f.session.controller._offset, previous)
    assert f.session._receipt_replan is None
    assert check(f,r) is None and not f.session.clutched


@pytest.mark.parametrize('bad', [0,-.001,True,False,'0.06',float('nan'),float('inf'),-float('inf'),[],{}])
def test_invalid_reserve_is_full_hold(bad):
    f,r,_=prepared()
    assert check(f,r,reserve=bad) is None and not f.session.clutched


def test_disabled_legacy_api_does_not_invent_budget_exception():
    f,r,_=prepared();f.clock=r.valid_until-.001
    assert type(check(f,r,reserve=None)) is SimCommandReceipt
    assert 'execution_budget' not in f.session.status


@pytest.mark.parametrize('installed,require', [(False,False),(True,True),(True,False)])
def test_reserve_is_only_legal_for_explicit_uninstalled_checks(installed,require):
    f,r,_=prepared()
    if installed:assert f.session.mark_receipt_installed(r,now=f.clock,clock_id='mono')
    assert check(f,r,require_uninstalled=require) is None and not f.session.clutched


@pytest.mark.parametrize('fault', ['expired','release','owner','epoch','stale','malformed','changed_q','changed_token'])
def test_invalidity_wins_over_insufficient_capacity(fault):
    f,r,_=prepared();f.clock=r.valid_until-.01
    packet=deepcopy(f.last['packet']);ui=deepcopy(f.last['ui'])
    if fault=='expired':f.clock=r.valid_until
    elif fault=='release':ui=race_ui(f,release=True)
    elif fault=='owner':ui=race_ui(f,client_id='other')
    elif fault=='epoch':packet['source_instance_id']='other'
    elif fault=='stale':packet['expires_at']=f.clock-.001
    elif fault=='malformed':packet['tracking_valid']=False
    elif fault=='changed_q':r=replace(r,q14=tuple([0.]*14))
    elif fault=='changed_token':r=replace(r,token='other')
    assert check(f,r,packet=packet,ui=ui) is None
    assert not f.session.clutched and f.session._receipt_replan is None


@pytest.mark.parametrize('action', ['clutch','confirm','select','assist_release'])
def test_genuine_UI_change_keeps_its_own_existing_category(action):
    from test_copilot_receipt_replan import changed_ui
    f,r,_=prepared();f.clock=r.valid_until-.05
    decision=check(f,r,ui=changed_ui(f,action))
    assert type(decision) is SimReceiptReplan
    assert f.session.status['receipt_replan_kind']=='valid_UI_change'
    assert f.session.status['reason']!='receipt_execution_budget_replan'
    assert 'execution_budget' not in f.session.status


@pytest.mark.parametrize('fault', ['forged','dict','receipt_copy','receipt_changed','replayed','expired','clock','superseded','status'])
def test_budget_decision_cannot_be_forged_replayed_extended_or_publicly_authorized(fault):
    f,r,decision,_=deferred();now=f.clock;clock='mono'
    if fault=='forged':decision=object.__new__(SimReceiptReplan)
    elif fault=='dict':decision={'receipt_replan_kind':'execution_budget'}
    elif fault=='receipt_copy':r=replace(r)
    elif fault=='receipt_changed':object.__setattr__(r,'valid_until',999.)
    elif fault=='replayed':assert f.session.consume_receipt_replan(decision,r,now=now,clock_id=clock)
    elif fault=='expired':f.clock=now=r.valid_until
    elif fault=='clock':clock='other'
    elif fault=='superseded':f.session.hold('stop',now=now,clock_id=clock)
    elif fault=='status':
        f.session._receipt_replan=None
        f.session.status.update(receipt_replan_kind='execution_budget',receipt_replan_required=True)
    assert not f.session.consume_receipt_replan(decision,r,now=now,clock_id=clock)
    assert not f.session.clutched


@pytest.mark.parametrize('stage', ['recheck','consume'])
@pytest.mark.parametrize('fault', ['expiry','backwards','nan','revocation'])
def test_actual_end_clock_and_intervening_stop_cannot_be_hidden(stage,fault):
    f,r,decision,_=deferred() if stage=='consume' else (*prepared()[:2],None,None)
    f.clock=r.valid_until-.02;now=f.clock;calls=[]
    def clock():
        calls.append(True)
        if len(calls)==2:
            if fault=='expiry':return r.valid_until
            if fault=='backwards':return now-.001
            if fault=='nan':return float('nan')
            f.session.hold('intervening_stop',now=now,clock_id='mono')
        return now
    f.session.clock=clock
    result=(check(f,r,now=now) if stage=='recheck' else
            f.session.consume_receipt_replan(decision,r,now=now,clock_id='mono'))
    assert not result and not f.session.clutched


def test_clock_delay_before_method_entry_is_included_in_budget():
    f,r,_=prepared();caller_now=r.valid_until-.061;f.clock=caller_now+.002
    assert type(check(f,r,now=caller_now)) is SimReceiptReplan
    assert f.session.status['execution_budget']['checked_at']==f.clock


@pytest.mark.parametrize('new_source,new_ui', [(True,False),(False,True),(False,False),(True,True)])
def test_budget_resume_needs_both_new_inputs_and_fresh_guard(new_source,new_ui):
    f,r,decision,_=deferred()
    assert f.session.consume_receipt_replan(decision,r,now=f.clock,clock_id='mono')
    old=deepcopy(f.last);calls=len(f.worker.calls);f.now=f.clock
    outcome=f.tick(delta=.005,packet=None if new_source else old['packet'],
        ui=None if new_ui else old['ui'],assist_held=True)
    assert f.session.clutched
    assert outcome['allowed'] is (new_source and new_ui)
    assert len(f.worker.calls)==calls+int(new_source and new_ui)


def test_prior_nonzero_installed_offset_survives_repeated_uninstalled_budget_rejections():
    f,r,_=prepared()
    assert f.session.mark_receipt_installed(r,now=f.clock,clock_id='mono')
    installed=f.session._committed_offset.copy()
    assert not np.array_equal(installed,np.eye(3))
    for held in (True,False,True,False):
        offer=f.tick(assist_held=held,delta=.02)
        r=receipt(f,offer)
        assert not np.array_equal(f.session._committed_offset,installed)
        f.clock=r.valid_until-.05;decision=check(f,r)
        assert type(decision) is SimReceiptReplan
        assert f.session.consume_receipt_replan(decision,r,now=f.clock,clock_id='mono')
        np.testing.assert_array_equal(f.session._committed_offset,installed)
        f.now=f.clock


def native_sink_case(codec, fault=None):
    from test_copilot_original_xr_sender import rig,assert_equal
    f=rig(codec,assisting=True);before=f.session._inflight_receipt['prior_committed_offset'].copy()
    f.clock=f.receipt.valid_until-.015
    if fault=='scene':f.sim_scene['obstacle'][0,3]+=.01
    elif fault=='target':f.target_check=lambda:dict(accepted=False)
    elif fault=='finger':f.robot.data.joint_pos[0,f.robot.joint_names.index('right_hand_index_0_joint')]+=.01
    elif fault=='stop':f.last['ui']=race_ui(f,release=True)
    elif fault=='expiry':f.clock=f.receipt.valid_until
    result=f.publish()
    assert not result['applied'] and not result['write_attempted'] and not result['delivery_unknown']
    assert len(f.target_calls)==1
    assert_equal(f.target_calls[-1],f.robot.data.joint_pos)
    assert f.controller.lowcmd_publisher.write_attempts==0
    if fault is not None:
        assert not f.session.clutched and not result.get('receipt_replan')
        return
    assert result['reason']=='execution_budget_replan_measured_hold' and result['receipt_replan']
    budget=result['execution_budget']
    assert budget['minimum_remaining_s']==.06 and budget['checked_at']==f.clock
    assert budget['receipt_valid_until']==f.receipt.valid_until and not budget['admitted']
    assert f.session.clutched and f.session.status['snapshot']['confirmed']
    assert_equal(f.session._committed_offset,before)
    assert 'receipt_installed_at' not in f.session.status
    assert f.sink._last_receipt is None and f.sink._last_deadline is None
    for _ in range(5):
        assert not f.sink.expire_before_physics_step()
        assert f.session.clutched
        assert_equal(f.target_calls[-1],f.robot.data.joint_pos)
    assert not f.publish()['applied'] and not f.session.clutched


def test_actual_codec_original_sink_capacity_refusal_and_invalidity_precedence():
    child("sys.path.insert(0,'tests')\nfrom test_copilot_execution_budget import native_sink_case\n"
          "for fault in (None,'scene','target','finger','stop','expiry'):\n"
          " native_sink_case(codec,fault)")


def test_native_sink_normal_write_and_postinstallation_expiry_remain_hard_hold():
    child("""sys.path.insert(0,'tests')
from test_copilot_original_xr_sender import rig,assert_equal
f=rig(codec,assisting=True)
result=f.publish()
assert result['applied'] and result['execution_budget']['admitted']
assert result['execution_budget']['minimum_remaining_s']==.06
f.clock=f.receipt.valid_until
assert not f.sink.expire_before_physics_step()
assert not f.session.clutched
assert f.sink.last_physics_interlock['reason']=='original_sender_lease_not_current'
assert_equal(f.target_calls[-1],f.robot.data.joint_pos)
""")
