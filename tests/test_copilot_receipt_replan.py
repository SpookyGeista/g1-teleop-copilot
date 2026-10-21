"""Local typed cancellation contracts; fake worker/physics, no browser or robot."""
from copy import copy, deepcopy
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest

from teleop_assist.copilot.sim_control import SimReceiptReplan
from teleop_assist.copilot.sim_actor import SimReceiptActor
from test_copilot_sim_control import race_ui
from test_copilot_sim_receipt import fixture, receipt, recheck_receipt
from test_copilot_sim_actor import ready, Codec


def changed_ui(f, action='assist_release'):
    actions = dict(assist_held=False)
    if action == 'select': actions['select_candidate_id'] = 'grasp'
    elif action != 'assist_release': actions[action] = True
    return race_ui(f, **actions)


def decision_fixture(*, installed=False, action='assist_release'):
    f, offer = fixture(assisting=True)
    prior = f.session._committed_offset.copy()
    r = receipt(f, offer)
    offered = f.session._committed_offset.copy()
    if installed:
        assert f.session.mark_receipt_installed(r, now=f.now, clock_id='mono')
    ui = changed_ui(f, action)
    decision = recheck_receipt(f, r, ui=ui)
    assert type(decision) is SimReceiptReplan
    return f, r, decision, ui, prior, offered


@pytest.mark.parametrize('action', ['clutch', 'select', 'confirm', 'assist_release'])
@pytest.mark.parametrize('installed', [False, True])
def test_valid_UI_change_burns_receipt_preserves_state_and_rolls_back_only_before_install(action, installed):
    f, r, decision, ui, prior, offered = decision_fixture(installed=installed, action=action)
    assert f.session.clutched and f.session._inflight_receipt is None and f.session._pending is None
    assert f.session._ui_revision < ui['revision']  # The action is not acknowledged here.
    np.testing.assert_array_equal(f.session._committed_offset, offered if installed else prior)
    assert f.session.consume_receipt_replan(decision, r, now=f.now+.006, clock_id='mono')
    assert f.session.clutched and f.session._receipt_replan is None
    assert f.session.status['receipt_was_installed'] is installed
    # The old command is terminal even though the deliberate clutch remains.
    assert recheck_receipt(f, r, ui=ui, now=f.now+.007) is None
    assert not f.session.clutched


def test_replan_waits_for_both_new_source_and_new_UI_then_fresh_guard():
    f, r, decision, ui, _, _ = decision_fixture()
    assert f.session.consume_receipt_replan(decision, r, now=f.now+.006, clock_id='mono')
    original = deepcopy(f.last['packet']); calls = len(f.worker.calls)
    assert not f.tick(packet=original, ui=ui)['allowed']
    assert not f.tick(ui=ui)['allowed']  # New source alone is insufficient.
    assert len(f.worker.calls) == calls and f.session.clutched
    fresh = f.tick(assist_held=False)
    assert fresh['allowed'] and len(f.worker.calls) == calls+1


def test_new_UI_without_new_source_cannot_resume_after_replan():
    f, r, decision, ui, _, _ = decision_fixture()
    assert f.session.consume_receipt_replan(decision, r, now=f.now+.006, clock_id='mono')
    old = deepcopy(f.last['packet'])
    f.revision = ui['revision']
    result = f.tick(packet=old, assist_held=False)
    assert not result['allowed'] and f.session.clutched
    assert result['reason'] == 'waiting_for_fresh_source_and_UI_after_receipt_replan'


@pytest.mark.parametrize('action', ['select', 'confirm', 'clutch'])
def test_unacknowledged_edge_is_processed_state_only_before_resume_gate(action):
    f, r, decision, ui, _, _ = decision_fixture(action=action)
    assert f.session.consume_receipt_replan(decision, r, now=f.now+.006, clock_id='mono')
    calls = len(f.worker.calls)
    result = f.tick(ui=ui)
    assert not result['allowed'] and result['snapshot']['last_ui_revision'] == ui['revision']
    assert f.session.clutched and len(f.worker.calls) == calls
    if action == 'confirm': assert result['snapshot']['confirmed']
    if action == 'select': assert result['snapshot']['selected_candidate_id'] == 'grasp'
    assert not f.tick(ui=ui)['allowed']  # Old one-shot cannot become a heartbeat.


def test_decision_is_not_constructible_copyable_serializable_or_subclassable():
    _, _, decision, _, _, _ = decision_fixture()
    for operation in (SimReceiptReplan, lambda: copy(decision), lambda: deepcopy(decision),
                      lambda: pickle.dumps(decision), lambda: json.dumps(decision),
                      lambda: type('ForgedSubclass', (SimReceiptReplan,), {})):
        with pytest.raises(TypeError): operation()


@pytest.mark.parametrize('attack', ['forged', 'dict', 'receipt_copy', 'receipt_changed', 'replayed',
                                    'expired', 'clock', 'superseded', 'status_string'])
def test_private_decision_cannot_be_forged_replayed_or_extended(attack):
    f, r, decision, ui, _, _ = decision_fixture()
    now, clock = f.now+.006, 'mono'
    if attack == 'forged': decision = object.__new__(SimReceiptReplan)
    elif attack == 'dict': decision = {'reason': 'valid_UI_replan_measured_hold'}
    elif attack == 'receipt_copy': r = replace(r)
    elif attack == 'receipt_changed': object.__setattr__(r, 'valid_until', 999.)
    elif attack == 'replayed':
        assert f.session.consume_receipt_replan(decision, r, now=now, clock_id=clock)
    elif attack == 'expired': now = r.valid_until
    elif attack == 'clock': clock = 'other'
    elif attack == 'superseded': f.tick(ui=ui)
    else:
        f.session._receipt_replan = None
        f.session.status.update(reason='valid_UI_replan_measured_hold', receipt_replan_required=True)
    assert not f.session.consume_receipt_replan(decision, r, now=now, clock_id=clock)
    assert not f.session.clutched and f.session._receipt_replan is None


@pytest.mark.parametrize('stage', ['recheck', 'decision_consume'])
def test_elapsed_work_expiry_with_valid_edge_still_causes_full_hold(monkeypatch, stage):
    f, offer = fixture(assisting=True); r = receipt(f, offer); ui = changed_ui(f)
    if stage == 'recheck':
        original = f.session._operator_sequence
        def delayed(*args):
            original(*args); f.clock += .151
        monkeypatch.setattr(f.session, '_operator_sequence', delayed)
        assert recheck_receipt(f, r, ui=ui) is None
    else:
        decision = recheck_receipt(f, r, ui=ui)
        times = iter([f.clock, f.clock+.151])
        f.session.clock = lambda: next(times)
        assert not f.session.consume_receipt_replan(decision, r, now=f.now+.006, clock_id='mono')
    assert not f.session.clutched and f.session._receipt_replan is None


@pytest.mark.parametrize('fault', ['release', 'owner', 'source_epoch', 'stale', 'expired'])
def test_invalid_source_or_explicit_stop_wins_over_simultaneous_valid_button_change(fault):
    f, offer = fixture(assisting=True); r = receipt(f, offer); ui = changed_ui(f)
    packet = deepcopy(f.last['packet']); now = f.now+.006
    if fault == 'release': ui['release'] = True
    elif fault == 'owner': ui['client_id'] = 'other'
    elif fault == 'source_epoch': packet['source_instance_id'] = 'other'
    elif fault == 'stale': packet['expires_at'] = f.now-.1
    else: now = r.valid_until
    assert recheck_receipt(f, r, ui=ui, operator=packet, now=now) is None
    assert not f.session.clutched and f.session._receipt_replan is None


@pytest.mark.parametrize('action', ['clutch', 'select', 'confirm', 'assist_release'])
def test_direct_actor_prewrite_replan_holds_without_offset_commit_or_ACK(action):
    f, r, bench, actor, apply = ready()
    previous = f.session._inflight_receipt['prior_committed_offset'].copy()
    revision = f.session._ui_revision
    def change():
        f.clock = f.now+.006; f.last['ui'] = changed_ui(f, action)
    actor.codec = Codec(change)
    result = apply()
    assert not result['applied'] and result['receipt_replan'] and bench.commands == [None]
    assert f.session.clutched and f.session._ui_revision == revision
    np.testing.assert_array_equal(f.session._committed_offset, previous)


@pytest.mark.parametrize('replacement', ['opaque', 'dict'])
def test_actor_does_not_trust_fake_decision_or_status_string(monkeypatch, replacement):
    f, r, bench, actor, apply = ready()
    fake = object.__new__(SimReceiptReplan) if replacement == 'opaque' else {'receipt_replan': True}
    f.session.status.update(reason='valid_UI_replan_measured_hold', receipt_replan_required=True)
    monkeypatch.setattr(f.session, 'recheck_receipt_inputs', lambda *args, **kwargs: fake)
    result = apply()
    assert not result['applied'] and not f.session.clutched and bench.commands == [None]


def test_second_direct_actor_cannot_reinstall_an_already_installed_receipt():
    f, r, bench, actor, apply = ready()
    assert apply()['applied']
    other = SimReceiptActor(f.session, bench, codec=Codec(), clock=lambda: f.clock)
    result = other.apply(r, read_inputs=lambda: (f.last['packet'], f.last['ui']))
    assert not result['applied'] and len([q for q in bench.commands if q is not None]) == 1
    assert not f.session.clutched


def test_uninstalled_consumed_receipt_does_not_accumulate_offset_in_later_step():
    f, offer = fixture(assisting=True); previous = f.session._committed_offset.copy()
    receipt(f, offer)
    assert not np.array_equal(f.session._committed_offset, previous)
    f.tick(assist_held=True)
    np.testing.assert_array_equal(f.session._committed_offset, previous)


def test_prewrite_rollback_preserves_previous_nonzero_actually_installed_offset():
    f, offer = fixture(assisting=True); first = receipt(f, offer)
    assert f.session.mark_receipt_installed(first, now=f.now, clock_id='mono')
    installed = f.session._committed_offset.copy()
    assert not np.array_equal(installed, np.eye(3))
    second_offer = f.tick(assist_held=True); second = receipt(f, second_offer)
    assert not np.array_equal(installed, f.session._committed_offset)
    decision = recheck_receipt(f, second, ui=changed_ui(f))
    assert type(decision) is SimReceiptReplan
    np.testing.assert_array_equal(f.session._committed_offset, installed)
    assert f.session.consume_receipt_replan(decision, second, now=f.now+.006, clock_id='mono')


@pytest.mark.parametrize('attack', ['changed', 'wrong_type', 'replayed', 'expired', 'clock'])
def test_install_ack_is_bound_one_shot_and_cannot_renew_authority(attack):
    f, offer = fixture(assisting=True); r = receipt(f, offer)
    now, clock = f.now, 'mono'
    if attack == 'changed': r = replace(r, q14=tuple([0.]*14))
    elif attack == 'wrong_type': r = {'token': r.token}
    elif attack == 'replayed': assert f.session.mark_receipt_installed(r, now=now, clock_id=clock)
    elif attack == 'expired': now = r.valid_until
    else: clock = 'other'
    assert not f.session.mark_receipt_installed(r, now=now, clock_id=clock)
    assert not f.session.clutched and f.session._inflight_receipt is None


@pytest.mark.parametrize('operation', ['install_ack', 'consume_replan'])
def test_clock_callback_revocation_cannot_be_overwritten_by_private_ack(operation):
    f, offer = fixture(assisting=True); r = receipt(f, offer)
    decision = recheck_receipt(f, r, ui=changed_ui(f)) if operation == 'consume_replan' else None
    calls = []
    def clock():
        calls.append(True)
        if len(calls) == 2: f.session.hold('intervening_stop', now=f.now, clock_id='mono')
        return f.clock
    f.session.clock = clock
    if operation == 'install_ack':
        assert not f.session.mark_receipt_installed(r, now=f.now+.006, clock_id='mono')
    else:
        assert not f.session.consume_receipt_replan(decision, r, now=f.now+.006, clock_id='mono')
    assert not f.session.clutched and f.session._inflight_receipt is None


def test_setter_finishing_after_deadline_is_followed_by_full_measured_hold():
    f, r, bench, actor, apply = ready(); command = bench.command
    def delayed(q, **kwargs):
        result = command(q, **kwargs)
        if q is not None: f.clock = r.valid_until
        return result
    bench.command = delayed
    result = apply()
    assert not result['applied'] and result['reason'] == 'receipt_install_ack_rejected'
    assert bench.commands == [(r.q14, r.valid_until), None]
    assert not f.session.clutched


@pytest.mark.parametrize('stage', ['recheck', 'decision_consume'])
def test_shorter_new_UI_lease_wins_even_while_original_receipt_is_fresh(monkeypatch, stage):
    f, offer = fixture(assisting=True); r = receipt(f, offer)
    ui = changed_ui(f); ui['expires_at'] = f.now+.009
    assert ui['expires_at'] < r.valid_until
    if stage == 'recheck':
        original = f.session._operator_sequence
        def delayed(*args):
            original(*args); f.clock += .005
        monkeypatch.setattr(f.session, '_operator_sequence', delayed)
        assert recheck_receipt(f, r, ui=ui, now=f.now+.006) is None
    else:
        decision = recheck_receipt(f, r, ui=ui, now=f.now+.006)
        assert type(decision) is SimReceiptReplan
        assert not f.session.consume_receipt_replan(decision, r, now=f.now+.009, clock_id='mono')
    assert not f.session.clutched
