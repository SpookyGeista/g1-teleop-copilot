"""A conscious UI edge is not a stale motion heartbeat after pending replan."""
from copy import deepcopy

import pytest

from test_copilot_sim_control import Fixture, race_ui, recheck


@pytest.mark.parametrize('action', ['select', 'confirm'])
def test_late_but_valid_one_shot_does_not_run_guard_or_lose_clutch(action):
    f = Fixture(); f.start()
    f.tick(select_candidate_id='grasp')
    offer = f.tick()
    ui = race_ui(f, **({'select_candidate_id': 'grasp'} if action == 'select' else {'confirm': True}))
    assert not recheck(f, ui=ui)
    assert f.session.clutched
    f.worker.delay = .08  # Would exhaust a 150-ms event already ~100 ms old.
    count = len(f.worker.calls)
    result = f.tick(ui=ui, delta=.105)
    assert result['reason'] == 'ui_action_waiting_for_fresh_control_sample'
    assert not result['allowed'] and result['snapshot']['clutched']
    assert result['snapshot']['last_ui_revision'] == ui['revision']
    assert result['snapshot']['confirmed'] is (action == 'confirm')
    assert len(f.worker.calls) == count
    assert f.consume(offer) is None  # The previous canceled offer never revives.
    assert f.session.clutched


@pytest.mark.parametrize('action', ['select', 'confirm'])
def test_repeated_action_cannot_become_motion_as_new_pose_packets_arrive(action):
    f = Fixture(); f.start(); f.tick(select_candidate_id='grasp')
    f.tick(**({'select_candidate_id': 'grasp'} if action == 'select' else {'confirm': True}))
    ui = deepcopy(f.last['ui'])
    count = len(f.worker.calls)
    for _ in range(3):
        result = f.tick(ui=ui)
        assert not result['allowed'] and result['snapshot']['clutched']
        assert len(f.worker.calls) == count
    # Fresh control heartbeat + new operator source can resume through the guard.
    result = f.tick()
    assert result['allowed'] and len(f.worker.calls) == count + 1


def test_new_heartbeat_without_new_pose_still_cannot_resume_after_confirmation():
    f = Fixture(); f.start(); f.tick(select_candidate_id='grasp'); f.tick(confirm=True)
    packet = deepcopy(f.last['packet'])
    count = len(f.worker.calls)
    result = f.tick(packet=packet)
    assert not result['allowed'] and result['snapshot']['confirmed']
    assert result['reason'] == 'waiting_for_new_XR_sample_after_clutch'
    assert len(f.worker.calls) == count
    assert f.tick(assist_held=True)['allowed']


def test_expired_confirmation_still_revokes_clutch_and_never_becomes_current():
    f = Fixture(); f.start(); f.tick(select_candidate_id='grasp')
    ui = race_ui(f, confirm=True)
    result = f.tick(ui=ui, delta=.16)
    assert not result['allowed'] and not result['snapshot']['clutched']
    assert 'ui_' in result['reason']
    assert not f.tick()['allowed']


def test_action_expires_during_controller_work_without_becoming_confirmed():
    f = Fixture(); f.start(); f.tick(select_candidate_id='grasp')
    original = f.session.controller.step
    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        f.clock += .16
        return result
    f.session.controller.step = delayed
    count = len(f.worker.calls)
    result = f.tick(confirm=True)
    assert result['reason'] == 'ui_action_lease_expired_during_transition'
    assert not result['snapshot']['clutched'] and not result['snapshot']['confirmed']
    assert not result['assistance_applied'] and not result['allowed']
    assert len(f.worker.calls) == count
