"""Endpoint work must not pin a fresh pose to an older VR heartbeat lease."""
from copy import deepcopy
from dataclasses import replace

import pytest

from tools.copilot.sim_runtime import intent_after_endpoint
from test_copilot_sim_control import Fixture, recheck


@pytest.mark.parametrize('live_ai', [False, True])
def test_fresh_heartbeat_keeps_its_own_lease_after_endpoint_delay(live_ai):
    f = Fixture(); f.start()
    prepared = deepcopy(f.last['ui'])
    prepared['expires_at'] = f.now + .15
    latest = dict(prepared, revision=f.revision+1,
                  capture_timestamp=f.now+.105, expires_at=f.now+.255)
    unchanged = deepcopy((prepared, latest))
    selected = intent_after_endpoint(prepared, latest, live_ai=live_ai)
    assert selected is latest
    f.worker.delay = .06
    result = f.tick(ui=selected, delta=.105)
    assert result['allowed'] and result['snapshot']['clutched']
    assert result['expires_at'] <= latest['expires_at']
    assert (prepared, latest) == unchanged
    # The actual old lease would expire during the same guard computation.
    g = Fixture(); g.start(); old = dict(g.last['ui'], expires_at=g.now+.15)
    g.worker.delay = .06
    assert not g.tick(ui=old, delta=.105)['allowed']


def test_new_live_selection_is_prepared_before_ack_and_cancels_old_offer():
    f = Fixture(); f.start()
    prepared = deepcopy(f.last['ui'])
    latest = dict(prepared, revision=f.revision+1, select_candidate_id='new_grasp',
                  capture_timestamp=f.now+.02, expires_at=f.now+.17)
    selected = intent_after_endpoint(prepared, latest, live_ai=True)
    assert selected is prepared
    offer = f.tick(ui=selected)
    assert offer['allowed']
    assert not recheck(f, ui=latest)
    assert f.session.clutched and f.consume(offer) is None
    assert f.session.status['snapshot']['last_ui_revision'] < latest['revision']
    # The next endpoint preparation supplies the genuinely selected candidate.
    batch = f.proposal()
    batch = replace(batch, candidates=tuple(replace(c, candidate_id='new_grasp') for c in batch.candidates))
    result = f.tick(ui=latest, batch=batch)
    assert not result['allowed'] and result['snapshot']['selected_candidate_id'] == 'new_grasp'
    assert result['reason'] == 'ui_action_waiting_for_fresh_control_sample'


@pytest.mark.parametrize('latest', [{}, None, {'release': True}, {'confirm': True}, {'clutch': True},
    {'release': True, 'select_candidate_id': 'contradictory_action'},
    {'revision': 4, 'select_candidate_id': 42}, {'revision': 4, 'select_candidate_id': ''}])
def test_no_release_invalid_input_or_other_action_is_hidden(latest):
    prepared = {'revision': 3, 'select_candidate_id': 'old'}
    assert intent_after_endpoint(prepared, latest, live_ai=True) is latest


def test_expired_latest_heartbeat_is_not_retimestamped_or_replaced_by_older_intent():
    f = Fixture(); f.start()
    latest = dict(f.last['ui'], revision=f.revision+1, capture_timestamp=f.now-.5, expires_at=f.now-.35)
    intent = intent_after_endpoint(f.last['ui'], latest, live_ai=True)
    result = f.tick(ui=intent)
    assert not result['allowed'] and not result['snapshot']['clutched']
    assert intent == latest
