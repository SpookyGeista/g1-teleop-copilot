import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.copilot.xr_adapter import PROJECT, XRAdapter, host_clock_id


def test_config_relative_mailboxes_resolve_from_project_not_staged_cwd(tmp_path, monkeypatch):
    config = tmp_path / 'copilot.json'
    config.write_text(json.dumps({'xr': {'mode':'shadow',
        'control_mailbox':'runs/copilot_live/copilot_control.json',
        'proposal_mailbox':'runs/copilot_live/latest_proposals.json',
        'status_path':str(tmp_path/'absolute_status.json')}}))
    staged_cwd = tmp_path / 'staging' / 'teleop'; staged_cwd.mkdir(parents=True)
    monkeypatch.chdir(staged_cwd)
    adapter = XRAdapter.from_config(config)
    assert adapter.config['control_mailbox'] == str(PROJECT/'runs/copilot_live/copilot_control.json')
    assert adapter.config['proposal_mailbox'] == str(PROJECT/'runs/copilot_live/latest_proposals.json')
    assert adapter.config['status_path'] == str(tmp_path/'absolute_status.json')


def envelope(now=10.):
    return dict(schema='punktnull.operator_envelope.v1', sequence=12,
                acquisition_timestamp=now - .02, clock_id=host_clock_id(),
                clock_uncertainty_s=.004, latency_unknown=False,
                sample_kind='webxr_pose_read', tracking_valid=True,
                received_at_host_monotonic=now - .01,
                backend='unitree_isaac_sim', dds_domain=1)


def poses():
    left, right = np.eye(4), np.eye(4)
    left[:3, 3], right[:3, 3] = [.3, .2, .4], [.3, -.2, .4]
    return left, right


def rotated(left, right, sample):
    right[:3, :3] = Rotation.from_euler('z', 3, degrees=True).as_matrix()
    return left, right, {'input_hold_required': False}


def isolated_test_guard(*args):
    return {'allowed': True, 'reason': 'isolated_contract_test_not_robot_validation'}


def test_shadow_preserves_actual_original_inputs_and_callback_inputs_are_copies():
    left, right = poses()
    adapter = XRAdapter(callback=rotated, clock=lambda: 10.)
    got = adapter.process(left, right, envelope())
    np.testing.assert_array_equal(got[0], left)
    np.testing.assert_array_equal(got[1], right)
    assert not adapter.diagnostics['applied']


def test_explicit_sim_assist_may_only_change_right_orientation():
    left, right = poses()
    adapter = XRAdapter({'mode': 'assist'}, callback=rotated, runtime_guard=isolated_test_guard, clock=lambda: 10.)
    got = adapter.process(left, right, envelope())
    np.testing.assert_array_equal(got[0], left)
    np.testing.assert_array_equal(got[1][:3, 3], right[:3, 3])
    assert not np.array_equal(got[1][:3, :3], right[:3, :3])
    assert adapter.diagnostics['applied']


@pytest.mark.parametrize('key,value', [
    ('acquisition_timestamp', None), ('clock_id', 'browser_without_mapping'),
    ('latency_unknown', True), ('clock_uncertainty_s', .1),
    ('acquisition_timestamp', 9.), ('acquisition_timestamp', 10.5),
    ('received_at_host_monotonic', 8.), ('tracking_valid', False),
    ('backend', 'real_robot'), ('dds_domain', 0),
])
def test_invalid_input_never_uses_fresh_receipt_as_capture_or_calls_assistant(key, value):
    left, right = poses()
    sample = envelope(); sample[key] = value
    def forbidden(*args):
        raise AssertionError('Invalid input must not reach assistant')
    adapter = XRAdapter({'mode': 'assist'}, callback=forbidden, clock=lambda: 10.)
    got = adapter.process(left, right, sample)
    assert adapter.input_hold_required
    assert adapter.diagnostics['reason'] != 'assistant_error'
    np.testing.assert_array_equal(got[1], right)


@pytest.mark.parametrize('broken', ['translation', 'left_arm', 'nonrigid'])
def test_callback_contract_violation_requests_hold_without_partial_command(broken):
    left, right = poses()
    def callback(l, r, sample):
        if broken == 'translation': r[0, 3] += .01
        if broken == 'left_arm': l[0, 3] += .01
        if broken == 'nonrigid': r[0, 0] = 2.
        return l, r, {}
    adapter = XRAdapter({'mode': 'assist'}, callback=callback, runtime_guard=isolated_test_guard, clock=lambda: 10.)
    got = adapter.process(left, right, envelope())
    assert adapter.input_hold_required
    assert adapter.diagnostics['reason'] == 'callback_broke_orientation_only_contract'
    np.testing.assert_array_equal(got[0], left)
    np.testing.assert_array_equal(got[1], right)


def test_control_lease_and_restart_replay_protection(tmp_path):
    mailbox = tmp_path / 'control.json'
    message = dict(revision=7, capture_timestamp=10., clock_id=host_clock_id(),
                   select_candidate_id='grasp', confirm=True, assist_held=True, release=False)
    mailbox.write_text(json.dumps(message))
    now = [10.]
    adapter = XRAdapter({'mode': 'assist', 'control_mailbox': str(mailbox)}, clock=lambda: now[0])
    calls = []
    class Controller:
        def step(self, sample, proposals, **kw):
            calls.append(kw['request'])
            return SimpleNamespace(left_base_ee=sample.left_base_ee, right_base_ee=sample.right_base_ee,
                snapshot={}, input_hold_required=False)
    adapter.controller = Controller()
    adapter.process(*poses(), envelope())
    assert not calls[-1].confirm and calls[-1].select_candidate_id is None
    # A newly written revision is an actual new UI edge.
    message['revision'] += 1
    mailbox.write_text(json.dumps(message))
    adapter.process(*poses(), envelope())
    assert calls[-1].confirm and calls[-1].select_candidate_id == 'grasp'
    adapter.process(*poses(), envelope())
    assert not calls[-1].confirm
    # Frozen assist-held cannot survive the 250 ms lease.
    now[0] = 10.3
    adapter.process(*poses(), envelope(now[0]))
    assert not calls[-1].assist_held and not calls[-1].release


def test_observation_extras_do_not_drop_structurally_valid_model_proposal(tmp_path):
    mailbox = tmp_path / 'proposals.json'
    record = dict(target_id='cup', observation_id='rgbd-4', capture_timestamp=9.98,
        clock_id=host_clock_id(), calibration_id='cal-1', motion_static=True, valid=True,
        inference_s=.12, object_points=500, descriptor_sha256='audited',
        candidates=[dict(candidate_id='g1', base_ee=np.eye(4).tolist(), model_score=.8,
            geometry_valid=True, ik_valid=True, collision_valid=True, source='model',
            world_canonical=np.eye(4).tolist(), q_solution=[0.]*14,
            validation_reason='accepted', preview_uv=[.3,.4])])
    mailbox.write_text(json.dumps(record))
    adapter = XRAdapter({'proposal_mailbox': str(mailbox)}, clock=lambda: 10.)
    received = []
    class Controller:
        def step(self, sample, proposals, **kw):
            received.append(proposals)
            return SimpleNamespace(left_base_ee=sample.left_base_ee, right_base_ee=sample.right_base_ee,
                snapshot={'reason':'preview_ready'}, input_hold_required=False)
    adapter.controller = Controller()
    adapter.process(*poses(), envelope())
    assert received[0].candidates[0].candidate_id == 'g1'
    assert adapter.diagnostics['reason'] == 'preview_ready'


def test_invalid_input_discards_previously_confirmed_controller():
    adapter = XRAdapter(clock=lambda: 10.)
    adapter.controller = object()
    invalid = envelope(); invalid['tracking_valid'] = False
    adapter.process(*poses(), invalid)
    assert adapter.controller is None and adapter.input_hold_required


@pytest.mark.parametrize('guard,reason', [(None, 'runtime_motion_guard_unavailable'),
    (lambda *args: {'allowed': False}, 'runtime_motion_guard_rejected')])
def test_endpoint_proposal_flags_cannot_replace_actual_motion_guard(guard, reason):
    left, right = poses()
    adapter = XRAdapter({'mode':'assist'}, callback=rotated, runtime_guard=guard, clock=lambda: 10.)
    got = adapter.process(left, right, envelope())
    assert adapter.input_hold_required and adapter.diagnostics['reason'] == reason
    np.testing.assert_array_equal(got[1], right)


@pytest.mark.parametrize('change', [{'capture_timestamp': None}, {'capture_timestamp': 11.}, {'clock_id': 'wrong'}])
def test_unknown_future_or_wrong_clock_control_cannot_hold_or_replay_edges(tmp_path, change):
    mailbox = tmp_path / 'control.json'
    adapter = XRAdapter({'mode': 'assist', 'control_mailbox': str(mailbox)}, clock=lambda: 10.)
    message = dict(revision=1, capture_timestamp=10., clock_id=host_clock_id(), confirm=True, assist_held=True)
    message.update(change); mailbox.write_text(json.dumps(message))
    calls = []
    class Controller:
        def step(self, sample, proposals, **kw):
            calls.append(kw['request'])
            return SimpleNamespace(left_base_ee=sample.left_base_ee, right_base_ee=sample.right_base_ee,
                snapshot={}, input_hold_required=False)
    adapter.controller = Controller()
    adapter.process(*poses(), envelope())
    assert not calls[-1].release and not calls[-1].assist_held and not calls[-1].confirm
