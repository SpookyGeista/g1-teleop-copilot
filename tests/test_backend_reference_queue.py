"""Exercise the real backend reference code with a small CPU plant double.

This proves reference cancellation/latching, not physical stopping distance.
The opt-in Isaac probe measures the latter separately.
"""
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from teleop_assist.isaac_backend import IsaacBackend, gravity_effort_budget


def backend_double():
    b = IsaacBackend.__new__(IsaacBackend)
    b.torch = torch
    b.closed = False
    b.start_pose = 'fixture'
    b.arm_solver = 'dls'
    b.arm_solver_sync = False
    b._unitree_ik = None
    b._arm_solver_diagnostics = {'name': 'dls', 'status': 'starting', 'accepted': False, 'solve_ms': 0.}
    b.arm_ids = list(range(7))
    b.finger_ids = [7]
    b.names = [f'arm_{i}' for i in range(7)] + ['right_hand_index_1_joint']
    b.body_id = 1
    b.lower, b.upper = np.full(8, -2.), np.full(8, 2.)
    b.velocity_limits = np.full(8, 10.)
    b.effort_limits = np.full(8, 8.)
    b.arm_total_effort_limits = np.full(7, 6.)
    b._arm_gravity_feedforward = np.zeros(7)
    b._arm_drive_effort_budget = np.zeros(7)
    b.joint_target = torch.zeros((1, 8))
    b.command_rejections = b.step_index = b.arm_reference_flushes = 0
    b.hold_reason = None
    b.last_grip = .5
    b._arm_reference_mode = 'active'
    b._external_hold_requested = False
    b._arm_hold_gravity_offset = np.zeros(7)
    b._arm_hold_gravity_forces = np.zeros(7)
    b._arm_hold_gravity_saturated = False
    q = torch.zeros((1, 8))
    jac = torch.zeros((1, 1, 6, 8))
    jac[0, 0, :6, :6] = torch.eye(6)
    data = SimpleNamespace(joint_pos=q, joint_vel=torch.zeros_like(q), applied_torque=torch.zeros_like(q),
                           joint_stiffness=torch.full_like(q, 100.), joint_effort_limits=torch.full_like(q, 6.))
    view = SimpleNamespace(get_jacobians=lambda: jac,
                           get_gravity_compensation_forces=lambda: torch.zeros_like(q),
                           get_dof_projected_joint_forces=lambda: torch.zeros_like(q))
    b.robot = SimpleNamespace(data=data, root_physx_view=view,
                              set_joint_position_target=lambda _: None,
                              write_joint_effort_limit_to_sim=lambda values, joint_ids: None,
                              set_joint_effort_target=lambda values, joint_ids: None,
                              write_data_to_sim=lambda: None, update=lambda _: None)
    b.sim = SimpleNamespace(step=lambda **_: None)
    b.contact = SimpleNamespace(is_initialized=True, body_names=['right_wrist_yaw_link'],
                                data=SimpleNamespace(net_forces_w=torch.zeros((1, 1, 3))),
                                update=lambda _: None)
    b._measured_ee = lambda: (np.array([.4, -.25, 1.]) + q[0, :3].numpy(),
                              np.array([0., 0., 0., 1.]), np.eye(3))
    b.fixture_joint_states = lambda: {}
    b._target_ee = b._measured_ee()[0].copy()
    return b


def queued_backend():
    b = backend_double()
    b.joint_target[0, :7] = .4
    b._target_ee += [.05, 0., 0.]
    return b


def unitree_backend(result=None):
    b = backend_double()
    b.arm_solver = 'unitree'
    b.arm_solver_sync = True
    b.arm_names = b.names[:7]
    b.torso_id = 0
    b.robot.data.body_link_pose_w = torch.tensor([[[0., 0., .8, 1., 0., 0., 0.]]])
    b._unitree_goal = None
    b._unitree_result_at = -float('inf')
    b._unitree_submit_step = -1000

    class Client:
        resets = 0
        args = None
        def solve(self, *args, **kwargs):
            self.args = args
            return result
        def reset(self):
            self.resets += 1
    b._unitree_ik = Client()
    return b


def test_unitree_target_uses_same_joint_speed_bound_and_measured_torso():
    values = {f'arm_{i}': .1 for i in range(7)}
    b = unitree_backend({'accepted': True, 'status': 'solved', 'joint_targets': values, 'solve_ms': 2.})
    state = b.step(b._measured_ee()[0]+[.01, 0., 0.])
    np.testing.assert_allclose(b.joint_target[0, :7], .8*b.dt)
    assert b._unitree_ik.args[2] is None  # position-only retreat must stay position-only
    np.testing.assert_allclose(b._unitree_ik.args[3], [0., 0., .8])
    np.testing.assert_allclose(b._unitree_ik.args[4], [0., 0., 0., 1.])
    assert state['arm_solver']['accepted']


@pytest.mark.parametrize('result', [
    {'accepted': False, 'status': 'infeasible'},
    {'accepted': True, 'joint_targets': {f'arm_{i}': float('nan') for i in range(7)}},
    {'accepted': True, 'joint_targets': {f'arm_{i}': 20. for i in range(7)}},
])
def test_bad_unitree_solution_flushes_existing_joint_queue(result):
    b = unitree_backend(result)
    b.joint_target[0, :7] = .4
    state = b.step(b._measured_ee()[0]+[.01, 0., 0.])
    np.testing.assert_allclose(b.joint_target[0, :7], 0.)
    assert state['hold_reason'] == 'arm_ik_rejected'
    assert not state['command_accepted']
    assert not state['arm_solver']['accepted']
    assert b._unitree_ik.resets == 1


def test_hold_invalidates_unitree_result_and_pending_generation_once():
    b = unitree_backend()
    b._unitree_goal = np.ones(7)
    b.joint_target[0, :7] = .4
    b.hold()
    b.hold()
    assert b._unitree_goal is None
    assert b._unitree_ik.resets == 1
    np.testing.assert_allclose(b.joint_target[0, :7], 0.)


def async_unitree_backend(result=None):
    b = unitree_backend()
    b.arm_solver_sync = False
    b._unitree_ik.poll = lambda: result
    b._unitree_ik.submit = lambda *args: None
    return b


def test_async_old_result_is_rejected_even_when_just_received():
    b = async_unitree_backend({'accepted': True, 'result_age_s': .151,
        'joint_targets': {f'arm_{i}': .1 for i in range(7)}})
    b.joint_target[0, :7] = .4
    state = b.step(b._measured_ee()[0]+[.01, 0., 0.])
    assert state['hold_reason'] == 'arm_ik_rejected'
    assert 'stale input' in state['arm_solver']['error']
    assert b._unitree_ik.resets == 1
    np.testing.assert_allclose(b.joint_target[0, :7], 0.)


def test_async_cached_goal_expires_and_flushes_when_worker_stops_responding():
    b = async_unitree_backend()
    b._unitree_goal = np.full(7, .1)
    b._unitree_result_at = time.monotonic()-.2
    b.joint_target[0, :7] = .4
    state = b.step(b._measured_ee()[0]+[.01, 0., 0.])
    assert state['hold_reason'] == 'arm_ik_rejected'
    assert 'stream stale' in state['arm_solver']['error']
    assert b._unitree_goal is None
    np.testing.assert_allclose(b.joint_target[0, :7], 0.)


def test_async_fresh_goal_is_applied_and_retained_between_solves():
    result = {'accepted': True, 'result_age_s': .005, 'request_id': 1,
              'joint_targets': {f'arm_{i}': .1 for i in range(7)}}
    b = async_unitree_backend(result)
    first = b.step(b._measured_ee()[0]+[.01, 0., 0.])
    b._unitree_ik.poll = lambda: None
    second = b.step(b._measured_ee()[0]+[.01, 0., 0.])
    assert first['command_accepted'] and second['command_accepted']
    np.testing.assert_allclose(b.joint_target[0, :7], 2*.8*b.dt)


def test_external_hold_flushes_both_queues_once_then_retains_pd_hold_reference():
    b = queued_backend()
    b.joint_target[0, 7] = .2
    b.hold()
    np.testing.assert_allclose(b.joint_target[0, :7], 0.)
    np.testing.assert_allclose(b._target_ee, b._measured_ee()[0])
    assert b.joint_target[0, 7] == pytest.approx(.2)  # independent gripper input
    assert b.arm_reference_flushes == 1
    b.step(b._measured_ee()[0], grip=.5)
    # Simulate gravity/contact deflection. A repeated hold must not ratchet
    # the new joint position into the reference and remove supporting torque.
    b.robot.data.joint_pos[0, 0] = -.025
    for _ in range(3):
        b.hold()
        state = b.step(b._measured_ee()[0], grip=.5)
    assert state['arm_reference_mode'] == 'hold'
    assert b.arm_reference_flushes == 1
    np.testing.assert_allclose(b.joint_target[0, :7], 0.)


def test_hold_followed_by_reverse_resumes_from_measured_q_and_rearms_next_flush():
    b = queued_backend()
    b.hold()
    state = b.step(b._measured_ee()[0] + [-.02, 0., 0.])
    assert state['arm_reference_mode'] == 'active'
    assert -.004 <= b.joint_target[0, 0] < 0.
    assert b.arm_reference_flushes == 1
    b.hold()
    assert b.arm_reference_flushes == 2
    np.testing.assert_allclose(b.joint_target[0, :7], 0.)


def test_hold_discards_motion_queue_while_continuous_gravity_torque_keeps_total_effort_bounded():
    b = queued_backend()
    gravity = torch.tensor([[3., -1., 20., 0., 0., 0., 0., 0.]])
    b.robot.root_physx_view.get_gravity_compensation_forces = lambda: gravity
    b.hold()
    b.step(b._measured_ee()[0])
    np.testing.assert_allclose(b.joint_target[0, :7], 0., atol=1e-7)
    np.testing.assert_allclose(b._arm_gravity_feedforward, [3., -1., 4.8, 0., 0., 0., 0.])
    np.testing.assert_allclose(b._arm_drive_effort_budget, [3., 5., 1.2, 6., 6., 6., 6.])
    np.testing.assert_allclose(np.abs(b._arm_gravity_feedforward)+b._arm_drive_effort_budget,b.arm_total_effort_limits)
    assert b._arm_hold_gravity_saturated
    np.testing.assert_allclose(b._arm_hold_gravity_forces, [3., -1., 20., 0., 0., 0., 0.])
    assert b.arm_reference_flushes == 1
    b.robot.data.joint_pos[0, 2] = -.02
    b.hold()
    np.testing.assert_allclose(b.joint_target[0, :7], 0., atol=1e-7)
    assert b.arm_reference_flushes == 1


@pytest.mark.parametrize('forces,ratings', [([float('nan')],[6.]),([1.],[0.]),([1.],[-1.]),([1.],[6.,6.])])
def test_gravity_budget_rejects_invalid_physical_values(forces, ratings):
    with pytest.raises(ValueError):gravity_effort_budget(forces,ratings)


def test_gravity_budget_covers_all_combined_drive_signs():
    gravity=np.linspace(-30.,30.,31);ratings=np.full(31,6.)
    ff,budget=gravity_effort_budget(gravity,ratings)
    for sign in (-1,1):
        assert np.all(np.abs(sign*budget+ff)<=ratings+1e-12)
    assert np.all(budget>=.2*ratings-1e-12)


@pytest.mark.parametrize('bad', ['cartesian', 'grip', 'orientation', 'workspace', 'contact'])
def test_internal_hold_reasons_flush_joint_queue_and_never_advance_far_target(bad):
    b = queued_backend()
    target, orientation, grip = b._measured_ee()[0]+[.05, 0., 0.], None, .5
    if bad == 'cartesian': target = [float('nan'), 0., 0.]
    if bad == 'grip': grip = 2.
    if bad == 'orientation': orientation = [0., 0., 0., 0.]
    if bad == 'workspace': target = [4., 0., 1.]
    if bad == 'contact': b.contact.data.net_forces_w[0, 0, 0] = -30.
    state = b.step(target, orientation, grip)
    assert state['hold_reason'] is not None
    assert state['arm_reference_mode'] == 'hold'
    assert b.arm_reference_flushes == 1
    np.testing.assert_allclose(b.joint_target[0, :7], 0.)
    np.testing.assert_allclose(b._target_ee, b._measured_ee()[0])
    b.robot.data.joint_pos[0, 0] = -.01
    b.step([float('nan'), 0., 0.])
    assert b.arm_reference_flushes == 1
    np.testing.assert_allclose(b.joint_target[0, :7], 0.)


def test_contact_retreat_cancels_inward_joint_queue_once_and_then_accumulates_outward():
    b = queued_backend()
    b.contact.data.net_forces_w[0, 0, 0] = -30.
    target = b._measured_ee()[0] + [-.02, .01, 0.]
    first = b.step(target, grip=.5)
    q1 = float(b.joint_target[0, 0])
    assert first['contact_retreat_active'] and first['hold_reason'] is None
    assert b.arm_reference_flushes == 1
    assert -.004 <= q1 < 0.
    np.testing.assert_allclose(b.joint_target[0, 1:7], 0., atol=1e-6)
    b.step(target, grip=.5)
    assert b.arm_reference_flushes == 1
    assert b.joint_target[0, 0] < q1
    # Reject an inward command after retreat: that transition cancels the
    # retreat queue, and later retreat has to establish a fresh reference.
    b.step(b._measured_ee()[0]+[.02, 0., 0.])
    assert b.arm_reference_flushes == 2
    b.step(target)
    assert b.arm_reference_flushes == 3
