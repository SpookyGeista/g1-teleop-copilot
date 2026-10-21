#!/usr/bin/env python3
"""Bounded real-physics six-dimensional controller and deadman-settling probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--angle-deg', type=float, default=10.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error('Use a fresh output directory')
if not 1. <= abs(args.angle_deg) <= 15.:
    parser.error('Probe rotation must be between 1 and 15 degrees')
args.output.mkdir(parents=True)
args.enable_cameras = True
args.device = 'cpu'
app = AppLauncher(args).app

try:
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from teleop_assist.core import AssistSession, AssistConfig, XR_TO_ROBOT
    from teleop_assist.isaac_backend import IsaacBackend

    backend = IsaacBackend(args.output/'sim', headless=args.headless)
    state = dict(backend.last_state)
    start_position = np.array(state['ee_position'])
    start_rotation = Rotation.from_quat(state['ee_quaternion'])
    world_axis = start_rotation.as_matrix()[:, 0]
    world_delta = Rotation.from_rotvec(world_axis*np.radians(args.angle_deg))
    desired_rotation = world_delta*start_rotation
    basis = Rotation.from_matrix(XR_TO_ROBOT)
    controller_delta = basis.inv()*world_delta*basis
    session = AssistSession(AssistConfig(assist_enabled=False, eye_enabled=False))
    sequence = 0

    def sample(hand_position, hand_rotation, clutch):
        global sequence
        sequence += 1
        session.command({'type': 'input', 'sample': {
            'source': 'synthetic_pose_probe', 'session_id': 'pose_probe_v1', 'seq': sequence,
            'orientation_mode': 'pose', 'xr_to_world_rotation': XR_TO_ROBOT.tolist(),
            'right': {'valid': True, 'position': np.asarray(hand_position).tolist(),
                      'quaternion_xyzw': hand_rotation.as_quat().tolist()},
            'axes': [0., 0.], 'trigger': 0., 'squeeze': 1. if clutch else 0.,
            'clutch': clutch}})

    sample(np.zeros(3), Rotation.identity(), False)
    session.robot = state
    session.command({'type': 'calibrate'})
    initial_frame = backend.capture()
    Image.fromarray(initial_frame['rgb']).save(args.output/'initial_rgb.png')
    max_command_angular_speed = 0.
    max_joint_violation = 0.
    invalid_states = 0
    previous_command = start_rotation
    pose_samples = []
    trace = (args.output/'pose_trace.jsonl').open('w')
    for step in range(1800):
        sample(np.zeros(3), controller_delta, True)
        reference = session.step(state, backend.dt)
        if reference.get('hold'):
            backend.hold()
        command_rotation = Rotation.from_quat(reference['ee_quat_xyzw'])
        command_speed = np.linalg.norm((command_rotation*previous_command.inv()).as_rotvec())/backend.dt
        previous_command = command_rotation
        max_command_angular_speed = max(max_command_angular_speed, float(command_speed))
        state = backend.step(reference['ee_target_world'], reference['ee_quat_xyzw'], reference['grip'])
        invalid_states += int(not state['valid'])
        max_joint_violation = max(max_joint_violation, state['joint_limit_violation'])
        measured_rotation = Rotation.from_quat(state['ee_quaternion'])
        position_error = float(np.linalg.norm(np.array(state['ee_position'])-start_position))
        angle_error = float(np.degrees(np.linalg.norm((desired_rotation*measured_rotation.inv()).as_rotvec())))
        pose_samples.append((position_error, angle_error))
        if step % 20 == 0:
            trace.write(json.dumps({'phase': 'rotate_in_place', 'state': state, 'reference': reference,
                                    'position_error_m': position_error, 'orientation_error_deg': angle_error})+'\n')
        if step % 40 == 0:
            backend.capture()
    final_rotation_state = dict(state)
    Image.fromarray(backend.capture()['rgb']).save(args.output/'rotated_rgb.png')

    # Start an operator-requested Cartesian move then release the deadman.
    # Measurement reports how far inertia/drive settling actually carries the
    # hand; a correct held reference does not imply an instantaneous stop.
    desired_position = start_position+np.array([.07, 0., .03])
    hand_position = XR_TO_ROBOT.T@(desired_position-start_position)
    for step in range(150):
        sample(hand_position, controller_delta, True)
        reference = session.step(state, backend.dt)
        state = backend.step(reference['ee_target_world'], reference['ee_quat_xyzw'], reference['grip'])
        if step % 40 == 0:
            backend.capture()
    stop_position = np.asarray(state['ee_position']).copy()
    stopped_references = True
    max_post_release_displacement = 0.
    post_release = []
    for step in range(400):
        sample(hand_position, controller_delta, False)
        reference = session.step(state, backend.dt)
        before = np.asarray(state['ee_position'])
        stopped_references &= bool(reference.get('hold') and reference['ee_quat_xyzw'] is None and
                                   np.linalg.norm(np.asarray(reference['ee_target_world'])-before) < 1e-8)
        backend.hold()
        reset_error = float(np.linalg.norm(backend._target_ee-before))
        state = backend.step(reference['ee_target_world'], reference['ee_quat_xyzw'], reference['grip'])
        displacement = float(np.linalg.norm(np.asarray(state['ee_position'])-stop_position))
        max_post_release_displacement = max(max_post_release_displacement, displacement)
        post_release.append(np.asarray(state['ee_position']))
        invalid_states += int(not state['valid'])
        if step % 20 == 0:
            trace.write(json.dumps({'phase': 'deadman_released', 'state': state,
                                    'held_reference': reference, 'filter_reset_error_m': reset_error,
                                    'distance_since_release_m': displacement})+'\n')
        if step % 40 == 0:
            backend.capture()
    trace.close()
    position_error, angle_error = map(float, np.mean(np.asarray(pose_samples[-200:]), axis=0))
    final_measured_rotation = Rotation.from_quat(final_rotation_state['ee_quaternion'])
    measured_turn = float(np.degrees(np.linalg.norm((final_measured_rotation*start_rotation.inv()).as_rotvec())))
    settled_speed = float(np.linalg.norm(post_release[-1]-post_release[-21])/(20*backend.dt))
    checks = {
        'rotation_accuracy': {'passed': angle_error < 2., 'mean_last_second_error_deg': float(angle_error),
                              'requested_turn_deg': args.angle_deg, 'measured_turn_deg': measured_turn},
        'tcp_position_held': {'passed': position_error < .015, 'mean_last_second_error_m': float(position_error)},
        'command_angular_speed': {'passed': max_command_angular_speed <= session.config.max_angular_speed+1e-4,
                                   'measured_max_rad_s': max_command_angular_speed,
                                   'configured_limit_rad_s': session.config.max_angular_speed},
        'deadman_reference_hold': {'passed': bool(stopped_references)},
        'physical_settling': {'passed': max_post_release_displacement < .025 and settled_speed < .01,
                              'max_displacement_after_release_m': max_post_release_displacement,
                              'final_speed_m_s': settled_speed},
        'physical_state': {'passed': invalid_states == 0 and max_joint_violation < .035,
                           'invalid_steps': invalid_states, 'max_joint_limit_violation_rad': max_joint_violation},
    }
    result = {'passed': all(check['passed'] for check in checks.values()), 'checks': checks,
              'source': 'real Isaac/PhysX, measured G1 end-effector state',
              'input': 'synthetic XR controller through production AssistSession; not a live headset',
              'fixed_base': True, 'position_start': start_position.tolist(),
              'rotation_start_xyzw': start_rotation.as_quat().tolist(),
              'rotation_goal_xyzw': desired_rotation.as_quat().tolist(),
              'rotation_final_xyzw': final_measured_rotation.as_quat().tolist()}
    (args.output/'pose_result.json').write_text(json.dumps(result, indent=2))
    Image.fromarray(backend.capture()['rgb']).save(args.output/'final_rgb.png')
    print('ASSIST_POSE_RESULT', json.dumps(result), flush=True)
    backend.close()
except BaseException:
    error = traceback.format_exc()
    (args.output/'pose_error.txt').write_text(error)
    print(error, flush=True)
    raise
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
