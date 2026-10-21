#!/usr/bin/env python3
"""Real Isaac multi-camera, calibrated raycast LiDAR and articulated scene probe.

External test forces validate passive joints. They are explicitly not robot
task completion, grasp attachment, policy output, or production control.
"""
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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error('Use a fresh output directory')
args.output.mkdir(parents=True)
args.enable_cameras = True
args.device = 'cpu'
app = AppLauncher(args).app

try:
    import numpy as np
    from PIL import Image
    from teleop_assist.isaac_backend import IsaacBackend

    backend = IsaacBackend(args.output/'backend', headless=args.headless, width=640, height=480,
                           wrist_width=480, wrist_height=360, scene_variant='interaction')
    initial = backend.capture_bundle()
    initial_joints = backend.fixture_joint_states()
    first_lidar = backend.capture_lidar()
    for name, frame in initial['frames'].items():
        Image.fromarray(frame['rgb']).save(args.output/(name+'_initial.png'))
        np.savez_compressed(args.output/(name+'_initial.npz'), rgb=frame['rgb'], depth=frame['depth'],
                            T_world_camera=frame['T_world_camera'], T_link_camera=frame['T_link_camera'],
                            metadata=json.dumps({k: v for k, v in frame.items() if not isinstance(v, np.ndarray)}))
    np.savez_compressed(args.output/'lidar_initial.npz', **{k: v for k, v in first_lidar.items() if isinstance(v, np.ndarray)})
    # Confirm wrist transforms follow the actual measured robot rather than
    # a virtual target by making a small bounded tool displacement.
    target = np.asarray(backend.last_state['ee_position']) + [-.01, -.035, .035]
    invalid_steps = 0
    for i in range(400):
        state = backend.step(target)
        invalid_steps += int(not state['valid'])
        if i % 100 == 0:
            backend.capture_bundle()
    after_motion = backend.capture_bundle()
    # Passive mechanics subsystem check: force/torque only, no pose writes.
    for i in range(120):
        for name, force, torque in [
            ('PullHandle', [-.8, 0, 0], [0, 0, 0]),
            ('BlueSelector', [0, 0, 0], [.005, 0, 0]),
            ('GreenButton', [1.5, 0, 0], [0, 0, 0]),
        ]:
            backend.fixture_body_views[name].apply_forces_and_torques_at_position(
                np.asarray([force], dtype=np.float32), np.asarray([torque], dtype=np.float32),
                None, np.array([0], dtype=np.uint32), True)
        state = backend.step()
        invalid_steps += int(not state['valid'])
    final = backend.capture_bundle()
    final_joints = backend.fixture_joint_states()
    for name, frame in final['frames'].items():
        Image.fromarray(frame['rgb']).save(args.output/(name+'_final.png'))
    camera_checks = {}
    for name, frame in after_motion['frames'].items():
        rotation = frame['T_world_camera'][:3, :3]
        camera_checks[name] = {
            'finite_depth_fraction': float(np.isfinite(frame['depth']).mean()),
            'rgb_standard_deviation': float(frame['rgb'].std()),
            'rigid_rotation': bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)),
            'mount_motion_m': float(np.linalg.norm(frame['T_world_camera'][:3, 3]-initial['frames'][name]['T_world_camera'][:3, 3])),
        }
    drawer_travel = final_joints['PullHandle']['position']-initial_joints['PullHandle']['position']
    knob_travel = final_joints['BlueSelector']['position']-initial_joints['BlueSelector']['position']
    checks = {
        'three_synchronized_cameras': set(final['frames']) == {'head', 'left_wrist', 'right_wrist'}
            and len({f['timestamp'] for f in final['frames'].values()}) == 1
            and len({f['sim_time'] for f in final['frames'].values()}) == 1,
        'rendered_nonempty_cameras': all(c['rgb_standard_deviation'] > 3 and c['finite_depth_fraction'] > .2
                                        for c in camera_checks.values()),
        'rigid_camera_calibrations': all(c['rigid_rotation'] for c in camera_checks.values()),
        'right_wrist_follows_measured_robot': camera_checks['right_wrist']['mount_motion_m'] > .01,
        'lidar_real_finite_hits': len(first_lidar['points_lidar']) > 10 and bool(np.isfinite(first_lidar['points_lidar']).all()),
        'drawer_physically_slides': drawer_travel < -.01 and -.165 < final_joints['PullHandle']['position'] <= .002,
        'knob_physically_rotates': abs(knob_travel) > .005 and abs(final_joints['BlueSelector']['position']) < np.pi/2+.02,
        'button_physically_presses': .006 < final_joints['GreenButton']['position'] < .014,
        'robot_remains_valid': invalid_steps == 0,
    }
    report = {'kind': 'real_isaac_multisensor_passive_mechanics_probe', 'checks': checks,
              'passed': all(checks.values()), 'camera_checks': camera_checks,
              'initial_joint_states': initial_joints, 'final_joint_states': final_joints,
              'lidar_hit_count': len(first_lidar['points_lidar']), 'lidar_self_hits': int(first_lidar['self_hit'].sum()),
              'drawer_displacement_m': drawer_travel, 'knob_rotation_rad': knob_travel,
              'invalid_robot_steps': invalid_steps,
              'limitations': ['Passive mechanics tested with external forces, not robot grasp success.',
                              'No headset hardware tested.', 'Ideal raycast LiDAR, not a physical sensor model.']}
    (args.output/'result.json').write_text(json.dumps(report, indent=2))
    print('MULTISENSOR_RESULT', json.dumps(report), flush=True)
    backend.close()
    if not report['passed']:
        raise RuntimeError('One or more multisensor checks failed')
except BaseException:
    error = traceback.format_exc()
    (args.output/'error.txt').write_text(error)
    print(error, flush=True)
    raise
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
