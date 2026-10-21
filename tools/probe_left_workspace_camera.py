#!/usr/bin/env python3
"""Render the fixed left workspace camera during a normal G1 finger closure.

This is an explicit oracle camera-coverage/robot-geometry test, not learned
perception or task acceptance. Known CAD and measured drawer joint state are
used only to evaluate visibility. No attachment, teleport or external force.
"""
import argparse
import hashlib
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
backend = None
try:
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from teleop_assist.isaac_backend import IsaacBackend
    from teleop_assist.fixture_assets import drawer_handle_components, _component_mesh
    from teleop_assist.recording import write_rgbd
    from audit_wrist_coverage import coverage, depth_visibility

    backend = IsaacBackend(args.output/'backend', headless=args.headless, width=640, height=480,
                           wrist_width=480, wrist_height=360, scene_variant='interaction')
    drawer = next(t for t in backend.scene['targets'] if t['id'] == 'PullHandle')
    origin = np.asarray(drawer['center'], dtype=float)
    grasp = origin+[-.056, 0., -.05]
    orientation = Rotation.from_euler('x', 90, degrees=True).as_quat()
    vertices = np.vstack([_component_mesh(c)[0] for c in drawer_handle_components()[1:]])
    report = {'kind': 'real_render_fixed_left_workspace_mount_oracle_visibility',
              'model_inference': False, 'headset_verified': False,
              'camera_calibration': backend.camera_calibration(), 'phases': {},
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in ['teleop_assist/sensor_geometry.py', 'teleop_assist/isaac_backend.py',
                                             'tools/probe_left_workspace_camera.py']},
              'limits': ['Oracle trajectory and CAD visibility evaluation only, no inferred object pose.',
                         'Normal IK, finger actuators and contact guards; no attachments, pose writes or external forces.',
                         'Left mount is a constant virtual bracket calibrated for the resting left arm.',
                         'Camera housing clearance and physical mounting are not simulated.']}

    def capture_phase(phase):
        bundle = backend.capture_bundle()
        joint = backend.fixture_joint_states()['PullHandle']['position']
        measured_origin = origin+[joint, 0., 0.]
        camera_reports = {}
        for name, frame in bundle['frames'].items():
            Image.fromarray(frame['rgb']).save(args.output/f'{phase}_{name}.png')
            write_rgbd(args.output/f'{phase}_{name}.npz', frame, compression=0)
            camera_reports[name] = {
                'T_link_camera': frame['T_link_camera'].tolist(),
                'coverage': coverage(vertices+measured_origin, measured_origin+[-.065, 0., 0.],
                                     frame['T_world_camera'], frame['intrinsics']),
                'depth_visibility': depth_visibility(measured_origin, frame['T_world_camera'],
                                                      frame['intrinsics'], frame['depth'])}
        forces = backend.contact.data.net_forces_w.detach().cpu().numpy()[0]
        finger_contacts = {name:float(np.linalg.norm(forces[i])) for i,name in enumerate(backend.contact.body_names)
                           if 'right_hand_' in name and np.linalg.norm(forces[i])>.02}
        report['phases'][phase] = {'frame_id': bundle['frame_id'], 'timestamp': bundle['timestamp'],
            'sim_time': bundle['sim_time'], 'ee_position': backend.last_state['ee_position'],
            'ee_quaternion': backend.last_state['ee_quaternion'],
            'accepted_grip': backend.last_state['accepted_grip'],
            'drawer_position_m_evaluation_only': joint, 'finger_contact_forces_N': finger_contacts,
            'cameras': camera_reports}
        print('WORKSPACE_CAMERA_PHASE', phase, json.dumps(report['phases'][phase]), flush=True)

    capture_phase('initial')
    invalid = guards = 0
    for phase, goal, grip, steps in [('align', grasp+[-.09,0,0],0.,1200),
                                      ('approach',grasp,0.,1000),
                                      ('half_close',grasp,.71,350),('closed',grasp,.85,350)]:
        start = np.array(backend.last_state['ee_position'])
        for step in range(steps):
            command = start+(goal-start)*min(1.,(step+1)/(steps*.75)) if phase=='approach' else goal
            state = backend.step(command, orientation, grip)
            invalid += int(not state['valid']); guards += int(state['contact_active'])
        capture_phase(phase)
    stages = list(report['phases'].values())
    report['checks'] = {
        'left_handle_inside_image': all(p['cameras']['left_wrist']['coverage']['projected_bbox_fraction_inside_image']>.98 for p in stages),
        'left_observed_CAD_depth_in_all_phases': all(p['cameras']['left_wrist']['depth_visibility']['depth_consistent_handle_pixels']>=80 for p in stages),
        'left_mount_identical_across_acquisitions': all(np.array_equal(np.array(p['cameras']['left_wrist']['T_link_camera']),
                     np.array(report['camera_calibration']['left_wrist']['T_link_camera'])) for p in stages),
        'robot_states_valid': invalid==0,
        'no_contact_guard': guards==0}
    report.update(passed=all(report['checks'].values()),invalid_robot_steps=invalid,contact_guard_steps=guards)
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('WORKSPACE_CAMERA_RESULT',json.dumps(report),flush=True)
    if not report['passed']:
        raise RuntimeError('Workspace camera checks failed; inspect saved real RGB-D evidence')
except BaseException:
    (args.output/'error.txt').write_text(traceback.format_exc())
    raise
finally:
    if backend is not None:backend.close()
    app.close(wait_for_replicator=False,skip_cleanup=True)
