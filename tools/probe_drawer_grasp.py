#!/usr/bin/env python3
"""Oracle kinematics audit of the real G1 hand grasping the PhysX drawer.

Known simulator coordinates only choose this test's trajectory. Production
perception does not receive them. There are no attachment joints, pose writes,
or externally applied forces: only the normal robot IK and finger drives act.
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
parser.add_argument('--depth-offset', type=float, default=.009,
                    help='Oracle EE offset beyond the handle bar centre along world X, meters')
parser.add_argument('--height-offset', type=float, default=-.05,
                    help='Oracle EE vertical offset from the handle bar centre, meters')
parser.add_argument('--grip', type=float, default=.85)
parser.add_argument('--pull-distance', type=float, default=.04)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error('Use a fresh output directory')
if not (0 <= args.grip <= 1):
    parser.error('grip must be between zero and one')
args.output.mkdir(parents=True)
args.enable_cameras = True
args.device = 'cpu'
app = AppLauncher(args).app

try:
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from teleop_assist.isaac_backend import IsaacBackend

    backend = IsaacBackend(args.output/'backend', headless=args.headless, width=640, height=480,
                           wrist_width=480, wrist_height=360, scene_variant='interaction')
    # Clearly labelled evaluation-only oracle, never a perception estimate.
    drawer = next(t for t in backend.scene['targets'] if t['id'] == 'PullHandle')
    bar = np.asarray(drawer['center']) + [-.065, 0., 0.]
    grasp = bar + [args.depth_offset, 0., args.height_offset]
    pregrasp = grasp + [-.09, 0., 0.]
    pull = grasp + [-args.pull_distance, 0., 0.]
    orientation = Rotation.from_euler('x', 90, degrees=True).as_quat()
    phases = [('align', pregrasp, 0., 1200), ('approach', grasp, 0., 1000),
              ('close', grasp, args.grip, 700), ('pull', pull, args.grip, 1600),
              ('open', pull, 0., 400)]
    def hand_markers():
        # Distal reference points derived from the actual hand STL extents.
        # This is a measured marker separation proxy, not surface clearance.
        points = {}
        for name, marker in [('thumb_2', [0., .05196, 0.]),
                             ('index_1', [.05196, 0., 0.]),
                             ('middle_1', [.05196, 0., 0.])]:
            body_id = list(backend.robot.body_names).index('right_hand_'+name+'_link')
            pose = backend.robot.data.body_link_pose_w[0, body_id].detach().cpu().numpy()
            points[name] = pose[:3]+backend._rotation(pose[3:7]) @ np.asarray(marker)
        midpoint = (points['index_1']+points['middle_1'])/2
        return {'thumb_to_finger_midpoint_m': float(np.linalg.norm(points['thumb_2']-midpoint)),
                'points_world': {n: p.tolist() for n, p in points.items()},
                'definition': 'Measured distal-link marker separation; not collision-surface aperture'}
    initial_markers = hand_markers()
    initial_joint = backend.fixture_joint_states()['PullHandle']['position']
    summaries = {}
    invalid = 0
    with (args.output/'trace.jsonl').open('w') as log:
        for phase, target, grip, steps in phases:
            start_joint = backend.fixture_joint_states()['PullHandle']['position']
            rows = []
            phase_start_ee = np.asarray(backend.last_state['ee_position'])
            for i in range(steps):
                # A continuous operator-like path avoids a sudden 10 cm goal
                # jump. The production IK, speed limits and contact guard
                # remain unchanged. Hold the endpoint for the final quarter.
                command = (phase_start_ee+(target-phase_start_ee)*min(1., (i+1)/(steps*.75))
                           if phase in ('approach', 'pull') else target)
                state = backend.step(command, orientation, grip)
                invalid += int(not state['valid'])
                forces = backend.contact.data.net_forces_w.detach().cpu().numpy()[0]
                names = list(backend.contact.body_names)
                finger_contacts = {n: float(np.linalg.norm(forces[j])) for j, n in enumerate(names)
                                   if 'hand_' in n and np.linalg.norm(forces[j]) > .02}
                row = {'phase': phase, 'step': i, 'requested_target': target.tolist(),
                       'requested_grip': grip, 'command_target': command.tolist(), 'finger_contact_forces_N': finger_contacts,
                       'robot': state, 'drawer_position_m': state['fixture_joint_states']['PullHandle']['position']}
                if i % 10 == 0:
                    log.write(json.dumps(row)+'\n')
                rows.append(row)
                if i % 200 == 0:
                    backend.capture_bundle()
            bundle = backend.capture_bundle()
            for name in ('head', 'right_wrist'):
                frame = bundle['frames'][name]
                Image.fromarray(frame['rgb']).save(args.output/(phase+'_'+name+'.png'))
                k = frame['intrinsics']
                np.savez_compressed(args.output/(phase+'_'+name+'.npz'),
                    rgb=frame['rgb'], depth=frame['depth'],
                    intrinsics=np.array([[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1]]),
                    T_world_camera=frame['T_world_camera'], T_link_camera=frame['T_link_camera'],
                    frame_id=frame['frame_id'], timestamp=frame['timestamp'], sim_time=frame['sim_time'])
            actual_orientation = Rotation.from_quat(state['ee_quaternion'])
            summaries[phase] = {
                'steps': steps, 'hand_aperture_proxy': hand_markers(), 'start_drawer_position_m': start_joint,
                'final_drawer_position_m': rows[-1]['drawer_position_m'],
                'minimum_drawer_position_m': min(r['drawer_position_m'] for r in rows),
                'contact_guard_steps': sum(r['robot']['contact_active'] for r in rows),
                'maximum_contact_N': max(r['robot']['contact_force_max'] for r in rows),
                'finger_contact_steps': sum(bool(r['finger_contact_forces_N']) for r in rows),
                'final_position_error_m': float(np.linalg.norm(target-np.asarray(state['ee_position']))),
                'final_orientation_error_rad': float((Rotation.from_quat(orientation)*actual_orientation.inv()).magnitude()),
                'final_ee': state['ee_position'], 'final_finger_contacts_N': rows[-1]['finger_contact_forces_N'],
                'final_joint_positions': {backend.names[j]: state['joints'][j] for j in backend.finger_ids},
            }
            print('DRAWER_PHASE', phase, json.dumps(summaries[phase]), flush=True)
    achieved_opening = initial_joint-summaries['pull']['final_drawer_position_m']
    released_opening = initial_joint-summaries['open']['final_drawer_position_m']
    checks = {'robot_valid': invalid == 0,
              'hand_reaches_grasp_region': summaries['approach']['final_position_error_m'] < .025,
              'finger_contacts_during_close': summaries['close']['finger_contact_steps'] > 30,
              'drawer_opens_through_robot_contact': achieved_opening > .03,
              'pull_tracks_requested_path': summaries['pull']['final_position_error_m'] < .025,
              'pull_finishes_with_finger_contact': bool(summaries['pull']['final_finger_contacts_N']),
              'pull_without_contact_guard': summaries['pull']['contact_guard_steps'] == 0,
              'drawer_stays_open_after_release': released_opening > .03}
    report = {'kind': 'real_physx_oracle_robot_drawer_grasp', 'passed': all(checks.values()),
              'checks': checks, 'phases': summaries, 'drawer_opening_m': achieved_opening,
              'held_drawer_opening_m': achieved_opening, 'released_drawer_opening_m': released_opening,
              'grasp_oracle_target': grasp.tolist(), 'initial_handle_bar_world': bar.tolist(),
              'invalid_robot_steps': invalid, 'grip': args.grip, 'pull_distance_m': args.pull_distance,
              'initial_hand_aperture_proxy': initial_markers,
              'limits': ['Oracle target, not perception/model validation.',
                         'No human input or headset tested.',
                         'No attachment, teleportation or external forces used.',
                         'Existing contact guard remains enabled.']}
    (args.output/'result.json').write_text(json.dumps(report, indent=2))
    print('DRAWER_GRASP_RESULT', json.dumps(report), flush=True)
    backend.close()
    if not report['passed']:
        raise RuntimeError('Physical robot grasp/open test did not pass')
except BaseException:
    error = traceback.format_exc()
    (args.output/'error.txt').write_text(error)
    print(error, flush=True)
    raise
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
