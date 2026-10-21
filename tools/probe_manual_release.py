#!/usr/bin/env python3
"""Oracle PhysX probe of a separately commanded manual release gesture.

The initial 40 mm pull is followed by a distinct operator-style 40 mm withdrawal
while opening. This is NOT an increase to assisted Execute travel and is NOT an
automatic compensation policy. Only ordinary arm IK and finger drives are used.
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
    parser.error('Use a new output directory')
args.output.mkdir(parents=True)
args.enable_cameras = True
args.device = 'cpu'
app = AppLauncher(args).app
result = {'kind': 'real_physx_oracle_explicit_manual_release_probe', 'passed': False,
          'scope': 'Oracle trajectory and explicit manual release gesture; no perception, production API or headset claim'}
exit_code = 1
backend = None
try:
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from teleop_assist.isaac_backend import IsaacBackend, EVALUATION_HAND_BODY_NAMES

    backend = IsaacBackend(args.output / 'backend', headless=args.headless, width=640,
        height=480, scene_variant='interaction', evaluation_contact_pairs=True)
    target = next(t for t in backend.scene['targets'] if t['id'] == 'PullHandle')
    grasp = np.asarray(target['center']) + [-.056, 0., -.050]
    pregrasp = grasp + [-.09, 0., 0.]
    pull = grasp + [-.04, 0., 0.]
    quat = Rotation.from_euler('x', 90, degrees=True).as_quat()
    phases = [('free', np.asarray(backend.last_state['ee_position']), None, 0., 200),
              ('align', pregrasp, quat, 0., 1200), ('approach', grasp, quat, 0., 1000),
              ('close', grasp, quat, .85, 700), ('pull', pull, quat, .85, 1600),
              ('manual_release', None, quat, 0., 400), ('released_hold', None, quat, 0., 200)]
    initial_joint = backend.fixture_joint_states()['PullHandle']['position']
    summaries = {}
    invalid = timestamp_mismatch = guards = joint_limit = velocity_limit = 0
    release_start = release_goal = None

    def images(phase):
        bundle = backend.capture_bundle()
        for name in ('head', 'left_wrist', 'right_wrist'):
            frame = bundle['frames'][name]
            Image.fromarray(frame['rgb']).save(args.output / f'{phase}_{name}.png')
            np.savez_compressed(args.output / f'{phase}_{name}.npz', rgb=frame['rgb'],
                depth=frame['depth'], T_world_camera=frame['T_world_camera'],
                intrinsics=json.dumps(frame['intrinsics']), timestamp=frame['timestamp'])

    with (args.output / 'contacts.jsonl').open('w', buffering=1) as log:
        for phase, goal, orientation, grip, steps in phases:
            start = np.asarray(backend.last_state['ee_position'])
            start_joint = backend.fixture_joint_states()['PullHandle']['position']
            if phase == 'manual_release':
                release_start = start.copy()
                release_goal = start + [-.04, 0., 0.]
                goal = release_goal
            elif phase == 'released_hold':
                goal = release_goal
            norms = []
            drawer = []
            positions = []
            per_body = {name: [] for name in EVALUATION_HAND_BODY_NAMES}
            for step in range(steps):
                requested_grip = grip
                if phase in ('approach', 'pull'):
                    command = start + (goal-start)*min(1., (step+1)/(steps*.75))
                elif phase == 'manual_release':
                    fraction = min(1., (step+1)/steps)
                    command = start + (goal-start)*fraction
                    requested_grip = .85*(1.-fraction)
                else:
                    command = goal
                if phase == 'free':
                    backend.hold()
                state = backend.step(command, orientation, requested_grip)
                contact = state['evaluation_hand_target_contacts']
                invalid += int(not state['valid'])
                guards += int(bool(state.get('collision_stop')))
                joint_limit += int(bool(state.get('joint_limit_violation')))
                velocity_limit += int(bool(state.get('joint_velocity_limit_violation')))
                timestamp_mismatch += int(contact['timestamp'] != state['timestamp'] or contact['sim_time'] != state['sim_time'])
                values = contact['per_body_normal_force_norms_N']
                norms.append(max(values.values()))
                for name in per_body:
                    per_body[name].append(values[name])
                drawer.append(state['fixture_joint_states']['PullHandle']['position'])
                positions.append(state['ee_position'])
                log.write(json.dumps({'phase': phase, 'phase_step': step,
                    'ee_position': state['ee_position'], 'ee_quaternion': state['ee_quaternion'],
                    'command_target': command.tolist(), 'requested_grip': requested_grip,
                    'accepted_grip': state['accepted_grip'], 'joint_names': state['joint_names'],
                    'joints': state['joints'], 'drawer_position_m_evaluation_only': drawer[-1],
                    'evaluation_hand_target_contacts': contact})+'\n')
                if phase == 'manual_release' and step in (99, 199, 299):
                    images(f'{phase}_{step+1:03d}')
            positions = np.asarray(positions)
            summaries[phase] = {'steps': steps, 'duration_sim_s': steps*backend.dt,
                'start_drawer_position_m': start_joint, 'end_drawer_position_m': drawer[-1],
                'drawer_displacement_m': start_joint-drawer[-1],
                'peak_partner_normal_force_N': max(norms),
                'samples_above_1e_minus4_N': sum(v>1e-4 for v in norms),
                'per_body_peak_partner_normal_force_N': {name:max(v) for name,v in per_body.items()},
                'start_ee': start.tolist(), 'final_ee': state['ee_position'],
                'actual_hand_displacement_m': float(np.linalg.norm(positions[-1]-start)),
                'actual_hand_path_length_m': float(np.linalg.norm(np.diff(np.vstack([start, positions]), axis=0),axis=1).sum()),
                'commanded_displacement_m': float(np.linalg.norm(goal-start)),
                'final_ee_error_m': float(np.linalg.norm(goal-positions[-1])),
                'final_accepted_grip': state['accepted_grip']}
            if phase in ('free', 'close', 'pull', 'manual_release', 'released_hold'):
                images(phase)
            print('MANUAL_RELEASE_PHASE', phase, json.dumps(summaries[phase]), flush=True)

    closure = summaries['close']['end_drawer_position_m']
    pulled = summaries['pull']['end_drawer_position_m']
    final = summaries['released_hold']['end_drawer_position_m']
    loss = final-pulled
    actual_full_travel = np.linalg.norm(np.asarray(summaries['released_hold']['final_ee']) - summaries['pull']['start_ee'])
    checks = {'valid_robot_state': invalid == 0, 'same_step_timestamps': timestamp_mismatch == 0,
        'no_collision_guard': guards == 0, 'no_joint_limit_violation': joint_limit == 0,
        'no_velocity_limit_violation': velocity_limit == 0,
        'positive_collisions_unchanged': not backend.evaluation_contact_collision_filtering['disabled'],
        'free_space_partner_channel_zero': summaries['free']['peak_partner_normal_force_N'] <= 1e-7,
        'drawer_moved_during_actual_pull': closure-pulled > .015,
        'pull_partner_contact_measured': summaries['pull']['samples_above_1e_minus4_N'] >= 25,
        'release_retains_pull_within_3mm': loss < .003,
        'release_retains_at_least_15mm_incremental_pull': closure-final > .015,
        'fully_open_at_final_rest': summaries['released_hold']['final_accepted_grip'] <= 1e-6,
        'all_finger_target_contacts_zero_during_final_1s_rest': summaries['released_hold']['peak_partner_normal_force_N'] <= 1e-7,
        'manual_release_target_reached_within_3mm': summaries['released_hold']['final_ee_error_m'] < .003}
    result.update(passed=all(checks.values()), checks=checks, phases=summaries,
        initial_drawer_position_m=initial_joint, release_opening_lost_m=loss,
        retained_incremental_pull_m=closure-final, final_total_drawer_opening_m=initial_joint-final,
        nominal_pull_command_m=.04, separate_manual_release_command_m=.04,
        actual_pull_plus_manual_release_hand_displacement_m=float(actual_full_travel),
        release_semantics='Separate explicit manual operator gesture after bounded pull; no assisted Execute cap modification or automatic arm motion',
        invalid_steps=invalid, timestamp_mismatches=timestamp_mismatch, collision_guard_steps=guards,
        joint_limit_violation_steps=joint_limit, velocity_limit_violation_steps=velocity_limit,
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT/'teleop_assist/isaac_backend.py']},
        scope_limits=['Known scene coordinates select this oracle kinematics probe; not learned perception.',
            'Compound drawer front plus U-handle partner, not contact-face classification.',
            'No production Session API, headset, hardware or automatic release validation.',
            'Additional manual motion is measured separately; assisted Execute remains limited to 40 mm.',
            'Normal G1 actuators only: no attachment, object pose writes or external object forces.'])
    exit_code = 0 if result['passed'] else 2
except BaseException:
    result['error'] = traceback.format_exc()
    print(result['error'], flush=True)
finally:
    (args.output/'report.json').write_text(json.dumps(result, indent=2))
    (args.output/'exit_status').write_text(str(exit_code))
    print('MANUAL_RELEASE_RESULT', json.dumps(result), flush=True)
    if backend:
        backend.close()
    app.close(wait_for_replicator=False, skip_cleanup=True)
raise SystemExit(exit_code)
