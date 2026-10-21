#!/usr/bin/env python3
"""Bounded contact/retraction audit in actual PhysX; no product or pose mutation."""
from __future__ import annotations

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
parser.add_argument('--direct-backend', action='store_true', help='Audit the backend gate in isolation')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error('Use a fresh output directory')
args.output.mkdir(parents=True)
args.enable_cameras = True
args.device = 'cpu'
app = AppLauncher(args).app
source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [Path(__file__).resolve(), ROOT/'teleop_assist/isaac_backend.py',
                           ROOT/'teleop_assist/core.py']}

try:
    import numpy as np
    from PIL import Image
    from teleop_assist.core import AssistSession, XR_TO_ROBOT
    from teleop_assist.isaac_backend import IsaacBackend

    backend = IsaacBackend(args.output/'sim', headless=args.headless, width=640, height=480)
    session = AssistSession()
    approach = np.array([.416, -.269, .985])
    contact_target = np.array([.64, -.269, .985])
    start = dict(backend.last_state)
    state = dict(start)
    sequence = 0

    def input_sample(target, grip, clutch=True):
        global sequence
        sequence += 1
        session.command({'type': 'input', 'sample': {
            'source': 'synthetic_contact_audit', 'session_id': 'contact_audit', 'seq': sequence,
            'orientation_mode': 'position', 'xr_to_world_rotation': XR_TO_ROBOT.tolist(),
            'right': {'valid': True, 'position': (XR_TO_ROBOT.T@(target-np.asarray(start['ee_position']))).tolist(),
                      'quaternion_xyzw': [0., 0., 0., 1.]},
            'axes': [0., 0.], 'trigger': grip, 'squeeze': 1. if clutch else 0., 'clutch': clutch}})

    input_sample(np.asarray(start['ee_position']), 0., False)
    session.robot = state
    session.command({'type': 'calibrate'})
    contact_state = None
    max_force = 0.
    invalid_steps = 0
    states = []
    phase_records = {}
    contact_grip_increment_max = 0.
    contact_finger_closure_reference_max = 0.
    Image.fromarray(backend.capture()['rgb']).save(args.output/'initial_rgb.png')
    with (args.output/'contact_trace.jsonl').open('w') as log:
        for phase, requested, max_steps, grip in [('approach', approach, 1600, .25),
                                                 ('seek_contact', contact_target, 1600, .25),
                                                 ('blocked_forward', contact_target, 1, 1.),
                                                 ('request_retreat', approach, 1000, 0.),
                                                 ('seek_contact_again', contact_target, 1600, .25),
                                                 ('request_retreat_under_contact', approach, 1000, 0.)]:
            phase_start = backend.last_state['sim_time']
            phase_rows = []
            for index in range(max_steps):
                input_sample(requested, grip)
                core_state = state
                core_reference = session.step(core_state, backend.dt)
                if args.direct_backend:
                    state = backend.step(requested, grip=grip)
                else:
                    if core_reference.get('hold'):
                        backend.hold()
                    state = backend.step(core_reference['ee_target_world'], core_reference['ee_quat_xyzw'],
                                         core_reference['grip'])
                row = {'phase': phase, 'requested_target_world': requested.tolist(),
                       'state': state, 'core_hold': bool(core_reference.get('hold')),
                       'core_reason': session.control['reason'], 'reference': core_reference,
                       'previous_state_contact_active': core_state.get('contact_active', False),
                       'requested_grip': grip}
                log.write(json.dumps(row)+'\n')
                phase_rows.append(row)
                max_force = max(max_force, state['contact_force_max'])
                invalid_steps += int(not state['valid'])
                if state['contact_active'] and 'accepted_grip' in state:
                    contact_grip_increment_max = max(contact_grip_increment_max,
                                                    state['accepted_grip']-core_state.get('accepted_grip', 0.))
                    for joint in backend.finger_ids:
                        name = backend.names[joint]
                        direction = -1. if ('thumb_1' in name or 'thumb_2' in name) else 1.
                        contact_finger_closure_reference_max = max(contact_finger_closure_reference_max,
                            direction*(state['joint_targets'][joint]-core_state['joints'][joint]))
                if index % 80 == 0:
                    backend.capture()
                if not state['valid']:
                    break
                if phase.startswith('seek_contact') and state['collision_stop']:
                    contact_state = dict(state)
                    Image.fromarray(backend.capture()['rgb']).save(args.output/'first_contact_rgb.png')
                    break
            phase_records[phase] = {'start_sim_time': phase_start,
                                    'end_sim_time': backend.last_state['sim_time'],
                                    'steps': len(phase_rows),
                                    'guard_steps': sum(int(r['state']['collision_stop']) for r in phase_rows),
                                    'core_collision_steps': sum(r['core_reason'] == 'collision' for r in phase_rows),
                                    'core_retreat_steps': sum(r['core_reason'] == 'contact_retreat' for r in phase_rows),
                                    'backend_retreat_steps': sum(bool(r['state'].get('contact_retreat_active')) for r in phase_rows),
                                    'max_accepted_grip_during_contact': max((r['state'].get('accepted_grip', 0.) for r in phase_rows
                                                                          if r['state']['contact_active']), default=None)}
            if phase.startswith('request_retreat'):
                states = phase_rows
            if invalid_steps:
                break
    final = dict(backend.last_state)
    recent = states[-200:]
    force_mean = float(np.mean([r['state']['contact_force_max'] for r in recent])) if recent else None
    retreat_blocked = bool(recent and all(r['state']['collision_stop'] and not r['state']['command_accepted'] for r in recent))
    displacement = float(contact_state['ee_position'][0]-final['ee_position'][0]) if contact_state else None
    result = {
        'kind': 'real_physx_contact_retraction_audit',
        'control_path': 'direct backend' if args.direct_backend else 'production AssistSession and IsaacBackend',
        'physical_contact_reached': contact_state is not None,
        'contact_guard_threshold_N': 18.,
        'max_measured_contact_force_N': max_force,
        'last_second_mean_contact_force_N': force_mean,
        'last_second_all_retreat_commands_blocked': retreat_blocked,
        'retreat_displacement_along_minus_world_X_m': displacement,
        'invalid_steps': invalid_steps,
        'maximum_grip_increase_while_contact_active': contact_grip_increment_max,
        'maximum_finger_closing_reference_beyond_measured_joint_rad': contact_finger_closure_reference_max,
        'phases': phase_records,
        'initial_state': start, 'contact_state': contact_state, 'final_state': final,
        'source_hashes': source_hashes,
        'limitations': ['One deliberate simulated cabinet contact with fixed pelvis and synthetic targets.',
                        'Targets are declared test fixtures, not outputs of perception or an operator.',
                        'The reactive 18N threshold is not a maximum force clamp or a safety certification.',
                        'This audit does not validate contact repair or recovery from opposing simultaneous contacts.',
                        'Contact forces are sampled by the existing backend guard before the next physics step.'],
        'conclusion': ('Sustained contact blocks the requested retreat in the final recorded second.'
                       if retreat_blocked else 'No sustained blocked-retreat condition established in the final recorded second.')}
    if not args.direct_backend:
        result['checks'] = {
            'contact_observed': contact_state is not None,
            'forward_held_by_core': phase_records.get('blocked_forward', {}).get('core_collision_steps', 0) > 0,
            'retreat_during_contact_core': any(p['core_retreat_steps'] > 0 for p in phase_records.values()),
            'retreat_during_contact_backend': any(p['backend_retreat_steps'] > 0 for p in phase_records.values()),
            'contact_cleared': force_mean is not None and force_mean < 1.,
            'measured_retreat_at_least_5cm': displacement is not None and displacement > .05,
            'additional_finger_closure_blocked': contact_grip_increment_max < 1e-6 and contact_finger_closure_reference_max < 1e-6,
            'physical_states_valid': invalid_steps == 0}
        result['passed'] = all(result['checks'].values())
    (args.output/'contact_result.json').write_text(json.dumps(result, indent=2))
    Image.fromarray(backend.capture()['rgb']).save(args.output/'final_rgb.png')
    print('ASSIST_CONTACT_RESULT', json.dumps({k:v for k,v in result.items() if k not in ('initial_state','contact_state','final_state')}), flush=True)
    backend.close()
except BaseException:
    error = traceback.format_exc()
    (args.output/'contact_error.txt').write_text(error)
    print(error, flush=True)
    raise
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
