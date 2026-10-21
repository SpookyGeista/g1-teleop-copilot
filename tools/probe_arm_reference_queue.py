#!/usr/bin/env python3
"""Opt-in real PhysX hold/reversal reference-queue audit.

Uses bounded robot-relative free-space commands; no perception or hardware.
Records residual physical motion without asserting an instantaneous stop.
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
args.enable_cameras, args.device = True, 'cpu'
app = AppLauncher(args).app

try:
    import numpy as np
    from teleop_assist.isaac_backend import IsaacBackend

    backend = IsaacBackend(args.output/'backend', headless=args.headless,
                           width=160, height=120, wrist_cameras=False,
                           scene_variant='interaction')
    ids = backend.arm_ids
    rows = []

    def record(phase, state):
        rows.append({'phase': phase, 'robot': state})
        return state

    def measured_q():
        return backend.robot.data.joint_pos[0].detach().cpu().numpy()[ids].copy()

    def target_q():
        return backend.joint_target[0, ids].detach().cpu().numpy().copy()

    initial = np.asarray(backend.last_state['ee_position'])
    goal = initial + [.04, 0., 0.]
    for _ in range(80):
        state = record('preload', backend.step(goal))
    hold_origin = np.asarray(state['ee_position'])
    before_hold_q = measured_q()
    queued_before_hold = target_q()-before_hold_q
    flushes_before = backend.arm_reference_flushes
    backend.hold()
    latched_hold_q = target_q()
    hold_gravity_offset = backend._arm_hold_gravity_offset.copy()
    hold_gravity_forces = backend._arm_hold_gravity_forces.copy()
    flush_error = float(np.max(np.abs(latched_hold_q-before_hold_q-hold_gravity_offset)))
    for _ in range(250):
        backend.hold()
        state = record('hold', backend.step(backend._measured_ee()[0]))
    hold_rows = [r['robot'] for r in rows if r['phase'] == 'hold']
    target_drift = max(float(np.max(np.abs(np.asarray(s['joint_targets'])[ids]-latched_hold_q)))
                       for s in hold_rows)
    hold_displacement = [float(np.linalg.norm(np.asarray(s['ee_position'])-hold_origin)) for s in hold_rows]
    sustained_flushes = backend.arm_reference_flushes-flushes_before
    goal = backend._measured_ee()[0]+[.035, 0., 0.]
    for _ in range(60):
        state = record('resume', backend.step(goal))
    reverse_origin = backend._measured_ee()[0]
    q_at_reverse = measured_q()
    queued_before_reverse = target_q()-q_at_reverse
    before_reverse_flushes = backend.arm_reference_flushes
    backend.hold()
    reverse_latched_q = target_q()
    reverse_flush_error = float(np.max(np.abs(reverse_latched_q-q_at_reverse-backend._arm_hold_gravity_offset)))
    reverse_goal = reverse_origin+[-.025, 0., 0.]
    first_reverse = record('reverse', backend.step(reverse_goal))
    first_reverse_reference_increment = float(np.max(np.abs(target_q()-reverse_latched_q)))
    first_reverse_cartesian_lead = np.asarray(first_reverse['servo_target'])-reverse_origin
    for _ in range(140):
        state = record('reverse', backend.step(reverse_goal))
    checks = {
        'preexisting_joint_queue_observed': float(np.linalg.norm(queued_before_hold)) > 1e-4,
        'hold_reference_is_measured_q_plus_limited_gravity_only': flush_error < 1e-6,
        'persistent_hold_flushes_once': sustained_flushes == 1,
        'persistent_hold_joint_reference_is_fixed': target_drift < 1e-7,
        'reverse_reference_is_measured_q_plus_limited_gravity_only': reverse_flush_error < 1e-6,
        'reverse_rearms_and_flushes_again': backend.arm_reference_flushes == before_reverse_flushes+1,
        'reverse_reference_joint_step_is_bounded': first_reverse_reference_increment <= .8*backend.dt+1e-6,
        'reverse_cartesian_reference_is_outward': bool(first_reverse_cartesian_lead[0] < 0.),
        'all_measured_states_valid': all(r['robot']['valid'] for r in rows),
        'free_space_probe_has_no_contact_guard': not any(r['robot']['contact_active'] for r in rows),
    }
    report = {'passed': all(checks.values()), 'checks': checks,
              'real_physx': True, 'physics_dt': backend.dt,
              'scope': 'Reference cancellation and fixed PD hold latch; no instantaneous physical-stop claim.',
              'input_kind': 'Explicit bounded robot-relative free-space commands; no perception/headset inputs.',
              'queued_arm_joint_error_before_hold_rad': queued_before_hold.tolist(),
              'hold_flush_error_rad': flush_error,
              'hold_reference_drift_rad': target_drift,
              'hold_gravity_offset_rad': hold_gravity_offset.tolist(),
              'hold_gravity_compensation_Nm': hold_gravity_forces.tolist(),
              'hold_gravity_saturated': hold_rows[0]['arm_hold_gravity_saturated'],
              'hold_peak_ee_displacement_m': max(hold_displacement),
              'hold_final_ee_displacement_m': hold_displacement[-1],
              'hold_final_arm_deflection_rad': (np.asarray(hold_rows[-1]['joints'])[ids]-latched_hold_q).tolist(),
              'queued_arm_joint_error_before_reverse_rad': queued_before_reverse.tolist(),
              'reverse_flush_error_rad': reverse_flush_error,
              'first_reverse_reference_joint_step_rad': first_reverse_reference_increment,
              'first_reverse_cartesian_reference_delta_m': first_reverse_cartesian_lead.tolist(),
              'reverse_final_measured_displacement_m': (np.asarray(state['ee_position'])-reverse_origin).tolist(),
              'contact_recovery_scope': 'Internal hold and retreat entry have CPU regression coverage; this run tests free-space dynamics only.'}
    (args.output/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print('ARM_REFERENCE_QUEUE_REPORT', json.dumps(report), flush=True)
    backend.close()
    if not report['passed']:
        raise RuntimeError('Reference queue probe checks failed; inspect report.json')
except BaseException:
    (args.output/'error.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
