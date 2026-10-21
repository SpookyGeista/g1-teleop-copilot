#!/usr/bin/env python3
"""Private JSON worker for original IK and complete-scene command checking."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import select
import signal
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--mode', choices=('guarded','unitree_manual_sim'), default='guarded')
    args = parser.parse_args()
    stopping = False
    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    # Install before original IK imports logging_mp. Its SIGTERM-ignoring
    # listener must receive the library's normal atexit shutdown sentinel.
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def lines():
        """Bounded framing that can stop even with an open or partial pipe."""
        pending = b''
        while not stopping:
            if not select.select([sys.stdin.fileno()], [], [], .1)[0]:
                continue
            chunk = os.read(sys.stdin.fileno(), 65536)
            if not chunk:
                return
            pending += chunk
            while b'\n' in pending and not stopping:
                line, pending = pending.split(b'\n', 1)
                if len(line) > 2*1024*1024:
                    raise ValueError('Oversized request')
                yield line
            if len(pending) > 2*1024*1024:
                raise ValueError('Oversized request')
    wire = sys.stdout
    def send(value):
        wire.write(json.dumps(value, allow_nan=False)+'\n'); wire.flush()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            from scipy.spatial.transform import Rotation
            from teleop_assist.copilot.guard import ActualMotionGuard
            from teleop_assist.copilot.endpoint_ik import EndpointIKValidator
            from teleop_assist.copilot.observations import rigid
            if args.mode == 'unitree_manual_sim':
                from teleop_assist.copilot.kinematics import OriginalIKValidator, arm_vector
                original = OriginalIKValidator(args.repository)
                manifest = dict(original.manifest, kind='original_Unitree_G1_29_continuous_manual_IK',
                    custom_motion_guard_enabled=False, simulation_only=True,
                    per_candidate_reset=False, trajectory_validated=False)
            else:
                guard = ActualMotionGuard(args.repository)
                endpoint_ik = EndpointIKValidator(guard.collision.model,guard.ik)
                guard.manifest['candidate_endpoint_ik']=endpoint_ik.manifest
                manifest = guard.manifest
        if stopping:
            return 0
        send({'status': 'ready', 'manifest': manifest})
    except Exception:
        if stopping:
            return 0
        send({'status': 'error', 'error': traceback.format_exc()}); return 1
    for line in lines():
        request_id = None
        try:
            if len(line) > 2*1024*1024: raise ValueError('Oversized request')
            value = json.loads(line); request_id = value['request_id']
            work_started=time.monotonic();cpu_started=time.process_time()
            with contextlib.redirect_stdout(sys.stderr):
                if args.mode == 'unitree_manual_sim':
                    if value['operation'] not in ('original_manual_ik','original_manual_reset'):
                        raise ValueError('Only original manual simulator IK is available')
                    current = arm_vector(value['current_q'])
                    if value['operation'] == 'original_manual_reset':
                        original.solver.init_data = current.copy()
                        original.solver.smooth_filter = original._filter_type(np.array([.4,.3,.2,.1]),14)
                        for _ in range(4): original.solver.smooth_filter.add_data(current.copy())
                        result = dict(reset=True, simulation_only=True)
                    else:
                        q, torque = original.solver.solve_ik(rigid(value['left_base_ee']),
                            rigid(value['right_base_ee']), current, arm_vector(value['current_dq']))
                        result = dict(q14=arm_vector(q).tolist(),
                            tau14=arm_vector(torque).tolist(),
                            custom_motion_guard_enabled=False, simulation_only=True)
                elif value['operation'] == 'validate':
                    result = guard.validate(value['left_base_ee'], value['right_base_ee'],
                        value['robot_state'], value['scene'], acquisition_timestamp=value['acquisition_timestamp'],
                        clock_id=value['clock_id'], now=time.monotonic(), tracking_valid=value.get('tracking_valid', False),
                        joint_reference=value.get('joint_reference'),
                        command_format=value.get('command_format', 'float64'))
                elif value['operation'] == 'validate_sim_candidate':
                    # Endpoint evidence only. The consumer still validates the
                    # actual human+assistance command and its short path.
                    state, scene = value['robot_state'], value['scene']
                    right, left = rigid(value['base_ee']), rigid(value['left_base_ee'])
                    full = guard.collision._q(state['joint_positions'])
                    ik = endpoint_ik.validate_pose(right, left, full)
                    checked = guard.collision.validate_request({'batch': {'candidates': [{
                        'candidate_id': 'sim_endpoint', 'base_ee': right.tolist(),
                        'ik_valid': ik['ik_valid'], 'kinematic_validation': ik}]},
                        'robot_state': state, 'scene': scene}, endpoint_only=True)
                    # validate_request establishes complete observed-state/FK
                    # consistency even if a direct home-to-grasp path fails.
                    evidence = checked['arm_collision_validation']
                    endpoint = checked['candidates'][0]['arm_collision_validation'].get('endpoint_precheck', {})
                    result = {'accepted': bool(evidence.get('reason') == 'evaluated' and ik['ik_valid']
                            and endpoint.get('collision_valid') is True),
                        'ik': ik, 'endpoint': endpoint, 'state_evidence': evidence,
                        'motion_authorized': False, 'scope': 'sim_candidate_endpoint_only'}
                elif value['operation'] == 'prepare_sim_initialization':
                    # A numerical initial-condition proposal, not an approach
                    # command. Isaac will explicitly initialize this test pose.
                    candidate = value['candidate']
                    state = value['robot_state']; scene = value['scene']
                    full = guard.collision._q(state['joint_positions'])
                    full[guard.collision.arm_indices] = candidate['kinematic_validation']['q_solution']
                    goal = rigid(candidate['base_ee'])
                    left = rigid(candidate['kinematic_validation']['left_target_base_ee'])
                    obstacles, allowed = guard.collision._obstacles(scene)
                    base = rigid(state['base_in_world'])
                    trials = []; result = {'accepted': False}
                    for axis in (2, 1, 0):
                        for sign in (1, -1):
                            raw = goal.copy(); rv = np.zeros(3); rv[axis] = sign*np.deg2rad(12)
                            raw[:3, :3] = Rotation.from_rotvec(rv).as_matrix() @ goal[:3, :3]
                            ik = guard.ik.validate_pose(raw, left, full[guard.collision.arm_indices])
                            q = full.copy(); q[guard.collision.arm_indices] = ik['q_solution']
                            collision = guard.collision._check_path(q, q, base, obstacles, allowed)
                            trials.append({'axis': axis, 'sign': sign, 'ik': ik, 'collision': collision})
                            if ik['ik_valid'] and collision['collision_valid']:
                                result = {'accepted': True, 'raw_right_base_ee': raw.tolist(),
                                    'left_base_ee': left.tolist(), 'goal_base_ee': goal.tolist(),
                                    'joint_positions': {name: float(q[guard.collision.model.joints[jid].idx_q])
                                        for jid, name in enumerate(guard.collision.model.names) if jid > 0},
                                    'initial_orientation_error_deg': 12.}
                                break
                        if result['accepted']: break
                    result.update(trials=trials, scope='offline_simulation_initial_condition_only', motion_authorized=False)
                else: raise ValueError('Unknown operation')
            if stopping:
                return 0
            send({'status': 'ok', 'request_id': request_id, 'result': result,
                'diagnostic_timing': dict(operation=value['operation'],
                    computation_wall_s=time.monotonic()-work_started,
                    computation_process_cpu_s=time.process_time()-cpu_started)})
        except Exception:
            if stopping:
                return 0
            send({'status': 'error', 'request_id': request_id, 'error': traceback.format_exc()})


if __name__ == '__main__':
    raise SystemExit(main())
