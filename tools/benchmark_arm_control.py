#!/usr/bin/env python3
"""Small real-Isaac arm benchmark; four runs are serialized with --suite.

Run with the installed Isaac Python interpreter. No headset/server/perception
process is started, and no RGB-D archive is recorded. Never reuse an output.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def source_hashes():
    files = ('teleop_assist/arm_benchmark.py', 'tools/benchmark_arm_control.py',
             'teleop_assist/core.py', 'teleop_assist/predictive.py',
             'teleop_assist/actions.py', 'teleop_assist/control_ownership.py',
             'teleop_assist/isaac_backend.py', 'teleop_assist/retained_reference.py',
             'teleop_assist/unitree_arm_ik.py', 'teleop_assist/unitree_ik_worker.py')
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in files if (ROOT/name).is_file()}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm-solver', choices=('dls', 'unitree'), default='dls')
    parser.add_argument('--assistance', choices=('manual', 'geometric'), default='manual')
    parser.add_argument('--suite', action='store_true', help='Four isolated solver/mode runs, sequentially')
    parser.add_argument('--live-scheduling', action='store_true',
                        help='Async production IK scheduling, wall-paced physics and host-clock input ages')
    parser.add_argument('--gui', action='store_true', help='Show actual Isaac viewport (default headless)')
    parser.add_argument('--headless', action='store_true', help='Explicit default; mutually exclusive with --gui')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.gui and args.headless:
        parser.error('Choose either --gui or --headless')
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error('Output must be a new directory; previous failures are preserved')
    return args


def suite(args):
    from teleop_assist.arm_benchmark import SCHEMA, compare_trials, scenario_contract
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': SCHEMA, 'synthetic_operator': True, 'headset_verified': False,
              'real_robot_verified': False, 'contract': scenario_contract(), 'runs': [],
              'source_sha256': source_hashes(), 'created_at': datetime.now(timezone.utc).isoformat()}
    for solver in ('dls', 'unitree'):
        for assistance in ('manual', 'geometric'):
            name = solver+'_'+assistance
            output = args.output/name
            command = [sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                       '--arm-solver', solver, '--assistance', assistance, '--seed', str(args.seed)]
            if args.gui:
                command.append('--gui')
            if args.live_scheduling:
                command.append('--live-scheduling')
            print('ARM_BENCHMARK_RUN '+name, flush=True)
            with (args.output/(name+'.log')).open('w') as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            result_path = output/'report.json'
            result = json.loads(result_path.read_text()) if result_path.is_file() else None
            entry = {'arm_solver': solver, 'assistance': assistance, 'exit_code': process.returncode,
                     'command': command, 'report': str(result_path),
                     'passed': process.returncode == 0 and bool(result and result.get('passed')),
                     'result': result}
            report['runs'].append(entry)
            write_json(args.output/'report.json', report)
    report['assistance_comparisons'] = {}
    for solver in ('dls', 'unitree'):
        rows = [r for r in report['runs'] if r['arm_solver'] == solver]
        metrics = {r['assistance']: ((r.get('result') or {}).get('scenarios', {}).get('misalignment', {}))
                   for r in rows}
        report['assistance_comparisons'][solver] = compare_trials(metrics['manual'], metrics['geometric'])
    report['passed'] = (all(r['passed'] for r in report['runs']) and
                        all(c['passed'] for c in report['assistance_comparisons'].values()))
    report['interpretation'] = ('All negative trials retained. Compare solver metrics and initial measured poses; '
                                'no winner is selected automatically. Simulation time is not headset latency.')
    write_json(args.output/'report.json', report)
    print('ARM_BENCHMARK_REPORT '+str(args.output/'report.json'), flush=True)
    return 0 if report['passed'] else 2


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'starting', 'passed': False, 'synthetic_operator': True,
              'headset_verified': False, 'real_robot_verified': False,
              'perception_evaluated': False, 'contact_task_evaluated': False,
              'arm_solver': args.arm_solver, 'assistance': args.assistance,
              'live_scheduling': args.live_scheduling,
              'created_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_hashes(),
              'scenarios': {}}
    write_json(args.output/'report.json', report)
    app = backend = None
    startup = time.perf_counter()
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=not args.gui, enable_cameras=True, device='cpu').app
        import numpy as np
        from teleop_assist.arm_benchmark import (INPUT_HZ, SCENARIOS, SCHEMA, SyntheticArmOperator,
                                                scenario_contract, scenario_duration, trial_metrics)
        from teleop_assist.isaac_backend import IsaacBackend
        backend = IsaacBackend(args.output/'sim', headless=not args.gui, seed=args.seed,
                               width=320, height=240, scene_variant='interaction', wrist_cameras=False,
                               start_pose='reference', arm_solver=args.arm_solver,
                               arm_solver_sync=not args.live_scheduling)
        dt = backend.dt
        packet_stride = int(round(1./INPUT_HZ/dt))
        if not math_isclose(packet_stride*dt, 1./INPUT_HZ):
            raise RuntimeError('Physics dt does not divide the fixed benchmark packet interval')
        report.update(schema=SCHEMA, contract=scenario_contract(), physics_dt_s=dt,
                      startup_wall_seconds=time.perf_counter()-startup,
                      backend_manifest=backend.manifest,
                      initial_state=backend.last_state,
                      solver_execution=('asynchronous_wall_paced' if args.live_scheduling else
                                        'synchronous_for_deterministic_simulation'),
                      trajectory_reset='physical_return_only', status='running')
        if args.live_scheduling:
            report['contract']['clock'] = ('Host monotonic input ages; physics and scenario trajectories '
                                           'remain simulation time. No network/headset latency is evaluated.')
        total_steps = 0
        benchmark_start = time.perf_counter()
        tick_deadline = benchmark_start+dt
        deadline_lateness = []
        def input_clock():
            return time.perf_counter()-benchmark_start if args.live_scheduling else total_steps*dt
        def pace():
            nonlocal tick_deadline
            if not args.live_scheduling:
                return
            remaining = tick_deadline-time.perf_counter()
            deadline_lateness.append(max(0., -remaining)*1000)
            if remaining > 0:
                time.sleep(remaining)
            # Do not fabricate burst-rate controller input after a slow step.
            tick_deadline = max(tick_deadline, time.perf_counter())+dt
        with (args.output/'events.jsonl').open('w', buffering=1) as events, \
                (args.output/'trajectory.jsonl').open('w', buffering=1) as trace:
            operator = SyntheticArmOperator(args.assistance,
                event_sink=lambda row: events.write(json.dumps(row, allow_nan=False)+'\n'))
            report['assist_config'] = asdict(operator.session.config)
            for scenario in SCENARIOS:
                operator.begin(scenario, backend.last_state, input_clock())
                samples = []
                scenario_start = time.perf_counter()
                for step in range(int(round(scenario_duration(scenario)/dt))):
                    if not app.is_running():
                        raise RuntimeError('Isaac stopped before the scripted trajectory finished')
                    loop_start = time.perf_counter()
                    reference, expected = operator.reference(step*dt, input_clock(),
                        backend.last_state, dt, send_packet=step % packet_stride == 0)
                    session_ms = (time.perf_counter()-loop_start)*1000
                    if reference['hold'] or reference.get('reset_pending', False):
                        backend.hold()
                    before_backend = time.perf_counter()
                    state = backend.step(reference['ee_target_world'], reference.get('ee_quat_xyzw'),
                                         reference['grip'])
                    backend_ms = (time.perf_counter()-before_backend)*1000
                    if args.gui and total_steps % 20 == 0:
                        backend.sim.render()
                    sample = {'scenario': scenario, 'scenario_time': step*dt,
                              'sim_elapsed_s': total_steps*dt, 'physics_time_s': state['sim_time'],
                              'wall_elapsed_s': time.perf_counter()-benchmark_start,
                              'session_wall_ms': session_ms, 'backend_wall_ms': backend_ms,
                              'command_position': reference['ee_target_world'],
                              'command_quaternion': reference.get('ee_quat_xyzw'),
                              'hold': reference['hold'], **expected,
                              'ee_position': state['ee_position'], 'ee_quaternion': state['ee_quaternion'],
                              'arm_joints': np.asarray(state['joints'])[backend.arm_ids].tolist(),
                              'arm_velocities': np.asarray(state['velocities'])[backend.arm_ids].tolist(),
                              'contact_force_max': state['contact_force_max'],
                              'valid': state['valid'], 'joint_limit_violation': state['joint_limit_violation'],
                              'arm_solver': state.get('arm_solver'),
                              'backend_hold_reason': state.get('hold_reason')}
                    samples.append(sample)
                    trace.write(json.dumps(sample, allow_nan=False)+'\n')
                    total_steps += 1
                    pace()
                metrics = trial_metrics(samples, scenario, args.assistance)
                metrics.update(wall_seconds=time.perf_counter()-scenario_start,
                               simulation_seconds=len(samples)*dt,
                               initial_measured_position=operator.origin.tolist(),
                               initial_measured_quaternion=operator.orientation.as_quat().tolist())
                report['scenarios'][scenario] = metrics
                write_json(args.output/'report.json', report)
                print('ARM_BENCHMARK_SCENARIO '+scenario+' '+json.dumps(metrics), flush=True)
            # Deliberately stop packet arrival while leaving the last packet's
            # clutch true. This tests expiration through production session code.
            stale = []
            for _ in range(int(round(.4/dt))):
                operator.now = 1000.+input_clock()
                reference = operator.session.step(backend.last_state, dt, now=operator.now)
                if reference['hold'] or reference.get('reset_pending', False):
                    backend.hold()
                state = backend.step(reference['ee_target_world'], reference.get('ee_quat_xyzw'),
                                     reference['grip'])
                stale.append({'input_age_sim_s': operator.now-operator.session.input_time,
                              'reason': operator.session.control['reason'],
                              'hold': reference['hold'], 'robot_valid': state['valid'],
                              'ee_position': state['ee_position']})
                total_steps += 1
                pace()
            expired = [row for row in stale if row['input_age_sim_s'] > operator.session.config.input_timeout]
            report['stale_input_check'] = {'passed': bool(expired) and all(
                row['hold'] and row['reason'] == 'input_stale' and row['robot_valid'] for row in expired),
                'scope': 'input expiry after an end hold; does not measure moving-stop distance',
                'samples': stale}
            backend.hold()
            backend.step(backend.last_state['ee_position'], None, 0.)
        report.update(status='complete', simulation_seconds=total_steps*dt,
                      wall_seconds=time.perf_counter()-benchmark_start,
                      passed=all(s['passed'] for s in report['scenarios'].values()) and
                             report['stale_input_check']['passed'])
        if deadline_lateness:
            report['wall_pacing'] = {'target_physics_hz': 1./dt,
                                    'overrun_steps': sum(value > 0 for value in deadline_lateness),
                                    'overrun_p95_ms': float(np.percentile(deadline_lateness, 95)),
                                    'overrun_max_ms': max(deadline_lateness),
                                    'note': 'Computation beyond a physics-tick budget, not headset/network latency'}
    except Exception as error:
        report.update(status='failed', passed=False, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        # Kit may implement skip_cleanup with exit(0). Evidence is committed
        # before close, and --suite checks this explicit passed field as well.
        report['source_sha256_at_finish'] = source_hashes()
        write_json(args.output/'report.json', report)
        print('ARM_BENCHMARK_REPORT '+str(args.output/'report.json'), flush=True)
        if backend is not None:
            backend.close()
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0 if report['passed'] else 2


def math_isclose(left, right):
    return abs(left-right) < 1e-9


def main(argv=None):
    args = arguments(argv)
    return suite(args) if args.suite else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
