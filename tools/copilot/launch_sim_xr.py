#!/usr/bin/env python3
"""One explicit launch of original input-only TeleVuer plus isolated PhysX.

No synthetic input, DDS transport, stock XR entry, or physical robot-camera stream.
Optional --xr-camera displays only this run's simulator camera.
--check is filesystem-only; --run starts the existing simulator/input processes.
"""
from __future__ import annotations
import argparse
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_copilot import configuration, group_process_details, stop_children


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='Validate files and print exact plan; no processes/network')
    mode.add_argument('--run', action='store_true', help='Start the actual input source and isolated simulator')
    p.add_argument('--config', type=Path, default=ROOT/'configs/copilot.local.json')
    p.add_argument('--runtime', type=Path, required=True, help='New session directory; never reuse an old mailbox')
    p.add_argument('--cert-file', type=Path, required=True, help='Existing TLS certificate; never generated or changed')
    p.add_argument('--key-file', type=Path, required=True, help='Existing TLS private key; contents never copied or logged')
    p.add_argument('--host', default='127.0.0.1', help='XR input bind IP; non-loopback must be chosen explicitly')
    p.add_argument('--xr-port', type=int, default=8012)
    p.add_argument('--dashboard-port', type=int, default=8770)
    p.add_argument('--duration', type=float, default=300., help='Whole session deadline including simulator startup, <=3600 s')
    p.add_argument('--startup-timeout', type=float, default=120., help='Bound for initial PhysX readiness, <=300 s')
    p.add_argument('--live-ai', action='store_true', help='Live learned proposals; manual mode adds opt-in bounded grasp alignment')
    p.add_argument('--control-mode',choices=('unitree_manual','guarded_assist'),default='unitree_manual',
        help='Default manual Unitree IK in isolated Isaac; guarded assistance is an explicit alternative')
    p.add_argument('--xr-camera', action='store_true',
        help='Read-only native operator image from this run only; no movement permission')
    p.add_argument('--camera-source', choices=('head', 'right_wrist', 'left_wrist'), default='head')
    p.add_argument('--operator-camera-source', choices=('head', 'right_wrist', 'left_wrist'), default='head',
        help='VR operator image, independent of --camera-source used for perception')
    p.add_argument('--prompt', default='red cylinder')
    p.add_argument('--sim-initialization', type=Path,
        default=ROOT/'teleop_assist/copilot/assets/manual_neutral/initialization.json',
        help='Simulation initial condition; default neutral untwisted wrists; no physical robot move')
    return p


def build_plan(args):
    if (not math.isfinite(args.duration) or not 1 <= args.duration <= 3600 or
            not math.isfinite(args.startup_timeout) or not 0 < args.startup_timeout <= 300):
        raise ValueError('Duration must be 1..3600 s; startup timeout must be positive and at most 300 s')
    if (not 1 <= args.xr_port <= 65535 or not 1 <= args.dashboard_port <= 65535 or
            args.xr_port == args.dashboard_port):
        raise ValueError('XR/dashboard ports must be different and in 1..65535')
    address = ipaddress.ip_address(args.host)
    if address.is_unspecified or address.is_multicast or getattr(address, 'scope_id', None):
        raise ValueError('Use a concrete loopback or reachable unicast IP, without an IPv6 zone suffix')
    config_path = args.config.resolve()
    _, paths = configuration(config_path)
    required = {'isaac_python', 'xr_python', 'isaac_repository', 'xr_repository'}
    if args.live_ai:
        required.update(('perception_python', 'grasp_python', 'grasp_repository', 'model_root'))
    missing = required - paths.keys()
    if missing:
        raise ValueError('Missing config paths: ' + ', '.join(sorted(missing)))
    for name in sorted(required):
        file = Path(paths[name])
        if name.endswith('_python'):
            if not file.is_file() or not os.access(file, os.X_OK):
                raise ValueError('Configured interpreter is not executable: ' + name)
        elif not file.is_dir():
            raise ValueError('Configured directory is unavailable: ' + name)
    for name in ('run_copilot.py', 'tools/copilot/xr_source.py'):
        if not (ROOT/name).is_file():
            raise ValueError('Required existing entry point missing: ' + name)
    runtime = args.runtime.resolve()
    if runtime.exists():
        raise ValueError('Session directory already exists; choose a new directory')
    cert, key = args.cert_file.resolve(), args.key_file.resolve()
    if cert == key or not cert.is_file() or not key.is_file():
        raise ValueError('Separate existing certificate and private-key files required')
    for item in (cert, key, config_path):
        if runtime == item or runtime in item.parents:
            raise ValueError('Config/TLS inputs must be outside the new session directory')
    authority = f'[{address}]:{args.xr_port}' if address.version == 6 else f'{address}:{args.xr_port}'
    socket_url = 'wss://' + authority + '/'
    browser_url = 'https://' + authority + '/?' + urlencode({'ws': socket_url})
    mailbox = runtime/'operator.json'
    sim = [paths['isaac_python'], str(ROOT/'run_copilot.py'), '--assist-sim',
        '--operator-mailbox', str(mailbox), '--sim-sender', 'guarded_original',
        '--sim-command-format', 'unitree_float32', '--config', str(config_path),
        '--runtime', str(runtime/'sim'), '--port', str(args.dashboard_port),
        '--duration', str(args.duration), '--camera-source', args.camera_source, '--prompt', args.prompt]
    if args.live_ai:
        sim.append('--live-ai')
    initialization_hash = None
    if args.sim_initialization is not None:
        initialization = args.sim_initialization.resolve(strict=True)
        content = initialization.read_bytes()
        condition = json.loads(content)
        if not isinstance(condition, dict) or condition.get('accepted') is not True:
            raise ValueError('Accepted simulation initial condition JSON required')
        sim += ['--sim-initialization', str(initialization)]
        initialization_hash = hashlib.sha256(content).hexdigest()
    if args.xr_camera:
        sim += ['--operator-camera-source', args.operator_camera_source]
    if args.control_mode == 'unitree_manual':
        if args.sim_initialization is None:
            raise ValueError('Manual simulation needs an initial condition')
        sim = [paths['isaac_python'], str(ROOT/'tools/copilot/manual_sim_runtime.py'), '--sim',
            '--initialization',str(args.sim_initialization.resolve()),
            '--operator-mailbox',str(mailbox),'--output',str(runtime/'sim/simulation'),
            '--port',str(args.dashboard_port),'--duration',str(args.duration),'--headless',
            '--repo',paths['isaac_repository'],'--xr-repo',paths['xr_repository'],
            '--ik-python',paths['xr_python'],'--operator-camera-source',args.operator_camera_source]
        if args.live_ai:
            sim += ['--grasp-assist','--config',str(config_path),'--prompt',args.prompt]
    source = [paths['xr_python'], str(ROOT/'tools/copilot/xr_source.py'), '--sim',
        '--repo', paths['xr_repository'], '--input-mode', 'controller', '--vr-buttons',
        '--host', str(address), '--port', str(args.xr_port), '--output', str(mailbox),
        '--stage', str(runtime/'xr_stage'), '--duration', str(args.duration),
        '--cert-file', str(cert), '--key-file', str(key)]
    # The inner run_copilot runtime is <session>/sim. No external camera-path
    # option exists: source starts only after this run's fresh PhysX status.
    camera_directory = runtime/'sim/simulation/operator_camera' if args.xr_camera else None
    if camera_directory is not None:
        source += ['--sim-camera', str(camera_directory)]
    return dict(schema='punktnull.input_only_xr_sim_plan.v1', runtime=str(runtime),
        source_command=source, simulation_command=sim,
        operator_file=str(mailbox), runtime_status=str(runtime/'sim/simulation/sim_runtime.json'),
        browser_url=browser_url, dashboard_url=f'http://127.0.0.1:{args.dashboard_port}',
        controls_url=f'http://127.0.0.1:{args.dashboard_port}/sim',
        duration_s=args.duration, startup_timeout_s=args.startup_timeout,
        live_ai=args.live_ai, control_mode=args.control_mode,
        custom_motion_guard_enabled=args.control_mode=='guarded_assist',
        input_only=True, xr_display_mode='pass-through',
        perception_camera_source='head' if args.control_mode=='unitree_manual' and args.live_ai else args.camera_source,
        operator_camera_source=args.operator_camera_source,
        sim_initialization_sha256=initialization_hash,
        xr_camera_requested=args.xr_camera,
        sim_camera_directory=str(camera_directory) if camera_directory is not None else None,
        sim_camera_scope='Optional read-only native ego image; no permission or verified browser/headset delivery',
        sim_camera_delivery_verified=False,
        robot_camera_stream_in_XR=False, synthetic_inputs=False, stock_XR_entry_executed=False,
        hardware_authorized=False, DDS_transport_requested=False,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())


def read_json(path):
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            return None
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def run_session(plan):
    out = Path(plan['runtime'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'launch_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    children, logs = [], []
    stopping = False
    report = dict(schema='punktnull.input_only_xr_sim_session.v1', started_at=time.monotonic(),
        launcher_completed=False, cleanup_accepted=False, synthetic_inputs=False,
        stock_XR_entry_executed=False, hardware_verified=False, headset_verified=False,
        xr_camera_requested=plan['xr_camera_requested'],
        sim_camera_directory=plan['sim_camera_directory'], sim_camera_delivery_verified=False,
        robot_camera_stream_in_XR=False, simulation_acceptance_claimed=False,
        commands={}, children={})
    deadline = report['started_at'] + plan['duration_s']
    def request_stop(*_):
        nonlocal stopping
        stopping = True
    previous = {number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)}
    def start(name, command):
        stream = (out/(name+'.log')).open('w')
        logs.append(stream)
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                     OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
            start_new_session=True)
        children.append((name, process))
        report['commands'][name] = command
        report['children'][name] = dict(pid=process.pid, own_new_session=True)
        return process
    def wait_until(predicate, until, label):
        while not predicate():
            if stopping:
                raise InterruptedError('Stop requested')
            if time.monotonic() >= until:
                raise TimeoutError(label)
            for name, process in children:
                if process.poll() is not None:
                    raise RuntimeError(name + ' exited before ' + label + ': ' + str(process.returncode))
            time.sleep(.1)
    clock_id = 'host_monotonic:' + Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    def sim_ready():
        status = read_json(plan['runtime_status'])
        if not status:
            return False
        stamp = status.get('timestamp')
        return (status.get('schema') == 'punktnull.sim_runtime.v1' and
                status.get('mode') == 'isolated_PhysX' and status.get('clock_id') == clock_id and
                status.get('robot_transport_started') is False and status.get('scripted_operator') is False and
                type(stamp) in (int, float) and 0 <= time.monotonic()-stamp < .5)
    try:
        simulation = start('simulation_launcher', plan['simulation_command'])
        wait_until(sim_ready, min(deadline, time.monotonic()+plan['startup_timeout_s']), 'fresh isolated PhysX status')
        source_command = list(plan['source_command'])
        # Source startup shares the original session deadline; no lease is extended.
        source_command[source_command.index('--duration')+1] = str(max(.01, deadline-time.monotonic()))
        source = start('xr_source', source_command)
        def endpoint_reported():
            log = out/'xr_source.log'
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size-65536))
                return b'punktnull.xr_input_endpoint.v1' in stream.read(65536)
        wait_until(endpoint_reported, min(deadline, time.monotonic()+25.), 'XR source endpoint report')
        print(json.dumps(dict(schema='punktnull.xr_sim_endpoints.v1',
            browser_url=plan['browser_url'], dashboard_url=plan['dashboard_url'], controls_url=plan['controls_url'],
            input_only=True, xr_display_mode='pass-through', robot_camera_stream_in_XR=False,
            xr_camera_requested=plan['xr_camera_requested'],
            sim_camera_directory=plan['sim_camera_directory'], sim_camera_delivery_verified=False,
            synthetic_inputs=False, headset_verified=False)), flush=True)
        report['endpoints_reported_at'] = time.monotonic()
        while not stopping and time.monotonic() < deadline:
            for name, process in children:
                if process.poll() is not None:
                    # At the same configured deadline the child may finish first.
                    if process.returncode == 0 and deadline-time.monotonic() < .5:
                        break
                    raise RuntimeError(name + ' exited during the session: ' + str(process.returncode))
            else:
                time.sleep(.1)
                continue
            break
        report['stop_reason'] = 'signal' if stopping else 'duration_finished'
        report['launcher_completed'] = True
    except InterruptedError:
        report.update(stop_reason='signal', launcher_completed=True)
    except Exception as error:
        report['failure'] = type(error).__name__ + ': ' + str(error)
    finally:
        # Stop input first. The existing simulation launcher owns separately
        # created GPU/model sessions and gets longer than its own cleanup bound.
        report['returncodes_before_cleanup'] = {name: process.poll() for name, process in children}
        forced, cleanup_errors = {}, {}
        for name, process in reversed(children):
            try:
                details = {}
                killed = stop_children([(name, process)], grace_s=30., forced_details=details)
                forced.update({item: details.get(item, []) for item in killed})
            except Exception as error:
                cleanup_errors[name] = type(error).__name__ + ': ' + str(error)
        report['forced_process_groups'] = forced
        report['cleanup_errors'] = cleanup_errors
        report['returncodes'] = {name: process.poll() for name, process in children}
        report['remaining_owned_groups'] = {name: group_process_details(process.pid) for name, process in children}
        # A clean nested supervisor exit is not enough: retain its own cleanup report.
        inner = read_json(out/'sim/session.json')
        report['nested_simulation_session'] = inner
        nested_clean = bool(inner and inner.get('cleanup_accepted') is True)
        report['cleanup_accepted'] = (not forced and not cleanup_errors and
            not any(report['remaining_owned_groups'].values()) and nested_clean and
            all(code == 0 for code in report['returncodes'].values()))
        if forced.get('simulation_launcher') or not nested_clean:
            report['open_cleanup_limit'] = ('Nested simulator/model groups are separately owned by run_copilot; '
                'without its clean session report this launcher cannot assert their cleanup.')
        for stream in logs:
            stream.close()
        report['finished_at'] = time.monotonic()
        (out/'launcher_session.json').write_text(json.dumps(report, indent=2) + '\n')
        for number, handler in previous.items():
            signal.signal(number, handler)
    print(json.dumps({key: report.get(key) for key in ('launcher_completed', 'cleanup_accepted', 'failure', 'stop_reason')}), flush=True)
    return 0 if report['launcher_completed'] and report['cleanup_accepted'] else 1


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        plan = build_plan(args)
    except (OSError, ValueError, TypeError, KeyError) as error:
        p.error(str(error))
    if args.check:
        print(json.dumps(dict(mode='FILESYSTEM_CHECK_ONLY', processes_started=False, network_started=False,
                              plan=plan), indent=2))
        return 0
    return run_session(plan)


if __name__ == '__main__':
    raise SystemExit(main())
