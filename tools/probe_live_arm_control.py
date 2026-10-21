#!/usr/bin/env python3
"""Opt-in 20 mm simulation-only HTTP arm probe under the live vision workload.

Requires the explicit live recording directory, a verified local assist_sim PID,
an unoccupied control lease and the reference-pose fixed-base G1. Never starts a
server, resumes an emergency stop, or connects to real robot hardware.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'synthetic_live_arm_probe'
RATE = 30.
TRAVEL = .020


def write_json(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def verify_recording(recording):
    recording = Path(recording).resolve()
    if not recording.is_relative_to(ROOT/'runs') or not recording.is_dir():
        raise ValueError('--recording must identify an existing project runs/ directory')
    path = recording/'manifest.json'
    manifest = json.loads(path.read_text())
    if Path(manifest.get('output', '')).resolve() != recording:
        raise ValueError('Recording manifest output does not match the explicit directory')
    if any(manifest.get(key) is not True for key in ('simulation_only', 'fixed_base', 'no_hardware_endpoints')):
        raise ValueError('Manifest must explicitly identify fixed-base simulation without hardware endpoints')
    url = str(manifest.get('url', '')).rstrip('/')
    parts = urlsplit(url)
    if (parts.scheme != 'http' or parts.hostname != '127.0.0.1' or parts.port != 8765
            or parts.path or parts.query or parts.fragment or parts.username or parts.password):
        raise ValueError('Probe only accepts the recorded local http://127.0.0.1:8765 server')
    pid = manifest.get('pid')
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError('Manifest PID is missing or invalid')
    process = Path('/proc')/str(pid)
    arguments = [s.decode() for s in (process/'cmdline').read_bytes().split(b'\0') if s]
    cwd = (process/'cwd').resolve(strict=True)
    def absolute(token):
        path = Path(token)
        return (cwd/path).resolve() if not path.is_absolute() else path.resolve()
    if not any(absolute(token) == ROOT/'tools/assist_sim.py' for token in arguments):
        raise ValueError('Recorded PID is not the project tools/assist_sim.py process')
    if '--output' not in arguments or absolute(arguments[arguments.index('--output')+1]) != recording:
        raise ValueError('Live process output differs from the supplied recording')
    if manifest.get('perception_backend') != 'learned' or manifest.get('pose_backend') != 'foundationpose':
        raise ValueError('This probe requires the full learned perception and FoundationPose workload')
    return recording, manifest, url, {'pid': pid, 'process_arguments': arguments,
        'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def assert_state(state, client=None):
    robot = state.get('robot', {})
    if state.get('viewer_only') or state.get('mode') != 'fixed_base_manipulation_fixture':
        raise ValueError('Expected the live fixed-base manipulation session')
    if robot.get('fixed_base') is not True or robot.get('start_pose') != 'reference':
        raise ValueError('Expected a fixed-base G1 in reference-pose mode')
    if state.get('stopped') is not False:
        raise ValueError('Emergency stop is set or unavailable; this probe never resumes it')
    if robot.get('valid') is not True:
        raise ValueError('Measured robot state is invalid')
    age = time.monotonic()-float(robot.get('timestamp', -1))
    if not np.isfinite(age) or not -.01 <= age <= .5:
        raise ValueError('Measured robot state is older than 500 ms')
    force = float(robot.get('contact_force_max', float('inf')))
    if not np.isfinite(force) or not 0. <= force <= 18.:
        raise ValueError('Contact guard is active')
    if robot.get('hold_reason'):
        raise ValueError('Robot reports a motion rejection: '+str(robot['hold_reason']))
    lease = state.get('control_session', {})
    if lease.get('read_only') is not False or lease.get('owner_client_id') != client:
        raise ValueError('Control lease is occupied, lost, or read-only')
    if client is None and (state.get('input_source') not in ('none', None) or state.get('input_age') is not None):
        raise ValueError('Existing operator input must be released before this synthetic test')
    position = np.asarray(robot.get('ee_position'), float)
    quaternion = np.asarray(robot.get('ee_quaternion'), float)
    if position.shape != (3,) or not np.isfinite(position).all() or quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError('Invalid measured Cartesian pose')
    return robot


def requested_forward(elapsed):
    def ease(t):
        t = float(np.clip(t, 0., 1.))
        return t*t*(3.-2.*t)
    if elapsed < 2.: return TRAVEL*ease(elapsed/2.), 'forward'
    if elapsed < 3.: return TRAVEL, 'outbound_hold'
    if elapsed < 5.: return TRAVEL*(1.-ease((elapsed-3.)/2.)), 'return'
    return 0., 'return_hold'


class Client:
    def __init__(self, url, event_stream):
        self.url = url
        self.id = SOURCE+'_'+uuid.uuid4().hex
        self.seq = 0
        self.counts = {'GET': 0, 'POST': 0, 'errors': 0, 'input': 0}
        self.events = event_stream
        self.last_input_at = -float('inf')
        # Never honor an environment proxy or follow an HTTP redirect for controls.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs): return None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, path, data=None):
        method = 'GET' if data is None else 'POST'
        self.counts[method] += 1
        started = time.monotonic()
        body = None if data is None else json.dumps(data, allow_nan=False).encode()
        request = urllib.request.Request(self.url+path, data=body, method=method,
                                        headers={'Content-Type': 'application/json'})
        event = {'at': started, 'method': method, 'path': path,
                 'type': data.get('type') if data else None}
        try:
            with self.opener.open(request, timeout=.8) as response:
                result = json.load(response)
                event['status'] = response.status
            if data is not None and (result.get('accepted') is False or result.get('error')):
                raise ValueError('Command rejected: '+str(result))
            return result
        except Exception as exc:
            self.counts['errors'] += 1
            event['error'] = str(exc)
            raise
        finally:
            event['duration_ms'] = (time.monotonic()-started)*1000
            self.events.write(json.dumps(event, allow_nan=False)+'\n')
            self.events.flush()

    def state(self): return self.request('/api/state')

    def command(self, kind, **fields):
        return self.request('/api/command', {'type': kind, 'client_id': self.id, **fields})

    def input(self, state, forward, active):
        if not np.isfinite(forward) or not 0. <= forward <= TRAVEL+1e-12:
            raise ValueError('Synthetic input exceeds the fixed 20 mm probe envelope')
        delay = 1./RATE-(time.monotonic()-self.last_input_at)
        if delay > 0: time.sleep(delay)
        self.seq += 1
        self.last_input_at = time.monotonic()
        self.counts['input'] += 1
        # Identity delta relative to the calibrated right-controller orientation
        # preserves the measured robot wrist orientation throughout the probe.
        return self.command('input', sample={'source': SOURCE, 'session_id': self.id,
            'client_id': self.id, 'seq': self.seq, 'orientation_mode': 'pose',
            'right': {'valid': True, 'position': [0., 0., -float(forward)],
                      'quaternion_xyzw': [0., 0., 0., 1.]},
            'xr_to_world_rotation': [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]],
            'clutch': bool(active), 'squeeze': float(active), 'trigger': 0., 'axes': [0., 0.],
            'primary': False, 'secondary': False, 'frame_id': state.get('frame_id', -1)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='New directory inside the selected recording')
    args = parser.parse_args(argv)
    recording = args.recording.resolve()
    output = (args.output or recording/('live_arm_probe_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])).resolve()
    if not output.is_relative_to(recording) or output == recording:
        parser.error('--output must be a new directory inside --recording')
    if not recording.is_relative_to(ROOT/'runs') or not recording.is_dir():
        parser.error('--recording must be an existing project runs/ directory')
    output.mkdir(parents=False, exist_ok=False)
    report = {'passed': False, 'synthetic_operator': True, 'headset_verified': False,
              'real_robot_verified': False, 'scope': 'live HTTP arm smoke under learned vision load; no headset latency claim',
              'recording': str(recording), 'input_hz_limit': RATE, 'travel_m': TRAVEL,
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                  for name in ('tools/probe_live_arm_control.py', 'tools/assist_sim.py',
                               'teleop_assist/core.py', 'teleop_assist/isaac_backend.py',
                               'teleop_assist/unitree_arm_ik.py', 'teleop_assist/control_ownership.py')
                  if (ROOT/name).is_file()}, 'failures': [], 'cleanup': {}}
    client = None
    claimed = False
    rows = []
    last_state = None
    origin = None
    try:
        recording, manifest, url, verification = verify_recording(recording)
        report['process_verification'] = verification
        report['live_manifest_source_hashes'] = manifest.get('source_hashes', {})
        report['workload'] = {key: manifest.get(key) for key in
                              ('perception_backend', 'pose_backend', 'arm_solver', 'wrist_cameras', 'lidar', 'model_worker', 'pose_worker')}
        with (output/'requests.jsonl').open('w') as events, (output/'trace.jsonl').open('w') as trace:
            client = Client(url, events)
            report['client_id'] = client.id
            try:
                before = client.state()
                last_state = before
                write_json(output/'initial_state.json', before)
                assert_state(before)
                report['before_frame_id'] = before.get('frame_id')
                report['before_semantic_frames'] = before.get('perception', {}).get('semantic_model_frames', 0)
                # Even a response timeout may follow an accepted claim. Cleanup
                # checks the actual owner before sending any released input.
                claimed = True
                result = client.command('control_claim', deadman_released=True)
                if result.get('control_session', {}).get('owner_client_id') != client.id:
                    raise ValueError('Control claim was not assigned to this probe')
                client.command('assist_enable', enabled=False)
                state = client.state()
                last_state = state
                assert_state(state, client.id)
                if state.get('settings', {}).get('assist_enabled') is not False:
                    raise ValueError('Assist-off setting was not confirmed')
                if abs(float(state.get('settings', {}).get('hand_scale', 0))-1.) > 1e-9:
                    raise ValueError('This 20 mm probe requires hand_scale=1')
                client.input(state, 0., False)
                client.command('calibrate')
                state = client.state()
                last_state = state
                client.input(state, 0., False)
                client.input(state, 0., True)
                state = client.state()
                last_state = state
                robot = assert_state(state, client.id)
                origin = np.asarray(robot['ee_position'], float)
                report['start_position'] = origin.tolist()
                report['start_quaternion'] = robot['ee_quaternion']
                rejections = int(robot.get('command_rejections', 0))
                begun = time.monotonic()
                while True:
                    elapsed = time.monotonic()-begun
                    forward, phase = requested_forward(elapsed)
                    state = client.state()
                    last_state = state
                    robot = assert_state(state, client.id)
                    if state.get('input_source') != SOURCE:
                        raise ValueError('Input source changed during the probe')
                    if state.get('settings', {}).get('assist_enabled') is not False:
                        raise ValueError('Assistance changed during the manual probe')
                    client.input(state, forward, True)
                    position = np.asarray(robot['ee_position'], float)
                    row = {'elapsed_s': elapsed, 'phase': phase, 'requested_forward_m': forward,
                           'ee_position': position.tolist(), 'ee_quaternion': robot['ee_quaternion'],
                           'robot_timestamp': robot['timestamp'], 'valid': robot['valid'],
                           'contact_N': robot['contact_force_max'], 'arm_solver': robot.get('arm_solver'),
                           'command_rejections': robot.get('command_rejections'),
                           'frame_id': state.get('frame_id'), 'input_age_s': state.get('input_age'),
                           'session_reason': state.get('assist', {}).get('reason')}
                    rows.append(row)
                    trace.write(json.dumps(row, allow_nan=False)+'\n'); trace.flush()
                    last_state = state
                    if np.linalg.norm(position-origin) > .035:
                        raise ValueError('Measured displacement exceeded the 35 mm smoke-test guard')
                    if int(robot.get('command_rejections', 0)) > rejections:
                        raise ValueError('Arm solver or backend rejected a live command')
                    if elapsed >= 6.: break
                report['end_position'] = rows[-1]['ee_position']
                report['actual_forward_m'] = max(row['ee_position'][0]-origin[0] for row in rows)
                report['return_error_m'] = float(np.linalg.norm(np.asarray(rows[-1]['ee_position'])-origin))
                report['max_contact_N'] = max(row['contact_N'] for row in rows)
                report['all_robot_states_valid'] = all(row['valid'] for row in rows)
                report['solver_rejection_count'] = max(int(row['command_rejections'] or 0) for row in rows)-rejections
                report['input_late_notices'] = [i for i, row in enumerate(rows)
                    if row['session_reason'] == 'input_stale' or float(row['input_age_s'] or 0) > .30]
                report['session_reasons'] = sorted({row['session_reason'] for row in rows})
                report['after_frame_id'] = state.get('frame_id')
                report['after_semantic_frames'] = state.get('perception', {}).get('semantic_model_frames', 0)
                report['samples'] = len(rows)
                write_json(output/'final_state.json', state)
                if report['actual_forward_m'] <= .015: report['failures'].append('forward_progress_not_over_15mm')
                if report['return_error_m'] >= .005: report['failures'].append('return_error_not_under_5mm')
                if report['input_late_notices']: report['failures'].append('input_expired_under_live_load')
                if report['after_frame_id'] <= report['before_frame_id']: report['failures'].append('camera_processing_did_not_advance')
                if report['after_semantic_frames'] <= report['before_semantic_frames']: report['failures'].append('learned_perception_did_not_advance')
            finally:
                if claimed:
                    try:
                        state = client.state()
                        if state.get('control_session', {}).get('owner_client_id') == client.id:
                            try:
                                client.input(state, 0., False)
                                report['cleanup']['released_input'] = True
                            finally:
                                client.command('control_release')
                                report['cleanup']['released_control'] = True
                        else:
                            report['cleanup']['owner_already_changed_or_expired'] = True
                    except Exception as exc:
                        report['cleanup']['error'] = str(exc)
                        report['failures'].append('cleanup_not_confirmed')
                if client is not None: report['requests'] = dict(client.counts)
    except (Exception, KeyboardInterrupt) as exc:
        report['failures'].append(type(exc).__name__+': '+str(exc))
        report['samples_before_failure'] = len(rows)
        if last_state is not None: write_json(output/'last_state_on_failure.json', last_state)
    if rows and origin is not None:
        # Preserve interpretable partial results as well as every request/trace
        # when a guard aborts before the six-second trajectory finishes.
        report.setdefault('actual_forward_m', max(row['ee_position'][0]-origin[0] for row in rows))
        report.setdefault('return_error_m', float(np.linalg.norm(np.asarray(rows[-1]['ee_position'])-origin)))
        report.setdefault('max_contact_N', max(row['contact_N'] for row in rows))
        report.setdefault('input_late_notices', [i for i, row in enumerate(rows)
            if row['session_reason'] == 'input_stale' or float(row['input_age_s'] or 0) > .30])
    report['passed'] = not report['failures'] and bool(rows)
    write_json(output/'report.json', report)
    print(json.dumps({'report': str(output/'report.json'), 'passed': report['passed'],
                      'actual_forward_m': report.get('actual_forward_m'),
                      'return_error_m': report.get('return_error_m'), 'failures': report['failures']}, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
