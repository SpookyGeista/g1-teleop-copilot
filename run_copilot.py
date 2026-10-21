#!/usr/bin/env python3
"""One entry point for the isolated G1 grasp copilot; no hardware mode."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def configuration(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get('paths'), dict):
        raise ValueError('Config must contain a paths object')
    paths = value['paths']
    if not paths or any(not isinstance(entry, str) or not entry.strip() for entry in paths.values()):
        raise ValueError('Config paths must be nonempty strings')
    # A venv's bin/python commonly symlinks to its base interpreter. Executing
    # the resolved target bypasses pyvenv.cfg and loses the installed models.
    # Normalize lexical paths without dereferencing executable symlinks.
    return value, {key: os.path.abspath(ROOT/Path(entry).expanduser()) for key, entry in paths.items()}


def group_process_details(pgid):
    """Only live members of our new-session group; never read argv or environ."""
    members=[]
    if not Path('/proc').is_dir():return members
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            stat=(entry/'stat').read_text()
            fields=stat.rsplit(')',1)[1].split()
            if int(fields[2])==pgid and int(fields[3])==pgid and fields[0] not in ('Z','X'):
                members.append(dict(pid=int(entry.name),ppid=int(fields[1]),pgid=int(fields[2]),
                    session=int(fields[3]),comm=stat[stat.index('(')+1:stat.rindex(')')]))
        except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError,IndexError): pass
    return sorted(members,key=lambda process:process['pid'])


def group_running(pgid):
    """Linux process groups can retain unreaped zombies, which cannot clean up."""
    if Path('/proc').is_dir():return bool(group_process_details(pgid))
    try: os.killpg(pgid,0);return True
    except ProcessLookupError:return False


def stop_children(children, grace_s=15., *, forced_details=None):
    """Terminate only process groups created by this launcher, including workers."""
    started = time.monotonic()
    deadline = started+grace_s
    parent_deadline = deadline-min(2.,max(0.,grace_s/3.))
    group_terminated = set()
    for _, child in children:
        # Let the owner finish its current operation and close its worker pipes.
        # Group-wide SIGTERM here also kills an in-flight IK/model worker before
        # the owner can serialize its normal stop/audit path.
        try:
            if child.poll() is None:child.send_signal(signal.SIGTERM)
        except ProcessLookupError: pass
    # A departed/uncooperative owner cannot coordinate remaining workers. Give
    # that exact group SIGTERM, preserving a bounded grace before forced kill.
    # The original total deadline applies to all groups, including descendants.
    while True:
        running = False
        for _,child in children:
            code=child.poll()
            alive=group_running(child.pid)
            running=running or alive
            if alive and child.pid not in group_terminated and (code is not None or time.monotonic()>=parent_deadline):
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                group_terminated.add(child.pid)
        if not running: break
        remaining=deadline-time.monotonic()
        if remaining<=0:break
        time.sleep(min(.05,remaining))
    killed = []
    cleanup_errors = []
    for name, child in children:
        # A child may exit while its inference worker ignores SIGTERM. The
        # original process group remains ours until all descendants have left.
        try:
            if group_running(child.pid):
                details=group_process_details(child.pid)
                os.killpg(child.pid, signal.SIGKILL)
                killed.append(name)
                if forced_details is not None:forced_details[name]=details
        except ProcessLookupError:
            pass
        except OSError as error:
            cleanup_errors.append(error)
    # Do not let a late wait/kill failure for one child prevent terminating the
    # other groups we own. Report failure only after attempting all cleanups.
    for _,child in children:
        try:child.wait(timeout=5)
        except (OSError,subprocess.TimeoutExpired) as error:cleanup_errors.append(error)
    if cleanup_errors:
        raise cleanup_errors[0]
    return killed


def scripted_completion(path):
    """Minimum report-based completion check, not a motion acceptance verdict."""
    try:
        with Path(path).open('rb') as stream:raw=stream.read(2*1024*1024+1)
        if len(raw)>2*1024*1024:return False,'report_too_large'
        report=json.loads(raw)
        if not isinstance(report,dict):return False,'report_invalid'
    except (OSError,ValueError,TypeError):return False,'report_missing_or_invalid'
    if report.get('completed') is not True:return False,'runtime_not_completed'
    if report.get('scripted_scenario_completed') is not True:return False,'scenario_not_completed'
    commands=report.get('commands_consumed')
    if type(commands) is not int or commands<=0:return False,'no_commands_consumed'
    return True,'sequence_completed_commands_observed_not_full_acceptance'


def scripted_acceptance(python, runtime):
    """Run the recorded-data checker after shutdown; process exit alone is insufficient."""
    output = runtime/'phase_acceptance.json'
    try:
        with (runtime/'phase_acceptance.log').open('w') as log:
            checked = subprocess.run([python, str(ROOT/'tools/copilot/check_sim_run.py'),
                str(runtime/'simulation'), '--require-measured-poses', '--output', str(output)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=60)
        with output.open('rb') as stream:
            raw = stream.read(2*1024*1024+1)
        if len(raw)>2*1024*1024:
            return False, 'acceptance_report_too_large'
        evidence=json.loads(raw)
        if checked.returncode != 0 or not isinstance(evidence,dict) or evidence.get('accepted') is not True:
            return False, 'recorded_phase_acceptance_failed'
        return True, 'recorded_phase_acceptance_passed_no_hardware_claim'
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        return False, 'acceptance_checker_failed_or_report_invalid'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--demo', action='store_true', help='Original G1 USD, live learned preview, no XR/DDS')
    actions.add_argument('--review', type=Path, help='Review an analyze_capture result, without loading models')
    actions.add_argument('--doctor', action='store_true', help='Read-only installation/provenance check')
    actions.add_argument('--xr-check', action='store_true', help='Check staged original XR code, without starting DDS')
    actions.add_argument('--assist-sim', action='store_true', help='Explicit isolated PhysX control with checked direct q14 handoff')
    parser.add_argument('--live-ai',action='store_true',help='Opt-in live learned proposals for --assist-sim; otherwise use the reproducible fixed fixture')
    parser.add_argument('--camera-source',choices=('head','right_wrist','left_wrist'),
                        help='Original camera preset for --assist-sim; default head')
    parser.add_argument('--operator-camera-source',choices=('head','right_wrist','left_wrist'),
                        help='Independent read-only VR image for --assist-sim')
    parser.add_argument('--sim-initialization',type=Path,
                        help='Explicit isolated-simulation initial condition JSON; no robot approach')
    parser.add_argument('--ticks',type=int,help='130..1000 scenario ticks, only with --scripted-test; default 145')
    parser.add_argument('--sim-command-format',choices=('float64','unitree_float32'),
                        help='Isolated simulation: direct default or checked Float32 plus local Unitree IDL round trip; no DDS transport')
    parser.add_argument('--sim-sender',choices=('direct','guarded_original'),
                        help='Isolated simulation receiver; guarded_original requires unitree_float32 and starts no DDS')
    operator_source=parser.add_mutually_exclusive_group()
    operator_source.add_argument('--operator-mailbox',type=Path,help='Explicit xr_source operator.json for --assist-sim')
    operator_source.add_argument('--scripted-test',action='store_true',help='Explicit synthetic-input integration test; no headset claim')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/copilot.local.json')
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--duration', type=float, default=300.)
    parser.add_argument('--prompt', default='red cylinder')
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args(argv)
    if not 0 < args.duration <= 3600 or not 1 <= args.port <= 65535:
        parser.error('Duration must be greater than 0 and at most 3600 seconds; port must be 1..65535')
    if args.assist_sim and not (args.operator_mailbox or args.scripted_test):
        parser.error('--assist-sim requires --operator-mailbox PATH or explicit --scripted-test')
    if not args.assist_sim and (args.operator_mailbox or args.scripted_test):
        parser.error('Operator sources are only allowed with --assist-sim')
    if args.live_ai and not args.assist_sim:
        parser.error('--live-ai is only allowed with --assist-sim')
    if args.camera_source is not None and not args.assist_sim:
        parser.error('--camera-source is only allowed with --assist-sim')
    if args.operator_camera_source is not None and not args.assist_sim:
        parser.error('--operator-camera-source is only allowed with --assist-sim')
    if args.sim_initialization is not None and not args.assist_sim:
        parser.error('--sim-initialization is only allowed with --assist-sim')
    if args.sim_command_format is not None and not args.assist_sim:
        parser.error('--sim-command-format is only allowed with --assist-sim')
    if args.sim_sender is not None and not args.assist_sim:
        parser.error('--sim-sender is only allowed with --assist-sim')
    if args.sim_sender=='guarded_original' and args.sim_command_format!='unitree_float32':
        parser.error('--sim-sender guarded_original requires --sim-command-format unitree_float32')
    if args.ticks is not None and (not args.scripted_test or not 130<=args.ticks<=1000):
        parser.error('--ticks requires --scripted-test and must be 130..1000')
    try:
        config, paths = configuration(args.config.resolve())
        required = {'isaac_python'}
        if args.demo: required.update(('xr_repository','xr_python','perception_python','grasp_python',
                                      'grasp_repository','model_root','isaac_repository'))
        if args.xr_check: required.update(('xr_repository','xr_python'))
        if args.assist_sim: required.update(('xr_repository','xr_python','isaac_repository'))
        if args.live_ai: required.update(('perception_python','grasp_python','grasp_repository','model_root'))
        if required-paths.keys(): raise ValueError('Missing config paths: '+', '.join(sorted(required-paths.keys())))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(f'Invalid config: {exc}')
    python = paths['isaac_python']
    if args.doctor:
        return subprocess.call([python, str(ROOT/'tools/copilot/doctor.py'), '--config', str(args.config.resolve())], cwd=ROOT)
    if args.xr_check:
        return subprocess.call([python, str(ROOT/'tools/copilot/launch_xr.py'), '--sim', '--check',
            '--assist-config', str(args.config.resolve()), '--repo', paths['xr_repository'],
            '--ik-python', paths['xr_python']], cwd=ROOT)
    batch = status = None
    if args.review:
        source = args.review.resolve()
        try:
            for name in ('latest_proposals.json', 'ui_state.json', 'frame.jpg'):
                if not (source/name).is_file(): raise ValueError(f'Review artifact missing: {source/name}')
            # Live motion evidence is published before expensive UI geometry.
            # Review the separately bound visual packet when available; the
            # same invalidation below applies to both old and new recordings.
            visual_path=source/'preview_proposals.json'
            batch = json.loads((visual_path if visual_path.is_file() else source/'latest_proposals.json').read_text())
            status = json.loads((source/'ui_state.json').read_text())
            if not isinstance(batch, dict) or not isinstance(status, dict) or not isinstance(batch.get('candidates'), list):
                raise ValueError('Review requires proposal/state objects and a candidates list')
            batch.update(valid=False, motion_static=False, recording_only=True, runtime_motion_guard_available=False)
            status.update(mode='review', status='preview')
            # Reject nonfinite values before creating any output directory.
            json.dumps(batch, allow_nan=False); json.dumps(status, allow_nan=False)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
    if args.assist_sim:
        assets=ROOT/'teleop_assist/copilot/assets/demo_cylinder'
        for name in ('candidate.json','initialization.json','provenance.json'):
            if not (assets/name).is_file(): parser.error(f'Missing isolated simulation asset: {name}')
        initialization = args.sim_initialization.resolve() if args.sim_initialization else assets/'initialization.json'
        if not initialization.is_file(): parser.error('Simulation initial condition file is missing')
        # Avoid starting Isaac for an already occupied dashboard port. The
        # runtime still owns the authoritative bind (and rejects later races).
        try:
            with socket.socket() as check_port:check_port.bind(('127.0.0.1',args.port))
        except OSError as exc:parser.error(f'Loopback control port unavailable: {exc}')
    runtime = (args.runtime or ROOT/'runs'/('copilot_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if runtime.exists(): parser.error('Runtime directory already exists; choose a new directory')
    runtime.mkdir(parents=True)
    ui = runtime/'dashboard'
    if not args.assist_sim:ui.mkdir()
    serve = [python, str(ROOT/'tools/copilot/serve.py'), '--port', str(args.port), '--duration', str(args.duration)]
    if args.assist_sim:
        serve=[python,str(ROOT/'tools/copilot/sim_runtime.py'),'--sim',
            '--candidate',str(assets/'candidate.json'),'--initialization',str(initialization),
            '--output',str(runtime/'simulation'),'--port',str(args.port),'--duration',str(args.duration),
            '--repo',paths['isaac_repository'],'--xr-repo',paths['xr_repository'],'--ik-python',paths['xr_python']]
        serve+=['--sender',args.sim_sender or 'direct']
        if args.operator_mailbox:serve+=['--operator-mailbox',str(args.operator_mailbox.resolve())]
        else:serve+=['--scripted-test','--ticks',str(args.ticks if args.ticks is not None else 145)]
        serve+=['--camera-source',args.camera_source or 'head']
        if args.operator_camera_source is not None:
            serve+=['--operator-camera-source',args.operator_camera_source]
        serve+=['--command-format',args.sim_command_format or 'float64']
        if not args.gui:serve+=['--headless']
        if args.live_ai:serve+=['--live-ai-frames',str(runtime/'frames')]
    elif args.review:
        shutil.copy2(source/'frame.jpg', ui/'frame.jpg')
        # Never turn an old recording into current input, even if a caller has
        # copied a batch from a previously active simulator session.
        (ui/'latest_proposals.json').write_text(json.dumps(batch, allow_nan=False))
        (ui/'ui_state.json').write_text(json.dumps(status, allow_nan=False))
        serve += ['--review', str(ui)]
    else:
        serve += ['--live-frames', str(runtime/'frames'), '--runtime', str(ui), '--prompt', args.prompt,
                  '--xr-repo', paths['xr_repository'], '--ik-python', paths['xr_python'],
                  '--model-python', paths['perception_python'], '--grasp-python', paths['grasp_python'],
                  '--grasp-repo', paths['grasp_repository'], '--model-root', paths['model_root']]
    logs=[]; children=[]; stop=False; stop_reason=None
    def request_stop(*_):
        nonlocal stop,stop_reason
        stop=True;stop_reason='signal'
    previous_signals = {number:signal.signal(number,request_stop) for number in (signal.SIGINT,signal.SIGTERM)}
    def start(command,name):
        stream=(runtime/(name+'.log')).open('w');logs.append(stream)
        environment=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',
                         OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
        if paths.get('isaac_repository'):
            environment['PUNKTNULL_ISAAC_REPOSITORY']=paths['isaac_repository']
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                 env=environment,start_new_session=True)
        children.append((name,process));return process
    started=time.monotonic();deadline=started+args.duration;exitcode=0;ready=False;forced=[];forced_details={}
    primary_name='simulation' if args.assist_sim else 'service'
    try:
        server=start(serve,primary_name)
        # serve.py emits this marker only after its loopback bind succeeds,
        # before model initialization. A port collision must never start Isaac.
        marker=(f'Isolated PhysX controls: http://127.0.0.1:{args.port}/sim' if args.assist_sim
                else f'Copilot dashboard: http://127.0.0.1:{args.port}')
        while not stop and server.poll() is None and time.monotonic()<deadline:
            log_path=runtime/(primary_name+'.log')
            with log_path.open('rb') as stream:
                stream.seek(max(0,log_path.stat().st_size-65536))
                ready=marker.encode() in stream.read(65536)
            if ready: break
            time.sleep(.05)
        if not ready:
            if not stop:
                stop_reason='duration_expired' if time.monotonic()>=deadline else 'service_start_failed'
                exitcode=1
            stop=True
        if args.demo and not stop:
            command=[python,str(ROOT/'tools/copilot/capture_unitree_scene.py'),'--repo',paths['isaac_repository'],
                     '--stream-seconds',str(args.duration),'--output',str(runtime/'frames')]
            if not args.gui:command.append('--headless')
            start(command,'simulator')
        if args.live_ai and ready and not stop and server.poll() is None:
            command=[python,str(ROOT/'tools/copilot/serve.py'),'--no-dashboard',
                '--complete-static-sim-preview',
                '--live-frames',str(runtime/'frames'),'--runtime',str(runtime/'simulation'),
                '--duration',str(args.duration),'--prompt',args.prompt,
                '--xr-repo',paths['xr_repository'],'--ik-python',paths['xr_python'],
                '--model-python',paths['perception_python'],'--grasp-python',paths['grasp_python'],
                '--grasp-repo',paths['grasp_repository'],'--model-root',paths['model_root']]
            start(command,'proposal_service')
        if ready and not stop:
            page='/sim' if args.assist_sim else ''
            label=('Isolierte PhysX-Steuerung · '+('synthetischer Test, kein Headset' if args.scripted_test else 'explizite XR-Eingabedatei')) if args.assist_sim else 'Vorschau, keine Roboterverbindung'
            if args.live_ai:label+=' · Live-KI-Vorschläge'
            print(f'G1 Copilot: http://127.0.0.1:{args.port}{page}\nRuntime: {runtime}\nModus: {label}',flush=True)
        while not stop and server.poll() is None:
            if time.monotonic()>=deadline:
                stop=True;stop_reason='duration_expired';break
            for name,child in children:
                stopped=child.poll() is not None
                unexpected_worker_end=(name=='proposal_service' and stopped and server.poll() is None)
                if stopped and (child.returncode != 0 or unexpected_worker_end):
                    print(f'{name} stopped with code {child.returncode}; see {runtime/(name+".log")}',file=sys.stderr)
                    exitcode=1;stop=True;stop_reason=name+'_failed'
            time.sleep(.1)
        if server.poll() not in (None,0):
            exitcode=1
            stop_reason=stop_reason or 'service_failed'
        # A worker may fail in the same interval in which the primary finishes;
        # still record that failure before initiating our own group shutdown.
        for name,child in children:
            if child is not server and child.poll() not in (None,0):
                exitcode=1;stop_reason=stop_reason or name+'_failed'
    except OSError as exc:
        print(f'Copilot could not start: {exc}',file=sys.stderr)
        exitcode=1;stop=True;stop_reason='process_start_failed'
    finally:
        before_cleanup={name:child.poll() for name,child in children}
        cleanup_error=None
        try:
            forced=stop_children(children,forced_details=forced_details)
        except (OSError,subprocess.TimeoutExpired) as exc:
            cleanup_error=type(exc).__name__
        for stream in logs:stream.close()
        process_stop_reason=stop_reason or 'service_completed'
        cleanup_failures={}
        for name,child in children:
            code=child.poll()
            # SIGTERM is expected only for a still-running owned process we
            # just stopped. A scripted primary must finish its own audit and
            # cleanly return zero; auxiliary workers may use default SIGTERM.
            expected_term=(code == -signal.SIGTERM and before_cleanup[name] is None
                and not (args.scripted_test and name == primary_name))
            if code != 0 and not expected_term:cleanup_failures[name]=code
        cleanup_accepted=not forced and not cleanup_failures and cleanup_error is None
        if not cleanup_accepted:
            exitcode=1;stop_reason=stop_reason or 'process_cleanup_failed'
        sequence_completed=completion_reason=phase_accepted=acceptance_reason=None
        if args.scripted_test:
            sequence_completed,completion_reason=scripted_completion(runtime/'simulation/report.json')
            if not sequence_completed:
                exitcode=1;stop_reason='test_incomplete'
                print('Synthetischer Ablauf unvollständig; siehe simulation/report.json und session.json.',file=sys.stderr)
            else:
                phase_accepted,acceptance_reason=scripted_acceptance(python,runtime)
                if not phase_accepted:
                    exitcode=1;stop_reason='test_acceptance_failed'
                    print('Simulationsabnahme fehlgeschlagen; siehe phase_acceptance.json und phase_acceptance.log.',file=sys.stderr)
        mode=('scripted_assist_sim' if args.scripted_test else 'xr_assist_sim') if args.assist_sim else ('demo_preview' if args.demo else 'recorded_review')
        report={'mode':mode,'wall_s':time.monotonic()-started,
            'live_ai':args.live_ai,
            'camera_source':(args.camera_source or 'head') if args.assist_sim else None,
            'operator_camera_source':args.operator_camera_source if args.assist_sim else None,
            'scripted_ticks':(args.ticks if args.ticks is not None else 145) if args.scripted_test else None,
            'scripted_sequence_completed':sequence_completed,'scripted_completion_reason':completion_reason,
            'scripted_phase_accepted':phase_accepted,'scripted_acceptance_reason':acceptance_reason,
            'processes':{name:child.returncode for name,child in children},'real_robot_connected':False,
            'runtime':str(runtime),'requested_stop':process_stop_reason=='signal',
            'stop_reason':stop_reason or process_stop_reason,'process_stop_reason':process_stop_reason,
            'service_ready':ready,'forced_process_groups':forced,'forced_process_details':forced_details}
        report.update(cleanup_accepted=cleanup_accepted,cleanup_failures=cleanup_failures,
            cleanup_error=cleanup_error,processes_before_cleanup=before_cleanup)
        (runtime/'session.json').write_text(json.dumps(report,indent=2)+'\n')
        for number,previous in previous_signals.items():signal.signal(number,previous)
    return exitcode


if __name__=='__main__':
    raise SystemExit(main())
