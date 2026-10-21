"""CPU-only launcher tests: every process and signal operation is substituted."""
import importlib.util
import hashlib
import json
from pathlib import Path
import signal
import socket
import sys
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1]/'tools/copilot/launch_sim_xr.py'
spec = importlib.util.spec_from_file_location('standalone_sim_xr_launcher', SOURCE)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


@pytest.fixture
def inputs(tmp_path):
    config = tmp_path/'config.json'
    paths = {name: str(tmp_path/name) for name in
             ('isaac_repository', 'xr_repository', 'grasp_repository', 'model_root')}
    for value in paths.values():
        Path(value).mkdir()
    for name in ('isaac_python', 'xr_python', 'perception_python', 'grasp_python'):
        paths[name] = sys.executable
    config.write_text(json.dumps({'paths': paths}))
    cert, key = tmp_path/'cert.pem', tmp_path/'key.pem'
    cert.write_text('test certificate, never parsed or used on network')
    key.write_text('test private key marker, not to be copied')
    return ['--check', '--config', str(config), '--runtime', str(tmp_path/'session'),
            '--cert-file', str(cert), '--key-file', str(key), '--duration', '1',
            '--control-mode', 'guarded_assist']


def test_manual_default_bypasses_entire_guarded_runtime(inputs):
    assert launch.parser().get_default('control_mode') == 'unitree_manual'
    plan=launch.build_plan(launch.parser().parse_args(inputs+['--control-mode','unitree_manual']))
    command=plan['simulation_command']
    assert command[1].endswith('/manual_sim_runtime.py')
    assert '--sim' in command and '--headless' in command
    assert '--sim-sender' not in command and '--assist-sim' not in command
    assert plan['custom_motion_guard_enabled'] is False
    assert plan['hardware_authorized'] is False and not plan['DDS_transport_requested']
    assert command[command.index('--operator-camera-source')+1]=='head'
    ai=launch.build_plan(launch.parser().parse_args(inputs+['--control-mode','unitree_manual','--live-ai']))
    assert '--grasp-assist' in ai['simulation_command']
    assert '--sim-sender' not in ai['simulation_command']
    assert ai['live_ai'] and not ai['hardware_authorized']


def test_exact_scope_defaults_and_optional_live_ai(inputs):
    plan = launch.build_plan(launch.parser().parse_args(inputs))
    sim, source = plan['simulation_command'], plan['source_command']
    assert sim[sim.index('--sim-sender')+1] == 'guarded_original'
    assert sim[sim.index('--sim-command-format')+1] == 'unitree_float32'
    assert source[source.index('--input-mode')+1] == 'controller'
    assert '--vr-buttons' in source and '--sim' in source and '--assist-sim' in sim
    assert '--scripted-test' not in sim and '--live-ai' not in sim
    assert source[source.index('--output')+1] == sim[sim.index('--operator-mailbox')+1]
    assert plan['browser_url'].startswith('https://127.0.0.1:8012/?ws=')
    assert plan['controls_url'] == 'http://127.0.0.1:8770/sim'
    assert not plan['robot_camera_stream_in_XR'] and not plan['stock_XR_entry_executed']
    assert plan['xr_camera_requested'] is False and plan['sim_camera_directory'] is None
    assert '--sim-camera' not in source
    other = launch.build_plan(launch.parser().parse_args(inputs+['--live-ai', '--camera-source', 'right_wrist']))
    assert '--live-ai' in other['simulation_command']
    assert other['simulation_command'][other['simulation_command'].index('--camera-source')+1] == 'right_wrist'


def test_operator_camera_is_independent_of_live_ai_and_perception(inputs):
    manual=launch.build_plan(launch.parser().parse_args(inputs+['--xr-camera']))
    assert '--live-ai' not in manual['simulation_command']
    baseline=launch.build_plan(launch.parser().parse_args(inputs+['--live-ai']))
    plan=launch.build_plan(launch.parser().parse_args(inputs+['--live-ai','--xr-camera','--camera-source','right_wrist']))
    expected=str(Path(plan['runtime'])/'sim/simulation/operator_camera')
    assert plan['sim_camera_directory']==expected and plan['xr_camera_requested'] is True
    assert plan['source_command'][-2:]==['--sim-camera',expected]
    assert plan['source_command'][:-2]==baseline['source_command']
    command=plan['simulation_command']
    assert command[command.index('--camera-source')+1]=='right_wrist'
    assert command[command.index('--operator-camera-source')+1]=='head'
    assert not plan['sim_camera_delivery_verified'] and not plan['synthetic_inputs']
    with pytest.raises(SystemExit):
        launch.parser().parse_args(inputs+['--live-ai','--sim-camera','/tmp/other_session'])


def test_explicit_sim_initialization_is_forwarded_and_fingerprinted(inputs, tmp_path):
    initialization=tmp_path/'symmetric.json'
    initialization.write_text(json.dumps({'accepted':True, 'motion_authorized':False}))
    plan=launch.build_plan(launch.parser().parse_args(inputs+['--sim-initialization',str(initialization)]))
    command=plan['simulation_command']
    assert command[command.index('--sim-initialization')+1]==str(initialization)
    assert plan['sim_initialization_sha256']==hashlib.sha256(initialization.read_bytes()).hexdigest()
    assert plan['hardware_authorized'] is False
    initialization.write_text(json.dumps({'accepted':False}))
    with pytest.raises(ValueError,match='initial condition'):
        launch.build_plan(launch.parser().parse_args(inputs+['--sim-initialization',str(initialization)]))


@pytest.mark.parametrize('extra', [
    ['--duration', 'nan'], ['--duration', '0.5'], ['--duration', '3601'], ['--startup-timeout', 'inf'],
    ['--xr-port', '8770'], ['--dashboard-port', '0'], ['--host', '0.0.0.0'],
    ['--host', '224.0.0.1'], ['--host', 'fe80::1%eth0']])
def test_invalid_plan_rejected_before_start(inputs, extra):
    with pytest.raises(ValueError):
        launch.build_plan(launch.parser().parse_args(inputs+extra))


def test_existing_runtime_and_missing_tls_rejected(inputs):
    args = launch.parser().parse_args(inputs)
    args.runtime.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        launch.build_plan(args)
    args.runtime.rmdir()
    args.key_file.unlink()
    with pytest.raises(ValueError, match='certificate'):
        launch.build_plan(args)


@pytest.mark.parametrize('extra',[[],['--live-ai','--xr-camera']])
def test_check_has_no_processes_network_or_file_writes(inputs, monkeypatch, capsys, extra):
    def forbidden(*_, **__):
        raise AssertionError('No process allowed during check')
    monkeypatch.setattr(launch.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(launch.subprocess, 'run', forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)
    args = launch.parser().parse_args(inputs)
    before = {p: p.read_bytes() for p in (args.cert_file, args.key_file, args.config)}
    assert launch.main(inputs+extra) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['processes_started'] is False and report['network_started'] is False
    assert not args.runtime.exists()
    assert all(p.read_bytes() == data for p, data in before.items())


def test_interpreter_symlink_is_not_dereferenced(inputs, tmp_path):
    args = launch.parser().parse_args(inputs)
    environment_python = tmp_path/'fake_venv/bin/python'
    environment_python.parent.mkdir(parents=True)
    environment_python.symlink_to(sys.executable)
    config = json.loads(args.config.read_text())
    config['paths']['xr_python'] = str(environment_python)
    args.config.write_text(json.dumps(config))
    assert launch.build_plan(args)['source_command'][0] == str(environment_python)


def mock_runtime(monkeypatch, plan, *, fail_source=False, nested_clean=True, forced=False):
    now = [100.]
    monkeypatch.setattr(launch, 'time', SimpleNamespace(monotonic=lambda: now[0], sleep=lambda value: now.__setitem__(0, now[0]+value)))
    handlers = {}
    def install(number, value):
        old = handlers.get(number)
        handlers[number] = value
        return old
    monkeypatch.setattr(launch, 'signal', SimpleNamespace(signal=install, SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM))
    processes, starts, stops = [], [], []
    clock = 'host_monotonic:' + Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    class Process:
        def __init__(self, name):
            self.pid = 1000 + len(processes)
            self.name, self.returncode = name, None
        def poll(self):
            if fail_source and self.name == 'xr_source':
                self.returncode = 3
            return self.returncode
    def popen(command, **kwargs):
        assert kwargs['start_new_session'] is True
        name = 'simulation_launcher' if 'run_copilot.py' in command[1] else 'xr_source'
        process = Process(name)
        processes.append(process)
        starts.append(name)
        if name == 'simulation_launcher':
            status = Path(plan['runtime_status'])
            status.parent.mkdir(parents=True)
            status.write_text(json.dumps(dict(schema='punktnull.sim_runtime.v1', mode='isolated_PhysX',
                timestamp=now[0], clock_id=clock, robot_transport_started=False, scripted_operator=False)))
        else:
            kwargs['stdout'].write('{"schema":"punktnull.xr_input_endpoint.v1"}\n')
            kwargs['stdout'].flush()
        return process
    monkeypatch.setattr(launch, 'subprocess', SimpleNamespace(Popen=popen, STDOUT=-2))
    def stop(children, grace_s, forced_details):
        assert grace_s >= 25.
        for name, process in children:
            stops.append(name)
            if process.returncode is None:
                process.returncode = 0
            if name == 'simulation_launcher':
                (Path(plan['runtime'])/'sim/session.json').write_text(json.dumps({'cleanup_accepted': nested_clean}))
        if forced:
            forced_details[children[0][0]] = [{'pid': children[0][1].pid}]
            return [children[0][0]]
        return []
    monkeypatch.setattr(launch, 'stop_children', stop)
    monkeypatch.setattr(launch, 'group_process_details', lambda _: [])
    return starts, stops


def test_supervision_starts_sim_first_and_stops_input_first(inputs, monkeypatch):
    plan = launch.build_plan(launch.parser().parse_args(inputs))
    starts, stops = mock_runtime(monkeypatch, plan)
    assert launch.run_session(plan) == 0
    assert starts == ['simulation_launcher', 'xr_source']
    assert stops == ['xr_source', 'simulation_launcher']
    report = json.loads((Path(plan['runtime'])/'launcher_session.json').read_text())
    assert report['cleanup_accepted'] and report['launcher_completed']
    assert not report['simulation_acceptance_claimed'] and not report['headset_verified']


def test_camera_session_keeps_request_distinct_from_delivery_and_acceptance(inputs,monkeypatch,capsys):
    plan=launch.build_plan(launch.parser().parse_args(inputs+['--live-ai','--xr-camera']))
    starts,stops=mock_runtime(monkeypatch,plan)
    assert launch.run_session(plan)==0
    assert starts==['simulation_launcher','xr_source'] and stops==['xr_source','simulation_launcher']
    report=json.loads((Path(plan['runtime'])/'launcher_session.json').read_text())
    assert report['xr_camera_requested'] is True and report['sim_camera_directory']==plan['sim_camera_directory']
    assert not report['sim_camera_delivery_verified'] and not report['headset_verified']
    assert not report['simulation_acceptance_claimed'] and not report['synthetic_inputs']
    endpoint=json.loads(capsys.readouterr().out.splitlines()[0])
    assert endpoint['xr_camera_requested'] is True and not endpoint['sim_camera_delivery_verified']


@pytest.mark.parametrize('options', [{'fail_source': True}, {'nested_clean': False}, {'forced': True}])
def test_process_failure_or_unconfirmed_nested_cleanup_never_passes(inputs, monkeypatch, options):
    plan = launch.build_plan(launch.parser().parse_args(inputs))
    _, stops = mock_runtime(monkeypatch, plan, **options)
    assert launch.run_session(plan) == 1
    assert stops == ['xr_source', 'simulation_launcher']
    report = json.loads((Path(plan['runtime'])/'launcher_session.json').read_text())
    assert report['cleanup_accepted'] is False


def test_no_mode_no_launch(inputs):
    with pytest.raises(SystemExit):
        launch.parser().parse_args(inputs[1:])
