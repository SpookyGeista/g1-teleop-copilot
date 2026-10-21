"""Independent cleanup regressions: no subprocesses, sockets, DDS or Isaac.

The launcher executes its actual main/finally control flow against explicit
process/checker fixtures. These fixtures do not establish physics acceptance.
"""
import ast
import json
from pathlib import Path
import signal
import subprocess
import threading

import numpy as np
import pytest

import run_copilot as runner
from teleop_assist.copilot.mailbox import atomic_json, read_json
from teleop_assist.copilot.service import ProposalSession


def run_cleanup_fixture(tmp_path, monkeypatch, *, final_codes=None,
                        forced=(), cleanup_exception=None, live_ai=False,
                        primary_running_at_cleanup=False, auxiliary_prior_exit=None):
    runtime = tmp_path/'runtime'
    state = {'cleaning': False, 'calls': 0}
    final_codes = final_codes or {'simulation': 0}
    handlers = {signal.SIGINT: object(), signal.SIGTERM: object()}
    original_handlers = dict(handlers)

    def set_signal(number, handler):
        previous = handlers[number]
        handlers[number] = handler
        return previous

    class NoSocket:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def bind(self, address):
            assert address == ('127.0.0.1', 8789)

    class ProcessFixture:
        def __init__(self, command, *, stdout, **kwargs):
            assert kwargs['start_new_session'] is True
            self.name = 'simulation' if Path(command[1]).name == 'sim_runtime.py' else 'proposal_service'
            self.pid = 900000000 + len(created)
            self.returncode = None
            self.polls = 0
            created.append(self)
            if self.name == 'simulation':
                stdout.write('Isolated PhysX controls: http://127.0.0.1:8789/sim\n')
                stdout.flush()
                (runtime/'simulation').mkdir()
                atomic_json(runtime/'simulation/report.json', dict(completed=True,
                    scripted_scenario_completed=True, commands_consumed=1,
                    test_fixture='NO_PHYSICS_CHECKER_ACCEPTANCE_STUB'))

        def poll(self):
            if state['cleaning']:
                return self.returncode
            self.polls += 1
            # Keep the primary running long enough to start its auxiliary,
            # then emulate its normal completion before group cleanup.
            if (self.name == 'simulation' and not primary_running_at_cleanup
                    and self.polls > (2 if live_ai else 1)):
                self.returncode = 0
            if self.name == 'proposal_service' and auxiliary_prior_exit is not None:
                self.returncode = auxiliary_prior_exit
            return self.returncode

    def cleanup(children, *, forced_details):
        state['cleaning'] = True
        state['calls'] += 1
        for name, process in children:
            process.returncode = final_codes.get(name, 0)
            if name in forced:
                forced_details[name] = [dict(pid=process.pid, ppid=1,
                    pgid=process.pid, session=process.pid, comm='test_fixture')]
        if cleanup_exception is not None:
            raise cleanup_exception
        return list(forced)

    paths = {key: '/unused/test_fixture' for key in (
        'isaac_python', 'xr_repository', 'xr_python', 'isaac_repository',
        'perception_python', 'grasp_python', 'grasp_repository', 'model_root')}
    created = []
    monkeypatch.setattr(runner, 'configuration', lambda _: ({}, paths))
    monkeypatch.setattr(runner.socket, 'socket', NoSocket)
    monkeypatch.setattr(runner.signal, 'signal', set_signal)
    monkeypatch.setattr(runner.subprocess, 'Popen', ProcessFixture)
    monkeypatch.setattr(runner, 'stop_children', cleanup)
    monkeypatch.setattr(runner, 'scripted_acceptance',
        lambda *_: (True, 'TEST_FIXTURE_ACCEPTED_NO_PHYSICS'))
    if primary_running_at_cleanup:
        elapsed = [0.]
        def monotonic():
            elapsed[0] += .1
            return elapsed[0]
        monkeypatch.setattr(runner.time, 'monotonic', monotonic)
        monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    arguments = ['--assist-sim', '--scripted-test', '--runtime', str(runtime),
                 '--port', '8789', '--duration', '1']
    if live_ai:
        arguments.append('--live-ai')
    exitcode = runner.main(arguments)
    report = json.loads((runtime/'session.json').read_text())
    assert handlers == original_handlers
    assert state['calls'] == 1
    assert report['scripted_sequence_completed'] is True
    assert report['scripted_phase_accepted'] is True
    assert not report['real_robot_connected']
    return exitcode, report


@pytest.mark.parametrize('final_code', [17, -signal.SIGTERM])
def test_scripted_primary_late_failure_cannot_be_hidden_by_valid_phase_report(
        tmp_path, monkeypatch, final_code):
    code, report = run_cleanup_fixture(tmp_path, monkeypatch,
        final_codes={'simulation': final_code})
    assert code == 1
    assert not report['cleanup_accepted']
    assert report['cleanup_failures'] == {'simulation': final_code}
    assert report['stop_reason'] == 'process_cleanup_failed'


@pytest.mark.parametrize('group', ['simulation', 'proposal_service'])
def test_forced_owned_group_fails_scripted_run_even_if_primary_completed(
        tmp_path, monkeypatch, group):
    code, report = run_cleanup_fixture(tmp_path, monkeypatch,
        live_ai=group == 'proposal_service', forced=(group,),
        final_codes={'simulation': 0, 'proposal_service': 0})
    assert code == 1
    assert not report['cleanup_accepted']
    assert report['forced_process_groups'] == [group]
    assert report['forced_process_details'][group][0]['comm'] == 'test_fixture'


def test_expected_running_auxiliary_sigterm_is_not_a_cleanup_failure(tmp_path, monkeypatch):
    code, report = run_cleanup_fixture(tmp_path, monkeypatch, live_ai=True,
        final_codes={'simulation': 0, 'proposal_service': -signal.SIGTERM})
    assert code == 0
    assert report['processes_before_cleanup']['proposal_service'] is None
    assert report['processes']['proposal_service'] == -signal.SIGTERM
    assert report['cleanup_accepted'] is True
    assert report['cleanup_failures'] == {}
    assert report['stop_reason'] == 'service_completed'


def test_sigterm_before_our_cleanup_is_an_auxiliary_failure(tmp_path, monkeypatch):
    code, report = run_cleanup_fixture(tmp_path, monkeypatch, live_ai=True,
        auxiliary_prior_exit=-signal.SIGTERM,
        final_codes={'simulation': 0, 'proposal_service': -signal.SIGTERM})
    assert code == 1
    assert report['processes_before_cleanup']['proposal_service'] == -signal.SIGTERM
    assert report['cleanup_failures'] == {'proposal_service': -signal.SIGTERM}


def test_running_scripted_primary_sigterm_is_not_expected_success(tmp_path, monkeypatch):
    code, report = run_cleanup_fixture(tmp_path, monkeypatch,
        primary_running_at_cleanup=True, final_codes={'simulation': -signal.SIGTERM})
    assert code == 1
    assert report['processes_before_cleanup']['simulation'] is None
    assert report['cleanup_failures'] == {'simulation': -signal.SIGTERM}
    assert report['cleanup_accepted'] is False


def test_unreaped_primary_after_cleanup_is_reported_as_failure(tmp_path, monkeypatch):
    code, report = run_cleanup_fixture(tmp_path, monkeypatch,
        final_codes={'simulation': None},
        cleanup_exception=subprocess.TimeoutExpired('owned fixture', timeout=5))
    assert code == 1
    assert report['cleanup_failures'] == {'simulation': None}
    assert report['cleanup_error'] == 'TimeoutExpired'


@pytest.mark.parametrize('failure', [OSError('fixture cleanup failure'),
    subprocess.TimeoutExpired('owned fixture', timeout=5)])
def test_cleanup_exception_is_persisted_and_signal_handlers_restored(
        tmp_path, monkeypatch, failure):
    code, report = run_cleanup_fixture(tmp_path, monkeypatch,
        cleanup_exception=failure)
    assert code == 1
    assert report['cleanup_error'] == type(failure).__name__
    assert report['cleanup_accepted'] is False
    assert report['stop_reason'] == 'process_cleanup_failed'


@pytest.mark.parametrize('original_audit_fails', [False, True])
def test_reselection_does_not_inherit_pending_or_failed_first_archive(
        tmp_path, monkeypatch, original_audit_fails):
    from teleop_assist.copilot import service as module
    started, finish = threading.Event(), threading.Event()
    original = module.write_failed_observation

    def writer(*args):
        started.set()
        assert finish.wait(2)
        if original_audit_fails:
            raise OSError('fixture first archive unavailable')
        return original(*args)

    class Engine:
        def close(self):
            pass

    def frame(number, stamp):
        return dict(frame_id=number, timestamp=stamp, clock_id='clock', calibration_id='cal',
            rgb=np.zeros((8, 8, 3), np.uint8), depth=np.ones((8, 8)),
            intrinsics=np.eye(3), T_world_camera=np.eye(4), T_world_base=np.eye(4))

    monkeypatch.setattr(module, 'write_failed_observation', writer)
    clock = [10.]
    session = ProposalSession(tmp_path, Engine(), clock=lambda: clock[0], clock_id='clock')
    try:
        atomic_json(tmp_path/'target_selection.json', dict(target_id='object', revision=1))
        session.update(frame(1, clock[0]), [], {})
        assert started.wait(1)
        session.publish()
        assert read_json(tmp_path/'ui_state.json')['first_failure']['replay_status'] == 'pending'
        clock[0] = 10.1
        atomic_json(tmp_path/'target_selection.json', dict(target_id='object', revision=2))
        session.update(frame(2, clock[0]), [], {})
        future = session.failure_audit
        finish.set()
        if original_audit_fails:
            with pytest.raises(OSError):
                future.result(2)
        else:
            future.result(2)
        session.publish()
        failure = read_json(tmp_path/'ui_state.json')['first_failure']
        assert failure['frame_id'] == 2
        assert failure['replay_status'] == 'not_captured_previous_failure_retained'
        assert 'replay_directory' not in failure and 'replay_error' not in failure
        assert session.failure_audit is future
        batch = read_json(tmp_path/'latest_proposals.json')
        assert batch['valid'] is False and batch['motion_static'] is False
        if not original_audit_fails:
            assert read_json(tmp_path/'first_failure/evidence.json')['frame_id'] == 1
            assert len(list(tmp_path.glob('first_failure*'))) == 1
    finally:
        finish.set()
        session.close()


def test_pure_audit_import_precedes_isaac_application_creation():
    source = runner.ROOT/'tools/copilot/sim_runtime.py'
    tree = ast.parse(source.read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and node.module == 'teleop_assist.copilot.audit_trace']
    creates = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name) and node.func.id == 'AppLauncher']
    assert len(imports) == len(creates) == 1
    assert imports[0].lineno < creates[0].lineno
