"""Adversarial pipe tests with real CPU subprocesses; no models or GPUs loaded."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pytest

from teleop_assist.copilot.observations import GeometryObservation


READY = "import json,os,sys,time\nprint(json.dumps({'status':'ready','manifest':{'fake_test_worker':True}}),flush=True)\n"


def observation():
    return GeometryObservation('object', 'obs', 12., 'test', 'cal', 'synthetic_transport_test',
                               np.ones((12, 12)), np.ones((12, 12), bool),
                               np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1.]]), np.eye(4), np.eye(4))


@pytest.fixture
def fake_process(monkeypatch):
    # Deliberately keep fake executable test files inside this workspace.
    directory = tempfile.TemporaryDirectory(prefix='ipc_fake_', dir=Path(__file__).parent)
    real_popen = subprocess.Popen
    children = []

    @contextmanager
    def make(kind, code, *, timeout=.12, startup_timeout=1., command_format='float64'):
        path = Path(directory.name) / 'fake_worker.py'
        path.write_text(code)
        def launch(command, **kwargs):
            child = real_popen([sys.executable, '-u', str(path)], **kwargs)
            children.append(child)
            return child
        module = importlib.import_module('teleop_assist.copilot.motion_process' if kind == 'motion'
                                         else 'teleop_assist.copilot.process')
        monkeypatch.setattr(module.subprocess, 'Popen', launch)
        engine = None
        try:
            if kind == 'motion':
                engine = module.MotionProcess(sys.executable, '/unused/fake', timeout=timeout,
                                              startup_timeout=startup_timeout,command_format=command_format)
            else:
                engine = module.GraspProcess(sys.executable, timeout=timeout, startup_timeout=startup_timeout)
            yield engine, module
        finally:
            if engine is not None:
                engine.close()
    yield make
    for child in children:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=1)
    directory.cleanup()


def invoke(kind, engine):
    return engine.call('validate', fixture='test') if kind == 'motion' else engine.infer(observation())


def test_requested_wire_format_reaches_worker_and_cannot_be_overridden(fake_process):
    code = READY + """
for line in sys.stdin:
    request=json.loads(line)
    print(json.dumps({'status':'ok','request_id':request['request_id'],
        'result':{'allowed':True,'command_format':request['command_format']}}),flush=True)
"""
    with fake_process('motion',code,timeout=1.,command_format='unitree_float32') as (engine,_):
        assert engine.manifest['requested_command_format']=='unitree_float32'
        assert engine.call('validate')['command_format']=='unitree_float32'
        with pytest.raises(ValueError,match='cannot override'):
            engine.call('validate',command_format='float64')


def test_worker_cannot_silently_downgrade_the_requested_wire_format(fake_process):
    code = READY + """
request=json.loads(sys.stdin.readline())
print(json.dumps({'status':'ok','request_id':request['request_id'],
    'result':{'allowed':True,'command_format':'float64'}}),flush=True)
"""
    with fake_process('motion',code,timeout=1.,command_format='unitree_float32') as (engine,_):
        with pytest.raises(RuntimeError,match='different command format'):engine.call('validate')
        assert engine.closed


def test_completed_motion_request_retains_exact_sent_bytes_then_clears(fake_process):
    import hashlib
    code = READY + """
import hashlib
for line in sys.stdin.buffer:
    request=json.loads(line)
    print(json.dumps({'status':'ok','request_id':request['request_id'],
        'result':{'allowed':False,'received_sha256':hashlib.sha256(line).hexdigest()}}),flush=True)
"""
    with fake_process('motion',code,timeout=1.) as (engine,_):
        payload={'acquisition_timestamp':12.25,'joint_positions':{'arm':.1}}
        result=engine.call('validate',robot_state=payload,now=12.3)
        raw=engine.last_call_request_bytes
        assert isinstance(raw,bytes) and hashlib.sha256(raw).hexdigest()==result['received_sha256']
        sent=json.loads(raw)
        assert sent['request_id']==engine.last_call_timing['request_id']
        assert sent['robot_state']==payload and sent['now']==12.3
        payload['joint_positions']['arm']=1.
        assert json.loads(raw)['robot_state']['joint_positions']['arm']==.1
        engine.call('validate_sim_candidate')
        assert engine.last_call_request_bytes is None
        engine.call('validate')
        assert engine.last_call_request_bytes is not None
        with pytest.raises(ValueError,match='cannot override'):
            engine.call('validate',command_format='unitree_float32')
        assert engine.last_call_request_bytes is None and engine.last_call_timing is None


@pytest.mark.parametrize('kind', ['motion', 'grasp'])
def test_fragmented_json_roundtrip_preserves_complete_message(kind, fake_process):
    code = READY + """
for line in sys.stdin:
    request=json.loads(line)
    answer=json.dumps({'status':'ok','request_id':request['request_id'],'result':{'value':'ä'+('x'*70000)}}).encode()+b'\\n'
    for start in range(0,len(answer),509):
        os.write(1,answer[start:start+509])
"""
    with fake_process(kind, code, timeout=1.) as (engine, _):
        result = invoke(kind, engine)
        assert result['value'] == 'ä' + 'x' * 70000


@pytest.mark.parametrize('kind', ['motion', 'grasp'])
def test_partial_json_response_cannot_bypass_deadline(kind, fake_process):
    code = READY + "sys.stdin.readline()\nos.write(1,b'{\"status\":\"ok\"')\ntime.sleep(30)\n"
    with fake_process(kind, code) as (engine, _):
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            invoke(kind, engine)
        assert time.monotonic() - start < 1.
        assert engine.closed and engine.process.poll() is not None


@pytest.mark.parametrize('kind', ['motion', 'grasp'])
def test_partial_startup_response_cannot_bypass_deadline(kind, fake_process):
    code = "import os,time\nos.write(1,b'{\"status\":')\ntime.sleep(30)\n"
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        with fake_process(kind, code, startup_timeout=.12):
            pytest.fail('Partial ready message must never initialize the worker')
    assert time.monotonic() - start < 1.


@pytest.mark.parametrize('kind', ['motion', 'grasp'])
@pytest.mark.parametrize('behavior', ['die', 'malformed', 'wrong_id', 'oversized'])
def test_broken_worker_fails_closed(kind, behavior, fake_process):
    endings = {
        'die': "sys.exit(0)",
        'malformed': "print('not-json',flush=True)",
        'wrong_id': "print(json.dumps({'status':'ok','request_id':99,'result':{}}),flush=True)",
        'oversized': "os.write(1,b'x'*(5*1024*1024));time.sleep(30)",
    }
    code = READY + "sys.stdin.readline()\n" + endings[behavior] + "\n"
    with fake_process(kind, code, timeout=1.) as (engine, _):
        with pytest.raises((ValueError, RuntimeError)):
            invoke(kind, engine)
        assert engine.closed and engine.process.poll() is not None


def test_motion_request_partial_writes_are_retried_without_data_loss(fake_process, monkeypatch):
    code = READY + """
for line in sys.stdin:
    request=json.loads(line)
    print(json.dumps({'status':'ok','request_id':request['request_id'],'result':{'count':len(request['padding'])}}),flush=True)
"""
    with fake_process('motion', code, timeout=2.) as (engine, module):
        original_write = module.os.write
        input_fd = engine.process.stdin.fileno()
        writes = []
        def partial(fd, data):
            if fd == input_fd:
                writes.append(len(data))
                return original_write(fd, data[:17])
            return original_write(fd, data)
        monkeypatch.setattr(module.os, 'write', partial)
        assert engine.call('validate', padding='a' * 20000)['count'] == 20000
        assert len(writes) > 100


def test_motion_blocked_reader_has_a_write_deadline(fake_process):
    # A watchdog prevents a regression to blocking write from hanging pytest.
    with fake_process('motion', READY + 'time.sleep(30)\n', timeout=.12) as (engine, _):
        finished = threading.Event()
        errors = []
        def call():
            try:
                engine.call('validate', padding='a' * (1024 * 1024))
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()
        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        bounded = finished.wait(timeout=1.)
        if not bounded:
            engine.process.kill()
            engine.process.wait(timeout=1.)
        worker.join(timeout=1.)
        assert bounded, 'MotionProcess blocked writing before its read timeout started'
        assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
        assert engine.closed


@pytest.mark.parametrize('kind', ['motion', 'grasp'])
def test_missing_interpreter_cleans_temporary_files(kind,monkeypatch,tmp_path):
    module = importlib.import_module('teleop_assist.copilot.motion_process' if kind=='motion'
                                     else 'teleop_assist.copilot.process')
    real_directory=tempfile.TemporaryDirectory
    created=[]
    def directory(*args,**kwargs):
        result=real_directory(*args,dir=tmp_path,**kwargs);created.append(Path(result.name));return result
    monkeypatch.setattr(module.tempfile,'TemporaryDirectory',directory)
    with pytest.raises(FileNotFoundError):
        if kind=='motion':module.MotionProcess('/missing/copilot-python','/unused')
        else:module.GraspProcess('/missing/copilot-python')
    assert created and all(not p.exists() for p in created)
