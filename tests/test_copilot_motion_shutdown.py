"""Real original-IK CPU lifecycle checks; no rendering, DDS, or arm commands."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import pytest

from run_copilot import group_process_details


ROOT = Path(__file__).resolve().parents[1]
IK_PYTHON = Path('/home/loya2/miniconda3/envs/to/bin/python')
XR_REPOSITORY = Path('/home/loya2/xr_teleoperate')


def original_ik_available():
    if not IK_PYTHON.is_file() or not (XR_REPOSITORY/'teleop/robot_control/robot_arm_ik.py').is_file():
        pytest.skip('Local original XR checkout and its CPU IK environment are unavailable')


def receive_line(process, timeout=45.):
    # Helper messages fit one pipe write; no buffered partial-read wait here.
    deadline = time.monotonic()+timeout
    message = b''
    while b'\n' not in message:
        remaining = deadline-time.monotonic()
        assert remaining > 0 and select.select([process.stdout], [], [], remaining)[0], 'Helper timeout'
        chunk = os.read(process.stdout.fileno(), 65536)
        assert chunk, 'Helper exited before its lifecycle result'
        message += chunk
    return json.loads(message.split(b'\n', 1)[0])


@pytest.mark.parametrize('stop_kind', ['eof', 'worker_term', 'partial_group_term'])
def test_real_original_ik_and_logging_listener_exit_together(stop_kind, tmp_path):
    original_ik_available()
    helper = tmp_path/'owner.py'
    helper.write_text('''
import json,os,select,signal,sys
from teleop_assist.copilot.motion_process import MotionProcess
stopping=False
def stop(*_):
    global stopping
    stopping=True
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
engine=MotionProcess(sys.argv[1],sys.argv[2],startup_timeout=40.)
try:
    print(json.dumps({'ready':True,'worker_pid':engine.process.pid}),flush=True)
    while not stopping and engine.process.poll() is None:
        if select.select([sys.stdin],[],[],.05)[0]:
            action=sys.stdin.readline().strip()
            if action=='partial':
                os.write(engine.process.stdin.fileno(),b'{"operation":')
                print(json.dumps({'partial_written':True}),flush=True)
            else:break
finally:
    engine.close()
print(json.dumps({'closed':True,'worker_exit':engine.process.returncode}),flush=True)
''')
    log = (tmp_path/'owner.stderr').open('w+')
    process = subprocess.Popen([sys.executable, str(helper), str(IK_PYTHON), str(XR_REPOSITORY)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, start_new_session=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
    initial = []
    try:
        ready = receive_line(process)
        assert ready['ready'] is True
        initial = group_process_details(process.pid)
        listeners = [item for item in initial if item['ppid']==ready['worker_pid']]
        assert listeners, 'The actual original logging_mp child must be present, not mocked'
        started = time.monotonic()
        if stop_kind == 'eof':
            process.stdin.close()
        elif stop_kind == 'worker_term':
            # Keep the owner's pipe open: stop cannot depend on receiving EOF.
            os.kill(ready['worker_pid'], signal.SIGTERM)
        else:
            process.stdin.write(b'partial\n'); process.stdin.flush()
            assert receive_line(process, 2.)['partial_written'] is True
            # This group belongs only to this test owner and its descendants.
            os.killpg(process.pid, signal.SIGTERM)
        closed = receive_line(process, 10.)
        assert closed == {'closed':True, 'worker_exit':0}
        assert process.wait(timeout=2.) == 0
        assert time.monotonic()-started < 10.
        assert group_process_details(process.pid) == [], 'No live owner, worker, or logger may remain'
        assert all(not Path(f'/proc/{item["pid"]}').exists() for item in listeners), 'Logger must be reaped'
    finally:
        # Only the session created above is eligible for test cleanup.
        if group_process_details(process.pid):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3.)
        if process.stdin and not process.stdin.closed: process.stdin.close()
        process.stdout.close(); log.close()


def test_worker_reports_initialization_error_with_nonzero_exit(tmp_path):
    original_ik_available()
    result = subprocess.run([str(IK_PYTHON),str(ROOT/'tools/copilot/motion_worker.py'),
        '--repository',str(tmp_path/'absent-checkout')], stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=15.,
        env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
    assert result.returncode != 0
    assert json.loads(result.stdout)['status']=='error'
