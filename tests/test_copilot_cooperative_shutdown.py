"""Actual subprocess regression for stop during an outstanding worker request."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import run_copilot as runner

from run_copilot import group_running, stop_children


def test_owner_can_finish_inflight_reply_before_shutting_down_worker(tmp_path):
    ready, premature, finished = (tmp_path/name for name in ('ready','premature','finished'))
    worker = '''import pathlib,signal,sys,time
def interrupted(*_):
 pathlib.Path(sys.argv[2]).write_text('worker received group TERM during pending request')
 sys.exit(17)
signal.signal(signal.SIGTERM,interrupted)
assert sys.stdin.readline().strip()=='request'
pathlib.Path(sys.argv[1]).write_text('working')
time.sleep(.25)
print('reply',flush=True)
assert sys.stdin.read()==''
'''
    parent = '''import pathlib,signal,subprocess,sys,time
stopping=False
def stop(*_):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop)
p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
p.stdin.write('request\\n');p.stdin.flush()
assert p.stdout.readline().strip()=='reply'
while not stopping:time.sleep(.01)
p.stdin.close();p.stdout.close()
assert p.wait(timeout=1)==0
pathlib.Path(sys.argv[4]).write_text('normal owner audit finished')
'''
    child = subprocess.Popen([sys.executable,'-c',parent,worker,str(ready),str(premature),str(finished)],
                             start_new_session=True)
    try:
        deadline=time.monotonic()+3
        while not ready.exists() and time.monotonic()<deadline:
            time.sleep(.01)
        assert ready.exists()
        assert stop_children([('owner',child)],grace_s=2.) == []
        assert child.returncode == 0 and finished.exists()
        assert not premature.exists() and not group_running(child.pid)
    finally:
        if group_running(child.pid):os.killpg(child.pid,signal.SIGKILL)
        child.wait(timeout=2)


def test_one_wait_timeout_cannot_skip_killing_and_reaping_later_owned_groups(monkeypatch):
    calls = []
    def waiting(name, fail=False):
        def wait(timeout):
            calls.append(('wait', name))
            if fail:raise subprocess.TimeoutExpired(name, timeout)
            return 0
        return wait
    children = [SimpleNamespace(pid=100+i, poll=lambda:None,
        send_signal=lambda sig:None, wait=waiting(str(i), fail=i == 0)) for i in range(2)]
    monkeypatch.setattr(runner, 'group_running', lambda pid: True)
    monkeypatch.setattr(runner, 'group_process_details', lambda pid: [{'pid':pid}])
    monkeypatch.setattr(runner.os, 'killpg', lambda pid,sig:calls.append(('signal',pid,sig)))
    with pytest.raises(subprocess.TimeoutExpired):
        runner.stop_children(list(zip(('first','second'),children)), grace_s=0)
    kills = [(index,item[1]) for index,item in enumerate(calls) if item[0]=='signal' and item[2]==signal.SIGKILL]
    assert [pid for _,pid in kills] == [100,101]
    assert all(index < calls.index(('wait','0')) for index,_ in kills)
    assert ('wait','1') in calls
