"""Launcher must preserve explicit failures despite Kit shutdown and signals."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def interpreter(tmp_path,body):
    fake=tmp_path/'fake_isaac_python'
    fake.write_text('#!'+sys.executable+'\n'+body)
    fake.chmod(0o755)
    return {**os.environ,'PUNKTNULL_ISAAC_PYTHON':str(fake)}

def test_explicit_application_failure_survives_zero_kit_exit(tmp_path):
    env=interpreter(tmp_path,"import os,pathlib\npathlib.Path(os.environ['PUNKTNULL_ASSIST_EXIT_FILE']).write_text('2')\n")
    result=subprocess.run([sys.executable,str(ROOT/'run_assist.py')],env=env,capture_output=True,timeout=10)
    assert result.returncode==2,result.stderr.decode()

def test_parent_termination_waits_for_child_status_flush(tmp_path):
    ready=tmp_path/'ready'
    body="""import os,pathlib,signal,time
done=False
def stop(*unused):
    global done
    done=True
signal.signal(signal.SIGTERM,stop)
pathlib.Path(os.environ['ASSIST_TEST_READY']).write_text('ready')
while not done:time.sleep(.02)
time.sleep(.4)
pathlib.Path(os.environ['PUNKTNULL_ASSIST_EXIT_FILE']).write_text('0')
"""
    env=interpreter(tmp_path,body);env['ASSIST_TEST_READY']=str(ready)
    process=subprocess.Popen([sys.executable,str(ROOT/'run_assist.py')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert ready.exists()
        process.send_signal(signal.SIGTERM)
        _,error=process.communicate(timeout=10)
        assert process.returncode==0,error.decode()
    finally:
        if process.poll() is None:process.kill();process.wait()
