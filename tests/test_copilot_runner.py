"""Release CLI/lifecycle tests. No model, simulator, DDS or GPU is started."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest
from run_copilot import configuration, stop_children, group_running, group_process_details, scripted_completion

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT/'run_copilot.py'


def test_configuration_preserves_virtualenv_python_symlink(tmp_path):
    executable=tmp_path/'venv'/'bin'/'python'
    executable.parent.mkdir(parents=True);executable.symlink_to(sys.executable)
    cfg=config(tmp_path/'config.json',executable)
    _,paths=configuration(cfg)
    assert paths['isaac_python']==str(executable)
    assert paths['isaac_python']!=str(executable.resolve())


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        return listener.getsockname()[1]


def source_review(path):
    path.mkdir()
    (path/'latest_proposals.json').write_text(json.dumps({'valid':True,'motion_static':True,
        'runtime_motion_guard_available':True,'capture_timestamp':1.,'candidates':[]}))
    (path/'ui_state.json').write_text(json.dumps({'mode':'assist','objects':[]}))
    (path/'frame.jpg').write_bytes(b'read-only-source-picture')
    return path


def config(path, python):
    path.write_text(json.dumps({'paths':dict.fromkeys(('isaac_python','xr_repository','xr_python',
        'perception_python','grasp_python','grasp_repository','model_root','isaac_repository'),str(python))}))
    return path


def run(arguments, *, env=None, timeout=25):
    return subprocess.run([sys.executable,str(RUNNER),*map(str,arguments)],
        capture_output=True,text=True,env=env,timeout=timeout)


@pytest.fixture
def harmless_backend(tmp_path):
    # It only emulates readiness/process lifecycle, and writes a marker if the
    # launcher would have started its simulator. It cannot import Isaac.
    path=tmp_path/'python_stub'
    path.write_text('#!'+sys.executable+'\n'+r'''
import json,os,pathlib,signal,socket,subprocess,sys,time
script=pathlib.Path(sys.argv[1]).name
if os.environ.get('PROCESS_RECORD_DIR'):
 record_dir=pathlib.Path(os.environ['PROCESS_RECORD_DIR']);record_dir.mkdir(exist_ok=True)
 (record_dir/(script+'.json')).write_text(json.dumps(dict(argv=sys.argv[1:],pid=os.getpid(),started=time.monotonic())))
if script=='check_sim_run.py':
 assert '--require-measured-poses' in sys.argv
 output=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
 mode=os.environ.get('STUB_PHASE_ACCEPTANCE','valid')
 if mode=='malformed':output.write_text('{broken')
 elif mode!='missing':output.write_text(json.dumps(dict(accepted=mode in ('valid','nonzero'),test_fixture='CLI_PROTOCOL_STUB_NO_PHYSICS')))
 sys.exit(1 if mode=='nonzero' else 0)
if os.environ.get('ASSIST_ARGS'):pathlib.Path(os.environ['ASSIST_ARGS']).write_text(json.dumps(sys.argv[1:]))
if os.environ.get('ASSIST_ENV'):pathlib.Path(os.environ['ASSIST_ENV']).write_text(os.environ.get('PUNKTNULL_ISAAC_REPOSITORY',''))
if script=='capture_unitree_scene.py':
 pathlib.Path(os.environ['SIM_MARKER']).write_text('simulator requested')
 while True:time.sleep(.05)
if os.environ.get('STUB_START_FAIL')=='1':sys.exit(17)
if os.environ.get('STUB_NO_READY')=='1':
 while True:time.sleep(.05)
if script=='serve.py' and '--no-dashboard' in sys.argv:
 assert (record_dir/'primary-ready').exists(), 'Proposal worker started before primary readiness'
 signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
 if os.environ.get('STUB_PROPOSAL_EXIT'):
  time.sleep(.1);sys.exit(int(os.environ['STUB_PROPOSAL_EXIT']))
 while True:time.sleep(.05)
port=int(sys.argv[sys.argv.index('--port')+1])
listener=socket.socket();listener.bind(('127.0.0.1',port));listener.listen()
if os.environ.get('STUB_WORKER'):
 child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])
 pathlib.Path(os.environ['STUB_WORKER']).write_text(str(child.pid))
time.sleep(.05)
def finish(*_):
 if script=='sim_runtime.py' and '--scripted-test' in sys.argv:
  output=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
  mode=os.environ.get('STUB_SIM_REPORT','incomplete')
  report=dict(completed=True,scripted_scenario_completed=mode in ('valid','no_commands'),
              commands_consumed=1 if mode=='valid' else 0,test_fixture='CLI_PROTOCOL_STUB_NO_PHYSICS')
  if mode=='malformed':(output/'report.json').write_text('{broken')
  elif mode!='missing':(output/'report.json').write_text(json.dumps(report))
 sys.exit(int(os.environ.get('STUB_CLEANUP_EXIT','0')))
signal.signal(signal.SIGTERM,finish)
if os.environ.get('PROCESS_RECORD_DIR'):(record_dir/'primary-ready').write_text(str(time.monotonic()))
print(('Isolated PhysX controls: http://127.0.0.1:'+str(port)+'/sim') if script=='sim_runtime.py' else ('Copilot dashboard: http://127.0.0.1:'+str(port)),flush=True)
started=time.monotonic()
while True:
 if os.environ.get('STUB_PRIMARY_EXIT_AFTER') and time.monotonic()-started>float(os.environ['STUB_PRIMARY_EXIT_AFTER']):finish()
 time.sleep(.05)
''')
    path.chmod(0o755)
    return path


@pytest.mark.parametrize('arguments',[['--review','missing','--duration','0'],
    ['--review','missing','--duration','nan'],['--review','missing','--port','0'],
    ['--review','missing','--port','65536'],['--demo','--review','missing'],
    ['--doctor','--sim-command-format','unitree_float32'],
    ['--demo','--sim-command-format','float64'],
    ['--assist-sim','--scripted-test','--sim-command-format','hardware']])
def test_invalid_cli_fails_without_runtime(arguments,tmp_path):
    runtime=tmp_path/'must-not-exist'
    result=run([*arguments,'--runtime',runtime])
    assert result.returncode==2 and not runtime.exists()


@pytest.mark.parametrize('arguments',[
    ['--demo','--sim-sender','guarded_original'],
    ['--assist-sim','--scripted-test','--sim-sender','guarded_original'],
    ['--assist-sim','--scripted-test','--sim-sender','guarded_original','--sim-command-format','float64']])
def test_original_sim_sender_requires_explicit_float32_simulator_scope(tmp_path,arguments):
    runtime=tmp_path/'must-not-exist'
    result=run([*arguments,'--runtime',runtime])
    assert result.returncode==2 and not runtime.exists()


def test_original_sim_sender_is_forwarded_to_owned_runtime(tmp_path,harmless_backend):
    cfg=config(tmp_path/'config.json',harmless_backend);runtime=tmp_path/'runtime';records=tmp_path/'records'
    result=run(['--assist-sim','--scripted-test','--sim-sender','guarded_original',
        '--sim-command-format','unitree_float32','--config',cfg,'--runtime',runtime,
        '--duration','.5','--port',free_port()],env=dict(os.environ,PROCESS_RECORD_DIR=str(records),STUB_SIM_REPORT='valid'))
    assert result.returncode==0,result.stderr
    command=json.loads((records/'sim_runtime.py.json').read_text())['argv']
    assert command[command.index('--sender')+1]=='guarded_original'
    assert '--require-measured-poses' in json.loads((records/'check_sim_run.py.json').read_text())['argv']


def test_invalid_review_is_checked_before_runtime_creation(tmp_path):
    cfg=config(tmp_path/'config.json',sys.executable)
    runtime=tmp_path/'runtime'
    result=run(['--review',tmp_path/'missing','--config',cfg,'--runtime',runtime])
    assert result.returncode==2 and not runtime.exists()


@pytest.mark.parametrize('separate_visual',[False,True])
def test_review_copies_without_mutation_and_invalidates_recording(tmp_path,separate_visual):
    source=source_review(tmp_path/'source');cfg=config(tmp_path/'config.json',sys.executable)
    if separate_visual:
        visual=json.loads((source/'latest_proposals.json').read_text())
        visual['visual_fixture']='exact_bound_projection'
        (source/'preview_proposals.json').write_text(json.dumps(visual))
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    runtime=tmp_path/'runtime'
    result=run(['--review',source,'--config',cfg,'--runtime',runtime,'--duration','1','--port',free_port()])
    assert result.returncode==0,result.stderr
    batch=json.loads((runtime/'dashboard/latest_proposals.json').read_text())
    assert batch['recording_only'] and not batch['valid'] and not batch['motion_static']
    assert not batch['runtime_motion_guard_available'] and batch['capture_timestamp']==1.
    assert ('visual_fixture' in batch)==separate_visual
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    report=json.loads((runtime/'session.json').read_text())
    assert report['service_ready'] and not report['real_robot_connected']


@pytest.mark.parametrize('failure',['port_collision','start_failure','missing_executable','readiness_deadline'])
def test_demo_start_failures_never_launch_simulator(tmp_path,harmless_backend,failure):
    runtime=tmp_path/'runtime';marker=tmp_path/'sim-started'
    env=dict(os.environ,SIM_MARKER=str(marker))
    executable=tmp_path/'not-installed' if failure=='missing_executable' else harmless_backend
    cfg=config(tmp_path/'config.json',executable)
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1',0));occupied.listen()
        port=occupied.getsockname()[1] if failure=='port_collision' else free_port()
        if failure=='start_failure':env['STUB_START_FAIL']='1'
        if failure=='readiness_deadline':env['STUB_NO_READY']='1'
        result=run(['--demo','--config',cfg,'--runtime',runtime,'--duration','.5','--port',port],env=env)
    assert result.returncode==1 and not marker.exists(),result.stdout+result.stderr
    report=json.loads((runtime/'session.json').read_text())
    assert not report['service_ready'] and not report['real_robot_connected']
    assert report['wall_s']<3


@pytest.mark.parametrize('interrupt',[False,True])
def test_deadline_or_interrupt_ends_descendant_workers(tmp_path,harmless_backend,interrupt):
    source=source_review(tmp_path/'source');cfg=config(tmp_path/'config.json',harmless_backend)
    runtime=tmp_path/'runtime';worker_pid=tmp_path/'worker.pid'
    env=dict(os.environ,STUB_WORKER=str(worker_pid),SIM_MARKER=str(tmp_path/'unused'))
    process=subprocess.Popen([sys.executable,str(RUNNER),'--review',str(source),'--config',str(cfg),
        '--runtime',str(runtime),'--duration','5' if interrupt else '1','--port',str(free_port())],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    try:
        deadline=time.monotonic()+3
        while not worker_pid.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert worker_pid.exists()
        if interrupt:
            time.sleep(.1);process.send_signal(signal.SIGINT)
        stdout,stderr=process.communicate(timeout=22)
        # The descendant deliberately ignores SIGTERM. Forced cleanup must
        # terminate it but cannot report a clean release lifecycle.
        assert process.returncode==1,stdout+stderr
        report=json.loads((runtime/'session.json').read_text())
        assert not report['cleanup_accepted']
        assert report['stop_reason']==('signal' if interrupt else 'duration_expired')
        assert report['requested_stop'] is interrupt
        assert 'service' in report['forced_process_groups']
        details=report['forced_process_details']['service']
        assert details and all(set(item)=={'pid','ppid','pgid','session','comm'} for item in details)
        stat=Path('/proc')/worker_pid.read_text()/'stat'
        assert not stat.exists() or stat.read_text().split()[2]=='Z'
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=3)


def test_parent_exit_does_not_cut_short_worker_sigterm_cleanup(tmp_path):
    ready=tmp_path/'ready';clean=tmp_path/'clean'
    worker='''import pathlib,signal,sys,time
def stop(*_):
 time.sleep(.2)
 pathlib.Path(sys.argv[2]).write_text('finished')
 sys.exit(0)
signal.signal(signal.SIGTERM,stop)
pathlib.Path(sys.argv[1]).write_text('ready')
while True:time.sleep(.05)
'''
    parent='''import signal,subprocess,sys,time
signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]])
while True:time.sleep(.05)
'''
    child=subprocess.Popen([sys.executable,'-c',parent,worker,str(ready),str(clean)],start_new_session=True)
    try:
        deadline=time.monotonic()+2
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert ready.exists()
        start=time.monotonic()
        assert stop_children([('worker-group',child)],grace_s=1.)==[]
        assert .18<=time.monotonic()-start<1.
        assert clean.read_text()=='finished' and not group_running(child.pid)
    finally:
        if group_running(child.pid):os.killpg(child.pid,signal.SIGKILL)
        child.wait(timeout=2)


def test_force_details_include_only_our_group_and_never_command_arguments(tmp_path):
    ready=tmp_path/'worker.pid'
    worker='''import os,pathlib,signal,sys,time
signal.signal(signal.SIGTERM,signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
while True:time.sleep(.02)
'''
    parent='''import subprocess,sys,time
subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],'DO_NOT_RECORD_FAKE_SECRET_2718'])
while True:time.sleep(.02)
'''
    child=subprocess.Popen([sys.executable,'-c',parent,worker,str(ready)],start_new_session=True)
    outsider=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
    details={}
    try:
        deadline=time.monotonic()+2
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert ready.exists();worker_pid=int(ready.read_text())
        assert {item['pid'] for item in group_process_details(child.pid)}=={child.pid,worker_pid}
        assert stop_children([('owned',child)],grace_s=.1,forced_details=details)==['owned']
        assert set(details)=={'owned'} and [item['pid'] for item in details['owned']]==[worker_pid]
        assert all(set(item)=={'pid','ppid','pgid','session','comm'} for item in details['owned'])
        assert all(item['pgid']==item['session']==child.pid for item in details['owned'])
        assert 'DO_NOT_RECORD_FAKE_SECRET' not in json.dumps(details)
        assert outsider.poll() is None  # unrelated group was neither listed nor killed
    finally:
        for process in (child,outsider):
            if group_running(process.pid):os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=2)


@pytest.mark.parametrize('arguments',[['--assist-sim'],['--demo','--scripted-test'],
    ['--assist-sim','--operator-mailbox','operator.json','--scripted-test']])
def test_assist_sim_requires_exactly_one_explicit_input_source(tmp_path,arguments):
    runtime=tmp_path/'runtime'
    result=run([*arguments,'--runtime',runtime])
    assert result.returncode==2 and not runtime.exists()


@pytest.mark.parametrize('scripted',[False,True])
@pytest.mark.parametrize('command_format',[None,'unitree_float32'])
def test_assist_sim_uses_one_isolated_process_and_correct_explicit_source(tmp_path,harmless_backend,scripted,command_format):
    cfg=config(tmp_path/'config.json',harmless_backend);runtime=tmp_path/'runtime';record=tmp_path/'args.json';environment_record=tmp_path/'isaac_path'
    source=['--scripted-test'] if scripted else ['--operator-mailbox',str(tmp_path/'operator.json')]
    if command_format:source+=['--sim-command-format',command_format]
    result=run(['--assist-sim',*source,'--config',cfg,'--runtime',runtime,'--duration','.5','--port',free_port()],
        env=dict(os.environ,ASSIST_ARGS=str(record),ASSIST_ENV=str(environment_record),STUB_SIM_REPORT='valid'))
    assert result.returncode==0,result.stdout+result.stderr
    command=json.loads(record.read_text())
    assert Path(command[0]).name=='sim_runtime.py' and '--sim' in command and '--headless' in command
    assert environment_record.read_text()==str(harmless_backend)
    assert ('--scripted-test' in command) is scripted
    assert ('--operator-mailbox' in command) is not scripted
    assert command[command.index('--camera-source')+1]=='head'
    assert command[command.index('--command-format')+1]==(command_format or 'float64')
    assert ('--ticks' in command) is scripted
    if scripted:assert command[command.index('--ticks')+1]=='145'
    assert '/sim' in result.stdout
    report=json.loads((runtime/'session.json').read_text())
    assert report['mode']==('scripted_assist_sim' if scripted else 'xr_assist_sim')
    assert set(report['processes'])=={'simulation'}
    assert report['camera_source']=='head' and report['scripted_ticks']==(145 if scripted else None)
    assert report['scripted_sequence_completed'] is (True if scripted else None)


def test_assist_sim_port_collision_does_not_start_backend(tmp_path,harmless_backend):
    cfg=config(tmp_path/'config.json',harmless_backend);runtime=tmp_path/'runtime';record=tmp_path/'args.json'
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1',0));occupied.listen()
        result=run(['--assist-sim','--scripted-test','--config',cfg,'--runtime',runtime,
            '--duration','.5','--port',occupied.getsockname()[1]],env=dict(os.environ,ASSIST_ARGS=str(record)))
    assert result.returncode==2 and not runtime.exists() and not record.exists()


@pytest.mark.parametrize('mode',[['--demo'],['--doctor'],['--review','missing']])
def test_live_ai_is_opt_in_only_for_assist_sim(tmp_path,mode):
    runtime=tmp_path/'runtime'
    result=run([*mode,'--live-ai','--runtime',runtime])
    assert result.returncode==2 and not runtime.exists()


def test_live_ai_requires_model_paths_before_starting_anything(tmp_path,harmless_backend):
    cfg=tmp_path/'config.json';runtime=tmp_path/'runtime'
    cfg.write_text(json.dumps({'paths':dict.fromkeys(('isaac_python','xr_repository','xr_python','isaac_repository'),str(harmless_backend))}))
    result=run(['--assist-sim','--scripted-test','--live-ai','--config',cfg,'--runtime',runtime])
    assert result.returncode==2 and not runtime.exists()


@pytest.mark.parametrize('mode',[['--demo'],['--assist-sim','--scripted-test','--live-ai']])
def test_configured_external_model_paths_reach_actual_service_process(tmp_path,harmless_backend,mode):
    cfg=config(tmp_path/'config.json',harmless_backend)
    value=json.loads(cfg.read_text())
    value['paths'].update(grasp_repository=str(tmp_path/'external grasp checkout'),
                          model_root=str(tmp_path/'external perception weights'))
    cfg.write_text(json.dumps(value))
    records=tmp_path/'records'
    result=run([*mode,'--config',cfg,'--runtime',tmp_path/'runtime','--duration','.6','--port',free_port()],
        env=dict(os.environ,PROCESS_RECORD_DIR=str(records),SIM_MARKER=str(tmp_path/'sim_marker'),
                 STUB_SIM_REPORT='valid'))
    assert result.returncode==0,result.stdout+result.stderr
    command=json.loads((records/'serve.py.json').read_text())['argv']
    for flag,key in [('--grasp-repo','grasp_repository'),('--model-root','model_root')]:
        assert command[command.index(flag)+1]==value['paths'][key]


@pytest.mark.parametrize('missing',['grasp_repository','model_root'])
def test_missing_external_model_location_fails_before_start(tmp_path,harmless_backend,missing):
    cfg=config(tmp_path/'config.json',harmless_backend)
    value=json.loads(cfg.read_text());value['paths'].pop(missing);cfg.write_text(json.dumps(value))
    runtime=tmp_path/'runtime'
    result=run(['--assist-sim','--scripted-test','--live-ai','--config',cfg,'--runtime',runtime])
    assert result.returncode==2 and missing in result.stderr and not runtime.exists()


@pytest.mark.parametrize('arguments',[
    ['--demo','--camera-source','head'],['--doctor','--camera-source','right_wrist'],
    ['--doctor','--operator-camera-source','head'],
    ['--doctor','--sim-initialization','missing.json'],
    ['--assist-sim','--operator-mailbox','operator.json','--ticks','150'],
    ['--assist-sim','--scripted-test','--ticks','129'],
    ['--assist-sim','--scripted-test','--ticks','1001'],
    ['--assist-sim','--scripted-test','--camera-source','unknown']])
def test_camera_and_ticks_reject_wrong_mode_or_unbounded_values(tmp_path,arguments):
    runtime=tmp_path/'runtime';result=run([*arguments,'--runtime',runtime])
    assert result.returncode==2 and not runtime.exists()


@pytest.mark.parametrize('camera,ticks',[('right_wrist',1000),('left_wrist',130)])
def test_explicit_original_camera_and_script_ticks_pass_to_primary_only(tmp_path,harmless_backend,camera,ticks):
    cfg=config(tmp_path/'config.json',harmless_backend);runtime=tmp_path/'runtime';records=tmp_path/'records'
    initialization=tmp_path/'initialization.json'
    initialization.write_text('{}')  # The simulator owns numerical validation.
    result=run(['--assist-sim','--scripted-test','--live-ai','--camera-source',camera,
        '--operator-camera-source','head','--ticks',ticks,
        '--sim-initialization',initialization,
        '--config',cfg,'--runtime',runtime,'--duration','.6','--port',free_port()],
        env=dict(os.environ,PROCESS_RECORD_DIR=str(records),STUB_SIM_REPORT='valid'))
    assert result.returncode==0,result.stdout+result.stderr
    command=json.loads((records/'sim_runtime.py.json').read_text())['argv']
    assert command[command.index('--camera-source')+1]==camera
    assert command[command.index('--operator-camera-source')+1]=='head'
    assert command[command.index('--initialization')+1]==str(initialization)
    assert command[command.index('--ticks')+1]==str(ticks)
    service=json.loads((records/'serve.py.json').read_text())['argv']
    assert '--camera-source' not in service and '--ticks' not in service
    assert '--operator-camera-source' not in service
    report=json.loads((runtime/'session.json').read_text())
    assert report['camera_source']==camera and report['scripted_ticks']==ticks
    assert report['operator_camera_source']=='head'


@pytest.mark.parametrize('ending',['duration','primary_completed','worker_error','worker_early_success','interrupt','primary_start_failure'])
def test_live_ai_worker_uses_primary_dashboard_and_lifecycle(tmp_path,harmless_backend,ending):
    cfg=config(tmp_path/'config.json',harmless_backend);runtime=tmp_path/'runtime';records=tmp_path/'records'
    env=dict(os.environ,PROCESS_RECORD_DIR=str(records))
    if ending=='primary_completed':env.update(STUB_PRIMARY_EXIT_AFTER='.3',STUB_SIM_REPORT='valid')
    if ending=='worker_error':env['STUB_PROPOSAL_EXIT']='17'
    if ending=='worker_early_success':env['STUB_PROPOSAL_EXIT']='0'
    if ending=='primary_start_failure':env['STUB_START_FAIL']='1'
    args=['--assist-sim','--scripted-test','--live-ai','--config',str(cfg),'--runtime',str(runtime),
          '--duration','.7' if ending=='duration' else '10','--prompt','blue test cup','--port',str(free_port())]
    process=subprocess.Popen([sys.executable,str(RUNNER),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    try:
        if ending=='interrupt':
            deadline=time.monotonic()+3
            while not (records/'serve.py.json').exists() and time.monotonic()<deadline:time.sleep(.01)
            assert (records/'serve.py.json').exists();time.sleep(.05);process.send_signal(signal.SIGINT)
        stdout,stderr=process.communicate(timeout=5)
        report=json.loads((runtime/'session.json').read_text())
        assert report['live_ai'] and not report['real_robot_connected']
        primary=json.loads((records/'sim_runtime.py.json').read_text())
        assert '--live-ai-frames' in primary['argv']
        assert primary['argv'][primary['argv'].index('--live-ai-frames')+1]==str(runtime/'frames')
        assert not group_running(primary['pid'])
        if ending=='primary_start_failure':
            assert process.returncode==1 and not (records/'serve.py.json').exists()
            assert set(report['processes'])=={'simulation'}
            assert report['stop_reason']=='test_incomplete' and not report['scripted_sequence_completed']
            return
        worker=json.loads((records/'serve.py.json').read_text())
        assert worker['started']>=float((records/'primary-ready').read_text())
        command=worker['argv']
        assert '--no-dashboard' in command and '--port' not in command
        for flag,value in (('--live-frames',str(runtime/'frames')),('--runtime',str(runtime/'simulation')),
            ('--prompt','blue test cup'),('--xr-repo',str(harmless_backend)),('--ik-python',str(harmless_backend)),
            ('--model-python',str(harmless_backend)),('--grasp-python',str(harmless_backend))):
            assert command[command.index(flag)+1]==value
        assert set(report['processes'])=={'simulation','proposal_service'}
        assert not group_running(worker['pid']) and not report['forced_process_groups']
        assert report['wall_s']<3
        if ending=='primary_completed':
            assert process.returncode==0,stdout+stderr
            assert report['scripted_sequence_completed']
        else:
            assert process.returncode==1 and report['stop_reason']=='test_incomplete'
            assert not report['scripted_sequence_completed']
        if ending.startswith('worker_'):assert report['process_stop_reason']=='proposal_service_failed'
        if ending=='interrupt':assert report['requested_stop']
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=2)
        if records.exists():
            for record in records.glob('*.json'):
                pid=json.loads(record.read_text())['pid']
                if group_running(pid):os.killpg(pid,signal.SIGKILL)


@pytest.mark.parametrize('mode',['incomplete','no_commands','missing','malformed'])
def test_zero_primary_exit_does_not_claim_scripted_completion(tmp_path,harmless_backend,mode):
    cfg=config(tmp_path/'config.json',harmless_backend);runtime=tmp_path/'runtime'
    result=run(['--assist-sim','--scripted-test','--config',cfg,'--runtime',runtime,
        '--duration','3','--port',free_port()],env=dict(os.environ,STUB_SIM_REPORT=mode,STUB_PRIMARY_EXIT_AFTER='.1'))
    report=json.loads((runtime/'session.json').read_text())
    assert report['processes']['simulation']==0 and result.returncode==1
    assert report['stop_reason']=='test_incomplete' and not report['scripted_sequence_completed']
    assert report['process_stop_reason']=='service_completed'


@pytest.mark.parametrize('mode', ['rejected','malformed','missing','nonzero','valid'])
def test_finished_simulation_requires_independent_phase_acceptance(tmp_path,harmless_backend,mode):
    cfg=config(tmp_path/'config.json',harmless_backend);runtime=tmp_path/'runtime'
    result=run(['--assist-sim','--scripted-test','--config',cfg,'--runtime',runtime,
        '--duration','3','--port',free_port()],env=dict(os.environ,STUB_SIM_REPORT='valid',
        STUB_PRIMARY_EXIT_AFTER='.1',STUB_PHASE_ACCEPTANCE=mode))
    report=json.loads((runtime/'session.json').read_text())
    assert report['processes']['simulation']==0 and report['scripted_sequence_completed']
    assert report['scripted_phase_accepted'] is (mode=='valid')
    assert result.returncode == (0 if mode=='valid' else 1)
    if mode!='valid':assert report['stop_reason']=='test_acceptance_failed'
    assert not report['real_robot_connected']


@pytest.mark.parametrize('report',[
    {'completed':False,'scripted_scenario_completed':True,'commands_consumed':2},
    {'completed':True,'scripted_scenario_completed':False,'commands_consumed':2},
    {'completed':True,'scripted_scenario_completed':True,'commands_consumed':True},
    {'completed':True,'scripted_scenario_completed':True,'commands_consumed':-1},
    {'completed':True,'scripted_scenario_completed':True,'commands_consumed':float('nan')},
    {'completed':True,'scripted_scenario_completed':True},[]])
def test_scripted_completion_rejects_missing_or_wrong_completion_fields(tmp_path,report):
    path=tmp_path/'report.json';path.write_text(json.dumps(report))
    assert scripted_completion(path)[0] is False


def test_completion_reads_saved_runtime_report_without_importing_simulator():
    path=ROOT/'runs/copilot_sim_live_003/simulation/report.json'
    if not path.is_file():pytest.skip('Local recorded incomplete run unavailable')
    assert scripted_completion(path)==(False,'scenario_not_completed')
