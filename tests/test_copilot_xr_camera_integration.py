"""Opt-in camera CLI and installer ordering, without constructing a server."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def source(monkeypatch):
    monkeypatch.setattr(sys,'path',list(sys.path))
    from tools.copilot import xr_source
    return xr_source


def wrapper_factory(source,monkeypatch,tmp_path):
    events=[];observed={}
    stage=tmp_path/'stage';stage.mkdir()
    public=stage/'public';public.mkdir()
    expected=stage/'teleop/televuer/src/televuer/televuer.py'
    original_type=type('StagedTeleVuer',(),{})
    def native_vuer(*a,**kw):
        observed['vuer']=kw
        return SimpleNamespace()
    module=SimpleNamespace(__file__=str(expected),Vuer=native_vuer,TeleVuer=original_type)
    def wrapper(**kwargs):
        events.append('wrapper');observed['wrapper']=kwargs
        return module.Vuer(host='0.0.0.0')
    monkeypatch.setattr(source.importlib,'import_module',lambda name:module if name=='televuer.televuer' else SimpleNamespace(TeleVuerWrapper=wrapper))
    monkeypatch.setattr(source,'prepare_public_web',lambda _: (public,public))
    monkeypatch.setattr(source,'install_public_file_routes',lambda *a:events.append('public_routes'))
    monkeypatch.setattr(source,'install_instrumentation',lambda root:events.append('input_instrumentation'))
    monkeypatch.setattr(source,'host_clock_id',lambda:'host_monotonic:TEST_FIXTURE')
    camera=ModuleType('teleop_assist.copilot.xr_camera')
    def install(cls,**kwargs):
        events.append('camera_install');observed['camera_class']=cls;observed['camera']=kwargs
        return dict(movement_authorized=False,headset_tested=False,control_mailboxes_written=False)
    camera.install_on_staged_televuer=install
    monkeypatch.setitem(sys.modules,camera.__name__,camera)
    return stage,expected,original_type,events,observed


def test_default_does_not_install_or_report_camera(source,monkeypatch,tmp_path):
    stage,_,_,events,observed=wrapper_factory(source,monkeypatch,tmp_path)
    source.prepare_wrapper(stage,host='127.0.0.1',port=8012,use_hand_tracking=False,
        cert_file='unused-cert',key_file='unused-key')
    assert events==['input_instrumentation','wrapper','public_routes']
    assert 'camera' not in observed and not (stage/'sim_camera_manifest.json').exists()


@pytest.mark.parametrize('hand_tracking',[False,True])
def test_opt_in_installs_after_instrumentation_before_constructor_with_own_host_clock(source,monkeypatch,tmp_path,hand_tracking):
    stage,expected,original_type,events,observed=wrapper_factory(source,monkeypatch,tmp_path)
    camera_dir=tmp_path/'current-simulation';camera_dir.mkdir()
    (camera_dir/'ui_state.json').write_text('{"TEST_FIXTURE": true}')
    before=(camera_dir/'ui_state.json').read_bytes()
    source.prepare_wrapper(stage,host='127.0.0.1',port=8012,use_hand_tracking=hand_tracking,
        cert_file='unused-cert',key_file='unused-key',sim_camera=camera_dir)
    assert events==['input_instrumentation','camera_install','wrapper','public_routes']
    assert observed['camera_class'] is original_type
    assert observed['camera']==dict(original_source=expected,directory=camera_dir,
        clock_id='host_monotonic:TEST_FIXTURE',width=640,height=480)
    assert observed['wrapper']['display_mode']=='pass-through'
    assert observed['wrapper']['use_hand_tracking'] is hand_tracking
    assert observed['wrapper']['zmq'] is False and observed['wrapper']['webrtc'] is False
    report=json.loads((stage/'sim_camera_manifest.json').read_text())
    assert report['directory']==str(camera_dir) and report['max_age_s']==.25
    assert report['movement_authorized'] is False and report['control_mailboxes_written'] is False
    assert (camera_dir/'ui_state.json').read_bytes()==before and len(list(camera_dir.iterdir()))==1


@pytest.mark.parametrize('opt_in',[False,True])
def test_check_reports_camera_configuration_without_loading_installer_or_starting_server(source,monkeypatch,tmp_path,capsys,opt_in):
    stage=tmp_path/'stage';camera=tmp_path/'simulation';camera.mkdir()
    calls=[]
    monkeypatch.setattr(source,'stage_checkout',lambda *a,**kw:calls.append('stage'))
    monkeypatch.setattr(source,'tls_pair',lambda args:(tmp_path/'not-read-cert',tmp_path/'not-read-key'))
    def forbidden(*a,**kw):raise AssertionError('Check must not construct source or display')
    monkeypatch.setattr(source,'prepare_wrapper',forbidden)
    args=['--sim','--check','--stage',str(stage)]
    if opt_in:args+=['--sim-camera',str(camera)]
    assert source.main(args)==0
    report=json.loads(capsys.readouterr().out)
    assert not report['server_started'] and not report['dds_imported'] and not report['live_headset_verified']
    assert calls==['stage']
    if opt_in:
        assert report['sim_camera']==dict(directory=str(camera),width=640,height=480,max_age_s=.25,
            read_only=True,movement_authorized=False,headset_delivery_verified=False)
    else:assert 'sim_camera' not in report


@pytest.mark.parametrize('kind', ['missing', 'ordinary_file'])
def test_missing_camera_directory_rejects_before_staging_or_server(source,monkeypatch,tmp_path,kind):
    def forbidden(*a,**kw):raise AssertionError('No side effect for missing configured directory')
    monkeypatch.setattr(source,'stage_checkout',forbidden)
    monkeypatch.setattr(source,'prepare_wrapper',forbidden)
    configured=tmp_path/kind
    if kind=='ordinary_file': configured.write_text('NOT_A_DIRECTORY')
    with pytest.raises(SystemExit) as error:
        source.main(['--sim','--check','--sim-camera',str(configured)])
    assert error.value.code==2


def test_installed_camera_matches_reviewed_size_and_disconnect_revision():
    # Adopted revision includes the independent operator stream and fresh
    # control-status header. Panel size and disconnect behavior are exercised
    # separately in test_copilot_xr_camera.py. No historical artifact is rewritten.
    assert hashlib.sha256((ROOT/'teleop_assist/copilot/xr_camera.py').read_bytes()).hexdigest() == \
        '0badf3f4cba9d5a591f1f6fae606614347e135b9bbfad3ee95491bcddff59c16'
