from copy import deepcopy
import json
from types import SimpleNamespace
import numpy as np
import pytest

from tools.copilot import xr_source
from tools.copilot.instrument_xr import SharedXRTelemetry
from teleop_assist.copilot.mailbox import host_clock_id,read_json


def data(now=10.):
    metadata=dict(schema='punktnull.xr_frame.v1',sequence=8,sample_kind='webxr_pose_read',
        sample_time_s=1.,sample_clock_id='browser:test',host_sample_time_s=now,
        host_clock_id=host_clock_id(),clock_uncertainty_s=.003,clock_sync_age_s=.1,
        sensor_capture_time_s=None,latency_unknown=False,left_tracking_valid=True,right_tracking_valid=True)
    shared=SharedXRTelemetry();pose=np.eye(4).flatten(order='F').tolist()
    shared.record('head',{'camera':{'matrix':pose},'_copilot':dict(metadata,sample_kind='camera_pose_callback')},now=now)
    shared.record('hand',{'left':pose*25,'right':pose*25,'leftState':{},'rightState':{},'_copilot':metadata},now=now)
    return SimpleNamespace(left_wrist_pose=np.eye(4),right_wrist_pose=np.eye(4),copilot_envelope=shared.envelope()),shared


def test_publication_sequence_never_renews_the_source_sample_or_expiry(tmp_path):
    now=[10.02];sample,_=data()
    publisher=xr_source.OperatorPublisher(tmp_path/'operator.json',clock=lambda:now[0])
    first=publisher.publish(sample);now[0]=10.1;second=publisher.publish(sample)
    assert second['publish_sequence']==first['publish_sequence']+1
    assert second['source_sequence']==first['source_sequence']==8
    assert second['acquisition_timestamp']==first['acquisition_timestamp']==10.
    assert second['expires_at']==first['expires_at']==10.15
    assert xr_source.validate_operator_packet(second,now=10.1) is None
    now[0]=10.16;expired=publisher.publish(sample)
    assert not expired['tracking_valid'] and expired['expires_at'] is None
    assert read_json(tmp_path/'operator.json')['publish_sequence']==3


def test_explicit_simulator_fixture_keeps_its_own_acquisition_provenance(tmp_path):
    sample,_=data();publisher=xr_source.OperatorPublisher(tmp_path/'operator.json',clock=lambda:10.02)
    packet=publisher.publish(sample)
    packet['envelope']['sample_kind']='simulation_pose_sample'
    packet['command_source']='scripted_test_fixture_NO_HEADSET'
    assert xr_source.validate_operator_packet(packet,now=10.02) is None
    packet['envelope']['sample_kind']='fresh_receipt_is_not_a_capture'
    assert xr_source.validate_operator_packet(packet,now=10.02)=='sample_time_provenance_unknown'


@pytest.mark.parametrize('broken',['unknown_head','missing_head','unknown_browser','old_head','bad_mapping'])
def test_unknown_head_or_browser_time_never_becomes_valid_from_publication(tmp_path,broken):
    sample,shared=data()
    if broken=='unknown_head':
        shared.record('head',{'camera':{'matrix':np.eye(4).flatten(order='F').tolist()}},now=10.02)
        sample.copilot_envelope=shared.envelope()
    elif broken=='missing_head':
        sample.copilot_envelope['tracking_valid']=False
    elif broken=='unknown_browser':sample.copilot_envelope.update(acquisition_timestamp=None,latency_unknown=True)
    elif broken=='old_head':sample.copilot_envelope['acquisition_timestamp']=9.
    else:sample.copilot_envelope['clock_uncertainty_s']=.2
    publisher=xr_source.OperatorPublisher(tmp_path/'operator.json',clock=lambda:10.02)
    packet=publisher.publish(sample)
    assert not packet['tracking_valid'] and packet['expires_at'] is None
    assert xr_source.validate_operator_packet(packet,now=10.02) is not None


@pytest.mark.parametrize('field,value',[('schema','bad'),('tracking_valid',False),
    ('target_frame','unverified_world'),('publish_sequence',True),('published_at',11.),
    ('clock_id','browser'),('source_sequence',9),('expires_at',999.),
    ('left_base_ee',[[1]]),('right_base_ee',None),('envelope',[]),('acquisition_timestamp',0.),
    ('published_at',10**500),('right_base_ee',[[10**500]*4]*4)])
def test_consumer_rejects_malformed_or_inconsistent_packets(tmp_path,field,value):
    sample,_=data();publisher=xr_source.OperatorPublisher(tmp_path/'operator.json',clock=lambda:10.02)
    packet=publisher.publish(sample);packet[field]=value
    assert xr_source.validate_operator_packet(packet,now=10.02) is not None


def test_nonfinite_metadata_and_pose_overwrite_old_valid_packet_with_invalid(tmp_path):
    sample,_=data();publisher=xr_source.OperatorPublisher(tmp_path/'operator.json',clock=lambda:10.02)
    assert publisher.publish(sample)['tracking_valid']
    sample.copilot_envelope['acquisition_timestamp']=float('nan')
    sample.left_wrist_pose[0,0]=float('nan')
    packet=publisher.publish(sample)
    assert not packet['tracking_valid'] and packet['left_base_ee'] is None
    assert not read_json(tmp_path/'operator.json')['tracking_valid']


def test_source_sequence_regression_and_shutdown_require_invalid_packet(tmp_path):
    sample,_=data();publisher=xr_source.OperatorPublisher(tmp_path/'operator.json',clock=lambda:10.02)
    publisher.publish(sample);sample.copilot_envelope['sequence']=7
    for record in sample.copilot_envelope['raw_event_metadata'].values():record['sequence']=7
    assert publisher.publish(sample)['reason']=='source_sequence_regressed'
    stopped=publisher.publish(reason='source_stopped')
    assert not stopped['tracking_valid'] and stopped['acquisition_timestamp'] is None


def test_new_head_event_counts_as_new_normalized_source(tmp_path):
    sample,shared=data();publisher=xr_source.OperatorPublisher(tmp_path/'operator.json',clock=lambda:10.02)
    first=publisher.publish(sample)
    metadata=dict(shared.read()['head'],sequence=9,host_sample_time_s=10.01)
    shared.record('head',{'camera':{'matrix':np.eye(4).flatten(order='F').tolist()},'_copilot':metadata},now=10.01)
    sample.copilot_envelope=shared.envelope();sample.left_wrist_pose[0,3]=.001
    second=publisher.publish(sample)
    assert second['tracking_valid'] and second['source_sequence']==9
    assert second['source_event_sequences']=={'head':9,'arms':8}
    assert second['acquisition_timestamp']==first['acquisition_timestamp']  # older arm still limits freshness


def test_browser_epoch_change_requires_invalid_sample_before_fresh_counter(tmp_path):
    sample,_=data();publisher=xr_source.OperatorPublisher(tmp_path/'operator.json',clock=lambda:10.02)
    first=publisher.publish(sample)
    for record in sample.copilot_envelope['raw_event_metadata'].values():
        record.update(sequence=1,sample_clock_id='browser:new')
    changed=publisher.publish(sample)
    assert not changed['tracking_valid'] and changed['reason']=='source_browser_changed'
    restarted=publisher.publish(sample)
    assert restarted['tracking_valid'] and restarted['source_sequence']==1
    assert restarted['source_event_clock_id']!=first['source_event_clock_id']
    assert restarted['source_instance_id']==first['source_instance_id']


def test_check_stages_without_constructing_vuer_or_publishing_input(tmp_path,monkeypatch,capsys):
    def forbidden(*args,**kwargs):raise AssertionError('Offline check must not start Vuer')
    monkeypatch.setattr(xr_source,'prepare_wrapper',forbidden)
    output=tmp_path/'operator.json';stage=tmp_path/'staged'
    assert xr_source.main(['--sim','--check','--output',str(output),'--stage',str(stage)])==0
    report=json.loads(capsys.readouterr().out)
    assert not report['server_started'] and not report['dds_imported']
    assert not report['live_headset_verified'] and report['host']=='127.0.0.1'
    assert (stage/'staging_manifest.json').exists() and not output.exists()


def test_original_vuer_factory_cannot_override_explicit_bind_address(tmp_path,monkeypatch):
    received=[]
    module=SimpleNamespace(__file__=str(tmp_path/'teleop/televuer/src/televuer/televuer.py'),
        Vuer=lambda *args,**kwargs:received.append(kwargs))
    class Wrapper:
        def __init__(self,**kwargs):
            assert kwargs['display_mode']=='pass-through' and not kwargs['zmq'] and not kwargs['webrtc']
            module.Vuer(host='0.0.0.0')
    monkeypatch.setattr(xr_source.importlib,'import_module',lambda name:module if name=='televuer.televuer' else SimpleNamespace(TeleVuerWrapper=Wrapper))
    monkeypatch.setattr(xr_source,'install_instrumentation',lambda *_:None)
    # Public-root/HTTP containment is exercised by test_copilot_xr_public_routes.
    # This fixture isolates the original constructor's host override only.
    monkeypatch.setattr(xr_source,'prepare_public_web',lambda *_:(tmp_path/'public-client',tmp_path/'public-static'))
    monkeypatch.setattr(xr_source,'install_public_file_routes',lambda *_:None)
    xr_source.prepare_wrapper(tmp_path,host='127.0.0.1',port=8012,use_hand_tracking=True,cert_file='cert',key_file='key')
    assert received==[{'host':'127.0.0.1','port':8012,'static_root':str(tmp_path/'public-static')}]


@pytest.mark.parametrize('host,authority',[('127.0.0.1','127.0.0.1:8823'),
    ('192.168.1.20','192.168.1.20:8823'),('::1','[::1]:8823'),
    ('2001:db8::1','[2001:db8::1]:8823'),('0.0.0.0','127.0.0.1:8823'),('::','[::1]:8823')])
def test_explicit_browser_url_preserves_secure_socket_host_and_port(host,authority):
    from urllib.parse import urlsplit,parse_qs
    endpoint=xr_source.browser_endpoint(host,8823);url=urlsplit(endpoint['browser_url'])
    assert url.scheme=='https' and url.netloc==authority and url.port==8823
    assert endpoint['websocket_url']==f'wss://{authority}/'
    assert parse_qs(url.query)=={'ws':[endpoint['websocket_url']]}


@pytest.mark.parametrize('host,port',[('localhost',8012),('127.0.0.1',True),
    ('127.0.0.1',0),('127.0.0.1',65536),('fe80::1%eth0',8012)])
def test_invalid_or_browser_unsupported_endpoint_rejected(host,port):
    with pytest.raises(ValueError):xr_source.browser_endpoint(host,port)


def test_server_cli_emits_machine_readable_full_url_without_starting_real_server(tmp_path,monkeypatch,capsys):
    cert=tmp_path/'cert';key=tmp_path/'key';cert.touch();key.touch();closed=[]
    wrapper=SimpleNamespace(tvuer=SimpleNamespace(process=SimpleNamespace(is_alive=lambda:False)),
        close=lambda:closed.append(True))
    monkeypatch.setattr(xr_source,'stage_checkout',lambda *args,**kwargs:None)
    monkeypatch.setattr(xr_source,'prepare_wrapper',lambda *args,**kwargs:wrapper)
    times=iter([0.,1.]);monkeypatch.setattr(xr_source.time,'monotonic',lambda:next(times))
    assert xr_source.main(['--sim','--host','::1','--port','8823','--duration','.001',
        '--stage',str(tmp_path/'stage'),'--output',str(tmp_path/'operator.json'),
        '--cert-file',str(cert),'--key-file',str(key)])==0
    message=json.loads(capsys.readouterr().out)
    assert message['schema']=='punktnull.xr_input_endpoint.v1'
    assert message['browser_url']==xr_source.browser_endpoint('::1',8823)['browser_url']
    assert message['live_headset_verified'] is False and message['robot_transport_started'] is False
    assert closed==[True] and read_json(tmp_path/'operator.json')['reason']=='source_stopped'
