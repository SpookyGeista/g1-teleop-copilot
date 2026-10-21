from copy import deepcopy
import pytest
from teleop_assist.copilot.frame_sync import await_preview_frame
from teleop_assist.copilot.mailbox import atomic_json


class Fixture:
    def __init__(self,path):
        self.path=path;self.now=10.;self.waited=[]
        self.frame=dict(frame_id=3,timestamp=9.99,clock_id='mono',calibration_id='cal')
        self.preview=dict(validation_observation_id='cal:3',capture_timestamp=9.99,
                          clock_id='mono',calibration_id='cal',valid=False)
    def write(self):atomic_json(self.path/'latest_proposals.json',self.preview)
    def sleep(self,seconds):self.waited.append(seconds);self.now+=seconds
    def run(self,**kwargs):return await_preview_frame(self.path,self.frame,clock=lambda:self.now,sleep=self.sleep,**kwargs)


def test_exact_frame_is_only_acknowledged_not_authorized(tmp_path):
    f=Fixture(tmp_path);f.write();before=deepcopy(f.frame)
    result=f.run()
    assert result['ready'] and not result['motion_authorized'] and not result['deadline_renewed']
    assert result['capture_timestamp']==9.99 and f.frame==before and f.waited==[]
    assert f.preview['valid'] is False  # Full validation remains the feed's job.


@pytest.mark.parametrize('key,value',[
    ('validation_observation_id','cal:2'),('capture_timestamp',10.),
    ('clock_id','other'),('calibration_id','other')])
def test_each_binding_mismatch_times_out(tmp_path,key,value):
    f=Fixture(tmp_path);f.preview[key]=value;f.write();result=f.run()
    assert not result['ready'] and result['reason']=='current_frame_preview_timeout'
    assert sum(f.waited)<=.100001 and result['elapsed_s']<=.100001


def test_waits_for_actual_new_publication_and_keeps_source_time(tmp_path):
    f=Fixture(tmp_path)
    def sleep(seconds):
        f.sleep(seconds)
        if f.now>=10.02:f.write()
    result=await_preview_frame(tmp_path,f.frame,clock=lambda:f.now,sleep=sleep)
    assert result['ready'] and .02<=result['elapsed_s']<.03
    assert result['capture_timestamp']==f.frame['timestamp']==9.99


def test_slow_read_cannot_ack_after_deadline(tmp_path,monkeypatch):
    f=Fixture(tmp_path)
    def read(*args):f.now+=.11;return f.preview
    monkeypatch.setattr('teleop_assist.copilot.frame_sync.read_json',read)
    result=f.run();assert not result['ready'] and result['elapsed_s']>.1


def test_cancelled_and_malformed_fail_closed(tmp_path):
    f=Fixture(tmp_path);f.write()
    assert f.run(cancelled=lambda:True)['reason']=='frame_wait_cancelled'
    (tmp_path/'latest_proposals.json').write_text('{')
    assert f.run()['reason']=='preview_mailbox_unreadable'


def test_regressed_clock_rejects_and_frozen_clock_is_bounded(tmp_path):
    f=Fixture(tmp_path)
    def backwards(seconds):f.now-=seconds
    result=await_preview_frame(tmp_path,f.frame,clock=lambda:f.now,sleep=backwards)
    assert not result['ready'] and result['reason']=='frame_wait_clock_invalid'
    waits=[]
    result=await_preview_frame(tmp_path,f.frame,clock=lambda:10.,sleep=waits.append)
    assert not result['ready'] and len(waits)<=52


def test_timeout_cannot_be_raised(tmp_path):
    f=Fixture(tmp_path)
    with pytest.raises(ValueError):f.run(timeout_s=.251)


@pytest.mark.parametrize('payload',['null','[]','7','"text"'])
def test_nonobject_json_is_rejected_by_bounded_mailbox(tmp_path,payload):
    f=Fixture(tmp_path);(tmp_path/'latest_proposals.json').write_text(payload)
    assert f.run()['reason']=='preview_mailbox_unreadable'


@pytest.mark.parametrize('value',[float('nan'),float('inf'),None])
def test_unknown_start_clock_produces_finite_failure_evidence(tmp_path,value):
    import json
    f=Fixture(tmp_path)
    result=await_preview_frame(tmp_path,f.frame,clock=lambda:value)
    assert not result['ready'] and result['reason']=='frame_wait_clock_invalid'
    json.dumps(result,allow_nan=False)


def test_unchanged_old_publication_is_not_reparsed_or_retimestamped(tmp_path,monkeypatch):
    from teleop_assist.copilot.mailbox import read_json
    f=Fixture(tmp_path);f.preview['validation_observation_id']='cal:2';f.write();calls=[]
    def count(*args,**kwargs):calls.append(1);return read_json(*args,**kwargs)
    monkeypatch.setattr('teleop_assist.copilot.frame_sync.read_json',count)
    result=f.run()
    assert not result['ready'] and len(calls)==1
    assert result['capture_timestamp']==9.99


def test_known_current_frame_rejection_stops_wait_without_refresh_or_permission(tmp_path):
    f=Fixture(tmp_path)
    f.preview.update(validation_observation_id='cal:2',capture_timestamp=9.5,
        processed_frame=dict(frame_id=3,capture_timestamp=9.99,clock_id='mono',calibration_id='cal',
                             rejected=True,reason='selected_object_not_observed'))
    f.write();result=f.run()
    assert not result['ready'] and result['reason']=='current_frame_rejected_by_producer'
    assert result['producer_reason']=='selected_object_not_observed'
    assert not f.waited and not result['motion_authorized'] and not result['deadline_renewed']


@pytest.mark.parametrize('key,value',[('frame_id',2),('frame_id',True),('capture_timestamp',9.98),
    ('capture_timestamp',True),('clock_id','other'),('calibration_id','other'),('rejected',False)])
def test_rejection_acknowledgment_must_match_exact_requested_frame(tmp_path,key,value):
    f=Fixture(tmp_path)
    f.preview.update(validation_observation_id='cal:2',capture_timestamp=9.5,
        processed_frame=dict(frame_id=3,capture_timestamp=9.99,clock_id='mono',calibration_id='cal',rejected=True))
    f.preview['processed_frame'][key]=value
    f.write();result=f.run()
    assert not result['ready'] and result['reason']=='current_frame_preview_timeout'
