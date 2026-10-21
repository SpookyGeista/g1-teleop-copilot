"""Counterexamples for an audit used to distinguish synthetic XR evidence."""
import hashlib
import json

import pytest

from teleop_assist.copilot.audit_trace import InputTrace
from tools.copilot.instrument_xr import SharedXRTelemetry
from tools.copilot.xr_source import OperatorPublisher
from test_copilot_xr_source import data


def test_packet_edits_after_read_cannot_rewrite_the_logged_source_or_button():
    trace = InputTrace('host:clock')
    operator = {'envelope': {'sample_kind': 'simulation_pose_sample'}, 'source_sequence': 17}
    intent = {'revision': 4, 'release': False}
    trace.capture(operator, intent, phase='session_step', captured_at=10.)
    operator['envelope']['sample_kind'] = 'webxr_pose_read'
    intent['release'] = True
    trace.capture(operator, intent, phase='actor_interlock', consumer='physics_substep',
                  substep_index=0, captured_at=10.01)
    evidence = trace.snapshot()
    first, second = evidence['entries']
    assert first['operator_sha256'] != second['operator_sha256']
    assert evidence['packets'][first['operator_sha256']]['envelope']['sample_kind'] == 'simulation_pose_sample'
    assert not evidence['packets'][first['intent_sha256']]['release']
    assert evidence['packets'][second['intent_sha256']]['release']
    for digest, value in evidence['packets'].items():
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
        assert hashlib.sha256(encoded.encode('ascii')).hexdigest() == digest
    evidence['packets'].clear()
    evidence['entries'][0]['phase'] = 'altered'
    assert trace.snapshot()['packets'] and trace.snapshot()['entries'][0]['phase'] == 'session_step'


def test_identical_read_keeps_event_identity_without_inventing_new_publication():
    trace = InputTrace('host:clock')
    for step in range(5):
        trace.capture({'source_sequence': 7}, {'revision': 4}, phase='actor_interlock',
                      consumer='physics_substep', substep_index=step, captured_at=10. + step * .005)
    evidence = trace.snapshot()
    assert len(evidence['packets']) == 2
    assert [r['substep_index'] for r in evidence['entries']] == list(range(5))
    assert len({r['operator_sha256'] for r in evidence['entries']}) == 1


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), object()])
def test_malformed_second_packet_cannot_partly_commit_the_first(bad):
    trace = InputTrace('host:clock')
    before = trace.snapshot()
    with pytest.raises((ValueError, TypeError)):
        trace.capture({'new': True}, {'invalid': bad}, phase='session_step', captured_at=10.)
    assert trace.snapshot() == before


@pytest.mark.parametrize('changes', [dict(captured_at=float('nan')), dict(captured_at=9.),
    dict(phase='other'), dict(phase='actor_interlock'),
    dict(phase='actor_interlock', consumer='physics_substep', substep_index=True),
    dict(phase='actor_interlock', consumer='physics_substep', substep_index=5),
    dict(phase='actor_interlock', consumer='actor_write', substep_index=0)])
def test_invalid_read_metadata_cannot_be_logged_as_an_accepted_read(changes):
    trace = InputTrace('host:clock')
    trace.capture({}, {}, phase='session_step', captured_at=10.)
    before = trace.snapshot()
    with pytest.raises(ValueError):
        trace.capture({}, {}, **(dict(phase='handoff_recheck', captured_at=10.01) | changes))
    assert trace.snapshot() == before


def test_read_and_packet_bounds_reject_without_partial_evidence():
    trace = InputTrace('host:clock')
    for i in range(16):
        trace.capture({}, {}, phase='session_step', captured_at=10. + i)
    before = trace.snapshot()
    with pytest.raises(ValueError, match='read count'):
        trace.capture({}, {}, phase='session_step', captured_at=26.)
    assert trace.snapshot() == before
    trace = InputTrace('host:clock')
    with pytest.raises(ValueError, match='2 MiB'):
        trace.capture({'large': 'x' * (2 * 1024 * 1024)}, {}, phase='session_step', captured_at=10.)
    assert not trace.snapshot()['entries'] and not trace.snapshot()['packets']


@pytest.mark.parametrize('kind,reason,description', [
    ('simulation_pose_sample', 'valid_simulation_pose_sample', 'Synthetic browser event'),
    ('webxr_pose_read', 'valid_webxr_api_sample', 'WebXR API pose-read')])
def test_source_labels_preserve_synthetic_versus_api_pose_provenance(tmp_path, kind, reason, description):
    sample, shared = data()
    # Change the stored event's actual source marker, then rederive the envelope.
    records = shared.read()
    records['arms']['sample_kind'] = kind
    shared.read = lambda: records
    sample.copilot_envelope = shared.envelope()
    packet = OperatorPublisher(tmp_path/'operator.json', clock=lambda: 10.02).publish(sample)
    assert packet['tracking_valid'] and packet['reason'] == reason
    assert packet['envelope']['sample_kind'] == kind
    assert packet['envelope']['timestamp_provenance'].startswith(description)
    assert packet['physical_sensor_capture_timestamp'] is None


def test_unknown_kind_does_not_claim_a_webxr_pose_read():
    telemetry = SharedXRTelemetry()
    assert telemetry.envelope()['timestamp_provenance'].startswith('Unknown pose source')


@pytest.mark.parametrize('kind', [[], {}, True, None])
def test_malformed_kind_remains_invalid_without_crashing_source(tmp_path, kind):
    sample, shared = data()
    records = shared.read()
    records['arms']['sample_kind'] = kind
    shared.read = lambda: records
    sample.copilot_envelope = shared.envelope()
    packet = OperatorPublisher(tmp_path/'operator.json', clock=lambda: 10.02).publish(sample)
    assert not packet['tracking_valid']
    assert packet['reason'] == 'sample_time_provenance_unknown'
    assert packet['envelope']['timestamp_provenance'].startswith('Unknown pose source')
