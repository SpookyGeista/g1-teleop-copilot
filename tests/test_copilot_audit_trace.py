import json
from copy import deepcopy

import pytest

from teleop_assist.copilot.audit_trace import FrozenAuditRows


def test_audit_retains_exact_nested_values_after_source_is_mutated(tmp_path):
    rows = FrozenAuditRows()
    row = dict(command_consumed=True, contact_force_n=0., controller={
        'evidence': [{'allowed': True, 'q14': [0.125]*14}], 'reason': None},
        timing={'step_at': 123.456}, text='Überprüfung')
    expected = deepcopy(row)
    rows.append(row)
    row['controller']['evidence'][0]['q14'][0] = 999.
    row['controller']['reason'] = 'runtime_stopped'
    rows.append(dict(command_consumed=False, contact_force_n=2.))
    rows.write(tmp_path/'rows.json')
    parsed = json.loads((tmp_path/'rows.json').read_text())
    assert parsed == [expected, dict(command_consumed=False, contact_force_n=2.)]
    assert len(rows) == 2 and rows.commands_consumed == 1 and rows.max_contact_force_n == 2.


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), object()])
def test_failed_append_preserves_previous_evidence_and_counts(tmp_path, invalid):
    rows = FrozenAuditRows()
    prior = dict(command_consumed=False, contact_force_n=0.3)
    rows.append(prior)
    with pytest.raises((ValueError, TypeError)):
        rows.append(dict(command_consumed=True, contact_force_n=1., invalid=invalid))
    rows.write(tmp_path/'rows.json')
    assert json.loads((tmp_path/'rows.json').read_text()) == [prior]
    assert len(rows) == 1 and rows.commands_consumed == 0 and rows.max_contact_force_n == .3


def test_empty_or_interrupted_run_writes_valid_atomic_json(tmp_path, monkeypatch):
    rows = FrozenAuditRows()
    path = tmp_path/'rows.json'
    rows.write(path)
    assert json.loads(path.read_text()) == []
    rows.append(dict(command_consumed=True, contact_force_n=0.))
    def interrupted(*_):
        raise OSError('simulated replacement failure')
    monkeypatch.setattr('teleop_assist.copilot.audit_trace.os.replace', interrupted)
    with pytest.raises(OSError):
        rows.write(path)
    assert json.loads(path.read_text()) == []
    assert list(tmp_path.iterdir()) == [path]
    assert len(rows) == 1
