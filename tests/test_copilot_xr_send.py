"""Final simulator-only transport gate adversaries; no SDK or DDS imports."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from teleop_assist.copilot.xr_send import SimLowCmdHandoff, snapshot_digest


class Fixture:
    def __init__(self):
        self.now = 10.
        self.state, self.scene = {'source': 'test_only', 'q43': [0.]*43}, {'source': 'test_scene'}
        self.gate = SimLowCmdHandoff(simulation_mode=True, clock_id='test', clock=lambda: self.now)
        self.checked = dict(allowed=True, hold_required=False, sequence=1,
            scope='static_simulation_numerical_precheck', command_format='unitree_float32',
            validated_at=10., valid_until=10.1, clock_id='test', q14_command=[.125]*14,
            snapshot_sha256=snapshot_digest(self.state, self.scene))
        self.writes = []
        self.token = self.gate.submit(self.checked)

    def message(self, q):
        motors = [NS(mode=1, q=0., dq=0., tau=0., kp=40., kd=1.5, reserve=0) for _ in range(35)]
        for i, value in enumerate(q): motors[15+i].q = value
        self.last_template = NS(mode_pr=0, mode_machine=0, motor_cmd=motors, reserve=[0]*4, crc=0)
        return self.last_template

    def write(self, message):
        self.writes.append(deepcopy(message))
        return True

    def publish(self, **changes):
        args = dict(read_snapshot=lambda: (self.state, self.scene), interlock=lambda: True,
            message_factory=self.message, crc=lambda message: 1234, write=self.write)
        args.update(changes)
        return self.gate.publish_once(self.token, **args)


def test_exact_private_q14_is_copied_and_sent_once_with_no_original_IK():
    f = Fixture()
    f.checked['q14_command'][0] = 123.
    result = f.publish()
    assert result['accepted'] and not result['hardware_authorized']
    assert [m.q for m in f.writes[0].motor_cmd[15:29]] == [.125]*14
    assert f.writes[0].crc == 1234 and f.last_template.crc == 0
    assert not f.publish()['write_attempted']
    assert len(f.writes) == 1


@pytest.mark.parametrize('change', [dict(allowed=False), dict(hold_required=True),
    dict(scope='hardware'), dict(command_format='float64'), dict(clock_id='other'),
    dict(sequence=True), dict(sequence=1), dict(validated_at=True), dict(valid_until=11.),
    dict(q14_command=[.1]*14), dict(q14_command=[float('nan')]*14),
    dict(q14_command=[0.]*13), dict(q14_command=[True]*14), dict(snapshot_sha256='')])
def test_invalid_new_submit_revokes_previous_command(change):
    f = Fixture()
    changed = dict(f.checked, sequence=2)
    changed.update(change)
    with pytest.raises((ValueError, OverflowError)):
        f.gate.submit(changed)
    assert not f.publish()['write_attempted'] and not f.writes


def test_superseded_token_cannot_consume_new_private_command():
    f = Fixture()
    new = f.gate.submit(dict(f.checked, sequence=2, q14_command=[.25]*14))
    assert not f.publish()['write_attempted']
    f.token = new
    assert f.publish()['q14'] == [.25]*14


@pytest.mark.parametrize('phase', ['initial', 'crc', 'snapshot_after_crc', 'interlock_after_crc'])
def test_expiry_at_each_blocking_boundary_prevents_Write(phase):
    f = Fixture(); calls = {'snapshot': 0, 'interlock': 0}
    if phase == 'initial': f.now = 10.1
    def crc(message):
        if phase == 'crc': f.now = 10.1
        return 1234
    def snapshot():
        calls['snapshot'] += 1
        if phase == 'snapshot_after_crc' and calls['snapshot'] == 2: f.now = 10.1
        return f.state, f.scene
    def interlock():
        calls['interlock'] += 1
        if phase == 'interlock_after_crc' and calls['interlock'] == 2: f.now = 10.1
        return True
    result = f.publish(crc=crc, read_snapshot=snapshot, interlock=interlock)
    assert not result['write_attempted'] and not f.writes
    assert result['reason'] == 'command_clock_or_lease_invalid'
    f.now = 10.05
    assert not f.publish()['write_attempted']  # Failed attempt cannot be retried.


@pytest.mark.parametrize('what', ['state', 'scene', 'interlock', 'cancel', 'submit', 'q', 'tau', 'kp', 'crc'])
def test_changes_during_crc_cannot_cross_the_final_Write_gate(what):
    f = Fixture(); permitted = [True]
    def crc(message):
        if what == 'state': f.state['q43'][42] += .01
        if what == 'scene': f.scene['new_obstacle'] = True
        if what == 'interlock': permitted[0] = False
        if what == 'cancel': f.gate.cancel()
        if what == 'submit': f.gate.submit(dict(f.checked, sequence=2))
        if what in ('q', 'tau', 'kp'): setattr(message.motor_cmd[15], what, 9.)
        return -1 if what == 'crc' else 1234
    result = f.publish(crc=crc, interlock=lambda: permitted[0])
    assert not result['write_attempted'] and not f.writes, result


@pytest.mark.parametrize('what', ['q', 'tau', 'dq', 'mode', 'kp', 'layout', 'mode_pr', 'nonfinite'])
def test_message_factory_cannot_transform_guarded_arm_command(what):
    f = Fixture()
    def factory(q):
        msg = f.message(q)
        if what in ('q', 'tau', 'dq'): setattr(msg.motor_cmd[15], what, .5)
        if what == 'mode': msg.motor_cmd[15].mode = 0
        if what == 'kp': msg.motor_cmd[15].kp = -1.
        if what == 'layout': msg.motor_cmd.pop()
        if what == 'mode_pr': msg.mode_pr = 1
        if what == 'nonfinite': msg.motor_cmd[15].kd = float('nan')
        return msg
    assert not f.publish(message_factory=factory)['write_attempted']
    assert not f.writes


@pytest.mark.parametrize('outcome', ['raises', 'false', 'none'])
def test_Write_failure_is_unknown_delivery_and_never_retried(outcome):
    f = Fixture()
    def write(message):
        f.writes.append(deepcopy(message))
        if outcome == 'raises': raise RuntimeError('transport_ambiguous_failure')
        return False if outcome == 'false' else None
    result = f.publish(write=write)
    assert result['write_attempted'] and result['delivery_unknown'] and not result['accepted']
    assert not f.publish()['write_attempted'] and len(f.writes) == 1


def test_stall_inside_writer_is_explicitly_outside_the_local_deadline_guarantee():
    f = Fixture()
    def write(message):
        f.now = 100.
        return f.write(message)
    result = f.publish(write=write)
    assert result['accepted'] and result['checked_before_write_at'] == 10.
    assert result['valid_until'] == 10.1 and f.now > result['valid_until']


@pytest.mark.parametrize('field', ['q', 'tau', 'crc', 'mode_machine', 'body', 'reserve'])
def test_crc_callback_retained_reference_cannot_mutate_the_eventual_private_message(field):
    f = Fixture(); retained = []; count = [0]
    def crc(message):
        retained.append(message)
        return 1234
    def snapshot():
        count[0] += 1
        if count[0] == 2:
            if field in ('q', 'tau'): setattr(retained[0].motor_cmd[15], field, .25)
            if field == 'body': retained[0].motor_cmd[0].q = .25
            if field in ('crc', 'mode_machine'): setattr(retained[0], field, 99)
            if field == 'reserve': retained[0].reserve[0] = 99
        return f.state, f.scene
    result = f.publish(crc=crc, read_snapshot=snapshot)
    assert result['accepted']
    sent = f.writes[0]
    assert sent.motor_cmd[15].q == .125 and sent.motor_cmd[15].tau == 0
    assert sent.motor_cmd[0].q == 0 and sent.crc == 1234
    assert sent.mode_machine == 0 and sent.reserve == [0]*4


def test_slow_final_reporting_hash_is_before_last_lease_check(monkeypatch):
    from teleop_assist.copilot import xr_send
    f = Fixture(); original = xr_send.hashlib.sha256
    def slow_arm_hash(data):
        if len(data) == 14*4: f.now = 10.2
        return original(data)
    monkeypatch.setattr(xr_send.hashlib, 'sha256', slow_arm_hash)
    result = f.publish()
    assert not result['write_attempted'] and not f.writes
    assert result['reason'] == 'command_clock_or_lease_invalid'


def test_synchronous_writer_can_cancel_reentrantly_without_deadlock():
    f = Fixture()
    def write(message):
        f.gate.cancel()
        return f.write(message)
    assert f.publish(write=write)['accepted']
    assert not f.publish()['write_attempted']


@pytest.mark.parametrize('mode', [False, None, 1, 'sim'])
def test_constructor_cannot_select_hardware(mode):
    with pytest.raises(ValueError, match='simulator_only'):
        SimLowCmdHandoff(simulation_mode=mode, clock_id='test')
