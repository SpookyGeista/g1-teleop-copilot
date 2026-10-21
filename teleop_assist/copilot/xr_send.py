"""One-shot arm-target handoff for an isolated simulator transport.

This component has no DDS import, transport constructor, thread or launcher.
It is not connected to the original XR controller. Its caller owns current
operator/clutch checks, the complete measured snapshot, and a simulator-only
writer. A rejected write is NOT a hardware hold: real actuators may retain their
last command. No torque controller, real watchdog or full LowCmd approval is
provided here. The only supported arm feedforward policy is explicit zero.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import struct
import threading
import time
import uuid

ARM_SLOTS = tuple(range(15, 29))  # Reviewed G1_29_JointArmIndex, left then right.


def snapshot_digest(state, scene):
    return hashlib.sha256(json.dumps({'robot_state': state, 'scene': scene},
        sort_keys=True, allow_nan=False).encode()).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _q14(value):
    if not isinstance(value, (tuple, list)) or len(value) != 14 or not all(map(_finite, value)):
        raise ValueError('fourteen_finite_joint_angles_required')
    wire = struct.pack('<14f', *value)
    unpacked = struct.unpack('<14f', wire)
    if not all(math.isfinite(x) for x in unpacked) or tuple(value) != unpacked:
        raise ValueError('guard_command_not_exactly_float32')
    return unpacked, wire


def _arm_fields(message):
    """Only the arm fields covered by this transport contract, not full LowCmd."""
    if len(message.motor_cmd) != 35 or message.mode_pr != 0:
        raise ValueError('G1_29_stationary_LowCmd_layout_required')
    fields = []
    for slot in ARM_SLOTS:
        motor = message.motor_cmd[slot]
        fields.append(tuple(getattr(motor, key) for key in ('mode', 'q', 'dq', 'tau', 'kp', 'kd')))
    if any(not _finite(value) for row in fields for value in row):
        raise ValueError('nonfinite_arm_message_fields')
    if any(row[0] != 1 or row[2] != 0 or row[3] != 0 or row[4] < 0 or row[5] < 0 for row in fields):
        raise ValueError('sim_position_command_with_zero_arm_feedforward_required')
    return tuple(fields)


def _message_fields(message):
    """Copy every scalar HG LowCmd field; detect mutation by the CRC callback."""
    motors = tuple(tuple(getattr(motor, key) for key in
        ('mode', 'q', 'dq', 'tau', 'kp', 'kd', 'reserve')) for motor in message.motor_cmd)
    return (message.mode_pr, message.mode_machine, motors, tuple(message.reserve), message.crc)


class SimLowCmdHandoff:
    """Private, copied command -> exactly one synchronous simulator Write call.

    submit() accepts a trusted ActualMotionGuard result, never a UI/JSON motion
    request. It checks its Float32 format and snapshot binding, but cannot prove
    where a caller obtained a dict. Each newer submit/cancel invalidates pending
    work; an attempted write burns the token even if CRC or the writer fails.

    Interlock must freshly check the upstream operator/ownership/clutch/target
    conditions and return exactly True. Snapshot reader must return the actual
    current (state, scene), not relabel old measurements. Both are re-read after
    message/CRC construction. Time is checked again immediately before Write.
    A process can still stall inside a writer/transport; no deadline on actuator
    receipt or guaranteed physical stop is claimed.
    """
    def __init__(self, *, simulation_mode, clock_id, clock=time.monotonic):
        if simulation_mode is not True:
            raise ValueError('simulator_only_handoff')
        if not isinstance(clock_id, str) or not clock_id:
            raise ValueError('named_clock_required')
        self.clock_id, self.clock = clock_id, clock
        self._lock = threading.RLock()
        self._generation = 0
        self._last_sequence = 0
        self._pending = None
        self._instance = uuid.uuid4().hex

    def cancel(self):
        with self._lock:
            self._generation += 1
            self._pending = None

    def _current(self, command):
        now = self.clock()
        if (not _finite(now) or not _finite(command.get('validated_at')) or
                not _finite(command.get('valid_until')) or
                not command['validated_at'] <= now < command['valid_until'] or
                command['valid_until'] - command['validated_at'] > .15 + 1e-9 or
                command.get('clock_id') != self.clock_id):
            raise ValueError('command_clock_or_lease_invalid')
        return now

    def submit(self, checked):
        with self._lock:
            # Even an invalid newer submission revokes an earlier command.
            self._generation += 1
            self._pending = None
            checked = deepcopy(checked)
            if (not isinstance(checked, dict) or checked.get('allowed') is not True or
                    checked.get('hold_required') is not False or
                    checked.get('scope') != 'static_simulation_numerical_precheck' or
                    checked.get('command_format') != 'unitree_float32'):
                raise ValueError('fresh_float32_simulation_guard_required')
            self._current(checked)
            sequence = checked.get('sequence')
            if type(sequence) is not int or sequence <= self._last_sequence:
                raise ValueError('guard_sequence_replayed_or_regressed')
            q14, wire = _q14(checked.get('q14_command'))
            digest = checked.get('snapshot_sha256')
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('guard_snapshot_digest_required')
            self._last_sequence = sequence
            token = f'{self._instance}:{self._generation}:{sequence}'
            self._pending = dict(token=token, generation=self._generation, checked=checked,
                q14=q14, wire_sha256=hashlib.sha256(wire).hexdigest())
            return token

    def publish_once(self, token, *, read_snapshot, interlock, message_factory, crc, write):
        """Record/send once; callbacks must be bound to an isolated simulator.

        message_factory gets an immutable tuple. Its complete message is copied
        before editing CRC. The factory owns non-arm fields and PD policy; those
        are outside the arm-position proof. CRC must return an unsigned32 value.
        write must synchronously accept this exact object and return True. An
        exception/negative result means outcome unknown; there is no retry.
        """
        write_started = False
        checked_at = None
        result = dict(accepted=False, write_attempted=False, write_acknowledged=False,
            delivery_unknown=False, hardware_authorized=False,
            scope='isolated_simulator_arm_target_handoff', reason='not_started')
        with self._lock:
            pending = self._pending
            if pending is None or token != pending['token']:
                # A bad token does not consume another, correctly addressed one.
                result['reason'] = 'missing_replayed_or_superseded_token'
                return result
            self._pending = None  # Burn before any callback, including exceptions.
        try:
            checked = pending['checked']
            self._current(checked)
            if interlock() is not True:
                raise ValueError('upstream_interlock_rejected')
            state, scene = read_snapshot()
            if snapshot_digest(state, scene) != checked['snapshot_sha256']:
                raise ValueError('snapshot_changed_after_guard')
            message = deepcopy(message_factory(pending['q14']))
            before = _arm_fields(message)
            actual_q, wire = _q14([row[1] for row in before])
            if actual_q != pending['q14']:
                raise ValueError('message_changed_guarded_joint_command')
            # No external callback receives our eventual private message before
            # Write. A CRC implementation can retain/mutate only this copy.
            message.crc = 0
            crc_input = deepcopy(message)
            before_crc = _message_fields(crc_input)
            message.crc = crc(crc_input)
            if type(message.crc) is not int or not 0 <= message.crc <= 0xffffffff:
                raise ValueError('invalid_LowCmd_crc')
            if _message_fields(crc_input) != before_crc:
                raise ValueError('message_mutated_during_crc')
            if interlock() is not True:
                raise ValueError('upstream_interlock_revoked_during_encoding')
            state, scene = read_snapshot()
            if snapshot_digest(state, scene) != checked['snapshot_sha256']:
                raise ValueError('snapshot_changed_during_encoding')
            # Every hash/list allocation and reporting step precedes the LAST
            # clock read. No CRC/snapshot/encoding work follows that read.
            result.update(q14=list(actual_q), arm_q14_f32_sha256=hashlib.sha256(wire).hexdigest(),
                valid_until=checked['valid_until'])
            with self._lock:
                if pending['generation'] != self._generation:
                    raise ValueError('command_revoked_or_superseded_during_encoding')
                checked_at = self._current(checked)
                write_started = True
                # Cancellation from another thread waits for this synchronous
                # call; same-thread failure handlers may re-enter cancel().
                # An already begun writer cannot be cancelled by this gate.
                acknowledged = write(message)
            result['write_acknowledged'] = acknowledged is True
            result['accepted'] = acknowledged is True
            result['delivery_unknown'] = acknowledged is not True
            result['reason'] = 'one_checked_sim_write_acknowledged' if acknowledged is True else 'write_not_acknowledged_no_retry'
        except Exception as error:
            result['reason'] = str(error) or type(error).__name__
            result['delivery_unknown'] = write_started
        result['write_attempted'] = write_started
        if checked_at is not None:
            result['checked_before_write_at'] = checked_at
        return result
