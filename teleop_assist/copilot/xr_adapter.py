"""Small pre-IK adapter; no model, DDS, Vuer or simulator imports.

Acquisition means the documented WebXR API pose-read time or a simulator's
sample time, never receipt time or the physical headset sensor exposure.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import fields
from pathlib import Path
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ARM_JOINT_NAMES = tuple(f'{side}_{name}_joint' for side in ('left', 'right') for name in (
    'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
    'wrist_roll', 'wrist_pitch', 'wrist_yaw'))


def host_clock_id():
    boot = Path('/proc/sys/kernel/random/boot_id')
    if not boot.is_file():
        raise RuntimeError('A shared host monotonic clock identifier is required')
    return 'host_monotonic:' + boot.read_text().strip()


def rigid(value):
    a = np.asarray(value, dtype=float)
    return (a.shape == (4, 4) and np.isfinite(a).all()
            and np.allclose(a[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            and np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            and abs(np.linalg.det(a[:3, :3]) - 1) <= 1e-5)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')
    os.replace(temporary, path)


def _read_json(path, maximum_bytes=1_000_000):
    if not path:
        return None
    path = Path(path)
    if not path.is_file() or path.stat().st_size > maximum_bytes:
        return None
    try:
        with path.open() as stream:
            text = stream.read(maximum_bytes + 1)
        if len(text) > maximum_bytes:
            return None
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except (OSError, ValueError, TypeError):
        return None


class XRAdapter:
    """process(left, right, envelope) -> left/right IK input matrices.

    Default shadow keeps both original matrices. A callback is injectable for
    isolated tests; the standard controller and typed mailboxes are loaded only
    on first use. Models are expected to run in another process.
    """
    def __init__(self, config=None, *, callback=None, runtime_guard=None, clock=time.monotonic):
        config = dict(config or {})
        self.config = dict(config.get('xr', config))
        # The staged upstream process runs from a temporary teleop/ directory.
        # Relative paths in the shared product config always belong to this
        # project, never to that cwd or the config file's parent directory.
        for key in ('proposal_mailbox', 'control_mailbox', 'status_path'):
            value = self.config.get(key)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f'{key} must be a nonempty path string')
                path = Path(value).expanduser()
                self.config[key] = str((path if path.is_absolute() else PROJECT/path).resolve())
        self.mode = self.config.get('mode', 'shadow')
        if self.mode not in ('shadow', 'assist'):
            raise ValueError('XR mode must be shadow or assist')
        self.callback = callback
        self.runtime_guard = runtime_guard
        self.clock = clock
        self.controller = None
        # A persisted confirmation is not a new click after restart.
        initial_control = _read_json(self.config.get('control_mailbox')) or {}
        self.last_control_revision = initial_control.get('revision')
        self.last_input_sequence = -1
        self.last_status_write = -math.inf
        self.input_hold_required = False
        self.diagnostics = {'reason': 'not_started', 'mode': self.mode, 'applied': False}

    @classmethod
    def from_config(cls, path):
        content = _read_json(path)
        if content is None:
            raise ValueError('Assist config must be an existing JSON object')
        return cls(content)

    def _operator_issue(self, envelope, now):
        if envelope.get('schema') != 'punktnull.operator_envelope.v1':
            return 'operator_schema_unknown'
        if envelope.get('tracking_valid') is not True:
            return 'tracking_invalid'
        stamp = envelope.get('acquisition_timestamp')
        if not isinstance(stamp, (int, float)) or isinstance(stamp, bool) or not math.isfinite(stamp):
            return 'acquisition_time_unknown'
        if envelope.get('clock_id') != host_clock_id():
            return 'clock_mismatch'
        if envelope.get('sample_kind') not in ('webxr_pose_read', 'simulation_pose_sample'):
            return 'sample_time_provenance_unknown'
        bound = envelope.get('clock_uncertainty_s')
        if not isinstance(bound, (float, int)) or not math.isfinite(bound) or not 0 <= bound <= .02:
            return 'clock_mapping_unbounded'
        if envelope.get('latency_unknown', True):
            return 'latency_unknown'
        if now - stamp > .15 + bound or stamp - now > bound + .01:
            return 'operator_sample_stale_or_future'
        receipt = envelope.get('received_at_host_monotonic')
        if not isinstance(receipt, (float, int)) or not math.isfinite(receipt) or not 0 <= now - receipt <= .15:
            return 'operator_event_stale'
        sequence = envelope.get('sequence')
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < self.last_input_sequence:
            return 'operator_sequence_regressed'
        self.last_input_sequence = sequence
        if envelope.get('backend') != 'unitree_isaac_sim' or envelope.get('dds_domain') != 1:
            return 'simulation_backend_required'
        return None

    def _standard_callback(self, left, right, envelope):
        from .controller import OrientationCopilot
        from .types import OperatorSample, ProposalBatch, GraspCandidate, CopilotRequest
        if self.controller is None:
            self.controller = OrientationCopilot()
        now = self.clock()
        record = _read_json(self.config.get('proposal_mailbox'))
        batch = None
        if record:
            try:
                candidate_keys = {field.name for field in fields(GraspCandidate)}
                batch_keys = {field.name for field in fields(ProposalBatch)} - {'candidates'}
                candidates = tuple(GraspCandidate(**{key: value for key, value in entry.items()
                    if key in candidate_keys}) for entry in record['candidates'])
                batch = ProposalBatch(candidates=candidates, **{key: value for key, value in record.items()
                    if key in batch_keys})
            except (TypeError, ValueError, KeyError):
                batch = None
        control = _read_json(self.config.get('control_mailbox')) or {}
        timestamp = control.get('capture_timestamp')
        fresh = (isinstance(timestamp, (float, int)) and math.isfinite(timestamp)
                 and control.get('clock_id') == host_clock_id() and 0 <= now - timestamp <= .25)
        revision = control.get('revision')
        edge = fresh and revision is not None and revision != self.last_control_revision
        if edge:
            self.last_control_revision = revision
        request = CopilotRequest(
            select_candidate_id=control.get('select_candidate_id') if edge else None,
            confirm=control.get('confirm') is True and edge,
            assist_held=control.get('assist_held') is True and fresh and self.mode == 'assist',
            # Deadman expiry ends held assistance (the controller fades its
            # offset). It must not erase an idle human's selected/confirmed
            # target before the next deliberate button press.
            release=control.get('release') is True and edge,
        )
        sample = OperatorSample(left_base_ee=left, right_base_ee=right,
            acquisition_timestamp=envelope['acquisition_timestamp'],
            clock_id=envelope['clock_id'], tracking_valid=envelope['tracking_valid'])
        return self.controller.step(sample, batch, now=now, clock_id=host_clock_id(),
            calibration_id=self.config.get('calibration_id', ''), request=request)

    def process(self, left4x4, right4x4, operator_envelope):
        left = np.array(left4x4, dtype=float, copy=True)
        right = np.array(right4x4, dtype=float, copy=True)
        now = self.clock()
        envelope = dict(operator_envelope or {})
        issue = 'original_pose_invalid' if not rigid(left) or not rigid(right) else self._operator_issue(envelope, now)
        self.input_hold_required = issue is not None
        if issue is not None:
            # The pre-controller gate must also erase a previous confirmation;
            # otherwise a recovered packet could silently resume old assist.
            self.controller = None
        diagnostics = {'mode': self.mode, 'applied': False, 'reason': issue,
                       'clock_id': host_clock_id(), 'timestamp': now,
                       'operator_envelope': envelope, 'robot_command_sent_by_adapter': False}
        if issue is None:
            try:
                result = (self.callback or self._standard_callback)(left.copy(), right.copy(), envelope)
                if isinstance(result, tuple):
                    proposed_left, proposed_right, snapshot = result
                    hold = bool(snapshot.get('input_hold_required', False))
                else:
                    proposed_left, proposed_right = result.left_base_ee, result.right_base_ee
                    snapshot = result.snapshot
                    hold = bool(result.input_hold_required)
                self.input_hold_required = hold
                diagnostics.update(reason=snapshot.get('reason', 'shadow' if self.mode == 'shadow' else 'controller'), controller=snapshot)
                if self.mode == 'assist' and not hold:
                    proposed_left = np.asarray(proposed_left, float)
                    proposed_right = np.asarray(proposed_right, float)
                    # Product v1 is right orientation only: independently guard
                    # that a callback cannot alter either translation/left arm.
                    if (not rigid(proposed_left) or not rigid(proposed_right)
                            or not np.array_equal(proposed_left, left)
                            or not np.array_equal(proposed_right[:3, 3], right[:3, 3])):
                        diagnostics['reason'] = 'callback_broke_orientation_only_contract'
                        self.input_hold_required = True
                    elif self.runtime_guard is None:
                        diagnostics['reason'] = 'runtime_motion_guard_unavailable'
                        self.input_hold_required = True
                    else:
                        guard = self.runtime_guard(left.copy(), right.copy(),
                            proposed_left.copy(), proposed_right.copy(), envelope)
                        diagnostics['runtime_guard'] = guard
                        if not isinstance(guard, dict) or guard.get('allowed') is not True:
                            diagnostics['reason'] = 'runtime_motion_guard_rejected'
                            self.input_hold_required = True
                        else:
                            right = proposed_right.copy()
                            diagnostics['applied'] = not np.array_equal(right, right4x4)
            except Exception as exc:
                # A failed assistant never injects a partially formed target.
                diagnostics.update(reason='assistant_error', error_type=type(exc).__name__, error=str(exc))
                self.input_hold_required = True
                self.controller = None
        diagnostics['input_hold_required'] = self.input_hold_required
        self.diagnostics = diagnostics
        status_path = self.config.get('status_path')
        if status_path and now - self.last_status_write >= .1:
            try:
                atomic_json(status_path, diagnostics)
                self.last_status_write = now
            except (OSError, TypeError, ValueError):
                self.diagnostics['status_write_failed'] = True
        return left, right
