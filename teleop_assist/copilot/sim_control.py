"""DDS-free session core for explicitly clutched, isolated simulator control.

Inputs and UI actions retain their source times and sequence numbers. A separate
runtime supplies an authorized ProposalBatch; preview-only service data is not
upgraded here. The consume methods expose the exact guarded q14 command once, after
checking its lease and the still-matching measured state/scene. No downstream IK
or robot transport exists in this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from tools.copilot.xr_source import validate_operator_packet
from .controller import OrientationCopilot, rigid
from .guard import JOINT_REFERENCE_SCHEMA, command_is_current
from .kinematics import ARM_JOINT_NAMES, arm_vector
from .types import CopilotRequest, GraspCandidate, OperatorSample, PreviewSelection, ProposalBatch
from .vr_buttons import controller_event_time_issue, has_vr_time_provenance


G1_JOINT_NAMES = frozenset(ARM_JOINT_NAMES) | frozenset(
    f'{side}_{joint}_joint' for side in ('left', 'right')
    for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll',
                  'hand_index_0', 'hand_index_1', 'hand_middle_0', 'hand_middle_1',
                  'hand_thumb_0', 'hand_thumb_1', 'hand_thumb_2')) | frozenset(
    ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'))


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SimCommandReceipt:
    """Immutable result of one successful simulator command consumption.

    This records a completed local check, not transport/actuator acceptance.
    ``valid_until`` is the private session deadline after every source recheck;
    it can be earlier than the untouched guard's own expiry in ``guard_json``.
    A downstream consumer must use this shorter deadline, its own final clock
    check, and current stop/state checks. JSON fields are immutable strings;
    parsing one creates a caller-owned copy and cannot change this receipt.

    Original and consumption snapshots retain their distinct acquisition times.
    Their digests do not assert byte identity when a fresh read has a new time.
    Only the existing complete-state/geometry continuity tests relate them.
    This Python value is not authentication for an untrusted caller or network.
    """
    q14: tuple[float, ...]
    valid_until: float
    clock_id: str
    token: str
    guard_sequence: int
    command_format: str
    guard_snapshot_sha256: str
    consumed_snapshot_sha256: str
    guard_json: str
    guarded_state_json: str
    guarded_scene_json: str
    measured_state_json: str
    scene_json: str
    scope: str = 'isolated_PhysX_consumed_arm_command'


_RECEIPT_FIELDS = tuple(field.name for field in fields(SimCommandReceipt))


def _receipt_binding(receipt):
    return tuple(getattr(receipt, name) for name in _RECEIPT_FIELDS)


class SimReceiptReplan:
    """Opaque local decision to hold a burned receipt without losing UI intent.

    There are no transferable fields. Only the issuing session's exact-object
    registry grants the decision meaning; constructing/copying/serializing it
    cannot create a valid cancellation. This is not a motion capability.
    """
    __slots__ = ()

    def __new__(cls):
        raise TypeError('Receipt replan decisions are issued only by the local session')

    def __reduce_ex__(self, protocol):
        raise TypeError('Receipt replan decisions cannot be serialized or copied')

    def __init_subclass__(cls, **kwargs):
        raise TypeError('Receipt replan decisions cannot be subclassed')


@dataclass(frozen=True)
class SimUIIntent:
    revision: int
    capture_timestamp: float | None
    clock_id: str | None
    client_id: str | None
    clutch: bool = False
    start: bool = False
    select_candidate_id: str | None = None
    confirm: bool = False
    assist_held: bool = False
    release: bool = False
    expires_at: float | None = None
    source_kind: str | None = None
    source_event_sequence: int | None = None
    clock_uncertainty_s: float | None = None
    received_at_host_monotonic: float | None = None

    @classmethod
    def from_packet(cls, packet):
        if not isinstance(packet, dict):
            raise ValueError('UI packet must be a mapping')
        required = ('revision', 'capture_timestamp', 'clock_id', 'client_id')
        if any(key not in packet for key in required):
            raise ValueError('UI provenance fields are missing')
        has_vr_time_provenance(packet)
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in packet.items() if key in names})


@dataclass(frozen=True)
class SimControlConfig:
    ui_max_age_s: float = .25
    state_max_age_s: float = .15
    scene_max_age_s: float = .15
    future_tolerance_s: float = .01
    max_step_s: float = .2
    static_position_tolerance_m: float = .001
    static_rotation_tolerance_rad: float = .005
    state_consumption_tolerance_rad: float = 1e-8

    def __post_init__(self):
        values = list(vars(self).values())
        if (not all(_number(value) for value in values) or
                not 0 < self.ui_max_age_s <= .25 or not 0 < self.state_max_age_s <= .15 or
                not 0 < self.scene_max_age_s <= .15 or not 0 <= self.future_tolerance_s <= .01 or
                not 0 < self.max_step_s <= .2 or not 0 <= self.static_position_tolerance_m <= .001 or
                not 0 <= self.static_rotation_tolerance_rad <= .005 or
                not 0 <= self.state_consumption_tolerance_rad <= 1e-6):
            raise ValueError('Invalid simulator control configuration')


class SimControlSession:
    """Synchronous core with an injectable MotionProcess-compatible CPU worker."""

    ui_action_policy = 'state_only_with_fresh_control_v1'

    def __init__(self, motion_worker, *, config=None, clock=time.monotonic):
        if not callable(getattr(motion_worker, 'call', None)):
            raise ValueError('Inject a worker with call(operation, **payload)')
        self.worker, self.config, self.clock = motion_worker, config or SimControlConfig(), clock
        self.controller = OrientationCopilot()
        self._committed_offset = np.eye(3)
        self._session_id = uuid.uuid4().hex
        self._generation = self._offer_sequence = 0
        self._last_now = self._clock_id = self._calibration_id = None
        self._ui_revision = None
        self._ui_signature = self._ui_client = None
        self._ui_seen_revision = self._ui_seen_signature = None
        self._publish_sequence = self._source_sequence = None
        self._source_signature = self._publication_signature = None
        self._last_acquisition = None
        self._source_epoch = None
        self._anchors = self._anchor_scene = self._anchor_base = None
        self._clutch_joint_reference = None
        self._target_id = None
        self._pending = None
        self._inflight_receipt = None
        self._receipt_replan = None
        self._resume_after_receipt_replan = None
        self._cancelled_token = None
        self._preview_selection_audit = None
        self._vr_start_required = False
        self._teleop_active = False
        self.status = self._status('clutch_required', now=None, clock_id=None)

    @property
    def clutched(self):
        return self._anchors is not None

    def _anchor_both_arms(self, raw, measured, measured_state, calibration_id):
        """Pair each current controller pose with its measured robot TCP.

        No robot movement is requested by anchoring. In particular, moving the
        controllers while stopped must not accumulate a command for AX start.
        """
        self._anchors = tuple((a.copy(), b.copy()) for a, b in zip(raw, measured))
        self._clutch_joint_reference = dict(schema=JOINT_REFERENCE_SCHEMA,
            source=measured_state['source'],
            acquisition_timestamp=measured_state['acquisition_timestamp'],
            clock_id=measured_state['clock_id'], calibration_id=calibration_id,
            joint_names=list(ARM_JOINT_NAMES),
            q14=[measured_state['joint_positions'][name] for name in ARM_JOINT_NAMES])

    def _status(self, reason, *, now, clock_id, controller=None, allowed=False, available=False):
        snapshot = deepcopy(controller or {})
        snapshot.update(clutched=self.clutched, clutch_required=not self.clutched,
                        clutch_available=bool(available), last_ui_revision=self._ui_revision,
                        input_hold_required=not allowed,
                        teleop_active=bool(self.clutched and self._teleop_active),
                        teleop_start_required=bool(self.clutched and self._vr_start_required
                                                   and not self._teleop_active))
        snapshot.setdefault('confirmed', False)
        snapshot.setdefault('confirmed_candidate_id', None)
        if self._preview_selection_audit is not None:
            snapshot['preview_selection'] = deepcopy(self._preview_selection_audit)
        return {'schema': 'punktnull.sim_runtime.v1', 'mode': 'isolated_PhysX',
                'ui_action_policy': self.ui_action_policy,
                'timestamp': now, 'clock_id': clock_id, 'reason': reason,
                'allowed': bool(allowed), 'input_hold_required': not allowed,
                'snapshot': snapshot, 'robot_transport_started': False,
                'real_robot_authorized': False, 'downstream_ik_allowed': False,
                'q14_command': None, 'command_pending': bool(allowed),
                'finger_control': 'measured_q43_values_unchanged', 'human_source': None,
                'assistance_applied': False, 'controller_reason': reason}

    def hold(self, reason='external_hold', *, now=None, clock_id=None):
        """Immediately revoke pending commands, clutch and confirmation."""
        self._generation += 1
        self._inflight_receipt = None
        self._receipt_replan = self._resume_after_receipt_replan = None
        self._pending = self._anchors = self._anchor_scene = self._anchor_base = None
        self._clutch_joint_reference = None
        self._target_id = None
        self._preview_selection_audit = None
        self._vr_start_required = False
        self._teleop_active = False
        self.controller = OrientationCopilot()
        self._committed_offset = np.eye(3)
        self.status = self._status(reason, now=now, clock_id=clock_id)
        return deepcopy(self.status)

    def _fresh(self, mapping, name, now, clock_id, maximum):
        if not isinstance(mapping, dict) or not mapping.get('source'):
            raise ValueError(name + '_source_unknown')
        stamp = mapping.get('acquisition_timestamp')
        if not _number(stamp) or mapping.get('clock_id') != clock_id:
            raise ValueError(name + '_clock_or_timestamp_invalid')
        if not -self.config.future_tolerance_s <= now - stamp <= maximum:
            raise ValueError(name + '_stale_or_future')
        return float(stamp + maximum)

    def _state(self, state, now, clock_id):
        deadline = self._fresh(state, 'robot_state', now, clock_id, self.config.state_max_age_s)
        named = state.get('joint_positions')
        if state.get('complete') is not True or not isinstance(named, dict) or set(named) != G1_JOINT_NAMES:
            raise ValueError('complete_named_q43_required')
        if not all(_number(value) for value in named.values()):
            raise ValueError('q43_nonfinite_or_invalid')
        base = rigid(state.get('base_in_world'), 'measured_base')
        bodies = state.get('observed_body_world_poses')
        if not isinstance(bodies, dict):
            raise ValueError('measured_body_poses_required')
        tcp = np.eye(4)
        tcp[0, 3] = .05
        measured = []
        for side in ('left', 'right'):
            wrist = rigid(bodies.get(side + '_wrist_yaw_link'), 'measured_wrist')
            measured.append(np.linalg.inv(base) @ wrist @ tcp)
        return deadline, base, tuple(measured)

    def _scene(self, scene, now, clock_id):
        deadline = self._fresh(scene, 'scene', now, clock_id, self.config.scene_max_age_s)
        if (not isinstance(scene.get('scene_id'), str) or not scene['scene_id'] or
                scene.get('geometry_complete') is not True or scene.get('static') is not True):
            raise ValueError('known_complete_static_scene_required')
        obstacles = scene.get('obstacles')
        if not isinstance(obstacles, list) or not obstacles:
            raise ValueError('scene_geometry_missing')
        ids = [item.get('id') for item in obstacles if isinstance(item, dict)]
        if len(ids) != len(obstacles) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('scene_obstacle_identity_invalid')
        _digest(scene)  # Never permit nonfinite geometry to bypass comparisons.
        return deadline

    def _pose_matches(self, a, b, *, position=None, rotation=None):
        a, b = rigid(a), rigid(b)
        return bool(np.linalg.norm(a[:3, 3] - b[:3, 3]) <= (
            self.config.static_position_tolerance_m if position is None else position) and
            Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude() <= (
                self.config.static_rotation_tolerance_rad if rotation is None else rotation))

    def _body_poses_match(self, before, current):
        """Exact consume tolerance with one established SciPy batch conversion.

        rigid() deliberately permits tiny nonorthogonality. Keep SciPy's
        projection and angle metric; a trace shortcut is not equivalent near
        the 1e-8 boundary. Preserve the old first-failure order: a later invalid
        body must not replace an earlier rotation mismatch's False result.
        """
        relative = []
        failure = None
        position_mismatch = False
        for name in before:
            try:
                a, b = rigid(before[name]), rigid(current[name])
                if not np.linalg.norm(a[:3, 3] - b[:3, 3]) <= 1e-8:
                    position_mismatch = True
                    break
                relative.append(a[:3, :3] @ b[:3, :3].T)
            except (TypeError, ValueError, OverflowError, KeyError) as error:
                failure = error
                break
        if relative and not np.all(Rotation.from_matrix(np.asarray(relative)).magnitude() <= 1e-8):
            return False
        if failure is not None:
            raise failure
        return not position_mismatch

    def _scene_matches(self, a, b, *, exact=False):
        if a['scene_id'] != b['scene_id'] or a.get('allowed_static_contacts', []) != b.get('allowed_static_contacts', []):
            return False
        aa, bb = ({item['id']: item for item in scene['obstacles']} for scene in (a, b))
        if set(aa) != set(bb):
            return False
        for name in aa:
            x, y = deepcopy(aa[name]), deepcopy(bb[name])
            pose_x, pose_y = x.pop('world_pose', None), y.pop('world_pose', None)
            if x != y:
                return False
            if (pose_x is None) != (pose_y is None):
                return False
            if pose_x is not None and not self._pose_matches(pose_x, pose_y,
                    position=1e-8 if exact else None, rotation=1e-8 if exact else None):
                return False
        return True

    def _operator_sequence(self, packet):
        epoch = (packet.get('source_instance_id'), packet.get('source_event_clock_id'))
        if any(value is not None and (not isinstance(value, str) or not value or len(value) > 256)
               for value in epoch):
            raise ValueError('operator_source_epoch_invalid')
        if self._source_epoch is not None and epoch != self._source_epoch:
            self._source_epoch = epoch
            self._publish_sequence = self._source_sequence = self._last_acquisition = None
            self._source_signature = self._publication_signature = None
            raise ValueError('operator_source_epoch_changed_reclutch_required')
        self._source_epoch = epoch
        publish, source = packet['publish_sequence'], packet['source_sequence']
        source_signature = _digest({key: packet[key] for key in (
            'source_sequence', 'acquisition_timestamp', 'clock_id', 'left_base_ee', 'right_base_ee', 'envelope')})
        publication_signature = _digest(packet)
        if self._publish_sequence is not None:
            if publish < self._publish_sequence or source < self._source_sequence:
                raise ValueError('operator_sequence_regressed')
            if publish == self._publish_sequence and publication_signature != self._publication_signature:
                raise ValueError('operator_publication_reused_with_changed_content')
            if source == self._source_sequence and source_signature != self._source_signature:
                raise ValueError('operator_source_sequence_reused_with_changed_content')
            if packet['acquisition_timestamp'] < self._last_acquisition:
                raise ValueError('operator_acquisition_regressed')
        self._publish_sequence, self._source_sequence = publish, source
        self._source_signature, self._publication_signature = source_signature, publication_signature
        self._last_acquisition = packet['acquisition_timestamp']

    def _ui(self, value, now, clock_id, *, consume_revision=True):
        ui = value if isinstance(value, SimUIIntent) else SimUIIntent.from_packet(value)
        if type(ui.revision) is not int or ui.revision < 1:
            raise ValueError('ui_revision_invalid')
        signature = _digest(vars(ui))
        edge = self._ui_revision is not None and ui.revision > self._ui_revision
        if self._ui_seen_revision is not None and ui.revision < self._ui_seen_revision:
            raise ValueError('ui_revision_regressed')
        if ui.revision == self._ui_seen_revision and signature != self._ui_seen_signature:
            raise ValueError('ui_revision_reused_with_changed_content')
        # Consume each revision even if its action must subsequently be denied.
        # Old clicks never become fresh actions when another input recovers.
        self._ui_seen_revision, self._ui_seen_signature = ui.revision, signature
        if consume_revision:
            self._ui_revision, self._ui_signature = ui.revision, signature
        if not isinstance(ui.client_id, str) or not 0 < len(ui.client_id) <= 256:
            raise ValueError('ui_client_identity_required')
        if self._ui_client is not None and ui.client_id != self._ui_client:
            if consume_revision:
                self._ui_client = ui.client_id
            raise ValueError('ui_owner_changed_reclutch_required')
        if consume_revision:
            self._ui_client = ui.client_id
        if not _number(ui.capture_timestamp) or ui.clock_id != clock_id:
            raise ValueError('ui_clock_or_timestamp_invalid')
        bounded = ui.clock_uncertainty_s is not None or ui.received_at_host_monotonic is not None
        if bounded:
            if (ui.source_kind != 'webxr_controller_event'
                    or type(ui.source_event_sequence) is not int or ui.source_event_sequence < 0):
                raise ValueError('ui_vr_source_provenance_invalid')
            issue = controller_event_time_issue(ui.capture_timestamp, ui.expires_at, now=now,
                clock_uncertainty_s=ui.clock_uncertainty_s,
                received_at_host_monotonic=ui.received_at_host_monotonic,
                future_tolerance_s=self.config.future_tolerance_s)
            if issue: raise ValueError('ui_' + issue)
            if now - ui.capture_timestamp > self.config.ui_max_age_s:
                raise ValueError('ui_stale_or_future')
        elif not 0 <= now - ui.capture_timestamp <= self.config.ui_max_age_s:
            raise ValueError('ui_stale_or_future')
        deadline = ui.capture_timestamp + self.config.ui_max_age_s
        if ui.expires_at is not None:
            if not _number(ui.expires_at) or not ui.capture_timestamp < ui.expires_at <= deadline:
                raise ValueError('ui_expiry_invalid')
            deadline = ui.expires_at
        if now >= deadline:
            raise ValueError('ui_expired')
        if not all(type(getattr(ui, name)) is bool for name in ('clutch', 'start', 'confirm', 'assist_held', 'release')):
            raise ValueError('ui_action_flags_invalid')
        if ui.select_candidate_id is not None and (not isinstance(ui.select_candidate_id, str) or
                                                   not 0 < len(ui.select_candidate_id) <= 256):
            raise ValueError('ui_candidate_identity_invalid')
        if sum((ui.clutch, ui.start, ui.confirm, ui.release, ui.select_candidate_id is not None)) > 1:
            raise ValueError('ui_actions_must_be_separate')
        if ui.clutch and ui.assist_held:
            raise ValueError('clutch_and_assistance_must_be_separate')
        return ui, edge, float(deadline)

    def _clutch_inputs_available(self, packet, state, scene, proposals, supplied, now, clock_id, calibration_id):
        """Read-only availability independent of the deliberately expired UI lease."""
        try:
            if validate_operator_packet(packet, now=now, clock_id=clock_id):
                return False
            _, _, measured = self._state(state, now, clock_id)
            self._scene(scene, now, clock_id)
            if any(not self._pose_matches(a, b, position=1e-5, rotation=1e-5) for a, b in zip(measured, supplied)):
                return False
            return bool(isinstance(proposals, ProposalBatch) and proposals.candidates and
                        all(isinstance(c, GraspCandidate) for c in proposals.candidates) and
                        not self.controller._batch_issue(proposals, now, clock_id, calibration_id))
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
            return False

    def step(self, operator_packet, ui_intent, measured_state, scene, proposals, *,
             measured_left_base_ee, measured_right_base_ee, now, clock_id, calibration_id):
        # A merely consumed receipt is not proof that a target was installed.
        # Superseding an unapplied receipt must not accumulate its correction.
        if self._inflight_receipt is not None and not self._inflight_receipt['installed']:
            self._committed_offset = self._inflight_receipt['prior_committed_offset'].copy()
        self._inflight_receipt = None
        self._receipt_replan = None
        self._pending = None  # Every newer step supersedes an unconsumed offer.
        started = self.clock()
        source = operator_packet.get('command_source') if isinstance(operator_packet, dict) else None
        neutral_ack = None
        try:
            if not _number(now) or not isinstance(clock_id, str) or not clock_id:
                raise ValueError('session_clock_invalid')
            clock_changed = self._clock_id is not None and clock_id != self._clock_id
            bad_interval = self._last_now is not None and not 0 < now - self._last_now <= self.config.max_step_s
            # A running but idle consumer still advances its local tick time.
            # Expired UI heartbeats must not create an artificial long pause
            # that rejects the user's eventual fresh clutch click.
            self._clock_id, self._last_now = clock_id, now
            ui, edge, ui_deadline = self._ui(ui_intent, now, clock_id)
            if clock_changed:
                raise ValueError('session_clock_changed')
            if bad_interval:
                raise ValueError('session_step_interval_invalid')
            if not isinstance(calibration_id, str) or not calibration_id:
                raise ValueError('calibration_required')
            if self._calibration_id is not None and calibration_id != self._calibration_id:
                self._calibration_id = calibration_id
                raise ValueError('calibration_changed_reclutch_required')
            self._calibration_id = calibration_id
            if edge and ui.release:
                raise ValueError('explicit_release_reclutch_required')
            issue = validate_operator_packet(operator_packet, now=now, clock_id=clock_id)
            if issue:
                raise ValueError(issue)
            self._operator_sequence(operator_packet)
            state_deadline, base, measured = self._state(measured_state, now, clock_id)
            scene_deadline = self._scene(scene, now, clock_id)
            supplied = (measured_left_base_ee, measured_right_base_ee)
            if any(not self._pose_matches(a, b, position=1e-5, rotation=1e-5) for a, b in zip(measured, supplied)):
                raise ValueError('measured_EE_must_match_observed_original_50mm_TCP')
            if (not self.clutched and self.controller._selected is None
                    and self.controller._confirmed is None and not self.controller._preview_pending
                    and self._preview_selection_audit is None
                    and not any((ui.clutch, ui.start, ui.confirm, ui.assist_held, ui.release,
                                 ui.select_candidate_id is not None))):
                # This validates only the original neutral input transaction.
                # Live proposals may legitimately be unavailable until a later
                # explicit selection. Never use proposal validity to bootstrap
                # the reconnect handshake, or turn this ACK into motion rights.
                neutral_ack = dict(schema='punktnull.neutral_source_ui_ack.v1',
                    motion_authorized=False, ui_revision=ui.revision, ui_client_id=ui.client_id,
                    ui_signature=_digest(vars(ui_intent) if isinstance(ui_intent, SimUIIntent) else ui_intent),
                    ui_capture_timestamp=ui.capture_timestamp, ui_expires_at=ui_deadline,
                    ui_source_event_sequence=ui.source_event_sequence,
                    source_instance_id=operator_packet.get('source_instance_id'),
                    source_event_clock_id=operator_packet.get('source_event_clock_id'),
                    source_sequence=operator_packet['source_sequence'],
                    operator_acquisition_timestamp=operator_packet['acquisition_timestamp'],
                    clock_id=clock_id, calibration_id=calibration_id,
                    valid_until=float(min(ui_deadline, operator_packet['expires_at'],
                                          state_deadline, scene_deadline)))
            if type(proposals) is PreviewSelection:
                # A preview is a separate non-motion transition, never a batch
                # with forged validation flags. The current UI must still be
                # exactly the one that staged this choice before any work.
                if (not edge or ui.select_candidate_id != proposals.candidate_id
                        or ui.revision != proposals.ui_revision or ui.client_id != proposals.ui_client_id
                        or _digest(vars(ui)) != proposals.ui_signature
                        or ui.clutch or ui.start or ui.confirm or ui.release or ui.assist_held):
                    raise ValueError('preview_requires_matching_fresh_select_edge')
                staged = OrientationCopilot()
                preview_snapshot = staged.select_preview(proposals,
                    OperatorSample(measured[0], measured[1], operator_packet['acquisition_timestamp'],
                                   operator_packet['clock_id'], True),
                    now=now, clock_id=clock_id, calibration_id=calibration_id)
                deadline = min(ui_deadline, operator_packet['expires_at'], state_deadline,
                               scene_deadline, proposals.capture_timestamp + .25)
                # Revoke any old clutch, installed offset and private receipts
                # before publishing the new, inert choice. No old action is
                # replayed when a later endpoint result becomes available.
                audit = dict(
                    schema='punktnull.preview_selection_commit.v1',
                    ui_revision=ui.revision, ui_signature=proposals.ui_signature,
                    ui_client_id=ui.client_id, target_revision=proposals.target_revision,
                    target_signature=proposals.target_signature, target_id=proposals.target_id,
                    candidate_id=proposals.candidate_id, model_source_id=proposals.model_source_id,
                    ui_capture_timestamp=ui.capture_timestamp, ui_expires_at=ui_deadline,
                    operator_expires_at=operator_packet['expires_at'], state_expires_at=state_deadline,
                    scene_expires_at=scene_deadline, preview_expires_at=proposals.capture_timestamp+.25,
                    commit_deadline=deadline,
                    motion_authorized=False)
                self.hold('preview_selected_awaiting_endpoint', now=now, clock_id=clock_id)
                staged_status = self._status('preview_selected_awaiting_endpoint', now=now,
                    clock_id=clock_id, controller=preview_snapshot, available=False)
                staged_status['snapshot']['preview_selection'] = deepcopy(audit)
                staged_status.update(human_source=source, operator_source_sequence=operator_packet['source_sequence'],
                    operator_publish_sequence=operator_packet['publish_sequence'],
                    operator_acquisition_timestamp=operator_packet['acquisition_timestamp'],
                    operator_expires_at=operator_packet['expires_at'], ui_expires_at=ui_deadline,
                    proposal_observation_id=None, proposal_capture_timestamp=None,
                    preview_observation_id=proposals.observation_id,
                    preview_capture_timestamp=proposals.capture_timestamp,
                    controller_reason='preview_selected_awaiting_endpoint')
                # This injected clock IS the caller's named monotonic clock.
                # Take the actual end sample, not caller_now plus a duration:
                # even the microseconds before method entry count at expiry.
                finished = self.clock()
                if not _number(finished) or finished < started or finished < now:
                    raise ValueError('preview_commit_clock_regressed_or_unknown')
                if finished >= deadline:
                    raise ValueError('preview_selection_lease_expired_during_transition')
                audit['committed_at'] = finished
                staged_status['snapshot']['preview_selection']['committed_at'] = finished
                staged_status['timestamp'] = finished
                self.controller = staged
                self._preview_selection_audit = audit
                self.status = staged_status
                return deepcopy(self.status)
            if not isinstance(proposals, ProposalBatch):
                raise ValueError('authorized_proposal_batch_required')
            if not isinstance(proposals.candidates, (tuple, list)) or any(
                    not isinstance(c, GraspCandidate) for c in proposals.candidates):
                raise ValueError('proposal_candidate_contract_invalid')
            proposal_issue = self.controller._batch_issue(proposals, now, clock_id, calibration_id)
            if proposal_issue:
                raise ValueError('proposal_' + proposal_issue)
            if not proposals.candidates:
                raise ValueError('proposal_candidates_lost')
            if self.clutched:
                if proposals.target_id != self._target_id:
                    raise ValueError('proposal_target_changed')
                if not self._pose_matches(base, self._anchor_base):
                    raise ValueError('robot_base_changed')
                if not self._scene_matches(scene, self._anchor_scene):
                    raise ValueError('scene_geometry_changed')
            raw = tuple(rigid(operator_packet[key], key) for key in ('left_base_ee', 'right_base_ee'))
            # Separate translational and rotational relative maps preserve
            # world/base-axis manual translations, instead of rotating them by
            # the wrist's orientation at clutch time.
            mapped = []
            starting = bool(edge and ui.start and self.clutched and not self._teleop_active)
            for index in range(2):
                pose = measured[index].copy()
                if self.clutched and not starting:
                    origin, robot_origin = self._anchors[index]
                    pose[:3, 3] = robot_origin[:3, 3] + raw[index][:3, 3] - origin[:3, 3]
                    # Each source passed the unchanged rigid-transform gate.
                    # WebXR Float32 roundoff can accumulate across these three
                    # valid rotations and make det(R) exceed that gate. Compose
                    # rotations on SO(3), rather than carry scale/shear into IK.
                    pose[:3, :3] = (Rotation.from_matrix(raw[index][:3, :3]) *
                        Rotation.from_matrix(origin[:3, :3]).inv() *
                        Rotation.from_matrix(robot_origin[:3, :3])).as_matrix()
                mapped.append(pose)
            previous_choice = self.status.get('snapshot', {}).get('confirmed_candidate_id') or self.status.get('snapshot', {}).get('selected_candidate_id')
            request = CopilotRequest(ui.select_candidate_id if edge else None, ui.confirm and edge,
                                     ui.assist_held and self.clutched and not (edge and ui.clutch), False)
            if edge and ui.clutch and self.clutched:
                # Rebase clears an existing correction/confirmation. A later
                # deliberate confirmation is required; no old click is replayed.
                self.controller = OrientationCopilot()
                previous_choice = None
                mapped = [pose.copy() for pose in measured]
            if (edge and ui.clutch) or starting:
                self._committed_offset = np.eye(3)
                self._resume_after_receipt_replan = None
            # New offers start at the last offset actually handed off by
            # consume(), not a discarded or superseded proposal. Keep the
            # controller's source/clock/opposition/selection history intact.
            self.controller._offset = self._committed_offset.copy()
            outcome = self.controller.step(OperatorSample(mapped[0], mapped[1],
                operator_packet['acquisition_timestamp'], operator_packet['clock_id'], True),
                proposals, now=now, clock_id=clock_id, calibration_id=calibration_id, request=request)
            if outcome.input_hold_required:
                raise ValueError(outcome.reason)
            if previous_choice and outcome.reason.startswith(('candidate_', 'confirmed_candidate_', 'selected_candidate_',
                                                               'selection_observation_', 'target_', 'calibration_')):
                raise ValueError('proposal_permission_lost:' + outcome.reason)
            if edge and ui.clutch:
                # Keep the exact measured posture and its original provenance.
                # It is only an IK hypothesis; fresh full-state validation is
                # mandatory every time, including after physical settling.
                self._anchor_both_arms(raw, measured, measured_state, calibration_id)
                self._anchor_scene, self._anchor_base = deepcopy(scene), base.copy()
                self._target_id = proposals.target_id
                self._clutch_source_sequence = operator_packet['source_sequence']
                self._vr_start_required = ui.source_kind == 'webxr_controller_event'
                self._teleop_active = not self._vr_start_required
                self._generation += 1
                self.status = self._status('clutched_waiting_for_next_XR_sample', now=now, clock_id=clock_id,
                                           controller=outcome.snapshot, available=True)
            elif not self.clutched:
                if edge and ui.start:
                    raise ValueError('calibration_required_before_start')
                self.status = self._status('clutch_required', now=now, clock_id=clock_id,
                                           controller=outcome.snapshot, available=True)
            elif edge and ui.start:
                if not self._vr_start_required:
                    raise ValueError('explicit_start_only_valid_for_vr_calibration')
                if starting:
                    # Fresh, already checked poses on BOTH sides are the new
                    # origins. AX itself remains a state-only event, and the
                    # next sample still goes through the full motion guard.
                    self._anchor_both_arms(raw, measured, measured_state, calibration_id)
                self._teleop_active = not self._teleop_active
                self._clutch_source_sequence = operator_packet['source_sequence']
                self._generation += 1
                self.status = self._status(
                    'teleop_started_waiting_for_next_XR_sample' if self._teleop_active else 'teleop_paused',
                    now=now, clock_id=clock_id, controller=outcome.snapshot, available=True)
            elif ui.clutch or ui.start or ui.confirm or ui.select_candidate_id is not None:
                # One-shot UI decisions acknowledge state only. Their original
                # short event lease is not a fresh motion heartbeat after a
                # prior tick has already waited for a slow guard. Never run IK
                # or create a receipt from a repeated old action packet.
                finished = self._time_after_work(now, started)
                deadline = min(ui_deadline, operator_packet['expires_at'], state_deadline,
                    scene_deadline, proposals.capture_timestamp + self.controller.config.target_max_age_s)
                if finished >= deadline:
                    raise ValueError('ui_action_lease_expired_during_transition')
                if edge:
                    self._clutch_source_sequence = operator_packet['source_sequence']
                self.status = self._status('ui_action_waiting_for_fresh_control_sample',
                    now=finished, clock_id=clock_id, controller=outcome.snapshot, available=True)
            elif self._vr_start_required and not self._teleop_active:
                self.status = self._status('teleop_start_required', now=now, clock_id=clock_id,
                                           controller=outcome.snapshot, available=True)
            elif operator_packet['source_sequence'] <= self._clutch_source_sequence:
                self.status = self._status('waiting_for_new_XR_sample_after_clutch', now=now, clock_id=clock_id,
                                           controller=outcome.snapshot, available=True)
            elif (self._resume_after_receipt_replan is not None and
                  (operator_packet['source_sequence'] <= self._resume_after_receipt_replan[0] or
                   ui.revision <= self._resume_after_receipt_replan[1])):
                self.status = self._status('waiting_for_fresh_source_and_UI_after_receipt_replan',
                    now=now, clock_id=clock_id, controller=outcome.snapshot, available=True)
            else:
                self._resume_after_receipt_replan = None
                offered_offset = self.controller._offset.copy()
                checked = self.worker.call('validate', left_base_ee=outcome.left_base_ee.tolist(),
                    right_base_ee=outcome.right_base_ee.tolist(), robot_state=deepcopy(measured_state),
                    scene=deepcopy(scene), acquisition_timestamp=operator_packet['acquisition_timestamp'],
                    clock_id=operator_packet['clock_id'], tracking_valid=True,
                    joint_reference=deepcopy(self._clutch_joint_reference))
                finished = now + max(0., self.clock() - started)
                if not isinstance(checked, dict) or checked.get('scope') != 'static_simulation_numerical_precheck':
                    raise ValueError('guard_result_scope_invalid')
                if not _number(checked.get('validated_at')) or not _number(checked.get('valid_until')):
                    raise ValueError('guard_result_timestamps_invalid')
                if not command_is_current(checked, now=finished, clock_id=clock_id):
                    raise ValueError('guard_rejected_or_expired:' + str(checked.get('reason')))
                q14 = arm_vector(checked.get('q14_command'), 'guard_q14_command')
                deadline = min(checked['valid_until'], operator_packet['expires_at'], ui_deadline,
                               state_deadline, scene_deadline,
                               proposals.capture_timestamp + self.controller.config.target_max_age_s)
                if finished >= deadline:
                    raise ValueError('source_lease_expired_during_guard')
                self._offer_sequence += 1
                token = f'{self._session_id}:{self._generation}:{self._offer_sequence}'
                self._pending = {'token': token, 'generation': self._generation, 'q14': q14.copy(),
                                 'guard': deepcopy(checked), 'deadline': float(deadline),
                                 'assist_held': ui.assist_held,
                                 'assist_offset': offered_offset,
                                 'state': deepcopy(measured_state), 'scene': deepcopy(scene)}
                self.status = self._status('guarded_sim_command_available', now=finished, clock_id=clock_id,
                                           controller=outcome.snapshot, allowed=True, available=True)
                self.status.update(command_token=token, expires_at=float(deadline),
                                   mapped_left_base_ee=mapped[0].tolist(), mapped_right_base_ee=mapped[1].tolist(),
                                   assisted_right_base_ee=outcome.right_base_ee.tolist(),
                                   assistance_applied=outcome.applied,
                                   guard={'sequence': checked.get('sequence'), 'reason': checked.get('reason'),
                                          'elapsed_s': checked.get('elapsed_s'), 'valid_until': checked['valid_until']})
            self.status.update(human_source=source, operator_source_sequence=operator_packet['source_sequence'],
                               operator_publish_sequence=operator_packet['publish_sequence'],
                               operator_acquisition_timestamp=operator_packet['acquisition_timestamp'],
                               operator_expires_at=operator_packet['expires_at'], ui_expires_at=ui_deadline,
                               proposal_observation_id=proposals.observation_id,
                               proposal_capture_timestamp=proposals.capture_timestamp,
                               controller_reason=outcome.reason,
                               assistance_applied=outcome.applied and self.status['allowed'])
            return deepcopy(self.status)
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError, RuntimeError, OSError, TimeoutError) as exc:
            result = self.hold(str(exc), now=now if _number(now) else None, clock_id=clock_id)
            # An expired/missing UI mailbox is an expected idle state. The next
            # explicit click can reacquire a lease; do not hide that button.
            if str(exc).startswith(('ui_', 'UI ')) and _number(now):
                available = self._clutch_inputs_available(operator_packet, measured_state, scene, proposals,
                    (measured_left_base_ee, measured_right_base_ee), now, clock_id, calibration_id)
                self.status['snapshot']['clutch_available'] = available
                result['snapshot']['clutch_available'] = available
            self.status['human_source'] = source
            result['human_source'] = source
            if neutral_ack is not None and str(exc).startswith(('proposal_', 'authorized_proposal_')):
                # Prepare reset/status copies before the real end-clock sample.
                # A proposal HOLD stays a HOLD, with separate non-motion proof
                # that its neutral UI/source transaction actually crossed the
                # owner, epoch, state, scene and EE gates while still current.
                ack_status, ack_result = deepcopy(neutral_ack), deepcopy(neutral_ack)
                try:
                    finished = self.clock()
                except (TypeError, ValueError, OverflowError, RuntimeError, OSError):
                    finished = None
                if (_number(finished) and _number(started) and finished >= started and finished >= now
                        and finished < neutral_ack['valid_until']):
                    ack_status['checked_at'] = ack_result['checked_at'] = finished
                    self.status['neutral_source_ui_ack'] = ack_status
                    result['neutral_source_ui_ack'] = ack_result
            return result

    def _cancel_pending(self, reason, *, now):
        """Cancel an undelivered command without discarding confirmed intent."""
        if self._pending is not None:
            self._cancelled_token = self._pending['token']
        self._pending = None
        self.status.update(allowed=False, command_pending=False, input_hold_required=True,
                           assistance_applied=False, reason=reason, timestamp=now)
        self.status['snapshot']['input_hold_required'] = True
        return False

    def _time_after_work(self, caller_now, started):
        """Keep the caller's named clock, accounting for time spent checking."""
        finished = self.clock()
        if not _number(started) or not _number(finished) or finished < started:
            raise ValueError('local_elapsed_clock_invalid')
        effective_now = caller_now + (finished-started)
        if not _number(effective_now):
            raise ValueError('local_elapsed_clock_invalid')
        return effective_now

    def recheck_inputs(self, operator_packet, ui_packet, *, now, clock_id, execution_reserve_s=0.):
        """Re-read stop/lease signals after blocking validation, before consume.

        This never authorizes a command, runs IK, acknowledges a UI edge, or
        renews the original source lease. New valid XR samples may arrive while
        the CPU worker runs; the pending command retains its earlier source and
        deadline. New one-shot UI actions and hold-button changes cancel only
        that pending command; step() subsequently processes the edge or bounded
        return to manual. Invalid data, changed owner/source epoch and explicit
        release revoke the clutch. False always means no handoff this cycle.
        """
        started = self.clock()
        if self._pending is None:
            return False
        try:
            if not _number(execution_reserve_s) or not 0 <= execution_reserve_s <= .1:
                raise ValueError('handoff_execution_reserve_invalid')
            pending = self._pending
            if (not _number(now) or clock_id != self._clock_id or now >= pending['deadline'] or
                    not command_is_current(pending['guard'], now=now, clock_id=clock_id)):
                raise ValueError('handoff_source_lease_expired_or_clock_invalid')
            ui, edge, ui_deadline = self._ui(ui_packet, now, clock_id, consume_revision=False)
            if ui.release:
                raise ValueError('handoff_explicit_release_reclutch_required')
            issue = validate_operator_packet(operator_packet, now=now, clock_id=clock_id)
            if issue:
                raise ValueError('handoff_' + issue)
            self._operator_sequence(operator_packet)
            if edge and (ui.clutch or ui.start or ui.confirm or ui.select_candidate_id is not None):
                return self._cancel_pending('handoff_ui_action_changed_pending_replan', now=now)
            if ui.assist_held != pending['assist_held']:
                return self._cancel_pending('handoff_assist_button_changed_pending_replan', now=now)
            # Only shorten: a newer publication/heartbeat never renews the
            # acquisition, state, proposal or UI lease used by the worker.
            pending['deadline'] = min(pending['deadline'], ui_deadline, operator_packet['expires_at'])
            if now >= pending['deadline']:
                raise ValueError('handoff_source_lease_expired')
            self.status.update(handoff_inputs_checked_at=now, expires_at=pending['deadline'])
            finished = self._time_after_work(now, started)
            if finished >= pending['deadline'] or not command_is_current(pending['guard'], now=finished, clock_id=clock_id):
                raise ValueError('handoff_source_lease_expired_during_recheck')
            if pending['deadline']-finished <= execution_reserve_s:
                # The source/UI checks above must still be fresh and valid.
                # Burn this undelivered offer, never extend its lease. A later
                # sample and UI revision must acquire a new guard result.
                self._resume_after_receipt_replan=(operator_packet['source_sequence'],ui.revision)
                return self._cancel_pending('handoff_execution_budget_pending_replan', now=finished)
            self.status['handoff_inputs_checked_at'] = finished
            return True
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError, RuntimeError) as exc:
            self.hold(str(exc), now=now if _number(now) else None, clock_id=clock_id)
            return False

    def consume(self, offer, *, now, clock_id, measured_state, scene):
        """Return the exact guarded q14 once, or None and revoke the clutch.

        Call immediately before setting the simulator target, with freshly read
        state/scene. Re-read source/UI and call recheck_inputs() immediately
        before this method when the worker call may have spanned new input.
        """
        return self._consume(offer, now=now, clock_id=clock_id,
            measured_state=measured_state, scene=scene, as_receipt=False)

    def consume_receipt(self, offer, *, now, clock_id, measured_state, scene):
        """Return a SimCommandReceipt once, sharing consume()'s private gate.

        The public offer contributes only its token and allowed flag. Its q14,
        expiry, clock or advertised guard data cannot override private evidence.
        Receipt construction is included in the final elapsed-time lease check.
        A downstream refusal must hold/reset the session, except for the exact
        locally registered valid-UI replan decision; there is no command retry.
        """
        return self._consume(offer, now=now, clock_id=clock_id,
            measured_state=measured_state, scene=scene, as_receipt=True)

    def _command_receipt(self, pending, measured_state, scene):
        guard = pending['guard']
        guarded_digest = _digest({'robot_state':pending['state'], 'scene':pending['scene']})
        if guard.get('snapshot_sha256') != guarded_digest:
            raise ValueError('receipt_original_guard_snapshot_binding_invalid')
        sequence = guard.get('sequence')
        command_format = guard.get('command_format', 'float64')
        if type(sequence) is not int or sequence < 1 or command_format not in ('float64','unitree_float32'):
            raise ValueError('receipt_guard_metadata_invalid')
        q14 = tuple(float(value) for value in pending['q14'])
        if tuple(arm_vector(guard.get('q14_command'))) != q14:
            raise ValueError('receipt_private_guard_command_mismatch')
        encode = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
        return SimCommandReceipt(q14=q14, valid_until=float(pending['deadline']),
            clock_id=guard['clock_id'], token=pending['token'], guard_sequence=sequence,
            command_format=command_format, guard_snapshot_sha256=guarded_digest,
            consumed_snapshot_sha256=_digest({'robot_state':measured_state, 'scene':scene}),
            guard_json=encode(guard), guarded_state_json=encode(pending['state']),
            guarded_scene_json=encode(pending['scene']), measured_state_json=encode(measured_state),
            scene_json=encode(scene))

    def recheck_receipt_inputs(self, receipt, operator_packet, ui_packet, *, now, clock_id,
                               require_uninstalled=False, execution_reserve_s=None):
        """Recheck live stop/source signals after encoding a consumed command.

        Only the exact locally consumed, still-inflight receipt can be checked.
        This method cannot create another command, acknowledge an action, run
        IK or renew a deadline. The returned immutable receipt may have a
        shorter deadline. A valid new UI action/button state returns an opaque
        SimReceiptReplan instead, burning the old receipt. The actor must consume
        that exact decision and install a measured hold; it cannot write q14.
        Every step()/hold() revokes this inflight context and unconsumed decision.
        A transport must still enforce one-shot delivery and check this deadline
        immediately before applying its decoded target to the isolated actor.

        A code-owned execution reserve may reject an otherwise current,
        UNINSTALLED receipt. This separate replan category holds measured joints
        and requires new source/UI samples and a new guard. It never extends the
        receipt deadline or supplies a worst-case execution-time guarantee.
        """
        started = self.clock()
        try:
            inflight = self._inflight_receipt
            if (inflight is None or type(receipt) is not SimCommandReceipt or
                    _receipt_binding(receipt) != inflight['binding'] or
                    inflight['generation'] != self._generation or not self.clutched):
                raise ValueError('receipt_missing_changed_or_superseded')
            if type(require_uninstalled) is not bool or (require_uninstalled and inflight['installed']):
                raise ValueError('receipt_already_installed_before_new_write')
            if execution_reserve_s is not None and (not _number(execution_reserve_s)
                    or execution_reserve_s <= 0 or require_uninstalled is not True):
                raise ValueError('receipt_execution_reserve_invalid')
            if (not _number(now) or clock_id != self._clock_id or now >= inflight['deadline'] or
                    not command_is_current(inflight['guard_lifetime'], now=now, clock_id=clock_id)):
                raise ValueError('receipt_source_lease_expired_or_clock_invalid')
            ui, edge, ui_deadline = self._ui(ui_packet, now, clock_id, consume_revision=False)
            if ui.release:
                raise ValueError('receipt_explicit_release_reclutch_required')
            issue = validate_operator_packet(operator_packet, now=now, clock_id=clock_id)
            if issue:
                raise ValueError('receipt_' + issue)
            self._operator_sequence(operator_packet)
            replan_reason = None
            if edge and (ui.clutch or ui.start or ui.confirm or ui.select_candidate_id is not None):
                replan_reason = 'receipt_ui_action_changed_replan'
            elif ui.assist_held != inflight['assist_held']:
                replan_reason = 'receipt_assist_button_changed_replan'
            deadline = min(inflight['deadline'], ui_deadline, operator_packet['expires_at'])
            updated = replace(receipt, valid_until=float(deadline))
            next_inflight = dict(inflight, deadline=float(deadline), binding=_receipt_binding(updated))
            self.status.update(receipt_inputs_checked_at=now, expires_at=float(deadline))
            # Keep the legacy UI-change path unchanged. Only the new capacity
            # decision needs a directly sampled end time in the local clock;
            # the caller's earlier timestamp cannot increase its headroom.
            budget_check = execution_reserve_s is not None and replan_reason is None
            budget = (dict(policy='preinstall_private_receipt_reserve_v1',
                minimum_remaining_s=float(execution_reserve_s),
                receipt_valid_until=float(deadline),
                latest_admission_at=float(deadline-execution_reserve_s),
                checked_at=None, remaining_s=None, admitted=False,
                motion_authorized=False, worst_case_execution_time_guaranteed=False)
                if budget_check else None)
            if budget_check:
                finished = self.clock()
                if (not _number(started) or not _number(finished) or
                        started < now or finished < started):
                    raise ValueError('receipt_execution_budget_clock_invalid')
            else:
                finished = self._time_after_work(now, started)
            if (self._inflight_receipt is not inflight or inflight['generation'] != self._generation or
                    not self.clutched):
                raise ValueError('receipt_revoked_during_recheck')
            if finished >= deadline or not command_is_current(inflight['guard_lifetime'], now=finished, clock_id=clock_id):
                raise ValueError('receipt_source_lease_expired_during_recheck')
            if budget_check:
                # Compare against the same representable cutoff in both tests
                # and production. Equality admits; expiry itself never does.
                budget.update(checked_at=finished, remaining_s=float(deadline-finished),
                    admitted=finished <= budget['latest_admission_at'])
                self.status['execution_budget'] = budget
                if not budget['admitted']:
                    replan_reason = 'receipt_execution_budget_replan'
            if replan_reason is not None:
                # All original and newly read leases still apply. No edge is
                # acknowledged; the next step processes it under fresh inputs.
                decision = object.__new__(SimReceiptReplan)
                self._generation += 1
                self._inflight_receipt = self._pending = None
                self._cancelled_token = receipt.token
                if not inflight['installed']:
                    self._committed_offset = inflight['prior_committed_offset'].copy()
                self.controller._offset = self._committed_offset.copy()
                self._resume_after_receipt_replan = (operator_packet['source_sequence'], ui.revision)
                self._receipt_replan = dict(decision=decision, receipt=receipt,
                    binding=_receipt_binding(receipt), generation=self._generation,
                    deadline=float(deadline), clock_id=clock_id,
                    guard_lifetime=inflight['guard_lifetime'], installed=inflight['installed'],
                    kind=('execution_budget' if replan_reason=='receipt_execution_budget_replan' else 'valid_UI_change'),
                    decision_at=finished)
                self.status.update(allowed=False, command_pending=False, input_hold_required=True,
                    assistance_applied=False, reason=replan_reason, timestamp=finished,
                    receipt_replan_required=True, receipt_was_installed=inflight['installed'],
                    receipt_replan_kind=self._receipt_replan['kind'])
                self.status['snapshot'].update(input_hold_required=True,
                    orientation_offset_rad=float(Rotation.from_matrix(self._committed_offset).magnitude()))
                return decision
            self._inflight_receipt = next_inflight
            self.status['receipt_inputs_checked_at'] = finished
            return updated
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError, RuntimeError) as exc:
            self.hold(str(exc), now=now if _number(now) else None, clock_id=clock_id)
            return None

    def consume_receipt_replan(self, decision, receipt, *, now, clock_id):
        """Consume one exact private replan decision before an actor holds.

        Public status/reason strings cannot grant this path. Replays, copies,
        wrong receipts, expiry and changed generations cause a full hold.
        """
        started = self.clock()
        registered = self._receipt_replan
        self._receipt_replan = None  # Burn even a failed/counterfeit attempt.
        try:
            if (type(decision) is not SimReceiptReplan or registered is None or
                    decision is not registered['decision'] or receipt is not registered['receipt'] or
                    type(receipt) is not SimCommandReceipt or
                    _receipt_binding(receipt) != registered['binding'] or
                    registered['generation'] != self._generation or not self.clutched or
                    self._inflight_receipt is not None or self._pending is not None):
                raise ValueError('receipt_replan_missing_changed_or_replayed')
            if not _number(now) or clock_id != registered['clock_id']:
                raise ValueError('receipt_replan_clock_invalid')
            if registered.get('kind') == 'execution_budget':
                finished = self.clock()
                if (not _number(started) or not _number(finished) or started < now
                        or finished < started or started < registered['decision_at']
                        or registered['installed']):
                    raise ValueError('receipt_execution_budget_clock_or_installation_invalid')
            else:
                finished = self._time_after_work(now, started)
            if registered['generation'] != self._generation or not self.clutched:
                raise ValueError('receipt_replan_revoked_during_consumption')
            if (finished >= registered['deadline'] or not command_is_current(
                    registered['guard_lifetime'], now=finished, clock_id=clock_id)):
                raise ValueError('receipt_replan_expired')
            self.status.update(receipt_replan_consumed_at=finished, receipt_replan_required=False)
            return True
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError, RuntimeError) as exc:
            self.hold(str(exc), now=now if _number(now) else None, clock_id=clock_id)
            return False

    def mark_receipt_installed(self, receipt, *, now, clock_id):
        """Actor acknowledgment immediately after its successful target setter.

        This cannot extend a lease, authorize another write, or resurrect a
        receipt. A failed acknowledgment requires the actor's full measured hold.
        """
        started = self.clock()
        try:
            inflight = self._inflight_receipt
            if (type(receipt) is not SimCommandReceipt or inflight is None or
                    _receipt_binding(receipt) != inflight['binding'] or inflight['installed'] or
                    inflight['generation'] != self._generation or not self.clutched):
                raise ValueError('receipt_install_ack_missing_changed_or_replayed')
            if not _number(now) or clock_id != self._clock_id:
                raise ValueError('receipt_install_ack_clock_invalid')
            finished = self._time_after_work(now, started)
            if (self._inflight_receipt is not inflight or inflight['generation'] != self._generation or
                    not self.clutched):
                raise ValueError('receipt_install_ack_revoked_during_check')
            if (finished >= inflight['deadline'] or not command_is_current(
                    inflight['guard_lifetime'], now=finished, clock_id=clock_id)):
                raise ValueError('receipt_install_ack_expired')
            self._inflight_receipt = dict(inflight, installed=True)
            self.status['receipt_installed_at'] = finished
            return True
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError, RuntimeError) as exc:
            self.hold(str(exc), now=now if _number(now) else None, clock_id=clock_id)
            return False

    def _consume(self, offer, *, now, clock_id, measured_state, scene, as_receipt):
        started = self.clock()
        if not isinstance(offer, dict) or offer.get('allowed') is not True:
            return None
        if self._cancelled_token is not None and offer.get('command_token') == self._cancelled_token:
            return None
        try:
            pending = self._pending
            if (pending is None or offer.get('command_token') != pending['token'] or
                    pending['generation'] != self._generation or not self.clutched):
                raise ValueError('command_missing_replayed_or_superseded')
            if not _number(now) or now >= pending['deadline'] or not command_is_current(pending['guard'], now=now, clock_id=clock_id):
                raise ValueError('command_expired_before_sim_handoff')
            self._state(measured_state, now, clock_id)
            self._scene(scene, now, clock_id)
            before, current = pending['state']['joint_positions'], measured_state['joint_positions']
            if any(abs(before[name] - current[name]) > self.config.state_consumption_tolerance_rad for name in G1_JOINT_NAMES):
                raise ValueError('robot_state_changed_after_guard')
            if not self._pose_matches(pending['state']['base_in_world'], measured_state['base_in_world'], position=1e-8, rotation=1e-8):
                raise ValueError('robot_base_changed_after_guard')
            prior_bodies, current_bodies = pending['state']['observed_body_world_poses'], measured_state['observed_body_world_poses']
            if set(prior_bodies) != set(current_bodies) or not self._body_poses_match(prior_bodies, current_bodies):
                raise ValueError('robot_body_observation_changed_after_guard')
            if not self._scene_matches(pending['scene'], scene, exact=True):
                raise ValueError('scene_changed_after_guard')
            command = pending['q14'].copy()
            output = self._command_receipt(pending, measured_state, scene) if as_receipt else command
            inflight = None
            if as_receipt:
                inflight = dict(binding=_receipt_binding(output), generation=pending['generation'],
                    deadline=pending['deadline'], assist_held=pending['assist_held'],
                    installed=False, prior_committed_offset=self._committed_offset.copy(),
                    guard_lifetime={key:pending['guard'][key] for key in
                        ('allowed','hold_required','clock_id','validated_at','valid_until')})
            committed_offset = pending['assist_offset'].copy()
            self._pending = None
            self.status.update(command_pending=False, command_consumed_once=True)
            finished = self._time_after_work(now, started)
            if finished >= pending['deadline'] or not command_is_current(pending['guard'], now=finished, clock_id=clock_id):
                raise ValueError('command_expired_during_sim_handoff_checks')
            # Provisional for a receipt: the actor must acknowledge actual
            # installation. A pre-write replan or superseding step rolls this
            # back; the legacy array API retains its immediate-handoff contract.
            self._committed_offset = committed_offset
            if as_receipt:
                self._inflight_receipt = inflight
            return output
        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
            self.hold('command_handoff_rejected_reclutch_required', now=now, clock_id=clock_id)
            self.status['handoff_rejection_reason'] = str(exc)
            return None
