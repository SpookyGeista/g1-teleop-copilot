"""Deliberate live-preview selection for the isolated simulator.

Only LiveSimProposals.authorize can produce a typed execution-side batch.
UI records remain compact previews; JSON permission flags are never forwarded.
The caller must hold its control session whenever reset_control is true.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import hashlib
import math
from pathlib import Path
import time

from .controller import rigid
from .live_sim_proposals import LiveSimProposals, OBSERVED_PREGRASP_REQUIRED
from .mailbox import read_json
from .frame_sync import await_preview_frame
from .sim_control import SimUIIntent
from .types import PreviewSelection
from .vr_buttons import controller_event_time_issue, has_vr_time_provenance, VR_TIME_FIELDS


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _identity(value):
    return isinstance(value, str) and 0 < len(value) <= 256


def _signature(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _diagnostic_text(value):
    # Runtime mailboxes can be malformed. Never let diagnostic collection
    # itself escape fail-closed handling or copy an unbounded source payload.
    return value[:256] if isinstance(value,str) else None


def _observed_collision_summary(value):
    """Retain the raw preview result without turning it into execution flags."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ('status', 'reason', 'collision_scope'):
        if isinstance(value.get(key), str) and len(value[key]) <= 512:
            result[key] = value[key]
    for key in ('hand_collision_free', 'pregrasp_collision_free', 'scene_coverage_unknown'):
        if type(value.get(key)) is bool:
            result[key] = value[key]
    for key in ('object_point_count', 'scene_point_count', 'scene_broad_phase_point_count'):
        if type(value.get(key)) is int and value[key] >= 0:
            result[key] = value[key]
    final = value.get('final')
    if isinstance(final, dict):
        result['final'] = {key: final[key] for key in
            ('forbidden_object_points', 'scene_collision_points', 'allowed_target_pad_contacts')
            if type(final.get(key)) is int and final[key] >= 0}
    return result


class SimTargetFeed:
    """Synchronous glue; no automatic candidate choice or confirmation."""

    def __init__(self, directory, worker, descriptor_hash, *, clock=time.monotonic,
                 policy=OBSERVED_PREGRASP_REQUIRED):
        if not callable(getattr(worker, 'call', None)) or not callable(clock):
            raise ValueError('Worker.call and an acquisition-clock callable are required')
        self.directory, self.worker, self.clock = Path(directory), worker, clock
        self.gate = LiveSimProposals(descriptor_sha256=descriptor_hash, policy=policy)
        self.policy = policy
        self.descriptor_hash = descriptor_hash
        self.active_candidate_id = None
        self._target_revision = self._target_signature = None
        self._ui_revision = self._ui_signature = None
        self._authorized_binding = None
        self._staged_selection = self._committed_selection = None
        # Recheck-only parsed values never enter worker requests or public UI
        # results. They are reused only after reading identical complete bytes.
        self.__recheck_json = {}

    def _read_recheck_json(self, filename):
        """Private parse reuse, never a cached validation or freshness result."""
        if filename not in ('target_selection.json', 'latest_proposals.json'):
            raise ValueError('unknown_recheck_mailbox')
        try:
            with (self.directory/filename).open('rb') as stream:
                raw = stream.read(2*1024*1024+1)
            if len(raw) > 2*1024*1024:
                raise ValueError('Mailbox too large')
            previous = self.__recheck_json.get(filename)
            if previous is not None and raw == previous[0]:
                return previous[1]
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError('Mailbox must contain an object')
            self.__recheck_json[filename] = (raw, value)
            return value
        except FileNotFoundError:
            self.__recheck_json.pop(filename, None)
            return None
        except Exception:
            self.__recheck_json.pop(filename, None)
            raise

    def record_frame(self, frame, scene):
        return self.gate.record_frame(frame, scene)

    def revoke_validation(self):
        """Drop authority without replaying old object or candidate clicks."""
        self.active_candidate_id = self._authorized_binding = None
        self._staged_selection = self._committed_selection = None
        self.gate.reset()

    def finish_preview_selection(self, status):
        """Observe the actual session commit; this grants no batch authority.

        Called immediately after the selection tick. A failed/expired session
        commit burns the staging state instead of leaving a Feed-only choice.
        """
        staged = self._staged_selection
        snapshot = status.get('snapshot', {}) if isinstance(status, dict) else {}
        audit = snapshot.get('preview_selection', {})
        expected = self._selection_identity(staged) if staged is not None else None
        if (expected is None or not isinstance(audit, dict)
                or any(audit.get(key) != value for key, value in expected.items())
                or snapshot.get('selected_candidate_id') != staged.candidate_id
                or snapshot.get('preview_selection_pending') is not True
                or status.get('allowed') is not False or snapshot.get('clutched') is not False):
            self.revoke_validation()
            return False
        self._committed_selection = expected
        self._staged_selection = None
        return True

    @staticmethod
    def _selection_identity(preview):
        return {key: getattr(preview, key) for key in ('ui_revision', 'ui_signature', 'ui_client_id',
            'target_revision', 'target_signature', 'target_id', 'candidate_id', 'model_source_id')}

    def recheck_preview_selection(self, preview, *, now, clock_id, calibration_id):
        """Last non-authorizing target read before the early source/UI refresh."""
        try:
            if type(preview) is not PreviewSelection or preview != self._staged_selection:
                raise ValueError('staged_preview_missing_or_changed')
            selection = self._read_recheck_json('target_selection.json')
            if _signature(selection) != preview.target_signature:
                raise ValueError('object_selection_changed_before_preview_commit')
            current = self._read_recheck_json('latest_proposals.json')
            self._ui_candidates(current, selection, now, clock_id, calibration_id, materialize=False)
            checked = self.gate.recheck(current, selected_candidate_id=preview.candidate_id,
                now=now, clock_id=clock_id, calibration_id=calibration_id)
            if not checked.ready or checked.model_source_id != preview.model_source_id:
                raise ValueError(checked.reason if not checked.ready else 'preview_model_source_changed')
            return dict(accepted=True, reason='staged_preview_still_current', motion_authorized=False)
        except (TypeError, ValueError, KeyError, AttributeError, OSError) as error:
            self.revoke_validation()
            return dict(accepted=False, reason=str(error), motion_authorized=False)

    def _target(self, packet, now, clock_id):
        if not isinstance(packet, dict) or type(packet.get('revision')) is not int or packet['revision'] < 1:
            raise ValueError('object_selection_missing_or_invalid')
        if packet.get('target_id') is not None and not _identity(packet['target_id']):
            raise ValueError('object_selection_identity_invalid')
        if (packet.get('clock_id') != clock_id or not _number(packet.get('capture_timestamp'))
                or packet['capture_timestamp'] > now+.01):
            raise ValueError('object_selection_clock_or_timestamp_invalid')
        revision, signature = packet['revision'], _signature(packet)
        if self._target_revision is not None and revision < self._target_revision:
            raise ValueError('object_selection_revision_regressed')
        if revision == self._target_revision and signature != self._target_signature:
            raise ValueError('object_selection_revision_reused_with_changed_content')
        changed = revision != self._target_revision
        self._target_revision, self._target_signature = revision, signature
        if changed:
            self.gate.reset()
            self.active_candidate_id = None
            self._staged_selection = self._committed_selection = None
        return changed

    def _intent(self, value, now, clock_id, selection):
        packet = asdict(value) if isinstance(value, SimUIIntent) else value
        if packet is None or packet == {}:
            return None
        if not isinstance(packet, dict) or type(packet.get('revision')) is not int or packet['revision'] < 1:
            raise ValueError('sim_selection_revision_invalid')
        revision, signature = packet['revision'], _signature(packet)
        if self._ui_revision is not None and revision < self._ui_revision:
            raise ValueError('sim_selection_revision_regressed')
        if revision == self._ui_revision:
            if signature != self._ui_signature:
                raise ValueError('sim_selection_revision_reused_with_changed_content')
            return None
        # Consume even invalid/expired selection clicks: data recovery must not
        # silently turn the same old action into a new deliberate selection.
        self._ui_revision, self._ui_signature = revision, signature
        candidate_id = packet.get('select_candidate_id')
        if candidate_id is None:
            return None
        if not _identity(candidate_id) or not _identity(packet.get('client_id')):
            raise ValueError('sim_selection_identity_invalid')
        stamp = packet.get('capture_timestamp')
        provenance = packet
        if isinstance(value, SimUIIntent) and all(getattr(value,name) is None for name in VR_TIME_FIELDS):
            # A typed legacy intent has None defaults; raw JSON containing
            # explicit null/partial metadata must still fail closed.
            provenance = {name:item for name,item in packet.items() if name not in VR_TIME_FIELDS}
        bounded = has_vr_time_provenance(provenance)
        if (packet.get('clock_id') != clock_id or not _number(stamp)
                or stamp < selection['capture_timestamp']):
            raise ValueError('sim_selection_stale_future_or_before_object_selection')
        expiry = packet.get('expires_at')
        if bounded:
            if (packet.get('source_kind') != 'webxr_controller_event'
                    or type(packet.get('source_event_sequence')) is not int or packet['source_event_sequence'] < 0):
                raise ValueError('sim_selection_vr_provenance_invalid')
            issue = controller_event_time_issue(stamp, expiry, now=now,
                **{name:packet[name] for name in VR_TIME_FIELDS})
            if issue: raise ValueError('sim_selection_' + issue)
            # A mapped time ahead of host time must not reorder an earlier
            # button event across the user's object choice. Require both its
            # real arrival and its entire capture interval to follow that choice.
            if (packet['received_at_host_monotonic'] < selection['capture_timestamp']
                    or stamp-packet['clock_uncertainty_s'] < selection['capture_timestamp']):
                raise ValueError('sim_selection_before_object_selection_with_clock_uncertainty')
        elif not 0 <= now-stamp < .25:
            raise ValueError('sim_selection_stale_future_or_before_object_selection')
        if expiry is not None and (not _number(expiry) or not stamp < expiry <= stamp+.25 or now >= expiry):
            raise ValueError('sim_selection_expired')
        flags = [packet.get(name, False) for name in ('clutch', 'confirm', 'assist_held', 'release')]
        if any(type(value) is not bool or value for value in flags):
            raise ValueError('sim_selection_must_be_a_separate_action')
        return candidate_id

    def _ui_candidates(self, preview, selection, now, clock_id, calibration_id, *, materialize=True):
        if (not isinstance(preview, dict) or preview.get('valid') is not False
                or preview.get('motion_static') is not True
                or preview.get('validation_reason') != 'preview_only_actual_command_path_not_validated'
                or preview.get('clock_id') != clock_id or preview.get('calibration_id') != calibration_id
                or preview.get('descriptor_sha256') != self.descriptor_hash
                or preview.get('target_id') != selection.get('target_id')
                or not _number(preview.get('capture_timestamp'))
                or not -.01 <= now-preview['capture_timestamp'] <= .25):
            raise ValueError('current_selected_object_preview_unavailable')
        candidates = preview.get('candidates')
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 128:
            raise ValueError('preview_candidate_list_invalid')
        result, ids = [], set()
        for candidate in candidates:
            if not isinstance(candidate, dict) or not _identity(candidate.get('candidate_id')):
                raise ValueError('preview_candidate_identity_invalid')
            identity = candidate['candidate_id']
            if identity in ids:
                raise ValueError('preview_candidate_identity_duplicated')
            ids.add(identity)
            observed = candidate.get('observed_hand_collision')
            observed_clear = isinstance(observed, dict) and observed.get('pregrasp_collision_free') is True
            if (candidate.get('geometry_valid') is not True or candidate.get('ik_valid') is not True
                    or candidate.get('collision_valid') is not False
                    or (self.policy == OBSERVED_PREGRASP_REQUIRED and not observed_clear)):
                continue
            if not _number(candidate.get('model_score')) or not _identity(candidate.get('source')):
                raise ValueError('preview_candidate_provenance_invalid')
            base_ee = rigid(candidate.get('base_ee'))
            world_canonical = rigid(candidate['world_canonical']) if 'world_canonical' in candidate else None
            if not materialize:
                # Recheck still validates every eligible pose, including rows
                # outside the UI's display bound; no result graph is needed.
                continue
            row = dict(candidate_id=identity, base_ee=base_ee.tolist(),
                model_score=candidate['model_score'], source=candidate['source'], preview_only=True,
                preview_collision_policy=self.policy, observed_hand_collision=_observed_collision_summary(observed))
            if world_canonical is not None:
                row['world_canonical'] = world_canonical.tolist()
            if len(result) < 32:
                result.append(row)
            elif identity == self.active_candidate_id:
                result[-1] = row
        return result

    def _preview_evidence(self, preview, selection, *, phase, now, clock_id,
                          calibration_id, entered_at=None, checked_at=None, reason=None):
        preview=preview if isinstance(preview,dict) else {}
        selection=selection if isinstance(selection,dict) else {}
        stamp=preview.get('capture_timestamp')
        age=now-stamp if _number(now) and _number(stamp) else None
        failed=[]
        checks=(('preview_valid_flag',preview.get('valid') is False),
            ('motion_static_unavailable',preview.get('motion_static') is True),
            ('service_validation_reason',preview.get('validation_reason')=='preview_only_actual_command_path_not_validated'),
            ('preview_clock_mismatch',preview.get('clock_id')==clock_id),
            ('preview_calibration_mismatch',preview.get('calibration_id')==calibration_id),
            ('preview_descriptor_mismatch',preview.get('descriptor_sha256')==self.descriptor_hash),
            ('preview_target_mismatch',preview.get('target_id')==selection.get('target_id')),
            ('preview_capture_timestamp_unknown',_number(stamp)))
        failed.extend(name for name,passed in checks if not passed)
        if age is not None:
            if age<-.01:failed.append('preview_capture_timestamp_future')
            elif age>.25:failed.append('preview_age_exceeds_250ms')
        return dict(schema='punktnull.sim_preview_check.v1',phase=phase,
            check_now=now if _number(now) else None,
            entered_at=entered_at if _number(entered_at) else None,
            checked_at=checked_at if _number(checked_at) else None,
            elapsed_s=checked_at-entered_at if _number(checked_at) and _number(entered_at)
                and _number(checked_at-entered_at) else None,
            preview_capture_timestamp=stamp if _number(stamp) else None,
            preview_age_s=age if _number(age) else None,
            max_preview_age_s=.25,future_tolerance_s=.01,
            preview_valid=preview.get('valid') if type(preview.get('valid')) is bool else None,
            motion_static=preview.get('motion_static') if type(preview.get('motion_static')) is bool else None,
            validation_reason=_diagnostic_text(preview.get('validation_reason')),
            validation_observation_id=_diagnostic_text(preview.get('validation_observation_id')),
            observation_id=_diagnostic_text(preview.get('observation_id')),
            model_source_observation_id=_diagnostic_text(preview.get('model_source_observation_id')),
            model_source_capture_timestamp=preview.get('model_source_capture_timestamp')
                if _number(preview.get('model_source_capture_timestamp')) else None,
            preview_clock_id=_diagnostic_text(preview.get('clock_id')),
            expected_clock_id=_diagnostic_text(clock_id),
            preview_calibration_id=_diagnostic_text(preview.get('calibration_id')),
            expected_calibration_id=_diagnostic_text(calibration_id),
            target_id=_diagnostic_text(preview.get('target_id')),
            selected_target_id=_diagnostic_text(selection.get('target_id')),
            selected_candidate_id=_diagnostic_text(self.active_candidate_id),
            model_source_id=_diagnostic_text((self._authorized_binding or {}).get('model_source_id')),
            failed_preview_checks=failed,reason=_diagnostic_text(reason),
            motion_authorized=False,deadline_renewed=False)

    def prepare(self, *, intent, session_status, robot_state, scene, left_base_ee,
                now, clock_id, calibration_id, current_frame=None, cancelled=lambda:False):
        self._authorized_binding = None
        preview=selection=None;phase='prepare_input';entered_at=now;checked_at=None
        result = dict(batch=None, candidates=[], goal=None, reason='choose_object',
                      preview_selection=None,
                      endpoint_evidence=None, reset_control=False,preview_evidence=None,rejection_evidence=None,
                      frame_barrier=None,frame_synchronization_s=0.)
        try:
            if not _number(now) or not _identity(clock_id) or not _identity(calibration_id):
                raise ValueError('current_time_invalid')
            selection = read_json(self.directory/'target_selection.json')
            result['reset_control'] = self._target(selection, now, clock_id)
            selected = self._intent(intent, now, clock_id, selection)
            if selection.get('target_id') is None:
                return result
            preview = read_json(self.directory/'latest_proposals.json')
            phase='prepare_preview'
            result['preview_evidence']=self._preview_evidence(preview,selection,phase=phase,
                now=now,clock_id=clock_id,calibration_id=calibration_id,entered_at=entered_at)
            result['candidates'] = self._ui_candidates(preview, selection, now, clock_id, calibration_id)
            if selected is not None:
                if selected not in {item['candidate_id'] for item in result['candidates']}:
                    raise ValueError('deliberately_selected_preview_candidate_unavailable')
                self.gate.reset()
                self._staged_selection = self._committed_selection = None
                self.active_candidate_id = selected
                result['reset_control'] = True
            if self.active_candidate_id is None:
                result['reason'] = 'choose_grasp_explicitly'
                return result
            snapshot = session_status.get('snapshot', {}) if isinstance(session_status, dict) else {}
            if selected is None:
                audit = snapshot.get('preview_selection', {})
                if (self._staged_selection is not None or self._committed_selection is None
                        or not isinstance(audit, dict)
                        or any(audit.get(key) != value for key, value in self._committed_selection.items())
                        or snapshot.get('selected_candidate_id') != self.active_candidate_id):
                    raise ValueError('preview_selection_not_committed_or_revoked')
            confirmed = None if result['reset_control'] else snapshot.get('confirmed_candidate_id')
            phase='prepare_inspect'
            prepared = self.gate.inspect(preview, selected_candidate_id=self.active_candidate_id,
                confirmed_candidate_id=confirmed, now=now, clock_id=clock_id, calibration_id=calibration_id)
            if not prepared.ready:
                raise ValueError(prepared.reason)
            if selected is not None:
                ui = intent if isinstance(intent, SimUIIntent) else SimUIIntent.from_packet(intent)
                candidate = prepared.candidate
                self._staged_selection = PreviewSelection(
                    target_id=selection['target_id'], target_revision=selection['revision'],
                    target_signature=self._target_signature,
                    observation_id=prepared.validation_observation_id,
                    capture_timestamp=prepared.validation_capture_timestamp, clock_id=clock_id,
                    calibration_id=calibration_id, candidate_id=candidate['candidate_id'],
                    base_ee=tuple(tuple(float(x) for x in row) for row in rigid(candidate['base_ee'])),
                    model_score=candidate['model_score'], source=candidate['source'],
                    model_source_id=prepared.model_source_id,
                    model_source_capture_timestamp=preview['model_source_capture_timestamp'],
                    model_source_clock_id=clock_id, ui_revision=ui.revision,
                    ui_signature=hashlib.sha256(json.dumps(vars(ui),sort_keys=True,
                        allow_nan=False).encode()).hexdigest(), ui_client_id=ui.client_id)
                result.update(preview_selection=self._staged_selection,
                              reason='preview_selection_awaiting_session_commit')
                return result
            state_copy, scene_copy = deepcopy(robot_state), deepcopy(scene)
            phase='prepare_endpoint'
            started = self.clock()
            endpoint = self.worker.call('validate_sim_candidate', base_ee=deepcopy(prepared.candidate['base_ee']),
                left_base_ee=rigid(left_base_ee).tolist(), robot_state=state_copy, scene=scene_copy)
            finished = self.clock()
            checked_at=finished
            result['preview_evidence']=self._preview_evidence(preview,selection,phase=phase,
                now=finished,clock_id=clock_id,calibration_id=calibration_id,
                entered_at=entered_at,checked_at=finished)
            result['endpoint_evidence'] = deepcopy(endpoint)
            # Diagnostics stay outside endpoint evidence and authorization.
            endpoint_worker_timing = deepcopy(getattr(self.worker, 'last_call_timing', None))
            if (not _number(started) or not _number(finished) or finished < started or finished < now):
                raise ValueError('endpoint_validation_clock_regressed_or_unknown')
            endpoint_elapsed=finished-started
            if current_frame is not None:
                # Current-image tracking runs in parallel with the endpoint
                # worker. No command/batch has yet been issued: construct the
                # NEW batch from genuinely measured current-frame evidence,
                # keeping the exact frozen candidate and original model time.
                phase='prepare_current_frame'
                synchronizing_at=self.clock()
                result['frame_barrier']=await_preview_frame(self.directory,current_frame,
                    clock=self.clock,cancelled=cancelled)
                checked_at=self.clock()
                result['frame_synchronization_s']=checked_at-synchronizing_at
                if not result['frame_barrier']['ready']:
                    raise ValueError(result['frame_barrier']['reason'])
                if _signature(read_json(self.directory/'target_selection.json')) != self._target_signature:
                    raise ValueError('object_selection_changed_during_endpoint_check')
                preview=read_json(self.directory/'latest_proposals.json')
                barrier=result['frame_barrier']
                if (preview.get('validation_observation_id')!=barrier['expected_validation_observation_id']
                        or preview.get('capture_timestamp')!=barrier['capture_timestamp']
                        or preview.get('clock_id')!=barrier['clock_id']
                        or preview.get('calibration_id')!=barrier['calibration_id']):
                    raise ValueError('acknowledged_frame_changed_before_authorization')
                checked_at=self.clock()
                result['candidates']=self._ui_candidates(preview,selection,checked_at,clock_id,calibration_id)
                current=self.gate.inspect(preview,selected_candidate_id=self.active_candidate_id,
                    confirmed_candidate_id=confirmed,now=checked_at,clock_id=clock_id,calibration_id=calibration_id)
                if not current.ready:raise ValueError(current.reason)
                if current.model_source_id!=prepared.model_source_id or current.candidate!=prepared.candidate:
                    raise ValueError('candidate_changed_during_endpoint_check')
                prepared=current
                finished=self.clock();checked_at=finished
                result['frame_synchronization_s']=finished-synchronizing_at
                result['preview_evidence']=self._preview_evidence(preview,selection,phase=phase,
                    now=finished,clock_id=clock_id,calibration_id=calibration_id,
                    entered_at=entered_at,checked_at=finished)
            phase='prepare_authorize'
            authorized = self.gate.authorize(prepared, endpoint, robot_state=state_copy, scene=scene_copy,
                now=finished, clock_id=clock_id)
            if not authorized.accepted:
                raise ValueError(authorized.reason)
            self._authorized_binding = dict(target_revision=self._target_revision,
                target_signature=self._target_signature, candidate_id=self.active_candidate_id,
                model_source_id=authorized.batch.model_source_id)
            result.update(batch=authorized.batch, goal=rigid(prepared.candidate['base_ee']),
                reason=authorized.reason, authorization_evidence=authorized.evidence)
            result['preview_evidence']['model_source_id']=_diagnostic_text(authorized.batch.model_source_id)
            result['preview_evidence']['endpoint_elapsed_s']=endpoint_elapsed
            result['preview_evidence']['endpoint_worker_timing']=endpoint_worker_timing
            return result
        except Exception as error:
            evidence=self._preview_evidence(preview,selection,phase=phase,
                now=checked_at if _number(checked_at) else now,clock_id=clock_id,
                calibration_id=calibration_id,entered_at=entered_at,checked_at=checked_at,reason=str(error))
            # Even an unreadable object file must consume a concurrent select
            # revision, so repairing that file cannot revive the old click.
            try:
                packet = asdict(intent) if isinstance(intent, SimUIIntent) else intent
                if isinstance(packet, dict) and type(packet.get('revision')) is int:
                    revision = packet['revision']
                    if self._ui_revision is None or revision > self._ui_revision:
                        self._ui_revision, self._ui_signature = revision, _signature(packet)
            except Exception:
                pass
            self.active_candidate_id = None
            self._staged_selection = self._committed_selection = None
            result.update(batch=None, goal=None, reset_control=True,
                reason='live_target_feed_rejected:'+str(error),preview_evidence=evidence,rejection_evidence=evidence)
            return result

    def recheck(self, *, session_status, now, clock_id, calibration_id):
        """Only revoke a pending handoff after blocking work; never renew it.

        Call after the session's source/UI recheck and immediately before its
        consume(). The already issued batch and command retain their original
        capture time and deadline, even when this reads a newer camera preview.
        No IK, worker call, authorization or selection action occurs here.
        """
        preview=selection=None;phase='recheck_input';entered_at=now;checked_at=None
        try:
            if not _number(now) or not _identity(clock_id) or not _identity(calibration_id):
                raise ValueError('current_time_invalid')
            started = self.clock()
            binding = self._authorized_binding
            if binding is None or self.active_candidate_id != binding['candidate_id']:
                raise ValueError('authorized_selection_missing_or_revoked')
            selection = self._read_recheck_json('target_selection.json')
            if (not isinstance(selection, dict) or selection.get('revision') != binding['target_revision']
                    or _signature(selection) != binding['target_signature']):
                raise ValueError('object_selection_changed_since_endpoint_check')
            preview = self._read_recheck_json('latest_proposals.json')
            phase='recheck_preview'
            self._ui_candidates(preview, selection, now, clock_id, calibration_id, materialize=False)
            snapshot = session_status.get('snapshot', {}) if isinstance(session_status, dict) else {}
            checked = self.gate.recheck(preview, selected_candidate_id=self.active_candidate_id,
                confirmed_candidate_id=snapshot.get('confirmed_candidate_id'), now=now,
                clock_id=clock_id, calibration_id=calibration_id)
            if not checked.ready:
                raise ValueError(checked.reason)
            if checked.model_source_id != binding['model_source_id']:
                raise ValueError('model_source_changed_since_endpoint_check')
            # Selection may have changed while parsing/checking the preview.
            if _signature(self._read_recheck_json('target_selection.json')) != binding['target_signature']:
                raise ValueError('object_selection_changed_during_handoff_recheck')
            finished = self.clock()
            checked_at=finished;phase='recheck_finished'
            if (not _number(started) or not _number(finished) or finished < started or finished < now):
                raise ValueError('handoff_recheck_clock_regressed_or_unknown')
            if not -.01 <= finished-preview['capture_timestamp'] <= .25:
                raise ValueError('preview_expired_during_handoff_recheck')
            return dict(accepted=True, reason='bound_live_selection_still_current',
                motion_authorized=False, deadline_renewed=False,
                preview_evidence=self._preview_evidence(preview,selection,phase=phase,now=finished,
                    clock_id=clock_id,calibration_id=calibration_id,entered_at=entered_at,checked_at=finished),
                rejection_evidence=None)
        except Exception as error:
            evidence=self._preview_evidence(preview,selection,phase=phase,
                now=checked_at if _number(checked_at) else now,clock_id=clock_id,
                calibration_id=calibration_id,entered_at=entered_at,checked_at=checked_at,reason=str(error))
            self.active_candidate_id = self._authorized_binding = None
            self.gate.reset()
            return dict(accepted=False, reason='live_target_recheck_rejected:'+str(error),
                motion_authorized=False, deadline_renewed=False,
                preview_evidence=evidence,rejection_evidence=evidence)
