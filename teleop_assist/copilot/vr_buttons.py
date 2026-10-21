"""Optional original-Vuer controller buttons for isolated simulation.

These are protocol field names, not a claim about Quest/Pico button labels.
No I/O, controller constructor, finger command or robot transport is used here.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math

SCHEMA = 'punktnull.vr_buttons.v1'
MAPPING = 'vuer_controller_v1'
BUTTONS = ('aButton', 'bButton', 'squeeze')
VR_TIME_FIELDS = ('clock_uncertainty_s', 'received_at_host_monotonic')


def number(value):
    try: return type(value) in (int, float) and math.isfinite(value)
    except OverflowError: return False


def controller_event_time_issue(stamp, expiry, *, now, clock_uncertainty_s,
                                received_at_host_monotonic, future_tolerance_s=.01):
    """Check a bounded mapped VR time without restamping or extending its lease.

    This is an explicit VR-only policy: an already received event's mapped time
    can be slightly ahead of host now within its measured clock uncertainty.
    It does not grant this tolerance to legacy or desktop UI timestamps.
    """
    bound, receipt = clock_uncertainty_s, received_at_host_monotonic
    if (not all(number(v) for v in (stamp, expiry, now, bound, receipt, future_tolerance_s))
            or not 0 <= bound <= .02 or not 0 <= future_tolerance_s <= .01):
        return 'vr_clock_provenance_invalid'
    if (receipt > now or stamp > receipt + bound
            or stamp > now + min(bound, future_tolerance_s)
            or not stamp < expiry or not now < expiry <= min(stamp + .15, receipt + .15)):
        return 'vr_event_stale_or_time_invalid'
    return None


def has_vr_time_provenance(mapping):
    """Return pair presence; reject partial/null metadata rather than downgrade."""
    present = [name in mapping for name in VR_TIME_FIELDS]
    if any(present) and (not all(present) or not all(number(mapping[name]) for name in VR_TIME_FIELDS)):
        raise ValueError('Incomplete or invalid VR clock provenance')
    return all(present)


def states_issue(states):
    if not isinstance(states, dict): return 'controller_states_missing'
    for side in ('left', 'right'):
        state = states.get(side)
        if not isinstance(state, dict) or any(type(state.get(name)) is not bool for name in BUTTONS):
            return 'controller_buttons_unknown'
        gamepad = state.get('gamepad')
        if (not isinstance(gamepad, dict) or gamepad.get('mapping') != 'xr-standard'
                or type(gamepad.get('button_count')) is not int or not 6 <= gamepad['button_count'] <= 128
                or gamepad.get('required_buttons_available') is not True):
            return 'controller_button_capabilities_unknown'
    return None


def source_payload(arms, operator):
    """Bind the exact controller event to the already validated pose packet."""
    issue = None
    if operator.get('tracking_valid') is not True: issue = 'operator_invalid'
    elif arms.get('event_kind') != 'controller': issue = 'controller_mode_required'
    else: issue = states_issue(arms.get('controller_states'))
    stamp, receipt = arms.get('host_sample_time_s'), arms.get('received_at_host_monotonic')
    bound = arms.get('clock_uncertainty_s')
    if not issue and (not number(stamp) or not number(receipt)
                      or not number(bound) or not 0 <= bound <= .02
                      or arms.get('host_clock_id') != operator.get('clock_id')
                      or stamp > receipt + bound or receipt > operator['published_at']):
        issue = 'controller_sample_time_invalid'
    return dict(schema=SCHEMA, mapping=MAPPING, valid=issue is None, reason=issue or 'valid_controller_event',
        event_sequence=arms.get('sequence'), acquisition_timestamp=stamp,
        clock_id=arms.get('host_clock_id'), expires_at=min(receipt + .15, stamp + .15) if not issue else None,
        clock_uncertainty_s=bound, received_at_host_monotonic=receipt,
        source_instance_id=operator.get('source_instance_id'),
        source_event_clock_id=operator.get('source_event_clock_id'),
        states=arms.get('controller_states', {}))


@dataclass(frozen=True)
class VRIntent:
    action: str | None
    reason: str
    client_id: str | None = None
    candidate_id: str | None = None
    acquisition_timestamp: float | None = None
    clock_id: str | None = None
    expires_at: float | None = None
    event_sequence: int | None = None
    clock_uncertainty_s: float | None = None
    received_at_host_monotonic: float | None = None

    def source_sample(self):
        """Preserve the original edge's complete timing, including deferred edges."""
        result = dict(acquisition_timestamp=self.acquisition_timestamp, clock_id=self.clock_id,
                      expires_at=self.expires_at, event_sequence=self.event_sequence)
        if self.clock_uncertainty_s is not None or self.received_at_host_monotonic is not None:
            result.update(clock_uncertainty_s=self.clock_uncertainty_s,
                          received_at_host_monotonic=self.received_at_host_monotonic)
        return result


class VRButtonParser:
    """Consume new source events once; rearm only after all buttons are up.

    The WebXR ``aButton``/``bButton`` fields represent X/Y on the left
    controller and A/B on the right controller.  The main state transitions
    follow SONIC's familiar chord layout:

    * A+B+X+Y: calibrate/arm, or stop when already calibrated;
    * A+X: start/pause teleoperation after calibration;
    * Y: next offered candidate; B: confirm; right squeeze: hold assistance.

    A new confirmation also requires squeeze-up before assistance can begin.
    """
    def __init__(self):
        self.epoch = None
        self.sequence = -1
        self.signature = None
        self.previous = None
        self.armed = False
        self.confirmed_id = None
        self.hold_armed = False
        self.client_id = None
        self.offered_choice = None

    def disarm(self):
        self.armed = self.hold_armed = False
        self.previous = None
        self.confirmed_id = None
        self.offered_choice = None

    def _reject(self, reason):
        self.disarm()
        return VRIntent('release', reason, self.client_id)

    def step(self, operator, *, now, clock_id, candidate_ids=(), snapshot=None):
        # Importing the source module only exposes pure validation; main is guarded.
        from tools.copilot.xr_source import validate_operator_packet
        issue = validate_operator_packet(operator, now=now, clock_id=clock_id)
        if issue: return self._reject(issue)
        payload = operator.get('vr_buttons')
        if not isinstance(payload, dict) or payload.get('schema') != SCHEMA or payload.get('mapping') != MAPPING:
            return self._reject('vr_buttons_not_enabled')
        if payload.get('valid') is not True: return self._reject('vr_button_source_invalid')
        issue = states_issue(payload.get('states'))
        if issue: return self._reject(issue)
        epoch = (operator.get('source_instance_id'), operator.get('source_event_clock_id'))
        if any(not isinstance(value, str) or not 0 < len(value) <= 128 for value in epoch):
            return self._reject('vr_source_identity_unknown')
        if (payload.get('source_instance_id'), payload.get('source_event_clock_id')) != epoch:
            return self._reject('vr_source_identity_mismatch')
        sequence = payload.get('event_sequence')
        event_sequences = operator.get('source_event_sequences')
        if (type(sequence) is not int or sequence < 0 or not isinstance(event_sequences, dict)
                or sequence != event_sequences.get('arms')):
            return self._reject('vr_event_sequence_invalid')
        stamp, expiry = payload.get('acquisition_timestamp'), payload.get('expires_at')
        try: bounded = has_vr_time_provenance(payload)
        except ValueError: return self._reject('vr_clock_provenance_invalid')
        if (not number(stamp) or not number(expiry) or payload.get('clock_id') != clock_id
                or not operator['acquisition_timestamp'] <= stamp or not now < expiry
                or expiry > stamp + .15):
            return self._reject('vr_event_stale_or_time_invalid')
        if bounded:
            bound, receipt = (payload[name] for name in VR_TIME_FIELDS)
            if (bound > operator['envelope']['clock_uncertainty_s']
                    or receipt < operator['envelope']['received_at_host_monotonic']
                    or receipt > operator['published_at']):
                return self._reject('vr_clock_provenance_mismatch')
            issue = controller_event_time_issue(stamp, expiry, now=now,
                clock_uncertainty_s=bound, received_at_host_monotonic=receipt)
            if issue: return self._reject(issue)
        elif stamp > now:
            return self._reject('vr_event_stale_or_time_invalid')
        states = payload['states']
        buttons = {side + '.' + name: states[side][name] for side in ('left', 'right') for name in BUTTONS}
        try: signature = json.dumps(payload, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError): return self._reject('vr_payload_not_serializable')
        event = dict(acquisition_timestamp=stamp, clock_id=clock_id,
                     expires_at=min(expiry, operator['expires_at']), event_sequence=sequence)
        if bounded: event.update({name: payload[name] for name in VR_TIME_FIELDS})
        if epoch != self.epoch:
            old_client = self.client_id
            changed = self.epoch is not None
            self.disarm(); self.epoch = epoch; self.sequence = sequence; self.signature = signature
            self.client_id = 'xr:' + epoch[0]
            self.previous = buttons
            self.armed = not any(buttons.values())
            if changed:
                # Reconnection revokes the old source first. A later fresh
                # all-buttons-up event supplies the new core baseline; the
                # release itself must never double as a new-source action.
                self.armed = False
                return VRIntent('release', 'vr_source_reconnected', old_client)
            return VRIntent(None, 'vr_neutral_baseline', self.client_id,
                            **(event if self.armed else {}))
        if sequence < self.sequence or (sequence == self.sequence and signature != self.signature):
            return self._reject('vr_event_regressed_or_reused')
        if sequence == self.sequence: return VRIntent(None, 'vr_event_already_consumed', self.client_id)
        self.sequence, self.signature = sequence, signature
        previous, self.previous = self.previous, buttons
        if not self.armed:
            self.armed = not any(buttons.values())
            return VRIntent(None, 'vr_neutral_baseline_required', self.client_id,
                            **(event if self.armed else {}))
        prior = previous or {}
        edges = {name for name, value in buttons.items() if value and not prior.get(name, True)}
        intent = dict(client_id=self.client_id, **event)
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        face = {
            'X': buttons['left.aButton'],
            'Y': buttons['left.bButton'],
            'A': buttons['right.aButton'],
            'B': buttons['right.bButton'],
        }
        prior_face = {
            'X': prior.get('left.aButton', False),
            'Y': prior.get('left.bButton', False),
            'A': prior.get('right.aButton', False),
            'B': prior.get('right.bButton', False),
        }
        start_stop = all(face.values())
        start_stop_edge = start_stop and not all(prior_face.values())
        ax = face['A'] and face['X'] and not face['B'] and not face['Y']
        prior_ax = (prior_face['A'] and prior_face['X'] and
                    not prior_face['B'] and not prior_face['Y'])
        # Releasing B/Y after calibration must not start teleoperation merely
        # because A/X remain down. Start needs an actual new A or X press.
        ax_edge = ax and not prior_ax and bool(edges & {'left.aButton', 'right.aButton'})
        squeeze_pressed = buttons['left.squeeze'] or buttons['right.squeeze']
        if start_stop_edge:
            if squeeze_pressed:
                return self._reject('vr_chord_requires_released_grips')
            if snapshot.get('clutched') is True:
                self.disarm()
                return VRIntent('release', 'vr_policy_stop_abxy', **intent)
            return VRIntent('clutch', 'vr_calibrate_abxy', **intent)
        if ax_edge:
            if squeeze_pressed:
                return self._reject('vr_chord_requires_released_grips')
            if snapshot.get('clutched') is not True:
                return VRIntent(None, 'vr_calibration_required_before_ax', **intent)
            return VRIntent('start', 'vr_teleop_toggle_ax', **intent)
        actions = [(name, action) for name, action in (
            ('left.bButton', 'select'), ('right.bButton', 'confirm')) if name in edges]
        if len(actions) > 1 or (actions and buttons['right.squeeze']):
            return self._reject('vr_actions_must_be_separate')
        confirmed = snapshot.get('confirmed_candidate_id') if snapshot.get('confirmed') is True else None
        if confirmed != self.confirmed_id:
            self.confirmed_id = confirmed; self.hold_armed = False
        if not buttons['right.squeeze']: self.hold_armed = True
        if actions:
            action = actions[0][1]
            if action == 'select':
                candidates = [c for c in candidate_ids if isinstance(c, str) and 0 < len(c) <= 256]
                if not candidates: return VRIntent(None, 'vr_no_offered_candidates', **intent)
                selected = self.offered_choice or snapshot.get('selected_candidate_id') or confirmed
                index = (candidates.index(selected) + 1) % len(candidates) if selected in candidates else 0
                self.offered_choice = candidates[index]
                return VRIntent('select', 'vr_next_candidate', candidate_id=candidates[index], **intent)
            return VRIntent(action, 'vr_' + action, **intent)
        if snapshot.get('clutched') is True:
            hold = buttons['right.squeeze'] and self.hold_armed and confirmed is not None
            return VRIntent('hold' if hold else 'idle', 'vr_hold' if hold else 'vr_manual_heartbeat', **intent)
        if self.offered_choice is not None:
            if self.offered_choice not in candidate_ids: return self._reject('vr_selected_candidate_unavailable')
            # Preserve an explicit offered choice while waiting for a conscious
            # clutch. Squeeze cannot turn this state into an assist command.
            return VRIntent('idle', 'vr_waiting_for_clutch', **intent)
        return VRIntent(None, 'vr_clutch_required', **intent)
