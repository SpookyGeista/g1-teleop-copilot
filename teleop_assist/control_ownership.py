"""Single explicit browser operator; observing never changes control input.

This is session arbitration on the local trusted network, not authentication.
Internal simulation drivers call AssistSession directly and do not use it.
"""
import math


class ControlConflict(ValueError):
    pass


class BrowserControlLease:
    def __init__(self, session, timeout=1.0):
        self.session = session
        self.timeout = float(timeout)
        self.owner = None
        self.updated_at = -math.inf
        self.read_only = bool(session.snapshot().get('viewer_only', False))

    def _expire(self):
        if self.owner is not None and self.session.clock()-self.updated_at > self.timeout:
            previous = self.owner
            self.owner = None
            self.session.clear_operator_reference('browser_control_expired')
            self.session.emit('browser_control_expired', client_id=previous)

    def snapshot(self):
        with self.session.lock:
            self._expire()
            return {'owner_client_id': self.owner, 'timeout_s': self.timeout,
                    'age_s': self.session.clock()-self.updated_at if self.owner else None,
                    'explicit_claim_required': True, 'read_only': self.read_only}

    def dispatch(self, data):
        with self.session.lock:
            if self.read_only:
                return {'accepted': False, 'reason': 'Read-only recorded session; no control commands are executed'}
            self._expire()
            kind = data.get('type')
            # A deliberate stop is available to everyone. External calibrated
            # eye devices are a sensor source, separate from arm ownership.
            sample = data.get('sample', data)
            if kind == 'stop' or (kind == 'gaze' and isinstance(sample, dict)
                                  and sample.get('source') != 'webxr_gaze_projection'):
                return self.session.command(data)
            client = data.get('client_id', sample.get('client_id') if isinstance(sample, dict) else None)
            if not isinstance(client, str) or not 1 <= len(client) <= 160:
                raise ControlConflict('A browser client_id and explicit control claim are required')
            if kind == 'control_claim':
                if data.get('deadman_released') is not True:
                    raise ControlConflict('Release the deadman before taking control')
                sample = self.session.input
                fresh = sample is not None and self.session.clock()-self.session.input_time <= self.session.config.input_timeout
                if self.owner != client and fresh and (sample.get('clutch', False) or sample.get('squeeze', 0) > .5):
                    raise ControlConflict('The current operator is holding the deadman')
                if self.owner != client:
                    previous = self.owner
                    self.session.clear_operator_reference('browser_control_claimed')
                    self.owner = client
                    self.session.emit('browser_control_claimed', client_id=client, previous_client_id=previous)
                self.updated_at = self.session.clock()
                return {'accepted': True, 'control_session': self.snapshot()}
            if self.owner != client:
                raise ControlConflict('Observer mode: take control explicitly before sending commands')
            if kind == 'control_heartbeat':
                self.updated_at = self.session.clock()
                return {'accepted': True}
            if kind == 'control_release':
                self.owner = None
                self.session.clear_operator_reference('browser_control_released')
                self.session.emit('browser_control_released', client_id=client)
                return {'accepted': True, 'control_session': self.snapshot()}
            # Validate/process before renewal. Rejected or duplicate inputs
            # cannot extend a dead owner's lease.
            result = self.session.command(data)
            if result.get('accepted', True):
                self.updated_at = self.session.clock()
            return result
