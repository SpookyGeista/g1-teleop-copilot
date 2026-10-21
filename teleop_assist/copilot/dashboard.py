"""Loopback UI with per-tab ownership. No robot transport is opened here."""
from __future__ import annotations
from copy import deepcopy
import hmac
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import socket
import threading
import time
from urllib.parse import urlparse, parse_qs
from .mailbox import ControlWriter, atomic_json, read_json, host_clock_id
from .vr_buttons import VRButtonParser, number as finite_number


class CommandConflict(ValueError):
    pass


def _neutral_ack_matches(runtime, baseline, epoch, now):
    """Only acknowledge our still-current neutral baseline; never motion."""
    ack = runtime.get('neutral_source_ui_ack')
    snapshot = runtime.get('snapshot')
    if (not isinstance(ack, dict) or not isinstance(baseline, dict)
            or not isinstance(snapshot, dict) or not isinstance(epoch, tuple) or len(epoch) != 2
            or any(not isinstance(value, str) or not value for value in epoch)):
        return False
    if (ack.get('schema') != 'punktnull.neutral_source_ui_ack.v1'
            or ack.get('motion_authorized') is not False
            or type(ack.get('ui_revision')) is not int
            or type(ack.get('ui_source_event_sequence')) is not int
            or runtime.get('allowed') is not False
            or runtime.get('input_hold_required') is not True
            or runtime.get('command_pending') is not False
            or snapshot.get('clutched') is not False):
        return False
    if (ack.get('ui_revision') != baseline.get('revision')
            or ack.get('ui_client_id') != baseline.get('client_id')
            or ack.get('ui_signature') != baseline.get('ui_signature')
            or ack.get('ui_capture_timestamp') != baseline.get('ui_capture_timestamp')
            or ack.get('ui_source_event_sequence') != baseline.get('ui_source_event_sequence')
            or ack.get('ui_expires_at') != baseline.get('expires_at')
            or (ack.get('source_instance_id'), ack.get('source_event_clock_id')) != epoch
            or epoch != baseline.get('source_epoch')
            or ack.get('clock_id') != runtime.get('clock_id')
            or ack.get('clock_id') != baseline.get('clock_id')
            or not isinstance(ack.get('calibration_id'), str) or not ack['calibration_id']):
        return False
    values = (ack.get('checked_at'), ack.get('valid_until'),
        ack.get('operator_acquisition_timestamp'), baseline.get('acquisition_timestamp'),
        baseline.get('expires_at'), baseline.get('registered_at'), runtime.get('timestamp'), now)
    if not all(finite_number(value) for value in values):
        return False
    return (type(ack.get('source_sequence')) is int
        and type(baseline.get('source_sequence')) is int
        and ack['source_sequence'] >= baseline['source_sequence']
        and ack['operator_acquisition_timestamp'] >= baseline['acquisition_timestamp']
        and baseline['registered_at'] <= ack['checked_at'] <= now
        and runtime['timestamp'] <= now
        and ack['checked_at'] < ack['valid_until']
        and now < ack['valid_until'] <= baseline['expires_at'])


class OperatorOwnership:
    """Selection ownership; the held-assistance lease separately lasts 250 ms."""
    def __init__(self, control, *, clock=time.monotonic, timeout=2.):
        self.control, self.clock, self.timeout = control, clock, timeout
        self.client_id = None
        self.expires_at = 0.

    def expire(self):
        if self.client_id is not None and self.clock() >= self.expires_at:
            self.control.write('release')
            self.client_id = None

    def heartbeat(self, client_id):
        self.expire()
        if self.client_id == client_id:
            self.expires_at = self.clock() + self.timeout

    def require(self, client_id, *, acquire=False):
        self.expire()
        if self.client_id is None and acquire:
            self.client_id = client_id
        if self.client_id != client_id:
            raise CommandConflict('Ein anderes Fenster steuert, oder es wurde noch kein Ziel gewählt.')
        self.expires_at = self.clock() + self.timeout

    def release(self, client_id):
        self.require(client_id)
        packet = self.control.write('release', client_id=client_id)
        self.client_id = None
        return packet

    def snapshot(self, client_id):
        self.expire()
        return dict(client_id=client_id, owner_client_id=self.client_id,
                    owned_by_this_client=self.client_id == client_id,
                    available=self.client_id is None,
                    lease_remaining_s=max(0., self.expires_at - self.clock()) if self.client_id else 0.)


class Dashboard:
    def __init__(self, directory, host='127.0.0.1', port=8770):
        if host not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('This release serves only on loopback')
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.control = ControlWriter(self.directory)
        self.ownership = OperatorOwnership(self.control)
        self.sim_control = ControlWriter(self.directory, filename='sim_control.json')
        self.sim_ownership = OperatorOwnership(self.sim_control)
        self.vr_buttons = VRButtonParser()
        self._vr_pending_revision = None
        self._vr_reconnect_epoch = None
        self._vr_reconnect_baseline = None
        self._vr_deferred_intent = None
        self.lock = threading.RLock()
        self.sessions = {}
        self._closed = self._started = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(2.)

            def log_message(self, *args):
                pass

            def send(self, code, body, kind='application/json'):
                if not isinstance(body, bytes):
                    body = json.dumps(body, allow_nan=False).encode()
                self.send_response(code)
                for key, value in {
                    'Content-Type': kind, 'Content-Length': str(len(body)),
                    'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                    'X-Frame-Options': 'DENY', 'Referrer-Policy': 'no-referrer',
                    'Content-Security-Policy': "default-src 'self'; connect-src 'self'; img-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
                }.items():
                    self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass

            def source_allowed(self):
                hosts = self.headers.get_all('Host', [])
                if len(hosts) != 1 or hosts[0] not in owner.allowed_hosts:
                    return False
                origin = self.headers.get('Origin')
                if origin is not None and origin != 'http://' + hosts[0]:
                    return False
                return self.headers.get('Sec-Fetch-Site') in (None, 'same-origin', 'none')

            def session(self):
                token = self.headers.get('X-Copilot-Token', '')
                if not token or len(token) > 128 or not token.isascii():
                    return None
                now = time.monotonic()
                with owner.lock:
                    for key, item in list(owner.sessions.items()):
                        if now - item['seen_at'] > 300:
                            del owner.sessions[key]
                        elif hmac.compare_digest(key, token):
                            item['seen_at'] = now
                            return item['client_id']
                return None

            def do_GET(self):
                if not self.source_allowed():
                    return self.send(403, {'error': 'Local same-origin request required'})
                path = urlparse(self.path).path
                if path == '/':
                    return self.send(200, Path(__file__).with_name('static').joinpath('index.html').read_bytes(), 'text/html; charset=utf-8')
                if path == '/sim':
                    return self.send(200, Path(__file__).with_name('static').joinpath('sim.html').read_bytes(), 'text/html; charset=utf-8')
                if path == '/api/session':
                    with owner.lock:
                        if owner._closed:
                            return self.send(503, {'error': 'Dashboard closed'})
                        now = time.monotonic()
                        owner.sessions = {key: item for key, item in owner.sessions.items() if now - item['seen_at'] <= 300}
                        if len(owner.sessions) >= 64:
                            return self.send(429, {'error': 'Too many dashboard sessions'})
                        token, client_id = secrets.token_urlsafe(32), secrets.token_hex(16)
                        owner.sessions[token] = dict(client_id=client_id, seen_at=now, last_sequence=0)
                        return self.send(200, {'token': token, 'client_id': client_id})
                if path == '/api/state':
                    client_id = self.session()
                    if client_id is None:
                        return self.send(403, {'error': 'Dashboard session required'})
                    try:
                        with owner.lock:
                            owner.ownership.heartbeat(client_id)
                            ownership = owner.ownership.snapshot(client_id)
                            owner.sim_ownership.heartbeat(client_id)
                            sim_ownership = owner.sim_ownership.snapshot(client_id)
                        return self.send(200, dict(
                            perception=read_json(owner.directory / 'ui_state.json', {}),
                            proposals=read_json(owner.directory / 'preview_proposals.json', None)
                                or read_json(owner.directory / 'latest_proposals.json', {}),
                            xr=read_json(owner.directory / 'xr_status.json', {}),
                            clock_id=host_clock_id(), ownership=ownership,
                            sim=read_json(owner.directory/'sim_runtime.json', {}),sim_ownership=sim_ownership,
                            server_monotonic=time.monotonic(),
                            frame_revision=(owner.directory / 'frame.jpg').stat().st_mtime_ns
                                if (owner.directory / 'frame.jpg').is_file() else None))
                    except (ValueError, OSError):
                        return self.send(503, {'error': 'Runtime state unavailable'})
                if path == '/frame.jpg':
                    frame = owner.directory / 'frame.jpg'
                    try:
                        if frame.is_file() and frame.stat().st_size <= 8 * 1024 * 1024:
                            contents=frame.read_bytes()
                            requested=parse_qs(urlparse(self.path).query,keep_blank_values=True).get('sha256')
                            if requested is not None:
                                if (len(requested)!=1 or len(requested[0])!=64
                                        or any(c not in '0123456789abcdef' for c in requested[0])):
                                    return self.send(400, {'error':'Invalid frame hash'})
                                if not hmac.compare_digest(hashlib.sha256(contents).hexdigest(),requested[0]):
                                    return self.send(409, {'error':'Frame changed; request a fresh bound preview'})
                            return self.send(200, contents, 'image/jpeg')
                    except OSError:
                        pass
                    return self.send(404, {'error': 'No camera frame'})
                return self.send(404, {'error': 'Not found'})

            def do_POST(self):
                if not self.source_allowed():
                    return self.send(403, {'error': 'Local same-origin request required'})
                client_id = self.session()
                if client_id is None:
                    return self.send(403, {'error': 'Dashboard session required'})
                path = urlparse(self.path).path
                if path not in ('/api/control', '/api/target', '/api/sim/control'):
                    return self.send(404, {'error': 'Not found'})
                try:
                    lengths = self.headers.get_all('Content-Length', [])
                    if self.headers.get('Transfer-Encoding') or len(lengths) != 1:
                        raise ValueError('Explicit request length required')
                    length = int(lengths[0])
                    if not 0 < length <= 8192 or self.headers.get('Content-Type', '').split(';', 1)[0].strip() != 'application/json':
                        raise ValueError('Expected small JSON request')
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError('Incomplete command')
                    value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
                    if not isinstance(value, dict):
                        raise ValueError('Command must be an object')
                    with owner.lock:
                        if owner._closed:
                            return self.send(503, {'error': 'Dashboard closed'})
                        sequence = int(self.headers.get('X-Copilot-Sequence', '0'))
                        session_item = next((item for item in owner.sessions.values() if item['client_id'] == client_id), None)
                        if session_item is None:
                            return self.send(403, {'error': 'Dashboard session expired'})
                        if not 0 < sequence < 2 ** 53 or sequence <= session_item['last_sequence']:
                            raise CommandConflict('Veralteter oder ungeordneter Bedienbefehl.')
                        # Late in-flight hold requests cannot overtake a later
                        # release/idle issued by the same browser tab.
                        session_item['last_sequence'] = sequence
                        if path == '/api/sim/control':
                            if set(value)-{'action','candidate_id'} or 'action' not in value:
                                raise ValueError('Unknown simulation command field')
                            action=value['action'];candidate=value.get('candidate_id')
                            if action not in ('select','confirm','clutch','start','hold','release','idle'):
                                raise ValueError('Unknown simulation action')
                            if action != 'select' and candidate is not None:
                                raise ValueError('Candidate only belongs to selection')
                            runtime=read_json(owner.directory/'sim_runtime.json',{})
                            if runtime.get('scripted_operator') is True or runtime.get('scripted_test') is True:
                                return self.send(409,{'error':'Synthetischer Test: nur Beobachtung; Stop über den Launcher.'})
                            if action == 'release':
                                packet=owner.sim_ownership.release(client_id)
                                return self.send(200,{'revision':packet['revision']})
                            timestamp=runtime.get('timestamp')
                            if (runtime.get('schema')!='punktnull.sim_runtime.v1' or runtime.get('mode')!='isolated_PhysX'
                                    or runtime.get('reason')=='runtime_stopped'
                                    or runtime.get('clock_id')!=host_clock_id() or not finite_number(timestamp)
                                    or not 0<=time.monotonic()-timestamp<=.5):
                                return self.send(409,{'error':'Keine frische isolierte PhysX-Simulation verfügbar.'})
                            if action == 'select':
                                if not isinstance(candidate,str) or not 0<len(candidate)<=256:
                                    raise ValueError('Invalid simulation candidate')
                                offered=runtime.get('candidates',[])
                                if not isinstance(offered,list) or not any(isinstance(c,dict) and c.get('candidate_id')==candidate for c in offered):
                                    raise ValueError('Simulation candidate not offered')
                            owner.sim_ownership.require(client_id,acquire=action in ('select','clutch'))
                            packet=owner.sim_control.write(action,candidate,client_id=client_id)
                            return self.send(200,{'revision':packet['revision']})
                        if path == '/api/control':
                            if set(value) - {'action', 'candidate_id'} or 'action' not in value:
                                raise ValueError('Unknown command field')
                            action = value['action']
                            if action not in ('select', 'confirm', 'hold', 'release', 'idle'):
                                raise ValueError('Unknown action')
                            runtime=read_json(owner.directory/'sim_runtime.json',{})
                            if (runtime.get('schema')=='punktnull.sim_runtime.v1'
                                    and runtime.get('mode')=='isolated_PhysX' and action!='release'):
                                return self.send(409,{'error':'Griffansicht ist lokal. Simulation ausschließlich auf /sim bedienen.'})
                            candidate = value.get('candidate_id')
                            if action == 'select':
                                if not isinstance(candidate, str) or not 0 < len(candidate) <= 256:
                                    raise ValueError('Invalid candidate')
                                proposals = read_json(owner.directory / 'latest_proposals.json', {})
                                if not any(isinstance(item, dict) and item.get('candidate_id') == candidate for item in proposals.get('candidates', [])):
                                    raise ValueError('Candidate is not currently offered')
                            elif candidate is not None:
                                raise ValueError('Candidate only belongs to selection')
                            if action == 'release':
                                packet = owner.ownership.release(client_id)
                            else:
                                owner.ownership.require(client_id, acquire=action == 'select')
                                packet = owner.control.write(action, candidate, client_id=client_id)
                            return self.send(200, {'revision': packet['revision']})
                        if set(value) != {'target_id'}:
                            raise ValueError('Unknown object command field')
                        runtime=read_json(owner.directory/'sim_runtime.json',{})
                        if (runtime.get('schema')=='punktnull.sim_runtime.v1'
                                and runtime.get('mode')=='isolated_PhysX'):
                            if runtime.get('scripted_operator') is True or runtime.get('scripted_test') is True:
                                return self.send(409,{'error':'Synthetischer Test: Objektwahl ist nur beobachtbar.'})
                            timestamp=runtime.get('timestamp')
                            if (runtime.get('reason')=='runtime_stopped' or runtime.get('clock_id')!=host_clock_id()
                                    or not finite_number(timestamp) or not 0<=time.monotonic()-timestamp<=.5):
                                return self.send(409,{'error':'Keine aktuelle isolierte Simulation für eine neue Objektwahl.'})
                            other=owner.sim_ownership.snapshot(client_id)
                            if not other['available'] and not other['owned_by_this_client']:
                                raise CommandConflict('Ein anderes Fenster oder VR-Controller bedient die Simulation.')
                        target = value['target_id']
                        if target is not None:
                            if not isinstance(target, str) or not 0 < len(target) <= 256:
                                raise ValueError('Invalid object')
                            perception = read_json(owner.directory / 'ui_state.json', {})
                            if not any(isinstance(item, dict) and item.get('target_id') == target for item in perception.get('objects', [])):
                                raise ValueError('Object is not currently offered')
                        owner.ownership.require(client_id, acquire=True)
                        packet = owner.control.write('release', client_id=client_id)
                        atomic_json(owner.directory / 'target_selection.json', {
                            'target_id': target, 'revision': packet['revision'],
                            'request_id': f'{client_id}:{sequence}',
                            'capture_timestamp': packet['capture_timestamp'], 'clock_id': packet['clock_id']})
                        return self.send(200, {'target_id': target})
                except CommandConflict as exc:
                    return self.send(409, {'error': str(exc)})
                except (ValueError, KeyError, TypeError, RecursionError, TimeoutError):
                    return self.send(400, {'error': 'Invalid command'})
                except OSError:
                    return self.send(503, {'error': 'Runtime mailbox unavailable'})

        class IPv6Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6
        binding = '127.0.0.1' if host == 'localhost' else host
        server_class = IPv6Server if binding == '::1' else ThreadingHTTPServer
        self.http = server_class((binding, port), Handler)
        self.http.daemon_threads = True
        self.address = (self.http.server_address[0], self.http.server_address[1])
        bound_host, bound_port = self.address
        host_label = f'[{bound_host}]' if ':' in bound_host else bound_host
        self.allowed_hosts = {f'{host_label}:{bound_port}', f'localhost:{bound_port}'}
        self.url = f'http://{host_label}:{bound_port}'
        self.thread = threading.Thread(target=self.http.serve_forever,
            kwargs={'poll_interval': .05}, daemon=True, name='copilot-dashboard')

    def consume_vr(self, operator_packet, *, runtime_status, now=None):
        """Optional in-process input bridge, called only by isolated runtime.

        Runtime must reread operator + controls after a slow guard and recheck
        them before applying its validated command. This method never actuates.
        """
        now = time.monotonic() if now is None else now
        with self.lock:
            if getattr(self,'_closed',False): return dict(written=False,reason='dashboard_closed')
            runtime = runtime_status if isinstance(runtime_status,dict) else {}
            if runtime.get('scripted_operator') is True or runtime.get('scripted_test') is True:
                return dict(written=False,reason='vr_disabled_for_scripted_test')
            self.sim_ownership.expire()
            stamp = runtime.get('timestamp')
            valid_runtime = (runtime.get('schema')=='punktnull.sim_runtime.v1'
                and runtime.get('mode')=='isolated_PhysX' and runtime.get('clock_id')==host_clock_id()
                and finite_number(stamp) and 0 <= now-stamp <= .5)
            if not valid_runtime:
                self.vr_buttons.disarm()
                self._vr_pending_revision=None
                self._vr_deferred_intent=None
                if self.sim_ownership.client_id == self.vr_buttons.client_id and self.vr_buttons.client_id is not None:
                    packet=self.sim_ownership.release(self.vr_buttons.client_id)
                    return dict(written=True,reason='isolated_runtime_unavailable',revision=packet['revision'])
                return dict(written=False,reason='isolated_runtime_unavailable')
            candidates=runtime.get('candidates',[])
            candidates=candidates if isinstance(candidates,list) else []
            snapshot=runtime.get('snapshot',{})
            snapshot=snapshot if isinstance(snapshot,dict) else {}
            reconnect_epoch=getattr(self,'_vr_reconnect_epoch',None)
            baseline=getattr(self,'_vr_reconnect_baseline',None)
            neutral_validated = _neutral_ack_matches(runtime, baseline, reconnect_epoch, now)
            if (reconnect_epoch is not None and baseline is not None
                    and snapshot.get('last_ui_revision')==baseline['revision']
                    and snapshot.get('clutched') is False
                    and (neutral_validated or (type(runtime.get('operator_source_sequence')) is int
                    and runtime['operator_source_sequence']>=baseline['source_sequence']
                    and finite_number(runtime.get('operator_acquisition_timestamp'))
                    and runtime['operator_acquisition_timestamp']>=baseline['acquisition_timestamp']))):
                # A UI ACK alone is insufficient: owner/epoch changes ACK UI
                # before rejecting the source. Successful core source
                # provenance proves this neutral baseline crossed both gates.
                self._vr_reconnect_epoch=self._vr_reconnect_baseline=None
                reconnect_epoch=None
            previous_epoch=self.vr_buttons.epoch
            deferred=getattr(self,'_vr_deferred_intent',None)
            if deferred is not None and now>=deferred.expires_at:
                # Fresh held samples cannot renew a previously observed edge.
                self.vr_buttons.disarm()
                self._vr_deferred_intent=None
                deferred=None
            # While neutral synchronization is pending, inspect but do not
            # consume a valid new action edge. A held edge can be consumed once
            # after ACK only from an event whose original XR lease is current.
            # Releases and invalid inputs still commit their disarmed state.
            manual_mode=runtime.get('control_mode')=='unitree_manual'
            parser=deepcopy(self.vr_buttons) if reconnect_epoch is not None or manual_mode else self.vr_buttons
            intent=parser.step(operator_packet,now=now,clock_id=host_clock_id(),
                candidate_ids=[c.get('candidate_id') for c in candidates if isinstance(c,dict)],snapshot=snapshot)
            if (manual_mode and intent.action=='release' and
                    intent.reason not in ('vr_policy_stop_abxy','vr_source_reconnected')):
                # In manual mode an invalid/expired event means no new UI
                # action. Its separate pose watchdog holds measured joints.
                # In particular pose expiry includes clock uncertainty whereas
                # button expiry does not: that boundary must not synthesize a
                # release, erase button history, or revoke the operator owner.
                return dict(written=False,reason='manual_tracking_hold:'+intent.reason)
            if (parser.epoch!=previous_epoch and parser.epoch is not None and
                    (previous_epoch is not None or snapshot.get('last_ui_revision') is not None)):
                self._vr_reconnect_epoch=parser.epoch
                self._vr_reconnect_baseline=None
                self._vr_pending_revision=None
                self._vr_deferred_intent=None
                deferred=None
                reconnect_epoch=parser.epoch
            if reconnect_epoch is not None and intent.action not in (None,'release'):
                if intent.action in ('clutch','start','select','confirm'):
                    if deferred is None:
                        self._vr_deferred_intent=intent
                    elif (intent.action,intent.client_id,intent.candidate_id)!=(
                            deferred.action,deferred.client_id,deferred.candidate_id):
                        parser.disarm()
                        self.vr_buttons=parser
                        self._vr_deferred_intent=None
                return dict(written=False,reason='vr_waiting_for_neutral_core_baseline')
            self.vr_buttons=parser
            if deferred is not None:
                self._vr_deferred_intent=None
                if (intent.action,intent.client_id,intent.candidate_id)==(
                        deferred.action,deferred.client_id,deferred.candidate_id):
                    # This is the first consumption of the original edge, not
                    # a new action licensed by a later held sample.
                    intent=deferred
            if intent.action is None:
                # The core deliberately ignores a persisted first UI action.
                # Prime its startup baseline from a fresh all-buttons-up event,
                # before the user's first real selection/clutch. This neither
                # acquires ownership nor confirms/clutches/holds anything.
                startup_neutral=(intent.reason in ('vr_neutral_baseline','vr_neutral_baseline_required')
                    and snapshot.get('last_ui_revision') is None and snapshot.get('clutched') is not True)
                reconnect_neutral=(reconnect_epoch is not None and intent.acquisition_timestamp is not None
                    and not any(value for side in operator_packet['vr_buttons']['states'].values()
                                for key,value in side.items() if key in ('aButton','bButton','squeeze')))
                if (intent.acquisition_timestamp is not None and (startup_neutral or reconnect_neutral)
                        and self.sim_ownership.client_id is None):
                    pending_baseline=getattr(self,'_vr_reconnect_baseline',None)
                    if reconnect_epoch is not None and pending_baseline is not None:
                        acknowledged=snapshot.get('last_ui_revision')==pending_baseline['revision']
                        expired_and_held=(now>=pending_baseline['expires_at']
                            and runtime['timestamp']>=pending_baseline['expires_at']
                            and runtime.get('input_hold_required') is True
                            and snapshot.get('clutched') is False)
                        if not acknowledged and not expired_and_held:
                            # Several input reads can share the owner's last
                            # published core status. Do not move the revision
                            # ahead of the ACK or refresh an unconsumed lease.
                            return dict(written=False,reason='vr_waiting_for_neutral_core_ack')
                    # A reconnect baseline can have been ACKed only by an
                    # owner/epoch rejection. Retry only from a genuinely fresh
                    # all-up event; never restamp the old mailbox or lease.
                    try:
                        packet=self.sim_control.write('idle',client_id=intent.client_id,
                            source_sample=intent.source_sample())
                    except ValueError:
                        return dict(written=False,reason='vr_neutral_baseline_expired')
                    if reconnect_epoch is not None:
                        self._vr_reconnect_baseline=dict(revision=packet['revision'],
                            registered_at=now,
                            client_id=packet.get('client_id'), clock_id=packet['clock_id'],
                            ui_signature=hashlib.sha256(json.dumps(packet,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                            ui_capture_timestamp=packet['capture_timestamp'],
                            ui_source_event_sequence=packet.get('source_event_sequence'),
                            source_epoch=reconnect_epoch,
                            source_sequence=operator_packet['source_sequence'],
                            acquisition_timestamp=operator_packet['acquisition_timestamp'],
                            expires_at=packet['expires_at'])
                    return dict(written=True,reason='vr_neutral_core_baseline',revision=packet['revision'])
                return dict(written=False,reason=intent.reason)
            if intent.action=='release':
                self._vr_pending_revision=None
                self._vr_deferred_intent=None
                if intent.client_id is None or self.sim_ownership.client_id!=intent.client_id:
                    return dict(written=False,reason=intent.reason)
                # Invalid tracking cancels under the host supervisor; it never
                # creates a fresh hold from an old XR sample.
                if intent.acquisition_timestamp is not None:
                    try:
                        packet=self.sim_control.write('release',client_id=intent.client_id,
                            source_sample=intent.source_sample())
                        self.sim_ownership.client_id=None
                    except ValueError:packet=self.sim_ownership.release(intent.client_id)
                else:packet=self.sim_ownership.release(intent.client_id)
                return dict(written=True,reason=intent.reason,revision=packet['revision'])
            try:
                manual_resume=(runtime.get('control_mode')=='unitree_manual'
                    and intent.action in ('start','idle') and snapshot.get('clutched') is True)
                self.sim_ownership.require(intent.client_id,acquire=intent.action in ('clutch','select') or manual_resume)
            except CommandConflict:
                self.vr_buttons.disarm();self._vr_pending_revision=None
                return dict(written=False,reason='vr_desktop_ownership_conflict')
            ack=snapshot.get('last_ui_revision')
            if self._vr_pending_revision is not None:
                if type(ack) is not int or ack<self._vr_pending_revision:
                    return dict(written=False,reason='vr_waiting_for_core_ack')
                self._vr_pending_revision=None
            try:
                packet=self.sim_control.write(intent.action,intent.candidate_id,client_id=intent.client_id,
                    source_sample=intent.source_sample())
            except ValueError:
                self.vr_buttons.disarm()
                self.sim_ownership.release(intent.client_id)
                return dict(written=False,reason='vr_command_expired_before_write')
            if intent.action in ('clutch','start','select','confirm'):
                self._vr_pending_revision=packet['revision']
            return dict(written=True,reason=intent.reason,revision=packet['revision'])

    def start(self):
        with self.lock:
            if self._closed:
                raise RuntimeError('Dashboard already closed')
            if not self._started:
                self._started = True
                self.thread.start()
        return self

    def close(self):
        with self.lock:
            if self._closed:
                return
            self._closed = True
            self.control.write('release')
            self.sim_control.write('release')
        if self._started:
            self.http.shutdown()
        self.http.server_close()
        if self._started:
            self.thread.join(timeout=3)
