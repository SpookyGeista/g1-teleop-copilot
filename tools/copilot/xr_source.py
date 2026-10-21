#!/usr/bin/env python3
"""Original TeleVuer wrist inputs as an expiring local mailbox; no robot/DDS.

--check only stages pinned source. A normal invocation explicitly starts a
TeleVuer HTTPS input server, in pass-through mode, loopback by default.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import ipaddress
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import uuid
from types import MethodType
from urllib.parse import urlencode
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from teleop_assist.copilot.mailbox import atomic_json, host_clock_id
from teleop_assist.copilot.xr_adapter import rigid
from teleop_assist.copilot.vr_buttons import source_payload
from tools.copilot.instrument_xr import BROWSER_CHUNK, DEFAULT_CLIENT, install_instrumentation, stage_checkout

SCHEMA = 'punktnull.xr_operator.v1'
FRAME = 'original_G1_29_L_ee_R_ee_nominal_base'
ENVELOPE_FIELDS = ('schema','sequence','acquisition_timestamp','clock_id','clock_uncertainty_s',
    'latency_unknown','sample_kind','tracking_valid','received_at_host_monotonic',
    'sensor_capture_timestamp','timestamp_provenance')


def number(value):
    try: return type(value) in (int,float) and math.isfinite(value)
    except OverflowError: return False


def envelope_issue(envelope, now, clock_id):
    if not isinstance(envelope, dict) or envelope.get('schema') != 'punktnull.operator_envelope.v1':
        return 'operator_schema_unknown'
    if envelope.get('tracking_valid') is not True: return 'tracking_invalid'
    stamp, bound = envelope.get('acquisition_timestamp'), envelope.get('clock_uncertainty_s')
    if not number(stamp): return 'acquisition_time_unknown'
    if envelope.get('clock_id') != clock_id: return 'clock_mismatch'
    if envelope.get('sample_kind') not in ('webxr_pose_read','simulation_pose_sample'): return 'sample_time_provenance_unknown'
    if not number(bound) or not 0 <= bound <= .02: return 'clock_mapping_unbounded'
    if envelope.get('latency_unknown') is not False: return 'latency_unknown'
    if not -.01-bound <= now-stamp <= .15+bound: return 'operator_sample_stale_or_future'
    receipt = envelope.get('received_at_host_monotonic')
    if not number(receipt) or not 0 <= now-receipt <= .15: return 'operator_event_stale'
    if type(envelope.get('sequence')) is not int or envelope['sequence'] < 0: return 'source_sequence_invalid'
    return None


def validate_operator_packet(packet, *, now, clock_id=None):
    """Return a rejection reason, or None; never infer freshness from receipt."""
    clock_id = clock_id or host_clock_id()
    if not isinstance(packet, dict) or packet.get('schema') != SCHEMA: return 'packet_schema_unknown'
    if packet.get('tracking_valid') is not True: return 'packet_tracking_invalid'
    if packet.get('target_frame') != FRAME: return 'target_frame_unknown'
    if type(packet.get('publish_sequence')) is not int or packet['publish_sequence'] < 1: return 'publish_sequence_invalid'
    published = packet.get('published_at')
    if packet.get('publication_clock_id') != clock_id or not number(published) or not 0 <= now-published <= .2:
        return 'publication_stale_or_unknown'
    envelope = packet.get('envelope')
    issue = envelope_issue(envelope, now, clock_id)
    if issue: return issue
    for field, source in (('acquisition_timestamp','acquisition_timestamp'),('clock_id','clock_id'),('source_sequence','sequence')):
        if packet.get(field) != envelope.get(source): return 'packet_envelope_mismatch'
    expiry = min(envelope['acquisition_timestamp']+.15+envelope['clock_uncertainty_s'],
                 envelope['received_at_host_monotonic']+.15)
    if not number(packet.get('expires_at')) or abs(packet['expires_at']-expiry)>1e-8 or now>expiry:
        return 'packet_expired_or_expiry_invalid'
    try:
        if not rigid(packet.get('left_base_ee')) or not rigid(packet.get('right_base_ee')): return 'pose_invalid'
    except (ValueError,TypeError,OverflowError): return 'pose_invalid'
    return None


class OperatorPublisher:
    def __init__(self, output, *, clock=time.monotonic, clock_id=None, vr_buttons=False):
        self.output=Path(output);self.clock=clock;self.clock_id=clock_id or host_clock_id()
        self.publish_sequence=0;self.last_source_sequence=-1
        self.source_instance_id=uuid.uuid4().hex;self.last_browser_clock=None
        self.vr_buttons=bool(vr_buttons)

    def publish(self, tele_data=None, *, reason=None):
        now=self.clock();self.publish_sequence+=1
        raw=getattr(tele_data,'copilot_envelope',{})
        envelope={key:raw.get(key) for key in ENVELOPE_FIELDS} if isinstance(raw,dict) else {}
        # Rejected malformed metadata must still yield a serializable invalid
        # packet; it must not leave the preceding valid mailbox untouched.
        try: json.dumps(envelope,allow_nan=False)
        except (ValueError,TypeError): envelope={}
        issue=reason or envelope_issue(envelope,now,self.clock_id)
        records=raw.get('raw_event_metadata',{}) if isinstance(raw,dict) else {}
        head,arms=(records.get(name,{}) for name in ('head','arms')) if isinstance(records,dict) else ({},{})
        head=head if isinstance(head,dict) else {};arms=arms if isinstance(arms,dict) else {}
        event_sequences={name:record.get('sequence') if type(record.get('sequence')) is int else None
                         for name,record in (('head',head),('arms',arms))}
        browser_clock=arms.get('sample_clock_id')
        same_browser=isinstance(browser_clock,str) and bool(browser_clock) and head.get('sample_clock_id')==browser_clock
        if not issue and (not same_browser or any(v is None or v<0 for v in event_sequences.values())):
            issue='source_event_identity_unknown'
        if not issue:
            # The normalized wrists depend on BOTH head and arm events. A new
            # head sample is a new operator source even without a new HAND_MOVE.
            envelope['sequence']=max(event_sequences.values())
            if self.last_browser_clock is not None and browser_clock!=self.last_browser_clock:
                self.last_source_sequence=-1;issue='source_browser_changed'
            self.last_browser_clock=browser_clock
        sequence=envelope.get('sequence')
        if not issue and sequence<self.last_source_sequence: issue='source_sequence_regressed'
        if not issue: self.last_source_sequence=sequence
        left=right=None
        try:
            left=np.asarray(tele_data.left_wrist_pose,dtype=float)
            right=np.asarray(tele_data.right_wrist_pose,dtype=float)
            if not rigid(left) or not rigid(right): raise ValueError('Invalid pose')
            left,right=left.tolist(),right.tolist()
        except (AttributeError,ValueError,TypeError,OverflowError):
            left=right=None;issue=issue or 'pose_invalid'
        expiry=(min(envelope['acquisition_timestamp']+.15+envelope['clock_uncertainty_s'],
                    envelope['received_at_host_monotonic']+.15) if not issue else None)
        packet=dict(schema=SCHEMA,publish_sequence=self.publish_sequence,source_sequence=sequence,
            source_instance_id=self.source_instance_id,
            source_event_clock_id=browser_clock if isinstance(browser_clock,str) else None,
            source_event_sequences=event_sequences,
            published_at=now,publication_clock_id=self.clock_id,
            acquisition_timestamp=envelope.get('acquisition_timestamp'),clock_id=envelope.get('clock_id'),
            tracking_valid=issue is None,reason=issue or ('valid_simulation_pose_sample'
                if envelope.get('sample_kind') == 'simulation_pose_sample' else 'valid_webxr_api_sample'),
            left_base_ee=left,right_base_ee=right,envelope=envelope,expires_at=expiry,
            command_source='original_Unitree_TeleVuer',target_frame=FRAME,
            physical_sensor_capture_timestamp=None,world_transform_calibrated=False,
            robot_transport_started=False)
        if self.vr_buttons:
            packet['vr_buttons']=source_payload(arms,packet)
            try: json.dumps(packet['vr_buttons'],allow_nan=False)
            except (ValueError,TypeError):
                packet['vr_buttons']={'schema':'punktnull.vr_buttons.v1','mapping':'vuer_controller_v1',
                                      'valid':False,'reason':'malformed_controller_metadata'}
        atomic_json(self.output,packet)
        return packet


def tls_pair(args):
    if bool(args.cert_file) != bool(args.key_file): raise ValueError('Provide both certificate and key paths')
    if args.cert_file: return args.cert_file,args.key_file
    if os.environ.get('XR_TELEOP_CERT') and os.environ.get('XR_TELEOP_KEY'):
        return Path(os.environ['XR_TELEOP_CERT']),Path(os.environ['XR_TELEOP_KEY'])
    user=Path.home()/'.config/xr_teleoperate'
    if (user/'cert.pem').is_file() and (user/'key.pem').is_file(): return user/'cert.pem',user/'key.pem'
    return args.repo/'teleop/televuer/cert.pem',args.repo/'teleop/televuer/key.pem'


def browser_endpoint(host,port):
    """Explicit WSS query avoids the pinned client's HTTPS port omission."""
    address=ipaddress.ip_address(host)
    if type(port) is not int or not 1<=port<=65535:raise ValueError('Invalid input server port')
    if getattr(address,'scope_id',None) is not None:
        raise ValueError('Scoped IPv6 browser URLs are unsupported; use a reachable unscoped address')
    if address.is_unspecified:address=ipaddress.ip_address('::1' if address.version==6 else '127.0.0.1')
    authority=f'[{address}]:{port}' if address.version==6 else f'{address}:{port}'
    websocket=f'wss://{authority}/'
    return dict(browser_url=f'https://{authority}/?'+urlencode({'ws':websocket}),websocket_url=websocket)


def prepare_public_web(staging, *, client_root=DEFAULT_CLIENT):
    """Copy only known public client assets, never the checkout or TLS files.

    The staged build links unchanged assets to the installed Vuer build. Verify
    each link against that exact source before dereferencing it into a public
    copy; arbitrary links added to the staging directory are never followed.
    """
    staging, original = Path(staging).resolve(), Path(client_root).resolve()
    staged = staging/'vuer_client_build'
    evidence = json.loads((staging/'staging_manifest.json').read_text())
    original_chunk = (original/BROWSER_CHUNK).read_bytes()
    if (evidence.get('schema') != 'punktnull.xr_staging.v1'
            or evidence.get('original_browser_sha256') != hashlib.sha256(original_chunk).hexdigest()):
        raise ValueError('Public client source differs from staging provenance')
    public = staging/'public_web'
    public.mkdir()  # A fresh directory is required; never reuse stale assets.
    client, static = public/'client', public/'static'
    client.mkdir(); static.mkdir()
    files = {}; total = 0
    for source in sorted(original.rglob('*')):
        relative = source.relative_to(original)
        name = relative.as_posix()
        if name not in ('index.html', 'editor/index.html') and relative.parts[0] != 'assets':
            continue
        if source.is_dir():
            if source.is_symlink():
                raise ValueError('Public client directory symlink is not permitted')
            continue
        resolved = source.resolve(strict=True)
        if not resolved.is_relative_to(original) or not source.is_file():
            raise ValueError('Public client asset escapes installed build')
        selected = staged/relative
        if name == BROWSER_CHUNK:
            if selected.is_symlink() or selected.resolve(strict=True) != selected:
                raise ValueError('Instrumented public chunk must be a local regular file')
            expected = evidence.get('staged_browser_sha256')
        else:
            if not selected.is_symlink() or selected.resolve(strict=True) != resolved:
                raise ValueError('Public asset is not its known staged source link')
            expected = hashlib.sha256(resolved.read_bytes()).hexdigest()
        size = selected.stat().st_size
        total += size
        if size > 32*1024*1024 or total > 256*1024*1024 or len(files) >= 10000:
            raise ValueError('Public client build exceeds bounded copy size')
        contents = selected.read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        if len(contents) != size or digest != expected:
            raise ValueError('Public client asset changed during copy')
        target = client/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(contents)
        files[name] = dict(sha256=digest, bytes=len(contents))
    if not all(name in files for name in ('index.html', 'editor/index.html', BROWSER_CHUNK)):
        raise ValueError('Required public client assets are missing')
    atomic_json(staging/'public_web_manifest.json', dict(schema='punktnull.xr_public_web.v1',
        files=files, bytes=total, static_root_empty=True,
        scope='Only known instrumented client files; no checkout, run, config or TLS files'))
    return client, static


def install_public_file_routes(vuer, client_root, static_root):
    """Contain every dynamic file route on this locally created Vuer only."""
    from aiohttp import web
    client_root, static_root = Path(client_root).resolve(strict=True), Path(static_root).resolve(strict=True)
    allowed = {client_root, client_root/'assets', static_root}

    def handler(root, fixed=None):
        root = Path(root).resolve(strict=True)
        if root not in allowed:
            raise ValueError('Vuer file route must use the isolated public root')
        async def serve(request):
            name = fixed if fixed is not None else request.match_info.get('filename')
            if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
                    or Path(name).is_absolute() or any(part in ('', '.', '..') for part in name.split('/'))):
                raise web.HTTPNotFound()
            try:
                path = (root/name).resolve(strict=True)
                if not path.is_relative_to(root) or not path.is_file():
                    raise web.HTTPNotFound()
            except (OSError, RuntimeError, ValueError):
                raise web.HTTPNotFound() from None
            return web.FileResponse(path)
        return serve

    def add_static(self, path, root):
        self._add_route(path+'/{filename:.*}', handler(root), method='GET')
    def static_file(self, path, root, filename=None):
        self._add_route(path, handler(root, filename), method='GET')
    vuer.client_root, vuer.static_root = client_root, str(static_root)
    vuer._add_static = MethodType(add_static, vuer)
    vuer._static_file = MethodType(static_file, vuer)


def prepare_wrapper(staging, *, host, port, use_hand_tracking, cert_file, key_file, sim_camera=None):
    """This function deliberately starts Vuer; imports/main --check never do."""
    sys.path.insert(0,str(staging/'teleop/televuer/src'))
    module=importlib.import_module('televuer.televuer')
    expected=(staging/'teleop/televuer/src/televuer/televuer.py').resolve()
    if Path(module.__file__).resolve()!=expected: raise RuntimeError('Expected isolated staged TeleVuer import')
    public_client, public_static = prepare_public_web(staging)
    real_vuer=module.Vuer
    def configured_vuer(*args,**kwargs):
        kwargs.update(host=host,port=port,static_root=str(public_static))
        vuer = real_vuer(*args,**kwargs)
        install_public_file_routes(vuer, public_client, public_static)
        return vuer
    module.Vuer=configured_vuer
    install_instrumentation(public_client)
    if sim_camera is not None:
        from teleop_assist.copilot.xr_camera import install_on_staged_televuer
        camera_report = install_on_staged_televuer(module.TeleVuer, original_source=expected,
            directory=Path(sim_camera).resolve(), clock_id=host_clock_id(), width=640, height=480)
        atomic_json(staging/'sim_camera_manifest.json', dict(camera_report,
            directory=str(Path(sim_camera).resolve()), width=640, height=480, max_age_s=.25))
    wrapper=importlib.import_module('televuer.tv_wrapper').TeleVuerWrapper
    return wrapper(use_hand_tracking=use_hand_tracking,binocular=False,img_shape=(480,640),
        display_mode='pass-through',zmq=False,webrtc=False,
        cert_file=str(cert_file),key_file=str(key_file))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim',action='store_true',help='Mandatory: input source is for isolated simulator only')
    parser.add_argument('--check',action='store_true',help='Stage/check only; no input server')
    parser.add_argument('--repo',type=Path,default=Path('/home/loya2/xr_teleoperate'))
    parser.add_argument('--stage',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/copilot_live/operator.json')
    parser.add_argument('--host',default='127.0.0.1',help='Loopback default; LAN address only when explicitly requested')
    parser.add_argument('--port',type=int,default=8012)
    parser.add_argument('--input-mode',choices=('hand','controller'),default='hand')
    parser.add_argument('--vr-buttons',action='store_true',help='Opt-in isolated-sim controls from instrumented controller events only')
    parser.add_argument('--sim-camera',type=Path,help='Opt-in read-only camera panel from this current simulation directory; no movement permission')
    parser.add_argument('--frequency',type=float,default=60.)
    parser.add_argument('--duration',type=float,default=3600.)
    parser.add_argument('--cert-file',type=Path);parser.add_argument('--key-file',type=Path)
    args=parser.parse_args(argv)
    if not args.sim: parser.error('--sim is mandatory; no robot transport is available')
    if args.vr_buttons and args.input_mode!='controller': parser.error('--vr-buttons requires --input-mode controller')
    if args.sim_camera is not None:
        args.sim_camera=args.sim_camera.expanduser().resolve()
        if not args.sim_camera.is_dir(): parser.error('--sim-camera requires an existing simulation directory')
    if not 1<=args.port<=65535 or not 1<=args.frequency<=120 or not 0<args.duration<=3600:
        parser.error('Invalid port, frequency or bounded duration')
    try: endpoint=browser_endpoint(args.host,args.port);cert,key=tls_pair(args)
    except ValueError as exc: parser.error(str(exc))
    staging=args.stage.resolve() if args.stage else Path(tempfile.mkdtemp(prefix='punktnull_xr_source_'))/'checkout'
    stage_checkout(args.repo,staging,client_root=DEFAULT_CLIENT)
    tls_available=cert.is_file() and key.is_file()
    report=dict(stage=str(staging),tls_files_present=tls_available,host=args.host,port=args.port,
        input_only=True,dds_imported=False,server_started=False,live_headset_verified=False,
        target_frame=FRAME,world_transform_calibrated=False,vr_buttons=args.vr_buttons)
    if args.sim_camera is not None:
        report['sim_camera']=dict(directory=str(args.sim_camera),width=640,height=480,max_age_s=.25,
            read_only=True,movement_authorized=False,headset_delivery_verified=False)
    if args.check:
        print(json.dumps(report,indent=2));return 0
    if not tls_available: parser.error('TLS files unavailable; configure the existing XR certificate/key paths')
    publisher=OperatorPublisher(args.output,vr_buttons=args.vr_buttons);publisher.publish(reason='source_starting')
    stop=False;wrapper=None
    def request_stop(*_):
        nonlocal stop
        stop=True
    previous={n:signal.signal(n,request_stop) for n in (signal.SIGINT,signal.SIGTERM)}
    started=time.monotonic()
    try:
        wrapper=prepare_wrapper(staging,host=args.host,port=args.port,use_hand_tracking=args.input_mode=='hand',
            cert_file=cert,key_file=key,sim_camera=args.sim_camera)
        endpoint_report=dict(schema='punktnull.xr_input_endpoint.v1',host=args.host,port=args.port,
            **endpoint,input_only=True,robot_transport_started=False,live_headset_verified=False)
        if args.sim_camera is not None: endpoint_report['sim_camera']=report['sim_camera']
        print(json.dumps(endpoint_report),flush=True)
        while not stop and time.monotonic()-started<args.duration:
            tick=time.monotonic()
            if not wrapper.tvuer.process.is_alive(): raise RuntimeError('TeleVuer input process stopped')
            publisher.publish(wrapper.get_tele_data())
            time.sleep(max(0.,1/args.frequency-(time.monotonic()-tick)))
    finally:
        publisher.publish(reason='source_stopped')
        if wrapper is not None:
            wrapper.close()
            if wrapper.tvuer.process.is_alive(): wrapper.tvuer.process.kill();wrapper.tvuer.process.join(timeout=2)
        for n,handler in previous.items():signal.signal(n,handler)
    return 0


if __name__=='__main__':raise SystemExit(main())
