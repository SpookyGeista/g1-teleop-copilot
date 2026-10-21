"""Isolated read-only simulator-camera candidate for the original Vuer scene.

Importing this module starts no process, server, browser, model or transport.
The original input-only TeleVuer constructor is deliberately NOT called here.
A future caller can install one display-method override before constructing its
staged input wrapper. No control packets, model choices or leases are written.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

ORIGINAL_TELEVUER_SHA256 = '48945f4e90eb158f62dcc0d6a7569e52960e8a5b462989544337af361861e15d'
BINDING_SCHEMA = 'punktnull.preview_frame.v1'
MAX_JPEG_BYTES = 8 * 1024 * 1024
MAX_STATE_BYTES = 2 * 1024 * 1024


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def binding_key(binding):
    # Exactly the existing ProposalSession._publish_image convention.
    return sha(json.dumps({k:v for k,v in binding.items() if k != 'key'},
                          sort_keys=True, allow_nan=False).encode())


def publish_operator_frame(directory, frame, *, grasp_preview=None):
    """Publish a separate display image without model or control ownership.

    The original acquisition time, camera identity and projection are retained.
    This directory cannot substitute for perception or grant movement rights.
    """
    from .mailbox import atomic_json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    encoded = io.BytesIO()
    rgb = np.asarray(frame['rgb'])
    Image.fromarray(rgb).save(encoded, format='JPEG', quality=85)
    jpeg = encoded.getvalue()
    projection = (np.asarray(frame['T_world_camera'], dtype='<f8').tobytes() +
                  np.asarray(frame['intrinsics'], dtype='<f8').tobytes())
    binding = dict(schema=BINDING_SCHEMA, frame_id=int(frame['frame_id']),
        capture_timestamp=float(frame['timestamp']), clock_id=frame['clock_id'],
        calibration_id=frame['calibration_id'], width=int(rgb.shape[1]), height=int(rgb.shape[0]),
        jpeg_sha256=sha(jpeg), projection_sha256=sha(projection))
    binding['key'] = binding_key(binding)
    state = dict(mode='live_operator_camera', frame_id=binding['frame_id'],
        camera_current=True, display_frame=binding,
        camera_source=frame.get('camera_source'), movement_authorized=False)
    if grasp_preview is not None:
        state['grasp_preview'] = dict(grasp_preview, preview_frame=binding)
    # One atomic snapshot for the VR reader. The legacy JPEG/JSON pair remains
    # available to the dashboard, but can straddle two publications.
    metadata = json.dumps(state, allow_nan=False,
        default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)).encode()
    require(len(metadata) <= MAX_STATE_BYTES and len(jpeg) <= MAX_JPEG_BYTES,
            'camera_file_exceeds_bound')
    packet = len(metadata).to_bytes(4, 'big') + metadata + jpeg
    temporary = directory/'display_frame.next.bin'
    temporary.write_bytes(packet)
    temporary.replace(directory/'display_frame.bin')
    temporary = directory/'frame.next.jpg'
    temporary.write_bytes(jpeg)
    temporary.replace(directory/'frame.jpg')
    if grasp_preview is not None:
        # This projection belongs to these exact camera pixels. It does not
        # renew the source proposal timestamp or change a selected grasp.
        atomic_json(directory/'grasp_preview.json', dict(grasp_preview, preview_frame=binding))
    atomic_json(directory/'ui_state.json', {k:v for k,v in state.items() if k != 'grasp_preview'})
    return binding


def bounded_read(path, limit):
    with Path(path).open('rb') as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, 'camera_file_exceeds_bound')
    return data


def read_state(directory):
    return decode_state(bounded_read(Path(directory)/'ui_state.json', MAX_STATE_BYTES))


def decode_state(data):
    def unique(pairs):
        result = {}
        for key,value in pairs:
            require(key not in result, 'duplicate_state_key')
            result[key] = value
        return result
    value = json.loads(data,
                       object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_state')))
    require(isinstance(value, dict), 'camera_state_not_object')
    return value


def read_display_packet(directory):
    try:
        data = bounded_read(Path(directory)/'display_frame.bin', 4+MAX_STATE_BYTES+MAX_JPEG_BYTES)
    except FileNotFoundError:
        return None
    require(len(data) >= 4, 'camera_packet_truncated')
    size = int.from_bytes(data[:4], 'big')
    require(0 < size <= MAX_STATE_BYTES and 4+size < len(data), 'camera_packet_metadata_size')
    jpeg = data[4+size:]
    require(len(jpeg) <= MAX_JPEG_BYTES, 'camera_file_exceeds_bound')
    return decode_state(data[4:4+size]), jpeg


@dataclass(frozen=True)
class DisplaySample:
    rgb: np.ndarray
    evidence: dict


class BoundSimCamera:
    """Read JPEG+binding every display cycle; never make a cached frame younger.

    Host monotonic identity and dimensions are explicit setup inputs. The age
    limit matches the existing 250-ms camera-current publication rule. This is
    an image-visibility rule, never an actuation or collision-safety rule.
    """
    def __init__(self, directory, *, clock_id, width=640, height=480,
                 max_age_s=.25, header_height=64, clock=time.monotonic):
        require(isinstance(clock_id,str) and clock_id.startswith('host_monotonic:'), 'explicit_host_clock_required')
        require(type(width) is int and type(height) is int and 160 <= width <= 4096 and 120 <= height <= 2160,
                'bounded_source_dimensions_required')
        require(number(max_age_s) and 0 < max_age_s <= .25, 'camera_age_limit_cannot_exceed_existing_limit')
        require(type(header_height) is int and 48 <= header_height <= 128, 'bounded_header_required')
        self.directory = Path(directory)
        self.clock_id, self.width, self.height = clock_id, width, height
        self.max_age_s, self.header_height, self.clock = max_age_s, header_height, clock
        self.last_binding = None
        self.examined_binding = None
        self.display_cycle = 0
        self.loaded_state = None
        self.font = ImageFont.load_default()

    def _load(self, now):
        packet = read_display_packet(self.directory)
        state = packet[0] if packet is not None else read_state(self.directory)
        binding = state.get('display_frame')
        require(isinstance(binding,dict) and binding.get('schema') == BINDING_SCHEMA, 'bound_display_frame_missing')
        require(binding.get('key') == binding_key(binding), 'display_binding_hash_mismatch')
        require(binding.get('clock_id') == self.clock_id, 'camera_clock_mismatch')
        require(type(binding.get('frame_id')) is int and binding['frame_id'] >= 0 and
                state.get('frame_id') == binding['frame_id'], 'camera_frame_identity_mismatch')
        require(number(binding.get('capture_timestamp')), 'camera_acquisition_time_missing')
        self.examined_binding = dict(binding)
        age = now - binding['capture_timestamp']
        require(-.01 <= age <= self.max_age_s, 'camera_frame_stale_or_future')
        require(state.get('camera_current') is True and state.get('mode') not in ('review','recorded','replay'),
                'producer_does_not_claim_current_camera')
        require((binding.get('width'),binding.get('height')) == (self.width,self.height), 'camera_dimensions_changed')
        require(isinstance(binding.get('calibration_id'),str) and binding['calibration_id'], 'camera_calibration_identity_missing')
        require(isinstance(binding.get('projection_sha256'),str) and len(binding['projection_sha256']) == 64,
                'camera_projection_binding_missing')
        if self.last_binding is not None:
            old = self.last_binding
            require(binding['calibration_id'] == old['calibration_id'], 'camera_calibration_changed')
            require(binding['frame_id'] >= old['frame_id'] and binding['capture_timestamp'] >= old['capture_timestamp'],
                    'camera_frame_regressed')
            if binding['frame_id'] == old['frame_id']:
                require(binding == old, 'camera_frame_identity_reused')
            else:
                require(binding['capture_timestamp'] > old['capture_timestamp'], 'camera_new_frame_reuses_acquisition_time')
        jpeg = packet[1] if packet is not None else bounded_read(self.directory/'frame.jpg', MAX_JPEG_BYTES)
        require(sha(jpeg) == binding.get('jpeg_sha256'), 'camera_jpeg_hash_mismatch')
        if packet is None:
            second = read_state(self.directory)
            require(second.get('display_frame') == binding and second.get('camera_current') is True,
                    'camera_binding_changed_during_read')
        with Image.open(io.BytesIO(jpeg)) as image:
            require(image.format == 'JPEG' and image.size == (self.width,self.height), 'camera_jpeg_format_or_size_mismatch')
            rgb = np.asarray(image.convert('RGB')).copy()
        # Decode/read time counts. This is a new check time, NOT a new capture.
        finished = self.clock()
        require(number(finished) and finished >= now and -.01 <= finished-binding['capture_timestamp'] <= self.max_age_s,
                'camera_expired_during_display_preparation')
        self.last_binding = dict(binding)
        self.loaded_state = state
        return rgb, dict(binding), finished

    def sample(self):
        self.display_cycle += 1
        checked = self.clock()
        self.examined_binding = None
        binding, camera, reason = None, None, 'camera_unavailable'
        try:
            require(number(checked), 'display_clock_invalid')
            try:
                camera, binding, checked = self._load(checked)
            except ValueError as error:
                # JPEG and metadata are individually atomic. A writer between
                # the two reads is a transient publication race: retry once
                # from disk, with the unchanged source timestamp and age limit.
                if str(error) not in ('camera_jpeg_hash_mismatch','camera_binding_changed_during_read'):
                    raise
                camera, binding, checked = self._load(self.clock())
            reason = 'bound_current_sim_camera'
        except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError,
                UnidentifiedImageError, Image.DecompressionBombError) as error:
            reason = str(error) if isinstance(error,ValueError) else type(error).__name__
            # Metadata is diagnostic only after a failed load; never decode or
            # show its unbound image, and never reuse the previous camera pixels.
            binding = self.examined_binding or self.last_binding
            checked = self.clock()
            if not number(checked): checked = None
        displayed = self._compose(camera, binding, checked, reason)
        # Canvas/text/copies/hash also cost time. A sample that expired while
        # preparing that output is replaced, never re-timestamped.
        if camera is not None and not self.is_current(displayed):
            return self.hidden('camera_expired_during_display_preparation', binding)
        return displayed

    def is_current(self, displayed):
        now = self.clock()
        captured = displayed.evidence['source_capture_timestamp']
        return (displayed.evidence['image_visible'] and number(now) and number(captured)
                and -.01 <= now-captured <= self.max_age_s
                and now >= displayed.evidence['checked_at_host_monotonic'])

    def hidden(self, reason, binding=None):
        checked = self.clock()
        return self._compose(None, binding or self.last_binding, checked if number(checked) else None, reason)

    def _compose(self, camera, binding, checked, reason):
        valid = camera is not None
        age = checked-binding['capture_timestamp'] if binding and number(checked) else None
        frame_label = str(binding['frame_id']) if binding else 'unknown'
        age_label = f'{age*1000:.0f} ms' if number(age) else 'unknown'
        lines = [f'{"SIM CAMERA" if valid else "NO CURRENT SIM IMAGE"} | frame {frame_label} | display cycle {self.display_cycle}',
                 f'Age at source check: {age_label}; NOT headset/display latency.',
                 'If this counter freezes, freshness is unknown. Image grants NO movement permission.']
        # Diagnostic feedback only; no image or movement authorization derives
        # from this adjacent status file. Missing/stale status is never ACTIVE.
        grasp_overlay = None
        if self.directory.name == 'operator_camera':
            control_label = 'STEUERSTATUS UNBEKANNT'
            operator_lines = []
            try:
                state = json.loads(bounded_read(self.directory.parent/'sim_runtime.json', MAX_STATE_BYTES))
                stamp = state.get('timestamp')
                if (number(checked) and number(stamp) and 0 <= checked-stamp <= .5
                        and state.get('clock_id') == self.clock_id
                        and state.get('mode') == 'isolated_PhysX'):
                    snapshot = state.get('snapshot', {})
                    if snapshot.get('clutched') is True:
                        control_label = ('TELEOP AKTIV | AX: Pause | ABXY: Stop' if
                            snapshot.get('teleop_active') is True else 'KALIBRIERT | AX: Start')
                    else:
                        control_label = 'BEWEGUNG GESPERRT | Tasten loslassen, dann ABXY'
                    ai=state.get('ai_assist')
                    if isinstance(ai,dict):
                        if valid:
                            from .xr_grasp_overlay import render_grasp_overlay
                            try:
                                preview = json.loads(bounded_read(self.directory/'grasp_preview.json', MAX_STATE_BYTES))
                            except (OSError, ValueError, TypeError, RecursionError):
                                preview = {}
                            camera, grasp_overlay, operator_lines = render_grasp_overlay(
                                camera, binding, state, preview, now=checked, clock_id=self.clock_id)
                        labels={'learned_grasp_alignment':'GREIFHILFE AKTIV',
                            'learned_approach_guidance':'FRUEHE FUEHRUNG | Kopfkamera',
                            'assist_waiting_for_wrist_depth':'Hilfe wartet auf Handkamera-Tiefe',
                            'move_hand_near_selected_grasp':'Hand naeher zum Griff fuehren'}
                        if ai.get('reason') in labels:
                            operator_lines.append(labels[ai['reason']])
                        distance=ai.get('hand_distance_m')
                        if ai.get('held') and number(distance):
                            operator_lines.append(f'Abstand zum Griff: {distance*100:.0f} cm')
            except (OSError, ValueError, TypeError, AttributeError):
                pass
            lines = [control_label, *operator_lines,
                     'Zeigefinger rechts: greifen | Seitengriff rechts: Hilfe',
                     f'SIM | Bild {frame_label} | Alter {age_label}']
        measure = ImageDraw.Draw(Image.new('RGB',(1,1)))
        wrapped = []
        for line in lines:
            current = ''
            for word in line.split():
                trial = (current+' '+word).strip()
                if current and measure.textbbox((0,0),trial,font=self.font)[2] > self.width-12:
                    wrapped.append(current);current=word
                else: current=trial
            wrapped.append(current)
        header_height = max(self.header_height,10+17*len(wrapped))
        canvas = Image.new('RGB', (self.width,self.height+header_height), (10,14,20))
        if valid:
            canvas.paste(Image.fromarray(camera), (0,header_height))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0,0,self.width-1,header_height-1), fill=(23,52,70) if valid else (102,30,27))
        for index,line in enumerate(wrapped):
            draw.text((6,5+index*17),line,fill=(255,255,255),font=self.font)
        if not valid:
            draw.text((12,header_height+18),'IMAGE HIDDEN: '+reason[:85],fill=(255,193,182),font=self.font)
        pixels = np.asarray(canvas).copy()
        pixels.flags.writeable = False
        return DisplaySample(pixels, dict(schema='punktnull.xr_camera_display_sample.v1',image_visible=valid,
            reason=reason, display_cycle=self.display_cycle, checked_at_host_monotonic=checked,
            camera_row_offset=header_height, header_lines=wrapped,
            grasp_overlay=grasp_overlay,
            source_capture_timestamp=binding['capture_timestamp'] if binding else None,
            source_frame_id=binding['frame_id'] if binding else None,
            source_clock_id=binding['clock_id'] if binding else None,
            source_age_at_preparation_s=age,source_binding_key=binding.get('key') if binding else None,
            source_jpeg_sha256=binding.get('jpeg_sha256') if binding else None,
            movement_authorized=False,headset_delivery_verified=False,
            display_rgb_sha256=sha(pixels.tobytes())))


def native_background(feed, image_schema):
    """Check original acquisition age AFTER native JPEG construction too.

    This bounds only server-side enqueue preparation. It cannot prove delivery
    age or remove the final client texture when the connection/browser freezes.
    """
    displayed = feed.sample()
    def encode(sample):
        return image_schema(sample.rgb,aspect=sample.rgb.shape[1]/sample.rgb.shape[0],
            height=1.8,distanceToCamera=2,format='jpeg',quality=80,key='background-mono',interpolate=True)
    background = encode(displayed)
    if displayed.evidence['image_visible'] and not feed.is_current(displayed):
        displayed = feed.hidden('camera_expired_during_native_jpeg_encoding')
        background = encode(displayed)
    return background, displayed


def install_on_staged_televuer(televuer_class, *, original_source, directory, clock_id,
                              width=640, height=480, display_fps=20., clock=time.monotonic):
    """Prepare one isolated display override; DOES NOT construct/start Vuer.

    Call after staging/instrumentation and before the existing input wrapper
    constructor forks Vuer. That later, explicitly started wrapper stays in
    pass-through mode with zmq=False/webrtc=False; this coroutine adds an ego
    camera panel through the native ImageBackground schema. Only the staged
    in-memory class is changed. No original source file is edited.
    """
    require(sha(Path(original_source).read_bytes()) == ORIGINAL_TELEVUER_SHA256, 'unreviewed_original_televuer_source')
    require(number(display_fps) and 1 <= display_fps <= 30, 'bounded_display_rate_required')
    original = televuer_class.main_pass_through
    require(Path(original.__code__.co_filename).resolve() == Path(original_source).resolve(),
            'display_method_does_not_belong_to_retained_original_source')
    require(getattr(original,'_punktnull_camera_candidate',False) is False, 'camera_candidate_already_installed')
    schemas = SimpleNamespace(**{name:original.__globals__[name] for name in ('Hands','MotionControllers','ImageBackground')})

    async def camera_scene(self, session):
        # Original Vuer close_ws removes this entry. Its display coroutine is
        # not automatically cancelled, so finish when that actual session ends.
        # There is no await between this check and upsert; unrelated schema or
        # serialization assertions must still propagate instead of being hidden.
        def session_open():
            socket = session.vuer.ws.get(session.CURRENT_WS_ID)
            return socket is not None and not socket.closed

        if not session_open():
            return
        # The input components and keys match the untouched original method.
        component = (schemas.Hands(stream=True,key='hands',hideLeft=True,hideRight=True) if self.use_hand_tracking else
                     schemas.MotionControllers(stream=True,key='motionControllers',left=True,right=True))
        session.upsert(component,to='bgChildren')
        if Path(directory).name == 'operator_camera':
            from .xr_panels import stream_operator_panels
            await stream_operator_panels(session, schemas.ImageBackground, directory=directory,
                clock_id=clock_id, width=width, height=height, fps=display_fps,
                clock=clock, session_open=session_open)
            return
        feed = BoundSimCamera(directory,clock_id=clock_id,width=width,height=height,clock=clock)
        pending_image = None
        while session_open():
            # Never accumulate camera frames behind a slow websocket. Keep the
            # original input component and other events untouched in the queue.
            queue=getattr(session,'uplink_queue',())
            if pending_image is not None and any(item is pending_image for item in queue):
                await asyncio.sleep(1./display_fps)
                continue
            # JPEG/mesh drawing must not block incoming controller events on
            # Vuer's asyncio loop. Only one preparation is ever in flight.
            background, displayed = await asyncio.to_thread(native_background,feed,schemas.ImageBackground)
            if displayed.evidence['image_visible'] and not feed.is_current(displayed):
                background, displayed = await asyncio.to_thread(native_background,feed,schemas.ImageBackground)
            if not session_open():
                return
            session.upsert([background],to='bgChildren')
            queue=getattr(session,'uplink_queue',())
            pending_image=queue[-1] if queue else None
            await asyncio.sleep(1./display_fps)

    camera_scene._punktnull_camera_candidate = True
    televuer_class.main_pass_through = camera_scene
    return dict(schema='punktnull.xr_camera_candidate_install.v1',original_source_sha256=ORIGINAL_TELEVUER_SHA256,
        scope='Native ego image panel in staged input-only Vuer; no server started by installer',
        display_layout=('centered_head_with_right_assistance' if Path(directory).name == 'operator_camera' else 'single'),
        display_continuity=('marked_last_valid_frame' if Path(directory).name == 'operator_camera' else 'hide_stale'),
        input_components_preserved=True,control_mailboxes_written=False,movement_authorized=False,
        actual_browser_tested=False,headset_tested=False)
