"""Independent operator and assistance screens for the isolated VR simulator.

Cached pixels are only a display continuity mechanism, never fresh observations
for control. Acquisition timestamps and the 250 ms perception gate stay intact.
"""
from __future__ import annotations

import asyncio
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .xr_camera import (BoundSimCamera, DisplaySample, MAX_STATE_BYTES,
                        bounded_read, number)


PANEL_HEADER = 128
PANEL_LAYOUT = {
    'head': dict(key='operator-head', height=1.8, position=[0, 0, 0]),
    'assist': dict(key='operator-assist', height=1.3, position=[1.74, 0, 0]),
}


class CameraPanel(BoundSimCamera):
    def __init__(self, directory, *, role, **kwargs):
        super().__init__(directory, **kwargs)
        if role not in PANEL_LAYOUT:
            raise ValueError('Unknown camera panel')
        self.role = role
        self.cached_rgb = None
        self.cached_binding = None
        self.cached_preview = {}
        try:
            self.panel_font = ImageFont.truetype('DejaVuSans.ttf', 16)
        except OSError:
            self.panel_font = self.font

    def sample(self):
        self.display_cycle += 1
        now = self.clock()
        reason = 'bound_current_sim_camera'
        try:
            camera, binding, now = self._load(now)
            self.cached_rgb, self.cached_binding = camera, binding
            self.cached_preview = (self.loaded_state or {}).get('grasp_preview', {})
        except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError,
                Image.DecompressionBombError) as error:
            reason = str(error) if isinstance(error, ValueError) else type(error).__name__
        return self.compose(self.clock(), reason)

    def compose(self, now, reason):
        binding = self.cached_binding
        age = now-binding['capture_timestamp'] if binding and number(now) else None
        current = (reason == 'bound_current_sim_camera' and number(age)
                   and -.01 <= age <= self.max_age_s)
        # Fixed dimensions, even if status text/selection changes. The operator
        # camera never passes through the mesh renderer or preview metadata.
        camera = self.cached_rgb.copy() if self.cached_rgb is not None else None
        overlay = None
        control = 'Steuerstatus unbekannt'
        state = {}
        try:
            state = json.loads(bounded_read(self.directory.parent/'sim_runtime.json', MAX_STATE_BYTES))
            stamp = state.get('timestamp')
            if not (number(now) and number(stamp) and 0 <= now-stamp <= .5
                    and state.get('clock_id') == self.clock_id and state.get('mode') == 'isolated_PhysX'):
                state = {}
            snapshot = state.get('snapshot', {})
            if state:
                control = ('AX: Start' if snapshot.get('clutched') else 'ABXY: Kalibrieren')
                if snapshot.get('teleop_active'):
                    control = 'TELEOP AKTIV | AX: Pause | ABXY: Stop'
        except (OSError, ValueError, TypeError, AttributeError, RecursionError):
            state = {}

        title = 'KOPFKAMERA' if self.role == 'head' else 'ASSISTENZ / GRIFFVORSCHAU'
        freshness = 'LIVE' if current else ('BILD VERZOEGERT' if age is not None and age <= 1. else 'STANDBILD - KEIN LIVEBILD')
        if camera is None:
            freshness = 'WARTE AUF KAMERABILD'
        lines = [f'{title} | {freshness}', control]
        if self.role == 'head':
            lines += ['Zeigefinger rechts: greifen | Seitengriff rechts: Hilfe']
        else:
            extra = ['Y: Griff wechseln | B: bestaetigen']
            if current:
                from .xr_grasp_overlay import render_grasp_overlay
                try:
                    camera, overlay, extra = render_grasp_overlay(camera, binding, state,
                        self.cached_preview, now=now, clock_id=self.clock_id)
                except Exception as error:
                    # Drawing is display-only. Keep its failure local to this
                    # panel and preserve the independent operator image.
                    extra = ['Griffvorschau voruebergehend nicht verfuegbar']
                    overlay = {'visible':False, 'reason':type(error).__name__}
            elif camera is not None:
                extra = ['Keine aktuelle Griffvorschau']
            lines += extra
            ai = state.get('ai_assist', {})
            if not isinstance(ai, dict):
                ai = {}
            labels = {'learned_grasp_alignment':'GREIFHILFE AKTIV',
                'learned_approach_guidance':'FRUEHE FUEHRUNG | Kopfkamera',
                'assist_waiting_for_wrist_depth':'Hilfe wartet auf Handkamera-Tiefe',
                'move_hand_near_selected_grasp':'Hand naeher zum Griff fuehren'}
            if ai.get('reason') in labels:
                lines.append(labels[ai['reason']])
        # The final line stays visible regardless of the amount of help text.
        detail = f'SIM | Bild {binding["frame_id"]} | Bildalter {age*1000:.0f} ms' if binding and number(age) else 'SIM | Keine Bilddaten'
        lines = lines[:5] + [detail]
        canvas = Image.new('RGB', (self.width, self.height+PANEL_HEADER), (30,38,48))
        if camera is not None:
            canvas.paste(Image.fromarray(camera), (0, PANEL_HEADER))
        draw = ImageDraw.Draw(canvas)
        color = (23,52,70) if current else (128,64,16)
        draw.rectangle((0,0,self.width-1,PANEL_HEADER-1), fill=color)
        for row, line in enumerate(lines):
            # Bound text to its own panel; never change texture size.
            while line and draw.textbbox((0,0),line,font=self.panel_font)[2] > self.width-16:
                line = line[:-1]
            draw.text((8,6+row*20), line, font=self.panel_font, fill='white')
        if not current and camera is not None:
            draw.rectangle((1,PANEL_HEADER,self.width-2,self.height+PANEL_HEADER-2),
                           outline=(255,170,55), width=5)
            # Long outages must be unmistakable, without flashing to black.
            if not number(age) or age > 1.:
                y = PANEL_HEADER + self.height//2
                draw.rectangle((0,y-16,self.width,y+20),fill=(128,64,16))
                draw.text((12,y-10),'STANDBILD - Verbindung / Simulation pruefen',
                          font=self.panel_font,fill='white')
        rgb = np.asarray(canvas).copy()
        rgb.flags.writeable = False
        return DisplaySample(rgb, dict(panel=self.role, image_visible=camera is not None,
            camera_current=current, cached_display=not current and camera is not None,
            reason=reason, header_lines=lines, camera_row_offset=PANEL_HEADER,
            source_capture_timestamp=binding['capture_timestamp'] if binding else None,
            source_frame_id=binding['frame_id'] if binding else None,
            source_age_at_preparation_s=age, checked_at_host_monotonic=now,
            grasp_overlay=overlay, movement_authorized=False, headset_delivery_verified=False))


def native_panel(feed, image_schema):
    sample = feed.sample()
    def encode(value):
        return image_schema(value.rgb, aspect=feed.width/(feed.height+PANEL_HEADER),
            distanceToCamera=2, format='jpeg', quality=80 if feed.role == 'head' else 70,
            interpolate=True, geometry='plane', **PANEL_LAYOUT[feed.role])
    result = encode(sample)
    if sample.evidence['camera_current'] and not feed.is_current(sample):
        sample = feed.compose(feed.clock(), 'camera_expired_during_native_jpeg_encoding')
        result = encode(sample)
    return result, sample


async def stream_operator_panels(session, image_schema, *, directory, clock_id,
                                 width, height, fps, clock, session_open):
    directory = Path(directory)
    feeds = {role:CameraPanel(path, role=role, clock_id=clock_id, width=width, height=height, clock=clock)
             for role,path in (('head',directory),('assist',directory.parent/'assist_camera'))}
    periods = {'head':1/fps, 'assist':1/min(fps,8.)}
    due = {role:0. for role in feeds}
    tasks = {}
    pending = None
    counts = Counter()
    samples = {}
    last_report = 0.
    loop = asyncio.get_running_loop()
    try:
        while session_open():
            queue = getattr(session, 'uplink_queue', ())
            if pending is not None and any(item is pending for item in queue):
                counts['queue_waits'] += 1
                await asyncio.sleep(.02)
                continue
            tick = loop.time()
            for role in feeds:
                if role not in tasks and tick >= due[role]:
                    tasks[role] = asyncio.create_task(asyncio.to_thread(native_panel, feeds[role], image_schema))
                    due[role] = tick+periods[role]
            if not tasks:
                await asyncio.sleep(max(.001,min(.05,min(due.values())-tick)))
                continue
            done, _ = await asyncio.wait(tasks.values(), timeout=.05, return_when=asyncio.FIRST_COMPLETED)
            elements = []
            for role, task in list(tasks.items()):
                if task not in done:
                    continue
                del tasks[role]
                try:
                    element, sample = task.result()
                except Exception:
                    # Retain this panel's last client texture; retry only this
                    # producer. In particular, preview encoding cannot stop head video.
                    counts[role+'_preparation_errors'] += 1
                    continue
                # Queuing or thread scheduling can consume the remaining age.
                if sample.evidence['camera_current'] and not feeds[role].is_current(sample):
                    due[role] = 0.
                    continue
                elements.append(element)
                samples[role] = sample.evidence
                counts[role+'_sent'] += 1
                counts[role+'_cached'] += int(sample.evidence['cached_display'])
                counts[role+'_read_'+sample.evidence['reason']] += 1
            if not session_open():
                return
            if elements:
                session.upsert(elements, to='bgChildren')
                queue = getattr(session, 'uplink_queue', ())
                pending = queue[-1] if queue else None
            if tick-last_report >= 2.:
                from .mailbox import atomic_json
                atomic_json(directory/'stream_health.json', dict(timestamp=clock(),
                    clock_id=clock_id, counters=dict(counts), panels=samples,
                    headset_delivery_verified=False))
                last_report = tick
    finally:
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
