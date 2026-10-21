"""Read-only display of the one grasp selected by the VR controller.

All hand outlines are supplied by HandPreview's real OPEN Dex3 mesh projection.
This module neither selects a grasp nor grants permission to move. It does not
infer depth or manufacture a hand shape when the bound projection is missing.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BINDING_SCHEMA = 'punktnull.preview_frame.v1'
YELLOW = (255, 209, 74)
GREEN = (82, 231, 151)
MAX_CANDIDATES = 512


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _fresh(value, now, maximum):
    return _number(value) and -.01 <= now - value <= maximum


def _pose(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(not isinstance(row, (list, tuple)) or len(row) != 4 for row in value)
            or any(not _number(v) for row in value for v in row)):
        return None
    pose = np.asarray(value, dtype=float)
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        return None
    return pose


def _same_pose(first, second):
    a, b = _pose(first), _pose(second)
    return a is not None and b is not None and np.allclose(a, b, atol=1e-8, rtol=0)


def _candidate_map(items):
    if not isinstance(items, list) or len(items) > MAX_CANDIDATES:
        return {}
    result = {}
    for item in items:
        if not isinstance(item, dict):
            return {}
        identity = item.get('candidate_id')
        if not isinstance(identity, str) or not 0 < len(identity) <= 512 or identity in result:
            return {}
        result[identity] = item
    return result


def build_operator_grasp_preview(candidates, selected, batch, frame, hand_preview):
    """Project offered goals against the latest independently validated batch.

    A newer observation may validate the same offered poses. Its own unchanged
    source timestamp is retained, never replaced by camera time. The selected
    candidate is only read: this function cannot choose or confirm a control goal.
    """
    batch = batch if isinstance(batch, dict) else {}
    frame = frame if isinstance(frame, dict) else {}
    processed = batch.get('processed_frame')
    current = (isinstance(frame.get('clock_id'), str) and bool(frame['clock_id'])
               and batch.get('clock_id') == frame['clock_id']
               and isinstance(processed, dict) and processed.get('rejected') is False)
    result = dict(candidates=[], clock_id=batch.get('clock_id'),
                  calibration_id=frame.get('calibration_id'),
                  capture_timestamp=batch.get('capture_timestamp'),
                  preview_current=current, motion_static=batch.get('motion_static'))
    if not current:
        return result
    offered, source = _candidate_map(candidates), _candidate_map(batch.get('candidates'))
    from .frames import preview_uv
    for identity, candidate in offered.items():
        original = source.get(identity)
        if not original or not all(_same_pose(candidate.get(key), original.get(key))
                                   for key in ('base_ee', 'world_canonical')):
            continue
        projected = {key: deepcopy(value) for key, value in candidate.items()
                     if key not in ('preview_uv', 'preview_hand_polygons')}
        if (isinstance(selected, dict) and selected.get('candidate_id') == identity
                and all(_same_pose(selected.get(key), candidate.get(key))
                        for key in ('base_ee', 'world_canonical'))):
            projected['preview_uv'] = preview_uv(projected['world_canonical'], frame)
            projected['preview_hand_polygons'] = hand_preview.project(projected['world_canonical'], frame)
        result['candidates'].append(projected)
    return result


def _point(value, width, height):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(_number(v) and abs(v) <= 8 * max(width, height) for v in value))


def _geometry(candidate, width, height):
    polygons = candidate.get('preview_hand_polygons')
    if not isinstance(polygons, list) or not 1 <= len(polygons) <= 64:
        return None
    shapes = []
    for polygon in polygons:
        if not isinstance(polygon, dict):
            return None
        points = polygon.get('uv')
        if (not isinstance(points, list) or not 3 <= len(points) <= 1024
                or not all(_point(p, width, height) for p in points)):
            return None
        shapes.append([tuple(p) for p in points])
    points = np.asarray([p for shape in shapes for p in shape])
    lo, hi = points.min(axis=0), points.max(axis=0)
    if (np.any(hi - lo < 1) or hi[0] < 0 or hi[1] < 0
            or lo[0] >= width or lo[1] >= height):
        return None
    triad = candidate.get('preview_uv')
    if not (isinstance(triad, list) and len(triad) == 4
            and all(_point(p, width, height) for p in triad)):
        triad = None
    return shapes, triad, lo, hi


@lru_cache(maxsize=8)
def _font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_hand(draw, shapes, triad, color, transform=lambda p: p, *, width=2):
    for polygon in shapes:
        points = [transform(p) for p in polygon]
        draw.polygon(points, fill=(*color, 20))
        draw.line(points + [points[0]], fill=(*color, 245), width=width, joint='curve')
    if triad:
        origin = transform(triad[0])
        for point, axis_color in zip(triad[1:], ((241, 107, 105), (100, 224, 146), (124, 178, 255))):
            draw.line([origin, transform(point)], fill=(*axis_color, 235), width=width)


def _hint(camera, total):
    image = Image.fromarray(camera).convert('RGBA')
    overlay = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(overlay)
    x1 = min(image.width - 8, 286)
    draw.rounded_rectangle((10, 10, x1, 83), radius=9, fill=(11, 20, 29, 225),
                           outline=(*YELLOW, 200), width=2)
    draw.text((21, 18), 'Y: Griff anzeigen', font=_font(19), fill=(*YELLOW, 255))
    draw.text((21, 48), f'{total} Vorschlaege bereit', font=_font(14), fill=(237, 242, 246, 255))
    return np.asarray(Image.alpha_composite(image, overlay).convert('RGB')).copy()


def _render(camera, geometry, *, index, total, confirmed):
    shapes, triad, lo, hi = geometry
    h, w = camera.shape[:2]
    color = GREEN if confirmed else YELLOW
    image = Image.fromarray(camera).convert('RGBA')
    overlay = Image.new('RGBA', image.size)
    _draw_hand(ImageDraw.Draw(overlay), shapes, triad, color)
    image = Image.alpha_composite(image, overlay)

    # Enlarge the actual camera pixels and the exact same mesh projection. No
    # invented perspective or alternative hand geometry is shown in this inset.
    panel_width = min(250, max(160, int(w * .38)), w - 20)
    panel_height = min(252, max(150, int(h * .53)), h - 20)
    panel_x = 10
    if lo[0] < panel_width + 20 and lo[1] < panel_height + 20:
        panel_x = w - panel_width - 10
    panel_y = 10
    inner_width, inner_height = panel_width - 16, panel_height - 88
    center = (lo + hi) / 2
    span = np.maximum(hi - lo, 20.) * 1.3
    span[0] = max(span[0], span[1] * inner_width / inner_height)
    span[1] = span[0] * inner_height / inner_width
    # Fit the crop to camera bounds while preserving its aspect ratio.
    scale = min(1., w / span[0], h / span[1])
    span *= scale
    start = np.clip(center - span / 2, [0, 0], [w, h] - span)
    box = (int(start[0]), int(start[1]), max(int(start[0] + span[0]), int(start[0]) + 1),
           max(int(start[1] + span[1]), int(start[1]) + 1))
    crop = Image.fromarray(camera).crop(box).resize((inner_width, inner_height), Image.Resampling.BILINEAR).convert('RGBA')
    enlarged = Image.new('RGBA', crop.size)
    def transform(point):
        return ((point[0] - box[0]) * inner_width / (box[2] - box[0]),
                (point[1] - box[1]) * inner_height / (box[3] - box[1]))
    _draw_hand(ImageDraw.Draw(enlarged), shapes, triad, color, transform, width=2)
    crop = Image.alpha_composite(crop, enlarged)
    panel = Image.new('RGBA', (panel_width, panel_height), (11, 20, 29, 240))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, panel_width-1, panel_height-1), radius=8,
                           outline=(*color, 255), width=2)
    draw.text((9, 7), f'Griff {index}/{total}', font=_font(18), fill=(*color, 255))
    draw.text((9, 31), 'Bestaetigt' if confirmed else 'Ausgewaehlt', font=_font(13), fill=(*color, 255))
    panel.paste(crop, (8, 52))
    draw.text((9, panel_height-28), 'Gruen: bestaetigt' if confirmed else 'Gelb: Vorschau',
              font=_font(12), fill=(*color, 255))
    image.alpha_composite(panel, (panel_x, panel_y))
    return np.asarray(image.convert('RGB')).copy(), [panel_x, panel_y, panel_width, panel_height]


def render_grasp_overlay(camera_rgb, binding, runtime, preview, *, now, clock_id):
    """Return ``(rgb, evidence, header_lines)`` without changing any input.

    The caller owns camera-image binding verification. The projection may come
    from a nearby frame only when both cameras have exactly the same calibrated
    projection and current stationary-scene evidence; frame IDs alone never
    make different camera views compatible. A hidden overlay preserves pixels.
    """
    camera = np.asarray(camera_rgb)
    evidence = dict(schema='punktnull.xr_grasp_overlay.v1', geometry_visible=False,
                    state='waiting', candidate_id=None, confirmed=False,
                    movement_authorized=False, controls_written=False)
    def hidden(reason, text='Griffvorschau wartet auf aktuelle Kameradaten'):
        return camera.copy(), dict(evidence, reason=reason), [text]
    if (camera.dtype != np.uint8 or camera.ndim != 3 or camera.shape[2] != 3
            or not 160 <= camera.shape[1] <= 4096 or not 120 <= camera.shape[0] <= 2160):
        return hidden('camera_format_invalid')
    if not all(isinstance(v, dict) for v in (binding, runtime, preview)) or not _number(now):
        return hidden('overlay_metadata_missing')
    h, w = camera.shape[:2]
    if (runtime.get('clock_id') != clock_id or not _fresh(runtime.get('timestamp'), now, .5)
            or runtime.get('mode') != 'isolated_PhysX'):
        return hidden('runtime_not_current', 'Griffvorschau wartet auf aktuellen Steuerstatus')
    frame = preview.get('preview_frame')
    if (not isinstance(frame, dict) or binding.get('schema') != BINDING_SCHEMA
            or frame.get('schema') != BINDING_SCHEMA):
        return hidden('projection_binding_missing')
    if (preview.get('clock_id') != clock_id or binding.get('clock_id') != clock_id
            or frame.get('clock_id') != clock_id
            or not all(_fresh(stamp, now, .25) for stamp in (
                preview.get('capture_timestamp'), binding.get('capture_timestamp'), frame.get('capture_timestamp')))):
        return hidden('preview_or_camera_not_current')
    if preview.get('preview_current') is not True or preview.get('motion_static') is not True:
        return hidden('preview_not_current_or_stationary')
    if (not isinstance(binding.get('projection_sha256'), str) or len(binding['projection_sha256']) != 64
            or frame.get('projection_sha256') != binding['projection_sha256']
            or not isinstance(binding.get('calibration_id'), str) or not binding['calibration_id']
            or frame.get('calibration_id') != binding['calibration_id']
            or preview.get('calibration_id') != binding['calibration_id']
            or any((b.get('width'), b.get('height')) != (w, h) for b in (binding, frame))
            or abs(binding['capture_timestamp'] - frame['capture_timestamp']) > .25):
        return hidden('preview_projection_mismatch', 'Griffvorschau wartet auf passende Kopfkamera')
    offered, projected = _candidate_map(runtime.get('candidates')), _candidate_map(preview.get('candidates'))
    if not offered:
        return hidden('no_offered_candidates', 'Griffvorschlaege werden vorbereitet')
    # Preserve the controller's Y-cycle ordering, never silently rerank it.
    eligible = {identity: c for identity, c in offered.items() if identity in projected
                and all(_same_pose(c.get(key), projected[identity].get(key))
                        for key in ('base_ee', 'world_canonical'))}
    snapshot = runtime.get('snapshot')
    if not isinstance(snapshot, dict):
        return hidden('selection_status_missing', 'Griffvorschau wartet auf aktuellen Steuerstatus')
    confirmed = snapshot.get('confirmed') is True
    identity = snapshot.get('confirmed_candidate_id') if confirmed else snapshot.get('selected_candidate_id')
    if identity is None and not confirmed:
        if not eligible:
            return hidden('no_matching_offered_preview')
        evidence.update(state='ready', reason='awaiting_vr_selection', candidate_count=len(offered))
        return _hint(camera, len(offered)), evidence, ['Y: Griff anzeigen']
    if not isinstance(identity, str) or identity not in eligible:
        return hidden('selected_candidate_not_matched', 'Gewaehlter Griff wartet auf passende Vorschau')
    geometry = _geometry(projected[identity], w, h)
    if geometry is None:
        return hidden('projected_hand_geometry_unavailable', 'Gewaehlter Griff: Handvorschau noch nicht sichtbar')
    index, total = list(offered).index(identity) + 1, len(offered)
    rgb, inset = _render(camera, geometry, index=index, total=total, confirmed=confirmed)
    evidence.update(geometry_visible=True, state='confirmed' if confirmed else 'selected',
                    reason='bound_selected_grasp_projection', candidate_id=identity, confirmed=confirmed,
                    candidate_index=index, candidate_count=total, inset_xywh=inset,
                    projection_sha256=binding['projection_sha256'],
                    preview_capture_timestamp=frame['capture_timestamp'])
    line = (f'Griff {index}/{total} bestaetigt · Seitengriff rechts halten' if confirmed else
            f'Griff {index}/{total} · B bestaetigen · Y weiter')
    return rgb, evidence, [line]
