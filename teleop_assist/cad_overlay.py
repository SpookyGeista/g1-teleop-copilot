"""Display-only CAD bounds transformed by an accepted measured object pose.

Call after task_profiles.annotate_tasks. This does not produce a pose, infer an
object identity, or alter grasp/control anchors. Bounds are CAD-local, never
simulator world truth. Failed validation leaves the measured display unchanged.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_CAD_CATALOG = Path(__file__).resolve().parents[1] / 'assets/cad/handle_pose/catalog.json'


def _fingerprint(path):
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


@lru_cache(maxsize=16)
def _catalog(fingerprint):
    value = json.loads(Path(fingerprint[0]).read_text())
    if value.get('schema') != 'punktnull.known_fixture_assets.v1' or value.get('units') != 'meters':
        raise ValueError('Known CAD catalog with meter units required')
    return value


def _bounds(path, expected_sha):
    # Filesystem timestamps can collide for rapid same-size writes. Read/hash
    # this small exported mesh on every check; stat metadata is not proof that
    # previously validated bytes are still present. Parsing stays cached by
    # the actual immutable content, so a cache hit cannot hide mesh tampering.
    data = path.read_bytes()
    if len(expected_sha) != 64 or hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError('CAD mesh SHA256 mismatch')
    return _parsed_bounds(data)


@lru_cache(maxsize=32)
def _parsed_bounds(data):
    vertices = []
    for line in data.decode('utf-8').splitlines():
        fields = line.split()
        if fields and fields[0] == 'v':
            if len(fields) != 4:
                raise ValueError('CAD OBJ vertices must be explicit meter XYZ triples')
            vertices.append([float(x) for x in fields[1:]])
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('CAD OBJ requires finite vertices')
    low, high = points.min(axis=0), points.max(axis=0)
    if np.any(high <= low):
        raise ValueError('CAD bounds must have nonzero volume')
    return low, high


def _rigid(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Finite 4x4 camera and estimated object transforms required')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError('Proper rigid transforms required')
    return matrix


def apply_registered_cad_overlays(targets, camera, *, now=None, catalog_path=None, max_pose_age=.5):
    """Return copied targets with precise CAD display cuboids where admissible.

    ``camera`` accepts the frame's dict intrinsics or the published 3x3 matrix,
    plus width/height and T_world_camera. Time is host monotonic acquisition
    time. Recheck freshness even if the earlier task annotation was accepted.
    ``cad_overlay_status`` explains a fallback; original segmentation remains
    available under segmentation_box_camera/bbox_px/polygon_px. This helper
    requires accepted CAD task annotation and never changes center_world or
    task_calibration. A near-plane crossing retains the measured display.
    """
    now = time.monotonic() if now is None else float(now)
    if not math.isfinite(now) or now < 0 or not math.isfinite(max_pose_age) or not 0 < max_pose_age <= .5:
        raise ValueError('Finite monotonic time and pose age in (0, .5] required')
    result = copy.deepcopy(targets)
    for target in result:
        # Support repeated use on an already annotated target without retaining
        # an expired CAD overlay or overwriting its segmentation evidence.
        if target.get('cad_overlay_status', {}).get('source') == 'registered_CAD_bounds':
            for key in ('box_camera', 'bbox_px', 'polygon_px'):
                backup = 'segmentation_' + key
                if backup in target:
                    target[key] = copy.deepcopy(target[backup])
        status = {'status': 'not_available', 'reason': 'accepted_measured_cad_pose_required'}
        target['cad_overlay_status'] = status
        if target.get('cad_grasp_status', {}).get('status') != 'accepted':
            continue
        status['status'] = 'rejected'
        try:
            if target.get('pose_valid') is not True or target.get('pose_source') not in ('foundationpose', 'foundationpose_rgbd_tracking'):
                raise ValueError('Valid approved measured pose required')
            age = now - float(target['pose_timestamp'])
            if not math.isfinite(age) or age < -.01 or age > max_pose_age:
                raise ValueError('CAD pose stale or future timestamp')
            if target['pose_source'] == 'foundationpose_rgbd_tracking':
                registration_age = now - float(target['pose_registration_timestamp'])
                if not math.isfinite(registration_age) or not 0 <= registration_age <= .7:
                    raise ValueError('Tracked CAD registration stale')
                if target.get('pose_tracking_method') != 'measured_point_translation_for_declared_prismatic_fixture':
                    raise ValueError('Unknown CAD pose tracking method')
            asset_id = target['cad_asset_id']
            if target['cad_grasp_status'].get('cad_asset_id') != asset_id:
                raise ValueError('Accepted CAD identity differs from measured pose')
            catalog_file = Path(catalog_path or DEFAULT_CAD_CATALOG)
            assets = [a for a in _catalog(_fingerprint(catalog_file))['objects'] if a.get('id') == asset_id]
            if len(assets) != 1 or assets[0].get('mesh_units') != 'meters':
                raise ValueError('Exactly one matching meter CAD asset required')
            asset = assets[0]
            mesh = Path(asset['cad_mesh'])
            if not mesh.is_absolute():
                mesh = catalog_file.parent / mesh
            if mesh.suffix.lower() != '.obj':
                raise ValueError('CAD overlay currently requires exported OBJ mesh')
            low, high = _bounds(mesh, asset['cad_sha256'])
            camera_object = np.linalg.inv(_rigid(camera['T_world_camera'])) @ _rigid(target['T_world_object'])
            rotation, translation = camera_object[:3, :3], camera_object[:3, 3]
            local_corners = np.array(list(itertools.product(*zip(low, high))))
            corners = local_corners @ rotation.T + translation
            if np.any(corners[:, 2] <= .001):
                raise ValueError('CAD bounds cross or lie behind camera near plane')
            intrinsics = camera['intrinsics']
            if isinstance(intrinsics, dict):
                intrinsics = [[intrinsics['fx'], 0, intrinsics['cx']], [0, intrinsics['fy'], intrinsics['cy']], [0, 0, 1]]
            k = np.asarray(intrinsics, dtype=float)
            if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
                raise ValueError('Finite calibrated intrinsics required')
            pixels_h = corners @ k.T
            pixels = pixels_h[:, :2] / pixels_h[:, 2, None]
            width, height = float(camera['width']), float(camera['height'])
            if not math.isfinite(width + height) or min(width, height) <= 0:
                raise ValueError('Positive image dimensions required')
            xy_min = np.maximum(pixels.min(axis=0), [0, 0])
            xy_max = np.minimum(pixels.max(axis=0), [width, height])
            if np.any(xy_max <= xy_min):
                raise ValueError('CAD bounds outside current camera image')
            for key in ('box_camera', 'bbox_px', 'polygon_px'):
                target.setdefault('segmentation_' + key, copy.deepcopy(target.get(key, [])))
            target['box_camera'] = {'center': (rotation @ ((low + high) / 2) + translation).tolist(),
                                    'size': (high - low).tolist(),
                                    'quaternion_xyzw': Rotation.from_matrix(rotation).as_quat().tolist(),
                                    'source': 'registered_CAD_bounds'}
            target['bbox_px'] = [*xy_min.tolist(), *xy_max.tolist()]
            # The 2-D UI intentionally falls back to the projected CAD rectangle.
            target['polygon_px'] = []
            target['cad_overlay_status'] = {'status': 'accepted', 'source': 'registered_CAD_bounds',
                'cad_asset_id': asset_id, 'cad_sha256': asset['cad_sha256'],
                'pose_source': target['pose_source'], 'pose_timestamp': target['pose_timestamp'], 'age_s': age,
                'bounds_frame': 'CAD_object_local', 'projection': 'clipped_projected_cuboid_bbox'}
        except (KeyError, TypeError, ValueError, OSError, UnicodeError) as error:
            status['reason'] = str(error)
    return result
