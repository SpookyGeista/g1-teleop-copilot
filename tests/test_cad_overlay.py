import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.cad_overlay import apply_registered_cad_overlays
from teleop_assist import cad_overlay


@pytest.fixture
def inputs(tmp_path):
    # Deliberately off-origin bounds catch confusing object origin and box centre.
    low, high = np.array([-.08, -.07, -.01]), np.array([-.02, .07, .01])
    mesh = tmp_path / 'handle.obj'
    mesh.write_text('\n'.join('v ' + ' '.join(map(str, p)) for p in itertools.product(*zip(low, high))))
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'schema': 'punktnull.known_fixture_assets.v1', 'units': 'meters',
        'objects': [{'id': 'handle', 'mesh_units': 'meters', 'cad_mesh': mesh.name,
                     'cad_sha256': hashlib.sha256(mesh.read_bytes()).hexdigest()}]}))
    t = np.eye(4); t[:3, 3] = [.1, .1, 1.]
    target = {'id': 'a', 'center_world': [.01, .02, 1], 'pose_valid': True,
        'pose_source': 'foundationpose', 'pose_timestamp': 10., 'cad_asset_id': 'handle',
        'cad_grasp_status': {'status': 'accepted', 'cad_asset_id': 'handle'}, 'T_world_object': t.tolist(),
        'box_camera': {'center': [0, 0, 1], 'size': [.4, .4, .4]},
        'bbox_px': [1, 2, 3, 4], 'polygon_px': [[1, 2], [3, 4], [5, 6]], 'task_calibration': {'keep': True}}
    camera = {'width': 640, 'height': 480, 'intrinsics': [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
              'T_world_camera': np.eye(4).tolist()}
    return target, camera, catalog, mesh, low, high


def apply(inputs, target=None, **kw):
    t, camera, catalog, *_ = inputs
    return apply_registered_cad_overlays([t if target is None else target], camera, now=kw.pop('now', 10.1), catalog_path=catalog, **kw)[0]


def test_rotated_box_projection_and_preserved_measured_geometry(inputs):
    t, camera, _, _, low, high = inputs
    world_camera = np.eye(4); world_camera[:3, :3] = Rotation.from_euler('z', 20, degrees=True).as_matrix()
    world_camera[:3, 3] = [.05, -.03, .04]
    camera['T_world_camera'] = world_camera.tolist()
    world_object = np.array(t['T_world_object']); world_object[:3, :3] = Rotation.from_euler('xyz', [10, 15, 35], degrees=True).as_matrix()
    t['T_world_object'] = world_object.tolist()
    before = copy.deepcopy(t)
    out = apply(inputs)
    expected = np.linalg.inv(world_camera) @ world_object
    b = out['box_camera']
    np.testing.assert_allclose(b['center'], expected[:3, :3] @ ((low + high) / 2) + expected[:3, 3])
    np.testing.assert_allclose(b['size'], high-low)
    np.testing.assert_allclose(Rotation.from_quat(b['quaternion_xyzw']).as_matrix(), expected[:3, :3])
    corners = np.array(list(itertools.product(*zip(low, high)))) @ expected[:3, :3].T + expected[:3, 3]
    ph = corners @ np.array(camera['intrinsics']).T; pixels = ph[:, :2] / ph[:, 2, None]
    np.testing.assert_allclose(out['bbox_px'], [*pixels.min(axis=0), *pixels.max(axis=0)])
    assert out['polygon_px'] == []
    assert out['segmentation_polygon_px'] == before['polygon_px']
    assert out['segmentation_box_camera'] == before['box_camera']
    assert out['center_world'] == before['center_world']
    assert out['task_calibration'] == before['task_calibration']
    assert t == before
    assert out['cad_overlay_status']['source'] == 'registered_CAD_bounds'


@pytest.mark.parametrize('change', [
    {'pose_timestamp': 9.}, {'pose_timestamp': 11.}, {'pose_valid': 1},
    {'pose_source': 'simulator_truth'}, {'cad_grasp_status': {'status': 'rejected'}},
    {'cad_asset_id': 'different'},
    {'pose_source': 'foundationpose_rgbd_tracking', 'pose_registration_timestamp': 9.,
     'pose_tracking_method': 'measured_point_translation_for_declared_prismatic_fixture'},
])
def test_rejects_invalid_unaccepted_or_stale_pose(inputs, change):
    t = copy.deepcopy(inputs[0]); t.update(change)
    out = apply(inputs, t)
    assert out['box_camera'] == t['box_camera']
    assert out['bbox_px'] == t['bbox_px']
    assert out['polygon_px'] == t['polygon_px']
    assert out['cad_overlay_status']['status'] != 'accepted'


def test_caches_bounds_but_detects_mesh_change(inputs, monkeypatch):
    mesh = inputs[3]
    unchanged_stat = mesh.stat()
    original_stat = Path.stat
    # Reproduce a same-size write within one filesystem timestamp tick,
    # independently of the host's timestamp resolution or scheduler timing.
    def frozen_mesh_stat(path, *args, **kwargs):
        return unchanged_stat if path == mesh else original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', frozen_mesh_stat)
    assert apply(inputs)['cad_overlay_status']['status'] == 'accepted'
    before = mesh.read_bytes()
    after = before.replace(b'-0.08', b'-0.09')
    assert before != after and len(before) == len(after)
    mesh.write_bytes(after)
    assert mesh.stat() == unchanged_stat
    out = apply(inputs)
    assert out['cad_overlay_status']['status'] == 'rejected'
    assert 'SHA256' in out['cad_overlay_status']['reason']


def test_unchanged_mesh_reuses_parsed_bounds(inputs):
    assert apply(inputs)['cad_overlay_status']['status'] == 'accepted'
    before = cad_overlay._parsed_bounds.cache_info()
    assert apply(inputs)['cad_overlay_status']['status'] == 'accepted'
    after = cad_overlay._parsed_bounds.cache_info()
    assert after.hits == before.hits + 1
    assert after.misses == before.misses


def test_expired_reapplication_restores_segmentation(inputs):
    first = apply(inputs)
    expired = apply(inputs, first, now=11.)
    for key in ('box_camera', 'bbox_px', 'polygon_px'):
        assert expired[key] == inputs[0][key]


@pytest.mark.parametrize('z', [-1., 0.])
def test_behind_or_crossing_camera_retains_measured_box(inputs, z):
    t = copy.deepcopy(inputs[0]); t['T_world_object'][2][3] = z
    out = apply(inputs, t)
    assert out['box_camera'] == t['box_camera']
    assert out['cad_overlay_status']['status'] == 'rejected'
