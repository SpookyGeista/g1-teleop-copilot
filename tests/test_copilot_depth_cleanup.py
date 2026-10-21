"""Depth connectivity tests without scene truth or category-shape assumptions."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop_assist.copilot.depth_cleanup import clean_object_mask, DepthCleanupSettings
from teleop_assist.copilot.observations import GeometryObservation


def scene():
    depth = np.full((40, 60), 1.)
    mask = np.zeros_like(depth, dtype=bool)
    mask[10:25, 20:40] = True
    depth[mask] = .5
    k = np.array([[400., 0., 30.], [0., 400., 20.], [0., 0., 1.]])
    return depth, mask, k, np.eye(4)


def test_connected_target_is_preserved_without_erosion_or_depth_edits():
    depth, mask, k, t = scene()
    before = depth.copy()
    clean, report = clean_object_mask(depth, mask, k, t)
    assert report['accepted']
    assert np.array_equal(clean, mask) and np.array_equal(depth, before)
    assert report['scene_truth_used'] is False
    json.dumps(report, allow_nan=False)


def test_adjacent_image_pixels_with_far_depth_are_removed_and_become_scene():
    depth, mask, k, t = scene()
    mask[10:15, 40:43] = True  # 15 pixels share an image boundary with target
    clean, report = clean_object_mask(depth, mask, k, t)
    assert report['accepted'] and report['removed_valid_mask_pixels'] == 15
    assert not clean[10:15, 40:43].any()
    observation = GeometryObservation('object', 'frame', 1., 'clock', 'cal', 'synthetic_test',
                                      depth, clean, k, t, t)
    obj, environment = observation.point_clouds(max_object_points=10000, max_scene_points=10000)
    original = replace(observation, mask=mask)
    old_obj, old_environment = original.point_clouds(max_object_points=10000, max_scene_points=10000)
    assert len(obj) == len(old_obj) - 15
    assert len(environment) == len(old_environment) + 15
    assert np.allclose(obj[:, 2], .5)


def test_equal_size_competing_components_abstain_instead_of_switching_object():
    depth, mask, k, t = scene()
    mask[10:25, 0:20] = True
    depth[10:25, 0:20] = .8
    clean, report = clean_object_mask(depth, mask, k, t)
    assert not report['accepted'] and not clean.any()
    assert report['reason'] == 'multiple_competing_depth_components'


def test_many_small_far_components_abstain_if_total_loss_is_high():
    depth, mask, k, t = scene()
    mask[:8, :40] = True
    # Individually disconnected metric components, no competing large object.
    depth[:8, :40] = np.where(np.indices((8, 40)).sum(0) % 2, .85, 1.4)
    clean, report = clean_object_mask(depth, mask, k, t)
    assert not report['accepted'] and not clean.any()
    assert report['reason'] in ('excessive_mask_loss', 'multiple_competing_depth_components')


def test_small_invalid_depth_fraction_removed_but_not_fabricated_as_scene():
    depth, mask, k, t = scene()
    depth[12, 23] = np.nan
    clean, report = clean_object_mask(depth, mask, k, t)
    assert report['accepted']
    assert report['invalid_depth_mask_pixels'] == 1
    assert not clean[12, 23]
    assert np.isnan(depth[12, 23])


def test_connected_foreign_surface_is_retained_and_limitation_is_explicit():
    depth, mask, k, t = scene()
    mask[10:15, 40:43] = True
    depth[10:15, 40:43] = .5
    clean, report = clean_object_mask(depth, mask, k, t)
    assert report['accepted'] and np.array_equal(clean, mask)
    assert any('Connected foreign surfaces' in text for text in report['limitations'])


def test_world_transform_does_not_change_connectivity():
    depth, mask, k, t = scene()
    first, report1 = clean_object_mask(depth, mask, k, t)
    t[:3, :3] = Rotation.from_euler('xyz', [.4, -.7, .6]).as_matrix()
    t[:3, 3] = [4., -3., 2.]
    second, report2 = clean_object_mask(depth, mask, k, t)
    assert np.array_equal(first, second)
    c1 = np.array(report1['retained_world_centroid_m'])
    assert np.allclose(report2['retained_world_centroid_m'], t[:3, :3] @ c1 + t[:3, 3])


@pytest.mark.parametrize('bad', ['float_mask', 'intrinsics', 'nonrigid'])
def test_invalid_inputs_raise(bad):
    depth, mask, k, t = scene()
    if bad == 'float_mask':
        mask = mask.astype(float)
    elif bad == 'intrinsics':
        k[0, 0] = 0
    else:
        t[0, 0] = 2
    with pytest.raises(ValueError):
        clean_object_mask(depth, mask, k, t)


def test_small_or_over_budget_masks_abstain():
    depth, mask, k, t = scene()
    clean, report = clean_object_mask(depth, mask, k, t,
                                     settings=DepthCleanupSettings(max_mask_pixels=200))
    assert not clean.any() and report['reason'] == 'mask_exceeds_bounded_CPU_budget'
    clean, report = clean_object_mask(depth, np.zeros_like(mask), k, t)
    assert not clean.any() and report['reason'] == 'insufficient_input_mask_pixels'


def test_fixed_real_capture_removes_depth_outliers_without_truth():
    capture = ROOT / 'runs/copilot_unitree_pipeline_001/observation.npz'
    if not capture.exists():
        pytest.skip('optional recorded camera fixture is not distributed with source')
    observation = GeometryObservation.from_npz(capture)
    clean, report = clean_object_mask(observation.depth, observation.mask, observation.intrinsics,
                                     observation.T_world_camera)
    assert report['accepted']
    assert 0 < report['removed_valid_mask_pixels'] < .15 * observation.mask.sum()
    assert clean.sum() >= .85 * observation.mask.sum()
    assert report['scene_truth_used'] is False
