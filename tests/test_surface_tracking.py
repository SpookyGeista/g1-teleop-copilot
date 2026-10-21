"""CPU RGB-D fixtures; no SAM, simulator truth input, GPU or motion authority."""
import copy
from dataclasses import replace

import cv2
import numpy as np
import pytest

from teleop_assist.learned_perception import LearnedPerceptionConfig, mask_to_target
from teleop_assist.surface_tracking import CONTRACT, measure


def plane(frame_id=0, timestamp=10., camera_x=0., camera_z=0., object_x=0., object_z=.5):
    h, w = 120, 160
    K = np.array([[200., 0, 80.], [0, 200., 60.], [0, 0, 1.]])
    T = np.eye(4); T[0, 3] = camera_x; T[2, 3] = camera_z
    y, x = np.indices((h, w)); z = object_z-camera_z
    wx = (x-K[0, 2])*z/K[0, 0]+camera_x
    wy = (y-K[1, 2])*z/K[1, 1]
    mask = (abs(wx-object_x) <= .050000001) & (abs(wy) <= .040000001)
    depth = np.full((h, w), .8-camera_z, np.float32); depth[mask] = z
    rgb = np.full((h, w, 3), [25, 80, 130], np.uint8); rgb[mask] = [175, 40, 30]
    return dict(frame_id=frame_id, timestamp=timestamp, camera_id='owner_primary_channel',
        clock_id='synthetic_monotonic', calibration_id='synthetic_rectified',
        intrinsics=K, T_world_camera=T, T_world_base=np.eye(4), depth=depth, rgb=rgb), mask


def initial():
    previous, mask = plane()
    target, reason = mask_to_target(mask, previous['depth'], previous['intrinsics'],
        previous['T_world_camera'], label='synthetic surface', detection_score=.9, mask_score=.9)
    assert reason is None
    target.update(id='selected', frame_id=0, timestamp=10., semantic_keyframe_timestamp=10.)
    return previous, mask, target


def evaluate(previous=None, mask=None, target=None, current=None, **kwargs):
    old, original, prior = initial()
    previous = old if previous is None else previous
    mask = original if mask is None else mask
    target = prior if target is None else target
    current = plane(1, 10.1)[0] if current is None else current
    kwargs.setdefault('now', current['timestamp'])
    kwargs.setdefault('selected_target_id', 'selected')
    return measure(previous, mask, target, current, kwargs.pop('semantic_timestamp', 10.), **kwargs)


def rejected(result, reason=None):
    assert result['target'] is None
    assert result['mask'] is None
    assert result['reason']
    if reason is not None:
        assert result['reason'] == reason
    assert result['quality']['semantic_identity_proven'] is False
    assert result['quality']['motion_authorized'] is False


def test_textureless_current_measurement_needs_no_lk_or_sam(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Surface measurement cannot require feature detection')
    monkeypatch.setattr(cv2, 'goodFeaturesToTrack', forbidden)
    result = evaluate()
    assert result['reason'] is None
    target = result['target']
    assert target['frame_id'] == 1 and target['timestamp'] == 10.1
    assert target['semantic_keyframe_timestamp'] == 10.
    assert target['semantic_keyframe_age_s'] == pytest.approx(.1)
    assert target['identity_proven'] is False and target['motion_authorized'] is False
    assert result['quality']['method'] == 'measured_rgbd_surface_reprojection'
    assert target['normal_source'] == 'supported_current_measured_depth'
    assert not {'flow_residual_px', 'inlier_count', 'inlier_fraction'} & result['quality'].keys()
    assert np.array_equal(result['mask'], initial()[1])


@pytest.mark.parametrize('selection', [None, '', 'wrong', True])
def test_explicit_matching_selection_required(selection):
    rejected(evaluate(selected_target_id=selection), 'selected_identity_mismatch')


@pytest.mark.parametrize('field', ['camera_id', 'clock_id', 'calibration_id', 'source_instance_id'])
def test_source_change_rejects(field):
    current = plane(1, 10.1)[0]; current[field] = 'different'
    rejected(evaluate(current=current), 'camera_clock_calibration_or_source_changed')


def test_explicit_matching_epoch_and_owner_channel_preserved():
    previous, mask, target = initial(); current = plane(1, 10.1)[0]
    previous['source_instance_id'] = current['source_instance_id'] = 'actual-owner-epoch'
    result = evaluate(previous, mask, target, current)
    assert result['reason'] is None
    assert result['quality']['camera_id'] == 'owner_primary_channel'
    assert result['quality']['clock_id'] == 'synthetic_monotonic'


@pytest.mark.parametrize('field', ['clock_id', 'camera_id', 'calibration_id'])
def test_missing_owner_frame_identity(field):
    current = plane(1, 10.1)[0]; current.pop(field)
    rejected(evaluate(current=current))


@pytest.mark.parametrize('field,value', [('frame_id', 0), ('timestamp', 10.),
    ('frame_id', True), ('timestamp', float('nan')), ('timestamp', float('inf'))])
def test_nonadvancing_or_invalid_source_identity(field, value):
    current = plane(1, 10.1)[0]; current[field] = value
    rejected(evaluate(current=current))


def test_gap_stale_future_and_consumer_semantic_expiry():
    rejected(evaluate(current=plane(2, 10.651)[0]), 'frame_gap_exceeded')
    rejected(evaluate(now=10.351), 'current_observation_stale')
    rejected(evaluate(now=10.089), 'current_observation_stale')
    previous, mask, target = initial(); previous['timestamp'] = target['timestamp'] = 10.6
    current = plane(1, 10.9)[0]
    rejected(evaluate(previous, mask, target, current, now=11.001), 'original_semantic_evidence_expired')


@pytest.mark.parametrize('semantic', [float('nan'), float('inf'), -1., 10.01, True])
def test_invalid_semantic_timestamp(semantic):
    rejected(evaluate(semantic_timestamp=semantic), 'original_semantic_evidence_expired')


def test_original_semantics_cannot_be_refreshed_by_argument_or_continuation():
    previous, mask, target = initial()
    target['semantic_keyframe_timestamp'] = 9.9
    rejected(evaluate(previous, mask, target), 'previous_target_provenance_mismatch')
    previous, mask, target = initial()
    for i, timestamp in enumerate((10.4, 10.8, 11.), start=1):
        current = plane(i, timestamp)[0]
        result = evaluate(previous, mask, target, current)
        assert result['reason'] is None
        assert result['target']['semantic_keyframe_timestamp'] == 10.
        previous, mask, target = current, result['mask'], result['target']
    rejected(evaluate(previous, mask, target, plane(4, 11.001)[0]), 'original_semantic_evidence_expired')


@pytest.mark.parametrize('field', ['center_world', 'normal_world'])
@pytest.mark.parametrize('value', [[float('nan'), 0., 1.], [float('inf'), 0., 1.], [0., 1.], [0., 0., 0.]])
def test_nonfinite_or_invalid_previous_geometry(field, value):
    previous, mask, target = initial(); target[field] = value
    result = evaluate(previous, mask, target)
    # A finite zero world center is a permissible coordinate, unlike zero normal.
    if field == 'center_world' and value == [0., 0., 0.]:
        rejected(result, 'previous_anchor_disagreement')
    else:
        rejected(result)


@pytest.mark.parametrize('field', ['detection_confidence', 'segmentation_confidence',
    'pose_confidence', 'position_sigma_m'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.01, True])
def test_invalid_prior_quality_is_never_used(field, value):
    previous, mask, target = initial(); target[field] = value
    rejected(evaluate(previous, mask, target))


@pytest.mark.parametrize('field,value', [('center', [float('nan'), 0, 1]), ('center', [0, 1]),
    ('size', [-1., .1, .1]), ('size', [.1, float('inf'), .1]),
    ('quaternion_xyzw', [0., 0., 0., 0.]), ('quaternion_xyzw', [0., 0., 0., 2.])])
def test_invalid_previous_box(field, value):
    previous, mask, target = initial(); target['box_camera'][field] = value
    rejected(evaluate(previous, mask, target))


@pytest.mark.parametrize('depth', [float('nan'), float('inf'), .3, .8, .513])
def test_missing_or_occluding_current_depth(depth):
    current, mask = plane(1, 10.1); current['depth'][mask] = depth
    rejected(evaluate(current=current), 'insufficient_native_surface_support')


def test_changed_rgb_cannot_be_substituted_for_current_measurement():
    current, mask = plane(1, 10.1); current['rgb'][mask] = [20, 200, 190]
    rejected(evaluate(current=current), 'insufficient_native_surface_support')
    # Strictly less than35, matching the unchanged exploratory prototype.
    current, mask = plane(1, 10.1); current['rgb'][mask] += 35
    rejected(evaluate(current=current), 'insufficient_native_surface_support')


def test_invalid_previous_depth_and_mask_types():
    previous, mask, target = initial(); previous['depth'][mask] = np.nan
    rejected(evaluate(previous, mask, target), 'insufficient_previous_depth')
    rejected(evaluate(mask=mask.astype(np.uint8)), 'invalid_previous_mask')


def test_sparse_failed_pixels_are_never_filled_or_dilated():
    current, mask = plane(1, 10.1)
    yy, xx = np.nonzero(mask); absent = (yy % 7 == 0) & (xx % 5 == 0)
    current['depth'][yy[absent], xx[absent]] = np.nan
    result = evaluate(current=current)
    assert result['reason'] is None
    assert not result['mask'][yy[absent], xx[absent]].any()
    assert not (result['mask'] & ~mask).any()


@pytest.mark.parametrize('camera_x,camera_z', [(.03, 0.), (0., .1)])
def test_measured_camera_motion_preserves_world_surface(camera_x, camera_z):
    current, actual = plane(1, 10.1, camera_x=camera_x, camera_z=camera_z)
    result = evaluate(current=current)
    assert result['reason'] is None
    assert not (result['mask'] & ~actual).any()
    assert np.linalg.norm(np.array(result['target']['center_world'])-initial()[2]['center_world']) < .003


def test_occluded_rear_surface_never_replaces_conflicting_front_zbuffer():
    previous, _, target = initial(); current = plane(1, 10.1)[0]
    mask = np.zeros(previous['depth'].shape, bool)
    mask[45:75, 40:50] = True; mask[45:75, 30:40] = True
    previous['depth'][45:75, 40:50] = .5; previous['depth'][45:75, 30:40] = 1.
    previous['rgb'][mask] = [175, 40, 30]
    current['T_world_camera'][0, 3] = .05
    current['depth'][:] = 1.; current['rgb'][:] = [175, 40, 30]
    result = evaluate(previous, mask, target, current)
    rejected(result, 'insufficient_native_surface_support')
    assert result['quality']['current_depth_supported_pixels'] == 0
    # x40..49 at z=.5 shifts20px; x30..39 at z=1 shifts10px.
    # Both are the same30x10 current pixels; rear matches may not replace front.
    assert result['quality']['unique_zbuffer_pixels'] == 300


def test_neighbor_ambiguity_or_invalid_evidence_rejects():
    target = initial()[2]
    rejected(evaluate(other_targets=[dict(id='another', center_world=target['center_world'])]), 'ambiguous_neighbour')
    rejected(evaluate(other_targets=[dict(id='another', center_world=[np.nan, 0, 1])]))
    rejected(evaluate(other_targets=[None]))


def test_supplied_geometry_config_is_used_without_threshold_tuning():
    config = replace(LearnedPerceptionConfig(), max_mask_fraction=.01)
    rejected(evaluate(geometry_config=config), 'current_geometry_mask_pixel_extent')
    rejected(evaluate(geometry_config=replace(config, max_mask_fraction=float('nan'))))
    assert CONTRACT['depth_tolerance_m'] == .012
    assert CONTRACT['minimum_native_support_fraction'] == .8
    assert CONTRACT['maximum_mean_rgb_error'] == 35.


@pytest.mark.parametrize('part', ['rgb_dtype', 'depth_shape', 'depth_object', 'intrinsic_nan', 'extrinsic_invalid', 'oversized'])
def test_bounded_native_frame_validation(part):
    current = plane(1, 10.1)[0]
    if part == 'rgb_dtype': current['rgb'] = current['rgb'].astype(float)
    elif part == 'depth_shape': current['depth'] = current['depth'][:10]
    elif part == 'depth_object': current['depth'] = current['depth'].astype(object)
    elif part == 'intrinsic_nan': current['intrinsics'][0, 1] = np.nan
    elif part == 'extrinsic_invalid': current['T_world_camera'][0, 0] = 2.
    else:
        current['depth'] = np.broadcast_to(np.array(.5), (4096, 2161))
        current['rgb'] = np.broadcast_to(np.zeros(3, np.uint8), (4096, 2161, 3))
    rejected(evaluate(current=current))


def test_input_arrays_and_targets_are_not_mutated_or_returned_as_aliases():
    previous, mask, target = initial(); current = plane(1, 10.1)[0]
    before = copy.deepcopy((previous, mask, target, current))
    result = evaluate(previous, mask, target, current)
    assert result['reason'] is None
    for frame, saved in ((previous, before[0]), (current, before[3])):
        for key in ('rgb', 'depth', 'intrinsics', 'T_world_camera'):
            assert np.array_equal(frame[key], saved[key])
    assert target == before[2] and np.array_equal(mask, before[1])
    result['mask'][:] = False
    assert np.array_equal(mask, before[1])


def test_indistinguishable_replacement_is_explicitly_not_identity_proof():
    result = evaluate()  # Identical synthetic measurements cannot encode object identity.
    assert result['reason'] is None
    assert result['target']['identity_proven'] is False
    assert result['quality']['semantic_identity_proven'] is False
