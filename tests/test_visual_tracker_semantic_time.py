"""Original semantic acquisition survives recovery and CPU LK propagation."""
import copy

import numpy as np
import pytest

from teleop_assist.learned_perception import mask_to_target
from teleop_assist.visual_tracker import RGBDMaskTracker


def fixture(timestamp=2.1, frame_id=21):
    rgb = np.zeros((100, 160, 3), np.uint8)
    rng = np.random.default_rng(81)
    masks = {}
    for identity, x in (('recovery', 20), ('dino', 95)):
        rgb[30:70, x:x+35] = rng.integers(20, 220, (40, 35, 3), dtype=np.uint8)
        mask = np.zeros(rgb.shape[:2], bool)
        mask[30:70, x:x+35] = True
        masks[identity] = mask
    frame = dict(rgb=rgb, depth=np.ones(rgb.shape[:2], np.float32),
                 intrinsics=dict(fx=200., fy=200., cx=80., cy=50.),
                 T_world_camera=np.eye(4), timestamp=timestamp, frame_id=frame_id)
    targets = {}
    for identity, mask in masks.items():
        target, reason = mask_to_target(mask, frame['depth'], frame['intrinsics'],
            frame['T_world_camera'], label='handle', detection_score=.6, mask_score=.9)
        assert reason is None
        targets[identity] = dict(target, id=identity, frame_id=frame_id, timestamp=timestamp)
    return frame, targets, masks


def cad(target, frame, **changes):
    return dict(kind='foundationpose_rgbd', accepted=True, target_id=target['id'],
        timestamp=frame['timestamp'], frame_id=frame['frame_id'], camera_id='head',
        center_world=target['center_world'], quality=dict(silhouette_iou=.9,
        measured_depth_support=.95, median_depth_error_m=.002, p90_depth_error_m=.004),
        pose_position_sigma_m=.012, pose_orientation_sigma_rad=.15, **changes)


@pytest.mark.parametrize('same_frame', [True, False])
def test_expired_recovery_reinitialization_never_renews_semantics(same_frame):
    frame, targets, masks = fixture()
    target = dict(targets['recovery'], semantic_keyframe_timestamp=1.)
    tracker = RGBDMaskTracker()
    tracker.initialize(frame, [target], masks)
    current = frame if same_frame else dict(frame, timestamp=2.2, frame_id=22)
    result = tracker.track(current)
    assert result['targets'] == []
    assert tracker.last_masks == {}
    assert result['diagnostics']['lost_tracks']['recovery'] == 'expired_semantic_and_cad_identity_evidence'
    assert not tracker.confirm_identity(cad(target, current))


def test_current_recovery_keeps_original_time_and_expires_after_original_ttl():
    frame, targets, masks = fixture(timestamp=1.8, frame_id=18)
    target = dict(targets['recovery'], semantic_keyframe_timestamp=1.)
    tracker = RGBDMaskTracker()
    tracker.initialize(frame, [target], masks)
    for timestamp, frame_id in ((1.8, 18), (1.9, 19), (2., 20)):
        observed = tracker.track(dict(frame, timestamp=timestamp, frame_id=frame_id))['targets']
        assert len(observed) == 1
        assert observed[0]['semantic_keyframe_timestamp'] == 1.
        assert observed[0]['semantic_keyframe_age_s'] == pytest.approx(timestamp-1.)
        assert observed[0]['timestamp'] == timestamp
        assert observed[0]['frame_id'] == frame_id
    assert tracker.track(dict(frame, timestamp=2.01, frame_id=21))['targets'] == []


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -0.1, 2.100001, None, True, '1.0'])
def test_invalid_explicit_semantic_time_never_falls_back_to_frame_time(invalid):
    frame, targets, masks = fixture()
    bad = dict(targets['recovery'], semantic_keyframe_timestamp=invalid)
    tracker = RGBDMaskTracker()
    tracker.initialize(frame, [bad, targets['dino']], masks)
    observed = tracker.track(frame)['targets']
    assert [target['id'] for target in observed] == ['dino']
    assert observed[0]['semantic_keyframe_timestamp'] == frame['timestamp']


def test_mixed_targets_have_independent_semantic_ages_and_original_expiry():
    frame, targets, masks = fixture(timestamp=1.8, frame_id=18)
    targets['recovery']['semantic_keyframe_timestamp'] = 1.
    tracker = RGBDMaskTracker()
    tracker.initialize(frame, list(targets.values()), masks)
    first = {t['id']: t for t in tracker.track(frame)['targets']}
    assert first['recovery']['semantic_keyframe_age_s'] == pytest.approx(.8)
    assert first['dino']['semantic_keyframe_age_s'] == 0.
    result = tracker.track(dict(frame, timestamp=2.1, frame_id=21))
    assert [t['id'] for t in result['targets']] == ['dino']
    assert result['targets'][0]['semantic_keyframe_timestamp'] == 1.8
    assert result['targets'][0]['semantic_keyframe_age_s'] == pytest.approx(.3)


def test_initialization_copies_semantic_time_and_input_target_fields():
    frame, targets, masks = fixture(timestamp=1.8, frame_id=18)
    target = dict(targets['recovery'], semantic_keyframe_timestamp=1.)
    before = copy.deepcopy(target)
    tracker = RGBDMaskTracker()
    tracker.initialize(frame, [target], masks)
    assert target == before
    target['semantic_keyframe_timestamp'] = 1.8
    assert tracker.track(frame)['targets'][0]['semantic_keyframe_timestamp'] == 1.


def test_normal_dino_source_time_and_continuing_lk_never_renew_it():
    frame, targets, masks = fixture(timestamp=1., frame_id=10)
    tracker = RGBDMaskTracker()
    tracker.initialize(frame, [targets['dino']], masks)
    for stamp, index in ((1., 10), (1.4, 14), (1.8, 18)):
        result = tracker.track(dict(frame, timestamp=stamp, frame_id=index))
        assert result['targets'][0]['semantic_keyframe_timestamp'] == 1.
        assert result['targets'][0]['semantic_keyframe_age_s'] == pytest.approx(stamp-1.)
    assert tracker.track(dict(frame, timestamp=2.01, frame_id=21))['targets'] == []


def test_fresh_accepted_cad_can_support_recovery_without_renewing_semantics():
    frame, targets, masks = fixture()
    target = dict(targets['recovery'], semantic_keyframe_timestamp=1.)
    tracker = RGBDMaskTracker()
    tracker.initialize(frame, [target], masks)
    assert tracker.confirm_identity(cad(target, frame))
    for stamp, index in ((2.1, 21), (2.3, 23)):
        observed = tracker.track(dict(frame, timestamp=stamp, frame_id=index))['targets']
        assert len(observed) == 1
        assert observed[0]['semantic_keyframe_timestamp'] == 1.
        assert observed[0]['semantic_keyframe_age_s'] == pytest.approx(stamp-1.)
        assert observed[0]['identity_validation_timestamp'] == 2.1
    assert tracker.track(dict(frame, timestamp=2.61, frame_id=27))['targets'] == []


def test_unaccepted_cad_does_not_rescue_stale_same_frame_recovery():
    frame, targets, masks = fixture()
    target = dict(targets['recovery'], semantic_keyframe_timestamp=1.)
    tracker = RGBDMaskTracker()
    tracker.initialize(frame, [target], masks)
    bad = cad(target, frame)
    bad['accepted'] = False
    assert not tracker.confirm_identity(bad)
    assert tracker.track(frame)['targets'] == []
