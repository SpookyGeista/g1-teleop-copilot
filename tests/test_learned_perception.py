"""Geometry/identity contracts only; real model inference is tools/probe_models.py."""
import dataclasses

import numpy as np
import pytest

from teleop_assist.learned_perception import LearnedCabinetPerception, LearnedPerceptionConfig, mask_to_target
from teleop_assist.model_registry import model_paths


def fixture():
    mask = np.zeros((100, 120), dtype=bool)
    mask[30:70, 40:80] = True
    depth = np.ones(mask.shape, dtype=np.float32)
    k = {'fx':200., 'fy':200., 'cx':60., 'cy':50.}
    return mask, depth, k, np.eye(4)


def convert(mask, depth, k, t):
    return mask_to_target(mask, depth, k, t, label='a door handle',
                          detection_score=.3, mask_score=.9)


def test_measured_surface_transform_and_separate_confidences():
    m, z, k, t = fixture()
    t[:3, 3] = [2., 3., 4.]
    target, reason = convert(m, z, k, t)
    assert reason is None
    assert target['label'] == 'handle'
    assert np.allclose(np.array(target['center_world'])-target['center_camera'], [2,3,4])
    assert np.allclose(target['normal_world'], [0,0,-1])
    assert target['detection_confidence'] == .3
    assert target['pose_confidence'] > .8
    assert target['position_sigma_m'] >= .004
    x, y = target['surface_pixel_px']
    assert m[y,x]
    assert target['center_camera'][2] == z[y,x]
    assert target['geometry_scope'] == 'visible_depth_surface'
    assert target['action_inferred'] is False


def test_hole_is_never_a_contact_sample():
    m, z, k, t = fixture()
    m[43:57, 53:67] = False
    z[43:57, 53:67] = np.nan
    target, reason = convert(m,z,k,t)
    assert reason is None
    x,y = target['surface_pixel_px']
    assert m[y,x] and np.isfinite(z[y,x])


def test_sparse_depth_fails_closed():
    m,z,k,t = fixture()
    z[30:60] = np.nan
    target, reason = convert(m,z,k,t)
    assert target is None and reason == 'insufficient_mask_depth'


def test_invalid_geometry_rejected():
    m,z,k,t = fixture()
    t[0,0] = 2
    with pytest.raises(ValueError):
        convert(m,z,k,t)
    with pytest.raises(ValueError):
        convert(m[:, :-1], z,k,np.eye(4))


def test_model_missing_does_not_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match='setup_models'):
        model_paths(tmp_path)


def tracker():
    # Deliberately exercise association without constructing neural models.
    p = LearnedCabinetPerception.__new__(LearnedCabinetPerception)
    p.config = LearnedPerceptionConfig()
    p.reset()
    return p


def det(x=0, label='handle'):
    return {'label':label, 'center_world':[x,0,1]}


def test_tracks_current_measurements_only_and_retain_identity():
    p=tracker(); m=np.ones((8,8),bool)
    first,_=p._associate([det()], [m], 1.)
    second,_=p._associate([det(.002)], [m], 1.1)
    assert first[0]['id']==second[0]['id']
    third, diag=p._associate([], [], 1.2)
    assert third==[] and diag['stale_track_ids']==[first[0]['id']]
    assert p.last_masks=={}


def test_ambiguous_neighbours_are_not_reassigned():
    p=tracker(); m=np.ones((8,8),bool)
    p._associate([det(-.01), det(.01)], [m,m], 1.)
    targets,diag=p._associate([det()], [m], 1.1)
    assert targets==[]
    assert diag['ambiguous_detection_count']==1


def test_expired_track_has_new_identity():
    p=tracker(); m=np.ones((8,8),bool)
    old,_=p._associate([det()], [m], 1.)
    new,diag=p._associate([det()], [m], 4.)
    assert old[0]['id']!=new[0]['id']
    assert diag['expired_track_ids']==[old[0]['id']]
