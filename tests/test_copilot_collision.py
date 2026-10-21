"""Actual Dex3 geometry controls for the deliberately scoped preview filter."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop_assist.copilot.collision import GraspCollisionFilter, CollisionSettings

ASSETS = ROOT / 'teleop_assist/copilot/assets/dex3_right'
FAR_OBJECT = np.array([[1., 1., 1.]])
FAR_SCENE = np.array([[-1., -1., -1.]])


@pytest.fixture(scope='module')
def checker():
    return GraspCollisionFilter(ASSETS)


def palm_center(checker):
    return checker.canonical_meshes['right_hand_palm_link'].convex_hull.center_mass


def pad_face(checker):
    patch = checker.pad_meshes['right_hand_index_1_link']
    center = patch.centroid
    index = np.argmin(np.linalg.norm(patch.triangles_center - center, axis=1))
    return patch.triangles_center[index], patch.face_normals[index]


def test_actual_dex3_far_points_clear_but_not_full_robot_certification(checker):
    report = checker.assess_grasp(np.eye(4), FAR_OBJECT, FAR_SCENE)
    assert report['hand_collision_free'] and report['pregrasp_collision_free']
    assert report['scene_coverage_unknown']
    assert report['collision_valid'] is False
    assert report['full_robot_collision_checked'] is False
    assert report['finger_closure_checked'] is False
    assert len(report['approach']) == 9
    assert np.isclose(report['approach_extra_inflation_m'], .005)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('source', ['target', 'scene'])
def test_palm_penetration_is_never_exempt_even_for_target(checker, source):
    point = palm_center(checker)[None]
    report = checker.assess_grasp(np.eye(4), point if source == 'target' else FAR_OBJECT,
                                 point if source == 'scene' else FAR_SCENE)
    assert not report['hand_collision_free']
    assert report['final']['links']['right_hand_palm_link']


def test_obstacle_at_pregrasp_is_rejected_even_when_final_pose_is_clear(checker):
    obstacle = palm_center(checker) - [0, 0, .080]
    report = checker.assess_grasp(np.eye(4), FAR_OBJECT, obstacle[None])
    assert report['hand_collision_free']
    assert not report['pregrasp_collision_free']
    assert report['approach'][0]['scene_collision_points'] == 1


def test_final_pad_contact_only_permitted_for_target_not_scene(checker):
    center, normal = pad_face(checker)
    contact = (center + .0005 * normal)[None]
    target = checker.assess_grasp(np.eye(4), contact, FAR_SCENE)
    scene = checker.assess_grasp(np.eye(4), FAR_OBJECT, contact)
    assert target['hand_collision_free']
    assert target['final']['allowed_target_pad_contacts'] == 1
    assert not scene['hand_collision_free']
    # Duplication into the environment cloud is never silently filtered away.
    duplicate = checker.assess_grasp(np.eye(4), contact, contact)
    assert not duplicate['hand_collision_free']


def test_deep_target_penetration_under_contact_pad_is_rejected(checker):
    center, normal = pad_face(checker)
    report = checker.assess_grasp(np.eye(4), (center - .004 * normal)[None], FAR_SCENE)
    assert not report['hand_collision_free']
    assert report['final']['forbidden_object_points'] == 1


def test_rigid_transform_preserves_collision_and_approach_results(checker):
    g = np.eye(4)
    g[:3, :3] = Rotation.from_euler('xyz', [.3, -.8, .5]).as_matrix()
    g[:3, 3] = [.4, -.2, .7]
    scene = np.stack([palm_center(checker) - [0, 0, .080], [-1, -1, -1]])
    before = checker.assess_grasp(np.eye(4), FAR_OBJECT, scene)
    after = checker.assess_grasp(g, FAR_OBJECT @ g[:3, :3].T + g[:3, 3],
                                scene @ g[:3, :3].T + g[:3, 3])
    assert before['final'] == after['final']
    assert before['approach'] == after['approach']


def test_missing_scene_does_not_claim_clearance(checker):
    report = checker.assess_grasp(np.eye(4), FAR_OBJECT, np.zeros((0, 3)))
    assert report['status'] == 'incomplete_observation'
    assert not report['hand_collision_free'] and not report['pregrasp_collision_free']


@pytest.mark.parametrize('kind', ['nonrigid', 'nan', 'empty_object'])
def test_invalid_inputs_fail_explicitly(checker, kind):
    g, obj = np.eye(4), FAR_OBJECT.copy()
    if kind == 'nonrigid':
        g[0, 0] = 2
    elif kind == 'nan':
        obj[0, 0] = np.nan
    else:
        obj = np.zeros((0, 3))
    with pytest.raises(ValueError):
        checker.assess_grasp(g, obj, FAR_SCENE)


def test_descriptor_tampering_fails_before_geometric_checks(tmp_path):
    descriptor = tmp_path / 'dex3_right'
    shutil.copytree(ASSETS, descriptor)
    with (descriptor / 'config.json').open('a') as file:
        file.write(' ')
    with pytest.raises(ValueError, match='hash mismatch'):
        GraspCollisionFilter(descriptor)


def test_actual_mesh_vertices_are_enclosed_by_each_zero_margin_hull(checker):
    for name, mesh in checker.canonical_meshes.items():
        selected = mesh.vertices[::113]
        assert len(checker._hits(name, selected, 0)) == len(selected)


def test_half_step_inflation_covers_between_sample_translation(checker):
    # An actual CAD vertex translated half a sampling step must be inside
    # that sample's inflated hull, even if it is not inside the static mesh.
    for name, mesh in checker.canonical_meshes.items():
        selected = mesh.vertices[::251] + [0, 0, .005]
        assert len(checker._hits(name, selected, .005)) == len(selected)


@pytest.mark.parametrize('kwargs', [{'approach_samples': 3}, {'clearance_m': -1},
                                   {'approach_m': float('nan')}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        CollisionSettings(**kwargs)
