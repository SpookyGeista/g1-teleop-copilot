import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.fixture_assets import write_cad_assets, pull_handle_grasp_asset
from teleop_assist.sensor_geometry import (
    camera_from_link_pose, measure_fixed_joint, wrist_camera_extrinsic,
    LEFT_WORKSPACE_CAMERA_QUATERNION_XYZW,
)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_wrist_mount_is_rigid_optical_and_follows_measured_link(side):
    mount = wrist_camera_extrinsic(side)
    rotation = Rotation.from_euler('xyz', [21, -33, 49], degrees=True)
    position = np.array([.22, -.14, 1.12])
    world = camera_from_link_pose(position, rotation.as_quat(), mount)
    np.testing.assert_allclose(world[:3, 3], position+rotation.apply(mount[:3, 3]))
    if side == 'right':
        nominal_grasp = np.array([.14, .05, 0.])
        nominal_forward = nominal_grasp-mount[:3, 3]
        nominal_forward /= np.linalg.norm(nominal_forward)
        optical_grasp = (nominal_grasp-mount[:3, 3]) @ mount[:3, :3]
        np.testing.assert_allclose(optical_grasp[:2], [0., 0.], atol=1e-10)
        assert optical_grasp[2] > .15
    else:
        nominal_forward = np.array([.07183456871782815, -.6221454476840839, .7795991512714044])
        np.testing.assert_allclose(mount[:3, :3], Rotation.from_quat(LEFT_WORKSPACE_CAMERA_QUATERNION_XYZW).as_matrix())
    np.testing.assert_allclose(world[:3, 2], rotation.apply(nominal_forward))
    np.testing.assert_allclose(world[:3, :3].T @ world[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(world[:3, :3]) == pytest.approx(1.)


def test_left_workspace_mount_is_constant_and_copies_cannot_mutate_calibration():
    initial = wrist_camera_extrinsic('left')
    changed = wrist_camera_extrinsic('left')
    changed[:] = 0
    np.testing.assert_array_equal(wrist_camera_extrinsic('left'), initial)
    # Rotation follows the measured wrist even if its rest posture changes;
    # there is no per-frame world-space workspace look-at correction.
    first = camera_from_link_pose([0, 0, 0], [0, 0, 0, 1], initial)
    turn = Rotation.from_euler('z', 45, degrees=True)
    second = camera_from_link_pose([.1, 0, 0], turn.as_quat(), initial)
    np.testing.assert_allclose(second[:3, 2], turn.apply(first[:3, 2]))
    assert np.linalg.norm(second[:3, 2]-first[:3, 2]) > .1


def test_resting_left_arm_observes_declared_workspace_with_fixed_mount():
    # Recorded resting-link pose, not a simulator object pose. The centre is a
    # declared workcell location, used only to verify this static calibration.
    rest = np.array([[.6206903362765473, -.0005632291805311888, .784055603401823, .085680291056633],
                     [.14616960015068878, .982551581824465, -.11500798687930049, .20978644490242004],
                     [-.7703102975065919, .1859894400957532, .6099425987157924, .7653028964996338],
                     [0, 0, 0, 1]])
    world_camera = rest @ wrist_camera_extrinsic('left')
    workspace = np.linalg.inv(world_camera) @ [.5, -.2, 1., 1.]
    np.testing.assert_allclose(workspace[:2], [0, 0], atol=1e-8)
    assert .5 < workspace[2] < .7
    assert world_camera[2, 2] > .25  # points upwards into workcell, not ground


def test_wrist_transform_rejects_missing_pose_and_bad_mount_side():
    with pytest.raises(ValueError):
        camera_from_link_pose([1, 2, 3], [0, 0, 0, 0], np.eye(4))
    with pytest.raises(ValueError):
        camera_from_link_pose([1, 2, float('nan')], [0, 0, 0, 1], np.eye(4))
    with pytest.raises(ValueError):
        wrist_camera_extrinsic('head')


def test_joint_measurement_reads_translation_with_explicit_units():
    joint = {'type': 'prismatic', 'axis_world': [1, 0, 0],
             'anchor_world': [.574, -.1, .955], 'limits': [-.16, 0.]}
    state = measure_fixed_joint([.514, -.1, .955, 0, 0, 0, 1], [-.12, 0, 0, 0, 0, 0], joint)
    assert state['position'] == pytest.approx(-.06)
    assert state['velocity'] == pytest.approx(-.12)
    assert state['unit'] == 'm' and state['evaluation_only']


def test_joint_measurement_reads_signed_rotation_without_degrees_confusion():
    joint = {'type': 'revolute', 'axis_world': [1, 0, 0],
             'anchor_world': [0, 0, 0], 'limits': [-np.pi/2, np.pi/2]}
    pose = [0, 0, 0, *Rotation.from_euler('x', -37, degrees=True).as_quat()]
    state = measure_fixed_joint(pose, [0, 0, 0, -.3, 0, 0], joint)
    assert state['position'] == pytest.approx(np.deg2rad(-37))
    assert state['velocity'] == pytest.approx(-.3) and state['unit'] == 'rad'


def test_cad_catalog_is_meter_scale_deterministic_and_has_no_world_poses(tmp_path):
    obj = {'id': 'KnownHandle', 'kind': 'handle', 'actions': ['pull'],
           'components': [{'shape': 'box', 'size': [.02, .12, .02], 'center': [-.06, 0, 0]}],
           'contact_point_object_m': [-.07, 0, 0],
           'mechanical_joint': {'type': 'prismatic', 'axis_object': [1, 0, 0]}}
    catalog = write_cad_assets([obj], tmp_path)
    rows = (tmp_path/'KnownHandle.obj').read_text().splitlines()
    vertices = np.array([[float(x) for x in row.split()[1:]] for row in rows if row.startswith('v ')])
    np.testing.assert_allclose(vertices.max(0)-vertices.min(0), [.02, .12, .02])
    np.testing.assert_allclose(vertices.mean(0), [-.06, 0, 0])
    assert catalog == write_cad_assets([obj], tmp_path)
    assert 'T_world_object' not in json.dumps(catalog)
    assert catalog['objects'][0]['pose_estimation_required']


def test_cylinder_obj_has_closed_endcaps_and_object_x_axis(tmp_path):
    write_cad_assets([{'id': 'Button', 'components': [{'shape': 'cylinder', 'size': [.04, .058, .058]}]}], tmp_path)
    rows = (tmp_path/'Button.obj').read_text().splitlines()
    vertices = np.array([[float(x) for x in row.split()[1:]] for row in rows if row.startswith('v ')])
    faces = [[int(x)-1 for x in row.split()[1:]] for row in rows if row.startswith('f ')]
    np.testing.assert_allclose(vertices.max(0)-vertices.min(0), [.04, .058, .058], atol=1e-8)
    edges = {}
    for a, b, c in faces:
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            edges[key] = edges.get(key, 0)+1
    assert set(edges.values()) == {2}


def test_handle_only_mesh_keeps_full_drawer_coordinate_origin(tmp_path):
    asset = pull_handle_grasp_asset([{'tool_point_object_m': [-.056, 0, -.05]}])
    catalog = write_cad_assets([asset], tmp_path)
    item = catalog['objects'][0]
    np.testing.assert_array_equal(item['T_parent_object'], np.eye(4))
    rows = (tmp_path/'PullHandleGrasp.obj').read_text().splitlines()
    vertices = np.array([[float(x) for x in row.split()[1:]] for row in rows if row.startswith('v ')])
    np.testing.assert_allclose(vertices.min(0), [-.076, -.07, -.011])
    np.testing.assert_allclose(vertices.max(0), [-.0005, .07, .011])
    assert item['grasp_templates'][0]['tool_point_object_m'] == [-.056, 0, -.05]
