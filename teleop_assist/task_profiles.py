"""Declared fixture capabilities plus optional measured CAD registration.

The legacy profile preserves surface-only approach actions for existing tests.
Its surface point is explicitly not a calibrated finger/tool grasp pose. A
fresh FoundationPose result for the known handle transforms the CAD-local G1
candidate into world coordinates. No simulator world poses are read here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_CAD_CATALOG = Path(__file__).resolve().parents[1] / 'assets/cad/handle_pose/catalog.json'
CAD_ASSET_ID = 'PullHandleGrasp'
GRASP_TEMPLATE_ID = 'right_g1_bounded_pull_v1'


def _vector(value, length, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {length} finite values')
    return result


def _rigid(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'{name} must be a finite 4x4 transform')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError(f'{name} must be a proper rigid transform')
    return matrix


def _nonnegative(value, name):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return number


def _load_handle_template(catalog_path):
    path = Path(catalog_path)
    catalog = json.loads(path.read_text())
    if catalog.get('schema') != 'punktnull.known_fixture_assets.v1' or catalog.get('units') != 'meters':
        raise ValueError('Known CAD catalog schema and meter units required')
    matches = [asset for asset in catalog.get('objects', []) if asset.get('id') == CAD_ASSET_ID]
    if len(matches) != 1:
        raise ValueError('Catalog requires exactly one matching handle CAD asset')
    asset = matches[0]
    if asset.get('mesh_units') != 'meters' or asset.get('parent_asset_id') != 'PullHandle':
        raise ValueError('Wrong CAD units or parent frame')
    if not np.allclose(_rigid(asset['T_parent_object'], 'T_parent_object'), np.eye(4), atol=1e-8):
        raise ValueError('This template requires the exported drawer-root object frame')
    mesh = Path(asset['cad_mesh'])
    if not mesh.is_absolute():
        mesh = path.parent / mesh
    if hashlib.sha256(mesh.read_bytes()).hexdigest() != asset.get('cad_sha256'):
        raise ValueError('CAD mesh hash does not match the grasp catalog')
    templates = [row for row in asset.get('grasp_templates', []) if row.get('id') == GRASP_TEMPLATE_ID]
    if len(templates) != 1:
        raise ValueError('Physically evaluated bounded G1 grasp template is unavailable')
    template = templates[0]
    point = _vector(template['tool_point_object_m'], 3, 'CAD tool point')
    quaternion = _vector(template['wrist_quaternion_object_xyzw'], 4, 'CAD wrist quaternion')
    expected_q = np.array([2.**-.5, 0., 0., 2.**-.5])
    if (not np.allclose(point, [-.056, 0., -.05], atol=1e-6)
            or min(np.linalg.norm(quaternion - expected_q), np.linalg.norm(quaternion + expected_q)) >= 1e-6):
        raise ValueError('Template differs from the declared physically evaluated tool pose')
    joint = asset['mechanical_joint']
    axis = _vector(joint['axis_object'], 3, 'CAD joint axis')
    approach = _vector(asset['approach_direction_object'], 3, 'CAD approach direction')
    if (joint.get('type') != 'prismatic' or not np.allclose(axis, [1, 0, 0])
            or not np.allclose(approach, [1, 0, 0])
            or not np.allclose(_vector(joint['limits_m'], 2, 'CAD joint limits'), [-.16, 0.])):
        raise ValueError('Handle joint prior differs from the declared prismatic fixture')
    if (not math.isclose(float(template['validated_pull_distance_m']), .04, abs_tol=1e-8)
            or not math.isclose(float(template['validated_pull_duration_s']), 6., abs_tol=1e-8)
            or not math.isclose(float(template['grip_scalar']), .85, abs_tol=1e-8)):
        raise ValueError('Unexpected bounded pull validation scope')
    _nonnegative(template['approach_standoff_m'], 'CAD approach standoff')
    _vector(asset['contact_point_object_m'], 3, 'CAD contact point')
    return asset, template


def _cad_grasp(target, now, max_pose_age, catalog_path):
    """Transform a known template only with fresh, valid estimated object pose.

    Timestamp uses the host-monotonic acquisition clock. Uncertainty is supplied
    by pose integration; the detector confidence is not a pose covariance.
    """
    present = any(key in target for key in ('T_world_object', 'pose_valid', 'pose_source', 'cad_asset_id'))
    if not present:
        return None, {'status': 'not_available', 'reason': 'measured_cad_pose_required'}
    status = {'status': 'rejected', 'cad_asset_id': target.get('cad_asset_id'), 'source': target.get('pose_source')}
    try:
        if target.get('pose_valid') is not True:
            raise ValueError('pose_valid must be a true JSON boolean')
        if target.get('pose_source') not in ('foundationpose', 'foundationpose_rgbd_tracking'):
            raise ValueError('Unapproved pose source')
        if target.get('cad_asset_id') != CAD_ASSET_ID:
            raise ValueError('No calibrated grasp template for this CAD asset')
        timestamp = float(target['pose_timestamp'])
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Pose acquisition timestamp must be finite and nonnegative')
        age = now - timestamp
        status['age_s'] = age
        if age < -.01 or age > max_pose_age:
            raise ValueError('Pose is stale or timestamp is in the future')
        if target['pose_source']=='foundationpose_rgbd_tracking':
            registration_time=float(target['pose_registration_timestamp'])
            registration_age=now-registration_time
            if not math.isfinite(registration_age) or not 0<=registration_age<=.7:
                raise ValueError('Tracked CAD registration is stale')
            status['registration_age_s']=registration_age
            if target.get('pose_tracking_method')!='measured_point_translation_for_declared_prismatic_fixture':
                raise ValueError('Unknown CAD pose tracking method')
        transform = _rigid(target['T_world_object'], 'T_world_object')
        if (target.get('pose_position_sigma_m', target.get('position_sigma_m')) is None
                or target.get('pose_orientation_sigma_rad', target.get('orientation_sigma_rad')) is None):
            raise ValueError('Pose position and orientation uncertainty must be supplied')
        position_sigma = _nonnegative(target.get('pose_position_sigma_m', target.get('position_sigma_m')), 'pose position uncertainty')
        orientation_sigma = _nonnegative(target.get('pose_orientation_sigma_rad', target.get('orientation_sigma_rad')), 'pose orientation uncertainty')
        if position_sigma > .025 or orientation_sigma > .35:
            raise ValueError('Pose uncertainty exceeds calibrated grasp limits')
        asset, template = _load_handle_template(catalog_path)
        local_point = np.asarray(template['tool_point_object_m'], dtype=float)
        effective_sigma = position_sigma + np.linalg.norm(local_point) * orientation_sigma
        if effective_sigma > .025:
            raise ValueError('Tool-point uncertainty including orientation exceeds 25 mm')
        rotation, origin = transform[:3, :3], transform[:3, 3]
        contact_world = rotation @ np.asarray(asset['contact_point_object_m']) + origin
        if np.linalg.norm(contact_world - np.asarray(target['center_world'])) > .12:
            raise ValueError('Registered CAD pose disagrees with the observed handle surface')
        point = rotation @ local_point + origin
        quaternion = (Rotation.from_matrix(rotation) * Rotation.from_quat(template['wrist_quaternion_object_xyzw'])).as_quat()
        outward = -(rotation @ np.asarray(asset['approach_direction_object']))
        pull_axis = -(rotation @ np.asarray(asset['mechanical_joint']['axis_object']))
        result = {'grasp_position_world': point.tolist(), 'grasp_quaternion_xyzw': quaternion.tolist(),
                  'approach_axis_world': outward.tolist(), 'motion_axis_world': pull_axis.tolist(),
                  'pivot_world': origin.tolist(), 'position_sigma_m': float(effective_sigma),
                  'orientation_sigma_rad': orientation_sigma, 'template': template,
                  'calibration_id': f'cad:{CAD_ASSET_ID}:{GRASP_TEMPLATE_ID}', 'pose_timestamp': timestamp}
        status.update(status='accepted', reason='fresh_measured_pose_and_known_local_grasp_template',
                      template_id=GRASP_TEMPLATE_ID, tool_position_sigma_m=float(effective_sigma),
                      engagement_verified=False, validation_scope=template['validation'])
        return result, status
    except (KeyError, TypeError, ValueError, OSError) as error:
        status['reason'] = str(error)
        return None, status


def annotate_tasks(targets, profile='interaction', standoff=.08, *, now=None,
                   catalog_path=None, max_pose_age=.5):
    if profile not in ('interaction', 'approach_only'):
        raise ValueError('Unknown task profile')
    now = time.monotonic() if now is None else float(now)
    if not math.isfinite(now) or now < 0 or not math.isfinite(max_pose_age) or not 0 < max_pose_age <= 2.:
        raise ValueError('Finite monotonic time and bounded positive pose age required')
    standoff = _nonnegative(standoff, 'approach standoff')
    if standoff > .3:
        raise ValueError('Approach standoff exceeds 30 cm')
    annotated = []
    # Surface-only fallback: the known fixed cabinet faces -worldX. This is not
    # a measured six-dimensional hand/tool calibration.
    mounting_outward = np.array([-1., 0., 0.])
    for original in targets:
        t = copy.deepcopy(original)
        center = _vector(t['center_world'], 3, 'surface center')
        normal = _vector(t['normal_world'], 3, 'surface normal')
        if np.linalg.norm(normal) < 1e-6:
            raise ValueError('Surface normal must be nonzero')
        normal /= np.linalg.norm(normal)
        label = str(t.get('semantic_label', t.get('label', ''))).lower()
        is_handle = any(word in label for word in ('handle', 'griff', 'drawer'))
        pose, pose_status = _cad_grasp(t, now, max_pose_age, catalog_path or DEFAULT_CAD_CATALOG) if is_handle else (
            None, {'status': 'not_applicable', 'reason': 'no_grasp_template_for_semantic_class'})
        t['position_sigma_m'] = max(_nonnegative(t.get('position_sigma_m', 0), 'surface position uncertainty'),
            .012, _nonnegative(t.get('surface_residual_m', 0), 'surface residual') * 3)
        t['cad_grasp_status'] = pose_status
        base = {'grasp_position_world': center.tolist(), 'approach_axis_world': normal.tolist(),
                'max_position_sigma_m': .025, 'grasp_pose_scope': 'visible_surface_approach_only',
                'has_calibrated_grasp_template': False, 'execution_ready': False}
        outward = mounting_outward
        if pose is not None:
            base.update({key: pose[key] for key in ('grasp_position_world', 'grasp_quaternion_xyzw', 'approach_axis_world')})
            base.update(grasp_pose_scope='cad_template_transformed_by_estimated_pose',
                        has_calibrated_grasp_template=True, execution_ready=profile == 'interaction',
                        recommended_grip_scalar=pose['template']['grip_scalar'],
                        validated_pull_distance_m=pose['template']['validated_pull_distance_m'],
                        validated_pull_duration_s=pose['template']['validated_pull_duration_s'],
                        pose_timestamp=pose['pose_timestamp'])
            outward = np.asarray(pose['motion_axis_world'])
            t['position_sigma_m'] = max(t['position_sigma_m'], pose['position_sigma_m'])
            t['orientation_sigma_rad'] = pose['orientation_sigma_rad']
        actions = [dict(base, id='grasp', label='Greifen / ausrichten', kind='grasp')]
        if profile == 'interaction':
            if is_handle:
                actions += [dict(base, id='pull', label='Schublade ziehen', kind='linear', motion_axis_world=outward.tolist()),
                            dict(base, id='push', label='Schublade schieben', kind='linear', motion_axis_world=(-outward).tolist())]
                if pose is not None:
                    for entry in actions:
                        if entry['id'] in ('pull','push'):
                            entry['execution_travel_limit_m']=.04
            elif any(word in label for word in ('knob', 'selector', 'drehknopf')):
                actions += [dict(base, id='rotate_left', label='Links drehen', kind='axial_rotation', motion_axis_world=outward.tolist(), pivot_world=center.tolist()),
                            dict(base, id='rotate_right', label='Rechts drehen', kind='axial_rotation', motion_axis_world=(-outward).tolist(), pivot_world=center.tolist())]
            elif any(word in label for word in ('button', 'taster', 'pushbutton')):
                actions += [dict(base, id='press', label='Drücken', kind='linear', motion_axis_world=(-outward).tolist())]
        calibration = {'id': f'{profile}:{t["id"]}', 'source': 'configured_fixture', 'profile': profile,
            'anchor_source': 'measured_rgbd_surface', 'mechanism_source': 'declared_fixture_class_mapping_not_inferred',
            'joint_axis_source': 'fixed_interaction_fixture_mount_world_X', 'approach_standoff_m': standoff,
            'has_calibrated_grasp_template': False, 'execution_ready': False, 'actions': actions}
        if pose is not None:
            calibration.update(id=f'{profile}:{t["id"]}:{pose["calibration_id"]}', source='cad_registration',
                anchor_source='estimated_T_world_object_times_CAD_local_tool_point',
                joint_axis_source='estimated_object_rotation_times_declared_CAD_joint_axis',
                cad_asset_id=CAD_ASSET_ID, grasp_template_id=GRASP_TEMPLATE_ID,
                approach_standoff_m=pose['template']['approach_standoff_m'],
                has_calibrated_grasp_template=True, execution_ready=profile == 'interaction',
                pose_timestamp=pose['pose_timestamp'], pose_max_age_s=max_pose_age,
                mechanism_origin_world=pose['pivot_world'],
                validation_scope=pose['template']['validation'], engagement_verified=False)
        t['task_calibration'] = calibration
        annotated.append(t)
    return annotated
