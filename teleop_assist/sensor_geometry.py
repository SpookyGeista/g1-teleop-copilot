"""CPU-testable optical camera calibration and joint measurement helpers.

Transforms map optical +X right, +Y down, +Z forward into link/world frames.
The camera mount is a rigid calibration, never aimed at simulator targets.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Optical frame -> left_wrist_yaw_link. Declared virtual bracket calibration
# for the fixed fixture's resting LEFT arm: a third view of the workspace.
# Chosen once using the nominal workspace centre and a measured rest-link pose.
# It never changes with object selection, estimated object pose or scene truth.
LEFT_WORKSPACE_CAMERA_QUATERNION_XYZW = (
    -.1264031726746106, -.3069571017293627,
    .8252774303313211, -.45685527097915146,
)


def wrist_camera_extrinsic(side):
    if side not in ('left', 'right'):
        raise ValueError('Wrist side must be left or right')
    transform = np.eye(4)
    # Both optical origins use the same virtual bracket above the wrist.
    transform[:3, 3] = [-.025, 0., .065]
    if side == 'left':
        transform[:3, :3] = Rotation.from_quat(LEFT_WORKSPACE_CAMERA_QUATERNION_XYZW).as_matrix()
        return transform
    # Right hand's existing close-range grasp view remains unchanged.
    nominal_grasp = np.array([.14, .05, 0.])
    forward = nominal_grasp-transform[:3, 3]
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform[:3, :3] = np.column_stack((right, down, forward))
    return transform


def camera_from_link_pose(position, quaternion_xyzw, T_link_camera):
    position = np.asarray(position, dtype=float)
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    extrinsic = np.asarray(T_link_camera, dtype=float)
    if position.shape != (3,) or quaternion.shape != (4,) or extrinsic.shape != (4, 4):
        raise ValueError('Invalid calibrated pose shapes')
    if not all(np.isfinite(a).all() for a in (position, quaternion, extrinsic)):
        raise ValueError('Camera pose must be finite')
    if np.linalg.norm(quaternion) < 1e-9:
        raise ValueError('Camera pose needs a valid quaternion')
    T_world_link = np.eye(4)
    T_world_link[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    T_world_link[:3, 3] = position
    return T_world_link @ extrinsic


def measure_fixed_joint(transform_xyzw, velocity, joint):
    """Read a world-anchored 1-DoF joint from measured PhysX body state.

    ``joint`` contains the initial anchor and known mechanical axis. This is
    evaluation telemetry, not an object-pose estimate for the vision stack.
    """
    transform = np.asarray(transform_xyzw, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    axis = np.asarray(joint['axis_world'], dtype=float)
    axis /= np.linalg.norm(axis)
    if joint['type'] == 'prismatic':
        coordinate = float((transform[:3] - np.asarray(joint['anchor_world'])) @ axis)
        rate = float(velocity[:3] @ axis)
        unit = 'm'
    elif joint['type'] == 'revolute':
        coordinate = float(Rotation.from_quat(transform[3:7]).as_rotvec() @ axis)
        rate = float(velocity[3:6] @ axis)
        unit = 'rad'
    else:
        raise ValueError('Unsupported fixture joint type')
    return {'position': coordinate, 'velocity': rate, 'unit': unit,
            'limits': list(joint['limits']), 'source': 'measured_PhysX_rigid_body_state',
            'evaluation_only': True}
