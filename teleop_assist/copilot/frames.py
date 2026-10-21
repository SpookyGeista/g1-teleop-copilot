"""Calibrated camera/robot recording contracts, with unchanged source clocks."""
from __future__ import annotations
import numpy as np
import os
from pathlib import Path
import tempfile
from scipy.spatial.transform import Rotation
from .kinematics import ARM_JOINT_NAMES
from .observations import GeometryObservation, rigid


def mounted_camera_pose(frame, parent_link, offset_position, offset_quaternion_wxyz):
    """ROS optical extrinsics from measured parent body and original fixed mount.

    Isaac's camera XForm view can omit the articulated parent's current pose
    on the CPU backend. Never substitute that stale view for measured links.
    """
    names=[str(name) for name in frame['body_names']]
    if len(names)!=len(set(names)) or parent_link not in names:
        raise ValueError('Camera parent must have one measured body pose')
    body=np.asarray(frame['body_poses'][names.index(parent_link)],dtype=float)
    position=np.asarray(offset_position,dtype=float)
    quaternion=np.asarray(offset_quaternion_wxyz,dtype=float)
    if (body.shape!=(7,) or position.shape!=(3,) or quaternion.shape!=(4,) or
            not np.isfinite(body).all() or not np.isfinite(position).all() or not np.isfinite(quaternion).all()):
        raise ValueError('Invalid measured camera mount')
    parent=np.eye(4);parent[:3,3]=body[:3]
    parent[:3,:3]=Rotation.from_quat(body[[4,5,6,3]]).as_matrix()
    mount=np.eye(4);mount[:3,3]=position
    mount[:3,:3]=Rotation.from_quat(quaternion[[1,2,3,0]]).as_matrix()
    return rigid(parent@mount)


def publish_frame(directory, frame):
    """Atomically publish the captured arrays without changing source times.

    Uncompressed NPZ avoids adding video compression work to the control loop;
    the reader sees either complete version, never a partially written file.
    """
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    handle,name=tempfile.mkstemp(prefix='.frame-',suffix='.npz',dir=directory)
    try:
        with os.fdopen(handle,'wb') as stream:
            np.savez(stream,**frame)
        Path(name).replace(directory/'frame.npz')
    finally:Path(name).unlink(missing_ok=True)


def load_frame(path):
    with np.load(path, allow_pickle=False) as data:
        frame = {name: data[name].copy() for name in data.files}
    for key in ('clock_id', 'calibration_id'):
        frame[key] = str(frame[key].item())
        if not frame[key]:
            raise ValueError(f'Missing {key}')
    frame['frame_id'] = int(frame['frame_id'])
    frame['timestamp'] = float(frame['timestamp'])
    if not np.isfinite(frame['timestamp']) or frame['timestamp'] < 0:
        raise ValueError('Invalid source timestamp')
    rgb, depth = frame['rgb'], frame['depth']
    if rgb.dtype != np.uint8 or rgb.shape != depth.shape + (3,):
        raise ValueError('Expected aligned RGB and metric depth')
    frame['T_world_base'] = rigid(frame['T_world_base'])
    frame['T_world_camera'] = rigid(frame['T_world_camera'])
    return frame


def geometry_observation(frame, target_id, mask):
    from .depth_cleanup import clean_object_mask
    cleaned, cleanup = clean_object_mask(frame['depth'], np.asarray(mask, dtype=bool),
                                        frame['intrinsics'], frame['T_world_camera'])
    if not cleanup['accepted']:
        raise ValueError(f"Ambiguous target depth: {cleanup.get('reason', cleanup.get('status'))}")
    return GeometryObservation(target_id=target_id,
        observation_id=f"{frame['calibration_id']}:{frame['frame_id']}",
        capture_timestamp=frame['timestamp'], clock_id=frame['clock_id'],
        calibration_id=frame['calibration_id'], source='observed_rgbd_and_pretrained_mask',
        depth=frame['depth'], mask=cleaned, intrinsics=frame['intrinsics'],
        T_world_camera=frame['T_world_camera'], T_world_base=frame['T_world_base'])


def robot_request(frame):
    names = [str(n) for n in frame['joint_names']]
    q = np.asarray(frame['joint_positions'], float)
    if len(names) != len(set(names)) or q.shape != (len(names),) or not np.isfinite(q).all():
        raise ValueError('Invalid named joint state')
    values = dict(zip(names, q.tolist()))
    arm = [values[name] for name in ARM_JOINT_NAMES]
    bodies = [str(n) for n in frame['body_names']]
    left_wrist = np.asarray(frame['body_poses'][bodies.index('left_wrist_yaw_link')], float)
    world_wrist = np.eye(4)
    world_wrist[:3, :3] = Rotation.from_quat(left_wrist[[4, 5, 6, 3]]).as_matrix()
    world_wrist[:3, 3] = left_wrist[:3]
    body_matrices = {}
    for name, pose in zip(bodies, frame['body_poses']):
        transform = np.eye(4)
        transform[:3, 3] = pose[:3]
        transform[:3, :3] = Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()
        body_matrices[name] = transform.tolist()
    wrist_ee = np.eye(4); wrist_ee[0, 3] = .05
    left = np.linalg.inv(frame['T_world_base']) @ world_wrist @ wrist_ee
    return {'current_arm_q': arm, 'left_base_ee': left.tolist(),
            'joint_state_acquisition_timestamp': frame['timestamp'],
            'joint_state_clock_id': frame['clock_id'],
            'joint_state_source': 'same_physics_step_recording',
            'robot_state': {'joint_positions': values,
                'base_in_world': frame['T_world_base'].tolist(),
                'observed_body_world_poses': body_matrices,
                'body_pose_quaternion_convention': 'wxyz',
                'acquisition_timestamp': frame['timestamp'], 'clock_id': frame['clock_id'],
                'source': 'original_Unitree_USD_simulator_measured_state', 'complete': len(values) == 43}}


def preview_uv(world_canonical, frame):
    """Projected axes for a selected grasp; no invented camera overlay poses."""
    G = rigid(world_canonical)
    local = np.array([[0, 0, 0, 1], [.04, 0, 0, 1], [0, .04, 0, 1], [0, 0, .04, 1]])
    camera = (np.linalg.inv(frame['T_world_camera']) @ G @ local.T).T[:, :3]
    if np.any(camera[:, 2] <= .03):
        return None
    pixel = camera @ np.asarray(frame['intrinsics']).T
    return (pixel[:, :2] / pixel[:, 2:]).tolist()
