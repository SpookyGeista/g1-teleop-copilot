"""Metric, time-stamped observed geometry for the grasp copilot."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def rigid(value, name='transform'):
    value = np.array(value, dtype=float, copy=True)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5)):
        raise ValueError(f'{name} must be a finite proper rigid transform')
    return value


@dataclass(frozen=True)
class GeometryObservation:
    target_id: str
    observation_id: str
    capture_timestamp: float
    clock_id: str
    calibration_id: str
    source: str
    depth: np.ndarray
    mask: np.ndarray
    intrinsics: np.ndarray
    T_world_camera: np.ndarray
    T_world_base: np.ndarray

    def validate(self):
        for field in ('target_id', 'observation_id', 'clock_id', 'calibration_id', 'source'):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f'Missing {field}')
        if not np.isfinite(self.capture_timestamp):
            raise ValueError('Acquisition time must be known in the named clock')
        depth, mask = np.asarray(self.depth), np.asarray(self.mask)
        if depth.ndim != 2 or depth.size > 4096 * 2160 or depth.shape != mask.shape or mask.dtype != bool:
            raise ValueError('Matching metric depth and boolean mask are required')
        K = np.asarray(self.intrinsics, float)
        if (K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
                or not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0):
            raise ValueError('Rectified pinhole calibration required')
        rigid(self.T_world_camera, 'T_world_camera')
        rigid(self.T_world_base, 'T_world_base')

    def arrays(self):
        self.validate()
        return {name: np.array(getattr(self, name), copy=True) for name in self.__dataclass_fields__}

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            values = {name: data[name].copy() for name in cls.__dataclass_fields__}
        for name in ('target_id', 'observation_id', 'clock_id', 'calibration_id', 'source'):
            values[name] = str(values[name].item())
        values['capture_timestamp'] = float(values['capture_timestamp'])
        result = cls(**values)
        result.validate()
        return result

    def point_clouds(self, max_object_points=3500, max_scene_points=40000, seed=0):
        self.validate()
        depth = np.asarray(self.depth, float)
        K = np.asarray(self.intrinsics, float)
        valid = np.isfinite(depth) & (depth > .03) & (depth < 3.)
        masked = np.asarray(self.mask) & valid
        if int(masked.sum()) < 100 or masked.sum() < .70 * max(1, self.mask.sum()):
            raise ValueError('Insufficient valid observed object depth; abstain')
        rng = np.random.default_rng(seed)
        def backproject(selection, maximum):
            y, x = np.nonzero(selection)
            if len(y) > maximum:
                keep = rng.choice(len(y), maximum, replace=False)
                y, x = y[keep], x[keep]
            z = depth[y, x]
            camera = np.column_stack(((x-K[0, 2])*z/K[0, 0], (y-K[1, 2])*z/K[1, 1], z))
            T = np.asarray(self.T_world_camera)
            return (camera @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        return backproject(masked, max_object_points), backproject(valid & ~self.mask, max_scene_points)


def canonical_to_original_ee(grasps_world, canonical_from_wrist, world_from_base):
    """GraspGenX places its canonical mesh root; do not append a fingertip TCP."""
    B = rigid(canonical_from_wrist, 'T_canonical_wrist')
    world_base = rigid(world_from_base, 'T_world_base')
    wrist_ee = np.eye(4)
    wrist_ee[0, 3] = .05  # original Unitree G1_29 R_ee definition
    poses = np.asarray(grasps_world, float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError('Expected Nx4x4 grasp poses')
    for pose in poses:
        rigid(pose, 'grasp')
    return np.linalg.inv(world_base)[None] @ poses @ B[None] @ wrist_ee[None]
