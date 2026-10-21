"""Pretrained GraspGenX worker backend. No robot, DDS or control imports."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from .observations import GeometryObservation, canonical_to_original_ee, rigid
from .assets import descriptor_evidence


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class GraspModel:
    def __init__(self, repository, descriptor, num_grasps=64, top_k=10):
        self.repository = Path(repository).resolve()
        self.descriptor_path = Path(descriptor).resolve()
        self.descriptor = json.loads(self.descriptor_path.read_text())
        self.descriptor_hash = sha256(self.descriptor_path)
        self.geometry_evidence = descriptor_evidence(self.descriptor_path)
        if self.descriptor.get('type') != 'revolute_3f':
            raise ValueError('This release requires a configured Dex3 three-finger descriptor')
        self.B = rigid(self.descriptor['base_rotation'], 'T_canonical_wrist')
        if not 10 <= top_k <= num_grasps <= 512:
            raise ValueError('Require 10 <= top_k <= num_grasps <= 512')
        self.num_grasps, self.top_k = int(num_grasps), int(top_k)
        checkpoints = self.repository / 'ext/graspgenx_checkpoints/release'
        assets = self.repository / 'ext/gripper_descriptions'
        required = [checkpoints / 'gen/epoch_736.pth', checkpoints / 'dis/epoch_1056.pth']
        if not all(p.is_file() for p in required) or not assets.is_dir():
            raise FileNotFoundError('Install pinned GraspGenX weights/assets before starting')
        os.environ['GRASPGENX_CHECKPOINT_DIR'] = str(checkpoints.parent)
        os.environ['GRASPGENX_GRIPPER_CFG_DIR'] = str(assets)
        sys.path.insert(0, str(self.repository))
        import torch
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.utils.checkpoint_io import load_model_cfg
        from graspgenx.x_grippers import make_sweep_volume_gripper_info
        if not torch.cuda.is_available():
            raise RuntimeError('GraspGenX CUDA unavailable; no synthetic candidates substituted')
        self.torch, self.sampler_class = torch, GraspGenXSampler
        cfg = load_model_cfg(checkpoints / 'gen', checkpoints / 'dis')
        sweep = self.descriptor['sweep_volume']
        if not all(np.asarray(sweep[k]).shape == (3,) and np.isfinite(sweep[k]).all()
                   for k in ('extents', 'offset', 'extents2', 'offset2')):
            raise ValueError('Invalid sweep volumes')
        if min(*sweep['extents'], *sweep['extents2']) <= 0:
            raise ValueError('Sweep volume extents must be positive')
        info = make_sweep_volume_gripper_info(extents_open=sweep['extents'], offset_open=sweep['offset'],
            extents_mid=sweep['extents2'], offset_mid=sweep['offset2'], gripper_type=2,
            fingertip_depth=float(self.descriptor['fingertip'][2]))
        self.sampler = GraspGenXSampler(cfg, gripper_info=info)
        self.manifest = {'backend': 'GraspGenX', 'descriptor_sha256': self.descriptor_hash,
                         'descriptor': str(self.descriptor_path), 'gpu': torch.cuda.get_device_name(),
                         'torch': torch.__version__, 'weights_sha256': {p.name: sha256(p) for p in required},
                         'descriptor_validation': self.geometry_evidence,
                         'motion_output': False}

    def infer(self, observation: GeometryObservation, seed=6103):
        points, scene = observation.point_clouds(seed=seed)
        self.torch.manual_seed(seed)
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        poses, scores = self.sampler_class.run_inference(points, self.sampler,
            num_grasps=self.num_grasps, topk_num_grasps=self.top_k, min_grasps=self.top_k,
            max_tries=1, remove_outliers=False)
        self.torch.cuda.synchronize()
        inference_s = time.perf_counter() - started
        poses, scores = poses.cpu().numpy(), scores.cpu().numpy()
        if not len(poses):
            raise ValueError('No grasp proposals')
        ee = canonical_to_original_ee(poses, self.B, observation.T_world_base)
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite grasp scores')
        candidates = [{'candidate_id': f'{observation.observation_id}:{i}', 'base_ee': pose.tolist(),
                       'world_canonical': canonical.tolist(), 'model_score': float(score),
                       'geometry_valid': True, 'ik_valid': False, 'collision_valid': False,
                       'source': 'GraspGenX_pretrained_current_Dex3',
                       'validation_reason': 'descriptor_verified_awaiting_kinematics_and_collision_checks'}
                      for i, (pose, canonical, score) in enumerate(zip(ee, poses, scores))]
        return {'target_id': observation.target_id, 'observation_id': observation.observation_id,
                'capture_timestamp': observation.capture_timestamp, 'clock_id': observation.clock_id,
                'calibration_id': observation.calibration_id, 'candidates': candidates,
                'inference_s': inference_s, 'object_points': len(points), 'scene_points': len(scene),
                'observed_centroid_world': points.mean(axis=0).tolist(),
                'descriptor_sha256': self.descriptor_hash, 'seed': seed}
