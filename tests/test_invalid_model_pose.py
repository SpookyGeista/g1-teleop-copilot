"""Bad model output resets tracking without relaxing calibration gates."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from teleop_assist.foundation_pose import FoundationPoseConfig, FoundationPoseEstimator


def estimator_with_output(pose):
    estimator = FoundationPoseEstimator.__new__(FoundationPoseEstimator)
    estimator.config = FoundationPoseConfig()
    estimator.torch = SimpleNamespace(inference_mode=nullcontext)
    estimator.estimator = SimpleNamespace(
        pose_last=np.eye(4), track_one=lambda **kwargs: pose,
        register=lambda **kwargs: pose)
    estimator._registered = True
    estimator._last_T_world_camera = None
    estimator._accepted_prior = {'timestamp':1., 'T_world_object':np.eye(4)}
    estimator._render_depth = lambda *args: np.ones((10,10), dtype=np.float32)
    return estimator


@pytest.mark.parametrize('kind', ['drift', 'reflection', 'nan', 'shape', 'bottom_row'])
@pytest.mark.parametrize('register', [False, True])
def test_invalid_model_output_is_rejected_and_requires_registration(kind, register):
    pose = np.eye(4)
    if kind == 'drift': pose[0,0] = 1.0002
    elif kind == 'reflection': pose[0,0] = -1
    elif kind == 'nan': pose[1,3] = np.nan
    elif kind == 'shape': pose = np.eye(3)
    else: pose[3,0] = .01
    estimator = estimator_with_output(pose)
    method = estimator.register if register else estimator.track
    args = (np.zeros((10,10,3), np.uint8), np.ones((10,10)),
            np.eye(3), np.ones((10,10), bool), np.eye(4))
    result = method(*args, timestamp=1.1)
    assert not result['accepted'] and not result['retained_tracking_prior']
    assert result['reason'] == 'invalid_model_rigid_transform'
    assert result['requires_registration']
    assert estimator._accepted_prior is None
    assert estimator.estimator.pose_last is None
    assert estimator._last_T_world_camera is None
    with pytest.raises(RuntimeError, match='registration'):
        estimator.track(*args, timestamp=1.2)


def test_bad_camera_calibration_still_raises():
    estimator = estimator_with_output(np.eye(4))
    camera = np.eye(4); camera[0,0] = 1.01
    with pytest.raises(ValueError, match='proper rigid transform'):
        estimator.track(np.zeros((10,10,3), np.uint8), np.ones((10,10)),
                        np.eye(3), np.ones((10,10), bool), camera, timestamp=1.1)
