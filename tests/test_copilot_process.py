import threading
import time
import numpy as np
import pytest

from teleop_assist.copilot.observations import GeometryObservation, canonical_to_original_ee
from teleop_assist.copilot.process import LatestGraspWorker


def observation(identifier='1'):
    return GeometryObservation('object', identifier, 12., 'test', 'cal', 'synthetic_test',
        np.ones((12, 12)), np.ones((12, 12), bool),
        np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1.]]), np.eye(4), np.eye(4))


def test_projected_points_and_frame_mapping():
    obs = observation()
    points, scene = obs.point_clouds()
    np.testing.assert_allclose(points[25], [.01, .02, 1], atol=1e-7)
    assert len(scene) == 0
    B = np.eye(4)
    B[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    W = np.eye(4); W[0, 3] = 1
    actual = canonical_to_original_ee(np.eye(4)[None], B, W)
    np.testing.assert_allclose(actual[0, :3, 3], [-1, .05, 0], atol=1e-7)
    np.testing.assert_allclose(actual[0, :3, :3], B[:3, :3])


def test_bad_depth_and_missing_capture_time_abstain():
    from dataclasses import replace
    obs = observation()
    with pytest.raises(ValueError, match='Acquisition time'):
        replace(obs, capture_timestamp=float('nan')).validate()
    with pytest.raises(ValueError, match='Insufficient'):
        replace(obs, depth=obs.depth * 1000).point_clouds()
    with pytest.raises(ValueError, match='proper rigid'):
        canonical_to_original_ee(np.zeros((1, 4, 4)), np.eye(4), np.eye(4))


class BlockingEngine:
    def __init__(self):
        self.started = threading.Event()
        self.proceed = threading.Event()
        self.seen = []
        self.closed = False

    def infer(self, obs):
        self.started.set()
        self.proceed.wait(timeout=2)
        self.seen.append((obs.observation_id, float(obs.depth[0, 0])))
        return {'observation_id': obs.observation_id, 'capture_timestamp': obs.capture_timestamp}

    def close(self):
        self.closed = True
        self.proceed.set()


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.001)
    assert predicate()


def test_latest_pending_replaces_old_work_without_blocking_or_sensor_buffer_alias():
    engine = BlockingEngine()
    worker = LatestGraspWorker(engine)
    try:
        obs = observation('first')
        worker.submit(obs)
        assert engine.started.wait(timeout=1)
        obs.depth[:] = 9
        worker.submit(observation('discarded'))
        worker.submit(observation('latest'))
        assert worker.status()['replaced'] == 1
        assert worker.status()['completed'] == 0  # submit did not wait for inference
        engine.proceed.set()
        wait_until(lambda: worker.status()['completed'] == 2)
        assert engine.seen == [('first', 1.), ('latest', 1.)]
        result = worker.poll()
        assert result == {'observation_id': 'latest', 'capture_timestamp': 12.}
        assert worker.poll() is None
    finally:
        worker.close()
    assert engine.closed and not worker.thread.is_alive()


def test_worker_failure_does_not_publish_fabricated_result():
    class Broken:
        def infer(self, obs):
            raise RuntimeError('inference failed')
        def close(self):
            pass
    worker = LatestGraspWorker(Broken())
    try:
        worker.submit(observation())
        wait_until(lambda: worker.status()['error'] is not None)
        assert worker.poll() is None
        with pytest.raises(RuntimeError, match='unavailable'):
            worker.submit(observation())
    finally:
        worker.close()
