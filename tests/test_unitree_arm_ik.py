"""The optional solver is tested in its own runtime, without changing Isaac."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from teleop_assist.unitree_arm_ik import (
    ARM_NAMES, AsyncUnitreeArmIK, PROJECT, PINNED_COMMIT, default_worker_python,
    pose_matrix, source_provenance,
)

URDF = Path(os.environ.get('PUNKTNULL_G1_REPO',
    '/home/loya2/WholeBodyTele/GR00T-WholeBodyControl')) / 'decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf'


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)


def independent_urdf_fk(named):
    """XML FK independent of Pinocchio, checking all fixed and movable origins."""
    joints = ET.parse(URDF).getroot().findall('joint')
    frames = {'pelvis': np.eye(4)}
    pending = list(joints)
    while pending:
        progressed = False
        for joint in pending[:]:
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent not in frames:
                continue
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            transform = np.eye(4)
            transform[:3, :3] = rotation([0, 0, 1], rpy[2])@rotation([0, 1, 0], rpy[1])@rotation([1, 0, 0], rpy[0])
            transform[:3, 3] = xyz
            if joint.get('type') != 'fixed':
                assert joint.get('type') == 'revolute'
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                transform[:3, :3] = transform[:3, :3]@rotation(axis, named.get(joint.get('name'), 0.))
            frames[child] = frames[parent]@transform
            pending.remove(joint)
            progressed = True
        assert progressed, 'Disconnected or cyclic test URDF'
    return frames


def quaternion(matrix):
    # Tests stay away from pi, so this independent trace formula is sufficient.
    w = np.sqrt(1+np.trace(matrix))/2
    return np.array([matrix[2, 1]-matrix[1, 2], matrix[0, 2]-matrix[2, 0],
                     matrix[1, 0]-matrix[0, 1], 4*w*w])/(4*w)


@pytest.fixture(scope='module')
def worker_python():
    if not URDF.exists():
        pytest.skip('Retained G1 URDF not installed')
    try:
        python = default_worker_python()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    result = subprocess.run([str(python), '-c', 'import pinocchio.casadi, casadi'], capture_output=True)
    if result.returncode:
        pytest.skip('Optional Pinocchio/CasADi worker runtime not installed')
    return python


def test_official_source_is_pinned_and_licensed(tmp_path):
    manifest = source_provenance()
    assert manifest['commit'] == PINNED_COMMIT
    source = PROJECT/'third_party/unitree_xr_ik'
    shutil.copytree(source, tmp_path/'source')
    (tmp_path/'source/robot_arm_ik.py').write_text('# changed reference\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        source_provenance(tmp_path/'source')


@pytest.mark.parametrize('position,orientation', [
    ([np.nan, 0, 0], [0, 0, 0, 1]), ([0, 0, 0], [0, 0, 0, 0]),
    ([0, 0, 0], [0, 0, 0, 2]), ([0, 0, 0], [0, 0, np.inf, 1]),
])
def test_invalid_world_poses_rejected(position, orientation):
    with pytest.raises(ValueError):
        pose_matrix(position, orientation)


def test_chain_mapping_and_tcp_match_independent_urdf_fk(worker_python):
    rng = np.random.default_rng(91)
    cases = []
    for _ in range(5):
        # Reversed dictionary insertion order catches accidental order-based mapping.
        named = dict(reversed(list(zip(ARM_NAMES, rng.uniform(-.2, .2, 7)))))
        named['right_elbow_joint'] = float(rng.uniform(.1, .7))
        named.update(waist_yaw_joint=.29, waist_roll_joint=-.13, waist_pitch_joint=.18,
                     left_shoulder_pitch_joint=.38, left_elbow_joint=.62)
        cases.append(named)
    code = """
import json, sys
from teleop_assist.unitree_arm_ik import UnitreeArmIK
s=UnitreeArmIK(sys.argv[1])
print(json.dumps([s.forward_torso(q).tolist() for q in json.loads(sys.stdin.read())]))
"""
    result = subprocess.run([str(worker_python), '-c', code, str(URDF)],
        input=json.dumps(cases), text=True, capture_output=True, cwd=PROJECT, timeout=20,
        env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    tcp = np.eye(4); tcp[:3, 3] = [.17, -.003, 0.]
    for named, matrix in zip(cases, actual):
        frames = independent_urdf_fk(named)
        expected = np.linalg.solve(frames['torso_link'], frames['right_wrist_yaw_link']@tcp)
        np.testing.assert_allclose(matrix, expected, atol=2e-12)


@pytest.fixture(scope='module')
def worker(worker_python, tmp_path_factory):
    with AsyncUnitreeArmIK(URDF, python=worker_python,
                          log_path=tmp_path_factory.mktemp('ik')/'worker.log') as worker:
        yield worker


def request_at_zero():
    named = dict.fromkeys(ARM_NAMES, 0.)
    frames = independent_urdf_fk(named)
    root = pose_matrix([.7, -.4, .85], [0., 0., np.sin(.3), np.cos(.3)])
    torso = root@frames['torso_link']
    tcp = np.eye(4); tcp[:3, 3] = [.17, -.003, 0.]
    target = root@frames['right_wrist_yaw_link']@tcp
    return [named, target[:3, 3], quaternion(target[:3, :3]),
            torso[:3, 3], quaternion(torso[:3, :3])]


def test_worker_solves_in_measured_world_torso_frame(worker):
    args = request_at_zero()
    result = worker.solve(*args)
    assert result['accepted'], result
    assert set(result['joint_targets']) == set(ARM_NAMES)
    assert result['position_error_m'] < 1e-5
    assert result['orientation_error_rad'] < 1e-5
    assert result['result_age_s'] >= result['solve_ms']/1000
    assert result['submitted_monotonic'] <= time.monotonic()
    assert worker.manifest['urdf_sha256'] == hashlib.sha256(URDF.read_bytes()).hexdigest()


def test_worker_handles_small_motion_and_position_only_retreat(worker):
    args = request_at_zero()
    args[1] = args[1]+[.015, 0., 0.]
    result = worker.solve(*args)
    assert result['accepted'], result
    assert result['position_error_m'] < .01
    assert any(abs(v) > .01 for v in result['joint_targets'].values())
    args[2] = None
    result = worker.solve(*args)
    assert result['accepted'], result
    assert result['orientation_error_rad'] is None


def test_infeasible_nonfinite_and_misnamed_inputs_never_produce_targets(worker):
    args = request_at_zero()
    args[1] = [100., 0., 0.]
    result = worker.solve(*args)
    assert not result['accepted'] and result['joint_targets'] == {}
    assert result['solver_status'] == 'target_outside_reach_bound'
    # Inside the coarse sphere but beyond this arm's kinematic reach: passing
    # the precheck must not turn a least-squares compromise into a valid target.
    args = request_at_zero()
    torso = pose_matrix(args[3], args[4])
    args[1] = torso[:3, 3]+torso[:3, :3]@np.array([1.1, 0., 0.])
    result = worker.solve(*args)
    assert not result['accepted'] and result['joint_targets'] == {}
    assert result['solver_status'] != 'target_outside_reach_bound'
    args = request_at_zero()
    args[0][ARM_NAMES[0]] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        worker.submit(*args)
    args = request_at_zero()
    del args[0][ARM_NAMES[3]]
    with pytest.raises(ValueError, match='Missing right-arm'):
        worker.submit(*args)


def test_reset_discards_pre_pause_results_and_latest_request_wins(worker):
    args = request_at_zero()
    worker.submit(*args)
    worker.reset()
    time.sleep(.05)
    assert worker.poll() is None
    last = None
    for _ in range(15):
        last = worker.submit(*args)
    deadline = time.monotonic()+2
    seen = None
    while time.monotonic() < deadline:
        result = worker.poll()
        if result is not None:
            seen = result
            if result['request_id'] == last:
                break
        time.sleep(.002)
    assert seen is not None and seen['request_id'] == last and seen['accepted']


def test_timeout_returns_no_target_and_next_request_can_recover(worker):
    worker.reset()
    result = worker.solve(*request_at_zero(), timeout=0.)
    assert not result['accepted'] and result['joint_targets'] == {}
    assert result['solver_status'] == 'timeout'
    result = worker.solve(*request_at_zero())
    assert result['accepted'], result
