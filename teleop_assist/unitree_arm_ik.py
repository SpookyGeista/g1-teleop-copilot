"""Optional, bounded comparison with Unitree's G1_29 optimization objective.

This is an adaptation, not an import of the entire upstream controller. It uses
our exact URDF/TCP, seven named right-arm joints, and the *measured* torso pose.
Unitree's 50/1/.02/.1 objective is retained; its bimanual moving filter and RNEA
feedforward are not. The caller keeps its measured PhysX dynamics, velocity
limits, contact supervision and deadman. No collision avoidance is provided.

Heavy dependencies live in a subprocess so Isaac's Python environment is not
modified. Submit/poll never block on optimization; only the explicit benchmark
solve() method waits. A pending request is replaced by the newest request.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import threading
import time

import numpy as np

ARM_NAMES = (
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
    'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
)
PROJECT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = '817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b'
PINNED_HASHES = {
    'robot_arm_ik.py': '42ea931d75ef70f67e55b7a61cf50cd7164dd1cef9ce83b8309ba7d144315250',
    'LICENSE': '63762ccebdb836cf2bae6b1ae0de526e7499dbf45e554c790ebc45942cce3e36',
}


def source_provenance(directory=None):
    """Fail closed if the stored official reference or license was modified."""
    directory = Path(directory) if directory else PROJECT/'third_party/unitree_xr_ik'
    manifest = json.loads((directory/'PROVENANCE.json').read_text())
    if manifest['commit'] != PINNED_COMMIT:
        raise ValueError('Unitree reference commit differs from the pinned revision')
    for name, expected in PINNED_HASHES.items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Unitree reference hash mismatch: {name}')
        if manifest['files'][name]['sha256'] != expected:
            raise ValueError(f'Unitree reference manifest hash mismatch: {name}')
    return manifest


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain exactly {size} finite values')
    return result


def pose_matrix(position, quaternion_xyzw):
    """Strict public XYZW quaternion to SE(3), without a SciPy dependency."""
    p = _vector(position, 3, 'position')
    q = _vector(quaternion_xyzw, 4, 'quaternion_xyzw')
    norm = float(np.linalg.norm(q))
    if abs(norm-1.) > .001:
        raise ValueError('quaternion_xyzw must be unit length')
    x, y, z, w = q/norm
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    matrix[:3, 3] = p
    return matrix


def _request(joint_positions, target_position_world, target_quaternion_xyzw,
             torso_position_world, torso_quaternion_xyzw):
    if not isinstance(joint_positions, Mapping):
        raise ValueError('joint_positions must be a mapping keyed by URDF joint name')
    missing = set(ARM_NAMES)-joint_positions.keys()
    if missing:
        raise ValueError(f'Missing right-arm joints: {sorted(missing)}')
    # Only these seven joints are optimized. All other chains end at or branch
    # away from the supplied measured torso frame and cannot change this FK.
    q = _vector([joint_positions[name] for name in ARM_NAMES], 7, 'right arm')
    world_target = pose_matrix(target_position_world, [0., 0., 0., 1.]
                               if target_quaternion_xyzw is None else target_quaternion_xyzw)
    world_torso = pose_matrix(torso_position_world, torso_quaternion_xyzw)
    target = np.linalg.solve(world_torso, world_target)
    return {'joint_positions': dict(zip(ARM_NAMES, q.tolist())),
            'orientation_weight': 0. if target_quaternion_xyzw is None else 1.,
            'target_torso': target.tolist()}


class UnitreeArmIK:
    """Synchronous solver; instantiate only where pinocchio.casadi is present."""

    def __init__(self, urdf, ee_offset=(.17, -.003, 0.),
                 max_position_error=.08, max_orientation_error=.35):
        import casadi as ca
        import pinocchio as pin
        from pinocchio import casadi as cpin

        provenance = source_provenance()
        self.ca, self.pin = ca, pin
        self.urdf = Path(urdf).resolve()
        self.offset = _vector(ee_offset, 3, 'ee_offset')
        self.max_position_error = float(max_position_error)
        self.max_orientation_error = float(max_orientation_error)
        if not (0 < self.max_position_error <= .2 and 0 < self.max_orientation_error <= .8):
            raise ValueError('Invalid IK acceptance thresholds')
        self.full_model = pin.buildModelFromUrdf(str(self.urdf))
        if any(not self.full_model.existJointName(name) for name in ARM_NAMES):
            raise ValueError('URDF is missing named G1 right-arm joints')
        locked = [i for i, name in enumerate(self.full_model.names)
                  if i and name not in ARM_NAMES]
        self.model = pin.buildReducedModel(self.full_model, locked, pin.neutral(self.full_model))
        if self.model.nq != 7 or self.model.nv != 7:
            raise ValueError('Expected seven scalar right-arm joints in the reduced URDF')
        self.indices = {name: self.model.joints[self.model.getJointId(name)].idx_q
                        for name in ARM_NAMES}
        if sorted(self.indices.values()) != list(range(7)):
            raise ValueError('Right-arm joint index map is not one-to-one')
        if any(self.model.joints[self.model.getJointId(name)].nq != 1 for name in ARM_NAMES):
            raise ValueError('Expected scalar right-arm joints')
        for name in ('torso_link', 'right_wrist_yaw_link'):
            if not self.model.existFrame(name):
                raise ValueError(f'URDF is missing required frame {name}')
        self.torso_id = self.model.getFrameId('torso_link')
        self.wrist_id = self.model.getFrameId('right_wrist_yaw_link')
        self.data = self.model.createData()
        pin.framesForwardKinematics(self.model, self.data, pin.neutral(self.model))
        self.base_torso = self.data.oMf[self.torso_id].copy()
        self.lower = np.asarray(self.model.lowerPositionLimit).copy()
        self.upper = np.asarray(self.model.upperPositionLimit).copy()

        cmodel = cpin.Model(self.model)
        cdata = cmodel.createData()
        cq = ca.SX.sym('right_arm_q', 7)
        cpin.framesForwardKinematics(cmodel, cdata, cq)
        wrist = cdata.oMf[self.wrist_id]
        torso_inverse = self.base_torso.inverse()
        tcp_position = ca.DM(torso_inverse.rotation) @ (
            wrist.translation + wrist.rotation@ca.DM(self.offset)) + ca.DM(torso_inverse.translation)
        tcp_rotation = ca.DM(torso_inverse.rotation)@wrist.rotation
        target = ca.SX.sym('target', 4, 4)
        position_error = ca.Function('position_error', [cq, target], [tcp_position-target[:3, 3]])
        rotation_error = ca.Function('rotation_error', [cq, target],
                                     [cpin.log3(tcp_rotation@target[:3, :3].T)])
        self.opti = ca.Opti()
        self.var_q = self.opti.variable(7)
        self.param_last = self.opti.parameter(7)
        self.param_target = self.opti.parameter(4, 4)
        self.param_orientation_weight = self.opti.parameter()
        self.opti.subject_to(self.opti.bounded(self.lower, self.var_q, self.upper))
        # Same G1_29 cost terms/weights as the pinned official source (line 177).
        self.opti.minimize(
            50*ca.sumsqr(position_error(self.var_q, self.param_target))
            + self.param_orientation_weight*ca.sumsqr(rotation_error(self.var_q, self.param_target))
            + .02*ca.sumsqr(self.var_q) + .1*ca.sumsqr(self.var_q-self.param_last))
        self.opti.solver('ipopt', {
            'expand': True, 'detect_simple_bounds': True, 'calc_lam_p': False,
            'print_time': False, 'ipopt.sb': 'yes', 'ipopt.print_level': 0,
            'ipopt.max_iter': 30, 'ipopt.max_cpu_time': .1,
            'ipopt.tol': 1e-4, 'ipopt.acceptable_tol': 5e-4,
            'ipopt.acceptable_iter': 5, 'ipopt.warm_start_init_point': 'yes',
            'ipopt.derivative_test': 'none', 'ipopt.jacobian_approximation': 'exact',
        })
        self.manifest = {
            'algorithm': 'Unitree G1_29 objective adapted to measured-torso right arm',
            'source': provenance, 'urdf': str(self.urdf),
            'urdf_sha256': hashlib.sha256(self.urdf.read_bytes()).hexdigest(),
            'arm_names': list(ARM_NAMES), 'model_q_indices': self.indices,
            'ee_offset_wrist_m': self.offset.tolist(), 'input_quaternion': 'XYZW',
            'weights': {'translation': 50., 'rotation': 1., 'regularization': .02, 'smooth': .1},
            'solver': 'IPOPT', 'max_iterations': 30, 'cpu_time_limit_s': .1,
            'max_position_error_m': self.max_position_error,
            'max_orientation_error_rad': self.max_orientation_error,
            'versions': {'pinocchio': pin.__version__, 'casadi': ca.__version__},
            'adaptations': ['measured torso frame instead of a zero waist',
                            'seven right-arm joints instead of fourteen bilateral joints',
                            'same hand URDF and manipulation TCP as Isaac',
                            'no upstream moving-average filter or RNEA output',
                            'reject failed/inaccurate/nonfinite outputs instead of debug iterates'],
            'collision_avoidance': False,
        }

    def configuration(self, joint_positions):
        named = _vector([joint_positions[name] for name in ARM_NAMES], 7, 'right arm')
        q = np.empty(7)
        for name, value in zip(ARM_NAMES, named):
            q[self.indices[name]] = value
        return q

    def forward_torso(self, joint_positions):
        q = self.configuration(joint_positions)
        self.pin.framesForwardKinematics(self.model, self.data, q)
        wrist = self.data.oMf[self.wrist_id]
        return (self.base_torso.inverse()*wrist*
                self.pin.SE3(np.eye(3), self.offset)).homogeneous.copy()

    def solve_request(self, request):
        started = time.perf_counter()
        result = {'request_id': request.get('request_id'), 'accepted': False,
                  'joint_targets': {}, 'position_error_m': None,
                  'orientation_error_rad': None, 'solver_status': 'invalid_input'}
        try:
            q = self.configuration(request['joint_positions'])
            target = np.asarray(request['target_torso'], float)
            orientation_weight = float(request.get('orientation_weight', 1.))
            if orientation_weight not in (0., 1.):
                raise ValueError('Orientation weight must be zero (position-only) or one (upstream objective)')
            if (target.shape != (4, 4) or not np.isfinite(target).all()
                    or not np.allclose(target[3], [0, 0, 0, 1], atol=1e-8)
                    or not np.allclose(target[:3, :3].T@target[:3, :3], np.eye(3), atol=1e-6)
                    or abs(np.linalg.det(target[:3, :3])-1) > 1e-6):
                raise ValueError('Target must be a finite rigid transform')
            if np.any(q < self.lower-.03) or np.any(q > self.upper+.03):
                raise ValueError('Measured right-arm configuration exceeds joint limits')
            # A deliberately loose sphere cheaply rejects grossly impossible
            # requests; final feasibility still comes from FK residual checks.
            if np.linalg.norm(target[:3, 3]) > 1.2:
                result['solver_status'] = 'target_outside_reach_bound'
            else:
                self.opti.set_initial(self.var_q, np.clip(q, self.lower, self.upper))
                self.opti.set_value(self.param_last, q)
                self.opti.set_value(self.param_target, target)
                self.opti.set_value(self.param_orientation_weight, orientation_weight)
                try:
                    solution = self.opti.solve()
                except RuntimeError:
                    result['solver_status'] = self.opti.stats().get('return_status', 'solver_failed')
                else:
                    candidate = np.asarray(solution.value(self.var_q), float).reshape(-1)
                    result['solver_status'] = self.opti.stats().get('return_status', 'unknown')
                    if (candidate.shape != (7,) or not np.isfinite(candidate).all()
                            or np.any(candidate < self.lower-1e-6) or np.any(candidate > self.upper+1e-6)):
                        result['solver_status'] = 'invalid_solver_output'
                    else:
                        named = {name: float(candidate[index]) for name, index in self.indices.items()}
                        actual = self.forward_torso(named)
                        position_error = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
                        rotation_error = float(np.linalg.norm(self.pin.log3(actual[:3, :3]@target[:3, :3].T)))
                        result.update(position_error_m=position_error,
                                      orientation_error_rad=rotation_error if orientation_weight else None)
                        result['accepted'] = (position_error <= self.max_position_error
                                              and (not orientation_weight or rotation_error <= self.max_orientation_error))
                        if result['accepted']:
                            result['joint_targets'] = named
                        else:
                            result['solver_status'] = 'residual_exceeds_threshold'
        except (ValueError, KeyError, TypeError) as exc:
            result['error'] = str(exc)
        result['solve_ms'] = 1000*(time.perf_counter()-started)
        return result

    def solve(self, joint_positions, target_position_world, target_quaternion_xyzw,
              torso_position_world, torso_quaternion_xyzw):
        return self.solve_request(_request(joint_positions, target_position_world,
            target_quaternion_xyzw, torso_position_world, torso_quaternion_xyzw))


def default_worker_python():
    """Use an explicitly configured runtime or the already audited local one."""
    configured = os.environ.get('PUNKTNULL_IK_PYTHON')
    candidate = Path(configured).expanduser() if configured else Path('/home/loya2/miniconda3/envs/to/bin/python')
    if not candidate.is_file():
        raise RuntimeError('Set PUNKTNULL_IK_PYTHON to Python with pinocchio.casadi and casadi; existing environments are never modified')
    return candidate


class AsyncUnitreeArmIK:
    """One bounded worker, one in-flight request and at most one pending request."""

    def __init__(self, urdf, ee_offset=(.17, -.003, 0.), python=None, log_path=None):
        source_provenance()
        self._condition = threading.Condition()
        self._pending = None
        self._latest = None
        self._completed = {}
        self._closed = False
        self._generation = 0
        self._counter = 0
        self._fatal = None
        python = Path(python) if python else default_worker_python()
        offset = _vector(ee_offset, 3, 'ee_offset')
        log_path = Path(log_path) if log_path else PROJECT/'.logs/unitree_arm_ik_worker.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open('a', buffering=1)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1')
        # Do not inherit Isaac/Kit's injected Python paths into the isolated
        # interpreter. Its own environment owns Pinocchio/CasADi and their ABI.
        env.pop('PYTHONHOME', None)
        env['PYTHONPATH'] = str(PROJECT)
        self._process = subprocess.Popen(
            [str(python), '-m', 'teleop_assist.unitree_ik_worker', '--urdf', str(Path(urdf).resolve()),
             '--ee-offset', *[str(value) for value in offset]],
            cwd=PROJECT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log, text=True, bufsize=1)
        try:
            if not select.select([self._process.stdout], [], [], 20.)[0]:
                raise RuntimeError(f'IK worker startup timed out; see {log_path}')
            ready = json.loads(self._process.stdout.readline())
            if ready.get('event') != 'ready':
                raise RuntimeError(f'IK worker failed to initialize: {ready}')
            self.manifest = {**ready['manifest'], 'worker_python': str(python),
                             'transport': 'latest-only JSON subprocess; no hardware access',
                             'worker_log': str(log_path)}
        except Exception:
            self._process.terminate()
            self._process.wait(timeout=3)
            self._log.close()
            raise
        self._thread = threading.Thread(target=self._exchange, name='unitree-ik', daemon=True)
        self._thread.start()

    def submit(self, joint_positions, target_position_world, target_quaternion_xyzw,
               torso_position_world, torso_quaternion_xyzw):
        request = _request(joint_positions, target_position_world, target_quaternion_xyzw,
                           torso_position_world, torso_quaternion_xyzw)
        with self._condition:
            if self._closed or self._fatal:
                raise RuntimeError(self._fatal or 'IK worker is closed')
            self._counter += 1
            request.update(request_id=self._counter, generation=self._generation,
                           submitted_at=time.monotonic())
            self._pending = request
            self._condition.notify_all()
            return self._counter

    def _exchange(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                request, self._pending = self._pending, None
            try:
                self._process.stdin.write(json.dumps(request, allow_nan=False)+'\n')
                self._process.stdin.flush()
                raw = self._process.stdout.readline()
                if not raw:
                    raise RuntimeError('IK worker exited unexpectedly')
                result = json.loads(raw)
                if result.get('request_id') != request['request_id']:
                    raise RuntimeError('IK worker response ID mismatch')
                result.update(submitted_at=request['submitted_at'],
                              submitted_monotonic=request['submitted_at'], received_at=time.monotonic(),
                              generation=request['generation'])
                with self._condition:
                    if request['generation'] == self._generation and not self._closed:
                        self._latest = result
                        self._completed[request['request_id']] = result
                        while len(self._completed) > 32:
                            self._completed.pop(next(iter(self._completed)))
                        self._condition.notify_all()
            except Exception as exc:
                with self._condition:
                    self._fatal = str(exc)
                    self._condition.notify_all()
                return

    def poll(self):
        with self._condition:
            result, self._latest = self._latest, None
            if result is not None:
                age = time.monotonic()-result['submitted_at']
                result = {**result, 'age_ms': 1000*age, 'result_age_s': age}
            return result

    def solve(self, joint_positions, target_position_world, target_quaternion_xyzw,
              torso_position_world, torso_quaternion_xyzw, timeout=2.):
        request_id = self.submit(joint_positions, target_position_world, target_quaternion_xyzw,
                                 torso_position_world, torso_quaternion_xyzw)
        with self._condition:
            self._condition.wait_for(lambda: request_id in self._completed or self._fatal or self._closed,
                                     timeout=max(0., float(timeout)))
            if request_id in self._completed:
                result = self._completed.pop(request_id)
                if self._latest and self._latest['request_id'] == request_id:
                    self._latest = None
                age = time.monotonic()-result['submitted_at']
                return {**result, 'age_ms': 1000*age, 'result_age_s': age}
            self._reset_locked()
            return {'request_id': request_id, 'accepted': False, 'joint_targets': {},
                    'solver_status': 'worker_failed' if self._fatal else 'timeout',
                    'error': self._fatal, 'position_error_m': None,
                    'orientation_error_rad': None, 'solve_ms': None}

    def _reset_locked(self):
        self._generation += 1
        self._pending = None
        self._latest = None
        self._completed.clear()
        self._condition.notify_all()

    def reset(self):
        """Discard requests from before a hold, recenter, calibration or stop."""
        with self._condition:
            self._reset_locked()

    def close(self):
        with self._condition:
            self._closed = True
            self._reset_locked()
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.)
        self._thread.join(timeout=2.)
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
