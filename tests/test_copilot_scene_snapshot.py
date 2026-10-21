"""AST-extracted actual bench methods; no Isaac/Torch module or constructor.

The frozen pre-extraction snapshot below is a behavioral reference, not an
alternate production implementation. Tensor-shaped doubles execute on NumPy.
"""
import ast
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.copilot.frames import robot_request
from teleop_assist.copilot.kinematics import ARM_JOINT_NAMES
from teleop_assist.copilot.original_sim_snapshot import CaptureContext, OriginalSimSnapshot

ROOT = Path(__file__).resolve().parents[1]
BENCH_SOURCE = ROOT/'teleop_assist/copilot/physx_bench.py'
RUNTIME_SOURCE = ROOT/'teleop_assist/copilot/original_sim_runtime.py'


class ArrayTensor:
    def __init__(self, values):
        self.values = np.asarray(values)
    def __getitem__(self, key):
        return ArrayTensor(self.values[key])
    def cpu(self):
        return self
    def numpy(self):
        return self.values


def quaternion_matrix(value):
    q = value.numpy()
    return ArrayTensor(Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix())


def extract_class(path, name, methods, namespace):
    tree = ast.parse(Path(path).read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    selected = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in selected} == set(methods)
    assert '__init__' not in methods
    cls = ast.ClassDef(name=name, bases=[], keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


def fixture_bench(*, legacy=False):
    clock = NS(value=17.125, calls=0)
    def monotonic():
        clock.calls += 1
        return clock.value
    namespace = dict(np=np, time=NS(monotonic=monotonic),
        matrix_from_quat=quaternion_matrix, robot_request=robot_request)
    cls = extract_class(BENCH_SOURCE, 'PhysXOrientationBench',
        ('pose', 'snapshot', 'scene_snapshot'), namespace)
    if legacy:
        exec(compile(textwrap.dedent(LEGACY_SNAPSHOT_SOURCE), '<frozen_snapshot_before_scene_extraction>', 'exec'), namespace)
        cls.snapshot = namespace['snapshot']
    bench = cls()
    names = list(ARM_JOINT_NAMES) + [f'fixture_other_joint_{i}' for i in range(29)]
    bodies = ['left_wrist_yaw_link', 'right_wrist_yaw_link'] + [f'fixture_body_{i}' for i in range(53)]
    body_poses = np.empty((1, 55, 13), dtype=np.float32)
    body_poses[:] = 0.
    for i in range(55):
        body_poses[0, i, :3] = [.1+i*.001, -.2+i*.002, .7+i*.001]
        body_poses[0, i, 3:7] = Rotation.from_euler('xyz', [i*.002, .1, -.2]).as_quat()[[3, 0, 1, 2]]
    base = np.zeros((1, 13), dtype=np.float32)
    base[0, :3] = [.2, -.1, .8]
    base[0, 3:7] = Rotation.from_euler('xyz', [.17, -.23, .31]).as_quat()[[3, 0, 1, 2]]
    target = np.zeros((1, 13), dtype=np.float32)
    target[0, :7] = [.4, -.18, .9, 1., 0., 0., 0.]
    target[0, 7:] = [.001, -.002, .003, -.004, .005, -.006]
    bench.robot = NS(joint_names=names, body_names=bodies, data=NS(
        root_state_w=ArrayTensor(base), joint_pos=ArrayTensor(np.linspace(-.2, .3, 43, dtype=np.float32)[None]),
        body_state_w=ArrayTensor(body_poses)))
    bench.object = NS(data=NS(root_state_w=ArrayTensor(target)))
    bench.clock_id, bench.calibration_id, bench.frame_id = 'fixture_clock', 'fixture_calibration', 19
    return bench, clock


def assert_exact(actual, expected):
    if isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_exact(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_exact(a, e)
    else:
        assert actual == expected


def test_full_snapshot_matches_frozen_previous_implementation_bit_for_bit():
    current, clock = fixture_bench()
    legacy, old_clock = fixture_bench(legacy=True)
    assert_exact(current.snapshot(), legacy.snapshot())
    assert clock.calls == old_clock.calls == 1


def test_scene_reads_new_target_pose_and_all_six_velocity_components():
    bench, clock = fixture_bench()
    first = bench.scene_snapshot(7.5)
    target = bench.object.data.root_state_w.values
    target[0, :3] = [.44, -.11, .97]
    target[0, 3:7] = Rotation.from_euler('xyz', [.3, -.2, .7]).as_quat()[[3, 0, 1, 2]]
    target[0, 7:] = [.01, -.02, .03, -.04, .05, -.06]
    second = bench.scene_snapshot(7.5)
    np.testing.assert_array_equal(second['obstacles'][1]['world_pose'], bench.pose(ArrayTensor(target[0, :7])))
    assert second['observed_target_velocity'] == target[0, 7:].tolist()
    assert first['obstacles'][1]['world_pose'] != second['obstacles'][1]['world_pose']
    assert first['observed_target_velocity'] != second['observed_target_velocity']
    assert second['acquisition_timestamp'] == first['acquisition_timestamp'] == 7.5
    assert clock.calls == 0


def test_scene_objects_are_private_and_mutating_return_cannot_change_later_reads():
    bench, _ = fixture_bench()
    first = bench.scene_snapshot(3.)
    expected = deepcopy(first)
    first['obstacles'][0]['size_xyz'][0] = 99.
    first['obstacles'][1]['world_pose'][0][3] = 99.
    first['obstacles'][2]['normal'][2] = -99.
    first['observed_target_velocity'][0] = 99.
    first['allowed_static_contacts'][0]['max_penetration_m'] = 99.
    assert_exact(bench.scene_snapshot(3.), expected)
    bench.object.data.root_state_w.values[:] = 0.
    assert expected['obstacles'][1]['world_pose'][0][3] != 0.


@pytest.mark.parametrize('stamp', [float('nan'), float('inf'), -float('inf'), None, True, '10', [10.], np.float64(10)])
def test_scene_rejects_unknown_or_nonfinite_acquisition_timestamp(stamp):
    bench, clock = fixture_bench()
    with pytest.raises(ValueError):
        bench.scene_snapshot(stamp)
    assert clock.calls == 0


@pytest.mark.parametrize('stamp', [0, 1, 3.125])
def test_scene_preserves_original_timestamp_without_robot_read_or_physics(stamp):
    bench, clock = fixture_bench()
    class Forbidden:
        def __getattr__(self, _):
            raise AssertionError('scene-only read touched robot or simulation')
    bench.robot = bench.sim = Forbidden()
    scene = bench.scene_snapshot(stamp)
    assert scene['acquisition_timestamp'] == stamp
    assert type(scene['acquisition_timestamp']) is type(stamp)
    assert clock.calls == 0


def test_original_runtime_reader_uses_scene_only_and_preserves_capture_context():
    tree = ast.parse(RUNTIME_SOURCE.read_text())
    reader = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'current_scene')
    bench, clock = fixture_bench()
    bench.snapshot = lambda: pytest.fail('duplicated full robot normalization')
    namespace = dict(bench=bench, asdict=asdict)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[reader], type_ignores=[])), str(RUNTIME_SOURCE), 'exec'), namespace)
    context = CaptureContext(8.5, bench.clock_id, 'fixture_instance', 71, bench.calibration_id)
    scene = namespace['current_scene'](object(), context)
    assert {key: scene[key] for key in asdict(context)} == asdict(context)
    assert scene['source'] == 'current_simulator_complete_collision_scene'
    assert clock.calls == 0


def test_runtime_sample_decodes_state_once_preserves_exact_tcps_and_private_outputs():
    bench, _ = fixture_bench()
    _, state, scene, _ = bench.snapshot()
    snapshot = OriginalSimSnapshot(json.dumps(state), json.dumps(scene), 'fixture_digest',
        17.125, 17.126, 'fixture_clock', 'fixture_source', 71, 2, 'fixture_calibration')
    cls = extract_class(RUNTIME_SOURCE, 'OriginalSimRuntime', ('_sample',), dict(np=np))
    class CountReads:
        state_reads = 0
        scene_reads = 0
        @property
        def robot_state(self):
            self.state_reads += 1
            return snapshot.robot_state
        @property
        def scene(self):
            self.scene_reads += 1
            return snapshot.scene
    counted = CountReads()
    nothing, output_state, output_scene, poses = cls._sample(counted)
    assert nothing is None and counted.state_reads == counted.scene_reads == 1
    np.testing.assert_array_equal(poses['left'], snapshot.left_base_ee)
    np.testing.assert_array_equal(poses['right'], snapshot.right_base_ee)
    output_state['base_in_world'][0][3] = 99.
    output_state['observed_body_world_poses']['left_wrist_yaw_link'][1][3] = 99.
    output_scene['obstacles'][1]['world_pose'][2][3] = 99.
    right_before = poses['right'].copy()
    poses['left'][:] = 99.
    np.testing.assert_array_equal(poses['right'], right_before)
    assert snapshot.robot_state == state and snapshot.scene == scene
    _, next_state, next_scene, next_poses = cls._sample(snapshot)
    assert next_state == state and next_scene == scene
    np.testing.assert_array_equal(next_poses['left'], snapshot.left_base_ee)
    np.testing.assert_array_equal(next_poses['right'], snapshot.right_base_ee)


# The exact pre-extraction method is embedded below for portable regression;
# no test relies on a /tmp checkout or imports the original Isaac module.
LEGACY_BENCH_SOURCE_SHA256 = '923544085bbb616718a07b8bbd9da11f9ac742e52c4e79e40992b6feef0a2e1d'
LEGACY_SNAPSHOT_SOURCE = r'''
    def snapshot(self):
        stamp = time.monotonic()
        base = self.pose(self.robot.data.root_state_w[0,:7])
        frame = dict(joint_names=np.asarray(self.robot.joint_names),
            joint_positions=self.robot.data.joint_pos[0].cpu().numpy().copy(),
            body_names=np.asarray(self.robot.body_names),
            body_poses=self.robot.data.body_state_w[0,:,:7].cpu().numpy().copy(),
            T_world_base=base, timestamp=stamp, clock_id=self.clock_id,
            calibration_id=self.calibration_id, frame_id=self.frame_id)
        table = np.eye(4); table[:3,3] = [.48,0,.84]
        target = self.pose(self.object.data.root_state_w[0,:7])
        scene = dict(scene_id='original_unitree_usd_minimal_table', geometry_complete=True, static=True,
            source='current_simulator_complete_collision_scene', acquisition_timestamp=stamp, clock_id=self.clock_id,
            obstacles=[dict(id='table',type='box',size_xyz=[.65,.8,.04],world_pose=table.tolist()),
                dict(id='target',type='cylinder',radius=.02,length=.08,world_pose=target.tolist()),
                dict(id='ground',type='plane',normal=[0,0,1],offset=0)],
            allowed_static_contacts=[dict(link=s+'_ankle_roll_link',obstacle='ground',max_penetration_m=.002) for s in ('left','right')])
        scene['observed_target_velocity'] = self.object.data.root_state_w[0,7:13].cpu().numpy().tolist()
        tcp = np.eye(4); tcp[0,3] = .05
        state=robot_request(frame)['robot_state']
        # Derive control TCPs from the exact same normalized body matrices
        # sent to the independent checker. A second float32 conversion differs
        # by ~1e-7 and must not be hidden by weakening identity checks.
        poses = {s:np.linalg.inv(base) @ np.asarray(state['observed_body_world_poses'][s+'_wrist_yaw_link']) @ tcp
            for s in ('left','right')}
        return frame, state, scene, poses
'''
