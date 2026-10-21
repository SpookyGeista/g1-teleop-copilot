#!/usr/bin/env python3
"""Execute selected, pinned original XR methods with local recording fixtures.

This is a source audit, not a robot controller. No original module, constructor,
SDK, transport, CRC implementation, serialization, or background thread is used.
The actual unmodified publication-loop AST runs synchronously with fake time.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import copy
from enum import IntEnum
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARM_SOURCE = Path('teleop/robot_control/robot_arm.py')
ARM_SHA256 = '793e0774bb0a7a29b0cd163336babd45b29a36c3518528c87a854969c4132f73'
MOTOR_SOURCE = Path('unitree_sdk2py/idl/unitree_hg/msg/dds_/_MotorCmd_.py')
MOTOR_SHA256 = '710a8321be4d6731093b6f8328c70ab11606372561ed01a7af9a187143f54c4d'
METHODS = ('ctrl_dual_arm', 'clip_arm_q_target', '_ctrl_motor_state',
           'get_current_motor_q', 'get_current_dual_arm_q', 'get_current_dual_arm_dq')
ENUMS = ('G1_29_JointArmIndex', 'G1_29_JointIndex')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_tree(path, expected):
    content = Path(path).read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError(f'Unreviewed source hash: {path}; expected {expected}, got {actual}')
    return ast.parse(content, filename=str(path))


def extracted_module(path):
    """Select original bodies verbatim; intentionally never compile __init__."""
    tree = pinned_tree(path, ARM_SHA256)
    classes = {item.name:item for item in tree.body if isinstance(item, ast.ClassDef)}
    original = classes['G1_29_ArmController']
    methods = {item.name:item for item in original.body if isinstance(item, ast.FunctionDef)}
    selected = [copy.deepcopy(methods[name]) for name in METHODS]
    if any(item.decorator_list or item.args.defaults or item.args.kw_defaults for item in selected):
        raise ValueError('Selected methods must not execute decorators or default expressions')
    enums = [copy.deepcopy(classes[name]) for name in ENUMS]
    for enum in enums:
        if enum.decorator_list or enum.keywords or len(enum.bases) != 1 or ast.unparse(enum.bases[0]) != 'IntEnum':
            raise ValueError('Only reviewed literal IntEnum definitions may execute')
        if any(not isinstance(item, ast.Assign) or not isinstance(item.value, ast.Constant)
               or type(item.value.value) is not int for item in enum.body):
            raise ValueError('Enum body must consist only of integer assignments')
    controller = ast.ClassDef(name='ExtractedArmMethods', bases=[], keywords=[],
                              body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=enums+[controller], type_ignores=[]))
    if any(isinstance(item, (ast.Import, ast.ImportFrom)) for item in ast.walk(module)):
        raise ValueError('Imports are forbidden in the extracted AST')
    locations = {name:{'first_line':methods[name].lineno, 'last_line':methods[name].end_lineno}
                 for name in METHODS}
    return module, locations


class AuditFinished(BaseException):
    """End a bounded number of complete original publication-loop iterations."""


class FakeTime:
    def __init__(self, max_ticks):
        if type(max_ticks) is not int or not 1 <= max_ticks <= 1000:
            raise ValueError('A bounded tick count between 1 and 1000 is required')
        self.now = 1000.
        self.ticks = 0
        self.max_ticks = max_ticks

    def time(self):
        return self.now

    def sleep(self, duration):
        if not np.isfinite(duration) or not 0 <= duration <= .1:
            raise AssertionError('Unexpected original loop delay')
        self.now += duration
        self.ticks += 1
        if self.ticks == self.max_ticks:
            raise AuditFinished()


class RecordingPublisher:
    def __init__(self, clock):
        self.clock = clock
        self.records = []

    def Write(self, message):
        # Critical: copy NOW. The original reuses and mutates one message.
        self.records.append({'time':self.clock.time(), 'crc':message.crc,
                             'motor_cmd':copy.deepcopy([vars(item) for item in message.motor_cmd])})
        return True


class RecordingCRC:
    def __init__(self):
        self.calls = 0

    def Crc(self, _message):
        self.calls += 1
        return 123456  # Fixture sentinel, explicitly NOT the SDK CRC algorithm.


def state(q, dq=None):
    q = np.asarray(q, dtype=float)
    dq = np.zeros(35) if dq is None else np.asarray(dq, dtype=float)
    assert q.shape == dq.shape == (35,)
    return SimpleNamespace(motor_state=[SimpleNamespace(q=float(a), dq=float(b)) for a,b in zip(q,dq)])


class FixtureBuffer:
    def __init__(self, samples, *, switch_after=None):
        self.samples = samples
        self.switch_after = switch_after
        self.calls = 0

    def GetData(self):
        index = (self.calls % len(self.samples) if self.switch_after is None
                 else min(self.calls // self.switch_after, len(self.samples)-1))
        value = self.samples[index]
        self.calls += 1
        return value


def controller_fixture(path, *, simulation_mode, ticks=1, current_q=None):
    module, locations = extracted_module(path)
    clock = FakeTime(ticks)
    namespace = {'__name__':'_offline_original_xr_ast',
        '__builtins__':{'__build_class__':builtins.__build_class__, 'enumerate':enumerate, 'max':max},
        'IntEnum':IntEnum, 'np':np, 'time':clock}
    exec(compile(module, str(path), 'exec'), namespace)
    cls = namespace['ExtractedArmMethods']
    # This artificial class has no original constructor or inherited controller.
    controller = object.__new__(cls)
    controller.motion_mode = False
    controller.simulation_mode = simulation_mode
    controller.q_target = np.zeros(14)
    controller.tauff_target = np.zeros(14)
    controller.ctrl_lock = threading.Lock()
    controller.control_dt = 1./250.
    controller.arm_velocity_limit = 20.
    controller._speed_gradual_max = False
    controller.msg = SimpleNamespace(crc=0, motor_cmd=[SimpleNamespace(
        q=-10.-i/1000., dq=0., tau=0., mode=1, kp=80., kd=3.) for i in range(35)])
    controller.lowstate_buffer = FixtureBuffer([state(np.zeros(35) if current_q is None else current_q)])
    controller.crc = RecordingCRC()
    controller.lowcmd_publisher = RecordingPublisher(clock)
    return controller, clock, locations, namespace


def run_loop(controller, clock):
    try:
        controller._ctrl_motor_state()
    except AuditFinished:
        pass
    records = controller.lowcmd_publisher.records
    if len(records) != clock.max_ticks or controller.crc.calls != clock.max_ticks:
        raise AssertionError('The original loop did not complete the requested writes')
    return records


def arm_values(record, key='q'):
    return [item[key] for item in record['motor_cmd'][15:29]]


def sdk_numeric_contract(path):
    tree = pinned_tree(path, MOTOR_SHA256)
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'MotorCmd_')
    fields = {item.target.id:ast.unparse(item.annotation) for item in node.body if isinstance(item, ast.AnnAssign)}
    expected = {'q','dq','tau','kp','kd'}
    if any(fields.get(name) != 'types.float32' for name in expected):
        raise ValueError('Reviewed float32 MotorCmd contract has changed')
    return {'source':str(path), 'sha256':sha256(path), 'fields':fields,
        'evidence':'Static AST annotations only; no DDS serialization was invoked',
        'quantization_example':{'python_float':.1, 'numpy_float32_as_python_float':float(np.float32(.1))}}


def audit(repository, sdk_repository):
    source = Path(repository)/ARM_SOURCE
    sdk_source = Path(sdk_repository)/MOTOR_SOURCE
    before = {str(source):sha256(source), str(sdk_source):sha256(sdk_source)}
    sim, clock, locations, namespace = controller_fixture(source, simulation_mode=True, ticks=302)
    target = np.linspace(-.3, .4, 14)
    torques = np.linspace(.2, .5, 14)
    submitted_q, submitted_tau = target.copy(), torques.copy()
    original_body = copy.deepcopy([vars(item) for item in sim.msg.motor_cmd])
    sim.ctrl_dual_arm(target, torques)
    alias = {'q_reference_preserved':sim.q_target is target, 'tau_reference_preserved':sim.tauff_target is torques}
    target[2] += .6
    torques[3] += .7
    writes = run_loop(sim, clock)
    q_first, tau_first = arm_values(writes[0]), arm_values(writes[0], 'tau')
    same = all(arm_values(item)==q_first and arm_values(item,'tau')==tau_first for item in writes)
    non_arm_slots = list(range(15))+list(range(29,35))
    body_unchanged = all(item['motor_cmd'][index] == original_body[index] for item in writes for index in non_arm_slots)
    alias.update(submitted_q14=submitted_q.tolist(), submitted_tau14=submitted_tau.tolist(),
        mutated_input_q14=target.tolist(), mutated_input_tau14=torques.tolist(),
        first_published_q14=q_first, first_published_tau14=tau_first,
        mutation_after_submit_was_published=q_first==target.tolist() and tau_first==torques.tolist())
    repeat = {'writes':len(writes), 'original_loop_dt_s':sim.control_dt,
        'first_write_fake_timestamp':writes[0]['time'], 'last_write_fake_timestamp':writes[-1]['time'],
        'last_write_age_since_only_submit_s':writes[-1]['time']-writes[0]['time'],
        'target_updates_after_submit':0, 'all_writes_repeat_same_mutated_target':same,
        'non_arm_slots_unchanged':body_unchanged,
        'note':'Fake time demonstrates absent target expiration; not measured wall-clock throughput.'}

    current = np.zeros(35)
    current[15:29] = np.linspace(-.1, .1, 14)
    distant = np.linspace(-.4, .4, 14)
    distant[5] = 1.4  # Maximum delta is away from slot 0; clipping scales ALL deltas.
    real, real_clock, _, _ = controller_fixture(source, simulation_mode=False, current_q=current)
    sim2, sim2_clock, _, _ = controller_fixture(source, simulation_mode=True, current_q=current)
    direct_clip = real.clip_arm_q_target(distant, real.arm_velocity_limit)
    test_tau = np.arange(14)/20.
    real.ctrl_dual_arm(distant.copy(), test_tau.copy())
    sim2.ctrl_dual_arm(distant.copy(), test_tau.copy())
    real_write = run_loop(real, real_clock)[0]
    real_q = arm_values(real_write)
    sim_q = arm_values(run_loop(sim2, sim2_clock)[0])
    clipping = {'input_q14':distant.tolist(), 'measured_q14':current[15:29].tolist(),
        'hardware_branch_published_q14':real_q, 'simulation_branch_published_q14':sim_q,
        'direct_original_clip_q14':direct_clip.tolist(), 'velocity_limit_rad_s':real.arm_velocity_limit,
        'max_hardware_step_rad':float(np.max(np.abs(np.asarray(real_q)-current[15:29]))),
        'hardware_branch_changed_q14':real_q!=distant.tolist(),
        'hardware_loop_equals_original_clip':real_q==direct_clip.tolist(),
        'simulation_branch_preserves_q14':sim_q==distant.tolist(),
        'hardware_branch_preserves_tau':arm_values(real_write,'tau')==test_tau.tolist(),
        'hardware_branch_zero_dq':arm_values(real_write,'dq')==[0]*14,
        'note':'Both branches run only as AST methods with a RecordingPublisher, never on hardware.'}

    getters = {}
    slot_values = np.arange(35)/1000.
    sample_a, sample_b = state(slot_values,slot_values+10.), state(slot_values+1.,slot_values+11.)
    for method, count, field in [('get_current_motor_q',35,'q'),('get_current_dual_arm_q',14,'q'),
                                 ('get_current_dual_arm_dq',14,'dq')]:
        real.lowstate_buffer = FixtureBuffer([sample_a,sample_b], switch_after=7)
        mixed = getattr(real, method)()
        indices = list(range(35)) if count == 35 else list(range(15,29))
        values = [[getattr(sample.motor_state[index],field) for index in indices]
                  for sample in (sample_a,sample_b)]
        control = FixtureBuffer([sample_a,sample_b], switch_after=7)
        single_snapshot = control.GetData()
        single_read_values = [getattr(single_snapshot.motor_state[index],field) for index in indices]
        getters[method] = {'joint_count':count, 'GetData_calls':real.lowstate_buffer.calls,
            'returned':mixed.tolist(), 'source_slot_indices':indices, 'source_snapshot_values':values,
            'buffer_replaced_after_reads':7,
            'not_equal_to_either_complete_source_snapshot':bool(all(not np.array_equal(mixed,value) for value in values)),
            'single_GetData_control_calls':control.calls,
            'single_GetData_control_equals_snapshot_A':single_read_values==values[0],
            'note':'One simulated buffer replacement after seven reads; possible interleaving, not an observed hardware race.'}
    checks = [alias['q_reference_preserved'], alias['tau_reference_preserved'],
        alias['mutation_after_submit_was_published'], same, body_unchanged,
        repeat['last_write_age_since_only_submit_s']>1., clipping['hardware_branch_changed_q14'],
        clipping['hardware_loop_equals_original_clip'], clipping['simulation_branch_preserves_q14'],
        clipping['hardware_branch_preserves_tau'], clipping['hardware_branch_zero_dq']]
    checks += [item['not_equal_to_either_complete_source_snapshot'] and item['GetData_calls']==item['joint_count']
               and item['single_GetData_control_calls']==1 and item['single_GetData_control_equals_snapshot_A']
               for item in getters.values()]
    after = {path:sha256(path) for path in before}
    return {'schema':'punktnull.original_xr_send_audit.v1',
        'scope':'Offline AST execution of pinned original methods with recording fixtures',
        'network_or_dds_started':False, 'original_controller_constructed':False,
        'original_modules_imported':False, 'hardware_or_webxr_verified':False,
        'original_crc_or_wire_serialization_verified':False,
        'findings_reproduced':bool(all(checks)), 'sources_unchanged':before==after,
        'source_hashes':before, 'selected_methods':locations,
        'arm_slot_order':[{'name':item.name,'slot':item.value} for item in namespace[ENUMS[0]]],
        'aliasing':alias, 'repeated_old_target':repeat, 'clipping':clipping, 'getter_snapshot_mixing':getters,
        'motor_command_numeric_contract':sdk_numeric_contract(sdk_source),
        'limits':['No transport timing, physical stop behavior, full q43 freshness, or collision safety is certified.',
                  'The CRC and message container are fixtures. DDS float32 fields are evidenced statically only.',
                  'A guarded q14 cannot rely on this original repeated-send path retaining its lease or identity.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path('/home/loya2/xr_teleoperate'))
    parser.add_argument('--sdk-repository', type=Path, default=Path('/home/loya2/unitree_sdk2_python'))
    parser.add_argument('--output', type=Path, default=ROOT/'runs/copilot_xr_send_001/original_audit.json')
    args = parser.parse_args(argv)
    report = audit(args.repository, args.sdk_repository)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report':str(args.output), 'findings_reproduced':report['findings_reproduced'],
                      'sources_unchanged':report['sources_unchanged'], 'network_or_dds_started':False}))
    return 0 if report['findings_reproduced'] and report['sources_unchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
