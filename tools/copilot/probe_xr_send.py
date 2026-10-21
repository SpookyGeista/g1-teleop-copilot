#!/usr/bin/env python3
"""Offline original-geometry -> Float32 -> original SDK CRC -> recording probe.

An archived snapshot is replayed at its original logical time. This is NOT live
state. No SDK/DDS module, controller constructor, socket or actuator is opened.
Run with the installed original CPU Pinocchio environment, not plain Python.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CRC_SHA256 = 'b95a530423f72c5acc96811f75677699cb3a185b6327353e7c44b355a425a20e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OfflineOriginalCRC:
    """Exact selected SDK method ASTs; SDK imports/constructor never execute."""
    def __init__(self, path):
        path = Path(path)
        if sha(path) != CRC_SHA256:
            raise ValueError('Original SDK CRC source drift: review before executing AST')
        tree = ast.parse(path.read_text())
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CRC')
        names = {'Crc', '__PackHGLowCmd', '__Trans', '_crc_py', '__Crc32'}
        methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in methods} != names:
            raise ValueError('Reviewed CRC methods missing')
        constructor = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        layout_node = next(node.value for node in ast.walk(constructor)
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and
                target.attr == '__packFmtHGLowCmd' for target in node.targets))
        # The format is an AST expression with string multiplication; evaluate
        # only its reviewed constant expression, never the SDK constructor.
        self.layout = eval(compile(ast.Expression(body=layout_node), str(path)+'#pinned_layout', 'eval'),
                           {'__builtins__': {}})
        self.raw = None
        def pack(*args):
            self.raw = struct.pack(*args)
            return self.raw
        selected = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), ast.ClassDef(name='CRC',
            bases=[], keywords=[], body=deepcopy(methods), decorator_list=[])], type_ignores=[])
        namespace = {'struct': SimpleNamespace(pack=pack)}
        exec(compile(ast.fix_missing_locations(selected), str(path)+'#selected_AST_only', 'exec'), namespace)
        cls = namespace['CRC']
        self.instance = cls.__new__(cls)
        self.instance.platform = 'offline_pure_python'
        self.instance._CRC__packFmtHGLowCmd = self.layout
        self.manifest = dict(source=str(path.resolve()), source_sha256=sha(path),
            original_methods=sorted(names), original_methods_unmodified=True,
            SDK_imported=False, constructor_called=False, CRC_backend='original_python_polynomial',
            HG_LowCmd_layout=self.layout, HG_LowCmd_bytes=struct.calcsize(self.layout))

    def crc(self, message):
        return self.instance.Crc(message)

    def payload(self, message):
        self.instance._CRC__PackHGLowCmd(message)
        return self.raw


def make_message(q14):
    """Recording fixture: zero FF; non-arm/gains are explicitly not authorized."""
    motors = [SimpleNamespace(mode=1, q=0., dq=0., tau=0., kp=40., kd=1.5, reserve=0)
              for _ in range(35)]
    for index, value in enumerate(q14): motors[15+index].q = value
    return SimpleNamespace(__idl_typename__='unitree_hg.msg.dds_.LowCmd_', mode_pr=0,
        mode_machine=0, motor_cmd=motors, reserve=[0]*4, crc=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=ROOT/'runs/copilot_sim_live_012/simulation/live_endpoint_evidence.json')
    parser.add_argument('--repo', type=Path, default=Path('/home/loya2/xr_teleoperate'))
    parser.add_argument('--sdk-crc', type=Path, default=Path('/home/loya2/unitree_sdk2_python/unitree_sdk2py/utils/crc.py'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists(): parser.error('Refusing to replace existing evidence')
    def forbid_network(event, values):
        if event.startswith('socket.'):
            raise RuntimeError('Network forbidden in offline recording probe: '+event)
    sys.addaudithook(forbid_network)
    import numpy as np
    from scipy.spatial.transform import Rotation
    from teleop_assist.copilot.guard import ActualMotionGuard
    from teleop_assist.copilot.xr_send import SimLowCmdHandoff
    original_hash = sha(args.snapshot)
    archived = json.loads(args.snapshot.read_text())['state_evidence']
    state, scene = archived['robot_state'], archived['scene']
    clock_id = state['clock_id']
    guard = ActualMotionGuard(args.repo)
    q = guard.collision._q(state['joint_positions'])
    guard.collision.pin.framesForwardKinematics(guard.collision.model, guard.collision.data, q)
    tcp = np.eye(4); tcp[0, 3] = .05
    left, right = [guard.collision.data.oMf[guard.collision.model.getFrameId(side+'_wrist_yaw_link')].homogeneous @ tcp
                   for side in ('left', 'right')]
    right[:3, :3] = Rotation.from_euler('z', 1, degrees=True).as_matrix() @ right[:3, :3]
    logical_now = max(state['acquisition_timestamp'], scene['acquisition_timestamp'])
    checked = guard.validate(left, right, state, scene, acquisition_timestamp=logical_now,
        clock_id=clock_id, now=logical_now, tracking_valid=True, command_format='unitree_float32')
    if not checked['allowed']: raise RuntimeError('Archived numerical guard rejected: '+checked['reason'])
    original_crc = OfflineOriginalCRC(args.sdk_crc)
    packets = []
    def record(message):
        payload = original_crc.payload(message)
        fields = [struct.unpack_from('<f', payload, 4+slot*28+4)[0] for slot in range(15, 29)]
        packets.append(dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
            q14_from_original_SDK_payload=fields, final_crc=struct.unpack_from('<I', payload, len(payload)-4)[0],
            recomputed_original_CRC=original_crc.crc(message)))
        return True
    def case(kind):
        now = [checked['validated_at']]
        gate = SimLowCmdHandoff(simulation_mode=True, clock_id=clock_id, clock=lambda: now[0])
        token = gate.submit(checked)
        inputs = [deepcopy(state), deepcopy(scene)]
        permit = [True]
        def crc(message):
            value = original_crc.crc(message)
            if kind == 'expired_after_CRC': now[0] = checked['valid_until']
            if kind == 'finger_changed_after_CRC':
                finger = next(name for name in inputs[0]['joint_positions'] if 'hand' in name)
                inputs[0]['joint_positions'][finger] += .001
            if kind == 'interlock_revoked_after_CRC': permit[0] = False
            if kind == 'command_mutated_after_CRC': message.motor_cmd[15].q += .125
            return value
        kwargs = dict(read_snapshot=lambda: tuple(inputs), interlock=lambda: permit[0],
            message_factory=make_message, crc=crc, write=record)
        first = gate.publish_once(token, **kwargs)
        replay = gate.publish_once(token, **kwargs)
        return dict(first=first, replay=replay)
    cases = {kind: case(kind) for kind in ('accepted', 'expired_after_CRC', 'finger_changed_after_CRC',
        'interlock_revoked_after_CRC', 'command_mutated_after_CRC')}
    checks = dict(guard_allowed=checked['allowed'], one_recorded_message=len(packets)==1,
        exact_float32_q14=bool(packets and packets[0]['q14_from_original_SDK_payload']==checked['q14_command']),
        original_crc_matches=bool(packets and packets[0]['final_crc']==packets[0]['recomputed_original_CRC']),
        positive_acknowledged=cases['accepted']['first']['accepted'],
        all_replays_rejected=all(not value['replay']['write_attempted'] for value in cases.values()),
        all_faults_rejected_before_write=all(not value['first']['write_attempted'] for name,value in cases.items() if name!='accepted'),
        archived_input_unchanged=sha(args.snapshot)==original_hash)
    imported = sorted(name for name in sys.modules if name.startswith(('unitree_sdk', 'cyclonedds')))
    checks['no_SDK_or_DDS_import'] = not imported
    result = dict(schema='punktnull.offline_xr_send_probe.v1', accepted=all(checks.values()),
        scope='archived_logical_time_CPU_guard_and_recorded_arm_transport_only',
        hardware_authorized=False, hardware_tested=False, DDS_started=False, GPU_used=False,
        live_authority=False, original_timestamps_preserved=True,
        source_hashes={str(path):sha(path) for path in (args.snapshot, Path(__file__),
            ROOT/'teleop_assist/copilot/guard.py', ROOT/'teleop_assist/copilot/xr_send.py')},
        CRC=original_crc.manifest, guard=checked, cases=cases, recorded_packets=packets,
        checks=checks, forbidden_imports_present=imported,
        limitations=['Recording fixture only; no original controller constructor, DDS serialization or real publisher was executed.',
            'SDK CRC packing is tested, not Cyclone DDS CDR serialization or actuator reception.',
            'Operator interlock is synthetic; no new complete assist/clutch integration is claimed.',
            'Non-arm motor values, PD gains and zero arm feedforward are recording fixtures, not an authorized torque policy.',
            'A writer stall after the final local check can exceed the deadline. No real watchdog or physical hold is implemented.',
            'This optional code is not wired into the original launcher or the released RC2 simulation.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(accepted=result['accepted'], checks=checks, output=str(args.output)), indent=2))
    return 0 if result['accepted'] else 1


if __name__ == '__main__': raise SystemExit(main())
