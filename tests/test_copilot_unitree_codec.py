"""Actual IDL tests run in children so offline suites retain no DDS imports."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from teleop_assist.copilot.unitree_codec import ReviewedHGCRC, UnitreeSimArmCodec

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = r'''
import importlib.abc,json,sys
attempts=[]
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in {'unitree_sdk2py.core.channel','unitree_sdk2py.utils.crc'} or fullname.split('.')[0] in {'robot_control','logging_mp','torch','omni','isaaclab'}:
   attempts.append(fullname);raise RuntimeError('Forbidden import '+fullname)
sys.meta_path.insert(0,Block())
def audit(event,args):
 if event.startswith('socket.') or event in {'subprocess.Popen','os.fork','os.system'}:
  attempts.append(event);raise RuntimeError('Forbidden effect '+event)
sys.addaudithook(audit)
from teleop_assist.copilot.unitree_codec import UnitreeSimArmCodec,CDR_HEADER,CDR_LENGTH
assert not any(name.startswith(('unitree_sdk2py','cyclonedds')) for name in sys.modules)
import cyclonedds,inspect,struct,copy
from cyclonedds.core import Entity
def deny(self,*args,**kwargs):
 attempts.append(type(self).__name__);raise RuntimeError('Forbidden DDS entity')
for mod in (cyclonedds.domain,cyclonedds.topic,cyclonedds.pub,cyclonedds.sub,cyclonedds.core,cyclonedds.builtin):
 for value in vars(mod).values():
  if inspect.isclass(value) and issubclass(value,Entity):value.__init__=deny
codec=UnitreeSimArmCodec(simulation_mode=True)
q=list(struct.unpack('<14f',struct.pack('<14f',*(.1+i*.03 for i in range(14)))))
payload=codec.encode_arm_targets(q)
def rejected(fn):
 try:fn()
 except ValueError as error:return str(error)
 raise AssertionError('Expected rejection')
'''


def child(code):
    environment = dict(os.environ)
    # Source-installed Cyclone probes ldconfig in a subprocess before its
    # configured install path. Resolve that already installed path explicitly
    # so the test can keep its stronger no-subprocess audit guard enabled.
    spec = importlib.util.find_spec('cyclonedds')
    if spec is not None and spec.origin:
        library = Path(spec.origin).parent/'__library__.py'
        if library.is_file():
            assignments = {node.targets[0].id:ast.literal_eval(node.value)
                for node in ast.parse(library.read_text()).body if isinstance(node,ast.Assign)
                and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                and isinstance(node.value,ast.Constant)}
            configured = Path(assignments.get('library_path',''))
            if assignments.get('in_wheel') is False and configured.is_file() and configured.parent.name=='lib':
                environment['CYCLONEDDS_HOME'] = str(configured.parent.parent)
    result = subprocess.run([sys.executable,'-c',PRELUDE+'\n'+code+'''\nassert not Entity._entities
assert not attempts
assert 'unitree_sdk2py.core.channel' not in sys.modules
print(json.dumps({'passed':True}))
'''],cwd=ROOT,env=environment,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr+result.stdout
    assert json.loads(result.stdout)['passed']


def test_import_is_lazy_and_real_mode_rejected_before_IDL_import():
    with pytest.raises(ValueError,match='simulator_only'):
        UnitreeSimArmCodec(simulation_mode=False)


def test_CRC_source_drift_rejected_without_original_import(tmp_path):
    path=tmp_path/'crc.py';path.write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError,match='unreviewed_original_CRC_source'):
        ReviewedHGCRC(path)


def test_real_IDL_roundtrip_has_exact_float32_values_fixed_layout_and_zero_carrier_fields():
    child('''assert codec.decode_arm_targets(payload)==q
assert len(payload)==CDR_LENGTH==1004 and payload[:4]==CDR_HEADER
message=codec._LowCmd.deserialize(payload)
assert [message.motor_cmd[15+i].q for i in range(14)]==q
assert all(m.mode==m.kp==m.kd==m.tau==m.dq==m.reserve==0 for m in message.motor_cmd)
assert all(m.q==0 for i,m in enumerate(message.motor_cmd) if not 15<=i<29)
assert message.crc==codec._crc.crc(message)
assert codec.encode_arm_targets(codec.decode_arm_targets(payload))==payload
assert codec.manifest['hardware_authorized'] is False
''')


@pytest.mark.parametrize('expression', ['[.1]*14','[True]*14','[float("nan")]*14',
    '[float("inf")]*14','[1e100]*14','[10**1000]*14','[0.]*13','[0.]*15','"not joints"'])
def test_encode_rejects_rounding_nonfinite_or_malformed_joint_vectors(expression):
    child(f'rejected(lambda:codec.encode_arm_targets({expression}))')


@pytest.mark.parametrize('expression', ['payload[:-1]','payload+b"x"','b"\\x00\\x00\\x00\\x00"+payload[4:]',
    'b"\\x00\\x07\\x00\\x00"+payload[4:]','bytearray(payload)'])
def test_decode_rejects_length_header_and_mutable_payload(expression):
    child(f'rejected(lambda:codec.decode_arm_targets({expression}))')


@pytest.mark.parametrize('assignment,reason', [
    ('message.mode_pr=1','header_policy'),('message.mode_machine=1','header_policy'),
    ('message.reserve[0]=1','header_policy'),('message.motor_cmd[15].mode=1','motor_policy'),
    ('message.motor_cmd[15].kp=1.','motor_policy'),('message.motor_cmd[15].kd=1.','motor_policy'),
    ('message.motor_cmd[15].dq=.125','motor_policy'),('message.motor_cmd[15].tau=.125','motor_policy'),
    ('message.motor_cmd[15].reserve=1','motor_policy'),('message.motor_cmd[0].q=.125','non_arm_q_policy')])
def test_valid_CRC_does_not_authorize_other_LowCmd_fields(assignment,reason):
    child('message=codec._LowCmd.deserialize(payload)\n'+assignment+f'''\nmessage.crc=codec._crc.crc(message)
changed=codec._serialize(message)
assert {reason!r} in rejected(lambda:codec.decode_arm_targets(changed))
''')


def test_tampered_joint_with_old_CRC_and_changed_CRC_are_rejected():
    child('''message=codec._LowCmd.deserialize(payload)
message.motor_cmd[18].q+=.125
assert 'CRC_mismatch' in rejected(lambda:codec.decode_arm_targets(codec._serialize(message)))
message=codec._LowCmd.deserialize(payload);message.crc^=1
assert 'CRC_mismatch' in rejected(lambda:codec.decode_arm_targets(codec._serialize(message)))
''')


def test_nonfinite_decoded_q_is_rejected_even_when_original_CRC_matches():
    child('''message=codec._LowCmd.deserialize(payload);message.motor_cmd[18].q=float('nan')
message.crc=codec._crc.crc(message)
assert 'finite' in rejected(lambda:codec.decode_arm_targets(codec._serialize(message)))
''')


def test_signed_zero_is_preserved_for_target_but_fixed_inactive_fields_are_canonical():
    child('''targets=[-0.]+q[1:]
roundtrip=codec.decode_arm_targets(codec.encode_arm_targets(targets))
assert struct.pack('<f',roundtrip[0])==struct.pack('<f',-0.)
message=codec._LowCmd.deserialize(payload);message.motor_cmd[0].q=-0.
message.crc=codec._crc.crc(message)
assert 'noncanonical' in rejected(lambda:codec.decode_arm_targets(codec._serialize(message)))
''')


def test_target_bits_survive_negative_zero_subnormal_and_finite_float32_extremes():
    child('''bits=[0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff,
0x3f800000,0xbf800000,0x3eaaaaab,0xbeaaaaab,0x00800000,0x80800000,0x3f000000,0xbf000000]
expected=struct.pack('<14I',*bits)
targets=list(struct.unpack('<14f',expected))
encoded=codec.encode_arm_targets(targets)
assert struct.pack('<14f',*codec.decode_arm_targets(encoded))==expected
actual=codec._LowCmd.deserialize(encoded)
assert struct.pack('<14f',*(actual.motor_cmd[15+i].q for i in range(14)))==expected
''')


def test_private_buffers_and_returned_lists_do_not_alias_later_roundtrips():
    child('''first=codec.decode_arm_targets(payload);first[0]=999.
other=UnitreeSimArmCodec(simulation_mode=True)
zero=other.encode_arm_targets([0.]*14)
assert codec.decode_arm_targets(payload)==q
assert codec.decode_arm_targets(zero)==[0.]*14
''')


def test_changed_CDR_padding_is_rejected_even_though_fields_and_CRC_still_decode():
    child('''# Find a real padding byte from the actual serializer; do not assume
# the SDK CRC-packing offsets equal the CDR field offsets.
original=codec._LowCmd.deserialize(payload)
from dataclasses import asdict
padding=None
for index in range(4,len(payload)):
 changed=bytearray(payload);changed[index]^=1
 try:decoded=codec._LowCmd.deserialize(bytes(changed))
 except Exception:continue
 if asdict(decoded)==asdict(original):padding=bytes(changed);break
assert padding is not None
assert 'noncanonical' in rejected(lambda:codec.decode_arm_targets(padding))
''')
