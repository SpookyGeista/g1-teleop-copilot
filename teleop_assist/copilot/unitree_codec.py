"""Strict simulator-only q14 carrier using the real Unitree HG IDL/CDR codec.

Importing this module loads only the standard library. Constructing the codec
loads generated IDL and Cyclone's serialization library, but never constructs
a DDS entity/channel or an original controller/CRC object. This carrier has
zero motor modes and PD gains: it is not an actuating LowCmd controller.
Lease, source identity, scene/state validity and final consumption remain the
receiver's responsibility; a successful decode supplies no motion authority.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import inspect
import math
from pathlib import Path
import struct
import threading


ARM_SLOTS = tuple(range(15, 29))
CDR_HEADER = b'\x00\x01\x00\x00'
CDR_LENGTH = 1004  # Measured XCDR1 LE; NOT the SDK CRC payload plus four bytes.
CRC_SHA256 = 'b95a530423f72c5acc96811f75677699cb3a185b6327353e7c44b355a425a20e'
IDL_SHA256 = {
    'LowCmd_':'a23cec88e4bf782263ac2bea29d3908c9d21a44a14361ffc0bf3424cd36597ae',
    'MotorCmd_':'710a8321be4d6731093b6f8328c70ab11606372561ed01a7af9a187143f54c4d',
}
_IDL_LOCK = threading.RLock()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _exact_float32_q14(values):
    try:
        finite = (isinstance(values,(list,tuple)) and len(values)==14
            and all(type(value) in (int,float) and math.isfinite(value) for value in values))
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError('fourteen_finite_builtin_joint_angles_required')
    try:
        normalized = struct.unpack('<14f',struct.pack('<14f',*values))
    except (OverflowError,struct.error) as error:
        raise ValueError('joint_angles_outside_float32') from error
    if any(not math.isfinite(value) for value in normalized) or tuple(values)!=normalized:
        raise ValueError('joint_angles_must_already_be_exact_float32')
    return normalized


class ReviewedHGCRC:
    """Pinned original SDK HG packing/polynomial methods, no SDK import/init."""
    def __init__(self, source):
        source = Path(source)
        actual = _sha(source)
        if actual != CRC_SHA256:
            raise ValueError('unreviewed_original_CRC_source')
        tree = ast.parse(source.read_text(),filename=str(source))
        original = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='CRC')
        names = {'Crc','__PackHGLowCmd','__Trans','_crc_py','__Crc32'}
        methods = [copy.deepcopy(node) for node in original.body if isinstance(node,ast.FunctionDef) and node.name in names]
        if {node.name for node in methods} != names:
            raise ValueError('reviewed_CRC_methods_missing')
        constructor = next(node for node in original.body if isinstance(node,ast.FunctionDef) and node.name=='__init__')
        layout_node = next(node.value for node in ast.walk(constructor) if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Attribute) and target.attr=='__packFmtHGLowCmd' for target in node.targets))
        # This constant string expression comes from the hash-pinned source.
        self.layout = eval(compile(ast.Expression(body=layout_node),str(source)+'#layout','eval'),{'__builtins__':{}})
        module = ast.fix_missing_locations(ast.Module(body=[
            ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
            ast.ClassDef(name='CRC',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]))
        namespace = {'struct':struct}
        exec(compile(module,str(source)+'#reviewed_methods','exec'),namespace)
        self._instance = object.__new__(namespace['CRC'])
        self._instance.platform = 'pure_python_no_CRC_constructor'
        self._instance._CRC__packFmtHGLowCmd = self.layout
        self.manifest = {'source':str(source),'sha256':actual,'original_methods':sorted(names),
            'original_methods_unmodified':True,'CRC_constructor_called':False,
            'backend':'original_python_polynomial','SDK_HG_payload_length':struct.calcsize(self.layout)}

    def crc(self,message):
        return self._instance.Crc(message)


class UnitreeSimArmCodec:
    """Encode/decode a fixed, inactive HG LowCmd q14 carrier for simulation.

    q14 values must ALREADY equal their float32 representation. No rounding,
    smoothing, IK, clipping or fresh timestamp is introduced by this codec.
    """
    def __init__(self, *, simulation_mode, sdk_crc_path=None):
        if simulation_mode is not True:
            raise ValueError('simulator_only_q14_codec')
        idl = importlib.import_module('unitree_sdk2py.idl.unitree_hg.msg.dds_')
        support = importlib.import_module('cyclonedds.idl._support')
        self._LowCmd, self._MotorCmd = idl.LowCmd_, idl.MotorCmd_
        self._Buffer, self._little = support.Buffer, support.Endianness.Little
        sources = {}
        for name in IDL_SHA256:
            path = Path(inspect.getfile(getattr(idl,name)))
            actual = _sha(path)
            if actual != IDL_SHA256[name]:
                raise ValueError('unreviewed_Unitree_IDL_source:'+name)
            sources[str(path)] = actual
        if sdk_crc_path is None:
            package = importlib.import_module('unitree_sdk2py')
            sdk_crc_path = Path(package.__file__).parent/'utils/crc.py'
        self._crc = ReviewedHGCRC(sdk_crc_path)
        self.manifest = {'scope':'inactive_simulator_q14_carrier_only','hardware_authorized':False,
            'transport_created':False,'arm_slots':list(ARM_SLOTS),'format':'XCDR1_little_endian',
            'cdr_header_hex':CDR_HEADER.hex(),'cdr_length':CDR_LENGTH,
            'motor_mode':0,'kp':0.,'kd':0.,'dq':0.,'tau':0.,'non_arm_q':0.,
            'IDL_source_hashes':sources,'CRC':dict(self._crc.manifest)}
        # Populate and verify this installed serializer once before use. A
        # fresh buffer per call avoids Cyclone's shared type-level buffer.
        self.encode_arm_targets([0.]*14)

    def _message(self,q14):
        motors = [self._MotorCmd(0,0.,0.,0.,0.,0.,0) for _ in range(35)]
        for slot,value in zip(ARM_SLOTS,q14):
            motors[slot].q = value
        return self._LowCmd(0,0,motors,[0]*4,0)

    def _serialize(self,message):
        with _IDL_LOCK:
            return message.serialize(buffer=self._Buffer(),endianness=self._little,use_version_2=False)

    def encode_arm_targets(self,q14):
        q14 = _exact_float32_q14(q14)
        message = self._message(q14)
        message.crc = self._crc.crc(message)
        payload = self._serialize(message)
        if len(payload)!=CDR_LENGTH or payload[:4]!=CDR_HEADER:
            raise ValueError('installed_CDR_format_differs_from_reviewed_contract')
        return payload

    def decode_arm_targets(self,payload):
        if type(payload) is not bytes:
            raise ValueError('immutable_CDR_bytes_required')
        if len(payload)!=CDR_LENGTH or payload[:4]!=CDR_HEADER:
            raise ValueError('CDR_header_or_length_invalid')
        try:
            with _IDL_LOCK:
                message = self._LowCmd.deserialize(payload)
        except Exception as error:
            raise ValueError('CDR_deserialization_failed') from error
        if message.mode_pr!=0 or message.mode_machine!=0 or message.reserve!=[0]*4 or len(message.motor_cmd)!=35:
            raise ValueError('inactive_sim_carrier_header_policy_violated')
        for slot,motor in enumerate(message.motor_cmd):
            if any(getattr(motor,name)!=0 for name in ('mode','dq','tau','kp','kd','reserve')):
                raise ValueError('inactive_sim_carrier_motor_policy_violated')
            if slot not in ARM_SLOTS and motor.q!=0:
                raise ValueError('inactive_sim_carrier_non_arm_q_policy_violated')
        q14 = _exact_float32_q14([message.motor_cmd[slot].q for slot in ARM_SLOTS])
        if message.crc != self._crc.crc(message):
            raise ValueError('original_HG_CRC_mismatch')
        # Rebuild fixed fields as well as padding. Numeric comparisons above
        # would otherwise accept negative-zero encodings in inactive fields.
        canonical = self._message(q14)
        canonical.crc = message.crc
        if self._serialize(canonical)!=payload:
            raise ValueError('noncanonical_CDR_padding_or_carrier_representation')
        return list(q14)
