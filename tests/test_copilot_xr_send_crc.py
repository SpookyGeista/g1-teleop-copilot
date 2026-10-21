"""Original SDK packing and Python CRC against its native library, no DDS."""
from copy import deepcopy
import ctypes
from pathlib import Path
import random
import struct
import sys

import pytest

from tools.copilot.probe_xr_send import OfflineOriginalCRC, make_message, sha

SDK_CRC = Path('/home/loya2/unitree_sdk2_python/unitree_sdk2py/utils/crc.py')
NATIVE_SHA = 'b136a91a7e99c5cf914f465e131f63c897a17107939184a6f85d1a57cf315ade'


@pytest.fixture(scope='module')
def original_crc():
    if not SDK_CRC.is_file(): pytest.skip('Reviewed original SDK CRC source unavailable')
    return OfflineOriginalCRC(SDK_CRC)


def test_original_SDK_import_and_constructor_are_not_executed(original_crc):
    assert original_crc.manifest['constructor_called'] is False
    assert original_crc.manifest['SDK_imported'] is False
    assert not any(name.startswith(('unitree_sdk2py', 'cyclonedds')) for name in sys.modules)


def test_pinned_CRC_source_drift_is_rejected(tmp_path):
    altered = tmp_path/'crc.py'; altered.write_text('raise RuntimeError("should not execute")')
    with pytest.raises(ValueError, match='source drift'): OfflineOriginalCRC(altered)


def test_32_varied_lowcmd_payloads_match_native_original_CRC_and_q_slots(original_crc):
    library_path = SDK_CRC.parent/'lib/crc_amd64.so'
    if not library_path.is_file(): pytest.skip('Reviewed x86_64 original CRC library unavailable')
    if sha(library_path) != NATIVE_SHA: pytest.fail('Native CRC library changed; re-audit before loading')
    library = ctypes.CDLL(str(library_path))
    library.crc32_core.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32)
    library.crc32_core.restype = ctypes.c_uint32
    rng = random.Random(20261004)
    for _ in range(32):
        q14 = struct.unpack('<14f', struct.pack('<14f', *(rng.uniform(-2, 2) for _ in range(14))))
        msg = make_message(q14)
        msg.mode_machine = rng.randrange(256)
        msg.reserve = [rng.randrange(2**32) for _ in range(4)]
        for motor in msg.motor_cmd:
            motor.kp, motor.kd, motor.reserve = rng.uniform(1,300), rng.uniform(.1,5), rng.randrange(2**32)
        before = deepcopy(msg)
        expected = original_crc.crc(msg)
        assert msg == before
        msg.crc = expected
        raw = original_crc.payload(msg)
        assert len(raw) == 1004
        assert struct.unpack_from('<I', raw, 1000)[0] == expected
        count = len(raw)//4 - 1
        words = (ctypes.c_uint32*count)(*struct.unpack('<'+'I'*count, raw[:-4]))
        assert library.crc32_core(words, count) == expected
        decoded = tuple(struct.unpack_from('<f', raw, 4+28*slot+4)[0] for slot in range(15,29))
        assert decoded == q14
        msg.motor_cmd[15].q += .125
        assert original_crc.crc(msg) != expected
