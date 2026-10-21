"""Post-fault diagnostics preserve a completed request, never a new lease."""
import hashlib
import json

import pytest

from tools.copilot.sim_runtime import retain_expired_guard_request

EXPIRED=[dict(reason='original_sender_lease_not_current',checked_at=12.151,valid_until=12.15)]
TIMING=dict(operation='validate',request_id=17)
RAW=b'{"operation":"validate","request_id":17,"now":12.0,"acquisition_timestamp":11.99}\n'


def test_exact_request_only_after_expiry_and_without_retimestamping(tmp_path):
    assert retain_expired_guard_request(tmp_path,RAW,TIMING,20,[]) is None
    assert list(tmp_path.iterdir())==[]
    evidence=retain_expired_guard_request(tmp_path,RAW,TIMING,21,EXPIRED)
    saved=(tmp_path/evidence['path']).read_bytes()
    assert saved==RAW and evidence['sha256']==hashlib.sha256(RAW).hexdigest()
    assert evidence['tick']==21 and evidence['request_id']==17
    assert evidence['diagnostic_only'] and evidence['motion_authorized'] is False
    assert json.loads(saved)['acquisition_timestamp']==11.99
    assert evidence['expired_interlocks']==EXPIRED
    with pytest.raises(FileExistsError):
        retain_expired_guard_request(tmp_path,RAW,TIMING,22,EXPIRED)
    assert (tmp_path/evidence['path']).read_bytes()==RAW


@pytest.mark.parametrize('raw,timing',[
    (None,TIMING),('',TIMING),(b'',TIMING),(b'x'*(2*1024*1024+1),TIMING),
    (RAW,None),(RAW,dict(operation='validate_sim_candidate',request_id=17)),
    (RAW,dict(operation='validate',request_id=18)),
    (b'not json',TIMING),(b'[]',TIMING),
    (b'{"operation":"validate_sim_candidate","request_id":17}',TIMING),
])
def test_unavailable_wrong_or_unbounded_request_is_not_saved(tmp_path,raw,timing):
    evidence=retain_expired_guard_request(tmp_path,raw,timing,1,EXPIRED)
    assert evidence['retained'] is False
    assert list(tmp_path.iterdir())==[]
