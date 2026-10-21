"""Bounded producer/consumer synchronization for the isolated simulator.

This is a scheduling barrier, not perception or motion authorization. It runs
before human/state acquisition and before any command is offered. A successful
acknowledgment still has to pass every independent feed and handoff check.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

from .mailbox import read_json


def await_preview_frame(directory, frame, *, timeout_s=.10, poll_s=.002,
                        clock=time.monotonic, sleep=time.sleep, cancelled=lambda: False):
    if not (0 < timeout_s <= .10 and 0 < poll_s <= timeout_s):
        raise ValueError('The simulator frame barrier is bounded to 100 ms')
    frame_id=frame.get('frame_id');stamp=frame.get('timestamp')
    clock_id=frame.get('clock_id');calibration=frame.get('calibration_id')
    if (type(frame_id) is not int or frame_id < 0 or type(stamp) not in (int,float)
            or not math.isfinite(stamp) or not isinstance(clock_id,str) or not clock_id
            or not isinstance(calibration,str) or not calibration):
        raise ValueError('An exact captured simulator frame is required')
    started=clock();last=started;expected=f'{calibration}:{frame_id}'
    valid_start=type(started) in (int,float) and math.isfinite(started)
    result=dict(schema='punktnull.sim_frame_barrier.v1',ready=False,
        reason='current_frame_preview_timeout',frame_id=frame_id,
        capture_timestamp=stamp,clock_id=clock_id,calibration_id=calibration,
        expected_validation_observation_id=expected,timeout_s=timeout_s,
        started_at=started if valid_start else None,elapsed_s=0.,motion_authorized=False,deadline_renewed=False)
    if not valid_start:
        result['reason']='frame_wait_clock_invalid'
        return result
    path=Path(directory)/'latest_proposals.json'
    last_signature=object();preview={}
    # Iteration bound also makes a faulty/non-progressing injected clock finite.
    for _ in range(math.ceil(timeout_s/poll_s)+2):
        if cancelled():result['reason']='frame_wait_cancelled';break
        try:
            try:
                stat=path.stat()
                signature=(stat.st_dev,stat.st_ino,stat.st_mtime_ns,stat.st_size)
            except FileNotFoundError:signature=None
            # Producers replace complete files atomically. Reparse only when
            # a new publication exists; polling does not renew source time.
            if signature!=last_signature:
                preview=read_json(path,{})
                last_signature=signature
        except (OSError,ValueError):
            result['reason']='preview_mailbox_unreadable';break
        now=clock()
        if type(now) not in (int,float) or not math.isfinite(now) or now < last:
            result['reason']='frame_wait_clock_invalid';break
        result['elapsed_s']=now-started
        if now-started>=timeout_s:break
        last=now
        processed=preview.get('processed_frame')
        if (isinstance(processed,dict) and processed.get('rejected') is True
                and type(processed.get('frame_id')) is int and processed['frame_id']==frame_id
                and processed.get('clock_id')==clock_id and processed.get('calibration_id')==calibration
                and type(processed.get('capture_timestamp')) in (int,float)
                and processed['capture_timestamp']==stamp):
            result.update(reason='current_frame_rejected_by_producer',
                producer_reason=str(processed.get('reason'))[:256])
            break
        if (preview.get('validation_observation_id')==expected
                and preview.get('clock_id')==clock_id
                and preview.get('calibration_id')==calibration
                and preview.get('capture_timestamp')==stamp):
            result.update(ready=True,reason='exact_current_frame_processed',
                processed_age_s=now-stamp)
            break
        sleep(min(poll_s,timeout_s-(now-started)))
    return result
