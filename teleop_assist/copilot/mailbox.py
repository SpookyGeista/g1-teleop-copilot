"""Small atomic local mailboxes; control messages have short monotonic leases."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import tempfile
import time

from .vr_buttons import controller_event_time_issue, has_vr_time_provenance, VR_TIME_FIELDS


def host_clock_id():
    return 'host_monotonic:' + Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, allow_nan=False, ensure_ascii=False,
                         default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)).encode()
    if len(payload) > 2 * 1024 * 1024:
        raise ValueError('Mailbox payload exceeds 2 MiB')
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path, default=None, limit=2 * 1024 * 1024):
    try:
        with Path(path).open('rb') as stream:
            payload = stream.read(limit + 1)
        if len(payload) > limit:
            raise ValueError('Mailbox too large')
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise ValueError('Mailbox must contain an object')
        return result
    except FileNotFoundError:
        return default


class ControlWriter:
    """A short, expiring hold; state changes are one-shot revisions."""
    def __init__(self, directory, clock=time.monotonic, clock_id=None, *, filename='copilot_control.json'):
        self.directory = Path(directory)
        if Path(filename).name != filename or not filename.endswith('.json'):
            raise ValueError('Control mailbox must be a JSON filename')
        self.path = self.directory/filename
        self.clock = clock
        self.clock_id = clock_id or host_clock_id()
        old = read_json(self.path, {})
        self.revision = int(old.get('revision', 0))
        self.write('release')

    def write(self, action, candidate_id=None, *, client_id=None, source_sample=None):
        if action not in ('select', 'confirm', 'hold', 'release', 'idle', 'clutch', 'start'):
            raise ValueError('Unknown command')
        if action == 'select' and (not isinstance(candidate_id, str) or not 0 < len(candidate_id) <= 256):
            raise ValueError('Choose a candidate')
        if action != 'select' and candidate_id is not None:
            raise ValueError('Candidate ID only belongs to selection')
        now = self.clock()
        stamp, deadline = now, None
        if source_sample is not None:
            if not isinstance(source_sample, dict): raise ValueError('Invalid command source sample')
            stamp, deadline = source_sample.get('acquisition_timestamp'), source_sample.get('expires_at')
            bounded = has_vr_time_provenance(source_sample)
            try:
                valid = (type(stamp) in (int,float) and type(deadline) in (int,float)
                    and math.isfinite(stamp) and math.isfinite(deadline)
                    and source_sample.get('clock_id') == self.clock_id
                    and ((controller_event_time_issue(stamp, deadline, now=now,
                        **{name: source_sample[name] for name in VR_TIME_FIELDS}) is None
                        and type(source_sample.get('event_sequence')) is int
                        and source_sample['event_sequence'] >= 0) if bounded
                        else (0 <= now-stamp < .25 and now < deadline <= stamp+.25)))
            except OverflowError: valid = False
            if not valid: raise ValueError('Source command lease expired or unknown')
        self.revision += 1
        packet = dict(revision=self.revision, capture_timestamp=stamp, clock_id=self.clock_id,
                      select_candidate_id=candidate_id if action == 'select' else None,
                      confirm=action == 'confirm', assist_held=action == 'hold', release=action == 'release',
                      clutch=action == 'clutch', start=action == 'start')
        if client_id is not None:
            packet['client_id'] = client_id
        if source_sample is not None:
            packet.update(expires_at=deadline,source_kind='webxr_controller_event',
                          source_event_sequence=source_sample.get('event_sequence'))
            if bounded: packet.update({name: source_sample[name] for name in VR_TIME_FIELDS})
        atomic_json(self.path, packet)
        return packet
