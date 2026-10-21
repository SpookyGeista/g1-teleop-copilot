"""Immutable simulator audit rows, without retaining a growing cyclic-GC graph.

This is a diagnostic log, never a control mailbox or source of permission.
Memory still grows with serialized run length. Rows are bounded by the runtime.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import tempfile


class InputTrace:
    """Frozen, per-tick read evidence for the asynchronous XR path.

    This log grants no authority and never supplies inputs to a controller.
    Identity hashes describe decoded JSON, not the mailbox's byte formatting.
    Bounds keep a malformed input from producing an unbounded audit graph.
    """
    def __init__(self, clock_id):
        self.clock_id = clock_id
        self._packets = {}
        self._entries = []
        self._bytes = 0

    def capture(self, operator, intent, *, phase, captured_at, consumer=None, substep_index=None):
        if phase not in ('session_step', 'handoff_recheck', 'actor_interlock'):
            raise ValueError('Unknown input audit phase')
        if (type(captured_at) not in (int, float) or not math.isfinite(captured_at)
                or (self._entries and captured_at < self._entries[-1]['captured_at'])):
            raise ValueError('Input audit clock invalid or regressed')
        if phase == 'actor_interlock':
            if consumer not in ('actor_write', 'physics_substep'):
                raise ValueError('Input audit consumer required')
            if consumer == 'physics_substep' and (type(substep_index) is not int or not 0 <= substep_index < 5):
                raise ValueError('Input audit substep invalid')
        elif consumer is not None or substep_index is not None:
            raise ValueError('Unexpected input audit consumer')
        if consumer != 'physics_substep' and substep_index is not None:
            raise ValueError('Unexpected input audit substep')
        if len(self._entries) >= 16:
            raise ValueError('Input audit read count exceeded')
        pending = {}
        keys = []
        for packet in (operator, intent):
            if not isinstance(packet, dict):
                raise ValueError('Input audit requires decoded mailbox objects')
            encoded = json.dumps(packet, sort_keys=True, separators=(',', ':'),
                                 ensure_ascii=True, allow_nan=False)
            if len(encoded) > 2 * 1024 * 1024:
                raise ValueError('Input audit packet exceeds 2 MiB')
            key = hashlib.sha256(encoded.encode('ascii')).hexdigest()
            keys.append(key)
            if key not in self._packets:
                pending[key] = encoded
        added = sum(len(value) for value in pending.values())
        if self._bytes + added > 8 * 1024 * 1024:
            raise ValueError('Input audit tick exceeds 8 MiB')
        self._packets.update(pending)
        self._bytes += added
        self._entries.append(dict(phase=phase, captured_at=captured_at,
            consumer=consumer, substep_index=substep_index,
            operator_sha256=keys[0], intent_sha256=keys[1]))

    def snapshot(self):
        return dict(schema='punktnull.sim_input_trace.v1', clock_id=self.clock_id,
            scope='Observed decoded inputs; diagnostic only, no command authority',
            entries=[dict(entry) for entry in self._entries],
            packets={key: json.loads(value) for key, value in self._packets.items()})


class FrozenAuditRows:
    def __init__(self):
        self._encoded = []
        self.commands_consumed = 0
        self.max_contact_force_n = None

    def __len__(self):
        return len(self._encoded)

    def append(self, row):
        # Serialization must succeed before either the log or counters change.
        # The stored text has no aliases to mutable controller/status objects.
        encoded = json.dumps(row, allow_nan=False, separators=(',', ':'))
        consumed = row['command_consumed']
        force = row['contact_force_n']
        maximum = force if self.max_contact_force_n is None else max(self.max_contact_force_n, force)
        self._encoded.append(encoded)
        self.commands_consumed += bool(consumed)
        self.max_contact_force_n = maximum

    def write(self, path):
        path = Path(path)
        descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
        try:
            with os.fdopen(descriptor, 'w') as stream:
                stream.write('[')
                for index, encoded in enumerate(self._encoded):
                    if index:
                        stream.write(',')
                    stream.write(encoded)
                stream.write(']')
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
