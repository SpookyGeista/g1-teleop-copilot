"""Stable acquisition identity during one paused-physics control transaction.

A new read is used only to prove continuity; it never extends the captured
sample's timestamp. This contract relies on a single owner of physics stepping.
Real asynchronous sensors cannot use a simulator generation as their time base.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math


def _encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _stamp(value):
    stamp = value.get('acquisition_timestamp')
    if type(stamp) not in (float, int) or not math.isfinite(stamp):
        raise ValueError('snapshot_acquisition_time_invalid')
    return stamp


@dataclass(frozen=True)
class SimSnapshotTransaction:
    state_json: str
    scene_json: str
    physics_step: int
    source_instance_id: str

    @classmethod
    def capture(cls, state, scene, *, physics_step, source_instance_id):
        if type(physics_step) is not int or physics_step < 0 or not isinstance(source_instance_id, str) or not source_instance_id:
            raise ValueError('simulator_generation_required')
        if (not isinstance(state, dict) or not isinstance(scene, dict) or
                not isinstance(state.get('clock_id'), str) or not state['clock_id'] or
                scene.get('clock_id') != state['clock_id'] or _stamp(state) != _stamp(scene)):
            raise ValueError('same_acquisition_and_clock_required')
        return cls(_encode(state), _encode(scene), physics_step, source_instance_id)

    def read(self, current_state, current_scene, *, physics_step, source_instance_id):
        if physics_step != self.physics_step or source_instance_id != self.source_instance_id:
            raise ValueError('physics_advanced_or_source_changed_after_capture')
        original = [json.loads(self.state_json), json.loads(self.scene_json)]
        for saved, current, saved_json in zip(original, (current_state, current_scene),
                                               (self.state_json, self.scene_json)):
            if not isinstance(current, dict) or _stamp(current) < _stamp(saved):
                raise ValueError('snapshot_read_time_regressed')
            # Ignore only the new read time when proving geometry continuity.
            # Return the unchanged original time, never the newer one.
            compared = dict(current, acquisition_timestamp=saved['acquisition_timestamp'])
            if _encode(compared) != saved_json:
                raise ValueError('measured_snapshot_changed_after_capture')
        return tuple(original)
