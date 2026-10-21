"""Proportional controller triggers for the two local simulated Dex3 hands.

This is a named-joint grip preset, not Unitree's landmark retargeter or a DDS
joint order. Arm calibration and original Unitree IK are independent of it.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .vr_buttons import SCHEMA, MAPPING, states_issue, number, controller_event_time_issue
from tools.copilot.xr_source import validate_operator_packet


def trigger_values(packet, *, now, clock_id):
    """Return only analog values bound to this exact, fresh controller event."""
    empty = dict(left=None, right=None)
    if validate_operator_packet(packet, now=now, clock_id=clock_id):
        return empty
    payload = packet.get('vr_buttons')
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA or payload.get('mapping') != MAPPING:
        return empty
    if payload.get('valid') is not True or states_issue(payload.get('states')):
        return empty
    for field in ('source_instance_id', 'source_event_clock_id', 'clock_id'):
        if not isinstance(packet.get(field), str) or payload.get(field) != packet[field]:
            return empty
    sequence = payload.get('event_sequence')
    sequences = packet.get('source_event_sequences')
    if type(sequence) is not int or not isinstance(sequences, dict) or sequences.get('arms') != sequence:
        return empty
    stamp, expiry = payload.get('acquisition_timestamp'), payload.get('expires_at')
    bound, receipt = payload.get('clock_uncertainty_s'), payload.get('received_at_host_monotonic')
    if controller_event_time_issue(stamp, expiry, now=now,
            clock_uncertainty_s=bound, received_at_host_monotonic=receipt):
        return empty
    if (stamp < packet['acquisition_timestamp'] or bound > packet['envelope']['clock_uncertainty_s']
            or receipt < packet['envelope']['received_at_host_monotonic'] or receipt > packet['published_at']):
        return empty
    return {side: (value if number(value) and 0 <= value <= 1 else None)
            for side in empty for value in (payload['states'][side].get('triggerValue'),)}


class Dex3TriggerControl:
    """Rate-limit each hand independently; missing input holds measured joints."""
    def __init__(self, joint_names, urdf, *, simulation_mode, speed_rad_s=2.):
        if simulation_mode is not True:
            raise ValueError('Dex3 controller triggers are available only in isolated simulation')
        if not number(speed_rad_s) or not 0 < speed_rad_s <= 2.:
            raise ValueError('Invalid simulated finger target speed')
        config = json.loads((Path(__file__).parent/'assets/dex3_right/config.json').read_text())
        limits = {joint.attrib['name']: joint.find('limit').attrib
                  for joint in ET.parse(urdf).getroot().findall('joint') if joint.find('limit') is not None}
        self.hands = {}
        for side in ('left', 'right'):
            names = [name.replace('right_', side+'_', 1) for name in config['open']]
            sign = -1. if side == 'left' else 1.
            opened = sign*np.array(list(config['open'].values()))
            closed = sign*np.array([config['close'][name] for name in config['open']])
            lower = np.array([float(limits[name]['lower']) for name in names])
            upper = np.array([float(limits[name]['upper']) for name in names])
            if any(np.any(q < lower) or np.any(q > upper) for q in (opened, closed)):
                raise ValueError('Dex3 preset violates original named joint limits')
            self.hands[side] = dict(names=names, ids=[joint_names.index(name) for name in names],
                opened=opened, closed=closed, lower=lower, upper=upper,
                speed=np.minimum(speed_rad_s, [float(limits[name]['velocity']) for name in names]), target=None)
        self.status = {}

    def update(self, measured, packet, *, enabled, now, clock_id, dt):
        if not number(dt) or not 0 < dt <= .025:
            raise ValueError('Finger control requires the actual bounded physics timestep')
        target = np.array(measured, dtype=float, copy=True)
        if target.ndim != 1 or not np.isfinite(target).all():
            raise ValueError('Invalid measured joint vector')
        values = trigger_values(packet, now=now, clock_id=clock_id) if enabled else dict(left=None,right=None)
        self.status = dict(enabled=bool(enabled), mapping='index_trigger_proportional', hands={})
        for side, hand in self.hands.items():
            q, value = target[hand['ids']], values[side]
            if value is None:
                hand['target'] = None
            else:
                goal = hand['opened'] + value*(hand['closed']-hand['opened'])
                previous = q if hand['target'] is None else hand['target']
                next_q = previous + np.clip(goal-previous, -hand['speed']*dt, hand['speed']*dt)
                # No accumulated closure trajectory behind an object/contact.
                hand['target'] = np.clip(np.clip(next_q,q-.15,q+.15),hand['lower'],hand['upper'])
                target[hand['ids']] = hand['target']
            self.status['hands'][side] = dict(trigger=value, command_applied=value is not None,
                reason='trigger' if value is not None else ('trigger_unavailable' if enabled else 'teleop_hold'))
        return target
