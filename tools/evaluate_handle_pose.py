#!/usr/bin/env python3
"""Offline error against measured PhysX drawer state; never imported by control.

Scoped to the interaction_v2 handle, whose CAD frame starts axis-aligned and
translates along world X. Source poses are deduplicated by acquisition time.
Exact recorded capture timestamps map pose acquisition time onto the simulation
clock. Wall-clock interpolation is invalid while rendering pauses PhysX.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if np.isfinite(value) else None


def _capture_clock(recording, frames):
    path = recording / 'captures.jsonl'
    rows = _rows(path) if path.exists() else frames
    source = path.name if path.exists() else 'frames.jsonl'
    mapping, ambiguous = {}, set()
    invalid = 0
    for row in rows:
        # sensors supports separately timestamped wrist acquisitions in older
        # processed-frame archives. New captures normally share one clock pair.
        entries = [row, *row.get('sensors', {}).values()]
        for entry in entries:
            timestamp = _number(entry.get('timestamp'))
            sim_time = _number(entry.get('sim_time'))
            if timestamp is None or sim_time is None:
                invalid += 1
                continue
            if timestamp in mapping and mapping[timestamp] != sim_time:
                ambiguous.add(timestamp)
            mapping[timestamp] = sim_time
    for timestamp in ambiguous:
        del mapping[timestamp]
    return mapping, ambiguous, {'source': source, 'distinct_unambiguous_timestamps': len(mapping),
                               'ambiguous_timestamps': len(ambiguous), 'invalid_clock_entries': invalid,
                               'matching': 'exact recorded timestamp; no nearest or wall-clock fallback'}


def _reference_clock(states):
    reference = {}
    invalid = duplicates = 0
    for row in states:
        robot = row.get('robot', {})
        sim_time = _number(robot.get('sim_time'))
        position = _number(robot.get('fixture_joint_states', {}).get('PullHandle', {}).get('position'))
        if sim_time is None or position is None:
            invalid += 1
            continue
        if sim_time in reference:
            if reference[sim_time] != position:
                raise ValueError(f'Conflicting PhysX positions at sim_time={sim_time}')
            duplicates += 1
        reference[sim_time] = position
    if not reference:
        raise ValueError('No recorded PhysX drawer states with valid robot.sim_time')
    times = np.asarray(sorted(reference))
    return times, np.asarray([reference[t] for t in times]), {
        'valid_distinct_sim_times': len(times), 'invalid_rows': invalid, 'duplicate_rows': duplicates}


def evaluate(recording):
    recording = Path(recording)
    truth = json.loads((recording / 'evaluation_truth.json').read_text())
    if truth['fixture'] != 'fixed_pelvis_g1_interaction_v2':
        raise ValueError('Only interaction_v2 has the declared CAD-frame convention')
    fixture = next(j for j in truth['fixture_joints'] if j['id'] == 'PullHandle')
    states = _rows(recording / 'control.jsonl')
    frames = _rows(recording / 'frames.jsonl')
    times, positions, reference_info = _reference_clock(states)
    capture_times, ambiguous_times, capture_info = _capture_clock(recording, frames)
    offset = np.array([-.056, 0., -.05])
    samples = []
    seen = set()
    skips = Counter()
    candidates = duplicates = 0
    for frame in frames:
        for target in frame['targets']:
            if not target.get('pose_valid') or target.get('cad_asset_id') != 'PullHandleGrasp':
                continue
            candidates += 1
            timestamp = _number(target.get('pose_timestamp'))
            if timestamp is None:
                skips['missing_or_invalid_pose_timestamp'] += 1
                continue
            if timestamp in seen:
                duplicates += 1
                continue
            seen.add(timestamp)
            if timestamp in ambiguous_times:
                skips['ambiguous_capture_timestamp'] += 1
                continue
            if timestamp not in capture_times:
                skips['missing_capture_timestamp'] += 1
                continue
            sim_time = capture_times[timestamp]
            if sim_time < times[0] or sim_time > times[-1]:
                skips['capture_sim_time_outside_reference'] += 1
                continue
            try:
                pose = np.asarray(target['T_world_object'], dtype=float)
            except (KeyError, TypeError, ValueError):
                pose = np.empty(0)
            if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                    or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
                    or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)
                    or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-3)):
                skips['invalid_object_pose'] += 1
                continue
            joint = float(np.interp(sim_time, times, positions))
            upper = int(np.searchsorted(times, sim_time))
            bracket_gap = 0. if times[upper] == sim_time else float(times[upper] - times[upper - 1])
            center = np.array(fixture['anchor_world']) + np.array(fixture['axis_world']) * joint
            samples.append({
                'timestamp': timestamp, 'capture_sim_time': sim_time, 'reference_bracket_gap_sim_s': bracket_gap,
                'center_error_m': float(np.linalg.norm(pose[:3, 3] - center)),
                'tool_point_error_m': float(np.linalg.norm(pose[:3, :3] @ offset + pose[:3, 3] - center - offset)),
                'orientation_error_deg': float(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3]) - 1) / 2, -1, 1)))),
            })
    report = {'recording': str(recording.resolve()), 'evaluation_only': True,
              'status': 'evaluated' if samples else 'no_evaluable_pose_samples',
              'distinct_pose_samples': len(samples), 'reference': 'independent recorded PhysX drawer state',
              'scope': 'Known axis-aligned interaction_v2 CAD handle; synthetic sensors, no real-camera accuracy claim',
              'reference_interpolation': 'exact pose acquisition timestamp -> capture sim_time; linear PhysX position interpolation at robot.sim_time; no extrapolation',
              'scope_reference_interpolation_simclock': True,
              'reference_clock': 'simulation_seconds',
              'capture_clock_mapping': capture_info, 'reference_clock_records': reference_info,
              'candidate_pose_observations': candidates,
              'duplicate_pose_observations': duplicates, 'distinct_pose_timestamps': len(seen),
              'skipped_pose_observations': sum(skips.values()), 'skip_counts': dict(skips),
              'skip_count_scope': 'After timestamp deduplication; observations without a valid timestamp are counted individually',
              'max_reference_sample_gap_s': float(np.diff(times).max()) if len(times) > 1 else 0.,
              'max_reference_sample_gap_sim_s': float(np.diff(times).max()) if len(times) > 1 else 0.,
              'max_evaluated_reference_bracket_gap_sim_s': max((s['reference_bracket_gap_sim_s'] for s in samples), default=None)}
    for key in ('center_error_m', 'tool_point_error_m', 'orientation_error_deg'):
        values = np.array([s[key] for s in samples])
        report[key] = ({'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)), 'max': float(values.max())}
                       if len(values) else None)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = evaluate(args.recording)
    payload = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)
