#!/usr/bin/env python3
"""Offline physical target-center audit; never imported by runtime control.

Simulator truth is consumed only after a completed recording. The independent
metric separates alignment to a detector's chosen surface point from alignment
to the modeled physical component's nominal front-face center.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

TRIALS = ('baseline_raw', 'eye_off_assisted', 'eye_on_no_device', 'eye_joystick')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def records(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty measurements required')
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'maximum': float(values.max()),
            'last': float(values[-1])}


def evaluate(run_dir):
    run = Path(run_dir).resolve()
    required = ('acceptance.json', 'evaluation_truth.json', 'control.jsonl', 'result.json', 'initial_state.json')
    for name in required:
        if not (run/name).is_file():
            raise FileNotFoundError(f'Recording input missing: {run/name}')
    acceptance = json.loads((run/'acceptance.json').read_text())
    truth = json.loads((run/'evaluation_truth.json').read_text())
    result = json.loads((run/'result.json').read_text())
    initial = json.loads((run/'initial_state.json').read_text())
    control = list(records(run/'control.jsonl'))
    if not acceptance.get('complete') or result.get('status') != 'completed':
        raise ValueError('Only completed recordings can receive a final geometric audit')
    if truth.get('evaluation_only') is not True:
        raise ValueError('Ground truth must be explicitly separated as evaluation-only')
    chosen = [e for e in acceptance['events'] if e.get('event') == 'perceived_goal_selected']
    if len(chosen) != 1:
        raise ValueError('This evaluator requires one unambiguous initially selected work target')
    perceived_center = np.asarray(chosen[0]['center'], float)
    perceived_goal = np.asarray(chosen[0]['approach'], float)
    standoff = float(initial['settings']['standoff'])
    perceived_normal = (perceived_goal-perceived_center)/standoff
    perceived_normal /= np.linalg.norm(perceived_normal)

    candidates = sorted(((float(np.linalg.norm(np.asarray(t['surface_position'])-perceived_center)), t)
                         for t in truth['targets']), key=lambda pair: pair[0])
    match_distance, target = candidates[0]
    separation = candidates[1][0]-match_distance if len(candidates) > 1 else None
    if match_distance > .035 or (separation is not None and separation < .035):
        raise ValueError('Perception-to-truth target association is too distant or ambiguous')
    if target.get('dynamic'):
        raise ValueError('Dynamic target evaluation requires a timestamped ground-truth trajectory')
    true_center = np.asarray(target['surface_position'], float)
    true_normal = np.asarray(target['surface_normal'], float)
    true_normal /= np.linalg.norm(true_normal)
    true_goal = true_center+standoff*true_normal
    target_geometry = np.asarray(target['size'], float)

    phases = [e for e in acceptance['events'] if e.get('event') == 'acceptance_phase']
    windows = {}
    for i, event in enumerate(phases):
        name = event['phase']
        if name in TRIALS:
            if i+1 >= len(phases):
                raise ValueError('Missing end boundary for a motion trial')
            windows[name] = (float(event['sim_time']), float(phases[i+1]['sim_time']))
    if set(windows) != set(TRIALS):
        raise ValueError('All four comparison trials must be present')
    reports = {}
    for name, (begin, end) in windows.items():
        all_samples = [r for r in control if begin <= float(r['robot']['sim_time']) < end]
        final = [r for r in all_samples if float(r['robot']['sim_time']) >= end-1.]
        if len(final) < 10:
            raise ValueError(f'Too few recorded physical samples in final second of {name}')
        ee = np.asarray([r['robot']['ee_position'] for r in final], dtype=float)
        to_true = ee-true_center
        true_axial = to_true@true_normal
        true_lateral = np.linalg.norm(to_true-true_axial[:, None]*true_normal, axis=1)
        to_perceived = ee-perceived_goal
        perceived_lateral = np.linalg.norm(to_perceived-(to_perceived@perceived_normal)[:, None]*perceived_normal, axis=1)
        true_distance = np.linalg.norm(ee-true_goal, axis=1)
        valid = all(r['robot'].get('valid', False) for r in all_samples)
        limit = max(float(r['robot'].get('joint_limit_violation', 0.)) for r in all_samples)
        reasons = sorted(set(r.get('assist', {}).get('reason', '') for r in all_samples if r.get('reference', {}).get('hold')))
        lateral_statistics = stats(true_lateral)
        reports[name] = {
            'sim_time_window_s': [begin, end], 'evaluation_window_s': [end-1., end],
            'recorded_trial_samples': len(all_samples), 'recorded_final_second_samples': len(final),
            'true_nominal_front_center_lateral_error_m': lateral_statistics,
            'true_approach_point_3d_error_m': stats(true_distance),
            'true_surface_normal_standoff_m': stats(true_axial),
            'perceived_initial_approach_lateral_error_m': stats(perceived_lateral),
            'mean_measured_ee_world': ee.mean(axis=0).tolist(),
            'all_measured_states_valid': bool(valid),
            'max_joint_limit_violation_rad': float(limit), 'hold_reasons': reasons,
            'within_12mm_true_lateral_alignment': bool(lateral_statistics['mean'] < .012),
            'meaning': 'Last simulated second of recorded measured PhysX tool point; no commanded positions in metric',
        }
    baseline = reports['baseline_raw']['true_nominal_front_center_lateral_error_m']['mean']
    for name in TRIALS[1:]:
        error = reports[name]['true_nominal_front_center_lateral_error_m']['mean']
        reports[name]['true_lateral_error_reduction_vs_baseline'] = float(1.-error/baseline)

    sim_times = np.asarray([float(r['robot']['sim_time']) for r in control])
    wall_times = np.asarray([float(r['timestamp']) for r in control])
    if not np.all(np.diff(sim_times) > 0.) or not np.all(np.diff(wall_times) > 0.):
        raise ValueError('Recording timestamps must increase strictly')
    wall_span = float(wall_times[-1]-wall_times[0])
    sim_span = float(sim_times[-1]-sim_times[0])
    frame_timing = {}
    if (run/'frames.jsonl').is_file():
        frames = list(records(run/'frames.jsonl'))
        intervals = np.diff([float(f['timestamp']) for f in frames])
        if len(intervals):
            frame_timing = {'logged_frames': len(frames),
                            'wall_interval_seconds': stats(intervals),
                            'mean_wall_camera_hz': float(1./intervals.mean())}
    prior_checks = acceptance.get('checks', {})
    failed_checks = [k for k, value in prior_checks.items() if not value.get('passed', False)]
    normal_error_deg = float(np.degrees(np.arccos(np.clip(perceived_normal@true_normal, -1., 1.))))
    perceived_center_error = perceived_center-true_center
    source_files = {name: {'sha256': digest(run/name), 'bytes': (run/name).stat().st_size} for name in required}
    return {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'recording': str(run), 'evaluator': str(Path(__file__).resolve()),
        'evaluator_sha256': digest(__file__), 'source_files': source_files,
        'ground_truth_usage': 'OFFLINE EVALUATION ONLY; never sent to detector, target selector or arm controller',
        'association': {
            'perceived_initial_target_id': chosen[0]['target_id'], 'physical_component_id': target['id'],
            'component_kind': target['kind'], 'component_dynamic': False,
            'method': 'Nearest nominal front-face center, checked distance and separation after completed recording',
            'association_distance_m': float(match_distance), 'next_candidate_distance_margin_m': separation,
            'physical_component_dimensions_m': target_geometry.tolist(),
            'true_front_center_world': true_center.tolist(), 'true_surface_normal_world': true_normal.tolist(),
            'perceived_initial_front_point_world': perceived_center.tolist(),
            'perceived_front_point_3d_error_m': float(np.linalg.norm(perceived_center_error)),
            'perceived_front_point_lateral_error_m': float(np.linalg.norm(perceived_center_error-true_normal*float(perceived_center_error@true_normal))),
            'perceived_surface_normal_error_deg': normal_error_deg,
            'configured_approach_standoff_m': standoff,
            'true_approach_world': true_goal.tolist(), 'perceived_initial_approach_world': perceived_goal.tolist(),
        },
        'trials': reports,
        'timing': {
            'control_records': len(control), 'recorded_sim_span_s': sim_span,
            'recorded_wall_span_s': wall_span, 'recorded_simulation_to_wall_time_ratio': sim_span/wall_span,
            'real_time_supported_by_this_recording': bool(sim_span/wall_span >= .95),
            'reported_total_wall_seconds': result.get('wall_seconds'),
            'reported_total_sim_seconds_including_backend_warmup': result.get('sim_seconds'),
            'render_median_ms': result.get('render_median_ms'),
            'perception_median_ms': result.get('perception_median_ms'),
            'camera_frames_skipped_busy': result.get('camera_frames_skipped_busy'), **frame_timing,
        },
        'original_acceptance': {'passed': bool(acceptance.get('passed')), 'failed_checks': failed_checks},
        'physical_alignment_validation': {
            'all_assisted_trials_under_12mm_true_lateral_error': all(reports[n]['within_12mm_true_lateral_alignment'] for n in TRIALS[1:]),
            'all_motion_trial_states_valid': all(reports[n]['all_measured_states_valid'] for n in TRIALS),
            'scope': 'Independent physical target-center alignment metric, not a full product or safety certification',
        },
        'limitations': [
            'One nominal cabinet layout and one selected lower rotary component; no statistical hardware-generalization claim.',
            'The simulated RGB-D image is clean; real sensor noise, calibration error and headset eye accuracy are not represented.',
            f'The tool uses a configured {standoff*100:g}cm visual-approach standoff in this run; no contact repair, screw turning or force-controlled grasp is proven.',
            'Fixed pelvis and ideal static geometry differ from a balancing humanoid working at a real cabinet.',
            'A detector surface point is not necessarily the physical part center. The two errors are reported separately.',
            'Recorded control samples are decimated relative to the 200Hz physics loop; this audit does not reconstruct missing physics samples.',
            'Eye and controller inputs are synthetic; live headset end-to-end operation remains unverified.',
            'Input expiry and target loss checks test recorded software responses. Detector dropout is explicitly injected.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir/'ground_truth_validation.json'
    if output.exists():
        parser.error('Evaluation output already exists; preserve previous evidence or choose a new output')
    report = evaluate(args.run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output.resolve()),
                      'true_lateral_error_mm': {k: v['true_nominal_front_center_lateral_error_m']['mean']*1000 for k, v in report['trials'].items()},
                      'recorded_sim_to_wall_ratio': report['timing']['recorded_simulation_to_wall_time_ratio']}, allow_nan=False))


if __name__ == '__main__':
    main()
