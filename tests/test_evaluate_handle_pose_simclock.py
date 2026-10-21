"""Offline ground-truth evaluation must align on acquisition simulation time."""
import copy
import json
from pathlib import Path
import importlib.util

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('evaluate_handle_pose', Path(__file__).resolve().parents[1] / 'tools/evaluate_handle_pose.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def recording(tmp_path, frames, controls, captures=None):
    (tmp_path / 'evaluation_truth.json').write_text(json.dumps({
        'fixture': 'fixed_pelvis_g1_interaction_v2',
        'fixture_joints': [{'id': 'PullHandle', 'anchor_world': [1, 0, 0], 'axis_world': [1, 0, 0]}],
    }))
    write_jsonl(tmp_path / 'frames.jsonl', frames)
    write_jsonl(tmp_path / 'control.jsonl', controls)
    if captures is not None:
        write_jsonl(tmp_path / 'captures.jsonl', captures)
    return tmp_path


def state(wall, sim, position):
    return {'timestamp': wall, 'robot': {'sim_time': sim,
            'fixture_joint_states': {'PullHandle': {'position': position}}}}


def target(timestamp, position=0.):
    pose = np.eye(4)
    pose[0, 3] = 1. + position
    return {'pose_valid': True, 'cad_asset_id': 'PullHandleGrasp',
            'pose_timestamp': timestamp, 'T_world_object': pose.tolist()}


def test_render_wall_pause_does_not_move_physx_reference(tmp_path):
    # PhysX stays at sim=0 until wall=10 while render blocks; only then does
    # motion begin. Linear wall interpolation would invent 20 mm at capture.
    controls = [state(0., 0., 0.), state(9., 0., 0.), state(11., 1., -.04)]
    frames = [{'timestamp': 10., 'sim_time': 0., 'targets': [target(10.)]},
              {'timestamp': 10.5, 'sim_time': .5, 'targets': [target(10.5, -.02)]}]
    report = tool.evaluate(recording(tmp_path, frames, controls))
    old_wall_reference = np.interp(10., [0., 9., 11.], [0., 0., -.04])
    assert old_wall_reference == pytest.approx(-.02)
    assert report['center_error_m']['max'] == pytest.approx(0.)
    assert report['tool_point_error_m']['max'] == pytest.approx(0.)
    assert report['distinct_pose_samples'] == 2
    assert report['reference_clock_records']['duplicate_rows'] == 1
    assert report['scope_reference_interpolation_simclock']
    assert report['reference_clock'] == 'simulation_seconds'
    assert report['max_reference_sample_gap_sim_s'] == 1.
    assert report['max_evaluated_reference_bracket_gap_sim_s'] == 1.
    assert report['capture_clock_mapping']['source'] == 'frames.jsonl'


def test_raw_capture_archive_maps_delayed_pose_not_processing_frame_time(tmp_path):
    # Result arrives much later, after its raw acquisition was dropped from the
    # processed frame stream. The all-raw capture archive supplies its clock.
    frames = [{'timestamp': 50., 'sim_time': 1., 'targets': [target(10.)]}]
    controls = [state(9., 0., 0.), state(50., 1., -.04)]
    captures = [{'timestamp': 10., 'sim_time': 0.}, {'timestamp': 50., 'sim_time': 1.}]
    report = tool.evaluate(recording(tmp_path, frames, controls, captures))
    assert report['center_error_m']['max'] == pytest.approx(0.)
    assert report['distinct_pose_samples'] == 1
    assert report['capture_clock_mapping']['source'] == 'captures.jsonl'


def test_missing_capture_timestamp_is_counted_without_nearest_or_legacy_fallback(tmp_path):
    absent = target(10.000001)
    missing = target(10.)
    del missing['pose_timestamp']
    frames = [{'timestamp': 10.000001, 'sim_time': 0., 'targets': [absent, copy.deepcopy(absent), missing]}]
    captures = [{'timestamp': 10., 'sim_time': 0.}]
    report = tool.evaluate(recording(tmp_path, frames, [state(9., 0., 0.), state(11., 1., -.04)], captures))
    assert report['distinct_pose_samples'] == 0
    assert report['status'] == 'no_evaluable_pose_samples'
    assert report['center_error_m'] is None
    assert report['duplicate_pose_observations'] == 1
    assert report['skipped_pose_observations'] == 2
    assert report['skip_counts'] == {'missing_capture_timestamp': 1, 'missing_or_invalid_pose_timestamp': 1}


def test_ambiguous_capture_and_out_of_range_simclock_fail_closed(tmp_path):
    frames = [{'timestamp': 10., 'sim_time': 0., 'targets': [target(10.), target(20.), target(30.)]}]
    captures = [{'timestamp': 10., 'sim_time': 0.}, {'timestamp': 10., 'sim_time': .5},
                {'timestamp': 20., 'sim_time': -.1}, {'timestamp': 30., 'sim_time': 1.1}]
    report = tool.evaluate(recording(tmp_path, frames, [state(9., 0., 0.), state(11., 1., -.04)], captures))
    assert report['distinct_pose_samples'] == 0
    assert report['skip_counts'] == {'ambiguous_capture_timestamp': 1, 'capture_sim_time_outside_reference': 2}
    assert report['capture_clock_mapping']['ambiguous_timestamps'] == 1


def test_duplicate_poses_count_once_and_invalid_pose_is_not_an_accuracy_sample(tmp_path):
    invalid = target(11., -.04)
    invalid['T_world_object'][0][0] = 2.
    frames = [{'timestamp': 10., 'sim_time': 0., 'targets': [target(10.), target(10.)]},
              {'timestamp': 11., 'sim_time': 1., 'targets': [invalid]}]
    report = tool.evaluate(recording(tmp_path, frames, [state(10., 0., 0.), state(11., 1., -.04)]))
    assert report['distinct_pose_samples'] == 1
    assert report['duplicate_pose_observations'] == 1
    assert report['skip_counts'] == {'invalid_object_pose': 1}
    assert report['candidate_pose_observations'] == (report['distinct_pose_samples']
            + report['duplicate_pose_observations'] + report['skipped_pose_observations'])


def test_conflicting_reference_at_same_simclock_is_not_arbitrarily_selected(tmp_path):
    frames = [{'timestamp': 10., 'sim_time': 0., 'targets': [target(10.)]}]
    controls = [state(9., 0., 0.), state(10., 0., -.01)]
    with pytest.raises(ValueError, match='Conflicting PhysX positions'):
        tool.evaluate(recording(tmp_path, frames, controls))
