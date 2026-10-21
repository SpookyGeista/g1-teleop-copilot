#!/usr/bin/env python3
"""Actual pretrained perception → current Dex3 grasp model → original G1 IK.

Offline evaluation only. Source timestamps stay unchanged. No DDS, robot
transport or controller is opened. Review output with the live dashboard.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from teleop_assist.model_bridge import ModelProcess
from teleop_assist.copilot.process import GraspProcess
from teleop_assist.copilot.frames import load_frame, geometry_observation, robot_request, preview_uv
from teleop_assist.copilot.kinematics import isolated_validate
from teleop_assist.copilot.mailbox import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt', default='red cylinder')
    parser.add_argument('--target-id')
    parser.add_argument('--seed', type=int, default=6103)
    parser.add_argument('--num-grasps', type=int, default=256)
    parser.add_argument('--top-k', type=int, default=64)
    parser.add_argument('--grasp-python', default=str(ROOT/'third_party/GraspGenX/.venv/bin/python'))
    parser.add_argument('--grasp-repo', type=Path, default=ROOT/'third_party/GraspGenX')
    parser.add_argument('--model-python', default=str(ROOT/'.venv-models/bin/python'))
    parser.add_argument('--model-root', type=Path, default=ROOT/'models')
    parser.add_argument('--ik-python', default='/home/loya2/miniconda3/envs/to/bin/python')
    parser.add_argument('--xr-repo', default='/home/loya2/xr_teleoperate')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    frame = load_frame(args.frame)
    started = time.perf_counter()
    model = ModelProcess(python=args.model_python, prompts=[args.prompt],
                         model_root=str(args.model_root.expanduser().resolve()))
    try:
        perception = model.process_frame(frame)
    finally:
        model.close()
    masks = perception.pop('_masks')
    perception.pop('_camera_observations', None)
    atomic_json(args.output/'perception.json', perception)
    np.savez_compressed(args.output/'masks.npz', **masks)
    Image.fromarray(frame['rgb']).save(args.output/'frame.jpg', quality=94)
    targets = perception['targets']
    if args.target_id:
        targets = [target for target in targets if target['id'] == args.target_id]
    if len(targets) != 1:
        atomic_json(args.output/'report.json', {'status': 'abstained', 'reason': 'expected_exactly_one_evaluation_target',
            'targets': len(targets), 'motion_output': False})
        return
    target = targets[0]
    observation = geometry_observation(frame, target['id'], masks[target['id']])
    np.savez_compressed(args.output/'observation.npz', **observation.arrays())
    descriptor = ROOT/'teleop_assist/copilot/assets/dex3_right/config.json'
    model = GraspProcess(python=args.grasp_python, repository=str(args.grasp_repo.expanduser().resolve()),
                         descriptor=str(descriptor), num_grasps=args.num_grasps, top_k=args.top_k)
    try:
        batch = model.infer(observation, seed=args.seed)
        atomic_json(args.output/'grasp_model_manifest.json', model.manifest)
    finally:
        model.close()
    for candidate in batch['candidates']:
        candidate['preview_uv'] = preview_uv(candidate['world_canonical'], frame)
    atomic_json(args.output/'raw_proposals.json', batch)
    request = robot_request(frame)
    request['batch'] = batch
    atomic_json(args.output/'ik_request.json', request)
    batch = isolated_validate(request, repository=args.xr_repo, python_executable=args.ik_python)
    from teleop_assist.copilot.collision import GraspCollisionFilter
    collision = GraspCollisionFilter(descriptor)
    from teleop_assist.copilot.preview import HandPreview
    preview = HandPreview(collision)
    points, scene = observation.point_clouds()
    for candidate in batch['candidates']:
        candidate['observed_hand_collision'] = collision.assess_grasp(candidate['world_canonical'], points, scene)
    batch.update(valid=False, motion_static=False, recording_only=True,
                 validation_reason='Recorded endpoint preview; no fresh live scene/robot path evidence')
    counts = {'total': len(batch['candidates']), 'ik_valid': sum(c['ik_valid'] for c in batch['candidates']),
        'observed_hand_clear': sum(c['observed_hand_collision']['hand_collision_free'] for c in batch['candidates']),
        'ik_and_hand_clear': sum(c['ik_valid'] and c['observed_hand_collision']['hand_collision_free'] for c in batch['candidates']),
        'ik_and_pregrasp_clear': sum(c['ik_valid'] and c['observed_hand_collision']['pregrasp_collision_free'] for c in batch['candidates'])}
    batch['candidate_counts'] = counts
    atomic_json(args.output/'validated_proposals.json', batch)
    batch['candidates'] = sorted(batch['candidates'], key=lambda c: (
        not(c['ik_valid'] and c['observed_hand_collision']['pregrasp_collision_free']), not c['ik_valid'], -c['model_score']))[:8]
    for candidate in batch['candidates']:
        candidate['preview_hand_polygons'] = preview.project(candidate['world_canonical'], frame)
    atomic_json(args.output/'latest_proposals.json', batch)
    atomic_json(args.output/'ui_state.json', {
        'mode': 'Aufzeichnung · Vorschau', 'status': 'preview', 'reason': batch['validation_reason'],
        'objects': [dict(target_id=t['id'], label=t['label'], bbox_px=t['bbox_px']) for t in targets],
        'selected_target_id': target['id'], 'frame_id': frame['frame_id'],
        'capture_timestamp': frame['timestamp'], 'clock_id': frame['clock_id']})
    atomic_json(args.output/'report.json', {'status': 'complete', 'source_frame': str(args.frame.resolve()),
        'wall_s': time.perf_counter()-started, 'targets': len(targets), 'proposals': counts['total'],
        'original_ik_pass': counts['ik_valid'], 'observed_hand_pass': counts['observed_hand_clear'],
        'ik_and_observed_hand_pass': counts['ik_and_hand_clear'],
        'motion_output': False, 'real_robot_tested': False,
        'source_timestamps_preserved': True})
    print((args.output/'report.json').read_text())


if __name__ == '__main__':
    main()
