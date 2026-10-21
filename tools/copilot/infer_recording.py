#!/usr/bin/env python3
"""Run the production isolated grasp worker on an explicitly recorded view."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from teleop_assist.copilot.observations import GeometryObservation
from teleop_assist.copilot.process import GraspProcess
from teleop_assist.copilot.mailbox import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--rgbd', type=Path)
    source.add_argument('--observation', type=Path, help='Saved GeometryObservation; preserve every source field')
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-python', type=Path, default=ROOT / 'third_party/GraspGenX/.venv/bin/python')
    parser.add_argument('--repository', type=Path, default=ROOT / 'third_party/GraspGenX')
    parser.add_argument('--descriptor', type=Path, default=ROOT / 'teleop_assist/copilot/assets/dex3_right/config.json')
    parser.add_argument('--num-grasps', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be new')
    if not 10 <= args.top_k <= args.num_grasps <= 512 or not 1 <= args.repeats <= 10:
        parser.error('Require 10 <= top-k <= num-grasps <= 512 and 1..10 repeats')
    if args.observation:
        if args.mask or args.calibration:
            parser.error('--observation already includes its mask and calibration')
        observation = GeometryObservation.from_npz(args.observation)
        inputs = [args.observation]
    else:
        if not args.mask or not args.calibration:
            parser.error('--rgbd requires --mask and --calibration')
        calibration = json.loads(args.calibration.read_text())
        with np.load(args.rgbd, allow_pickle=False) as frame, np.load(args.mask, allow_pickle=False) as masks:
            observation = GeometryObservation('recorded_handle', f'recorded:{int(frame["frame_id"])}',
                float(frame['timestamp']), 'archived_capture_clock', calibration['calibration_id'],
                'recorded_Isaac_RGBD_and_neural_mask', frame['depth'], masks['mask'], frame['intrinsics'],
                frame['T_world_camera'], np.asarray(calibration['T_world_base']))
        inputs = [args.rgbd, args.mask, args.calibration]
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'observation.npz', **observation.arrays())
    started = time.monotonic()
    worker = GraspProcess(args.model_python, repository=str(args.repository),
                          descriptor=str(args.descriptor), num_grasps=args.num_grasps,
                          top_k=args.top_k, timeout=20.)
    try:
        results = [worker.infer(observation, seed=6103+i) for i in range(args.repeats)]
        atomic_json(args.output / 'model_manifest.json', worker.manifest)
        for i, result in enumerate(results):
            atomic_json(args.output / f'proposals_{i}.json', result)
        atomic_json(args.output / 'report.json', dict(status='inference_completed',
            wall_s=time.monotonic()-started, frame_source='recording_not_live',
            inference_s=[r['inference_s'] for r in results],
            proposals=[len(r['candidates']) for r in results],
            model_python=str(args.model_python.absolute()), repository=str(args.repository.resolve()),
            num_grasps=args.num_grasps, top_k=args.top_k,
            input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            geometry_validation_pending=True, kinematics_validation_pending=True,
            collision_validation_pending=True, real_robot=False))
        print((args.output / 'report.json').read_text())
    finally:
        worker.close()


if __name__ == '__main__':
    main()
