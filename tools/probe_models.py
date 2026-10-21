#!/usr/bin/env python3
"""Run the actual pinned models on recorded Isaac RGB-D; preserve evidence."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from teleop_assist.learned_perception import LearnedCabinetPerception, LearnedPerceptionConfig
from teleop_assist.model_registry import DEFAULT_MODEL_ROOT, MODELS, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', type=Path, default=ROOT/'runs/assist_live_recording_003')
    p.add_argument('--output', type=Path, default=ROOT/'runs/learned_model_probe')
    p.add_argument('--model-root', type=Path, default=DEFAULT_MODEL_ROOT)
    p.add_argument('--config', type=Path, help='Explicit domain prompt/threshold JSON')
    p.add_argument('--device', default='cuda')
    p.add_argument('--frames', type=int, default=3)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--detector-model', choices=('grounding_dino', 'grounding_dino_base'), default=None)
    p.add_argument('--prompts', nargs='+', default=None)
    args = p.parse_args()
    if args.frames < 1 or args.stride < 1:
        p.error('frames and stride must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'report.json').exists():
        p.error('Use a new output directory to preserve previous probe evidence')
    paths = sorted((args.recording/'rgbd').glob('*.npz'))[::args.stride][:args.frames]
    if not paths:
        p.error('No RGB-D recordings found')
    cfg = json.loads(args.config.read_text()) if args.config else {}
    cfg.update(model_root=str(args.model_root.resolve()), device=args.device)
    if args.threshold is not None:
        cfg['box_threshold'] = args.threshold
    if args.detector_model is not None:
        cfg['detector_model'] = args.detector_model
    if args.prompts:
        cfg['prompts'] = tuple(args.prompts)
    loaded = time.perf_counter()
    pipeline = LearnedCabinetPerception(LearnedPerceptionConfig(**cfg))
    load_s = time.perf_counter()-loaded
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            rgb = a['rgb']
            result = pipeline.process(rgb, a['depth'], a['intrinsics'], a['T_world_camera'],
                                      int(a['frame_id']), float(a['timestamp']))
        stem = path.stem
        (args.output/f'{stem}.json').write_text(json.dumps(result, indent=2)+'\n')
        np.savez_compressed(args.output/f'{stem}_masks.npz', **pipeline.last_masks)
        display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for index, target in enumerate(result['targets']):
            color = [(70, 220, 240), (240, 210, 70), (220, 70, 200), (100, 240, 90)][index % 4]
            mask = pipeline.last_masks[target['id']]
            display[mask] = (.60*display[mask]+.40*np.array(color)).astype(np.uint8)
            x1, y1, x2, y2 = target['bbox_px']
            cv2.rectangle(display, (x1,y1), (x2,y2), color, 1)
            cv2.putText(display, f"{target['id']} {target['label']} {target['detection_confidence']:.2f}",
                        (max(0, x1), max(12, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .36, color, 1, cv2.LINE_AA)
            cv2.drawMarker(display, tuple(target['surface_pixel_px']), (255,255,255), markerSize=7)
        cv2.imwrite(str(args.output/f'{stem}_overlay.jpg'), display)
        records.append({'file':str(path.resolve()), 'sha256':sha256(path), 'frame_id':result['diagnostics']['frame_id'],
                        'target_count':len(result['targets']), 'labels':[x['label'] for x in result['targets']],
                        'diagnostics':result['diagnostics']})
        print(json.dumps({'frame':stem, 'targets':records[-1]['labels'],
                          'seconds':result['diagnostics']['total_seconds']}), flush=True)
    import torch, transformers
    report = {'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),
              'kind':'real_pretrained_inference_on_recorded_Isaac_RGBD',
              'inference_completed':True, 'nonempty_targets':any(r['target_count'] for r in records),
              'models':{key:MODELS[key] for key in (pipeline.config.detector_model,'sam2')}, 'model_load_seconds':load_s,
              'python':platform.python_version(), 'torch':torch.__version__,
              'transformers':transformers.__version__, 'device':str(pipeline.device),
              'gpu':torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
              'frames':records,
              'limits':['Not a semantic accuracy benchmark; no human labels scored.',
                        'No headset or live robot was involved.',
                        'Surface boxes do not establish grasp feasibility or mechanical action axes.',
                        'Occluded and unrecognized objects are omitted, never replaced by scene truth.']}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'MODEL_PROBE_REPORT {args.output.resolve()/"report.json"}')


if __name__ == '__main__':
    main()
