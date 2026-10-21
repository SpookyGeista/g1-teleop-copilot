#!/usr/bin/env python3
"""Actual pretrained CAD registration using saved learned masks and Isaac RGB-D."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from teleop_assist.foundation_pose import FoundationPoseEstimator
from teleop_assist.model_registry import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame',type=Path,required=True)
    p.add_argument('--detections',type=Path,required=True)
    p.add_argument('--masks',type=Path,required=True)
    p.add_argument('--catalog',type=Path,default=ROOT/'assets/cad/handle_pose/catalog.json')
    p.add_argument('--label',default='handle')
    p.add_argument('--target-id',default=None)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():
        p.error('Choose a new output directory')
    detection=json.loads(args.detections.read_text())
    candidates=[t for t in detection['targets'] if t['label']==args.label and
                (args.target_id is None or t['id']==args.target_id)]
    if len(candidates)!=1:
        p.error(f'Need exactly one observed {args.label}; got {len(candidates)}. Select --target-id explicitly.')
    target=candidates[0]
    catalog=json.loads(args.catalog.read_text())
    asset=catalog['objects'][0]
    mesh=Path(asset['cad_mesh'])
    if sha256(mesh)!=asset['cad_sha256']:
        raise RuntimeError('CAD catalog hash mismatch')
    before=time.perf_counter()
    estimator=FoundationPoseEstimator(mesh,config={'resolve_x_half_turn_with_world_up':True},debug_dir=args.output/'debug')
    load_seconds=time.perf_counter()-before
    with np.load(args.frame,allow_pickle=False) as a,np.load(args.masks,allow_pickle=False) as masks:
        rgb=a['rgb'];mask=masks[target['id']]
        result=estimator.register(rgb,a['depth'],a['intrinsics'],mask,a['T_world_camera'],asset['grasp_templates'][0])
        result.update(frame_id=int(a['frame_id']),timestamp=float(a['timestamp']),target_id=target['id'])
    render=estimator.last_rendered_depth>.001
    overlay=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    overlay[mask]=(.6*overlay[mask]+.4*np.array([50,220,80])).astype(np.uint8)
    contours,_=cv2.findContours(render.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,contours,-1,(220,80,240),1)
    cv2.imwrite(str(args.output/'cad_mask_overlay.jpg'),overlay)
    np.save(args.output/'rendered_depth.npy',estimator.last_rendered_depth)
    report={'kind':'real_FoundationPose_on_neural_mask_and_recorded_Isaac_RGBD',
            'frame':str(args.frame.resolve()),'frame_sha256':sha256(args.frame),
            'mask_file':str(args.masks.resolve()),'mask_sha256':sha256(args.masks),
            'load_seconds':load_seconds,'result':result,
            'limits':['Observed CAD fit is not a completed physical manipulation test.',
                      'The known handle X-half-turn symmetry uses an explicit world-up mounting prior.',
                      'No simulator world object pose is used by this probe.']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if result['accepted'] else 2


if __name__=='__main__':
    raise SystemExit(main())
