#!/usr/bin/env python3
"""Diagnose local-normal changes on real RGBD; relaxed gate is QA-only."""
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from teleop_assist.model_bridge import ModelProcess
from teleop_assist.foundation_pose import FoundationPoseEstimator
from teleop_assist.visual_tracker import RGBDMaskTracker
from tools.probe_pose_latency import load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-mask-only',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    root=ROOT/'runs/vision_grasp_acceptance_004';frames={i:load(root/'rgbd'/f'{i:06d}.npz') for i in range(69,92)}
    rows={r['frame_id']:r for r in map(json.loads,(root/'frames.jsonl').read_text().splitlines())}
    target=next(t for t in rows[75]['targets'] if t['label']=='handle');identity=target['id']
    if a.prepare_mask_only:
        from teleop_assist.learned_perception import LearnedCabinetPerception
        model=LearnedCabinetPerception(json.loads((ROOT/'configs/perception.interaction.json').read_text()))
        proposal=target.get('segmentation_bbox_px',target['bbox_px'])
        measured=model.segment_boxes(frames[75]['rgb'],[proposal])[0]
        np.savez(a.output/'initial_sam_mask.npz',mask=measured['mask'])
        (a.output/'initial_sam_source.json').write_text(json.dumps({'source':'actual archived learned image bbox, newly inferred SAM pixels',
            'frame_id':75,'bbox_px':proposal,'mask_confidence':measured['confidence'],'scene_truth_used':False},indent=2)+'\n')
        return
    with np.load(a.output/'initial_sam_mask.npz') as d:mask=d['mask']
    tracker=RGBDMaskTracker();tracker.initialize(frames[75],[target],{identity:mask})
    fp=FoundationPoseEstimator(ROOT/'assets/cad/handle_pose/PullHandleGrasp.obj',config={'resolve_x_half_turn_with_world_up':True},debug_dir=a.output/'debug')
    records=[]
    for i in range(75,92):
        f=frames[i];angle=0.;rejected=None
        if i>75:
            gray=cv2.cvtColor(f['rgb'],cv2.COLOR_RGB2GRAY);dt=f['timestamp']-frames[i-1]['timestamp']
            observed,warped,quality,rejected=tracker._one(f,gray,target,mask,dt)
            if rejected=='inconsistent_surface_normal':
                original=tracker.config;tracker.config=replace(original,max_normal_angle_deg=180.)
                observed,warped,quality,reason=tracker._one(f,gray,target,mask,dt);tracker.config=original
                if reason:raise RuntimeError(reason)
            elif observed is None:raise RuntimeError(rejected)
            angle=float(np.degrees(np.arccos(np.clip(np.asarray(target['normal_world'])@observed['normal_world'],-1,1))))
            target=observed;mask=warped;tracker._gray=gray
        method=fp.register if i==75 else fp.track
        pose=method(f['rgb'],f['depth'],f['intrinsics'],mask,f['T_world_camera'],timestamp=f['timestamp'])
        record={'frame_id':i,'local_normal_world':target['normal_world'],'normal_angle_deg':angle,
            'old_tracker_rejected':rejected,'pose':pose};records.append(record)
        np.savez(a.output/f'{i:06d}.npz',raw_mask=mask,filtered_mask=fp.last_mask,rendered_depth=fp.last_rendered_depth)
        overlay=f['rgb'].copy().astype(float);filtered=fp.last_mask
        overlay[filtered]=.6*overlay[filtered]+.4*np.array([0,255,0]);overlay[mask & ~filtered]=.3*overlay[mask & ~filtered]+.7*np.array([255,0,255])
        Image.fromarray(overlay.astype(np.uint8)).save(a.output/f'{i:06d}.png')
        print('FRAME',i,'normal_angle',angle,'original_rejection',rejected,'cad_accepted',pose['accepted'],'quality',pose.get('quality'),flush=True)
    (a.output/'report.json').write_text(json.dumps({'kind':'Actual SAM+RGBD+FoundationPose replay; QA-only normal gate bypass records otherwise withheld masks',
        'records':records,'runtime_normal_gate_modified':False,'scene_truth_used':False},indent=2)+'\n')


if __name__=='__main__':main()
