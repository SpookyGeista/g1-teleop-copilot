#!/usr/bin/env python3
"""Actual private-model replay of CAD observability and grasp operating bounds."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.probe_pose_latency import load
from teleop_assist.learned_perception import mask_to_target
from teleop_assist.visual_tracker import RGBDMaskTracker
from teleop_assist.pose_bridge import PoseProcess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,default=ROOT/'runs/vision_grasp_acceptance_005')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-mask-only',action='store_true')
    p.add_argument('--start',type=int,default=105);p.add_argument('--end',type=int,default=116);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows={r['frame_id']:r for r in map(json.loads,(args.recording/'frames.jsonl').read_text().splitlines())}
    f=load(args.recording/'rgbd'/f'{args.start:06d}.npz')
    proposal=next(t for t in rows[args.start]['targets'] if t['label']=='handle')
    if args.prepare_mask_only:
        from teleop_assist.learned_perception import LearnedCabinetPerception
        model=LearnedCabinetPerception(json.loads((ROOT/'configs/perception.interaction.json').read_text()))
        box=proposal.get('segmentation_bbox_px',proposal['bbox_px'])
        measured=model.segment_boxes(f['rgb'],[box])[0]
        np.savez(args.output/'initial_sam_mask.npz',mask=measured['mask'])
        (args.output/'initial_sam_source.json').write_text(json.dumps({'source':'actual archived learned image bbox; new pretrained SAM pixels',
            'frame_id':args.start,'bbox_px':box,'mask_confidence':measured['confidence'],'scene_truth_used':False},indent=2)+'\n')
        return
    with np.load(args.output/'initial_sam_mask.npz') as a:mask=a['mask']
    target,reason=mask_to_target(mask,f['depth'],f['intrinsics'],f['T_world_camera'],label='handle',
        detection_score=proposal['detection_confidence'],mask_score=proposal['segmentation_confidence'])
    if target is None:raise RuntimeError(reason)
    identity=proposal['id'];target['id']=identity;tracker=RGBDMaskTracker();tracker.initialize(f,[target],{identity:mask})
    worker=PoseProcess();records=[]
    try:
        for number in range(args.start,args.end+1):
            f=load(args.recording/'rgbd'/f'{number:06d}.npz');tracked=tracker.track(f)
            if identity not in tracker.last_masks:raise RuntimeError(f'Lost actual RGBD observation: {tracked}')
            target=next(t for t in tracked['targets'] if t['id']==identity)
            start=time.monotonic();result=worker.estimate(f,tracker.last_masks[identity],identity,reset=number==args.start)
            roundtrip=time.monotonic()-start
            if result['accepted']:
                tracker.confirm_identity({'kind':'foundationpose_rgbd','accepted':True,'target_id':identity,
                    'timestamp':f['timestamp'],'frame_id':number,'camera_id':'head','center_world':target['center_world'],
                    'quality':result['quality'],'pose_position_sigma_m':max(.012,result['pose_position_sigma_m']),
                    'pose_orientation_sigma_rad':result['pose_orientation_sigma_rad'],
                    'T_world_object':result['T_world_object'],'cad_asset_id':'PullHandleGrasp','mesh_sha256':result['mesh_sha256']})
            bound=(max(.012,result['pose_position_sigma_m'])+np.linalg.norm([-.056,0,-.05])*result['pose_orientation_sigma_rad']
                   if result.get('pose_position_sigma_m') is not None and result.get('pose_orientation_sigma_rad') is not None else None)
            old=rows[number]['diagnostics']['cad_pose']['result']
            old_bound=.012+np.linalg.norm([-.056,0,-.05])*old['pose_orientation_sigma_rad']
            record={'frame_id':number,'timestamp':f['timestamp'],'roundtrip_ms':roundtrip*1000,'result':result,
                'old_live_tool_bound_m':old_bound,'replay_tool_bound_m':bound,
                'replay_tool_within_unchanged_25mm_gate':bool(result['accepted'] and bound is not None and bound<=.025)}
            records.append(record);np.savez(args.output/f'{number:06d}_mask.npz',mask=tracker.last_masks[identity])
            print(number,result['accepted'],result.get('reason'),'sigma',result.get('pose_orientation_sigma_rad'),'tool',bound,
                  'rank',result.get('pose_observability',{}).get('scaled_rank'),'roundtrip_ms',roundtrip*1000,flush=True)
        report={'kind':'Actual isolated private FoundationPose worker on recorded RGBD and newly inferred SAM mask',
            'recording':str(args.recording),'records':records,'accepted':sum(r['result']['accepted'] for r in records),
            'within_tool_bound':sum(r['replay_tool_within_unchanged_25mm_gate'] for r in records),'count':len(records),
            'scene_truth_used':False,'probabilistically_calibrated':False,
            'limits':['Local point-plane operating bounds use a white-noise spread approximation with explicit bias and floors.',
                      'Archived live binary masks were unavailable; replay uses freshly inferred SAM pixels.',
                      'No concurrent Isaac load; full-stack acceptance remains separate.']}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:worker.close()


if __name__=='__main__':main()
