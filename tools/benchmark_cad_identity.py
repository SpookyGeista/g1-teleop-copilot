#!/usr/bin/env python3
"""CPU A/B of lazy CAD comparison on recorded poses, depth and proposal masks."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.probe_pose_latency import load
from teleop_assist.cad_identity import reconcile_current_proposals
from teleop_assist.learned_perception import mask_to_target


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=5);args=p.parse_args()
    spec=importlib.util.spec_from_file_location('teleop_assist._benchmark_cad_identity_baseline',args.baseline)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    root=ROOT/'runs/vision_grasp_acceptance_009';cases=[]
    for row in map(json.loads,(root/'frames.jsonl').read_text().splitlines()):
        result=row['diagnostics'].get('cad_pose',{}).get('result',{})
        if not result.get('accepted') or result.get('frame_id')!=row['frame_id']:continue
        selected=next((t for t in row['targets'] if t['id']=='learned_004'),None)
        if selected is None:continue
        frame=load(root/'rgbd'/f"{row['frame_id']:06d}.npz");masks={}
        for target in row['targets']:
            if target['id']=='learned_004':continue
            # Live binary masks were not archived. Filled measured bounding
            # rectangles conservatively contain those masks: proving disjoint
            # rectangles suffices for the same broad-phase rejection.
            box=target.get('segmentation_bbox_px',target['bbox_px']);x1,y1,x2,y2=np.asarray(box,int)
            mask=np.zeros(frame['depth'].shape,bool);h,w=mask.shape
            mask[max(0,y1):min(h,y2+1),max(0,x1):min(w,x2+1)]=True;masks[target['id']]=mask
        cases.append((f"recorded009_frame{row['frame_id']:03d}_bbox_supersets",frame,row['targets'],masks,
            {**result,'cad_asset_id':'PullHandleGrasp'},selected['center_world']))
    probe=ROOT/'runs/cad_identity_reconciliation_probe_001'
    recorded=json.loads((probe/'report.json').read_text());frame=load(ROOT/'runs/vision_grasp_acceptance_008/rgbd/000109.npz')
    with np.load(probe/'000109_masks.npz') as a:mask=a['proposal'].copy()
    target,reason=mask_to_target(mask,frame['depth'],frame['intrinsics'],frame['T_world_camera'],
        label='handle',detection_score=.42842158675193787,mask_score=.90052729845047)
    if target is None:raise RuntimeError(reason)
    target.update(id='learned_022',frame_id=109,timestamp=frame['timestamp'],semantic_keyframe_timestamp=870364.280820945)
    result=recorded['diagnostics']['result'];evidence={**result,'cad_asset_id':'PullHandleGrasp'}
    cases.append(('actual008_frame109_alias',frame,[target],{'learned_022':mask},evidence,recorded['targets'][0]['center_world']))
    records=[]
    for name,frame,targets,masks,evidence,center in cases:
        parameters=(frame,targets,masks,'learned_004',evidence,np.asarray(center))
        a=old.reconcile_current_proposals(*parameters);b=reconcile_current_proposals(*parameters)
        comparable={k:v for k,v in b.items() if k!='comparison_depth_evaluated'}
        if a!=comparable:raise AssertionError(f'Decision/diagnostic drift in {name}: {a} != {comparable}')
        timings={'baseline_ms':[],'optimized_ms':[]}
        for _ in range(args.repeats):
            for key,method in [('baseline_ms',old.reconcile_current_proposals),('optimized_ms',reconcile_current_proposals)]:
                started=time.perf_counter();method(*parameters);timings[key].append((time.perf_counter()-started)*1000)
        records.append({'case':name,'same_decision_and_gate_diagnostics':True,'status':b['status'],
            'comparison_depth_evaluated':b.get('comparison_depth_evaluated'),
            **{key:float(np.median(value)) for key,value in timings.items()}})
    negative=[r for r in records if r['status']=='no_duplicate']
    report={'kind':'CPU-only same-input A/B, original complete comparison versus conservative lazy projection broad phase',
        'records':records,'all_decisions_and_gate_diagnostics_identical':True,
        'negative_case_count':len(negative),'negative_median_baseline_ms':float(np.median([r['baseline_ms'] for r in negative])),
        'negative_median_optimized_ms':float(np.median([r['optimized_ms'] for r in negative])),
        'limits':['Run009 did not archive binary masks: negative cases use larger measured bbox masks as conservative upper bounds.',
                  'The positive008 case uses the saved actual DINO/SAM/flow mask from the real model replay.',
                  'CPU isolation; concurrent live scheduling must still be checked separately.']}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
