#!/usr/bin/env python3
"""Real pretrained inference for the Grasp008 semantic re-identification loss."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from tools.probe_pose_latency import load
from teleop_assist.pose_bridge import PoseProcess,PoseRuntime
from teleop_assist.visual_tracker import RGBDMaskTracker
from teleop_assist.cad_identity import current_cad_comparison_mask


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=ROOT/'runs/vision_grasp_acceptance_008'
    rows={r['frame_id']:r for r in map(json.loads,(root/'frames.jsonl').read_text().splitlines())}
    first=load(root/'rgbd/000108.npz');first['camera_id']='head'
    frame=load(root/'rgbd/000109.npz');frame['camera_id']='head'
    if args.prepare_only:
        from teleop_assist.learned_perception import LearnedCabinetPerception
        model=LearnedCabinetPerception(json.loads((ROOT/'configs/perception.interaction.json').read_text()))
        result=model.process(first['rgb'],first['depth'],first['intrinsics'],first['T_world_camera'],first['frame_id'],first['timestamp'])
        target=next(t for t in result['targets'] if t['label']=='handle');mask=model.last_masks[target['id']]
        # The canonical-vs-proposal IDs reproduce the recorded registry event;
        # target pixels/geometry are freshly measured by the real models.
        target['id']='learned_022';target['semantic_keyframe_timestamp']=first['timestamp']
        (args.output/'proposal.json').write_text(json.dumps(target,indent=2)+'\n')
        np.savez(args.output/'proposal_mask.npz',mask=mask)
        (args.output/'semantic_diagnostics.json').write_text(json.dumps(result['diagnostics'],indent=2)+'\n')
        return
    target=json.loads((args.output/'proposal.json').read_text())
    with np.load(args.output/'proposal_mask.npz') as a:mask=a['mask'].copy()
    tracker=RGBDMaskTracker();tracker.initialize(first,[target],{'learned_022':mask})
    tracked=tracker.track(frame)
    if not tracked['targets']:raise RuntimeError('Actual proposal RGBD tracking failed: '+str(tracked))
    proposal=tracked['targets'][0];current_mask=tracker.last_masks['learned_022']
    canonical=copy.deepcopy(target);canonical['id']='learned_004'
    canonical['semantic_keyframe_timestamp']=next(t for t in rows[108]['targets'] if t['id']=='learned_004')['semantic_keyframe_timestamp']
    runtime=PoseRuntime(PoseProcess(),interval=0.)
    try:
        runtime.update(first,[canonical],{'learned_004':mask},'learned_004')
        if runtime.future:runtime.future.result(timeout=30.)
        runtime._complete_future()
        if not runtime.accepted:raise RuntimeError('Initial actual FoundationPose registration rejected')
        targets=[proposal];start=time.monotonic()
        runtime.update(frame,targets,{'learned_022':current_mask},'learned_004')
        elapsed=time.monotonic()-start
        positive=copy.deepcopy(runtime.diagnostics);output=copy.deepcopy(targets)
        cases={}
        # Recheck the same real accepted pose/current mask, without model calls,
        # to exercise existing-canonical and deliberately ambiguous alternatives.
        existing=[copy.deepcopy(canonical),copy.deepcopy(proposal)]
        cases['existing_canonical_deduplicated']=runtime._reconcile_current_identity(frame,existing,{'learned_022':current_mask},'learned_004') and [t['id'] for t in existing]==['learned_004']
        twin={**copy.deepcopy(proposal),'id':'synthetic_second_proposal'}
        ambiguous=[copy.deepcopy(proposal),twin]
        cases['synthetic_two_matching_proposals_withheld']=not runtime._reconcile_current_identity(frame,ambiguous,
            {'learned_022':current_mask,'synthetic_second_proposal':current_mask.copy()},'learned_004')
        evidence={**runtime.accepted['result'],'cad_asset_id':'PullHandleGrasp'}
        comparison=current_cad_comparison_mask(frame,evidence)
        rgb=frame['rgb'].astype(float).copy();rgb[current_mask]=.6*rgb[current_mask]+.4*np.array([255,0,255])
        rgb[comparison]=.6*rgb[comparison]+.4*np.array([0,255,0])
        Image.fromarray(rgb.astype(np.uint8)).save(args.output/'000109_proposal_CAD_overlay.png')
        np.savez(args.output/'000109_masks.npz',proposal=current_mask,CAD_comparison=comparison)
        report={'kind':'Actual DINO/SAM108, RGBD flow109 and private pretrained CAD register108/recovery109',
            'recording':str(root),'scene_truth_used':False,'target_ids':[t['id'] for t in output],
            'current_frame_elapsed_ms':elapsed*1000,'targets':output,'diagnostics':positive,'CPU_counterexamples':cases,
            'limits':['New actual neural masks; original live binary masks were not archived.',
                      'Recorded registry IDs assigned to newly measured proposals to reproduce the ID-change event.',
                      'No concurrent Isaac load; complete live grasp remains a separate test.']}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('targets','diagnostics')},indent=2),flush=True)
        if report['target_ids']!=['learned_004'] or not all(cases.values()):raise RuntimeError('Identity reconciliation checks failed')
    finally:runtime.close()


if __name__=='__main__':main()
