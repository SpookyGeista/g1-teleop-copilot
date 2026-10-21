#!/usr/bin/env python3
"""Real private CAD-worker recovery on frames whose upstream LK track was lost."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from tools.probe_pose_latency import load
from teleop_assist.learned_perception import mask_to_target
from teleop_assist.pose_bridge import PoseProcess,PoseRuntime
from teleop_assist.perception_runtime import AsyncPerception


class NoSemantic:
    pass


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=ROOT/'runs/vision_grasp_acceptance_006';args.output.mkdir(parents=True,exist_ok=True)
    rows={r['frame_id']:r for r in map(json.loads,(root/'frames.jsonl').read_text().splitlines())}
    f=load(root/'rgbd/000110.npz');proposal=next(t for t in rows[110]['targets'] if t['label']=='handle')
    with np.load(args.output/'initial_sam_mask.npz') as a:mask=a['mask'].copy()
    target,reason=mask_to_target(mask,f['depth'],f['intrinsics'],f['T_world_camera'],label='handle',
        detection_score=proposal['detection_confidence'],mask_score=proposal['segmentation_confidence'])
    if target is None:raise RuntimeError(reason)
    identity=proposal['id'];target.update(id=identity,timestamp=f['timestamp'],semantic_keyframe_timestamp=f['timestamp'])
    runtime=PoseRuntime(PoseProcess(),interval=0.);perception=AsyncPerception(NoSemantic());perception.selected_target_id=identity
    records=[]
    try:
        runtime.update(f,[target],{identity:mask},identity)
        if runtime.future is not None:runtime.future.result(timeout=30.)
        runtime._complete_future()
        if not runtime.accepted:raise RuntimeError('Initial actual CAD registration rejected')
        initial=copy.deepcopy(runtime.diagnostics)
        for number in (111,112):
            f=load(root/'rgbd'/f'{number:06d}.npz');targets=[]
            start=time.monotonic();returned=runtime.update(f,targets,{},identity);elapsed=time.monotonic()-start
            published=next((t for t in targets if t['id']==identity),None)
            reinitialized=perception.confirm_cad_identity(runtime.identity_confirmation)
            record={'frame_id':number,'timestamp':f['timestamp'],'original_live_had_handle':any(t['label']=='handle' for t in rows[number]['targets']),
                'recovered':published is not None,'same_list_in_place':returned is targets,'reinitialized_actual_mask':reinitialized,
                'elapsed_ms':elapsed*1000,'target':copy.deepcopy(published),'diagnostics':copy.deepcopy(runtime.diagnostics)}
            records.append(record)
            if published:
                measured=runtime.recovered_observation['mask'];np.savez(args.output/f'{number:06d}_recovered_mask.npz',mask=measured)
                overlay=f['rgb'].astype(float).copy();overlay[measured]=.55*overlay[measured]+.45*np.array([0,255,0])
                Image.fromarray(overlay.astype(np.uint8)).save(args.output/f'{number:06d}_recovered_overlay.png')
            print(number,'recovered',record['recovered'],'same_frame_pose',published.get('pose_timestamp') if published else None,
                  'ms',record['elapsed_ms'],'worker_status',runtime.diagnostics.get('status'),flush=True)
        # Explicitly altered sensor input is a negative test, never a model
        # success claim on a recorded frame: zero depth must produce no target.
        invalid={**f,'frame_id':113,'timestamp':f['timestamp']+.1,'depth':np.zeros_like(f['depth'])}
        targets=[];runtime.update(invalid,targets,{},identity)
        negative={'input':'synthetically removed current depth after real frame112','withheld':not targets,
                  'diagnostics':copy.deepcopy(runtime.diagnostics)}
        report={'kind':'Actual FoundationPose worker on actual frames110–112; recorded upstream111/112 LK dropout',
            'recording':str(root),'initial':initial,'records':records,'negative_depth_test':negative,
            'scene_truth_used':False,'new_SAM_claim_for_recovery':False,'freshness_gate_s':.5,
            'limits':['Initial mask is freshly inferred SAM using an archived learned image proposal.',
                      'Recovery mask is current measured depth conditioned on prior accepted CAD, not new SAM.',
                      'No concurrent Isaac load; full-stack grasp acceptance remains separate.']}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if not all(r['recovered'] and r['reinitialized_actual_mask'] for r in records) or not negative['withheld']:
            raise RuntimeError('Current CAD recovery acceptance checks failed')
    finally:runtime.close();perception.close()


if __name__=='__main__':main()
