#!/usr/bin/env python3
"""Actual async RGBD tracking / private CAD-worker handoff on a saved grasp."""
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
    def process_frame(self,*args):raise AssertionError('No unrequested semantic inference')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=ROOT/'runs/vision_grasp_acceptance_007';args.output.mkdir(parents=True,exist_ok=True)
    rows={r['frame_id']:r for r in map(json.loads,(root/'frames.jsonl').read_text().splitlines())}
    proposal=next(t for t in rows[108]['targets'] if t['label']=='handle');identity=proposal['id']
    f=load(root/'rgbd/000108.npz');f['camera_id']='head'
    source_mask=ROOT/'runs/wrist_cad_handoff_probe_001/initial_sam_mask.npz'
    with np.load(source_mask) as a:mask=a['mask'].copy()
    target,reason=mask_to_target(mask,f['depth'],f['intrinsics'],f['T_world_camera'],label='handle',
        detection_score=proposal['detection_confidence'],mask_score=proposal['segmentation_confidence'])
    if target is None:raise RuntimeError(reason)
    target.update(id=identity,timestamp=f['timestamp'],semantic_keyframe_timestamp=f['timestamp'])
    runtime=PoseRuntime(PoseProcess(),interval=0.)
    perception=AsyncPerception(NoSemantic(),semantic_interval=float('inf'));perception.selected_target_id=identity
    perception._initialize(f,{'targets':[target],'_masks':{identity:mask},'diagnostics':{}})
    perception.last_submit=time.monotonic()
    records=[]
    try:
        runtime.update(f,[target],{identity:mask},identity)
        if runtime.future is not None:runtime.future.result(timeout=30.)
        runtime._complete_future();runtime._annotate(f,[target]);perception.confirm_cad_identity(runtime.identity_confirmation)
        if not runtime.accepted:raise RuntimeError('Initial actual CAD registration rejected')
        for number in range(109,115):
            bundle={'frames':{}}
            for camera in ('head','right_wrist','left_wrist'):
                folder='rgbd' if camera=='head' else 'sensors/'+camera
                view=load(root/folder/f'{number:06d}.npz');view['camera_id']=camera;bundle['frames'][camera]=view
            f=bundle['frames']['head'];observed=perception.process(bundle,identity);targets=observed['targets']
            start=time.monotonic();runtime.update(f,targets,perception.tracker.last_masks,identity,
                observations=perception.observations,recovery_frames=bundle['frames']);elapsed=time.monotonic()-start
            confirmed=perception.confirm_cad_identity(runtime.identity_confirmation)
            target=next((t for t in targets if t['id']==identity),None)
            record={'frame_id':number,'target_present':target is not None,'target':copy.deepcopy(target),
                'confirmed_identity':confirmed,'elapsed_ms':elapsed*1000,'diagnostics':copy.deepcopy(runtime.diagnostics),
                'upstream_tracking_diagnostics':observed['diagnostics']};records.append(record)
            recovery=runtime.recovered_observation
            if recovery:
                camera=recovery['camera_id'];mask=recovery['mask'];image=recovery['frame']['rgb'].astype(float).copy()
                image[mask]=.55*image[mask]+.45*np.array([0,255,0]);Image.fromarray(image.astype(np.uint8)).save(args.output/f'{number:06d}_{camera}_overlay.png')
                np.savez(args.output/f'{number:06d}_{camera}_mask.npz',mask=mask)
            print(number,runtime.diagnostics['status'],runtime.diagnostics.get('observation_camera'),
                [(v['camera_id'],v['accepted']) for v in runtime.diagnostics['view_attempts']],
                'target',target is not None,'milliseconds',elapsed*1000,flush=True)
        report={'kind':'Actual RGBD tracker and private FoundationPose worker, head rejection to wrist same-frame alternative',
            'recording':str(root),'initial_SAM_mask':str(source_mask),'records':records,'scene_truth_used':False,
            'accepted_wrist_frames':sum(r['diagnostics']['status']=='accepted' and r['diagnostics']['observation_camera']=='right_wrist' for r in records),
            'limits':['New SAM initialization; original live binary masks were not archived.',
                      'No concurrent Isaac load; live grasp completion remains a separate acceptance test.']}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:runtime.close();perception.close()


if __name__=='__main__':main()
