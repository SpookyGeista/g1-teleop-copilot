#!/usr/bin/env python3
"""A/B real-recording replay of sequential versus direct semantic-mask rebase."""
import argparse,copy,importlib.util,json,shutil,sys,time
from concurrent.futures import Future
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from teleop_assist.model_bridge import ModelProcess
from teleop_assist.perception_runtime import AsyncPerception
from teleop_assist.model_registry import sha256


class NoInference:
    def process_frame(self,*args):raise AssertionError('Replay cannot schedule extra inference')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,default=ROOT/'runs/vision_grasp_acceptance_001')
    parser.add_argument('--baseline-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():parser.error('Use new output directory')
    baseline_path=args.output/'before_runtime.py';shutil.copyfile(args.baseline_source,baseline_path)
    spec=importlib.util.spec_from_file_location('teleop_assist._before_rebase',baseline_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows={r['frame_id']:r for r in map(json.loads,(args.recording/'frames.jsonl').read_text().splitlines())}
    bundles={}
    for number in range(47,62):
        bundle={'frames':{}}
        for camera,folder in (('head','rgbd'),('right_wrist','sensors/right_wrist'),('left_wrist','sensors/left_wrist')):
            with np.load(args.recording/folder/f'{number:06d}.npz') as d:frame={k:d[k].copy() for k in d.files}
            frame['frame_id']=int(frame['frame_id']);frame['timestamp']=float(frame['timestamp']);frame['camera_id']=camera
            bundle['frames'][camera]=frame
        bundles[number]=bundle
    model=ModelProcess(**json.loads((ROOT/'configs/perception.interaction.json').read_text()))
    measured={};selected=None
    try:
        for number in (47,50,53,56):
            bundle=bundles[number];result=model.process_frame(bundle['frames']['head'],bundle,selected)
            if selected is None:selected=next(t['id'] for t in result['targets'] if t['label']=='handle')
            measured[number]=result
            arrays={**result['_masks']}
            for camera,obs in result['_camera_observations'].items():
                arrays.update({camera+'__'+k:v for k,v in obs['masks'].items()})
            np.savez(args.output/f'{number:06d}_masks.npz',**arrays)
    finally:model.close()
    print('GPU_RELEASED',flush=True)
    samples=[]
    for repeat in range(5):
        modes=[('before',module.AsyncPerception),('direct_rebase',AsyncPerception)]
        if repeat%2:modes.reverse()
        for name,cls in modes:
            runtime=cls(NoInference(),semantic_interval=float('inf'))
            runtime.selected_target_id=selected
            runtime._initialize(bundles[47]['frames']['head'],copy.deepcopy(measured[47]))
            runtime.last_submit=time.monotonic()
            try:
                for number in range(48,62):
                    source={49:47,53:50,56:53,59:56}.get(number)
                    if source is not None:
                        runtime.source=bundles[source];runtime.future=Future()
                        runtime.future.set_result(copy.deepcopy(measured[source]))
                    start=time.perf_counter();out=runtime.process(bundles[number],selected);elapsed=time.perf_counter()-start
                    cad=rows[number]['diagnostics']['cad_pose']['result'];cad_source=rows[cad['frame_id']]
                    actual=next((t for t in cad_source['targets'] if t['id']==selected),None)
                    if actual and cad.get('accepted'):
                        runtime.confirm_cad_identity({'kind':'foundationpose_rgbd','accepted':True,'target_id':selected,
                            'timestamp':cad['timestamp'],'frame_id':cad['frame_id'],'camera_id':'head',
                            'center_world':actual['center_world'],'quality':cad['quality'],
                            'pose_position_sigma_m':cad['pose_position_sigma_m'],'pose_orientation_sigma_rad':cad['pose_orientation_sigma_rad']})
                    target=next((t for t in out['targets'] if t['id']==selected),None)
                    samples.append({'repeat':repeat,'mode':name,'frame_id':number,'rebase':source is not None,
                        'milliseconds':elapsed*1000,'selected_visible':target is not None,
                        'selected_center_world':target['center_world'] if target else None,
                        'target_ids':[t['id'] for t in out['targets']]})
            finally:runtime.close()
    summary={}
    for name in ('before','direct_rebase'):
        rows_mode=[r for r in samples if r['mode']==name];rebases=[r['milliseconds'] for r in rows_mode if r['rebase']]
        summary[name]={'median_all_ms':float(np.median([r['milliseconds'] for r in rows_mode])),
            'median_rebase_ms':float(np.median(rebases)),'max_rebase_ms':max(rebases),
            'selected_visible_frames':sum(r['selected_visible'] for r in rows_mode),'frames':len(rows_mode)}
    report={'kind':'Actual learned keyframes and recorded RGBD, CPU rebase A/B; archived CAD evidence replayed',
        'recording':str(args.recording),'baseline_source_sha256':sha256(baseline_path),
        'current_source_sha256':sha256(ROOT/'teleop_assist/perception_runtime.py'),
        'flow_depth_motion_time_gates_changed':False,'summary':summary,'samples':samples}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
