"""Private JSON-lines worker for the local learned perception environment."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
import traceback
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--config', default='{}')
    args = parser.parse_args()
    directory = args.directory.resolve()
    wire = sys.stdout

    def send(value):
        wire.write(json.dumps(value, allow_nan=False,
                              default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
        wire.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            from dataclasses import replace
            from teleop_assist.learned_perception import LearnedCabinetPerception, mask_to_target
            from teleop_assist.multiview_tracker import LockedTargetWristRecovery
            from teleop_assist.sensor_fusion import wrist_proposals, refine_with_wrist, lidar_support
            config = json.loads(args.config)
            pipeline = LearnedCabinetPerception(config=config or None)
            recovery = LockedTargetWristRecovery()
        send({'status':'ready', 'backend':'grounding_dino_sam2_rgbd'})
    except BaseException:
        send({'status':'error', 'error':traceback.format_exc()})
        return 1
    for line in sys.stdin:
        stem = None
        try:
            packet = json.loads(line)
            stem = packet['stem']
            if not isinstance(stem,str) or not stem.isdecimal() or len(stem)!=8:
                raise ValueError('Invalid private frame key')
            with np.load(directory/(stem+'.npz'), allow_pickle=False) as data:
                with contextlib.redirect_stdout(sys.stderr):
                    started=time.monotonic()
                    result = pipeline.process(data['rgb'],data['depth'],data['intrinsics'],
                                              data['T_world_camera'],int(data['frame_id']),
                                              float(data['timestamp']))
                    head={k:data[k] for k in ('rgb','depth','intrinsics','T_world_camera','timestamp','frame_id')}
                    head['timestamp']=float(head['timestamp']);head['frame_id']=int(head['frame_id'])
                    wrists={}
                    for name in ('right_wrist','left_wrist'):
                        if name+'_rgb' not in data:continue
                        wrist={k:data[name+'_'+k] for k in ('rgb','depth','intrinsics','T_world_camera','timestamp')}
                        wrist['timestamp']=float(wrist['timestamp']);wrist['camera_id']=name
                        wrist['frame_id']=int(data[name+'_frame_id']) if name+'_frame_id' in data else head['frame_id']
                        wrists[name]=wrist
                    sensor_status={'head':{'status':'semantic_rgbd'}}
                    source_observations=[];export_masks=dict(pipeline.last_masks)

                    def export_observation(identity,camera,frame,mask,confidence,head_target):
                        measured,reason=mask_to_target(mask,frame['depth'],frame['intrinsics'],frame['T_world_camera'],
                            label=head_target.get('semantic_phrase',head_target['label']),
                            detection_score=head_target.get('detection_confidence',.3),mask_score=confidence,
                            config=replace(pipeline.config,max_mask_fraction=.8))
                        if measured is None:return
                        measured.update(id=identity,frame_id=int(frame['frame_id']),timestamp=float(frame['timestamp']),
                                        observed=True,stale=False,observation_camera=camera)
                        key='obs__'+camera+'__'+identity
                        export_masks[key]=np.asarray(mask,dtype=bool)
                        # Camera-space target is required for tracking that view;
                        # head_target preserves the calibrated overlay anchor.
                        record={'camera_id':camera,'target_id':identity,'target':measured,
                                'head_target':dict(head_target),'mask_key':key,'timestamp':float(frame['timestamp'])}
                        source_observations[:]=[r for r in source_observations
                            if (r['camera_id'],r['target_id'])!=(camera,identity)]
                        source_observations.append(record)

                    selected=packet.get('selected_target_id')
                    # A fresh detector proposal cannot silently add disconnected
                    # neighboring objects to an already confirmed instance.
                    if selected in pipeline.last_masks:
                        target=next((t for t in result['targets'] if t['id']==selected),None)
                        if target is not None:
                            cleaned,component_report=recovery.filter_confirmed_mask(selected,head,pipeline.last_masks[selected])
                            sensor_status['confirmed_components']=component_report
                            if component_report.get('status')!='no_previous_surface':
                                measured,reason=mask_to_target(cleaned,head['depth'],head['intrinsics'],head['T_world_camera'],
                                    label=target.get('semantic_phrase',target['label']),
                                    detection_score=target['detection_confidence'],mask_score=target['segmentation_confidence'],
                                    config=pipeline.config)
                                if measured is None:
                                    result['targets']=[t for t in result['targets'] if t['id']!=selected]
                                    pipeline.last_masks.pop(selected,None);export_masks.pop(selected,None)
                                    sensor_status['confirmed_components']['geometry_rejection']=reason
                                else:
                                    target.update(measured,perception_method='confirmed_identity_head_sam2_rgbd')
                                    pipeline.last_masks[selected]=cleaned;export_masks[selected]=cleaned
                                    pipeline.synchronize_confirmed_observation(target)
                    recovery.observe(head,result['targets'],pipeline.last_masks)
                    selected=packet.get('selected_target_id')
                    if selected and not any(t['id']==selected for t in result['targets']):
                        recovered=recovery.recover(selected,head,{'head':head,**wrists},
                                                   pipeline.segment_boxes,result['targets'],segment_points=pipeline.segment_points)
                        sensor_status['confirmed_identity_recovery']=recovered['diagnostics']
                        result['targets'].extend(recovered['targets'])
                        for target in recovered['targets']:
                            pipeline.synchronize_confirmed_observation(target)
                            observation=recovered['observations'][target['id']]
                            if observation['camera_id']=='head':
                                pipeline.last_masks[target['id']]=observation['mask']
                                export_masks[target['id']]=observation['mask']
                            else:
                                export_observation(target['id'],observation['camera_id'],observation['frame'],
                                    observation['mask'],target['wrist_recovery']['mask_confidence'],target)
                    for name,wrist in wrists.items():
                        candidates=result['targets']
                        if selected:candidates=[t for t in candidates if t['id']==selected]
                        proposals=wrist_proposals(candidates,head,wrist)[:4]
                        records=[]
                        if proposals:
                            segments=pipeline.segment_boxes(wrist['rgb'],[p['bbox_px'] for p in proposals])
                            records=refine_with_wrist(result['targets'],head,wrist,proposals,segments)
                            validated={r['id'] for r in records if r['status']=='surface_refined'}
                            for proposal,segment in zip(proposals,segments):
                                if proposal['id'] in validated:
                                    target=next(t for t in result['targets'] if t['id']==proposal['id'])
                                    export_observation(target['id'],name,wrist,segment['mask'],
                                                       float(segment['confidence']),target)
                        sensor_status[name]={'status':'processed' if proposals else 'target_outside_view',
                                             'proposals':len(proposals),'refinements':records}
                    result['source_observations']=source_observations
                    scan=json.loads(str(data['lidar_json'])) if 'lidar_json' in data else None
                    sensor_status['lidar']=lidar_support(result['targets'],head,pipeline.last_masks,scan)
                    result['diagnostics']['sensors']=sensor_status
                    result['diagnostics']['inference_ms']=(time.monotonic()-started)*1000
                    result['diagnostics']['detected_count']=len(result['targets'])
                    np.savez(directory/(stem+'.masks.npz'),**export_masks)
            send({'status':'ok','stem':stem,'result':result})
        except BaseException:
            send({'status':'error','stem':stem,'error':traceback.format_exc()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
