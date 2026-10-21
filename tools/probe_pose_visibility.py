#!/usr/bin/env python3
"""Actual FoundationPose replay with measured foreground visibility accounting."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from tools.probe_pose_latency import load
from teleop_assist.foundation_pose import FoundationPoseEstimator,rendered_sensor_consistency
from teleop_assist.learned_perception import mask_to_target
from teleop_assist.visual_tracker import RGBDMaskTracker


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--restore-recorded-prior',action='store_true');args=p.parse_args()
    root=ROOT/'runs/vision_grasp_acceptance_011';args.output.mkdir(parents=True,exist_ok=True)
    first=load(root/'rgbd/000105.npz')
    with np.load(args.output/'initial_sam_mask.npz') as a:mask=a['mask'].copy()
    target,reason=mask_to_target(mask,first['depth'],first['intrinsics'],first['T_world_camera'],
        label='handle',detection_score=.4,mask_score=.9313684105873108)
    if target is None:raise RuntimeError(reason)
    target['id']='learned_004';tracker=RGBDMaskTracker();tracker.initialize(first,[target],{'learned_004':mask})
    estimator=FoundationPoseEstimator(ROOT/'assets/cad/handle_pose/PullHandleGrasp.obj',
        config={'resolve_x_half_turn_with_world_up':True},debug_dir=args.output/'debug')
    archived=None
    if args.restore_recorded_prior:
        rows={r['frame_id']:r for r in map(json.loads,(root/'frames.jsonl').read_text().splitlines())}
        archived=rows[105]['diagnostics']['cad_pose']['result']
        if not archived['accepted'] or archived['mesh_sha256']!=__import__('teleop_assist.model_registry',fromlist=['sha256']).sha256(estimator.mesh_path):
            raise RuntimeError('Exact accepted recorded model/CAD prior required')
        world=np.asarray(archived['T_world_object']);camera=np.asarray(first['T_world_camera'])
        centered=estimator.estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
        native=estimator.torch.as_tensor(np.linalg.inv(camera)@world@np.linalg.inv(centered),
            device='cuda',dtype=estimator.torch.float32)
        estimator.estimator.pose_last=native.clone();estimator._registered=True
        estimator._last_T_world_camera=camera.copy()
        estimator._accepted_prior={'T_world_object':world.copy(),'T_world_camera':camera.copy(),
            'timestamp':first['timestamp'],'native_pose':native.clone()}
    records=[]
    for number in (105,106,107):
        f=load(root/'rgbd'/f'{number:06d}.npz');tracked=tracker.track(f);camera='head'
        observed=tracker.last_masks.get('learned_004');start=time.monotonic()
        if number==105 and archived is not None:
            records.append({'frame_id':105,'camera_id':'head','result':archived,
                'replay_source':'restored previously accepted archived model pose, no new measurement claim'})
            continue
        if number==105:
            result=estimator.register(f['rgb'],f['depth'],f['intrinsics'],observed,f['T_world_camera'],timestamp=f['timestamp'])
        elif observed is not None:
            result=estimator.track(f['rgb'],f['depth'],f['intrinsics'],observed,f['T_world_camera'],timestamp=f['timestamp'])
        else:result=estimator.track_visible_depth(f['rgb'],f['depth'],f['intrinsics'],f['T_world_camera'],f['timestamp'])
        attempts=[{'camera_id':'head','result':result}]
        if not result['accepted'] and result.get('retained_tracking_prior'):
            f=load(root/'sensors/right_wrist'/f'{number:06d}.npz');camera='right_wrist'
            result=estimator.track_visible_depth(f['rgb'],f['depth'],f['intrinsics'],f['T_world_camera'],f['timestamp'])
            attempts.append({'camera_id':camera,'result':result})
        elapsed=time.monotonic()-start
        result.update(frame_id=number,timestamp=f['timestamp'])
        records.append({'frame_id':number,'camera_id':camera,'elapsed_ms':elapsed*1000,'result':result,
            'view_attempts':attempts,'image_tracker':tracked['diagnostics']})
        if not result['accepted']:raise RuntimeError(f'Actual frame{number} rejected: {result}')
        if tracked['targets'] and camera=='head':
            tracker.confirm_identity({'kind':'foundationpose_rgbd','accepted':True,'target_id':'learned_004',
                'timestamp':f['timestamp'],'frame_id':number,'camera_id':'head','center_world':tracked['targets'][0]['center_world'],
                'quality':result['quality'],'pose_position_sigma_m':max(.012,result['pose_position_sigma_m']),
                'pose_orientation_sigma_rad':result['pose_orientation_sigma_rad'],
                'T_world_object':result['T_world_object'],'cad_asset_id':'PullHandleGrasp','mesh_sha256':result['mesh_sha256']})
        z=f['depth'];render=estimator.last_rendered_depth;actualmask=estimator.last_mask
        foreground=(render>.001)&np.isfinite(z)&(z>.05)&(z<render-.012)
        rgb=f['rgb'].astype(float).copy();rgb[foreground]=.5*rgb[foreground]+.5*np.array([255,0,0])
        rgb[actualmask&~foreground]=.5*rgb[actualmask&~foreground]+.5*np.array([0,255,0])
        Image.fromarray(rgb.astype(np.uint8)).save(args.output/f'{number:06d}_{camera}_measured_visibility.png')
        np.savez(args.output/f'{number:06d}_mask_render.npz',mask=actualmask,rendered_depth=render)
        print(number,camera,result['accepted'],result['quality'],'rank',result['pose_observability'].get('scaled_rank'),'ms',elapsed*1000,flush=True)
    # Wrong rendering is an explicitly synthetic negative, never an actual
    # model result or relabelled camera observation.
    shifted=np.where(render>.001,render+.05,0)
    wrong=rendered_sensor_consistency(z,actualmask,shifted,allow_measured_occlusion=True)
    negative={'synthetic_wrong_pose_depth_offset_50mm_rejected':not wrong[0],'wrong_pose_quality':wrong[1]}
    removed=np.where(np.isfinite(z)&(z>.05),z*.5,z)
    failed=estimator.track_visible_depth(f['rgb'],removed,f['intrinsics'],f['T_world_camera'],f['timestamp']+.1)
    negative['synthetically_fully_occluded_current_depth_rejected']=not failed['accepted'];negative['fully_occluded_result']=failed
    report={'kind':'Actual SAM105 / RGBD tracker / pretrained FoundationPose on recorded Grasp011 head and right wrist images',
        'records':records,'accepted':sum(r['result']['accepted'] for r in records),'negative_tests':negative,
        'scene_truth_used':False,'silhouette_gate_changed':'measured foreground is excluded only for fresh registered tracking; raw IoU retained',
        'unchanged_gates':['0.45 IoU threshold','0.70 depth support','depth residual thresholds','full6D observability','0.5s prior acquisition age'],
        'new_extent_limits':{'min_observed_pixels':80,'min_observed_CAD_fraction':.20},
        'restored_recorded_accepted_prior':bool(archived is not None),
        'limits':['New actual SAM105 pixels; archived live binary masks unavailable.',
                  'No concurrent Isaac load; complete live grasp remains separate.']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not all(negative[k] for k in ('synthetic_wrong_pose_depth_offset_50mm_rejected','synthetically_fully_occluded_current_depth_rejected')):
        raise RuntimeError('Negative visibility checks failed')


if __name__=='__main__':main()
