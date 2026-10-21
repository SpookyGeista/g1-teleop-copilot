#!/usr/bin/env python3
"""Verify RGB-D/FK alignment across recorded isolated-simulator arm postures.

Postures are assigned as simulator test initializations, not commanded robot
trajectories. The red-pixel/cylinder oracle is used ONLY for this calibration
test; it supplies no perception, target selection or motion authorization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim',action='store_true',required=True)
    parser.add_argument('--recording',type=Path,required=True,help='Actual rows.json with measured q43')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render-passes',type=int,choices=(1,2,3,4),default=1,
        help='Diagnostic repeated render/read at unchanged physics state')
    parser.add_argument('--repo',type=Path,default=Path('/home/loya2/unitree_sim_isaaclab'))
    parser.add_argument('--xr-repo',type=Path,default=Path('/home/loya2/xr_teleoperate'))
    args=parser.parse_args()
    if args.output.exists():parser.error('Output already exists')
    rows=json.loads(args.recording.read_text())
    selected=[r for r in rows if r.get('scenario_tick') in range(11,19)]
    if [r['scenario_tick'] for r in selected]!=list(range(11,19)):
        parser.error('Recording must contain unique ordered scenario ticks11..18')
    initialization=json.loads((ROOT/'teleop_assist/copilot/assets/demo_cylinder/initialization.json').read_text())
    os.environ['PROJECT_ROOT']=str(args.repo.resolve())
    os.environ['PUNKTNULL_ISAAC_REPOSITORY']=str(args.repo.resolve())
    from isaaclab.app import AppLauncher
    from teleop_assist.copilot.render_sync import zero_delay_kit_args
    app=AppLauncher(headless=True,enable_cameras=True,device='cpu',kit_args=zero_delay_kit_args()).app
    try:
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation
        from PIL import Image
        from teleop_assist.copilot.physx_bench import PhysXOrientationBench
        from teleop_assist.copilot.observations import GeometryObservation
        bench=PhysXOrientationBench(args.repo,args.xr_repo,initialization,camera_source='right_wrist')
        args.output.mkdir(parents=True)
        reports=[]
        # Reverse order also exercises the opposite camera displacement.
        for index,row in enumerate(selected+list(reversed(selected))):
            named=dict(zip(row['actual_joint_names'],row['actual_q43']))
            if set(named)!=set(bench.robot.joint_names):raise ValueError('Complete matching q43 required')
            state=torch.tensor([[named[n] for n in bench.robot.joint_names]],dtype=torch.float32)
            if not torch.isfinite(state).all():raise ValueError('Nonfinite input posture')
            bench.robot.write_joint_state_to_sim(state,torch.zeros_like(state))
            bench.robot.set_joint_position_target(state)
            bench.step(1)
            started=time.monotonic()
            for _ in range(args.render_passes):frame=bench.camera_frame()
            elapsed=time.monotonic()-started
            rgb=frame['rgb'].astype(float)
            mask=(rgb[:,:,0]>70)&(rgb[:,:,0]>1.6*rgb[:,:,1])&(rgb[:,:,0]>1.6*rgb[:,:,2])
            observation=GeometryObservation('test_oracle_only',f'frame:{index}',frame['timestamp'],
                frame['clock_id'],frame['calibration_id'],'RGB_test_oracle_not_runtime_perception',
                frame['depth'],mask,frame['intrinsics'],frame['T_world_camera'],frame['T_world_base'])
            points,_=observation.point_clouds(max_object_points=20000)
            scene=json.loads(frame['simulation_scene_json'])
            target=next(item for item in scene['obstacles'] if item['id']=='target')
            transform=np.asarray(target['world_pose'])
            local=(points-transform[:3,3])@transform[:3,:3]
            radial=np.linalg.norm(local[:,:2],axis=1);height=abs(local[:,2])
            radius,half_height=target['radius'],target['length']/2
            side=np.hypot(radial-radius,np.maximum(height-half_height,0))
            cap=np.hypot(np.maximum(radial-radius,0),height-half_height)
            distances=np.minimum(side,cap)
            stem=f'frame_{index:02d}'
            np.savez_compressed(args.output/(stem+'.npz'),**frame)
            if index in (0,3,8,11):Image.fromarray(frame['rgb']).save(args.output/(stem+'.png'))
            report=dict(index=index,source_runtime_tick=row['tick'],source_scenario_tick=row['scenario_tick'],
                capture_timestamp=frame['timestamp'],render_and_read_s=elapsed,point_count=len(points),
                median_surface_error_m=float(np.median(distances)),p95_surface_error_m=float(np.quantile(distances,.95)),
                T_world_camera=frame['T_world_camera'].tolist(),file=stem+'.npz')
            reports.append(report)
        reference=np.asarray(reports[0]['T_world_camera'])
        excursion=max(float(Rotation.from_matrix(np.asarray(r['T_world_camera'])[:3,:3]@reference[:3,:3].T).magnitude())
            for r in reports)
        result=dict(schema='punktnull.dynamic_camera_calibration_probe.v1',
            accepted=excursion>=np.deg2rad(5) and all(r['p95_surface_error_m']<=.002 for r in reports),
            maximum_camera_rotation_excursion_deg=float(np.rad2deg(excursion)),minimum_camera_rotation_excursion_deg=5.,
            render_read_passes_per_state=args.render_passes,
            threshold_p95_surface_error_m=.002,frames=reports,render_settings=bench.render_sync,
            recording_sha256=hashlib.sha256(args.recording.read_bytes()).hexdigest(),
            bench_sha256=hashlib.sha256((ROOT/'teleop_assist/copilot/physx_bench.py').read_bytes()).hexdigest(),
            render_sync_sha256=hashlib.sha256((ROOT/'teleop_assist/copilot/render_sync.py').read_bytes()).hexdigest(),
            dds_imported=any(n.startswith(('unitree_sdk2py','dds.')) for n in sys.modules),
            hardware_tested=False,scope='same_state_simulator_RGBD_extrinsics_across_posture_changes',
            limitations=['Recorded simulator states are assigned; no actuator-trajectory validation.',
                'Known synthetic cylinder and red-pixel test oracle; not a production detector.',
                'No real hand-eye calibration, headset or grasp-success test.'])
        (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2),flush=True)
    finally:app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=='__main__':main()
