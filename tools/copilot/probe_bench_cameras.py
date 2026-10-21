#!/usr/bin/env python3
"""Bounded capture of one original camera in the isolated PhysX demo bench.

No DDS, XR server, learned command, camera-preset edits or hardware connection.
--check validates files/arguments without importing or launching Isaac.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim',action='store_true',help='Mandatory isolated-simulation acknowledgement')
    parser.add_argument('--check',action='store_true',help='Check inputs only; no Isaac imports, GPU, output or server')
    parser.add_argument('--camera',choices=('head','right_wrist','left_wrist'),default='head')
    parser.add_argument('--repo',type=Path,default=Path('/home/loya2/unitree_sim_isaaclab'))
    parser.add_argument('--xr-repo',type=Path,default=Path('/home/loya2/xr_teleoperate'))
    parser.add_argument('--initialization',type=Path,default=ROOT/'teleop_assist/copilot/assets/demo_cylinder/initialization.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--settling-steps',type=int,default=20,help='Additional .005-s physics steps after the bench constructor settles 100 steps')
    parser.add_argument('--frames',type=int,default=3,help='Capture 1..10 frames, one physics step between frames')
    parser.add_argument('--headless',action='store_true',help='Run the explicitly requested capture without a GUI')
    args=parser.parse_args(argv)
    if not args.sim:parser.error('--sim is required; this tool has no hardware mode')
    if args.output.exists():parser.error('Output already exists; choose a new directory')
    if not 0<=args.settling_steps<=200 or not 1<=args.frames<=10:
        parser.error('Use 0..200 additional settling steps and 1..10 frames')
    required=(args.repo/'robots/unitree.py',args.repo/'tasks/common_config/camera_configs.py',
              args.xr_repo/'assets/g1/g1_body29_hand14.urdf',args.initialization)
    missing=[str(path) for path in required if not path.is_file()]
    if missing:parser.error('Missing probe inputs: '+', '.join(missing))
    try:
        initialization=json.loads(args.initialization.read_text())
        joints=initialization.get('joint_positions') if isinstance(initialization,dict) else None
        if (initialization.get('accepted') is not True or not isinstance(joints,dict) or len(joints)!=43
                or any(not isinstance(name,str) or type(value) not in (int,float) or not math.isfinite(value)
                       for name,value in joints.items())):
            raise ValueError('Expected accepted demo initialization with 43 finite named joints')
    except (OSError,ValueError,TypeError,AttributeError,OverflowError) as exc:parser.error(str(exc))
    return args,initialization


def camera_config(camera):
    """Serialize observed effective configuration; never change preset values."""
    cfg=camera.cfg
    return dict(prim_path=cfg.prim_path,height=int(cfg.height),width=int(cfg.width),
        data_types=list(cfg.data_types),update_period=float(cfg.update_period),
        update_latest_camera_pose=bool(cfg.update_latest_camera_pose),
        offset=dict(pos=list(cfg.offset.pos),rot_wxyz=list(cfg.offset.rot),convention=cfg.offset.convention),
        pinhole={name:list(getattr(cfg.spawn,name)) if name=='clipping_range' else float(getattr(cfg.spawn,name))
            for name in ('focal_length','focus_distance','horizontal_aperture','clipping_range')})


def main(argv=None):
    args,initialization=arguments(argv)
    camera_file=args.repo/'tasks/common_config/camera_configs.py'
    source=dict(camera=args.camera,repository=str(args.repo.resolve()),xr_repository=str(args.xr_repo.resolve()),
        original_camera_config=str(camera_file.resolve()),original_camera_config_sha256=sha256(camera_file),
        initialization=str(args.initialization.resolve()),initialization_sha256=sha256(args.initialization),
        original_preset=('CameraBaseCfg.get_camera_config' if args.camera=='head'
            else 'CameraPresets.'+args.camera.replace('_wrist','_dex3_wrist_camera')),
        capture_only=True,dds_started=False,hardware_tested=False,
        requested_frames=args.frames,additional_settling_steps=args.settling_steps)
    if args.check:
        print(json.dumps(dict(source,status='inputs_checked',isaac_started=False),indent=2));return 0
    os.environ['PROJECT_ROOT']=str(args.repo.resolve())
    os.environ['PUNKTNULL_ISAAC_REPOSITORY']=str(args.repo.resolve())
    from isaaclab.app import AppLauncher
    # Explicit kwargs follow the installed AppLauncher API. No second context.
    from teleop_assist.copilot.render_sync import zero_delay_kit_args
    app=AppLauncher(headless=args.headless,enable_cameras=True,device='cpu',
        kit_args=zero_delay_kit_args()).app
    stop=False;previous={};frames=[];started=time.monotonic()
    def request_stop(*_):
        nonlocal stop
        stop=True
    try:
        previous={number:signal.signal(number,request_stop) for number in (signal.SIGTERM,signal.SIGINT)}
        import numpy as np
        from PIL import Image
        from teleop_assist.copilot.physx_bench import PhysXOrientationBench
        from teleop_assist.copilot.frames import publish_frame,load_frame
        from teleop_assist.copilot.mailbox import atomic_json
        args.output.mkdir(parents=True)
        bench=PhysXOrientationBench(args.repo,args.xr_repo,initialization,camera_source=args.camera)
        effective=camera_config(bench.camera)
        extra_contact=0.
        for _ in range(args.settling_steps):
            if stop:break
            extra_contact=max(extra_contact,bench.step(1,render=True))
        for index in range(args.frames):
            if stop or not app.is_running():break
            if index:bench.step(1,render=True)
            frame=bench.camera_frame();frame['camera_source']=args.camera
            # No physics step occurs between camera_frame and this ground-truth
            # snapshot; preserve their individual observation timestamps.
            _,_,scene,_=bench.snapshot()
            scene_file=f'scene_{index:04d}.json'
            atomic_json(args.output/scene_file,scene)
            camera_link='d435_link' if args.camera=='head' else args.camera.split('_')[0]+'_hand_camera_base_link'
            link_index=list(frame['body_names']).index(camera_link)
            archive=args.output/f'frame_{index:04d}.npz'
            np.savez_compressed(archive,**frame)
            checked=load_frame(archive)
            Image.fromarray(checked['rgb']).save(args.output/f'frame_{index:04d}.jpg',quality=95)
            publish_frame(args.output,frame)
            Image.fromarray(checked['rgb']).save(args.output/'frame.jpg',quality=95)
            valid_depth=np.isfinite(checked['depth'])&(checked['depth']>0)
            frames.append(dict(index=index,path=archive.name,sha256=sha256(archive),
                frame_id=checked['frame_id'],capture_timestamp=checked['timestamp'],clock_id=checked['clock_id'],
                calibration_id=checked['calibration_id'],T_world_camera=checked['T_world_camera'].tolist(),
                T_world_base=checked['T_world_base'].tolist(),intrinsics=checked['intrinsics'].tolist(),
                scene_file=scene_file,scene_capture_timestamp=scene['acquisition_timestamp'],
                observed_camera_link=camera_link,observed_camera_link_pose_wxyz=frame['body_poses'][link_index].tolist(),
                observed_target_world_pose=next(item['world_pose'] for item in scene['obstacles'] if item['id']=='target'),
                rgb_shape=list(checked['rgb'].shape),finite_positive_depth_fraction=float(valid_depth.mean()),
                depth_range_m=[float(checked['depth'][valid_depth].min()),float(checked['depth'][valid_depth].max())] if valid_depth.any() else None))
        dds_imported=any(name.startswith(('unitree_sdk2py','dds.')) for name in sys.modules)
        report=dict(source,status='captured' if frames else 'cancelled_before_capture',isaac_started=True,
            world_camera_convention='ROS optical frame: x right, y down, z forward',
            timestamp_provenance='Host monotonic after simulated camera update; no physical sensor capture',
            effective_camera_config=effective,frames=frames,dds_imported=dds_imported,
            constructor_settling_steps=100,constructor_settling_contact_n=bench.settling_contact_n,
            additional_settling_contact_n=extra_contact,gravity_saturation_steps=bench.gravity_saturation_steps,
            elapsed_s=time.monotonic()-started,stop_requested=stop,
            limitations=['Original camera preset on a custom isolated table scene',
                'Capture and visibility probe only; no motion or successful grasp claim',
                'No headset test or physical camera calibration'])
        atomic_json(args.output/'capture_report.json',report)
        print(json.dumps(report,indent=2),flush=True)
        if dds_imported:raise RuntimeError('Unexpected robot transport import in isolated capture')
        return 0 if frames else 2
    finally:
        app.close(wait_for_replicator=False,skip_cleanup=True)
        for number,handler in previous.items():signal.signal(number,handler)


if __name__=='__main__':raise SystemExit(main())
