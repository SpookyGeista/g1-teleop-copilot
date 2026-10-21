#!/usr/bin/env python3
"""Actual Isaac RGB-D, G1 physics, perception, assistance and VR service."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def arguments():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--cert',type=Path)
    p.add_argument('--key',type=Path)
    p.add_argument('--duration',type=float,default=0.,help='Wall seconds after setup; zero serves until stopped')
    p.add_argument('--camera-hz',type=float,default=15.)
    p.add_argument('--camera-width',type=int,choices=(640,960,1280),default=640)
    p.add_argument('--standoff',type=float,default=.14,help='Visual approach guide distance in metres; operator owns approach axis')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--perception',choices=('learned','geometric'),default='learned',help='Real pretrained DINO/SAM; geometric is an explicit comparison baseline')
    p.add_argument('--model-python',type=Path,default=None)
    p.add_argument('--pose-python',type=Path,default=None)
    p.add_argument('--pose-backend',choices=('foundationpose','surface'),default='foundationpose',help='CAD pose model, or explicit measured-surface comparison')
    p.add_argument('--semantic-hz',type=float,default=2.,help='Semantic keyframe rate; measured RGB-D tracking fills current frames')
    p.add_argument('--model-config',type=Path,default=None,help='JSON model/prompt configuration; interaction profile used when present')
    p.add_argument('--scene',choices=('cabinet','interaction'),default='interaction')
    p.add_argument('--start-pose',choices=('reference','fixture'),default=None,
                   help='Reference arms for live teleop; acceptance runs retain their fixture pose')
    p.add_argument('--arm-solver',choices=('dls','unitree'),default='dls',
                   help='Existing differential IK or isolated Unitree-objective arm IK comparison')
    p.add_argument('--wrist-cameras',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--lidar',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--require-action',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--frame-timeout',type=float,default=.7,help='Maximum measurement age accepted by assistance')
    p.add_argument('--eye-enabled',action='store_true',help='Optional external gaze; defaults OFF')
    p.add_argument('--assist-enabled',action='store_true',help='Defaults OFF until explicitly enabled')
    p.add_argument('--osc-port',type=int,default=0,help='Optional local eye adapter port; 0 disables')
    p.add_argument('--stack-acceptance',action='store_true',help='Scripted operator exercises real learned/action pipeline; no headset claim')
    p.add_argument('--vision-grasp-acceptance',action='store_true',help='Synthetic operator attempts a measured-CAD grasp and bounded physical pull through the production controller')
    p.add_argument('--manual-release-acceptance',action='store_true',help='With vision grasp acceptance: separately evaluate explicit manual withdrawal while opening; no autonomous release motion')
    p.add_argument('--negative-contact-control',action='store_true',help='Evaluation only: disable hand/selected-fixture collision pairs; the physical grasp report must fail')
    p.add_argument('--acceptance',action='store_true',help='Bounded synthetic operator through production pipeline')
    p.add_argument('--no-record-rgbd',action='store_true',help='Retain video/telemetry but omit per-frame depth archive')
    p.add_argument('--archive-compression',type=int,choices=(0,1),default=0,help='0: faster lossless raw NPZ, higher disk use; 1: compressed')
    AppLauncher.add_app_launcher_args(p)
    a=p.parse_args()
    if not .2<=a.semantic_hz<=10:p.error('--semantic-hz must be .2..10')
    if sum((a.acceptance,a.stack_acceptance,a.vision_grasp_acceptance))>1:p.error('Choose a single acceptance driver')
    if a.negative_contact_control and not a.vision_grasp_acceptance:p.error('--negative-contact-control requires --vision-grasp-acceptance')
    if a.manual_release_acceptance and not a.vision_grasp_acceptance:p.error('--manual-release-acceptance requires --vision-grasp-acceptance')
    if a.manual_release_acceptance and a.negative_contact_control:p.error('Manual release and negative contact control are separate evaluations')
    if a.vision_grasp_acceptance and (not a.require_action or a.perception!='learned' or a.pose_backend!='foundationpose' or a.scene!='interaction'):
        p.error('--vision-grasp-acceptance requires learned perception, FoundationPose, action selection and the interaction scene')
    if a.perception=='geometric' and a.pose_backend!='surface':p.error('Geometric baseline requires --pose-backend surface')
    if a.stack_acceptance and not a.require_action:p.error('--stack-acceptance requires action selection')
    if not .1<=a.frame_timeout<=2.:p.error('--frame-timeout must be 0.1..2 seconds')
    if a.acceptance and (a.require_action or a.perception!='geometric' or a.scene!='cabinet'):
        p.error('Legacy --acceptance requires --no-require-action --perception geometric --scene cabinet; use tools/validate_stack.py for the action stack')
    if not 1<=a.camera_hz<=30:p.error('--camera-hz must be 1..30')
    if not math.isfinite(a.duration) or a.duration<0:p.error('--duration must be finite and nonnegative')
    if not .04<=a.standoff<=.25:p.error('--standoff must be 0.04..0.25 metres')
    if bool(a.cert)!=bool(a.key):p.error('--cert and --key must be supplied together')
    a.enable_cameras=True;a.device='cpu'
    if a.start_pose is None:
        a.start_pose='fixture' if (a.acceptance or a.stack_acceptance or a.vision_grasp_acceptance) else 'reference'
    a.output=(a.output or ROOT/'runs'/datetime.now().strftime('assist_%Y%m%d_%H%M%S')).resolve()
    if a.output.exists():p.error('Output must be a new directory to preserve recordings')
    a.output.mkdir(parents=True)
    return a,AppLauncher

def main():
    args,AppLauncher=arguments()
    app=None;backend=None;service=None;osc=None;video=None;camera_worker=None;archive_worker=None;model=None;perception_runtime=None;pose_runtime=None;pose_worker=None
    result={'status':'starting','headset_verified':False,'external_eye_hardware_verified':False,
            'negative_contact_control':args.negative_contact_control,
            'manual_release_acceptance':args.manual_release_acceptance}
    gc_events=deque(maxlen=256);gc_starts={};gc_totals={}
    def trace_gc(phase,info):
        # Measure rare process-wide Python GC pauses without changing GC policy.
        stamp=time.monotonic();generation=info['generation'];thread_id=threading.get_ident()
        key=(thread_id,generation)
        if phase=='start':gc_starts[key]=stamp
        elif phase=='stop':
            began=gc_starts.pop(key,stamp);duration=stamp-began
            gc_events.append({'start':began,'end':stamp,'generation':generation,
                'thread_id':thread_id,'duration_ms':duration*1000,
                'collected':info['collected'],'uncollectable':info['uncollectable']})
            total=gc_totals.setdefault(generation,{'count':0,'seconds':0.,'max_seconds':0.})
            total['count']+=1;total['seconds']+=duration;total['max_seconds']=max(total['max_seconds'],duration)
    write_lock=threading.Lock()
    events=(args.output/'events.jsonl').open('w',buffering=1)
    trace=(args.output/'control.jsonl').open('w',buffering=1)
    camera_log=(args.output/'frames.jsonl').open('w',buffering=1)
    capture_log=(args.output/'captures.jsonl').open('w',buffering=1)
    def sink(value):
        from teleop_assist.server import json_bytes
        with write_lock:events.write(json_bytes(value).decode()+'\n')
    def save_json(path,value):
        from teleop_assist.server import json_bytes
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_bytes(json_bytes(value));temporary.replace(path)
    exit_code=0
    try:
        if args.perception=='learned':
            from teleop_assist.model_bridge import ModelProcess
            config_path=args.model_config or (ROOT/'configs/perception.interaction.json' if args.scene=='interaction' else None)
            model_config=json.loads(config_path.read_text()) if config_path and config_path.is_file() else {}
            if args.model_config and not args.model_config.is_file():raise FileNotFoundError(args.model_config)
            model=ModelProcess(python=args.model_python,**model_config)
            if args.pose_backend=='foundationpose':
                from teleop_assist.pose_bridge import PoseProcess,PoseRuntime
                pose_worker=PoseProcess(python=args.pose_python)
                pose_runtime=PoseRuntime(pose_worker,interval=1/args.camera_hz)
        app=AppLauncher(args).app
        import cv2
        cv2.setNumThreads(4)
        import numpy as np
        from teleop_assist.core import AssistConfig,AssistSession
        from teleop_assist.perception import CabinetPerception
        from teleop_assist.server import AssistHTTPServer,GazeOSCReceiver
        from teleop_assist.isaac_backend import IsaacBackend
        from teleop_assist.recording import write_rgbd
        from teleop_assist.task_profiles import annotate_tasks
        backend=IsaacBackend(args.output/'sim',headless=args.headless,seed=args.seed,width=args.camera_width,height=args.camera_width*3//4,
                            scene_variant=args.scene,wrist_cameras=args.wrist_cameras,
                            evaluation_contact_pairs=args.vision_grasp_acceptance,
                            evaluation_disable_hand_target_collisions=args.negative_contact_control,
                            start_pose=args.start_pose,arm_solver=args.arm_solver)
        session=AssistSession(AssistConfig(eye_enabled=args.eye_enabled,assist_enabled=args.assist_enabled,standoff=args.standoff,require_action=args.require_action,frame_timeout=args.frame_timeout),event_sink=sink)
        session.step(backend.last_state,backend.dt)
        perception=CabinetPerception() if args.perception=='geometric' else None
        service=AssistHTTPServer(session,args.host,args.port,args.cert,args.key)
        driver=None
        if args.acceptance:
            from teleop_assist.evaluation import AcceptanceDriver
            driver=AcceptanceDriver(session)
        elif args.stack_acceptance:
            from teleop_assist.stack_evaluation import StackAcceptanceDriver
            driver=StackAcceptanceDriver(session,require_cad_pose=args.pose_backend=='foundationpose')
        elif args.vision_grasp_acceptance:
            if args.manual_release_acceptance:
                from teleop_assist.manual_release_evaluation import ManualReleaseGraspDriver
                driver=ManualReleaseGraspDriver(session)
            else:
                from teleop_assist.vision_grasp_evaluation import VisionGraspPullDriver
                driver=VisionGraspPullDriver(session)
        def stop_signal(*unused):session.stop_requested=True
        signal.signal(signal.SIGINT,stop_signal);signal.signal(signal.SIGTERM,stop_signal)
        if args.osc_port:
            osc=GazeOSCReceiver(session,port=args.osc_port);osc.start()
        image_dir=args.output/'rgbd';image_dir.mkdir()
        video=cv2.VideoWriter(str(args.output/'robot_camera.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.camera_hz,(backend.width,backend.height))
        if not video.isOpened():raise RuntimeError('Video writer could not open')
        if model:
            from teleop_assist.perception_runtime import AsyncPerception
            perception_runtime=AsyncPerception(model,semantic_interval=1/args.semantic_hz)
        # Bound profiling memory during an hours-long headset session. Short
        # acceptance runs fit entirely; longer runs report recent cost windows.
        timing_window=20000
        frame_count=0;capture_count=0;archived_count=0
        capture_seconds=deque(maxlen=timing_window);perception_seconds=deque(maxlen=timing_window)
        gc.callbacks.append(trace_gc)
        sensor_dir=args.output/'sensors';sensor_dir.mkdir()
        def process_frame(bundle):
            nonlocal frame_count
            frame=bundle['frames']['head']
            started=time.monotonic()
            stage_started=started;stage_costs={}
            def stage_done(name):
                nonlocal stage_started
                now=time.monotonic();stage_costs[name]=(now-stage_started)*1000;stage_started=now
            with session.lock:
                selected_id=session.selection.id if session.selection.mode=='confirmed' else None
            stage_done('read_selection')
            if pose_runtime:
                # Start the confirmed CAD's actual current-depth measurement
                # while CPU semantic/flow bookkeeping processes the same bundle.
                # Initial registration and all publication gates stay in update.
                pose_runtime.prefetch(frame,selected_id,recovery_frames=bundle['frames'])
            stage_done('pose_prefetch')
            measured=(perception_runtime.process(bundle,selected_id) if model else
                      perception.process(frame['rgb'],frame['depth'],frame['intrinsics'],frame['T_world_camera'],frame['frame_id'],frame['timestamp']))
            stage_done('image_tracking')
            if pose_runtime:
                pose_runtime.update(frame,measured['targets'],perception_runtime.tracker.last_masks,selected_id,
                                    observations=perception_runtime.observations,recovery_frames=bundle['frames'])
                stage_done('pose_update')
                perception_runtime.confirm_cad_identity(pose_runtime.identity_confirmation)
                measured['diagnostics']['cad_pose']=dict(pose_runtime.diagnostics)
                stage_done('identity_confirmation')
            perception_seconds.append(time.monotonic()-started)
            k=frame['intrinsics']
            camera={'width':backend.width,'height':backend.height,
                    'intrinsics':[[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]],
                    'T_world_camera':frame['T_world_camera'].tolist(),'depth_convention':frame['depth_convention'],
                    'source':frame['source']}
            targets=annotate_tasks(measured['targets'],profile='interaction' if args.scene=='interaction' else 'approach_only',standoff=args.standoff) if args.require_action else measured['targets']
            stage_done('task_annotation')
            from teleop_assist.cad_overlay import apply_registered_cad_overlays
            targets=apply_registered_cad_overlays(targets,camera)
            stage_done('cad_overlay')
            sensor_metadata={}
            for camera_id,sensor in bundle['frames'].items():
                if camera_id=='head':continue
                service.publish_frame(sensor,camera_id=camera_id)
                sensor_metadata[camera_id]={key:sensor[key] for key in ('frame_id','timestamp','sim_time','intrinsics','T_world_camera','source') if key in sensor}
                for key in ('T_link_camera','source_link','camera_id'):
                    if key in sensor:sensor_metadata[camera_id][key]=sensor[key]
            scan=bundle.get('lidar')
            if scan is not None:
                sensor_metadata['lidar']={key:scan[key] for key in ('timestamp','sim_time','source','T_world_lidar') if key in scan}
            # Publish payload before its state identifier, so clients cannot see a half-published frame.
            service.publish_frame(frame)
            stage_done('publish_sensor_payloads')
            with session.lock:
                if driver and hasattr(driver,'filter_targets'):targets=driver.filter_targets(targets)
                session.update_frame(targets,frame['frame_id'],frame['timestamp'],camera,measured['diagnostics'])
            stage_done('publish_control_frame')
            published_at=time.monotonic()
            metadata={'frame_id':frame['frame_id'],'timestamp':frame['timestamp'],'sim_time':frame['sim_time'],
                      'processing_started_timestamp':started,'publish_timestamp':published_at,
                      'capture_to_publish_s':published_at-float(frame['timestamp']),
                      'pipeline_cost_ms':stage_costs,'archive_enqueued':bundle.get('_archive_enqueued',False),
                      'overlapping_gc_events':[dict(e) for e in list(gc_events) if e['end']>=started and e['start']<=published_at],
                      'camera':camera,'targets':targets,'diagnostics':measured['diagnostics'],'sensors':sensor_metadata}
            from teleop_assist.server import json_bytes
            camera_log.write(json_bytes(metadata).decode()+'\n')
            if frame_count==0:
                cv2.imwrite(str(args.output/'initial_rgb.png'),cv2.cvtColor(frame['rgb'],cv2.COLOR_RGB2BGR))
                save_json(args.output/'initial_state.json',session.snapshot())
            frame_count+=1
        def archive_bundle(bundle):
            nonlocal archived_count
            # This single bounded worker owns the video encoder and sensor
            # archive. All captured frames are offered, including frames that
            # the slower perception worker intentionally supersedes.
            frame=bundle['frames']['head']
            for camera_id,sensor in bundle['frames'].items():
                if camera_id=='head' or args.no_record_rgbd:continue
                camera_dir=sensor_dir/camera_id;camera_dir.mkdir(exist_ok=True)
                write_rgbd(camera_dir/f"{sensor['frame_id']:06d}.npz",sensor,compression=args.archive_compression)
            scan=bundle.get('lidar')
            if scan is not None:
                lidar_dir=sensor_dir/'lidar';lidar_dir.mkdir(exist_ok=True)
                np.savez_compressed(lidar_dir/f"{frame['frame_id']:06d}.npz",**scan)
            bgr=cv2.cvtColor(frame['rgb'],cv2.COLOR_RGB2BGR);video.write(bgr)
            if not args.no_record_rgbd:
                stem=f"{frame['frame_id']:06d}"
                cv2.imwrite(str(image_dir/(stem+'.jpg')),bgr,[cv2.IMWRITE_JPEG_QUALITY,90])
                write_rgbd(image_dir/(stem+'.npz'),frame,compression=args.archive_compression)
            capture_log.write(json.dumps({'frame_id':int(frame['frame_id']),
                'timestamp':float(frame['timestamp']),'sim_time':float(frame['sim_time']),
                'archived_timestamp':time.monotonic(),'rgbd_archived':not args.no_record_rgbd,
                'camera_ids':list(bundle['frames'])})+'\n')
            archived_count+=1
        def capture_frame():
            nonlocal capture_count
            begun=time.monotonic();bundle=backend.capture_bundle()
            if args.lidar:bundle['lidar']=backend.capture_lidar()
            capture_seconds.append(time.monotonic()-begun)
            # Isaac acquisition stays on the main thread. Fresh raw video does
            # not publish targets or renew a control/perception timestamp.
            service.publish_live_bundle(bundle)
            capture_count+=1
            # Give the recorder its own outer descriptor before annotating the
            # perception descriptor; both only read the owned sensor arrays.
            bundle['_archive_enqueued']=archive_worker.submit(dict(bundle))
            if not bundle['_archive_enqueued']:
                sink({'type':'recording_frame_dropped','timestamp':time.monotonic(),
                      'frame_id':bundle['frames']['head']['frame_id'],
                      'reason':'bounded_archive_queue_full'})
            return bundle
        from teleop_assist.pipeline_workers import LatestBundleWorker,BoundedArchiveWorker
        archive_worker=BoundedArchiveWorker(archive_bundle,capacity=4)
        process_frame(capture_frame())
        # Clear startup/SDK cycles before accepting an operator. Automatic GC
        # remains enabled and its later pauses remain observable in telemetry.
        gc_started=time.monotonic();startup_collected=gc.collect()
        result['startup_gc']={'collected':startup_collected,'seconds':time.monotonic()-gc_started,
                              'before_operator_service':True}
        gc_events.clear();gc_totals.clear()
        service.start()
        camera_worker=LatestBundleWorker(process_frame)
        url=f'{service.scheme}://{args.host}:{service.address[1]}'
        manifest={'url':url,'pid':__import__('os').getpid(),'output':str(args.output),'eye_default':args.eye_enabled,
                  'arm_solver':args.arm_solver,
                  'launch_arguments':sys.argv[1:],'seed':args.seed,
                  'execution_schedule':'wall_paced_physics_independent_raw_capture_latest_pending_perception_bounded_archive',
                  'opencv_cpu_threads':cv2.getNumThreads(),
                  'perception_pending_capacity':1,'archive_pending_capacity':4,
                  'live_display_scope':'metric_RGBD_reprojection_in_3D; flat_2D_requires_identical_camera; measurement IDs and ages unchanged',
                  'assist_default':args.assist_enabled,'camera_hz_requested':args.camera_hz,
                  'recording':'actual RGB-D, commands, measured PhysX state, selections and reference commands',
                  'input':'synthetic acceptance driver' if driver else 'WebXR / desktop operator',
                  'simulation_only':True,'fixed_base':True,'no_hardware_endpoints':True,
                  'evaluation_contact_pairs':args.vision_grasp_acceptance,
                  'negative_contact_control':args.negative_contact_control,
                  'manual_release_acceptance':args.manual_release_acceptance,
                  'perception_backend':args.perception,'scene_variant':args.scene,
                  'require_action':args.require_action,'wrist_cameras':args.wrist_cameras,'lidar':args.lidar,
                  'model_worker':model.manifest if model else None,'model_config':model_config if model else None,'semantic_hz_requested':args.semantic_hz,
                  'archive_compression':args.archive_compression,'pose_backend':args.pose_backend,
                  'pose_hz_requested':args.camera_hz if pose_runtime else None,
                  'pose_worker':pose_worker.manifest if pose_worker else None}
        from threadpoolctl import threadpool_info
        manifest['numerical_thread_pools']=threadpool_info()
        manifest['declared_model_assets']={name:json.loads((ROOT/'models'/name).read_text())
            for name in ('manifest.json','foundationpose_manifest.json')
            if model and (ROOT/'models'/name).is_file()}
        save_json(args.output/'known_task_catalog.json',backend.known_task_metadata())
        import hashlib
        manifest['source_hashes']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT/'teleop_assist').rglob('*')) if path.is_file() and path.suffix in ('.py','.mjs','.js','.html','.css')}
        manifest['source_hashes']['tools/assist_sim.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest['source_hashes']['run_assist.py']=hashlib.sha256((ROOT/'run_assist.py').read_bytes()).hexdigest()
        save_json(args.output/'manifest.json',manifest)
        save_json(args.output/'evaluation_truth.json',backend.evaluation_truth())
        print('ASSIST_READY '+url+' output='+str(args.output),flush=True)
        begun=time.monotonic();next_tick=begun;next_capture=backend.last_state['sim_time']+1/args.camera_hz;last_status=0.
        state=backend.last_state;step=0
        loop_costs={key:deque(maxlen=timing_window) for key in ('operator_update','control_step','physics_step')}
        while app.is_running() and not session.stop_requested:
            camera_worker.check();archive_worker.check()
            wall=time.monotonic()
            if args.duration and wall-begun>=args.duration:break
            if driver:
                tick_started=time.monotonic()
                with session.lock:driver.update(state,state['sim_time'])
                loop_costs['operator_update'].append(time.monotonic()-tick_started)
                if driver.done:break
            tick_started=time.monotonic()
            reference=session.step(state,backend.dt)
            loop_costs['control_step'].append(time.monotonic()-tick_started)
            tick_started=time.monotonic()
            if reference['hold'] or reference.get('reset_pending',False):backend.hold()
            state=backend.step(reference['ee_target_world'],reference.get('ee_quat_xyzw'),reference['grip'])
            loop_costs['physics_step'].append(time.monotonic()-tick_started)
            if driver:
                with session.lock:driver.observe(state,reference,state['sim_time'])
            if step%10==0:
                trace.write(json.dumps({'timestamp':time.monotonic(),'robot':state,'reference':reference,
                                        'selection':session.selection.snapshot(),'action':session.snapshot().get('action'),'assist':session.control})+'\n')
            if state['sim_time']>=next_capture:
                camera_worker.submit(capture_frame())
                next_capture+=1/args.camera_hz
                if next_capture<=state['sim_time']:
                    next_capture=state['sim_time']+1/args.camera_hz
            if wall-last_status>1:
                save_json(args.output/'state.json',session.snapshot());last_status=wall
            # The scripted operator uses the same asynchronous camera and
            # wall-clock pacing as an actual headset. Prediction derivatives
            # must not run against a different clock from the physical plant.
            next_tick+=backend.dt
            current=time.monotonic()
            if current-next_tick>.1:next_tick=current
            remaining=next_tick-current
            if remaining>0:time.sleep(remaining)
            step+=1
        camera_worker.close()
        process_frame(capture_frame())
        archive_worker.close()
        cv2.imwrite(str(args.output/'final_rgb.png'),cv2.cvtColor(backend.last_capture['rgb'],cv2.COLOR_RGB2BGR))
        save_json(args.output/'state.json',session.snapshot())
        result.update(status='completed',steps=step,frames=frame_count,sim_seconds=state['sim_time'],
                      wall_seconds=time.monotonic()-begun,real_isaac_render=True,real_physx=True,
                      camera_captures_deferred_busy=0,camera_frames_captured=capture_count,frames_archived=archived_count,
                      perception_worker=camera_worker.statistics(),archive_worker=archive_worker.statistics(),
                      render_median_ms=float(np.median(capture_seconds)*1000),
                      perception_median_ms=float(np.median(perception_seconds)*1000),http_errors=service.errors)
        result.update(achieved_physics_hz=step/result['wall_seconds'],
                      real_time_factor=step*backend.dt/result['wall_seconds'],
                      achieved_camera_hz=max(0,capture_count-2)/result['wall_seconds'],
                      achieved_perception_hz=max(0,frame_count-2)/result['wall_seconds'],
                      lossless_rgbd_recorded=not args.no_record_rgbd,
                      recording_complete=archived_count==capture_count)
        result['timing_statistics_scope']={'type':'bounded_recent_samples','maximum_samples_per_stage':timing_window,
            'render_samples':len(capture_seconds),'perception_samples':len(perception_seconds)}
        result['main_loop_cost_ms']={key:{'median':float(np.median(values)*1000),
            'p95':float(np.quantile(values,.95)*1000),'samples':len(values)}
            for key,values in loop_costs.items() if values}
        result['garbage_collection_totals']=gc_totals
        result['garbage_collection_scope']='after_startup_collection; automatic_GC_enabled'
        if driver:
            report=driver.report();save_json(args.output/'acceptance.json',report);result['acceptance']=report
            if not report.get('passed',False):result['status']='acceptance_failed';exit_code=2
        if not result['recording_complete']:
            result['status']='recording_incomplete';exit_code=2
        print('ASSIST_RESULT '+json.dumps(result),flush=True)
    except BaseException as error:
        exit_code=1;result.update(status='error',error=traceback.format_exc())
        failure=getattr(error,'failure',None)
        if failure is not None:
            result['worker_failure']={'exception_type':failure.exception_type,
                'message':failure.message,'traceback':failure.traceback}
        (args.output/'error.txt').write_text(result['error']);print(result['error'],flush=True)
    finally:
        if trace_gc in gc.callbacks:gc.callbacks.remove(trace_gc)
        for worker in (camera_worker,archive_worker):
            if worker is not None:
                try:worker.close()
                except Exception as error:
                    result.setdefault('cleanup_errors',[]).append(repr(error))
                    exit_code=1;result['status']='error' 
        if service:
            if service.thread.is_alive():service.close()
            else:service.httpd.server_close()
        if osc:osc.close()
        if video:video.release()
        events.close();trace.close();camera_log.close();capture_log.close()
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
        exit_file=__import__('os').environ.get('PUNKTNULL_ASSIST_EXIT_FILE')
        if exit_file:Path(exit_file).write_text(str(exit_code))
        if perception_runtime:perception_runtime.close()
        if pose_runtime:pose_runtime.close()
        elif pose_worker:pose_worker.close()
        if model:model.close()
        if backend:backend.close()
        if app:app.close(wait_for_replicator=False,skip_cleanup=True)
    return exit_code

if __name__=='__main__':raise SystemExit(main())
