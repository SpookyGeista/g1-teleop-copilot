#!/usr/bin/env python3
"""Original continuous Unitree arm IK connected directly to isolated Isaac.

No custom collision/path guard, proposal permission, command receipt, DDS or
physical robot connection. --scripted-test is explicitly synthetic, not Quest.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim',action='store_true',required=True)
    parser.add_argument('--initialization',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--operator-mailbox',type=Path)
    source.add_argument('--scripted-test',action='store_true')
    parser.add_argument('--contact-test',action='store_true',help='Synthetic manual table-contact scenario')
    parser.add_argument('--finger-test',action='store_true',help='Synthetic independent trigger/hold/release test')
    parser.add_argument('--grasp-assist',action='store_true',help='Opt-in learned alignment of the right hand in this same scene')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/copilot.local.json')
    parser.add_argument('--prompt',default='red cylinder')
    parser.add_argument('--ai-primary-camera',choices=('head','right_wrist'),default='head')
    parser.add_argument('--ai-test',action='store_true',help='Synthetic software integration test with real learned proposals')
    parser.add_argument('--duration',type=float,default=3600.)
    parser.add_argument('--port',type=int,default=8778)
    parser.add_argument('--repo',type=Path,default=Path('/home/loya2/unitree_sim_isaaclab'))
    parser.add_argument('--xr-repo',default='/home/loya2/xr_teleoperate')
    parser.add_argument('--ik-python',default='/home/loya2/miniconda3/envs/to/bin/python')
    parser.add_argument('--operator-camera-source',choices=('head','right_wrist','left_wrist'),default='head')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.contact_test and not args.scripted_test:parser.error('--contact-test requires --scripted-test')
    if args.finger_test and (not args.scripted_test or args.contact_test):
        parser.error('--finger-test requires --scripted-test and excludes --contact-test')
    if args.ai_test and (not args.scripted_test or not args.grasp_assist or args.finger_test or args.contact_test):
        parser.error('--ai-test requires --grasp-assist --scripted-test only')
    if args.output.exists() or not 1<=args.duration<=3600 or not 1<=args.port<=65535:
        parser.error('New output and bounded duration/port required')
    content=args.initialization.read_bytes();setup=json.loads(content)
    if setup.get('accepted') is not True:parser.error('Accepted simulation initialization required')
    args.output.mkdir(parents=True)
    args.enable_cameras=True;args.device='cpu'
    os.environ['PROJECT_ROOT']=str(args.repo.resolve())
    os.environ['PUNKTNULL_ISAAC_REPOSITORY']=str(args.repo.resolve())
    from teleop_assist.copilot.render_sync import zero_delay_kit_args
    args.kit_args=zero_delay_kit_args(args.kit_args)
    app=AppLauncher(args).app
    import numpy as np
    from scipy.spatial.transform import Rotation
    from teleop_assist.copilot.physx_bench import PhysXOrientationBench
    from teleop_assist.copilot.motion_process import MotionProcess
    from teleop_assist.copilot.manual_sim_control import ManualSimControl
    from teleop_assist.copilot.manual_fingers import Dex3TriggerControl
    from teleop_assist.copilot.manual_grasp_assist import ManualGraspAssist
    from teleop_assist.copilot.camera_publisher import CameraPublisher
    from teleop_assist.copilot.vr_buttons import source_payload
    from teleop_assist.copilot.dashboard import Dashboard
    from teleop_assist.copilot.mailbox import atomic_json,read_json
    from teleop_assist.copilot.kinematics import ARM_JOINT_NAMES
    from tools.copilot.sim_runtime import scripted_packet,write_audit
    from tools.copilot.xr_source import validate_operator_packet
    stop=False;worker=dashboard=bench=proposal_process=proposal_log=None;rows=[];error=None;completed=False;own_gc_freeze=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    status={};session=None;camera_publisher=None
    try:
        bench=PhysXOrientationBench(args.repo,args.xr_repo,setup,
            camera_source='right_wrist' if args.grasp_assist else args.operator_camera_source,
            operator_camera_source=args.operator_camera_source,
            support_camera_resolution=(320,240) if args.grasp_assist and args.operator_camera_source=='head' else None)
        worker=MotionProcess(args.ik_python,args.xr_repo,mode='unitree_manual_sim')
        assist=ManualGraspAssist(simulation_mode=True,guidance_profile='early_guidance',
            support_camera='head' if args.ai_primary_camera=='right_wrist' else 'right_wrist') if args.grasp_assist else None
        session=ManualSimControl(worker,simulation_mode=True,target_filter=assist.apply if assist else None)
        fingers=Dex3TriggerControl(bench.robot.joint_names,
            Path(args.xr_repo)/'assets/g1/g1_body29_hand14.urdf',simulation_mode=True)
        dashboard=Dashboard(args.output,port=args.port);dashboard.start()
        atomic_json(args.output/'runtime_source_manifest.json',dict(
            control_mode='unitree_manual',custom_motion_guard_enabled=False,
            manual_ik=worker.manifest,initial_condition_sha256=hashlib.sha256(content).hexdigest(),
            runtime_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            finger_control='independent_index_triggers',finger_speed_rad_s=2.,
            learned_grasp_assist=args.grasp_assist,
            physics_substeps_per_control=2 if args.grasp_assist else 5,
            support_camera_resolution=[320,240] if args.grasp_assist and args.operator_camera_source=='head' else None,
            assist_implementation_sha256=hashlib.sha256((ROOT/'teleop_assist/copilot/manual_grasp_assist.py').read_bytes()).hexdigest(),
            robot_transport_started=False,hardware_authorized=False))
        initial=bench.snapshot()[3]
        if assist:
            from run_copilot import configuration
            _,paths=configuration(args.config.resolve())
            proposal_log=(args.output/'proposal_service.log').open('w')
            proposal_process=subprocess.Popen([sys.executable,str(ROOT/'tools/copilot/serve.py'),
                '--no-dashboard','--complete-static-sim-preview','--recover-transient-tracking',
                '--live-frames',str(args.output/'ai_frames'),
                '--runtime',str(args.output),'--duration',str(args.duration),'--prompt',args.prompt,
                '--xr-repo',args.xr_repo,'--ik-python',args.ik_python,
                '--model-python',paths['perception_python'],'--grasp-python',paths['grasp_python'],
                '--grasp-repo',paths['grasp_repository'],'--model-root',paths['model_root']],
                # AppLauncher overwrites OPENBLAS_NUM_THREADS (24 on this host).
                # Do not pass Isaac's pool size to current-frame geometry work.
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                stdout=proposal_log,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic();tick=0;last_camera=0.;latest_test=None;wrist_frame=None;support_frame=None
        camera_publisher=CameraPublisher(args.output,assistance=bool(assist))
        camera_publish_work_s=0.
        # Keep the large, long-lived Isaac startup heap out of recurring
        # generation-2 scans. New allocations still use ordinary cyclic GC.
        gc.collect()
        if gc.get_freeze_count()==0:gc.freeze();own_gc_freeze=True
        ai_test_phase=None;ai_choice=None;ai_raw_origin=None
        def read_inputs():
            if args.scripted_test:return latest_test
            packet=read_json(args.operator_mailbox,{})
            runtime=dict(assist.annotate(session.status) if assist else session.status,
                timestamp=time.monotonic(),clock_id=bench.clock_id,scripted_operator=False)
            if not assist:runtime['candidates']=[]
            # An invalid pose is handled by the manual hold watchdog. Feeding
            # it to the assist-mode button supervisor would revoke ownership
            # on one dropped frame and prevent the retained calibration resuming.
            if validate_operator_packet(packet,now=time.monotonic(),clock_id=bench.clock_id) is None:
                dashboard.consume_vr(packet,runtime_status=runtime,now=time.monotonic())
            return packet,read_json(args.output/'sim_control.json',{})
        def hold():bench.robot.set_joint_position_target(bench.robot.data.joint_pos.clone())
        test_ticks=360 if args.finger_test else 180
        while not stop and time.monotonic()-start<args.duration and (args.ai_test or not args.scripted_test or tick<test_ticks):
            begin=time.monotonic();camera_work_s=0.
            if proposal_process is not None and proposal_process.poll() is not None:
                raise RuntimeError('Learned proposal service exited; see proposal_service.log')
            camera_publish_work_s=camera_publisher.poll()
            if not camera_publisher.busy and begin-last_camera>=(.05 if assist else 1/30):
                camera_begin=time.monotonic()
                frames=bench.camera_frames((args.operator_camera_source,'right_wrist') if assist else (args.operator_camera_source,))
                frame=frames[args.operator_camera_source]
                primary=frame
                if assist:
                    wrist_frame=frames['right_wrist']
                    primary=wrist_frame if args.ai_primary_camera=='right_wrist' else frame
                    support_frame=frame if args.ai_primary_camera=='right_wrist' else wrist_frame
                camera_publisher.submit(frame,primary,assist.candidates if assist else [],
                    (assist.confirmed or assist.selected) if assist else None)
                camera_work_s=time.monotonic()-camera_begin
                last_camera=begin
            control_begin=time.monotonic()
            _,state,_,poses=bench.snapshot()
            if args.scripted_test:
                raw=[np.array(setup['left_base_ee']),np.array(setup['raw_right_base_ee'])]
                amount=max(0,min(tick-20,50))*.0004
                if args.contact_test:
                    progress=min(1.,max(0.,(tick-20)/60.))
                    if tick>125:progress=max(0.,(165-tick)/40.)
                    amount=-.12*progress
                    for pose in raw:
                        pose[0,3]+=.10*progress
                        pose[:3,:3]=Rotation.from_euler('y',1.2*progress).as_matrix()@pose[:3,:3]
                raw[0][2,3]+=amount;raw[1][2,3]+=amount
                stamp=time.monotonic()
                packet=scripted_packet(0,stamp,bench.clock_id,*raw)
                packet.update(source_sequence=tick,publish_sequence=tick+1,
                    source_instance_id='manual_sim_test',source_event_clock_id='synthetic_test')
                packet['envelope']['sequence']=tick
                if not args.contact_test and not args.finger_test and not args.ai_test and (50<=tick<52 or 140<=tick<150):
                    packet['tracking_valid']=False;packet['envelope']['tracking_valid']=False
                    if tick>=140:time.sleep(.04)
                ui=dict(revision=tick+1,capture_timestamp=stamp,clock_id=bench.clock_id,
                    client_id='synthetic_NO_HEADSET',clutch=tick==5,start=tick in (10,90,110,155),release=tick==170)
                if args.contact_test:
                    ui['start']=tick==10
                    # Exercise the actual source-tolerance mismatch too.
                    for key in ('left_base_ee','right_base_ee'):
                        pose=np.array(packet[key]);pose[:3,:3]*=1.000001;packet[key]=pose.tolist()
                left_trigger=right_trigger=0.
                if args.finger_test:
                    ui.update(start=tick in (10,260,280,340),release=tick==350)
                    if 20<=tick<80:right_trigger=1.
                    elif 80<=tick<140:left_trigger=1.
                    elif 140<=tick<200:left_trigger=right_trigger=.5
                    elif tick>=260:left_trigger=right_trigger=1.
                    if 320<=tick<335:
                        packet['tracking_valid']=False;packet['envelope']['tracking_valid']=False
                        time.sleep(.04)
                if args.ai_test:
                    # The integration test selects the visible cylinder, then
                    # a real learned candidate. No fixture target substitutes it.
                    perception=read_json(args.output/'ui_state.json',{})
                    if not (args.output/'target_selection.json').exists() and perception.get('objects'):
                        atomic_json(args.output/'target_selection.json',dict(target_id=perception['objects'][0]['target_id'],
                            revision=1,source='explicit_synthetic_AI_software_test',capture_timestamp=stamp,clock_id=bench.clock_id))
                    choices=assist.candidates
                    if ai_choice is None and choices:
                        ai_choice=min(choices,key=lambda c:Rotation.from_matrix(
                            np.asarray(c['base_ee'])[:3,:3]@poses['right'][:3,:3].T).magnitude())
                        ai_test_phase=0;ai_raw_origin=(raw[0].copy(),raw[1].copy())
                    ui.update(clutch=False,start=False,release=False)
                    if ai_test_phase is not None:
                        phase=ai_test_phase
                        ui.update(clutch=phase==2,start=phase==5,select_candidate_id=ai_choice['candidate_id'] if phase==8 else None,
                            confirm=phase==10,assist_held=phase>=80 and phase<160)
                        goal=np.asarray(ai_choice['base_ee'])
                        # Hold the human reference stationary. Any measured
                        # change during assistance is then from the AI path.
                        fraction=0.
                        dest=goal[:3,3]-[.02,0,0]
                        raw[0]=ai_raw_origin[0].copy();raw[1]=ai_raw_origin[1].copy()
                        raw[1][:3,3]+=fraction*(dest-initial['right'][:3,3])
                        desired_rotation=Rotation.from_euler('z',10,degrees=True).as_matrix()@goal[:3,:3]
                        relative=Rotation.from_matrix(desired_rotation@initial['right'][:3,:3].T)
                        raw[1][:3,:3]=Rotation.from_rotvec(relative.as_rotvec()*fraction).as_matrix()@ai_raw_origin[1][:3,:3]
                        packet['left_base_ee']=raw[0].tolist();packet['right_base_ee']=raw[1].tolist()
                        if phase>=175:break
                        ai_test_phase+=1
                packet['source_event_sequences']=dict(arms=tick,head=tick)
                arms=dict(event_kind='controller',sequence=tick,host_sample_time_s=stamp,
                    received_at_host_monotonic=stamp,host_clock_id=bench.clock_id,clock_uncertainty_s=.001,
                    controller_states={side:dict(aButton=False,bButton=False,squeeze=False,triggerValue=value,
                        gamepad=dict(mapping='xr-standard',button_count=6,required_buttons_available=True))
                        for side,value in (('left',left_trigger),('right',right_trigger))})
                packet['vr_buttons']=source_payload(arms,packet)
                latest_test=(packet,ui)
            operator,ui=read_inputs()
            if assist:
                assist.update(read_json(args.output/'latest_proposals.json',{}),ui,support_frame,
                    now=time.monotonic(),clock_id=bench.clock_id,active=session.active)
            q=bench.robot.data.joint_pos[0,bench.arm_ids].cpu().numpy().copy()
            dq=bench.robot.data.joint_vel[0,bench.arm_ids].cpu().numpy().copy()
            command=session.step(operator,ui,(poses['left'],poses['right']),q,dq,
                now=time.monotonic(),clock_id=bench.clock_id)
            installed=False
            if command is not None:
                packet,latest_ui=read_inputs()
                if session.input_is_current(packet,latest_ui,now=time.monotonic(),clock_id=bench.clock_id):
                    target=bench.robot.data.joint_pos.clone()
                    target[0,bench.arm_ids]=target.new_tensor(command)
                    bench.robot.set_joint_position_target(target)
                    installed=True
            if not installed:hold()
            def watchdog():
                nonlocal installed
                packet,latest_ui=read_inputs()
                now=time.monotonic()
                current=session.input_is_current(packet,latest_ui,now=now,clock_id=bench.clock_id) if session.active else False
                if not current:installed=False
                finger_enabled=(current and session.missing_since is None
                    and session.status['reason'] in ('unitree_manual_ik','waiting_for_new_XR_sample')
                    and not any(latest_ui.get(key) for key in ('clutch','start','release')))
                target=bench.robot.data.joint_pos.clone()
                values=fingers.update(target[0].cpu().numpy(),packet,enabled=finger_enabled,
                    now=now,clock_id=bench.clock_id,dt=.005)
                target[0,bench.finger_ids]=target.new_tensor(values[bench.finger_ids])
                if installed:target[0,bench.arm_ids]=target.new_tensor(command)
                bench.robot.set_joint_position_target(target)
            control_work_s=time.monotonic()-control_begin
            physics_begin=time.monotonic()
            contact=bench.step(2 if assist else 5,before_step=watchdog)
            physics_work_s=time.monotonic()-physics_begin
            _,_,_,after=bench.snapshot()
            status=dict(assist.annotate(session.status) if assist else session.status,
                timestamp=time.monotonic(),clock_id=bench.clock_id,
                scripted_test=args.scripted_test,scripted_operator=args.scripted_test,live_ai=bool(assist),
                operator_camera_source=args.operator_camera_source,
                initialization=setup.get('initial_condition_description'),command_consumed=installed,
                manual_baseline=not bool(assist),headset_verified=False,finger_control=fingers.status)
            if not assist:status['candidates']=[]
            atomic_json(args.output/'sim_runtime.json',status)
            row=dict(tick=tick,command_consumed=installed,reason=status['reason'],snapshot=status['snapshot'],
                source_reason=operator.get('reason'),source_sequence=operator.get('source_sequence'),
                tracking_valid=operator.get('tracking_valid'),timestamp=time.monotonic(),
                actual_left_base_ee=after['left'].tolist(),actual_right_base_ee=after['right'].tolist(),
                contact_force_n=contact,loop_s=time.monotonic()-begin,
                camera_work_s=camera_work_s,control_work_s=control_work_s,physics_work_s=physics_work_s,
                camera_publish_work_s=camera_publish_work_s,
                ai_assist=status.get('ai_assist'),assistance_applied=status.get('assistance_applied',False),
                ai_test_phase=ai_test_phase,
                finger_control=fingers.status,
                actual_finger_joints={name:float(bench.robot.data.joint_pos[0,i])
                    for side,hand in fingers.hands.items() for name,i in zip(hand['names'],hand['ids'])},
                ik_wall_s=(worker.last_call_timing or {}).get('client_wall_s'),
                source_age_s=(begin-operator['acquisition_timestamp']
                    if type(operator.get('acquisition_timestamp')) in (float,int) else None))
            rows.append(row)
            # Preserve the first fault even if shutdown itself is interrupted.
            if tick==0 or (len(rows)>1 and (row['reason'],row['snapshot']['teleop_active']) !=
                    (rows[-2]['reason'],rows[-2]['snapshot']['teleop_active'])):
                with (args.output/'transitions.jsonl').open('a') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
            tick+=1
        completed=(ai_test_phase is not None and ai_test_phase>=175) if args.ai_test else (not args.scripted_test or tick==test_ticks)
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        atomic_json(args.output/'runtime_error.json',error)
        raise
    finally:
        if camera_publisher is not None:camera_publisher.close()
        if bench is not None:bench.robot.set_joint_position_target(bench.robot.data.joint_pos.clone())
        if session is not None:
            session.pause('runtime_stopped')
            atomic_json(args.output/'sim_runtime.json',dict(session.status,timestamp=time.monotonic(),
                clock_id=bench.clock_id,scripted_operator=args.scripted_test))
        if dashboard is not None:dashboard.close()
        if proposal_process is not None:
            from run_copilot import stop_children
            stop_children([('learned_proposals',proposal_process)],grace_s=15.)
        if proposal_log is not None:proposal_log.close()
        if worker is not None:worker.close()
        if own_gc_freeze:gc.unfreeze()
        write_audit(args.output/'rows.json',rows)
        maxima={side:max((float(np.linalg.norm(np.asarray(r['actual_'+side+'_base_ee'])[:3,3]-initial[side][:3,3]))
            for r in rows),default=0.) for side in ('left','right')} if bench is not None and rows else {}
        checks=dict(both_arms_moved=bool(maxima) and min(maxima.values())>.01,
            brief_tracking_gap_recovers=any(r['command_consumed'] for r in rows if 52<=r['tick']<60),
            AX_pause_no_commands=not any(r['command_consumed'] for r in rows if 90<=r['tick']<110),
            tracking_loss_pauses=not any(r['command_consumed'] for r in rows if 140<=r['tick']<155),
            release_no_commands=not any(r['command_consumed'] for r in rows if r['tick']>=170))
        accepted=completed and error is None and (not args.scripted_test or all(checks.values()))
        if args.contact_test:
            contacts=[r for r in rows if r['contact_force_n']>2.]
            checks=dict(contact_observed=bool(contacts),commands_during_contact=any(r['command_consumed'] for r in contacts),
                no_contact_declutch=all(r['snapshot']['teleop_active'] and r['snapshot']['clutched']
                    for r in rows if 11<=r['tick']<170),
                commands_after_contact=bool(contacts) and any(r['command_consumed'] and r['tick']>contacts[-1]['tick'] for r in rows),
                release_no_commands=not any(r['command_consumed'] for r in rows if r['tick']>=170))
            accepted=completed and error is None and all(checks.values())
        if args.finger_test:
            def index_at(tick,side):
                row=next((row for row in rows if row['tick']==tick),None)
                return abs(row['actual_finger_joints'][side+'_hand_index_0_joint']) if row else float('inf')
            def no_fingers(start,end):
                return not any(hand['command_applied'] for row in rows if start<=row['tick']<end
                    for hand in row['finger_control']['hands'].values())
            checks=dict(right_closes_independently=.75<index_at(79,'right')<1.05 and index_at(79,'left')<.1,
                left_closes_independently=.75<index_at(139,'left')<1.05 and index_at(139,'right')<.1,
                half_trigger_proportional=all(.35<index_at(199,side)<.55 for side in ('left','right')),
                release_opens=all(index_at(259,side)<.1 for side in ('left','right')),
                AX_pause_holds=no_fingers(260,281),tracking_loss_holds=no_fingers(320,341),
                release_holds=no_fingers(350,360),
                AX_resume_closes=all(.6<index_at(319,side)<1.05 for side in ('left','right')))
            accepted=completed and error is None and all(checks.values())
        if args.ai_test:
            def error_at(phase):
                row=next((r for r in rows if r.get('ai_test_phase')==phase),None)
                if row is None or ai_choice is None:return None
                actual=np.asarray(row['actual_right_base_ee']);goal=np.asarray(ai_choice['base_ee'])
                return dict(position_m=float(np.linalg.norm(actual[:3,3]-goal[:3,3])),
                    rotation_deg=float(np.rad2deg(Rotation.from_matrix(actual[:3,:3]@goal[:3,:3].T).magnitude())))
            before,after=error_at(80),error_at(160)
            checks=dict(real_model_candidate_selected=ai_choice is not None,
                wrist_depth_accepted=any((r.get('ai_assist') or {}).get('wrist_depth',{}).get('accepted') for r in rows),
                assistance_applied=any(r['assistance_applied'] and r['command_consumed'] for r in rows),
                measurable_translation_correction=any((r.get('ai_assist') or {}).get('translation_offset_m',0)>.005 for r in rows),
                measured_hand_moves_toward_learned_position=bool(before and after and before['position_m']-after['position_m']>.003),
                manual_release_stops_correction=not any(r['assistance_applied'] for r in rows if (r.get('ai_test_phase') or 0)>161))
            accepted=completed and error is None and all(checks.values())
        report=dict(completed=completed,accepted=accepted,scripted_test=args.scripted_test,
            custom_motion_guard_enabled=False,control_mode='unitree_manual',checks=checks,
            maximum_TCP_displacement_m=maxima,commands=sum(r['command_consumed'] for r in rows),
            contact_test=args.contact_test,finger_test=args.finger_test,
            ai_test=args.ai_test,live_ai=args.grasp_assist,
            learned_alignment_error_before=before if args.ai_test else None,
            learned_alignment_error_after=after if args.ai_test else None,
            maximum_contact_force_n=max((r['contact_force_n'] for r in rows),default=0.),
            ticks=len(rows),runtime_error=error,headset_verified=False,hardware_authorized=False)
        atomic_json(args.output/'report.json',report)
        atomic_json(args.output.parent/'session.json',dict(cleanup_accepted=error is None,
            control_mode='unitree_manual',report=report))
        app.close(wait_for_replicator=False,skip_cleanup=True)
    return 0 if accepted else 1


if __name__=='__main__':raise SystemExit(main())
