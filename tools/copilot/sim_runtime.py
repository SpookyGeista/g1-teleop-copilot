#!/usr/bin/env python3
"""Interactive, isolated PhysX orientation bench with direct checked q14 handoff.

Uses an explicitly initialized near-grasp scene and a previously inferred,
frozen GraspGenX candidate. No autonomous approach, finger closure or DDS.
--scripted-test is a synthetic-input integration test, never a headset test.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
import resource
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def write_audit(path, value):
    """Atomic run artifact; unlike a control mailbox this may contain all ticks."""
    descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump(value,stream,allow_nan=False)
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def scripted_packet(tick, stamp, clock, left, right):
    # Explicit simulator-only fixture matching the XR wire contract. There is
    # no physical sensor timestamp or claim of real WebXR acquisition.
    stale = 94 <= tick < 101
    acquired = stamp-.4 if stale else stamp
    envelope = dict(schema='punktnull.operator_envelope.v1',sequence=tick,
        acquisition_timestamp=acquired,clock_id=clock,clock_uncertainty_s=.001,
        latency_unknown=False,sample_kind='simulation_pose_sample',tracking_valid=True,
        received_at_host_monotonic=acquired,sensor_capture_timestamp=None,
        timestamp_provenance='synthetic_test_fixture_NO_HEADSET')
    return dict(schema='punktnull.xr_operator.v1',publish_sequence=tick+1,source_sequence=tick,
        published_at=stamp,publication_clock_id=clock,acquisition_timestamp=acquired,clock_id=clock,
        tracking_valid=True,left_base_ee=left.tolist(),right_base_ee=right.tolist(),envelope=envelope,
        expires_at=acquired+.15,command_source='scripted_test_fixture_NO_HEADSET',
        target_frame='original_G1_29_L_ee_R_ee_nominal_base',world_transform_calibrated=False,
        physical_sensor_capture_timestamp=None,robot_transport_started=False)


def scripted_intent(tick, stamp, clock, candidate_id):
    return dict(revision=tick+1,capture_timestamp=stamp,clock_id=clock,client_id='scripted_PhysX_test',
        clutch=tick in (4,74,112),select_candidate_id=candidate_id if tick in (7,77,115) else None,
        confirm=tick in (9,79,117),assist_held=12<=tick<65 or 82<=tick<112 or 120<=tick,
        release=tick in (0,70))


def intent_after_endpoint(prepared, latest, *, live_ai):
    """Use the newly acquired control heartbeat after slow endpoint work.

    Live grasp selection is prepared against a particular UI choice. A new
    selection must first pass that preparation on the next tick; do not ACK it
    against the previous candidate batch. The unchanged handoff recheck sees
    the new mailbox immediately and cancels any older command. Release and
    other action types reach the core without delay. Full input validation
    remains in the core and final handoff; this helper grants no permission
    and creates no timestamps, leases or revisions.
    """
    if (live_ai and isinstance(latest, dict) and
            isinstance(latest.get('select_candidate_id'), str) and
            0 < len(latest['select_candidate_id']) <= 256 and
            not any(latest.get(key) for key in ('release', 'clutch', 'confirm')) and
            (not isinstance(prepared, dict) or
             (latest.get('revision'), latest.get('select_candidate_id')) !=
             (prepared.get('revision'), prepared.get('select_candidate_id')))):
        return prepared
    return latest


def retain_expired_guard_request(directory, raw, timing, tick, interlocks):
    """Retain one exact completed request after a recorded physics lease fault.

    This diagnostic runs only after physics and grants no motion permission.
    It preserves the original request times; it is not a replay command.
    """
    expired = [item for item in interlocks
               if item.get('reason') == 'original_sender_lease_not_current']
    if not expired:
        return None
    if (not isinstance(raw, bytes) or not 1 <= len(raw) <= 2*1024*1024 or
            not isinstance(timing, dict) or timing.get('operation') != 'validate'):
        return dict(retained=False, reason='exact_completed_guard_request_unavailable')
    try:
        request = json.loads(raw)
    except (ValueError, UnicodeError):
        return dict(retained=False, reason='invalid_guard_request_diagnostic')
    if (not isinstance(request,dict) or request.get('operation') != 'validate' or
            request.get('request_id') != timing.get('request_id')):
        return dict(retained=False, reason='guard_request_timing_identity_mismatch')
    name = 'first_lease_expiry_guard_request.json'
    with (Path(directory)/name).open('xb') as stream:
        stream.write(raw)
    return dict(retained=True, path=name, sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw), tick=tick, request_id=request['request_id'],
        expired_interlocks=expired, diagnostic_only=True, timestamps_unchanged=True,
        motion_authorized=False)


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim',action='store_true',help='Required explicit isolated simulation mode')
    parser.add_argument('--initialization',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--operator-mailbox',type=Path)
    source.add_argument('--scripted-test',action='store_true')
    parser.add_argument('--ticks',type=int,default=145)
    parser.add_argument('--duration',type=float,default=180.)
    parser.add_argument('--port',type=int,default=8771)
    parser.add_argument('--live-ai-frames',type=Path,help='Publish frames here and consume current learned proposals from the output directory')
    parser.add_argument('--camera-source',choices=('head','right_wrist','left_wrist'),default='head')
    parser.add_argument('--operator-camera-source',choices=('head','right_wrist','left_wrist'))
    parser.add_argument('--command-format',choices=('float64','unitree_float32'),default='float64')
    parser.add_argument('--sender',choices=('direct','guarded_original'),default='direct')
    parser.add_argument('--repo',type=Path,default=Path('/home/loya2/unitree_sim_isaaclab'))
    parser.add_argument('--xr-repo',default='/home/loya2/xr_teleoperate')
    parser.add_argument('--ik-python',default='/home/loya2/miniconda3/envs/to/bin/python')
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    if not args.sim or args.output.exists() or not 1<=args.port<=65535 or not 1<=args.duration<=3600:
        parser.error('--sim, a new output, valid port and bounded duration are required')
    if not 130<=args.ticks<=1000:parser.error('Scripted test needs 130..1000 ticks')
    if args.sender=='guarded_original' and args.command_format!='unitree_float32':
        parser.error('The guarded original simulator sender requires exact unitree_float32 receipts')
    initialization_content=args.initialization.read_bytes()
    setup=json.loads(initialization_content);candidate=json.loads(args.candidate.read_text())
    provenance=json.loads((args.candidate.parent/'provenance.json').read_text())
    if hashlib.sha256(args.candidate.read_bytes()).hexdigest() != provenance.get('files',{}).get('candidate.json'):
        parser.error('Archived candidate differs from its provenance hash')
    model_source_id='archived_model:'+hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    if setup.get('accepted') is not True or candidate.get('geometry_valid') is not True:
        parser.error('Validated simulation initialization and real model candidate required')
    if setup.get('goal_base_ee') != candidate.get('base_ee'):
        parser.error('Initialization and selected candidate must have the same goal pose')
    args.output.mkdir(parents=True)
    args.enable_cameras=True;args.device='cpu';os.environ['PROJECT_ROOT']=str(args.repo.resolve())
    os.environ['PUNKTNULL_ISAAC_REPOSITORY']=str(args.repo.resolve())
    stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    from teleop_assist.copilot.render_sync import zero_delay_kit_args
    from teleop_assist.copilot.audit_trace import FrozenAuditRows, InputTrace
    args.kit_args=zero_delay_kit_args(args.kit_args)
    app=AppLauncher(args).app
    # AppLauncher installs its own signal handlers during startup. Register
    # afterwards so a launcher deadline reaches our hold/audit/finally path.
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    worker=dashboard=session=bench=feed=actor=codec=None;rows=FrozenAuditRows();completed=False;gc_events=[];gc_started={}
    scenario_tick=0;live_reason=None;script_selection_trials=[]
    own_gc_freeze=False;frozen_objects=0;rss_start_kib=None;max_audit_append_s=0.;runtime_error=None
    def observe_gc(phase, info):
        generation=info['generation']
        if phase=='start':gc_started[generation]=time.monotonic()
        elif generation in gc_started:
            end=time.monotonic();begin=gc_started.pop(generation)
            gc_events.append(dict(generation=generation,started=begin,finished=end,elapsed_s=end-begin,
                collected=info.get('collected'),uncollectable=info.get('uncollectable')))
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation
        from PIL import Image
        from teleop_assist.copilot.physx_bench import PhysXOrientationBench
        from teleop_assist.copilot.motion_process import MotionProcess
        from teleop_assist.copilot.sim_control import SimControlSession,SimUIIntent
        from teleop_assist.copilot.sim_actor import SimReceiptActor
        from teleop_assist.copilot.types import ProposalBatch,GraspCandidate
        from teleop_assist.copilot.mailbox import atomic_json,read_json
        from teleop_assist.copilot.dashboard import Dashboard
        from teleop_assist.copilot.preview import HandPreview
        from teleop_assist.copilot.collision import GraspCollisionFilter
        from teleop_assist.copilot.frames import preview_uv,publish_frame
    except BaseException:
        app.close(wait_for_replicator=False,skip_cleanup=True)
        raise
    status={}
    try:
        bench=PhysXOrientationBench(args.repo,args.xr_repo,setup,camera_source=args.camera_source,
            operator_camera_source=args.operator_camera_source)
        worker=MotionProcess(args.ik_python,args.xr_repo,rejection_directory=args.output/'motion_rejections',
            command_format=args.command_format)
        atomic_json(args.output/'motion_guard_manifest.json',worker.manifest)
        session=SimControlSession(worker)
        if args.command_format == 'unitree_float32':
            from teleop_assist.copilot.unitree_codec import UnitreeSimArmCodec
            codec=UnitreeSimArmCodec(simulation_mode=True)
        actor=SimReceiptActor(session,bench,codec=codec)
        command_samples=bench
        if args.sender=='guarded_original':
            from teleop_assist.copilot.original_sim_runtime import OriginalSimRuntime
            actor=OriginalSimRuntime(session,bench,args.xr_repo,codec=codec)
            command_samples=actor
        from dataclasses import asdict
        from teleop_assist.copilot.source_evidence import source_manifest
        source=source_manifest(ROOT,'isolated_PhysX_runtime')
        source.update(controller_configuration=asdict(session.controller.config),
            initial_condition_path=str(args.initialization.resolve()),
            initial_condition_sha256=hashlib.sha256(initialization_content).hexdigest(),
            ui_action_policy=session.ui_action_policy,
            session_configuration=asdict(session.config),render_settings=bench.render_sync,
            command_format=args.command_format,command_codec=codec.manifest if codec else None,
            sender=args.sender,sender_manifest=getattr(actor,'manifest',None),
            external_configuration_sources=bench.external_source_manifest,
            geometry_limitations=bench.open_geometry_limitations)
        atomic_json(args.output/'runtime_source_manifest.json',source)
        dashboard=Dashboard(args.output,port=args.port);dashboard.start()
        print(f'Isolated PhysX controls: http://127.0.0.1:{args.port}/sim',flush=True)
        preview=HandPreview(GraspCollisionFilter(ROOT/'teleop_assist/copilot/assets/dex3_right/config.json'))
        feed=None
        if args.live_ai_frames:
            from teleop_assist.copilot.sim_target_feed import SimTargetFeed
            descriptor_hash=hashlib.sha256((ROOT/'teleop_assist/copilot/assets/dex3_right/config.json').read_bytes()).hexdigest()
            feed=SimTargetFeed(args.output,worker,descriptor_hash,policy='complete_static_sim_orientation')
        goal=np.asarray(candidate['base_ee']);raw_left=np.asarray(setup['left_base_ee']);raw_right=np.asarray(setup['raw_right_base_ee'])
        initial_frame,_,_,_=bench.snapshot()
        reference_base=initial_frame['T_world_base'];invalid_scene=None
        candidate_ui=dict(candidate_id=candidate['candidate_id'],model_score=candidate['model_score'],
            source=candidate['source'],base_ee=candidate['base_ee'])
        candidates_ui=[] if feed is not None else [candidate_ui]
        script_candidate=None;script_object_selected=False;tried_candidates=set()
        script_pending_candidate=None
        scenario_tick=0;scenario_started=feed is None;live_reason=None;live_evidence_saved=False
        # A measured generation-2 scan of the warmed Isaac heap paused the
        # handoff for 147 ms. Keep this long-lived startup graph out of scans;
        # normal cyclic GC remains enabled for newly allocated control data.
        # This is a bounded, private simulator process, never a real-time claim.
        gc.collect()
        if gc.get_freeze_count()==0:
            gc.freeze();own_gc_freeze=True
        frozen_objects=gc.get_freeze_count()
        rss_start_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if args.sender=='guarded_original':
            # The original constructor consumed a bootstrap acquisition before
            # potentially slow startup. Advance one real measured-HOLD step;
            # do not retimestamp that old generation into a command lease.
            actor.sink.idle_hold()
            bench.step(1,before_step=actor.before_physics_step)
        started=time.monotonic();tick=0;previous_step_at=None;lease_fault_recorded=False
        gc.callbacks.append(observe_gc)
        while not stop and time.monotonic()-started<args.duration and (not args.scripted_test or scenario_tick<args.ticks):
            loop_started=time.monotonic();loop_cpu_started=time.process_time()
            prepared_preview_evidence=prepare_rejection_evidence=handoff_target_recheck=None
            frame_barrier=None
            frame_synchronization_s=0.
            captured_frame=None
            preview_selection=None
            vr_ingestion=None
            if feed is not None or args.operator_camera_source is not None or tick%4==0:
                frame=bench.camera_frame()
                if args.operator_camera_source is not None:
                    from teleop_assist.copilot.xr_camera import publish_operator_frame
                    operator_frame = (frame if args.operator_camera_source == args.camera_source else
                                      bench.camera_frame(source=args.operator_camera_source))
                    publish_operator_frame(args.output/'operator_camera', operator_frame)
                if feed is None:
                    temporary=args.output/'frame.next.jpg';Image.fromarray(frame['rgb']).save(temporary,quality=85)
                    temporary.replace(args.output/'frame.jpg')
                    candidate_ui['preview_uv']=preview_uv(candidate['world_canonical'],frame)
                    candidate_ui['preview_hand_polygons']=preview.project(candidate['world_canonical'],frame)
                else:
                    # This scene belongs to the just-captured rendered image;
                    # camera_frame already sampled it before any physics step.
                    camera_scene=json.loads(frame['simulation_scene_json'])
                    feed.record_frame(frame,camera_scene)
                    publish_frame(args.live_ai_frames,frame)
                    captured_frame=frame
            camera_finished=time.monotonic();camera_cpu_finished=time.process_time()
            frame,state,scene,poses=bench.snapshot();stamp=frame['timestamp']
            target=np.asarray(scene['obstacles'][1]['world_pose'])
            displacement=float(np.linalg.norm(target[:3,3]-[.4,-.18,.9]))
            rotation=float(Rotation.from_matrix(target[:3,:3]).magnitude())
            velocity=np.asarray(scene['observed_target_velocity'])
            if (displacement>.001 or rotation>.01 or np.linalg.norm(velocity[:3])>.01
                    or np.linalg.norm(velocity[3:])>.02
                    or not np.allclose(frame['T_world_base'],reference_base,atol=1e-6,rtol=0)):
                invalid_scene='fixture_target_or_base_moved'
            if bench.settling_contact_n>2.:invalid_scene='initial_contact_exceeds_threshold'
            # Source time is retained for a real mailbox. New synthetic samples
            # are generated only in explicitly labelled integration-test mode.
            now=time.monotonic()
            if args.scripted_test:
                operator={}
                if feed is not None and not scenario_started:
                    intent=dict(revision=tick+1,capture_timestamp=now,clock_id=bench.clock_id,client_id='scripted_PhysX_test')
                    ui=read_json(args.output/'ui_state.json',{})
                    if not script_object_selected and ui.get('objects'):
                        atomic_json(args.output/'target_selection.json',dict(target_id=ui['objects'][0]['target_id'],
                            revision=1,clock_id=bench.clock_id,capture_timestamp=now,
                            source='explicit_synthetic_integration_test_selection'))
                        script_object_selected=True
                    if script_candidate is None:
                        # Choose from exactly the compact list presented by
                        # the feed, not an undisplayed raw model candidate.
                        eligible=[item for item in candidates_ui if item['candidate_id'] not in tried_candidates]
                        if eligible:
                            # Prespecified synthetic test choice: smallest
                            # orientation change among the offered clear grasps.
                            selected=min(eligible,key=lambda c:Rotation.from_matrix(np.asarray(c['base_ee'])[:3,:3]@poses['right'][:3,:3].T).magnitude())
                            script_candidate=selected['candidate_id'];intent['select_candidate_id']=script_candidate
                            tried_candidates.add(script_candidate)
                else:
                    intent=scripted_intent(scenario_tick,now,bench.clock_id,script_candidate or candidate['candidate_id'])
                    intent['revision']=tick+1
                    if feed is not None and scenario_tick in (8,78,116):intent['clutch']=True
            else:
                try:operator=read_json(args.operator_mailbox,{})
                except (OSError,ValueError):operator={}
                vr_ingestion=dashboard.consume_vr(operator,runtime_status=status,now=time.monotonic())
                try:intent=read_json(args.output/'sim_control.json',{})
                except (OSError,ValueError):intent={}
            if feed is None:
                # Refresh actual endpoint evidence, including measured fingers
                # and the other arm, against the complete current sim scene.
                endpoint=worker.call('validate_sim_candidate',base_ee=goal.tolist(),left_base_ee=poses['left'].tolist(),
                    robot_state=state,scene=scene)
                if tick==0:atomic_json(args.output/'initial_endpoint_evidence.json',endpoint)
                accepted=endpoint['accepted'] and invalid_scene is None
                chosen=GraspCandidate(candidate['candidate_id'],goal,candidate['model_score'],True,
                    endpoint['ik']['ik_valid'],accepted,candidate['source'])
                batch=ProposalBatch(target_id='fixed_sim_cylinder',observation_id=f'current_sim_endpoint:{tick}',
                    capture_timestamp=stamp,clock_id=bench.clock_id,calibration_id=bench.calibration_id,
                    candidates=(chosen,),motion_static=invalid_scene is None,valid=accepted,
                    model_source_id=model_source_id,
                    model_source_capture_timestamp=provenance['model_source_capture_timestamp'],
                    model_source_clock_id=provenance['model_source_clock_id'])
            else:
                bundle=feed.prepare(intent=intent,session_status=session.status,robot_state=state,scene=scene,
                    left_base_ee=poses['left'],now=time.monotonic(),clock_id=bench.clock_id,calibration_id=bench.calibration_id,
                    current_frame=captured_frame,cancelled=lambda:stop)
                frame_barrier=bundle.get('frame_barrier')
                frame_synchronization_s=bundle.get('frame_synchronization_s',0.)
                prepared_preview_evidence=bundle.get('preview_evidence')
                prepare_rejection_evidence=bundle.get('rejection_evidence')
                candidates_ui=bundle['candidates'];live_reason=bundle['reason']
                preview_selection=bundle.get('preview_selection')
                if args.scripted_test and not scenario_started and intent.get('select_candidate_id'):
                    script_pending_candidate=intent['select_candidate_id']
                if args.scripted_test and not scenario_started and script_pending_candidate is not None and preview_selection is None:
                    # Synthetic-only trial accounting follows its endpoint
                    # reply on the NEXT tick; a real user's old click is never
                    # replayed or retried by this fixture bookkeeping.
                    script_selection_trials.append(dict(candidate_id=script_pending_candidate,
                        reason=live_reason,accepted=bundle['batch'] is not None,
                        selection_policy='synthetic_test_only_next_smallest_orientation_among_displayed_untried_candidates'))
                    trial_directory=args.output/'selection_trials';trial_directory.mkdir(exist_ok=True)
                    write_audit(trial_directory/f'trial_{len(script_selection_trials):03d}.json',dict(
                        candidate_id=script_pending_candidate,reason=live_reason,
                        robot_state=state,scene=scene,left_base_ee=poses['left'].tolist(),
                        endpoint_evidence=bundle['endpoint_evidence']))
                    if bundle['batch'] is None:script_candidate=None
                    script_pending_candidate=None
                if bundle['reset_control']:session.hold('deliberate_live_grasp_reselection',now=time.monotonic(),clock_id=bench.clock_id)
                batch=bundle['batch']
                accepted=batch is not None and invalid_scene is None
                if bundle['goal'] is not None:goal=np.asarray(bundle['goal'])
                if accepted and not live_evidence_saved:
                    atomic_json(args.output/'live_endpoint_evidence.json',bundle['endpoint_evidence']);live_evidence_saved=True
                if accepted and args.scripted_test and not scenario_started:
                    scenario_started=True;scenario_tick=-1
                if preview_selection is not None and invalid_scene is None:
                    batch=preview_selection
                elif not accepted:
                    batch=ProposalBatch('pending_live_target','no_execution_evidence',stamp,bench.clock_id,
                        bench.calibration_id,(),False,False)
            endpoint_finished=time.monotonic();endpoint_cpu_finished=time.process_time()
            # Capture the actual robot/scene before reading the newest input.
            # Snapshot acquisition does not depend on operator intent and can
            # otherwise spend part of an already sampled XR lease. No physics
            # step occurs between this capture, checking and command use.
            _,current_state,current_scene,poses=command_samples.capture_command_sample()
            if preview_selection is not None:
                preview_current=feed.recheck_preview_selection(preview_selection,
                    now=time.monotonic(),clock_id=bench.clock_id,calibration_id=bench.calibration_id)
                if not preview_current['accepted']:
                    live_reason='preview_selection_recheck_rejected:'+preview_current['reason']
                    # Pass no authorization to the ordinary session failure
                    # path. The concurrent UI edge is still consumed once.
                    batch=None
            # Neither endpoint nor snapshot work consumes human input. Read a
            # genuine latest acquisition now; never retimestamp an old packet.
            # The labelled synthetic source also samples only at this point.
            if args.scripted_test:
                operator=scripted_packet(scenario_tick,time.monotonic(),bench.clock_id,raw_left,raw_right)
                operator.update(publish_sequence=tick+1,source_sequence=tick)
                operator['envelope']['sequence']=tick
            else:
                try:operator=read_json(args.operator_mailbox,{})
                except (OSError,ValueError):operator={}
                # The UI heartbeat carries the same original VR event lease.
                # Refresh it along with the pose after endpoint work; keeping
                # the earlier UI sample needlessly spends one input frame of
                # the receipt lifetime. One-shot live selection still needs
                # the matching endpoint preparation on the next tick.
                vr_ingestion=dashboard.consume_vr(operator,runtime_status=status,now=time.monotonic())
                try:latest_step_intent=read_json(args.output/'sim_control.json',{})
                except (OSError,ValueError):latest_step_intent={}
                # A staged non-motion selection must see a newer release or
                # conflicting UI unchanged. Only its exact original edge can
                # commit this preview; never conceal a later action behind it.
                intent=(latest_step_intent if preview_selection is not None else
                        intent_after_endpoint(intent,latest_step_intent,live_ai=feed is not None))
            input_trace = None if args.scripted_test else InputTrace(bench.clock_id)
            if input_trace is not None:
                input_trace.capture(operator,intent,phase='session_step',captured_at=time.monotonic())
            session_step_at=time.monotonic()
            offer=session.step(operator,intent,current_state,current_scene,batch,
                measured_left_base_ee=poses['left'],measured_right_base_ee=poses['right'],
                now=session_step_at,clock_id=bench.clock_id,calibration_id=bench.calibration_id)
            if preview_selection is not None:
                feed.finish_preview_selection(offer)
            guard_finished=time.monotonic();guard_cpu_finished=time.process_time()
            guard_worker_timing=(worker.last_call_timing if preview_selection is None and
                (worker.last_call_timing or {}).get('operation')=='validate' else None)
            latest_operator,latest_intent=operator,intent
            if not args.scripted_test:
                try:latest_operator=read_json(args.operator_mailbox,{})
                except (OSError,ValueError):latest_operator={}
                vr_ingestion=dashboard.consume_vr(latest_operator,runtime_status=status,now=time.monotonic())
                try:latest_intent=read_json(args.output/'sim_control.json',{})
                except (OSError,ValueError):latest_intent={}
            if input_trace is not None and offer.get('allowed') is True:
                input_trace.capture(latest_operator,latest_intent,phase='handoff_recheck',captured_at=time.monotonic())
            handoff_ok=offer.get('allowed') is True and session.recheck_inputs(latest_operator,latest_intent,
                now=time.monotonic(),clock_id=bench.clock_id,execution_reserve_s=.02)
            recheck_finished=time.monotonic()
            try:
                _,handoff_state,handoff_scene,_=command_samples.read_command_sample()
            except ValueError as error:
                session.hold(str(error),now=time.monotonic(),clock_id=bench.clock_id)
                handoff_ok=False
            snapshot_finished=time.monotonic()
            if handoff_ok and feed is not None:
                latest_target=feed.recheck(session_status=session.status,now=time.monotonic(),
                    clock_id=bench.clock_id,calibration_id=bench.calibration_id)
                handoff_target_recheck=latest_target
                if not latest_target['accepted']:
                    session.hold(latest_target['reason'],now=time.monotonic(),clock_id=bench.clock_id)
                    handoff_ok=False
            receipt=(session.consume_receipt(offer,now=time.monotonic(),clock_id=bench.clock_id,
                measured_state=handoff_state,scene=handoff_scene) if handoff_ok else None)
            consumed_at=time.monotonic()
            input_consumer='actor_write'; input_substep=None
            def final_inputs():
                if args.scripted_test:return operator,intent
                try: current_operator=read_json(args.operator_mailbox,{})
                except (OSError,ValueError):current_operator={}
                vr_ingestion=dashboard.consume_vr(current_operator,runtime_status=status,now=time.monotonic())
                try:current_intent=read_json(args.output/'sim_control.json',{})
                except (OSError,ValueError):current_intent={}
                input_trace.capture(current_operator,current_intent,phase='actor_interlock',
                    captured_at=time.monotonic(),consumer=input_consumer,substep_index=input_substep)
                return current_operator,current_intent
            def final_target():
                return feed.recheck(session_status=session.status,now=time.monotonic(),
                    clock_id=bench.clock_id,calibration_id=bench.calibration_id)
            actor_result=actor.apply(receipt,read_inputs=final_inputs,
                recheck_target=final_target if feed is not None else None)
            q14=receipt.q14 if actor_result['applied'] else None
            controller_at_handoff=deepcopy(session.status)
            actor_finished=time.monotonic();actor_cpu_finished=time.process_time()
            def audited_physics_step():
                nonlocal input_consumer,input_substep
                input_consumer='physics_substep'
                input_substep=0 if input_substep is None else input_substep+1
                actor.before_physics_step()
            before_step=(audited_physics_step if input_trace is not None else actor.before_physics_step) if args.sender=='guarded_original' else None
            contact=bench.step(5,before_step=before_step)
            if args.sender=='guarded_original':
                actor_result['physics_step_interlocks']=list(actor.physics_interlocks)
            physics_finished=time.monotonic();physics_cpu_finished=time.process_time()
            lease_fault_evidence=None
            if not lease_fault_recorded:
                lease_fault_evidence=retain_expired_guard_request(args.output,
                    getattr(worker,'last_call_request_bytes',None),guard_worker_timing,tick,
                    actor_result.get('physics_step_interlocks',[]))
                lease_fault_recorded=lease_fault_evidence is not None
            if contact>2.:invalid_scene='contact_exceeds_threshold'
            after_frame,_,_,after=bench.snapshot()
            row=dict(tick=tick,scenario_tick=scenario_tick if scenario_started else None,
                command_consumed=q14 is not None,endpoint_accepted=accepted,live_proposal_reason=live_reason,
                vr_button_ingestion=vr_ingestion,
                source_sequence=operator.get('source_sequence'),scene_error=invalid_scene,
                input_trace=input_trace.snapshot() if input_trace is not None else None,
                prepared_preview_evidence=prepared_preview_evidence,
                frame_barrier=frame_barrier,
                prepare_rejection_evidence=prepare_rejection_evidence,
                handoff_target_recheck=handoff_target_recheck,
                final_actor=actor_result,
                expired_guard_request_evidence=lease_fault_evidence,
                orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(after['right'][:3,:3]@goal[:3,:3].T).magnitude())),
                actual_left_base_ee=after['left'].tolist(),actual_right_base_ee=after['right'].tolist(),
                actual_joint_names=after_frame['joint_names'].tolist(),actual_q43=after_frame['joint_positions'].tolist(),
                contact_force_n=contact,controller=controller_at_handoff,
                post_physics_controller=deepcopy(session.status),timing=dict(
                    loop_started=loop_started,session_step_at=session_step_at,
                    session_step_interval_s=session_step_at-previous_step_at if previous_step_at is not None else None,
                    camera_preview_s=camera_finished-loop_started,
                    camera_cpu_s=camera_cpu_finished-loop_cpu_started,
                    endpoint_and_snapshot_cpu_s=endpoint_cpu_finished-camera_cpu_finished,
                    controller_and_guard_cpu_s=guard_cpu_finished-endpoint_cpu_finished,
                    handoff_cpu_s=actor_cpu_finished-guard_cpu_finished,
                    physics_cpu_s=physics_cpu_finished-actor_cpu_finished,
                    actor_s=actor_finished-consumed_at,physics_s=physics_finished-actor_finished,
                    guard_worker=guard_worker_timing,
                    endpoint_and_snapshot_s=endpoint_finished-camera_finished-frame_synchronization_s,
                    frame_barrier_s=frame_synchronization_s,
                    controller_and_guard_s=guard_finished-endpoint_finished,
                    source_recheck_s=recheck_finished-guard_finished,
                    handoff_snapshot_s=snapshot_finished-recheck_finished,
                    consume_s=consumed_at-snapshot_finished,
                    handoff_and_physics_s=time.monotonic()-guard_finished,
                    loop_s=time.monotonic()-loop_started),
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            previous_step_at=session_step_at
            status=dict(session.status)
            status.update(schema='punktnull.sim_runtime.v1',mode='isolated_PhysX',timestamp=time.monotonic(),
                clock_id=bench.clock_id,candidates=candidates_ui,live_ai=feed is not None,live_proposal_reason=live_reason,
                perception_camera_source=args.camera_source,operator_camera_source=args.operator_camera_source,
                scripted_operator=args.scripted_test,scripted_test=args.scripted_test,
                headset_verified=False,robot_transport_started=False,vr_button_ingestion=vr_ingestion,
                command_format=args.command_format,command_codec_scope='local_IDL_roundtrip' if codec else None,
                scene_error=invalid_scene,orientation_error_deg=row['orientation_error_deg'],
                guard_allowed=q14 is not None,model_source='live_GraspGenX_with_independent_sim_evidence' if feed is not None else 'frozen_actual_GraspGenX_inference_pipeline_003',
                initialization=setup.get('initial_condition_description',
                    'explicit_near_grasp_simulation_pose; no_home_approach'),
                target_observation='fresh_camera_observation_and_complete_sim_scene' if feed is not None else 'fresh_complete_simulator_scene; not_live_semantic_reinference')
            atomic_json(args.output/'sim_runtime.json',status)
            # Include dashboard publication in loop timing. Final immutable
            # encoding is separately measured; the next step interval includes
            # it as well as all time between iterations.
            row['timing'].update(loop_s=time.monotonic()-loop_started,
                loop_cpu_s=time.process_time()-loop_cpu_started,
                loop_measurement_ends='before_final_immutable_audit_append')
            audit_started=time.monotonic()
            rows.append(row)
            max_audit_append_s=max(max_audit_append_s,time.monotonic()-audit_started)
            tick+=1
            if scenario_started:scenario_tick+=1
        completed=True
    except BaseException as error:
        # Kit shutdown may terminate before Python emits the pending exception.
        # Preserve the original failure before entering simulator cleanup.
        runtime_error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        write_audit(args.output/'runtime_error.json',runtime_error)
        raise
    finally:
        if observe_gc in gc.callbacks:gc.callbacks.remove(observe_gc)
        sender_close_error=None
        try:
            if args.sender=='guarded_original' and actor is not None and hasattr(actor,'close'):
                try:actor.close()
                except Exception as error:
                    sender_close_error=str(error) or type(error).__name__
                    completed=False
            if session is not None:
                status.update(session.hold('runtime_stopped',now=time.monotonic(),clock_id=bench.clock_id))
                status.update(guard_allowed=False,assistance_applied=False)
                atomic_json(args.output/'sim_runtime.json',status)
            if bench:
                rows.write(args.output/'rows.json')
                write_audit(args.output/'gc_events.json',gc_events)
                atomic_json(args.output/'report.json',dict(completed=completed,ticks=len(rows),
                    commands_consumed=rows.commands_consumed,
                    max_contact_force_n=rows.max_contact_force_n,
                    max_audit_append_s=max_audit_append_s,
                    gravity_saturation_steps=bench.gravity_saturation_steps,
                    motion_worker_pid=worker.process.pid if worker else None,
                    render_settings=bench.render_sync,
                    controller_configuration=vars(session.controller.config) if session else None,
                    ui_action_policy=session.ui_action_policy if session else None,
                    startup_heap_frozen_objects=frozen_objects,cyclic_gc_enabled=gc.isenabled(),
                    max_rss_start_kib=rss_start_kib,max_rss_end_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    scripted_operator=args.scripted_test,headset_verified=False,real_robot_tested=False,
                    live_ai=feed is not None,scripted_scenario_completed=args.scripted_test and scenario_tick>=args.ticks,
                    camera_source=args.camera_source,
                    command_format=args.command_format,command_codec=codec.manifest if codec else None,
                    sender=args.sender,sender_manifest=getattr(actor,'manifest',None),
                    external_configuration_sources=bench.external_source_manifest,
                    geometry_limitations=bench.open_geometry_limitations,
                    sender_close_error=sender_close_error,
                    runtime_error=runtime_error,
                    DDS_transport_started=False,
                    transport_modules_loaded=[n for n in sys.modules if n.startswith(('unitree_sdk2py.core.channel','dds.'))],
                    preview_collision_policy='complete_static_sim_orientation' if feed is not None else 'frozen_fixture_with_current_full_sim_checks',
                    live_proposal_reason=live_reason,
                    synthetic_selection_trials=script_selection_trials,
                    dds_imported=any(n.startswith('unitree_sdk2py') or n.startswith('dds.') for n in sys.modules),
                    candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                    candidate_source=str(args.candidate.resolve()),physics_step_s=.005,physics_steps_per_control_tick=5,
                    physical_contact_sensor_scope='16 right arm/hand/wrist-camera bodies vs table and target only',
                    limitations=['single rigid stationary known simulation object','explicit initialized near-grasp pose',
                        'fixed measured fingers','no grasp/lift','upstream PhysX self-collision disabled']))
        finally:
            if own_gc_freeze:gc.unfreeze()
            try:
                if dashboard:dashboard.close()
            finally:
                try:
                    if worker:worker.close()
                finally:
                    app.close(wait_for_replicator=False,skip_cleanup=True)
    if sender_close_error is not None:
        raise SystemExit('Measured simulator hold failed during shutdown; inspect report.json')
    if args.scripted_test and (not completed or scenario_tick < args.ticks):
        raise SystemExit('Synthetic acceptance sequence did not finish; inspect report.json')


if __name__=='__main__':main()
