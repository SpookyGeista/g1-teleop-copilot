"""Non-actuating proposal service shared by live preview and offline review.

Inference and endpoint checks run off the camera/UI thread. A confirmed proposal
never follows a new diffusion result. Live output remains a preview until a
separate guard validates the actual arm command, current robot and full scene.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import time
import tempfile
import numpy as np
from PIL import Image
from .collision import GraspCollisionFilter
from .frames import geometry_observation, robot_request, preview_uv
from .kinematics import isolated_validate
from .mailbox import atomic_json, read_json, host_clock_id
from .stationarity import ObservedStationarity
from .preview import HandPreview


def write_proposal_audit(path, value):
    """Bounded scientific artifact; never a control mailbox."""
    payload=json.dumps(value,allow_nan=False).encode()
    if len(payload)>64*1024*1024:raise ValueError('Proposal audit exceeds 64 MiB')
    descriptor,temporary=tempfile.mkstemp(prefix='.proposal-audit-',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as stream:stream.write(payload)
        os.replace(temporary,path)
    finally:Path(temporary).unlink(missing_ok=True)


def write_failed_observation(directory, observation, frame, evidence):
    """CPU-only exact replay input. Compression never runs on the UI thread."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    # Inputs were copied when the failure occurred; never substitute a newer
    # camera image while this independent audit worker is waiting to run.
    if observation is not None:
        np.savez_compressed(directory/'observation.npz',**observation.arrays())
    np.savez_compressed(directory/'frame.npz',**frame)
    files={name:dict(sha256=hashlib.sha256((directory/name).read_bytes()).hexdigest())
        for name in (('observation.npz','frame.npz') if observation is not None else ('frame.npz',))}
    atomic_json(directory/'evidence.json',dict(evidence,files=files,
        scope='failed_observed_geometry_replay_only' if observation is not None else 'failed_camera_frame_without_target_observation',
        motion_authorized=False))
    return str(directory)


SELECTED_REPLAY_MAX_BYTES=16*1024*1024
SELECTED_REPLAY_METADATA_MAX_BYTES=64*1024


def freeze_selected_failure_inputs(frame, tracking_failure):
    """Copy exact bounded local inputs before an asynchronous writer can run."""
    if not isinstance(frame,dict) or len(frame)>64:
        raise ValueError('Selected replay frame field limit')
    arrays={};other={};total=0
    for key,value in frame.items():
        if not isinstance(key,str) or not key.isidentifier() or len(key)>128 or key=='_nonarray_metadata_json':
            raise ValueError('Selected replay frame key invalid')
        if isinstance(value,dict):
            other[key]=value;continue
        if isinstance(value,str) and len(value)>SELECTED_REPLAY_METADATA_MAX_BYTES:
            raise ValueError('Selected replay string limit')
        array=np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError('Selected replay object arrays prohibited')
        total+=array.nbytes
        if total>SELECTED_REPLAY_MAX_BYTES:raise ValueError('Selected replay byte limit')
        arrays[key]=array
    def numeric_metadata(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError('Unsupported selected replay metadata')
    encode=lambda value:json.dumps(value,allow_nan=False,default=numeric_metadata)
    other_json=encode(other)
    if len(other_json.encode())>SELECTED_REPLAY_METADATA_MAX_BYTES:
        raise ValueError('Selected replay metadata limit')
    tracking_arrays={};tracking_metadata=None
    if tracking_failure is not None:
        tracking_metadata={key:value for key,value in tracking_failure.items() if key!='arrays'}
        supplied=tracking_failure.get('arrays',{})
        if not isinstance(supplied,dict) or len(supplied)>5:
            raise ValueError('Selected replay tracking field limit')
        for key,value in supplied.items():
            if key not in ('previous_gray','previous_feature_mask','corner_points','previous_rgb','previous_depth','previous_mask') or not isinstance(value,np.ndarray):
                raise ValueError('Selected replay tracking arrays invalid')
            if value.dtype.hasobject:raise ValueError('Selected replay tracking object arrays prohibited')
            total+=value.nbytes
            if total>SELECTED_REPLAY_MAX_BYTES:raise ValueError('Selected replay byte limit')
            tracking_arrays[key]=value
    tracking_json=encode(tracking_metadata)
    if len(tracking_json.encode())>SELECTED_REPLAY_METADATA_MAX_BYTES:
        raise ValueError('Selected replay tracking metadata limit')
    if total+len(other_json.encode())+len(tracking_json.encode())>SELECTED_REPLAY_MAX_BYTES:
        raise ValueError('Selected replay byte limit')
    # Budget validation precedes copies; never resize/downsample 'exact' data.
    frozen={key:value.copy() for key,value in arrays.items()}
    if other:frozen['_nonarray_metadata_json']=np.asarray(other_json)
    copied_tracking={key:value.copy() for key,value in tracking_arrays.items()}
    return frozen,copied_tracking,tracking_json,total


def write_selected_target_failure(directory, frame, tracking_arrays, tracking_json, evidence, *,
                                  scope='first_selected_target_failure_exact_local_replay_only'):
    """One extra selected-loss replay, with exclusive creation and no authority."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(directory/'frame.npz',**frame)
    np.savez_compressed(directory/'tracking_inputs.npz',metadata_json=np.asarray(tracking_json),**tracking_arrays)
    files={name:dict(sha256=hashlib.sha256((directory/name).read_bytes()).hexdigest(),
                    bytes=(directory/name).stat().st_size) for name in ('frame.npz','tracking_inputs.npz')}
    atomic_json(directory/'evidence.json',dict(evidence,files=files,
        scope=scope,motion_authorized=False))
    return files


def write_selected_feature_failure(directory, frame, tracking_arrays, tracking_json, evidence):
    """Independent one-shot feature diagnostic; not evidence of target loss."""
    return write_selected_target_failure(directory,frame,tracking_arrays,tracking_json,evidence,
        scope='first_selected_feature_failure_exact_local_replay_only')


class FrameTimingTrace:
    """Keep bounded selected-phase timing evidence; write only off-thread."""
    def __init__(self,max_records=5000,max_bytes=5*1024*1024):
        self.max_records,self.max_bytes=max_records,max_bytes
        self.lines=[];self.bytes=0;self.dropped=0;self.future=None;self.write_error=None

    def append(self,record):
        line=(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n').encode()
        if len(self.lines)>=self.max_records or self.bytes+len(line)>self.max_bytes:
            self.dropped+=1;return
        self.lines.append(line);self.bytes+=len(line)

    def publish(self,pool,directory,*,force=False):
        if not self.lines:return
        if not force and (len(self.lines)%64 or self.future is not None and not self.future.done()):return
        lines=tuple(self.lines)
        metadata=dict(schema='punktnull.selected_frame_timing_trace.v1',records=len(lines),
            bytes=self.bytes,dropped=self.dropped,max_records=self.max_records,max_bytes=self.max_bytes,
            scope='service_observation_and_publication_timing_only',motion_authorized=False)
        self.future=pool.submit(self._write_checked,Path(directory),lines,metadata)

    def status(self):
        # Reading a completed Future is nonblocking; keep failure evidence even
        # if a later snapshot succeeds, instead of silently dropping audit loss.
        if self.future is not None and self.future.done():
            failure=self.future.exception()
            if failure:self.write_error=type(failure).__name__
        state=('failed' if self.write_error else 'empty' if self.future is None
            else 'complete' if self.future.done() else 'pending')
        return dict(status=state,write_error=self.write_error,records=len(self.lines),
            bytes=self.bytes,dropped=self.dropped,motion_authorized=False)

    def _write_checked(self,directory,lines,metadata):
        try:self._write(directory,lines,metadata)
        except Exception as exc:
            self.write_error=type(exc).__name__
            raise

    @staticmethod
    def _write(directory,lines,metadata):
        temporary=directory/'frame_timing.next.jsonl'
        with temporary.open('wb') as stream:stream.writelines(lines)
        temporary.replace(directory/'frame_timing.jsonl')
        atomic_json(directory/'frame_timing_meta.json',metadata)


class ProposalEngine:
    def __init__(self, grasp, descriptor, *, xr_repository, ik_python, audit_directory=None,
                 complete_static_sim_preview=False):
        self.grasp = grasp
        self.collision = GraspCollisionFilter(descriptor)
        self.preview = HandPreview(self.collision)
        self.repository, self.ik_python = xr_repository, ik_python
        self.audit_directory = Path(audit_directory) if audit_directory is not None else None
        self.complete_static_sim_preview=complete_static_sim_preview

    def infer(self, observation, frame):
        input_evidence = None
        if self.audit_directory is not None:
            # Runs on the proposal worker, never the command handoff thread.
            # Retain the exact source images/mask/times for rejected runs too.
            key=hashlib.sha256((observation.observation_id+'|'+observation.clock_id).encode()).hexdigest()[:20]
            directory=self.audit_directory/'model_inputs'/key
            directory.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(directory/'observation.npz',**observation.arrays())
            np.savez_compressed(directory/'frame.npz',**frame)
            input_evidence={name:dict(path=str((directory/name).relative_to(self.audit_directory)),
                sha256=hashlib.sha256((directory/name).read_bytes()).hexdigest())
                for name in ('observation.npz','frame.npz')}
        batch = self.grasp.infer(observation)
        if input_evidence is not None:batch['model_input_evidence']=input_evidence
        request = robot_request(frame)
        request['batch'] = batch
        if self.complete_static_sim_preview:
            from .sim_preview import isolated_sim_preview
            source_scene=frame.get('simulation_scene_json')
            if isinstance(source_scene,np.ndarray):source_scene=source_scene.item()
            if not isinstance(source_scene,str) or len(source_scene)>1024*1024:
                raise ValueError('Same-frame simulator geometry is required for preview ranking')
            batch=isolated_sim_preview(dict(policy='complete_static_sim_orientation',batch=batch,
                robot_state=request['robot_state'],scene=json.loads(source_scene)),
                repository=self.repository,python_executable=self.ik_python)
        else:
            batch = isolated_validate(request, repository=self.repository, python_executable=self.ik_python)
        points, scene = observation.point_clouds()
        # Full-body ranking precedes the UI cutoff. Observed hand/path checks
        # remain advisory and unmodified for the eight displayed sim goals.
        observed=batch['candidates'][:8] if self.complete_static_sim_preview else batch['candidates']
        for candidate in observed:
            candidate['observed_hand_collision'] = self.collision.assess_grasp(candidate['world_canonical'], points, scene)
            candidate['collision_valid'] = False
        batch['observed_hand_checked_count']=len(observed)
        if self.complete_static_sim_preview:batch['preview_ranking_policy']='complete_static_sim_orientation'
        return batch

    def close(self):
        self.grasp.close()


class ProposalSession:
    def __init__(self, directory, engine, *, clock=time.monotonic, clock_id=None, recover_transient_tracking=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.engine, self.clock = engine, clock
        self.recover_transient_tracking=recover_transient_tracking
        self.clock_id = clock_id or host_clock_id()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='copilot-proposal-validation')
        self.audit_pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='copilot-failure-audit')
        self.failure_audit = None
        self._failure_audit_owner = None
        self.selected_failure_audit = None
        self.first_selected_target_failure = None
        self.selected_feature_audit = None
        self.first_selected_feature_failure = None
        self._tracking_failure = None
        self.current_observation = None
        self.frame_trace = FrameTimingTrace()
        self._batch_version = 0
        self._frame_update_version = 0
        self._published_key = None
        self._critical_published_key = None
        self._image_version = None
        self.publication_timing = {}
        self.future = None
        self.generation = 0
        self.pending_generation = None
        self.selection_key = None
        self.selected = None
        self.gate = None
        self.batch = None
        self.latest_frame = None
        self.display_frame = None
        self.last_preview = None
        self.last_targets = []
        self.stationary_report = {}
        self.frame_diagnostics = {}
        self.status = 'choose_object'
        self.error = None
        self.first_failure = None
        self.closed = False
        self.publish()

    def _selection(self):
        selection = read_json(self.directory/'target_selection.json', {})
        key = (selection.get('target_id'), selection.get('revision', selection.get('request_id')))
        if key != self.selection_key:
            self.generation += 1
            self.selection_key = key
            self.selected = key[0]
            self.batch = None
            self.last_preview = None
            self.gate = None
            self.error = None
            self.first_failure = None
            self.status = 'waiting_for_selected_object' if self.selected else 'choose_object'

    def update(self, frame, targets, masks, *, tracking_failure=None, feature_failure=None):
        work_started=time.perf_counter()
        if self.closed:
            raise RuntimeError('Proposal session is closed')
        self._frame_update_version += 1
        self._selection()
        self.current_observation = None
        self.latest_frame = frame
        self._tracking_failure = tracking_failure
        # A mask/box from a semantic keyframe may be used only after a tracker
        # has explicitly measured it in THIS RGB-D frame. Missing provenance
        # is not silently replaced with the service's receipt timestamp.
        def matches(target):
            stamp=target.get('timestamp')
            return (target.get('frame_id') == frame['frame_id']
                and isinstance(stamp,(int,float)) and not isinstance(stamp,bool)
                and np.isfinite(stamp) and abs(stamp-frame['timestamp'])<=1e-6
                and target.get('stale') is not True and target.get('observed',True) is True)
        self.last_targets = deepcopy([target for target in targets if matches(target)])
        targets = self.last_targets
        now = self.clock()
        if frame['clock_id'] != self.clock_id or not -.01 <= now-frame['timestamp'] <= .25:
            self.invalidate('camera_clock_unknown_or_stale')
        elif not self.selected:
            # Initial semantic inference can outlive its first camera frame.
            # With no operator selection there is no frozen target to re-arm:
            # a genuinely new fresh frame may restore the object chooser.
            # Selected targets keep their strict, deliberate-reselection latch.
            self.error = None
            self.first_failure = None
            self.status = 'choose_object'
        elif self.selected:
            target = next((t for t in targets if t.get('id', t.get('target_id')) == self.selected), None)
            if target is None or self.selected not in masks:
                self.invalidate('selected_object_not_observed')
            else:
                try:
                    observation = geometry_observation(frame, self.selected, masks[self.selected])
                    self.current_observation = observation
                    if self.recover_transient_tracking and self.error in (
                            'selected_object_not_observed','camera_stream_expired','camera_clock_unknown_or_stale',
                            'observation_sequence_gap'):
                        if self.gate is None or self.gate.recover_current(observation,now=now,clock_id=self.clock_id):
                            self.error=None
                            if self.batch is None and self.future is None:self.gate=None
                            self.status='revalidating_current_target'
                    if self.gate is None and not self.error:
                        self.gate = ObservedStationarity(observation)
                    if self.gate is not None:
                        self.stationary_report = self.gate.observe(observation, now=now, clock_id=self.clock_id)
                        if self.gate.invalid_reason:
                            self.invalidate(self.gate.invalid_reason)
                        elif self.batch is None and self.future is None and not self.error:
                            self.pending_generation = self.generation
                            self.future = self.pool.submit(self.engine.infer, deepcopy(observation), deepcopy(frame))
                            self.status = 'calculating_proposals'
                except (ValueError, KeyError, TypeError) as exc:
                    self.invalidate('invalid_target_geometry', str(exc))
        self._capture_selected_feature_failure(feature_failure)
        self.poll()
        if self.selected is not None:
            now=self.clock();stationarity=self.stationary_report
            record=dict(schema='punktnull.selected_frame_timing.v1',frame_id=int(frame['frame_id']),
                capture_timestamp=float(frame['timestamp']),observed_at=now,
                frame_age_s=now-frame['timestamp'],update_work_s=time.perf_counter()-work_started,
                target_id=self.selected,clock_id=frame['clock_id'],calibration_id=frame['calibration_id'],
                status=self.status,reason=str(self.error)[:256] if self.error else None,
                motion_static=stationarity.get('motion_static') is True,
                model_busy=self.future is not None,
                preview_capture_timestamp=self.last_preview.get('capture_timestamp') if self.last_preview else None,
                validation_observation_id=self.last_preview.get('validation_observation_id') if self.last_preview else None)
            for key in ('received_age_s','perception_s','post_perception_age_s'):
                value=self.frame_diagnostics.get(key)
                record[key]=float(value) if isinstance(value,(int,float)) and np.isfinite(value) else None
            for key in ('centroid_change_m','surface_p95_m'):
                value=stationarity.get(key)
                record[key]=float(value) if isinstance(value,(int,float)) and np.isfinite(value) else None
            record.update(self.publication_timing)
            self.frame_trace.append(record)
            self.frame_trace.publish(self.audit_pool,self.directory)

    def _capture_selected_feature_failure(self, capture):
        """One bounded diagnostic slot, independent of any target/error latch."""
        if self.first_selected_feature_failure is not None or self.selected is None:
            return
        evidence=None
        try:
            frame=self.latest_frame
            if (not isinstance(capture,dict) or frame is None
                    or capture.get('reason') not in (
                        'insufficient_visual_features','forward_flow_failed',
                        'backward_flow_failed','inconsistent_optical_flow','selected_surface_measurement_failed')
                    or capture.get('target_id')!=self.selected
                    or capture.get('current_frame_id')!=frame['frame_id']
                    or capture.get('current_timestamp')!=frame['timestamp']
                    or capture.get('phase') not in ('semantic_keyframe_rebase','current_tracking')
                    or capture.get('tracker_stage') not in ('head_rebase','confirmed_head_rebase','head','confirmed_head')):
                return
            evidence=dict(reason=capture['reason'],observed_at=self.clock(),
                frame_id=frame['frame_id'],capture_timestamp=frame['timestamp'],
                selected_target_id=self.selected,selection_revision=self.selection_key[1],
                phase=capture['phase'],tracker_stage=capture['tracker_stage'],
                selected_target_observed_after_processing=any(
                    t.get('id',t.get('target_id'))==self.selected for t in self.last_targets),
                replay_directory='first_selected_feature_failure',replay_status='pending',
                tracking_inputs_status=capture.get('capture_status'),
                max_raw_bytes=SELECTED_REPLAY_MAX_BYTES,motion_authorized=False)
            # Reserve before copying/writing. Failure or reselection cannot
            # overwrite evidence or turn this into an unbounded retry archive.
            self.first_selected_feature_failure=evidence
            frozen,tracking_arrays,tracking_json,size=freeze_selected_failure_inputs(frame,capture)
            evidence['raw_array_bytes']=size
            self.selected_feature_audit=self.audit_pool.submit(write_selected_feature_failure,
                self.directory/'first_selected_feature_failure',frozen,tracking_arrays,tracking_json,deepcopy(evidence))
        except Exception as exc:
            if evidence is not None:
                evidence.update(replay_status='failed',replay_error=type(exc).__name__)

    def invalidate(self, reason, detail=None):
        if (self.error is None and reason != 'service_stopped'
                and (not self.recover_transient_tracking or self.first_failure is None)):
            now = self.clock()
            self.first_failure = {'reason':reason, 'observed_at':now,
                'camera_age_s':now-self.latest_frame['timestamp'] if self.latest_frame else None,
                'frame_id':self.latest_frame['frame_id'] if self.latest_frame else None,
                'selected_target_id':self.selected,
                'frame_timing':deepcopy(self.frame_diagnostics),
                'stationarity':deepcopy(self.stationary_report),
                'model_input_evidence':deepcopy(self.batch.get('model_input_evidence')) if self.batch else None}
            if self.latest_frame is not None and self.failure_audit is None:
                copied_observation=deepcopy(self.current_observation)
                copied_frame=deepcopy(self.latest_frame)
                directory=self.directory/'first_failure'
                self.first_failure['replay_directory']='first_failure'
                self._failure_audit_owner=self.first_failure
                self.failure_audit=self.audit_pool.submit(write_failed_observation,directory,
                    copied_observation,copied_frame,deepcopy(self.first_failure))
        if (reason=='selected_object_not_observed' and self.selected is not None
                and self.first_selected_target_failure is None and self.latest_frame is not None):
            # Independent of any prior camera/error latch: observing this loss
            # records diagnostics only, never clears that latch or re-arms it.
            evidence=dict(reason=reason,observed_at=self.clock(),frame_id=self.latest_frame['frame_id'],
                capture_timestamp=self.latest_frame['timestamp'],selected_target_id=self.selected,
                selection_revision=self.selection_key[1],replay_directory='first_selected_target_failure',
                replay_status='pending',max_raw_bytes=SELECTED_REPLAY_MAX_BYTES,motion_authorized=False)
            self.first_selected_target_failure=evidence
            capture=self._tracking_failure
            if not (isinstance(capture,dict) and capture.get('target_id')==self.selected
                    and capture.get('current_frame_id')==self.latest_frame['frame_id']
                    and capture.get('current_timestamp')==self.latest_frame['timestamp']):
                capture=None
            evidence['tracking_inputs_status']=capture.get('capture_status') if capture else 'unavailable_for_this_target_and_frame'
            try:
                frozen,tracking_arrays,tracking_json,size=freeze_selected_failure_inputs(self.latest_frame,capture)
                evidence['raw_array_bytes']=size
                self.selected_failure_audit=self.audit_pool.submit(write_selected_target_failure,
                    self.directory/'first_selected_target_failure',frozen,tracking_arrays,tracking_json,deepcopy(evidence))
            except Exception as exc:
                # Recording failure cannot alter detection or motion policy.
                evidence.update(replay_status='failed',replay_error=type(exc).__name__)
        if self.gate is not None:
            self.gate.invalidate(reason)
        self.status = reason
        self.error = detail or reason
        if self.batch is not None:
            self.batch['valid'] = False
            self.batch['motion_static'] = False

    def poll(self):
        self._selection()
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
                if (self.pending_generation == self.generation and not self.error
                        and result['target_id'] == self.selected
                        and result['observation_id'] == self.gate.anchor.observation_id):
                    self.batch = deepcopy(result)
                    self._batch_version += 1
                    self.batch['model_source_capture_timestamp'] = result['capture_timestamp']
                    self.batch['model_source_observation_id'] = result['observation_id']
                    # Keep the full rejection audit; the UI presents a compact
                    # set ranked by explicit geometry/IK checks, then model score.
                    write_proposal_audit(self.directory/'proposal_audit.json', self.batch)
                    def rank(c):
                        if self.batch.get('preview_ranking_policy')=='complete_static_sim_orientation':
                            return (not (c.get('recorded_sim_endpoint_clear') is True),not c['ik_valid'],-c['model_score'])
                        observed = c.get('observed_hand_collision', {})
                        return (not (c['ik_valid'] and observed.get('pregrasp_collision_free') is True),
                                not c['ik_valid'], -c['model_score'])
                    candidates = self.batch['candidates']
                    self.batch['candidate_counts'] = {
                        'total': len(candidates),
                        'ik_valid': sum(c['ik_valid'] is True for c in candidates),
                        'ik_and_pregrasp_clear': sum(c['ik_valid'] is True and
                            c.get('observed_hand_collision', {}).get('pregrasp_collision_free') is True
                            for c in candidates)}
                    if self.batch.get('preview_ranking_policy')=='complete_static_sim_orientation':
                        self.batch['candidate_counts'].update(
                            observed_hand_checked=self.batch.get('observed_hand_checked_count',0),
                            recorded_sim_endpoint_clear=sum(c.get('recorded_sim_endpoint_clear') is True for c in candidates))
                    self.batch['candidates'] = sorted(self.batch['candidates'], key=rank)[:8]
                    self.status = 'preview_ready'
            except Exception as exc:
                self.invalidate('proposal_pipeline_failed', f'{type(exc).__name__}: {exc}')
            finally:
                self.future = None
        if self.latest_frame is not None and self.clock()-self.latest_frame['timestamp'] > .25:
            self.invalidate('camera_stream_expired')
        self.publish()

    def _publish_image(self):
        if self.latest_frame is None or self._image_version==self._frame_update_version:return
        frame=self.latest_frame
        temporary = self.directory/'frame.next.jpg'
        encoded=io.BytesIO();Image.fromarray(frame['rgb']).save(encoded,format='JPEG',quality=85)
        jpeg=encoded.getvalue();temporary.write_bytes(jpeg)
        temporary.replace(self.directory/'frame.jpg')
        projection=np.asarray(frame['T_world_camera'],dtype='<f8').tobytes()+np.asarray(frame['intrinsics'],dtype='<f8').tobytes()
        binding=dict(schema='punktnull.preview_frame.v1',frame_id=int(frame['frame_id']),
            capture_timestamp=float(frame['timestamp']),clock_id=frame['clock_id'],
            calibration_id=frame['calibration_id'],width=int(frame['rgb'].shape[1]),height=int(frame['rgb'].shape[0]),
            jpeg_sha256=hashlib.sha256(jpeg).hexdigest(),projection_sha256=hashlib.sha256(projection).hexdigest())
        binding['key']=hashlib.sha256(json.dumps(binding,sort_keys=True,allow_nan=False).encode()).hexdigest()
        self.display_frame=binding;self._image_version=self._frame_update_version

    def _critical_batch(self):
        # Keep every model/geometry/validation field. Only display coordinates
        # are separated; nested model evidence is immutable here and atomic_json
        # serializes synchronously before any subsequent UI copy is modified.
        batch={key:value for key,value in (self.batch or {}).items()
            if key not in ('preview_frame','preview_current','candidates')}
        batch['candidates']=[{key:value for key,value in candidate.items()
            if key not in ('preview_uv','preview_hand_polygons')}
            for candidate in (self.batch or {}).get('candidates',[])]
        if self.batch and not self.error and self.latest_frame is not None:
            batch['capture_timestamp']=self.latest_frame['timestamp']
            batch['validation_observation_id']=self.gate.last_observation
            batch['motion_static']=self.stationary_report.get('motion_static') is True
        else:batch['motion_static']=False
        batch.update(valid=False,runtime_motion_guard_available=False,preview_current=False,
            validation_reason=self.error or 'preview_only_actual_command_path_not_validated')
        if self.latest_frame is not None:
            # Processing acknowledgments include rejected observations. They
            # can stop a frame wait early, but cannot refresh model evidence or
            # authorize a command. The original batch time remains unchanged.
            frame=self.latest_frame
            batch['processed_frame']=dict(frame_id=int(frame['frame_id']),
                capture_timestamp=float(frame['timestamp']),clock_id=frame['clock_id'],
                calibration_id=frame['calibration_id'],rejected=self.error is not None,
                reason=self.error)
        return batch

    def publish(self):
        publish_started=time.perf_counter()
        trace_status=self.frame_trace.status()
        feature_failure=self.first_selected_feature_failure
        if feature_failure is not None and self.selected_feature_audit is not None:
            if self.selected_feature_audit.done():
                failure=self.selected_feature_audit.exception()
                feature_failure['replay_status']='failed' if failure else 'complete'
                if failure:feature_failure['replay_error']=type(failure).__name__
                else:feature_failure['files']=self.selected_feature_audit.result()
            else:feature_failure['replay_status']='pending'
        selected_failure=self.first_selected_target_failure
        if selected_failure is not None and self.selected_failure_audit is not None:
            if self.selected_failure_audit.done():
                failure=self.selected_failure_audit.exception()
                selected_failure['replay_status']='failed' if failure else 'complete'
                if failure:selected_failure['replay_error']=type(failure).__name__
                else:selected_failure['files']=self.selected_failure_audit.result()
            else:selected_failure['replay_status']='pending'
        if self.first_failure is not None and self.failure_audit is not None:
            if self.first_failure is not self._failure_audit_owner:
                # Keep one bounded replay per run. A deliberately reselected
                # object's later failure must never claim that earlier image.
                self.first_failure['replay_status']='not_captured_previous_failure_retained'
            elif self.failure_audit.done():
                failure=self.failure_audit.exception()
                self.first_failure['replay_status']='failed' if failure else 'complete'
                if failure:self.first_failure['replay_error']=type(failure).__name__
            else:self.first_failure['replay_status']='pending'
        camera_current=bool(self.latest_frame is not None and not self.closed
            and self.latest_frame['clock_id']==self.clock_id
            and -.01<=self.clock()-self.latest_frame['timestamp']<=.25)
        key=(self.generation,self._batch_version,self._frame_update_version,
            self.status,self.error,self.closed,self.future is not None,camera_current,
            (self.first_failure or {}).get('replay_status'),(self.first_failure or {}).get('replay_error'),
            (selected_failure or {}).get('replay_status'),(selected_failure or {}).get('replay_error'),
            (feature_failure or {}).get('replay_status'),(feature_failure or {}).get('replay_error'),
            trace_status['status'],trace_status['write_error'])
        # poll() still checks source changes, model completion and expiry before
        # reaching this point. Identical source data must not be retimed merely
        # because the UI polls again, and needs no repeated ghost/JSON work.
        if key==self._published_key:return
        critical=self._critical_batch()
        if key!=self._critical_published_key:
            atomic_json(self.directory/'latest_proposals.json',critical)
            self._critical_published_key=key
        critical_at=self.clock()
        self.publication_timing=dict(critical_published_at=critical_at,
            critical_publish_age_s=critical_at-self.latest_frame['timestamp'] if self.latest_frame else None,
            critical_publish_work_s=time.perf_counter()-publish_started)
        image_started=time.perf_counter();self._publish_image()
        self.publication_timing['jpeg_work_s']=time.perf_counter()-image_started
        projection_started=time.perf_counter()
        batch = deepcopy(critical)
        if self.batch and not self.error and self.latest_frame is not None:
            # A later observation is explicitly identified as validation evidence;
            # it does not pretend the model ran again on the current frame.
            batch['capture_timestamp'] = self.latest_frame['timestamp']
            batch['validation_observation_id'] = self.gate.last_observation
            batch['motion_static'] = self.stationary_report.get('motion_static') is True
            for candidate in batch['candidates']:
                candidate['preview_uv'] = preview_uv(candidate['world_canonical'], self.latest_frame)
                if hasattr(self.engine, 'preview'):
                    candidate['preview_hand_polygons'] = self.engine.preview.project(candidate['world_canonical'], self.latest_frame)
            batch['preview_frame']=deepcopy(self.display_frame)
            self.last_preview=deepcopy(batch)
        elif (self.batch and self.last_preview and self.display_frame
                and self.last_preview.get('preview_frame',{}).get('key')==self.display_frame['key']):
            # Retain the last geometrically bound preview for recording review,
            # without claiming a new model run or current validation at shutdown.
            batch=deepcopy(self.last_preview)
            batch['motion_static']=False
        else:
            batch['motion_static'] = False
        batch['preview_current']=bool(self.batch and not self.error and self.latest_frame is not None)
        batch['valid'] = False  # actual runtime path guard is a separate gate
        batch['runtime_motion_guard_available'] = False
        batch['validation_reason'] = self.error or 'preview_only_actual_command_path_not_validated'
        atomic_json(self.directory/'preview_proposals.json', batch)
        self.publication_timing['ui_projection_and_batch_write_s']=time.perf_counter()-projection_started
        self.publication_timing['publication_work_s']=time.perf_counter()-publish_started
        objects = [dict(target_id=t.get('id', t.get('target_id')), label=t.get('label', 'Objekt'),
                        bbox_px=t.get('bbox_px')) for t in self.last_targets]
        atomic_json(self.directory/'ui_state.json', {
            'mode': 'KI-Vorschau · keine Bewegungsfreigabe', 'status': self.status,
            'reason': self.error or 'Objekt und Griff bewusst auswählen. Bewegung noch gesperrt.',
            'objects': objects, 'selected_target_id': self.selected,
            'stationarity': self.stationary_report, 'model_busy': self.future is not None,
            'timing': self.frame_diagnostics,
            'publication_timing':self.publication_timing,
            'timing_trace':trace_status,
            'first_failure': self.first_failure,
            'first_selected_target_failure': selected_failure,
            'first_selected_feature_failure': feature_failure,
            'display_frame':self.display_frame,
            'camera_current':camera_current,
            'frame_id': self.latest_frame['frame_id'] if self.latest_frame else None})
        self._published_key=key

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.invalidate('service_stopped')
        self.publish()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.frame_trace.publish(self.audit_pool,self.directory,force=True)
        self.audit_pool.shutdown(wait=True, cancel_futures=False)
        self.engine.close()
        self.publish()
