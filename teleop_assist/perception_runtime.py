"""Slow semantic keyframes with current, independently tracked RGB-D views."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import copy
import time

import numpy as np

from .multiview_tracker import LockedTargetWristRecovery
from .sensor_fusion import target_corners_world
from .visual_tracker import RGBDMaskTracker


class AsyncPerception:
    def __init__(self, model, semantic_interval=.5, *, selected_surface_primary=False):
        if type(selected_surface_primary) is not bool:
            raise ValueError('selected_surface_primary must be an explicit boolean')
        self.selected_surface_primary=selected_surface_primary
        self.model=model
        self.tracker=RGBDMaskTracker(selected_surface_primary=selected_surface_primary)
        self.confirmed_head_tracker=RGBDMaskTracker(selected_surface_primary=selected_surface_primary)
        for tracker in (self.tracker,self.confirmed_head_tracker):
            tracker.set_surface_target(None,camera_role='head')
        self.selected_target_id=None
        self.wrist_trackers={name:RGBDMaskTracker(selected_surface_primary=selected_surface_primary)
                             for name in ('right_wrist','left_wrist')}
        # Close-range object masks can occupy more of the wrist image.
        from dataclasses import replace
        for tracker in self.wrist_trackers.values():
            tracker.geometry_config=replace(tracker.geometry_config,max_mask_fraction=.8)
        self.wrist_anchors={name:{} for name in self.wrist_trackers}
        self.observations={}
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='semantic-keyframe')
        self.future=None;self.source=None
        self.initialized=False
        self.last_submit=-float('inf');self.interval=float(semantic_interval)
        self.last_model_diagnostics={};self.model_frames=0
        self.history=deque(maxlen=32);self.discarded_keyframes=0
        self.selected_tracking_failure=None
        self.selected_feature_failure=None
        self._head_neighbour_source=None
        self._head_neighbours={}
        self._head_neighbour_overflow_until=-float('inf')

    def _remember_head_neighbours(self,frame,targets):
        """Bounded negative evidence shared by both trackers of the same camera.

        A lost neighbour does not become an empty scene. Keep its last actual
        position for at most the unchanged one-second semantic interval. This
        is rejection evidence only, never a new target or a refreshed identity.
        """
        source=(frame.get('clock_id'),frame.get('calibration_id'),frame.get('source_instance_id'))
        if any(not isinstance(value,str) or not value for value in source[:2]):
            return
        if source!=self._head_neighbour_source:
            self._head_neighbours={};self._head_neighbour_overflow_until=-float('inf')
            self._head_neighbour_source=source
        stamp=float(frame['timestamp'])
        for target in targets:
            identity=target.get('id')
            if (not isinstance(identity,str) or not 0<len(identity)<=256
                    or target.get('observed') is not True or target.get('stale')
                    or target.get('observation_camera','head')!='head' or target.get('head_visible',True) is not True
                    or target.get('frame_id')!=frame['frame_id'] or target.get('timestamp')!=stamp):
                continue
            previous=self._head_neighbours.get(identity)
            if previous is not None and previous['timestamp']>stamp:
                continue  # A delayed semantic result cannot rewind newer evidence.
            try:
                center=target.get('center_world')
                if not isinstance(center,(list,tuple,np.ndarray)) or len(center)!=3:
                    raise ValueError('Invalid neighbour center')
                center=[float(value) for value in center]
                if not np.isfinite(center).all():
                    center=None
            except (TypeError,ValueError,OverflowError):
                center=None
            self._head_neighbours[identity]=dict(timestamp=stamp,frame_id=int(frame['frame_id']),center=center)
        if len(self._head_neighbours)>128:
            self._head_neighbour_overflow_until=max(self._head_neighbour_overflow_until,stamp+1.)
            self._head_neighbours=dict(sorted(self._head_neighbours.items(),
                key=lambda item:item[1]['timestamp'],reverse=True)[:128])

    def _filter_merged_surface(self,frame,result,selected_target_id):
        """Apply one neighbourhood decision after all same-camera fallbacks."""
        self._remember_head_neighbours(frame,result['targets'])
        stamp=float(frame['timestamp'])
        self._head_neighbours={identity:item for identity,item in self._head_neighbours.items()
                               if 0<=stamp-item['timestamp']<=1.}
        target=next((item for item in result['targets'] if item['id']==selected_target_id),None)
        if target is None:
            return
        generic_reason=result['diagnostics'].get('lost_tracks',{}).get(selected_target_id)
        ambiguity_reasons={'ambiguous_current_neighbour','unresolved_previous_neighbour',
                           'invalid_neighbour_evidence','overlapping_instance_identity'}
        if (target.get('perception_method')!='measured_rgbd_surface_reprojection'
                and generic_reason not in ambiguity_reasons):
            return
        reason=generic_reason if generic_reason in ambiguity_reasons else None
        neighbour=None
        quality=target.get('tracking_quality',{})
        displacement=quality.get('visible_world_displacement_m',0.)
        center=np.asarray(target.get('center_world'),dtype=float)
        if (center.shape!=(3,) or not np.isfinite(center).all()
                or not np.isfinite(displacement) or displacement<0):
            reason='invalid_surface_neighbour_geometry'
        elif stamp<=self._head_neighbour_overflow_until:
            reason='surface_neighbour_history_capacity'
        elif reason is None:
            current_ids={item['id'] for item in result['targets']}
            for identity,item in self._head_neighbours.items():
                if identity==selected_target_id:
                    continue
                position=np.asarray(item['center'],dtype=float)
                if position.shape!=(3,) or not np.isfinite(position).all():
                    reason='invalid_neighbour_evidence';neighbour=identity;break
                if np.linalg.norm(center-position)<displacement+.025:
                    reason=('ambiguous_current_neighbour' if identity in current_ids
                            else 'unresolved_previous_neighbour')
                    neighbour=identity;break
        report=dict(reason=reason or 'known_neighbours_clear',neighbour_id=neighbour,
                    evidence_count=len(self._head_neighbours),motion_authorized=False)
        result['diagnostics']['multiview']['selected_surface_neighbour_gate']=report
        if reason is not None:
            result['targets']=[item for item in result['targets'] if item['id']!=selected_target_id]
            result['diagnostics'].setdefault('lost_tracks',{})[selected_target_id]=reason
            self.tracker.last_masks.pop(selected_target_id,None)
            self.observations.pop(selected_target_id,None)
            result['diagnostics']['multiview']['pose_observation_cameras'].pop(selected_target_id,None)
            result['diagnostics']['detected_count']=len(result['targets'])

    def _retain_feature_failure(self, tracker, frame, *, phase, tracker_stage):
        """Keep the first actual selected correspondence failure in this call.

        A semantic rebase candidate can be discarded while the older tracker
        still observes the target. Retain its already copied diagnostic inputs
        before discarding it; this never influences adoption or target validity.
        """
        try:
            capture=tracker.last_feature_failure
            if (self.selected_feature_failure is None and self.selected_target_id is not None
                    and isinstance(capture,dict) and capture.get('reason') in (
                        'insufficient_visual_features','forward_flow_failed',
                        'backward_flow_failed','inconsistent_optical_flow','selected_surface_measurement_failed')
                    and capture.get('target_id')==self.selected_target_id
                    and capture.get('current_frame_id')==int(frame['frame_id'])
                    and capture.get('current_timestamp')==float(frame['timestamp'])):
                # The tracker owns bounded copies, so a discarded candidate
                # can safely go out of scope. The service copies before async I/O.
                self.selected_feature_failure=dict(capture,phase=phase,tracker_stage=tracker_stage)
        except Exception:
            # Optional diagnosis cannot change the tracking decision.
            pass

    def confirm_cad_identity(self,evidence):
        """Consume real PoseRuntime evidence only for the confirmed visible ID."""
        if not evidence or evidence.get('target_id')!=self.selected_target_id:return False
        observation=evidence.get('_recovered_observation')
        evidence={key:value for key,value in evidence.items() if not key.startswith('_')}
        if observation is not None:
            if self.selected_surface_primary:
                # The stationary-pregrasp selected owner cannot be resurrected
                # by a separate CAD or camera recovery path between process calls.
                return False
            # A failed LK track is restarted only after a fresh same-frame
            # pretrained CAD result, using its actual observed-depth mask.
            frame=observation['frame'];target=observation['target'];camera=observation.get('camera_id','head')
            if (camera not in ('head',*self.wrist_trackers) or camera!=evidence.get('camera_id','head')
                    or target.get('id')!=self.selected_target_id
                    or int(frame['frame_id'])!=int(evidence.get('frame_id',-1))
                    or abs(float(frame['timestamp'])-float(evidence['timestamp']))>1e-6
                    or target.get('perception_method')!='foundationpose_current_depth_CAD_recovery'):
                return False
            semantic_time=float(target.get('semantic_keyframe_timestamp',float('nan')))
            if not np.isfinite(semantic_time) or not 0<=semantic_time<=float(frame['timestamp']):return False
            previous=self.confirmed_head_tracker if camera=='head' else self.wrist_trackers[camera]
            candidate=RGBDMaskTracker(previous.config,selected_surface_primary=self.selected_surface_primary)
            candidate.geometry_config=previous.geometry_config
            if camera=='head':
                candidate.set_surface_target(self.selected_target_id,camera_role='head')
            candidate.initialize(frame,[target],{target['id']:observation['mask']})
            if not candidate.confirm_identity(evidence):return False
            # This is a fresh current-depth observation, never a new SAM call.
            candidate._keyframe_timestamp=semantic_time
            if camera=='head':
                self.confirmed_head_tracker=candidate
                self.tracker.last_masks[target['id']]=observation['mask']
            else:
                self.wrist_trackers[camera]=candidate
                self.wrist_anchors[camera]={target['id']:{
                    'target':copy.deepcopy(observation['head_target']),
                    'source_center_world':np.asarray(target['center_world']),
                    'corners_world':target_corners_world(target,frame)}}
                # A wrist measurement never creates a fictitious head mask.
            self.observations[target['id']]=observation
            return True
        # The accepted pose used this camera's actual observed mask and depth.
        # It is independent identity evidence, never a new neural-mask claim.
        camera=evidence.get('camera_id','head')
        if camera=='head':return self.confirmed_head_tracker.confirm_identity(evidence)
        tracker=self.wrist_trackers.get(camera)
        return tracker.confirm_identity(evidence) if tracker is not None else False

    def _initialize(self, frame, result, current_bundle=None):
        self._remember_head_neighbours(frame,result['targets'])
        head_masks=result.pop('_masks')
        initialized=[];rebase_diagnostics={}
        current_head=(current_bundle or {}).get('frames',{}).get('head',frame)
        candidate=RGBDMaskTracker(self.tracker.config,selected_surface_primary=self.selected_surface_primary)
        candidate.geometry_config=self.tracker.geometry_config
        candidate.set_diagnostic_target(self.selected_target_id)
        candidate.set_surface_target(self.selected_target_id,camera_role='head')
        candidate.initialize(frame,result['targets'],head_masks)
        projected=candidate.track(current_head)
        self._retain_feature_failure(candidate,current_head,
            phase='semantic_keyframe_rebase',tracker_stage='head_rebase')
        # A delayed semantic frame may pass the semantic-age gate while its
        # direct flow/depth rebase fails. Do not rewind a current tracker before
        # that measurement succeeds. process() still tracks the current image
        # with the retained tracker; neither its source time nor TTL is renewed.
        current_empty=(not result['targets'] and not head_masks
            and int(frame['frame_id'])==int(current_head['frame_id'])
            and float(frame['timestamp'])==float(current_head['timestamp']))
        adopted=bool(projected['targets']) or current_empty
        selected_rebase_missing=(self.selected_surface_primary and self.selected_target_id is not None
            and self.selected_target_id in self.tracker._tracks
            and not any(target['id']==self.selected_target_id for target in projected['targets'])
            and int(frame['frame_id'])<int(current_head['frame_id'])
            and float(frame['timestamp'])<float(current_head['timestamp']))
        if selected_rebase_missing:
            # An unselected success in a delayed SAM candidate is not a new
            # selected observation. Keep the prior owner for an actual current
            # measurement in process(), with its original per-target TTL.
            adopted=False
        if adopted:
            self.tracker=candidate
            initialized.append('head')
        rebase_diagnostics['head']={**projected['diagnostics'],
            'adopted':adopted,
            'adoption_reason':('current_empty_semantic_keyframe' if current_empty else
                'selected_projection_missing_previous_owner_retained' if selected_rebase_missing else
                'current_projection_accepted' if adopted else 'current_projection_rejected_previous_tracker_retained'),
            'source_frame_id':int(frame['frame_id']),
            'source_timestamp':float(frame['timestamp']),
            'active_semantic_keyframe_timestamp':self.tracker._keyframe_timestamp}
        confirmed=[target for target in result['targets'] if target['id']==self.selected_target_id
                   and target['id'] in head_masks]
        if confirmed:
            candidate=RGBDMaskTracker(self.confirmed_head_tracker.config,
                                      selected_surface_primary=self.selected_surface_primary)
            candidate.geometry_config=self.confirmed_head_tracker.geometry_config
            candidate.set_diagnostic_target(self.selected_target_id)
            candidate.set_surface_target(self.selected_target_id,camera_role='head')
            candidate.initialize(frame,confirmed,head_masks)
            projected=candidate.track(current_head)
            self._retain_feature_failure(candidate,current_head,
                phase='semantic_keyframe_rebase',tracker_stage='confirmed_head_rebase')
            rebase_diagnostics['confirmed_head']=projected['diagnostics']
            if any(t['id']==self.selected_target_id for t in projected['targets']):
                self.confirmed_head_tracker=candidate
                initialized.append('confirmed_head')
            # A rejected delayed neural candidate cannot erase the existing
            # current, independently measured confirmed track.

        for camera,observation in result.pop('_camera_observations',{}).items():
            if camera not in self.wrist_trackers:continue
            candidate=RGBDMaskTracker(self.wrist_trackers[camera].config,
                                      selected_surface_primary=self.selected_surface_primary)
            candidate.geometry_config=self.wrist_trackers[camera].geometry_config
            candidate.initialize(observation['frame'],observation['targets'],observation['masks'])
            current_view=(current_bundle or {}).get('frames',{}).get(camera,observation['frame'])
            projected=candidate.track(current_view)
            rebase_diagnostics[camera]=projected['diagnostics']
            if not projected['targets']:continue
            self.wrist_trackers[camera]=candidate
            self.wrist_anchors[camera]={target['id']:{
                'target':copy.deepcopy(observation['head_targets'][target['id']]),
                'source_center_world':np.asarray(target['center_world']),
                'corners_world':target_corners_world(observation['head_targets'][target['id']],frame)}
                for target in observation['targets']}
            initialized.append(camera)
        self.last_model_diagnostics=copy.deepcopy(result['diagnostics'])
        self.last_model_diagnostics['direct_current_frame_rebase']=rebase_diagnostics
        self.model_frames+=1;self.initialized=True
        return initialized

    def _current_views(self,bundle,result,selected_target_id):
        """Retain identities only with fresh, validated flow and current depth."""
        head=bundle['frames']['head'];now=float(head['timestamp'])
        by_id={target['id']:target for target in result['targets']}
        primary_selected=self.selected_surface_primary and selected_target_id is not None
        if primary_selected and selected_target_id not in by_id:
            self.tracker.last_masks.pop(selected_target_id,None)
        confirmed=self.confirmed_head_tracker.track(head)
        for target in confirmed['targets']:
            if not primary_selected and target['id']==selected_target_id and target['id'] not in by_id:
                target.update(head_visible=True,observation_camera='head')
                result['targets'].append(target);by_id[target['id']]=target
                self.tracker.last_masks[target['id']]=self.confirmed_head_tracker.last_masks[target['id']]
        self.observations={identity:{'camera_id':'head','frame':head,'mask':mask}
                           for identity,mask in self.tracker.last_masks.items() if identity in by_id}
        view_diagnostics={};recoveries=[]
        for camera,tracker in self.wrist_trackers.items():
            frame=bundle['frames'].get(camera)
            if frame is None or abs(float(frame['timestamp'])-now)>.04:
                view_diagnostics[camera]={'status':'missing_or_stale'};continue
            tracked=tracker.track(frame)
            view_diagnostics[camera]=tracked['diagnostics']
            for measured in tracked['targets']:
                identity=measured['id']
                if primary_selected and identity==selected_target_id:
                    continue
                # Only the operator-selected identity may bridge a head occlusion.
                if identity!=selected_target_id:continue
                anchor=self.wrist_anchors[camera].get(identity)
                if anchor is None:continue
                delta=np.asarray(measured['center_world'])-anchor['source_center_world']
                if np.linalg.norm(delta)>.04:continue
                if identity not in by_id:
                    old=anchor['target']
                    center=np.asarray(old['center_world'])+delta
                    neighbors=[np.linalg.norm(np.asarray(t['center_world'])-center) for t in by_id.values()]
                    if neighbors and min(neighbors)<.04:continue
                    projected=LockedTargetWristRecovery._head_overlay_target(old,measured,center,
                        anchor['corners_world']+delta,head)
                    if projected is None:continue
                    projected.update(id=identity,frame_id=int(head['frame_id']),timestamp=now,
                        observed=True,stale=False,head_visible=False,observation_camera=camera,
                        perception_method='confirmed_identity_wrist_lk_rgbd',measurement_sources=[camera+'_rgbd'],
                        position_sigma_m=max(.012,measured['position_sigma_m']),
                        semantic_keyframe_timestamp=measured.get('semantic_keyframe_timestamp',float(frame['timestamp'])),
                        semantic_keyframe_age_s=measured.get('semantic_keyframe_age_s',0.),
                        tracking_quality=measured.get('tracking_quality'))
                    result['targets'].append(projected);by_id[identity]=projected;recoveries.append(identity)
                # Preserve the measured camera/mask pair, never a projected head
                # silhouette. Head is preferred while actually visible.
                if identity not in self.observations:
                    self.observations[identity]={'camera_id':camera,'frame':frame,
                                                'mask':tracker.last_masks[identity]}
        result['diagnostics']['multiview']={'confirmed_head':confirmed['diagnostics'],'views':view_diagnostics,'recovered_ids':recoveries,
            'pose_observation_cameras':{identity:obs['camera_id'] for identity,obs in self.observations.items()}}
        if primary_selected:
            result['diagnostics']['multiview']['selected_measurement_owner']='general_head_rgbd_tracker'
        result['diagnostics']['detected_count']=len(result['targets'])
        return result

    def process(self,bundle,selected_target_id=None):
        frame=bundle['frames']['head'];self.history.append(bundle)
        self.selected_tracking_failure=None
        self.selected_feature_failure=None
        if selected_target_id!=self.selected_target_id:
            self.confirmed_head_tracker.reset()
            self.selected_target_id=selected_target_id
        for tracker in (self.tracker,self.confirmed_head_tracker):
            tracker.set_diagnostic_target(selected_target_id)
            tracker.set_surface_target(selected_target_id,camera_role='head')
        initial=not self.initialized
        if initial:
            result=self.model.process_frame(frame,bundle,selected_target_id)
            self._initialize(frame,result)
            self.last_submit=-float('inf')
        elif self.future is not None and self.future.done():
            received=self.future.result()
            source_head=self.source['frames']['head']
            age=float(frame['timestamp'])-float(source_head['timestamp'])
            if 0<=age<=self.tracker.config.max_keyframe_age_s:
                # One measured direct warp uses the same forward/backward
                # flow, world-motion, depth and max-frame-gap gates. Replaying
                # every intermediate image for every object was delaying the
                # freshly captured control observation by multiple frame costs.
                self._initialize(source_head,received,current_bundle=bundle)
            else:self.discarded_keyframes+=1
            self.future=None
        result=self.tracker.track(frame)
        self._current_views(bundle,result,selected_target_id)
        self._filter_merged_surface(frame,result,selected_target_id)
        for role,tracker in (('head',self.tracker),('confirmed_head',self.confirmed_head_tracker)):
            self._retain_feature_failure(tracker,frame,phase='current_tracking',tracker_stage=role)
        capture=self.selected_feature_failure
        if capture is not None:
            if (capture.get('target_id')!=selected_target_id
                    or capture.get('current_frame_id')!=int(frame['frame_id'])
                    or capture.get('current_timestamp')!=float(frame['timestamp'])):
                self.selected_feature_failure=None
            else:
                # Presence is evidence only, not a new identity/validity gate.
                capture['selected_target_observed_after_processing']=any(
                    t['id']==selected_target_id for t in result['targets'])
        # Private in-process arrays, separate from diagnostics/RPC/UI JSON.
        # A failed rebase or one-view loss is not a selected-target loss when
        # another current measurement still observes that identity.
        if selected_target_id is not None and not any(t['id']==selected_target_id for t in result['targets']):
            for role,tracker in (('head',self.tracker),('confirmed_head',self.confirmed_head_tracker)):
                capture=tracker.last_feature_failure
                if (capture is not None and capture['target_id']==selected_target_id and
                        capture['current_frame_id']==int(frame['frame_id']) and
                        capture['current_timestamp']==float(frame['timestamp'])):
                    self.selected_tracking_failure=dict(capture,camera_role=role)
                    break
        result['diagnostics'].update(semantic_keyframe=copy.deepcopy(self.last_model_diagnostics),
            semantic_model=True,semantic_model_frames=self.model_frames,
            discarded_expired_keyframes=self.discarded_keyframes)
        result['diagnostics']['sensors']={
            'head':{'status':'fresh_rgbd_tracking'},
            'wrists':{'status':'independent_current_rgbd_tracking'},
            'lidar':{'status':'keyframe_consistency','timestamp':self.last_model_diagnostics.get('timestamp')}}
        if not initial and self.future is None and time.monotonic()-self.last_submit>=self.interval:
            self.source=bundle;self.last_submit=time.monotonic()
            self.future=self.pool.submit(self.model.process_frame,frame,bundle,selected_target_id)
        return result

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
