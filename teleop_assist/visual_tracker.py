"""Fast CPU RGB-D mask tracking between pretrained semantic keyframes.

SAM masks use verified optical flow and an inlier-supported image transform.
An explicitly selected surface can instead use independent measured RGBD
correspondence after feature detection or native optical correspondence fails.
Track loss, missing depth or inconsistent geometry withholds the target; the
tracker never substitutes a neighboring detection or invents an object ID.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import numbers
import time

import cv2
import numpy as np

from .learned_perception import LearnedPerceptionConfig, mask_to_target
from .perception import _calibration
from .cad_surface import current_cad_surface_support


def _bounded_diagnostic_metadata(value):
    """Bound the trusted local metadata graph before its diagnostic deepcopy."""
    pending=[(value,0)];nodes=used=0
    while pending:
        item,depth=pending.pop();nodes+=1
        if nodes>2048 or depth>8:raise ValueError('Diagnostic metadata structure limit')
        if isinstance(item,np.ndarray):
            if item.dtype.hasobject:raise ValueError('Diagnostic metadata object array')
            used+=item.nbytes
        elif isinstance(item,dict):
            if len(item)>128 or any(not isinstance(k,str) or len(k)>256 for k in item):
                raise ValueError('Diagnostic metadata mapping limit')
            used+=64+sum(len(k)*4 for k in item)
            pending.extend((v,depth+1) for v in item.values())
        elif isinstance(item,(list,tuple)):
            if len(item)>1024:raise ValueError('Diagnostic metadata sequence limit')
            used+=64+len(item)*8;pending.extend((v,depth+1) for v in item)
        elif isinstance(item,str):used+=len(item)*4
        elif item is None or isinstance(item,(int,float,bool,np.generic)):used+=32
        else:raise ValueError('Diagnostic metadata type invalid')
        if used>64*1024:raise ValueError('Diagnostic metadata byte limit')
    return copy.deepcopy(value)


@dataclass(frozen=True)
class VisualTrackerConfig:
    max_keyframe_age_s: float = 1.0
    max_frame_gap_s: float = .65
    min_flow_points: int = 3
    max_flow_points: int = 40
    max_forward_backward_error_px: float = 1.0
    max_photometric_error: float = 35.0
    min_inlier_fraction: float = .60
    ransac_error_px: float = 1.8
    max_normal_angle_deg: float = 40.
    max_depth_center_jump_m: float = .08
    max_world_speed_m_s: float = .5
    world_motion_slack_m: float = .015


class RGBDMaskTracker:
    def __init__(self, config=None, *, selected_surface_primary=False):
        if type(selected_surface_primary) is not bool:
            raise ValueError('selected_surface_primary must be an explicit boolean')
        self.selected_surface_primary = selected_surface_primary
        self.config = VisualTrackerConfig(**config) if isinstance(config, dict) else config or VisualTrackerConfig()
        self.geometry_config = LearnedPerceptionConfig()
        self._diagnostic_target_id = None
        self._surface_target_id = None
        self._surface_camera_role = None
        self.reset()

    def reset(self):
        self._frame = None
        self._gray = None
        self._tracks = {}
        self._keyframe_timestamp = None
        self.last_masks = {}
        self._identity_evidence = {}
        self.last_feature_failure = None
        self._surface_frame = None

    def set_surface_target(self, identity, *, camera_role):
        """Opt in only the selected ID to stationary surface correspondence.

        The role names the owning input channel, not a field claimed to have
        come from the camera. Its physical mount remains bound by calibration_id.
        This enables a measurement method; it does not confer control authority.
        """
        if not isinstance(camera_role,str) or not 0 < len(camera_role) <= 64:
            raise ValueError('A bounded camera owner role is required')
        if identity is not None and (not isinstance(identity,str) or not 0 < len(identity) <= 256):
            raise ValueError('A selected target ID or None is required')
        if camera_role != self._surface_camera_role:
            self._surface_frame = None
        self._surface_camera_role = camera_role
        self._surface_target_id = identity

    def _surface_copy(self, frame):
        """Bounded sensor-only snapshot; unavailable provenance disables fallback."""
        if self._surface_camera_role is None:
            return None
        try:
            rgb,depth=np.asarray(frame['rgb']),np.asarray(frame['depth'])
            if (rgb.dtype != np.uint8 or depth.dtype.kind not in 'fi'
                    or rgb.shape != depth.shape+(3,) or depth.ndim != 2
                    or rgb.nbytes+depth.nbytes > 16*1024*1024):
                return None
            if any(not isinstance(frame.get(key),str) or not frame[key]
                   for key in ('clock_id','calibration_id')):
                return None
            snapshot={key:copy.deepcopy(frame[key]) for key in (
                'frame_id','timestamp','intrinsics','T_world_camera','T_world_base',
                'clock_id','calibration_id')}
            snapshot.update(rgb=rgb.copy(),depth=depth.copy(),
                camera_id=self._surface_camera_role,camera_id_source='tracker_owner_channel')
            if 'source_instance_id' in frame:
                snapshot['source_instance_id']=frame['source_instance_id']
            return snapshot
        except (KeyError,ValueError,TypeError,MemoryError):
            return None

    def set_diagnostic_target(self, identity):
        """Enable only an optional local replay copy; never alter tracking."""
        identity = identity if isinstance(identity,str) and 0 < len(identity) <= 256 else None
        if identity != self._diagnostic_target_id:
            self.last_feature_failure = None
        self._diagnostic_target_id = identity

    def _capture_feature_failure(self, frame, old, feature_mask, points, *,
                                 reason='insufficient_visual_features', flow_inputs=None):
        if old.get('id') != self._diagnostic_target_id or self._diagnostic_target_id is None:
            return
        # These are the actual inputs to this failed native optical call. Never
        # reconstruct it later from the next gray/mask or rerun feature finding.
        record = dict(schema='punktnull.selected_feature_failure.v1',
            target_id=self._diagnostic_target_id, reason=reason, tracking_failure_trigger=reason,
            failure_stage=('feature_detection' if reason=='insufficient_visual_features' else 'optical_correspondence'),
            current_frame_id=int(frame['frame_id']), current_timestamp=float(frame['timestamp']),
            points_were_none=points is None, point_count=0 if points is None else len(points),
            feature_parameters=dict(maxCorners=self.config.max_flow_points,qualityLevel=.01,
                minDistance=3,blockSize=3,min_flow_points=self.config.min_flow_points),
            max_array_bytes=8*1024*1024, motion_authorized=False)
        self.last_feature_failure = record
        try:
            if flow_inputs is not None:
                # At most40 point records, stored as JSON metadata rather than
                # extra full-image arrays. Nonfinite native values stay explicit
                # nulls with a count; strict JSON replay serialization still works.
                flow={}
                for name,value in flow_inputs.items():
                    if value is None:
                        flow[name]={'available':False};continue
                    array=np.asarray(value)
                    if array.dtype.kind not in 'bifu' or array.size>80:
                        raise ValueError('Optical replay vector limit')
                    flat=array.reshape(-1);finite=np.isfinite(flat);valid=flat[finite]
                    flow[name]={'available':True,'shape':list(array.shape),
                        'values':[float(x) if ok else None for x,ok in zip(flat,finite)],
                        'finite_count':int(finite.sum()),'nonfinite_count':int((~finite).sum()),
                        'positive_count':int(np.count_nonzero(valid>0)),
                        'min':float(valid.min()) if len(valid) else None,
                        'median':float(np.median(valid)) if len(valid) else None,
                        'max':float(valid.max()) if len(valid) else None}
                record['optical_flow']=_bounded_diagnostic_metadata(flow)
            arrays=dict(previous_gray=self._gray,previous_feature_mask=feature_mask,
                corner_points=np.empty((0,1,2),np.float32) if points is None else points)
            if (self._surface_frame is not None
                    and self._surface_frame['frame_id']==self._frame['frame_id']
                    and self._surface_frame['timestamp']==self._frame['timestamp']):
                arrays.update(previous_rgb=self._surface_frame['rgb'],
                              previous_depth=self._surface_frame['depth'])
                record['surface_previous_frame']=_bounded_diagnostic_metadata({
                    key:value for key,value in self._surface_frame.items() if key not in ('rgb','depth')})
            size=sum(value.nbytes for value in arrays.values())
            record['raw_array_bytes']=size
            if (size>record['max_array_bytes'] or any(value.dtype.hasobject for value in arrays.values())
                    or (points is not None and len(points)>self.config.max_flow_points)):
                record['capture_status']='size_or_dtype_limit';return
            record.update(_bounded_diagnostic_metadata(dict(previous_frame=self._frame,
                current_frame=self._metadata(frame),previous_target=old,
                tracker_config=asdict(self.config),geometry_config=asdict(self.geometry_config),
                library_versions=dict(opencv=cv2.__version__,numpy=np.__version__))))
            record['arrays']={name:value.copy() for name,value in arrays.items()}
            record['capture_status']='captured'
        except Exception as exc:
            record.pop('arrays',None)
            record['capture_status']='capture_failed';record['capture_error']=type(exc).__name__

    def _capture_surface_failure(self, frame, old, mask, reason, quality):
        """Exact surface inputs only; no feature detector or LK call occurred."""
        if old.get('id') != self._diagnostic_target_id or self._diagnostic_target_id is None:
            return
        record=dict(schema='punktnull.selected_feature_failure.v1',
            target_id=self._diagnostic_target_id, reason='selected_surface_measurement_failed',
            tracking_failure_trigger='selected_surface_measurement_failed',
            failure_stage='surface_measurement', surface_measurement_reason=reason,
            current_frame_id=int(frame['frame_id']), current_timestamp=float(frame['timestamp']),
            max_array_bytes=8*1024*1024, motion_authorized=False)
        self.last_feature_failure=record
        try:
            if self._surface_frame is None:
                record['capture_status']='previous_rgbd_unavailable';return
            arrays=dict(previous_rgb=self._surface_frame['rgb'],
                        previous_depth=self._surface_frame['depth'],previous_mask=mask)
            size=sum(value.nbytes for value in arrays.values())
            record['raw_array_bytes']=size
            if size>record['max_array_bytes'] or any(value.dtype.hasobject for value in arrays.values()):
                record['capture_status']='size_or_dtype_limit';return
            record.update(_bounded_diagnostic_metadata(dict(previous_frame=self._frame,
                current_frame=self._metadata(frame),previous_target=old,
                surface_previous_frame={key:value for key,value in self._surface_frame.items()
                                        if key not in ('rgb','depth')},
                surface_measurement_quality=quality,tracker_config=asdict(self.config),
                geometry_config=asdict(self.geometry_config),
                library_versions=dict(opencv=cv2.__version__,numpy=np.__version__))))
            record['arrays']={name:value.copy() for name,value in arrays.items()}
            record['capture_status']='captured'
        except Exception as exc:
            record.pop('arrays',None)
            record['capture_status']='capture_failed';record['capture_error']=type(exc).__name__

    @staticmethod
    def _validate(frame):
        rgb = np.asarray(frame['rgb'])
        depth = np.asarray(frame['depth'])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or depth.shape != rgb.shape[:2]:
            raise ValueError('Tracker requires matching uint8 RGB and optical depth')
        _calibration(frame['intrinsics'], frame['T_world_camera'])
        if not np.isfinite(float(frame['timestamp'])) or float(frame['timestamp']) < 0:
            raise ValueError('Finite acquisition timestamp required')
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def initialize(self, frame, targets, masks):
        gray = self._validate(frame)
        timestamp = float(frame['timestamp'])
        tracks = {}
        for target in targets:
            identity = target['id']
            if identity not in masks or target.get('stale') or not target.get('observed', True):
                continue
            mask = np.asarray(masks[identity], dtype=bool)
            if mask.shape != gray.shape or mask.sum() < self.geometry_config.min_mask_pixels:
                continue
            # A recovered current mask is not a new semantic detection. Each
            # target carries its own original acquisition time; malformed
            # explicit times must never fall back to the new frame's time.
            semantic = target.get('semantic_keyframe_timestamp', timestamp)
            if (isinstance(semantic, (bool, np.bool_)) or not isinstance(semantic, numbers.Real)
                    or not np.isfinite(semantic) or not 0 <= semantic <= timestamp):
                continue
            semantic = float(semantic)
            measured = dict(target, semantic_keyframe_timestamp=semantic,
                            semantic_keyframe_age_s=timestamp-semantic)
            tracks[identity] = {'target': measured, 'mask': mask.copy(),
                                'semantic_keyframe_timestamp': semantic}
        self._tracks = tracks
        self._identity_evidence = {}
        self._frame = self._metadata(frame)
        self._gray = gray
        self._surface_frame = self._surface_copy(frame)
        # Compatibility/diagnostic summary only. Authority is per track, so
        # one fresh DINO detection cannot renew an older recovered instance.
        self._keyframe_timestamp = min((state['semantic_keyframe_timestamp']
                                       for state in tracks.values()), default=timestamp)
        self.last_masks = {key: value['mask'].copy() for key, value in tracks.items()}

    def confirm_identity(self,evidence):
        """Extend an existing track only with an accepted, recent CAD observation.

        This never changes the SAM keyframe timestamp or revives a lost track.
        All following RGB/flow/depth gates still run on every current image.
        """
        if not evidence or self._frame is None or evidence.get('kind')!='foundationpose_rgbd':return False
        identity=evidence.get('target_id');state=self._tracks.get(identity)
        if state is None or state['target'].get('label')!='handle' or evidence.get('accepted') is not True:return False
        timestamp=float(evidence.get('timestamp',float('nan')))
        age=float(self._frame['timestamp'])-timestamp
        if not np.isfinite(timestamp) or not 0<=age<=.5:return False
        if not 0<=int(evidence.get('frame_id',-1))<=int(self._frame['frame_id']):return False
        q=evidence.get('quality',{})
        bounds=(q.get('silhouette_iou',0)>=.45,q.get('measured_depth_support',0)>=.7,
                0<=q.get('median_depth_error_m',float('inf'))<=.01,
                0<=q.get('p90_depth_error_m',float('inf'))<=.02,
                0<=evidence.get('pose_position_sigma_m',float('inf'))<=.025,
                0<=evidence.get('pose_orientation_sigma_rad',float('inf'))<=.35)
        if not all(bounds):return False
        center=np.asarray(evidence.get('center_world',[]),float)
        if center.shape!=(3,) or not np.isfinite(center).all():return False
        if np.linalg.norm(np.asarray(state['target']['center_world'])-center)>.04:return False
        previous=self._identity_evidence.get(identity)
        if previous and timestamp<=previous['timestamp']:return False
        self._identity_evidence[identity]=dict(evidence)
        return True

    @staticmethod
    def _metadata(frame):
        return {key:frame[key] for key in ('frame_id','timestamp','intrinsics','T_world_camera')}

    def track(self, frame):
        start = time.perf_counter()
        gray = self._validate(frame)
        timestamp = float(frame['timestamp'])
        surface_current = self._surface_copy(frame)
        if (self.last_feature_failure is not None and
                (self.last_feature_failure['current_frame_id'] != int(frame['frame_id']) or
                 self.last_feature_failure['current_timestamp'] != timestamp)):
            self.last_feature_failure = None
        diagnostic = {'method':'lk_ransac_sam2_mask_current_rgbd', 'semantic_model':True,
                      'frame_id':int(frame['frame_id']), 'timestamp':timestamp,
                      'scene_truth_used':False, 'semantic_inference_this_frame':False,
                      'stale':False, 'lost_tracks':{}}
        self.last_masks = {}
        if self._frame is None:
            diagnostic.update(reason='semantic_keyframe_required', detected_count=0, stale=True)
            return {'targets':[], 'diagnostics':diagnostic}
        dt = timestamp-float(self._frame['timestamp'])
        age = timestamp-self._keyframe_timestamp
        diagnostic['semantic_keyframe_age_s'] = age
        diagnostic['semantic_keyframe_ages_s'] = {
            identity: timestamp-state['semantic_keyframe_timestamp']
            for identity, state in self._tracks.items()}
        if dt == 0 and int(frame['frame_id']) == int(self._frame['frame_id']):
            # The image is current, but an explicit recovered identity can
            # already be expired. Apply the same semantic/CAD lease as on an
            # advancing frame before exposing the measured same-frame mask.
            current_tracks = {}; diagnostic['identity_validations'] = {}
            for identity, state in self._tracks.items():
                semantic = state['semantic_keyframe_timestamp']
                track_age = timestamp-semantic
                evidence = self._identity_evidence.get(identity)
                evidence_age = timestamp-evidence['timestamp'] if evidence else float('inf')
                cad_current = evidence is not None and 0 <= evidence_age <= .5
                if self.selected_surface_primary and identity==self._surface_target_id:
                    checked_at=time.monotonic()
                    if (not np.isfinite(checked_at) or checked_at<0
                            or max(track_age,checked_at-semantic)>min(1.,self.config.max_keyframe_age_s)):
                        diagnostic['lost_tracks'][identity]='original_semantic_evidence_expired'
                        continue
                if track_age > self.config.max_keyframe_age_s and not cad_current:
                    diagnostic['lost_tracks'][identity] = 'expired_semantic_and_cad_identity_evidence'
                    continue
                target = dict(state['target'], semantic_keyframe_timestamp=semantic,
                              semantic_keyframe_age_s=track_age)
                if cad_current:
                    target.update(identity_validation_source='foundationpose_rgbd',
                        identity_validation_timestamp=evidence['timestamp'], identity_validation_age_s=evidence_age)
                    diagnostic['identity_validations'][identity] = {
                        'source':'foundationpose_rgbd', 'timestamp':evidence['timestamp'], 'age_s':evidence_age}
                current_tracks[identity] = {**state, 'target':target}
            self._tracks = current_tracks
            targets = [dict(value['target']) for value in current_tracks.values()]
            self.last_masks = {key:value['mask'].copy() for key,value in current_tracks.items()}
            if any(target.get('perception_method')=='measured_rgbd_surface_reprojection' for target in targets):
                diagnostic['method']='lk_and_selected_surface_current_rgbd'
            diagnostic.update(reason='current_semantic_keyframe' if targets else
                              'expired_or_nonadvancing_tracking_frame' if diagnostic['lost_tracks'] else
                              'current_semantic_keyframe',
                              stale=bool(diagnostic['lost_tracks']) and not targets,
                              detected_count=len(targets))
            return {'targets':targets,'diagnostics':diagnostic}
        if (dt <= 0 or int(frame['frame_id']) <= int(self._frame['frame_id'])
                or dt > self.config.max_frame_gap_s
                or gray.shape != self._gray.shape):
            self._tracks.clear()
            self._surface_frame = None
            diagnostic.update(reason='expired_or_nonadvancing_tracking_frame', detected_count=0, stale=True)
            return {'targets':[], 'diagnostics':diagnostic}
        current_tracks = {}
        diagnostic['identity_validations']={}
        for identity, state in self._tracks.items():
            semantic = state['semantic_keyframe_timestamp']
            track_age = timestamp-semantic
            evidence=self._identity_evidence.get(identity)
            evidence_age=timestamp-evidence['timestamp'] if evidence else float('inf')
            cad_current=evidence is not None and 0<=evidence_age<=.5
            if track_age>self.config.max_keyframe_age_s and not cad_current:
                diagnostic['lost_tracks'][identity]='expired_semantic_and_cad_identity_evidence'
                continue
            identity_age=min(track_age,evidence_age) if cad_current else track_age
            old_target, mask = state['target'], state['mask']
            target, warped, quality, reason = self._one(frame, gray, old_target, mask, dt,
                cad_evidence=evidence if cad_current else None,surface_current=surface_current)
            if target is None:
                diagnostic['lost_tracks'][identity] = reason
                continue
            surface_measurement=quality.get('method')=='measured_rgbd_surface_reprojection'
            target.update(id=identity, frame_id=int(frame['frame_id']), timestamp=timestamp,
                          semantic_keyframe_timestamp=semantic,
                          semantic_keyframe_age_s=track_age, tracking_quality=quality,
                          perception_method=('measured_rgbd_surface_reprojection' if surface_measurement
                                             else 'lk_ransac_sam2_mask_current_rgbd'))
            if surface_measurement:
                target.update(mask_source='current_depth_and_rgb_correspondence_to_previous_measured_mask',
                    normal_source='current_measured_depth_on_correspondence_mask',
                    segmentation_confidence_source='previous_SAM_only_not_current_semantic_inference')
            elif old_target.get('mask_source')=='current_measured_depth_near_previous_accepted_CAD_surface':
                target.update(mask_source=old_target['mask_source'],
                    normal_source='current_CAD_surface_mask_local_measured_depth',
                    perception_method='lk_ransac_CAD_depth_mask_current_rgbd',
                    segmentation_confidence_source='original_SAM_proposal_not_current_depth_mask')
            # Each propagated keyframe introduces additional correspondence
            # uncertainty; a new neural mask resets it. No score inflation.
            if cad_current:
                target.update(identity_validation_source='foundationpose_rgbd',
                    identity_validation_timestamp=evidence['timestamp'],identity_validation_age_s=evidence_age)
                diagnostic['identity_validations'][identity]={'source':'foundationpose_rgbd',
                    'timestamp':evidence['timestamp'],'age_s':evidence_age}
            if surface_measurement:
                # Keep previous uncertainty; rounded reprojection is a pixel
                # correspondence, not an invented zero-error LK transform.
                k,_=_calibration(frame['intrinsics'],np.eye(4))
                rounding_m=np.sqrt(.5)*target['center_camera'][2]/min(k['fx'],k['fy'])
                target['position_sigma_m']=max(old_target['position_sigma_m'],
                    target['position_sigma_m']+rounding_m)+.002*dt
            else:
                target['position_sigma_m'] += .002*identity_age + quality['flow_residual_px']*target['center_camera'][2]/float(_calibration(frame['intrinsics'],np.eye(4))[0]['fx'])
            target['pose_confidence'] = min(target['pose_confidence'],old_target.get('pose_confidence',.8))
            current_tracks[identity] = {'target':target,'mask':warped,
                                        'semantic_keyframe_timestamp':semantic}
        # Overlapping propagated instances indicate identity ambiguity. Withhold
        # both until semantic detection resolves them.
        keys = list(current_tracks)
        ambiguous = set()
        for i,key in enumerate(keys):
            for other in keys[i+1:]:
                a,b = current_tracks[key]['mask'],current_tracks[other]['mask']
                if np.count_nonzero(a & b)/max(1,min(np.count_nonzero(a),np.count_nonzero(b))) > .50:
                    ambiguous.update((key,other))
        for key in ambiguous:
            current_tracks.pop(key)
            diagnostic['lost_tracks'][key] = 'overlapping_instance_identity'
        # A supported surface must not silently replace a nearby competing
        # instance. Current neighbours are independently remeasured above. A
        # previously nearby instance that was lost this frame stays unresolved;
        # its absence from current_tracks is not evidence that it disappeared.
        for identity,state in list(current_tracks.items()):
            quality=state['target'].get('tracking_quality',{})
            if quality.get('method')!='measured_rgbd_surface_reprojection':
                continue
            center=np.asarray(state['target']['center_world'])
            margin=quality['visible_world_displacement_m']+.025
            reason=None
            for other,previous in self._tracks.items():
                if other==identity:
                    continue
                current=current_tracks.get(other)
                position=np.asarray((current or previous)['target']['center_world'])
                if position.shape!=(3,) or not np.isfinite(position).all():
                    reason='invalid_neighbour_evidence';break
                if np.linalg.norm(center-position)<margin:
                    reason='ambiguous_current_neighbour' if current else 'unresolved_previous_neighbour'
                    break
            if reason is not None:
                current_tracks.pop(identity)
                diagnostic['lost_tracks'][identity]=reason
        self._tracks = current_tracks
        self._gray = gray
        self._frame = self._metadata(frame)
        self._surface_frame = surface_current
        self.last_masks = {key:value['mask'] for key,value in current_tracks.items()}
        targets = [value['target'] for value in current_tracks.values()]
        if any(target.get('perception_method')=='measured_rgbd_surface_reprojection' for target in targets):
            diagnostic['method']='lk_and_selected_surface_current_rgbd'
        diagnostic.update(reason='observed_tracked_components' if targets else ('expired_or_nonadvancing_tracking_frame' if age>self.config.max_keyframe_age_s else 'tracking_lost'),
                          stale=not targets and age>self.config.max_keyframe_age_s,
                          detected_count=len(targets), total_seconds=time.perf_counter()-start)
        return {'targets':targets,'diagnostics':diagnostic}

    def _optical_surface_fallback(self,frame,old,mask,feature_mask,points,dt,surface_current,
                                  *,trigger,flow_inputs=None):
        """Independent measurement after four explicit optical failure gates only."""
        try:self._capture_feature_failure(frame,old,feature_mask,points,reason=trigger,flow_inputs=flow_inputs)
        except Exception as exc:
            # Optional recording never changes the measurement decision.
            self.last_feature_failure=dict(target_id=old.get('id'),current_frame_id=int(frame['frame_id']),
                current_timestamp=float(frame['timestamp']),capture_status='capture_failed',
                capture_error=type(exc).__name__,reason=trigger,tracking_failure_trigger=trigger)
        if old.get('id') != self._surface_target_id or self._surface_target_id is None:
            return None,None,None,trigger
        if self._surface_frame is None or surface_current is None:
            return None,None,None,trigger
        from .surface_tracking import measure
        measured=measure(self._surface_frame,mask,old,surface_current,
            old['semantic_keyframe_timestamp'],selected_target_id=self._surface_target_id,
            geometry_config=self.geometry_config)
        measured['quality']['optical_failure_trigger']=trigger
        if self.last_feature_failure is not None:
            self.last_feature_failure['surface_fallback']={
                'accepted_measurement':measured['target'] is not None,
                'reason':measured['reason'],'quality':copy.deepcopy(measured['quality'])}
        if measured['target'] is None:
            return None,None,None,trigger+'_surface_'+str(measured['reason'])
        target,warped,quality=measured['target'],measured['mask'],measured['quality']
        geometry,reason=self._geometry_consistency(frame,old,target,warped,dt)
        if reason is not None:
            if self.last_feature_failure is not None:
                self.last_feature_failure['surface_fallback'].update(accepted_measurement=False,reason=reason)
            return None,None,None,reason
        quality.update(geometry)
        return target,warped,quality,None

    def _one(self,frame,gray,old,mask,dt,cad_evidence=None,surface_current=None):
        if (self.selected_surface_primary and self._surface_target_id is not None
                and old.get('id')==self._surface_target_id):
            return self._primary_surface(frame,old,mask,dt,surface_current)
        c=self.config
        feature_mask=mask.astype(np.uint8)*255
        points=cv2.goodFeaturesToTrack(self._gray,maxCorners=c.max_flow_points,qualityLevel=.01,
                                      minDistance=3,mask=feature_mask,blockSize=3)
        if points is None or len(points)<c.min_flow_points:
            return self._optical_surface_fallback(frame,old,mask,feature_mask,points,dt,surface_current,
                trigger='insufficient_visual_features')
        nxt,status,error=cv2.calcOpticalFlowPyrLK(self._gray,gray,points,None,
                    winSize=(19,19),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.02))
        if nxt is None:
            return self._optical_surface_fallback(frame,old,mask,feature_mask,points,dt,surface_current,
                trigger='forward_flow_failed',flow_inputs=dict(forward_points=nxt,forward_status=status,photometric_error=error))
        back,back_status,back_error=cv2.calcOpticalFlowPyrLK(gray,self._gray,nxt,None,winSize=(19,19),maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.02))
        if back is None:
            return self._optical_surface_fallback(frame,old,mask,feature_mask,points,dt,surface_current,
                trigger='backward_flow_failed',flow_inputs=dict(forward_points=nxt,forward_status=status,
                    photometric_error=error,backward_points=back,backward_status=back_status,backward_photometric_error=back_error))
        fb=np.linalg.norm((back-points).reshape(-1,2),axis=1)
        usable=(status.ravel()>0)&(back_status.ravel()>0)&(fb<c.max_forward_backward_error_px)&(error.ravel()<c.max_photometric_error)
        src,dst=points.reshape(-1,2)[usable],nxt.reshape(-1,2)[usable]
        if len(src)<c.min_flow_points:
            return self._optical_surface_fallback(frame,old,mask,feature_mask,points,dt,surface_current,
                trigger='inconsistent_optical_flow',flow_inputs=dict(forward_points=nxt,forward_status=status,
                    photometric_error=error,backward_points=back,backward_status=back_status,
                    backward_photometric_error=back_error,forward_backward_error_px=fb,usable=usable))
        affine,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,
                    ransacReprojThreshold=c.ransac_error_px,maxIters=300,confidence=.99,refineIters=5)
        if affine is None or inliers is None or inliers.sum()<c.min_flow_points or inliers.mean()<c.min_inlier_fraction:
            return None,None,None,'insufficient_ransac_consensus'
        scale=float(np.linalg.norm(affine[:,0]))
        if not .7<scale<1.4:
            return None,None,None,'excessive_view_change'
        transformed=src@affine[:,:2].T+affine[:,2]
        residual=float(np.median(np.linalg.norm(transformed-dst,axis=1)[inliers.ravel()>0]))
        warped=cv2.warpAffine(mask.astype(np.uint8),affine,(gray.shape[1],gray.shape[0]),
                             flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).astype(bool)
        target,reason=mask_to_target(warped,frame['depth'],frame['intrinsics'],frame['T_world_camera'],
            label=old.get('semantic_phrase',old['label']),detection_score=old.get('detection_confidence',old.get('confidence',.5)),
            mask_score=old.get('segmentation_confidence',.8),config=self.geometry_config)
        if target is None:
            return None,None,None,reason
        geometry,reason=self._geometry_consistency(frame,old,target,warped,dt,cad_evidence)
        if reason is not None:
            return None,None,None,reason
        quality={'inlier_count':int(inliers.sum()),'inlier_fraction':float(inliers.mean()),
                 'flow_residual_px':residual,'median_forward_backward_error_px':float(np.median(fb[usable])),
                 **geometry}
        return target,warped,quality,None

    def _primary_surface(self,frame,old,mask,dt,surface_current):
        """Selected stationary-pregrasp measurement owner; failure has no LK rescue."""
        from .surface_tracking import measure
        if self._surface_frame is None or surface_current is None:
            measured=dict(target=None,mask=None,quality={},reason='surface_rgbd_provenance_unavailable')
        else:
            measured=measure(self._surface_frame,mask,old,surface_current,
                old['semantic_keyframe_timestamp'],selected_target_id=self._surface_target_id,
                geometry_config=self.geometry_config)
        target,warped,quality,reason=measured['target'],measured['mask'],measured['quality'],measured['reason']
        if target is not None:
            geometry,reason=self._geometry_consistency(frame,old,target,warped,dt)
            if reason is None:
                quality.update(geometry)
                return target,warped,quality,None
        try:
            self._capture_surface_failure(frame,old,mask,reason,quality)
        except Exception as exc:
            # Diagnostic failure never permits another measurement or changes loss.
            self.last_feature_failure=dict(target_id=old.get('id'),current_frame_id=int(frame['frame_id']),
                current_timestamp=float(frame['timestamp']),capture_status='capture_failed',
                capture_error=type(exc).__name__,reason='selected_surface_measurement_failed',
                failure_stage='surface_measurement',surface_measurement_reason=reason)
        return None,None,None,'selected_surface_'+str(reason)

    def _geometry_consistency(self,frame,old,target,warped,dt,cad_evidence=None):
        """Unchanged measured-motion and normal gates shared by both methods."""
        c=self.config
        displacement=float(np.linalg.norm(np.asarray(target['center_world'])-old['center_world']))
        if displacement>c.world_motion_slack_m+c.max_world_speed_m_s*dt:
            return None,'inconsistent_measured_world_motion'
        # Unexplained surface discontinuities often indicate an occluding hand.
        # Camera-induced range changes are excluded by comparing in world space.
        normal=np.asarray(old['normal_world'])
        if abs((np.asarray(target['center_world'])-old['center_world'])@normal)>c.max_depth_center_jump_m:
            return None,'occluding_depth_discontinuity'
        angle=float(np.degrees(np.arccos(np.clip(np.asarray(target['normal_world'])@normal,-1,1))))
        cad_surface=None
        if angle>c.max_normal_angle_deg:
            # A local patch can switch between different faces of one handle.
            # Only fresh accepted CAD identity plus newly measured depth on
            # that actual CAD surface can replace this normal heuristic.
            # The flow, depth-validity and world-motion gates above still apply.
            cad_surface=current_cad_surface_support(frame,warped,cad_evidence)
            if not cad_surface['accepted']:
                return None,'inconsistent_surface_normal'
        quality={'visible_world_displacement_m':displacement,'local_normal_change_deg':angle}
        if cad_surface is not None:quality['normal_change_CAD_surface_validation']=cad_surface
        return quality,None
