"""Recover a confirmed target through a fresh calibrated wrist observation.

This module retains only a short geometric hypothesis for the selected identity.
That hypothesis proposes a SAM box; a target is returned only after a fresh
neural mask and current wrist depth agree. It does not fill a hidden head mask.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import numpy as np
import cv2

from .learned_perception import LearnedPerceptionConfig, mask_to_target
from .sensor_fusion import project, target_corners_world
from .perception import _calibration, backproject


@dataclass(frozen=True)
class WristRecoveryConfig:
    max_observation_age_s: float = 1.2
    max_semantic_age_s: float = 30.
    max_sensor_skew_s: float = .04
    minimum_sam_score: float = .70
    volume_slack_m: float = .025
    minimum_depth_volume_fraction: float = .80
    max_anchor_change_m: float = .040
    minimum_bbox_visible_fraction: float = .65


class LockedTargetWristRecovery:
    def __init__(self, config=None):
        self.config=WristRecoveryConfig(**config) if isinstance(config,dict) else config or WristRecoveryConfig()
        self._memory={}

    def observe(self,head_frame,targets,masks=None):
        """Cache current head semantic measurements, without changing identity."""
        timestamp=float(head_frame['timestamp'])
        for target in targets:
            if target.get('stale') or not target.get('observed',True):
                continue
            self._memory[target['id']]={'target':deepcopy(target),
                'corners_world':target_corners_world(target,head_frame),
                'timestamp':timestamp,'semantic_timestamp':timestamp,
                'surface_points_world':self._surface_points(head_frame,(masks or {}).get(target['id']))}
        self._memory={key:item for key,item in self._memory.items()
                      if timestamp-item['timestamp']<=self.config.max_observation_age_s}

    def recover(self,selected_target_id,head_frame,wrist_frames,segment_boxes,other_targets=(),segment_points=None):
        """Return measured target + source observations, or an explicit loss.

        ``segment_boxes`` is the real pipeline SAM method. ``observations``
        retains the actual wrist frame/mask for a CAD pose worker. The returned
        target's overlay box is projected into the head frame, but no head mask
        is manufactured. Only call this for an explicitly confirmed missing ID.
        """
        c=self.config;now=float(head_frame['timestamp']);diagnostics={'method':'confirmed_identity_wrist_recovery',
                'selected_target_id':selected_target_id,'scene_truth_used':False,'views':{}}
        memory=self._memory.get(selected_target_id)
        if memory is None:
            return {'targets':[],'observations':{},'diagnostics':{**diagnostics,'reason':'no_confirmed_identity_history'}}
        if now-memory['timestamp']>c.max_observation_age_s or now-memory['semantic_timestamp']>c.max_semantic_age_s:
            return {'targets':[],'observations':{},'diagnostics':{**diagnostics,'reason':'identity_history_expired'}}
        old=memory['target'];corners=np.asarray(memory['corners_world'])
        candidates=[]
        for name,wrist in wrist_frames.items():
            record={};diagnostics['views'][name]=record
            if abs(float(wrist['timestamp'])-now)>c.max_sensor_skew_s:
                record['reason']='stale_wrist_frame';continue
            uv,z=project(corners,wrist)
            if not np.isfinite(uv).all() or min(z)<.08 or max(z)>1.5:
                record['reason']='outside_wrist_depth_range';continue
            lo,hi=uv.min(axis=0)-5,uv.max(axis=0)+5
            h,w=wrist['depth'].shape
            clo=np.maximum(lo,[1,1]);chi=np.minimum(hi,[w-2,h-2])
            visible=float(np.prod(np.maximum(0,chi-clo))/max(1,np.prod(hi-lo)))
            if visible<c.minimum_bbox_visible_fraction or min(chi-clo)<8:
                record['reason']='outside_wrist_field_of_view';continue
            box=[*clo.tolist(),*chi.tolist()]
            segments=[('box',segment) for segment in segment_boxes(wrist['rgb'],[box])]
            if segment_points is not None:
                positive,negative=self._point_prompts(memory,wrist,clo,chi)
                record['positive_points']=positive;record['negative_points']=negative
                if len(positive)>=3:
                    segments.extend(('measured_rgbd_points',segment) for segment in
                        segment_points(wrist['rgb'],box,positive,negative))
            for prompt_mode,segment in segments:
                if float(segment['confidence'])<c.minimum_sam_score:
                    record['reason']='uncertain_wrist_segmentation';continue
                mask=np.asarray(segment['mask'],dtype=bool)
                if len(memory.get('surface_points_world',[])):
                    mask,component_support=self._supported_components(wrist,mask,memory['surface_points_world'])
                    record['component_support']=component_support
                    if not mask.any():
                        record['reason']='no_component_supported_by_previous_measured_surface';continue
                k,t=_calibration(wrist['intrinsics'],wrist['T_world_camera'])
                valid=mask&np.isfinite(wrist['depth'])&(wrist['depth']>.08)&(wrist['depth']<1.5)
                ys,xs=np.nonzero(valid)
                if len(xs)<40:
                    record['reason']='insufficient_wrist_depth';continue
                points=backproject(np.column_stack((xs,ys)),wrist['depth'][ys,xs],k)@t[:3,:3].T+t[:3,3]
                lo_world,hi_world=corners.min(axis=0)-c.volume_slack_m,corners.max(axis=0)+c.volume_slack_m
                inside=np.all((points>=lo_world)&(points<=hi_world),axis=1)
                support=float(inside.mean())
                if support<c.minimum_depth_volume_fraction:
                    record.update(reason='wrist_mask_conflicts_with_known_target_volume',depth_volume_fraction=support);continue
                measured,reason=mask_to_target(mask,wrist['depth'],wrist['intrinsics'],wrist['T_world_camera'],
                    label=old.get('semantic_phrase',old['label']),detection_score=old.get('detection_confidence',.3),
                    mask_score=float(segment['confidence']),
                    config=replace(LearnedPerceptionConfig(),max_mask_fraction=.80))
                if measured is None:
                    record['reason']=reason;continue
                delta=np.asarray(measured['center_world'])-old['center_world']
                distance=float(np.linalg.norm(delta))
                if distance>c.max_anchor_change_m:
                    record.update(reason='wrist_anchor_disagreement',anchor_change_m=distance);continue
                neighbours=[np.linalg.norm(np.asarray(target['center_world'])-measured['center_world'])
                            for target in other_targets if target['id']!=selected_target_id]
                if neighbours and min(neighbours)<distance+.025:
                    record['reason']='ambiguous_neighbour';continue
                # Keep visual surface drift bounded; FoundationPose should determine
                # the action frame from the actual wrist image/mask separately.
                correction=delta.copy()
                if distance>.010:
                    correction*=.010/distance
                anchor=np.asarray(old['center_world'])+correction
                moved_corners=corners+correction
                if name=='head':
                    out={**deepcopy(old),**measured}
                    anchor=np.asarray(measured['center_world'])
                    moved_corners=target_corners_world(out,head_frame)
                else:
                    out=self._head_overlay_target(old,measured,anchor,moved_corners,head_frame)
                if out is None:
                    record['reason']='target_behind_head_camera';continue
                out.update(id=selected_target_id,frame_id=int(head_frame['frame_id']),timestamp=now,
                    head_visible=name=='head',observation_camera=name,observed=True,stale=False,
                    measurement_sources=[name+'_rgbd'],perception_method='confirmed_identity_wrist_sam2_rgbd' if name!='head' else 'confirmed_identity_head_sam2_rgbd',
                    semantic_keyframe_timestamp=memory['semantic_timestamp'],
                    semantic_keyframe_age_s=now-memory['semantic_timestamp'],
                    wrist_recovery={'mask_confidence':float(segment['confidence']),
                                    'depth_volume_fraction':support,'anchor_difference_m':distance,
                                    'source_timestamp':float(wrist['timestamp'])})
                out['position_sigma_m']=max(.010,measured['position_sigma_m'])
                record.update(reason='observed_in_'+name,depth_volume_fraction=support,anchor_change_m=distance,prompt_mode=prompt_mode)
                candidates.append((support*float(segment['confidence']),out,moved_corners,wrist,mask,name,deepcopy(record)))
        if not candidates:
            diagnostics['reason']='no_valid_current_wrist_observation'
            return {'targets':[],'observations':{},'diagnostics':diagnostics}
        _,target,newcorners,frame,mask,name,accepted_record=max(candidates,key=lambda candidate:(candidate[5]=='head',candidate[0]))
        diagnostics['views'][name]=accepted_record
        self._memory[selected_target_id]={'target':deepcopy(target),'corners_world':newcorners,
            'timestamp':now,'semantic_timestamp':memory['semantic_timestamp'],
            'surface_points_world':self._surface_points(frame,mask)}
        diagnostics['reason']='confirmed_target_observed_in_wrist'
        return {'targets':[target],'observations':{selected_target_id:{'camera_id':name,'frame':frame,'mask':mask}},
                'diagnostics':diagnostics}

    def filter_confirmed_mask(self,identity,frame,mask):
        prior=self._memory.get(identity)
        if prior is None or not len(prior.get('surface_points_world',[])):
            return np.asarray(mask,dtype=bool),{'status':'no_previous_surface'}
        if not 0<=float(frame['timestamp'])-prior['timestamp']<=self.config.max_observation_age_s:
            return np.zeros_like(mask,dtype=bool),{'status':'expired_previous_surface'}
        return self._supported_components(frame,mask,prior['surface_points_world'])

    @staticmethod
    def _supported_components(frame,mask,points_world):
        """Keep actual SAM pixels only in components with measured correspondence."""
        mask=np.asarray(mask,dtype=bool);h,w=mask.shape
        uv,z=project(points_world,frame)
        visible=np.isfinite(uv).all(1)&(z>.08)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        uv,z=uv[visible],z[visible]
        xy=np.rint(uv).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
        depth=frame['depth'][xy[:,1],xy[:,0]]
        supported=np.isfinite(depth)&(abs(depth-z)<.012)
        xy=xy[supported]
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
        observed_pixels=np.zeros(mask.shape,np.uint8)
        if len(xy):observed_pixels[xy[:,1],xy[:,0]]=1
        supported_area=cv2.dilate(observed_pixels,np.ones((5,5),np.uint8)).astype(bool)
        keep=np.zeros(mask.shape,bool);components=[]
        for component in range(1,count):
            component_mask=labels==component
            # A two-pixel boundary tolerance only associates measured points;
            # dilation is not copied into the returned object mask.
            association=cv2.dilate(component_mask.astype(np.uint8),np.ones((5,5),np.uint8))
            hits=int(association[xy[:,1],xy[:,0]].sum()) if len(xy) else 0
            pixels=int(stats[component,cv2.CC_STAT_AREA])
            coverage=float(np.count_nonzero(component_mask&supported_area)/max(1,pixels))
            accepted=hits>=3 and pixels>=24 and coverage>=.20
            if accepted:keep|=component_mask
            components.append({'pixels':int(stats[component,cv2.CC_STAT_AREA]),
                               'measured_surface_support_points':hits,'measured_pixel_coverage':coverage,'accepted':accepted})
        return keep,{'status':'component_correspondence_checked','components':components,
                     'removed_pixels':int(mask.sum()-keep.sum()),'retained_pixels':int(keep.sum())}

    @staticmethod
    def _surface_points(frame,mask):
        if mask is None:return np.empty((0,3))
        depth=np.asarray(frame['depth'])
        valid=np.asarray(mask,dtype=bool)&np.isfinite(depth)&(depth>.08)&(depth<1.5)
        y,x=np.nonzero(valid)
        if not len(x):return np.empty((0,3))
        # Deterministic spatial sample from actual pixels; no filled geometry.
        index=np.linspace(0,len(x)-1,min(len(x),800)).astype(int)
        k,t=_calibration(frame['intrinsics'],frame['T_world_camera'])
        return backproject(np.column_stack((x[index],y[index])),depth[y[index],x[index]],k)@t[:3,:3].T+t[:3,3]

    @staticmethod
    def _spread_points(points,count=6,separation=6.):
        points=np.asarray(points,dtype=float)
        if not len(points):return []
        selected=[int(np.argmin(np.linalg.norm(points-np.median(points,axis=0),axis=1)))]
        distances=np.linalg.norm(points-points[selected[0]],axis=1)
        for _ in range(count-1):
            index=int(np.argmax(distances))
            if distances[index]<separation:break
            selected.append(index);distances=np.minimum(distances,np.linalg.norm(points-points[index],axis=1))
        return points[selected].tolist()

    def _point_prompts(self,memory,frame,lo,hi):
        """Reproject visible measured surface samples; reject occluded pixels."""
        points=np.asarray(memory.get('surface_points_world',[]))
        if points.ndim!=2 or len(points)<3:return [],[]
        uv,z=project(points,frame);h,w=frame['depth'].shape
        good=np.isfinite(uv).all(axis=1)&(z>.08)&(uv[:,0]>=lo[0])&(uv[:,1]>=lo[1])&(uv[:,0]<hi[0])&(uv[:,1]<hi[1])
        uv,z=uv[good],z[good]
        if len(uv)<3:return [],[]
        xy=np.rint(uv).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
        current_depth=frame['depth'][xy[:,1],xy[:,0]]
        consistent=np.isfinite(current_depth)&(abs(current_depth-z)<.012)
        positive=self._spread_points(xy[consistent],count=6)
        if len(positive)<3:return [],[]
        # Negative points are real foreground/background depth outside the
        # remembered target volume, not inferred simulator semantic labels.
        x0,y0=np.floor(lo).astype(int);x1,y1=np.ceil(hi).astype(int)
        yy,xx=np.mgrid[y0:y1:3,x0:x1:3];depth=frame['depth'][yy,xx]
        valid=np.isfinite(depth)&(depth>.08)&(depth<1.5)
        pixels=np.column_stack((xx[valid],yy[valid]))
        k,t=_calibration(frame['intrinsics'],frame['T_world_camera'])
        measured=backproject(pixels,depth[valid],k)@t[:3,:3].T+t[:3,3]
        corners=np.asarray(memory['corners_world'])
        outside=np.any((measured<corners.min(0)-self.config.volume_slack_m)|(measured>corners.max(0)+self.config.volume_slack_m),axis=1)
        negatives=pixels[outside]
        if len(negatives):
            away=np.min(np.linalg.norm(negatives[:,None,:]-np.asarray(positive)[None,:,:],axis=2),axis=1)>7.
            negatives=negatives[away]
        return positive,self._spread_points(negatives,count=8)

    @staticmethod
    def _head_overlay_target(old,measured,anchor,corners,head):
        k,t=_calibration(head['intrinsics'],head['T_world_camera'])
        camera_corners=(corners-t[:3,3])@t[:3,:3]
        if min(camera_corners[:,2])<=.01:
            return None
        lo,hi=camera_corners.min(axis=0),camera_corners.max(axis=0)
        uv,_=project(corners,head)
        plo,phi=uv.min(axis=0),uv.max(axis=0)
        out=deepcopy(old)
        out.update(center_world=anchor.tolist(),center_camera=((anchor-t[:3,3])@t[:3,:3]).tolist(),
            box_camera={'center':((lo+hi)/2).tolist(),'size':(hi-lo).tolist(),'quaternion_xyzw':[0,0,0,1]},
            bbox_px=[*plo.tolist(),*phi.tolist()],polygon_px=[plo.tolist(),[phi[0],plo[1]],phi.tolist(),[plo[0],phi[1]]],
            pose_confidence=min(old.get('pose_confidence',.7),measured['pose_confidence']))
        out.pop('surface_pixel_px',None)
        return out
