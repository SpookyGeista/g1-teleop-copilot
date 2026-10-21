"""Private FoundationPose worker with short, honestly timestamped CAD pose hold.

A visible mask centroid never translates the accepted object pose. The latest
actual model measurement is retained for at most 0.5 seconds with its original
acquisition time and explicit uncertainty growth.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import copy
import json
from pathlib import Path
import time
import numpy as np
from .model_bridge import ModelProcess
from .perception import _calibration
from .sensor_fusion import target_corners_world
from .multiview_tracker import LockedTargetWristRecovery


class PoseProcess(ModelProcess):
    def __init__(self,python=None,**config):
        root=Path(__file__).resolve().parents[1]
        config.setdefault('mesh_path',str(root/'assets/cad/handle_pose/PullHandleGrasp.obj'))
        config.setdefault('resolve_x_half_turn_with_world_up',True)
        super().__init__(python=python or root/'.venv-pose/bin/python',worker='pose_worker.py',**config)

    def estimate(self,frame,mask,target_id,reset=False):
        with self.lock:
            if self.closed:raise RuntimeError('Pose worker closed')
            self.counter+=1;stem=f'{self.counter:08d}'
            request=self.work/(stem+'.npz')
            k=frame['intrinsics']
            if isinstance(k,dict):k=[[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]]
            np.savez(request,rgb=frame['rgb'],depth=frame['depth'],intrinsics=k,
                     T_world_camera=frame['T_world_camera'],frame_id=frame['frame_id'],
                     timestamp=frame['timestamp'],**({'mask':np.asarray(mask,dtype=bool)} if mask is not None else {}))
            try:
                self.process.stdin.write(json.dumps({'stem':stem,'target_id':target_id,'reset':reset,
                    'recover_current_depth':mask is None})+'\n')
                self.process.stdin.flush()
                result=self._read(self.timeout)
                if result.get('stem')!=stem or result.get('status')!='ok':
                    raise RuntimeError('Pose inference failed: '+str(result))
                value=result['result']
                if value.get('recovered_from_current_depth'):
                    with np.load(self.work/(stem+'.recovered-mask.npz'),allow_pickle=False) as a:
                        observed=np.asarray(a['mask'],bool).copy()
                    if observed.shape!=frame['depth'].shape:raise ValueError('Recovery mask/frame shape mismatch')
                    value['_recovered_mask']=observed
                return value
            except BaseException:
                self.close();raise
            finally:
                request.unlink(missing_ok=True)
                (self.work/(stem+'.recovered-mask.npz')).unlink(missing_ok=True)


class PoseRuntime:
    def __init__(self,worker,interval=.2,current_frame_wait_s=.18):
        self.worker=worker
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='cad-pose')
        self.future=None;self.source=None;self.accepted=None
        self.last_submit=-float('inf');self.interval=interval
        self.current_frame_wait_s=max(0.,float(current_frame_wait_s))
        self.diagnostics={'status':'awaiting_observed_handle','backend':'foundationpose'}
        self.reset_next=True
        self.identity_confirmation=None
        self.completed_results_this_update=[]
        self.recovered_observation=None
        self.identity_reconciliation=None
        self._current_non_vetoing_proposals=set()
        self._active_frame_key=None
        self._last_submission_frame=None

    @staticmethod
    def _frame_key(frame):
        return (int(frame['frame_id']),float(frame['timestamp']))

    def _begin_frame(self,frame):
        key=self._frame_key(frame)
        if key!=self._active_frame_key:
            self.completed_results_this_update=[]
            self._active_frame_key=key

    def prefetch(self,frame,selected_id=None,*,recovery_frames=None):
        """Start current-depth CAD tracking before slower CPU image tracking.

        Call and ``update`` serially from the camera-processing thread. A single
        executor owns all native worker operations. Initial registration still
        needs the ordinary semantic target/mask in update; this entry point can
        only track the explicitly confirmed, fresh registered CAD identity.
        """
        self._begin_frame(frame);self._complete_future()
        prior=self.accepted;key=self._frame_key(frame)
        if (self.future is not None or key==self._last_submission_frame or self.reset_next
                or selected_id is None or not prior or prior['id']!=selected_id
                or prior.get('target',{}).get('semantic_label',prior.get('target',{}).get('label'))!='handle'
                or not prior['result'].get('accepted') or not prior['result'].get('mesh_sha256')
                or not 0<=float(frame['timestamp'])-prior['timestamp']<=.5
                or time.monotonic()-self.last_submit<self.interval):return False
        camera=prior.get('camera_id','head')
        view=(recovery_frames or {}).get(camera) if camera!='head' else frame
        if (view is None or int(view['frame_id'])!=int(frame['frame_id'])
                or abs(float(view['timestamp'])-float(frame['timestamp']))>.04):
            camera='head';view=frame
        view={**view,'camera_id':camera}
        self.source={k:copy.deepcopy(prior[k]) for k in ('id','target','center_world','normal_world')}
        self.source.update(timestamp=float(view['timestamp']),frame_id=int(view['frame_id']),
            camera_id=camera,reset=False,recovery=True,prefetched=True,observation_frame=view,
            submitted_monotonic=time.monotonic())
        # A small immutable pose snapshot permits lazy fallback ranking inside
        # the background job, without reading a newer main-thread measurement.
        rank_prior={'timestamp':prior['timestamp'],'result':copy.deepcopy(prior['result'])}
        alternatives=lambda:self._rank_recovery_views(frame,recovery_frames,camera,prior=rank_prior)
        self.future=self.pool.submit(self._estimate,view,None,selected_id,False,alternatives)
        self.last_submit=time.monotonic();self._last_submission_frame=key
        return True

    def _estimate(self,frame,mask,identity,reset,fallback_views=None):
        started=time.monotonic()
        result=self.worker.estimate(frame,mask,identity,reset)
        attempts=[{'camera_id':frame.get('camera_id','head'),'frame_id':int(frame['frame_id']),
            'accepted':bool(result.get('accepted')),'reason':result.get('reason'),'quality':result.get('quality')}]
        source_override=None
        if not reset and not result.get('accepted') and result.get('retained_tracking_prior'):
            if callable(fallback_views):fallback_views=fallback_views()
            for view in fallback_views or []:
                # The prior/native state was restored after the rejected view.
                # Each alternate is a genuinely current RGBD frame, never an
                # old camera image relabelled with the head's newer timestamp.
                alternate=self.worker.estimate(view,None,identity,False)
                attempts.append({'camera_id':view['camera_id'],'frame_id':int(view['frame_id']),
                    'accepted':bool(alternate.get('accepted')),'reason':alternate.get('reason'),
                    'quality':alternate.get('quality')})
                result=alternate
                source_override={'camera_id':view['camera_id'],'frame_id':int(view['frame_id']),
                    'timestamp':float(view['timestamp']),'recovery':True,'observation_frame':view}
                if result.get('accepted') or not result.get('retained_tracking_prior'):break
        completed=time.monotonic()
        return {'result':result,'roundtrip_seconds':completed-started,'completed_monotonic':completed,
                'source_override':source_override,'view_attempts':attempts}

    def _rank_recovery_views(self,head,frames,exclude,*,prior=None):
        prior=self.accepted if prior is None else prior
        if not prior or not 0<=float(head['timestamp'])-prior['timestamp']<=.5:return []
        from .cad_surface import CATALOG,_triangles,point_triangle_surface_distance
        try:
            measured=prior['result'];pose=np.asarray(measured['T_world_object'])
            triangles=_triangles('PullHandleGrasp',measured['mesh_sha256'],str(CATALOG),CATALOG.stat().st_mtime_ns)
            world=triangles.reshape(-1,3)@pose[:3,:3].T+pose[:3,3]
        except (KeyError,ValueError,OSError):return []
        ranked=[]
        for camera in ('head','right_wrist','left_wrist'):
            view=(frames or {}).get(camera)
            if view is None or camera==exclude:continue
            if (int(view['frame_id'])!=int(head['frame_id'])
                    or abs(float(view['timestamp'])-float(head['timestamp']))>.04):continue
            k,transform=_calibration(view['intrinsics'],view['T_world_camera'])
            points=(world-transform[:3,3])@transform[:3,:3]
            if np.min(points[:,2])<=.05:continue
            uv=np.column_stack((points[:,0]*k['fx']/points[:,2]+k['cx'],points[:,1]*k['fy']/points[:,2]+k['cy']))
            h,w=view['depth'].shape;lo=np.maximum(np.floor(uv.min(0)).astype(int),[0,0]);hi=np.minimum(np.ceil(uv.max(0)).astype(int),[w,h])
            if np.any(hi-lo<8):continue
            yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];z=view['depth'][yy,xx]
            valid=np.isfinite(z)&(z>.05)&(z<3.);yy,xx,z=yy[valid],xx[valid],z[valid]
            if len(z)<80:continue
            take=np.linspace(0,len(z)-1,min(384,len(z)),dtype=int);yy,xx,z=yy[take],xx[take],z[take]
            camera_points=np.column_stack(((xx-k['cx'])*z/k['fx'],(yy-k['cy'])*z/k['fy'],z))
            object_camera=np.linalg.inv(pose)@transform;local=camera_points@object_camera[:3,:3].T+object_camera[:3,3]
            support=point_triangle_surface_distance(local,triangles)<=.012
            # Ranking/proposal only: final acceptance still uses full rendered
            # silhouette, residual and observability gates in the actual worker.
            if support.sum()<8:continue
            ranked.append((int(support.sum()),{**view,'camera_id':camera}))
        return [view for _,view in sorted(ranked,key=lambda item:item[0],reverse=True)]

    def _complete_future(self):
        if self.future is not None and self.future.done():
            completed=self.future.result();self.future=None
            if completed.get('source_override'):self.source={**self.source,**completed['source_override']}
            result=completed['result']
            recovered_mask=result.pop('_recovered_mask',None)
            if self.source.get('recovery') and result.get('accepted'):
                same_measurement=(int(result.get('frame_id',-1))==self.source['frame_id']
                    and abs(float(result.get('timestamp',-1))-self.source['timestamp'])<=1e-6
                    and result.get('target_id',self.source['id'])==self.source['id'])
                if not same_measurement:
                    result={**result,'accepted':False,'reason':'recovery_result_acquisition_mismatch',
                            'retained_tracking_prior':False};recovered_mask=None
            self.diagnostics={'status':'accepted' if result.get('accepted') else 'rejected',
                'backend':'foundationpose','result':result,'target_id':self.source['id'],
                'acquisition_timestamp':self.source['timestamp'],
                'observation_camera':self.source.get('camera_id','head'),
                'prefetched_before_image_tracking':bool(self.source.get('prefetched',False)),
                'submitted_monotonic':self.source.get('submitted_monotonic'),
                'worker_roundtrip_ms':completed['roundtrip_seconds']*1000,
                'worker_completed_monotonic':completed['completed_monotonic'],
                'consumed_monotonic':time.monotonic()}
            self.diagnostics['view_attempts']=completed.get('view_attempts',[])
            # An older pending result and the new same-frame result can both
            # complete during one update. Preserve both timing events rather
            # than silently overwriting the scheduling tail in diagnostics.
            self.completed_results_this_update.append({
                'frame_id':result.get('frame_id',self.source.get('frame_id')),
                'timestamp':result.get('timestamp',self.source['timestamp']),
                'accepted':bool(result.get('accepted')),'reason':result.get('reason'),
                'operation':result.get('operation'),'worker_roundtrip_ms':completed['roundtrip_seconds']*1000,
                'worker_completed_monotonic':completed['completed_monotonic'],
                'consumed_monotonic':self.diagnostics['consumed_monotonic']})
            if result.get('accepted'):
                self.accepted={**self.source,'result':result,'recovered_mask':recovered_mask};self.reset_next=False
            else:
                # A rejected update is not a measurement, but it does not
                # erase the preceding accepted measurement. _annotate keeps
                # its original time and the existing 0.5-second expiry.
                same_identity=bool(self.accepted and self.accepted['id']==self.source['id'])
                if not same_identity:self.accepted=None
                self.reset_next=not (same_identity and result.get('retained_tracking_prior',False))
                self.diagnostics['retained_last_accepted_measurement']=same_identity

    def _annotate(self,frame,targets):
        by_id={target['id']:target for target in targets}
        self.identity_confirmation=None
        if self.accepted:
            prior=self.accepted;target=by_id.get(prior['id'])
            age=float(frame['timestamp'])-prior['timestamp']
            if target is not None and 0<=age<=.5:
                # A changing visible-mask centroid is not object motion. Keep
                # the last actual CAD measurement and its acquisition time.
                measured=prior['result'];q=measured['quality']
                sigma=max(.012,float(measured.get('pose_position_sigma_m',.025)),
                          float(q['p90_depth_error_m'])+.004+age*.020)
                orientation_sigma=max(.10,float(measured.get('pose_orientation_sigma_rad',.35)))
                target.update(T_world_object=measured['T_world_object'],cad_asset_id='PullHandleGrasp',
                    pose_source='foundationpose',pose_valid=sigma<=.025 and orientation_sigma<=.35,
                    pose_timestamp=prior['timestamp'],pose_registration_timestamp=prior['timestamp'],
                    pose_age_s=age,pose_tracking_method='last_accepted_CAD_measurement_without_centroid_translation',
                    pose_position_sigma_m=sigma,pose_orientation_sigma_rad=orientation_sigma,
                    pose_uncertainty_source='conservative_operating_bounds_not_calibrated_probability',pose_quality=q)
                if self.identity_reconciliation is not None:
                    target['identity_reconciliation']=copy.deepcopy(self.identity_reconciliation)
                if target['pose_valid']:
                    self.identity_confirmation={'kind':'foundationpose_rgbd','accepted':True,
                        'target_id':prior['id'],'timestamp':prior['timestamp'],
                        'frame_id':prior.get('frame_id',measured.get('frame_id',-1)),
                        'camera_id':prior.get('camera_id','head'),'center_world':prior['center_world'].tolist(),
                        'T_world_object':measured['T_world_object'],'cad_asset_id':'PullHandleGrasp',
                        'mesh_sha256':measured.get('mesh_sha256'),
                        'quality':q,'pose_position_sigma_m':sigma,'pose_orientation_sigma_rad':orientation_sigma}
                    if self.recovered_observation is not None:
                        self.identity_confirmation['_recovered_observation']=self.recovered_observation

    def _reconcile_current_identity(self,frame,targets,masks,selected_id):
        """Validate neural re-identification from this frame's actual CAD pose."""
        self.identity_reconciliation=None
        self._current_non_vetoing_proposals=set()
        prior=self.accepted
        if (not prior or selected_id is None or prior['id']!=selected_id
                or prior.get('frame_id')!=int(frame['frame_id'])
                or abs(prior['timestamp']-float(frame['timestamp']))>.04):return True
        from .cad_identity import reconcile_current_proposals
        evidence={**prior['result'],'cad_asset_id':'PullHandleGrasp'}
        result=reconcile_current_proposals(frame,targets,masks,selected_id,evidence,prior['center_world'])
        self.diagnostics['identity_reconciliation']=result
        if result['status']=='ambiguous':
            # Preserve the other proposals for inspection. The confirmed ID is
            # withheld; multiple compatible or conflicting proposals are not
            # silently converted into a persistent alias.
            targets[:]=[t for t in targets if t['id']!=selected_id]
            self.diagnostics['recovery_publication_rejected']='ambiguous_current_CAD_identity_proposals'
            return False
        if result['status']=='unique_duplicate':
            targets[:]=[t for t in targets if t['id']!=result['proposal_id']]
            self.identity_reconciliation=result
        non_vetoing=result.get('non_vetoing_unassigned_proposal_ids',[])
        if non_vetoing:
            self._current_non_vetoing_proposals=set(non_vetoing)
            self.identity_reconciliation=result
            for target in targets:
                if target['id'] in self._current_non_vetoing_proposals:
                    target['association_to_confirmed_CAD']={
                        'status':'unassigned_current_depth_disproves_association',
                        'confirmed_id':selected_id,'frame_id':int(frame['frame_id']),
                        'cad_timestamp':self.accepted['timestamp'],'persistent_alias':False}
        return True

    def _publish_current_recovery(self,frame,targets,selected_id):
        """Recreate a missing selected ID only from this frame's accepted model result."""
        self.recovered_observation=None
        prior=self.accepted
        if not prior or not prior.get('recovery') or prior['id']!=selected_id:return
        source_frame=prior.get('observation_frame',frame)
        camera=prior.get('camera_id','head')
        if (prior.get('frame_id')!=int(frame['frame_id'])
                or abs(prior['timestamp']-float(source_frame['timestamp']))>1e-6
                or abs(prior['timestamp']-float(frame['timestamp']))>.04):return
        existing=next((t for t in targets if t['id']==selected_id),None)
        if existing is not None and camera=='head' and not prior.get('prefetched'):return
        mask=prior.get('recovered_mask')
        if mask is None or not prior['result'].get('recovered_from_current_depth'):return
        from .learned_perception import mask_to_target
        old=prior['target'];measured,reason=mask_to_target(mask,source_frame['depth'],source_frame['intrinsics'],source_frame['T_world_camera'],
            label=old.get('semantic_phrase','handle'),detection_score=old.get('detection_confidence',old.get('confidence',.5)),
            mask_score=old.get('segmentation_confidence',.8))
        if measured is None:
            self.diagnostics['recovery_publication_rejected']=reason;return
        if any(t['id']!=selected_id and t['id'] not in self._current_non_vetoing_proposals
               and np.linalg.norm(np.asarray(t['center_world'])-measured['center_world'])<.04 for t in targets):
            self.diagnostics['recovery_publication_rejected']='ambiguous_neighbor_near_current_CAD_surface';return
        semantic_time=float(old.get('semantic_keyframe_timestamp',old.get('timestamp',prior['timestamp'])))
        measured.update(id=selected_id,frame_id=int(source_frame['frame_id']),timestamp=float(source_frame['timestamp']),
            semantic_keyframe_timestamp=semantic_time,semantic_keyframe_age_s=float(frame['timestamp'])-semantic_time,
            perception_method='foundationpose_current_depth_CAD_recovery',
            mask_source='current_measured_depth_near_previous_accepted_CAD_surface',
            normal_source='current_CAD_surface_mask_local_measured_depth',
            segmentation_confidence_source='original_SAM_proposal_not_current_depth_mask',
            measurement_sources=[camera+'_rgbd','known_CAD','foundationpose'],head_visible=camera=='head',observation_camera=camera)
        target=measured
        if camera!='head':
            target=LockedTargetWristRecovery._head_overlay_target(measured,measured,np.asarray(measured['center_world']),
                target_corners_world(measured,source_frame),frame)
            if target is None:
                self.diagnostics['recovery_publication_rejected']='current_wrist_geometry_cannot_project_into_head';return
            target.update(frame_id=int(frame['frame_id']),timestamp=float(source_frame['timestamp']))
        if existing is None:targets.append(target)
        else:existing.clear();existing.update(target)
        prior['center_world']=np.asarray(measured['center_world']).copy();prior['target']=copy.deepcopy(target)
        self.recovered_observation={'camera_id':camera,'frame':source_frame,'mask':mask,
                                    'target':copy.deepcopy(measured),'head_target':copy.deepcopy(target)}
        self.diagnostics['current_frame_recovered_id']=selected_id

    def update(self,frame,targets,masks,selected_id=None,observations=None,recovery_frames=None):
        observations=observations or {}
        self._begin_frame(frame)
        self.recovered_observation=None
        for target in targets:
            if target.get('pose_source','').startswith('foundationpose'):target['pose_valid']=False
        self._complete_future()
        if (self.future is None and self._frame_key(frame)!=self._last_submission_frame
                and time.monotonic()-self.last_submit>=self.interval):
            available={identity:{'camera_id':'head','frame':frame,'mask':mask} for identity,mask in masks.items()}
            for identity,observation in observations.items():
                source_frame=observation['frame']
                if abs(float(source_frame['timestamp'])-float(frame['timestamp']))<=.04:
                    available[identity]=observation
            candidates=[t for t in targets if t.get('semantic_label',t.get('label'))=='handle' and t['id'] in available]
            if selected_id is not None:candidates=[t for t in candidates if t['id']==selected_id]
            if len(candidates)==1:
                target=candidates[0]
                normal=np.asarray(target['normal_world'],float);normal/=max(np.linalg.norm(normal),1e-9)
                observation=available[target['id']];source_frame=observation['frame']
                self.source={'id':target['id'],'timestamp':float(source_frame['timestamp']),
                    'camera_id':observation.get('camera_id','head'),'frame_id':int(source_frame['frame_id']),
                    'center_world':np.asarray(target['center_world']).copy(),'normal_world':normal,
                    'target':copy.deepcopy(target),'recovery':False}
                reset=self.reset_next or not self.accepted or self.accepted['id']!=target['id']
                self.source['reset']=reset
                alternatives=self._rank_recovery_views(frame,recovery_frames,observation.get('camera_id','head')) if selected_id==target['id'] and not reset else []
                self.future=self.pool.submit(self._estimate,source_frame,observation['mask'],target['id'],reset,alternatives)
                self.last_submit=time.monotonic();self._last_submission_frame=self._frame_key(frame)
            elif (selected_id is not None and self.accepted and self.accepted['id']==selected_id
                  and not any(t['id']==selected_id for t in targets) and not self.reset_next
                  and 'target' in self.accepted
                  and 0<=float(frame['timestamp'])-self.accepted['timestamp']<=.5):
                # Independent CAD tracking must get a current sensor observation
                # even when upstream optical-flow correspondence has failed.
                # The missing target stays absent unless this same-frame model
                # result passes every existing pose quality/observability gate.
                self.source={key:copy.deepcopy(self.accepted[key]) for key in ('id','target','center_world','normal_world')}
                self.source.update(timestamp=float(frame['timestamp']),frame_id=int(frame['frame_id']),
                    camera_id='head',reset=False,recovery=True)
                alternatives=self._rank_recovery_views(frame,recovery_frames,'head')
                self.future=self.pool.submit(self._estimate,frame,None,selected_id,False,alternatives)
                self.last_submit=time.monotonic();self._last_submission_frame=self._frame_key(frame)
        # Typical warm tracking is 17-30 ms, but concurrent Isaac/semantic GPU
        # scheduling exceeded the former 80-ms budget. Permit a bounded 180-ms
        # tail in this camera pass, avoiding a full extra publication period.
        # Registration remains asynchronous. Neither waiting nor timeout can
        # change acquisition timestamps or bypass the 0.5-second pose gate.
        wait_started=time.monotonic();timed_out=False
        if self.future is not None and not self.source.get('reset',True) and self.current_frame_wait_s:
            try:self.future.result(timeout=self.current_frame_wait_s)
            except TimeoutError:timed_out=True
        self._complete_future()
        if self._reconcile_current_identity(frame,targets,masks,selected_id):
            self._publish_current_recovery(frame,targets,selected_id)
        self._annotate(frame,targets)
        self.diagnostics.update(current_frame_pose_wait_ms=(time.monotonic()-wait_started)*1000,
            current_frame_wait_timed_out=timed_out,
            current_frame_wait_budget_ms=self.current_frame_wait_s*1000,
            completed_results_this_update=list(self.completed_results_this_update),
            pose_result_consumed_in_same_frame=bool(self.accepted and self.accepted.get('frame_id')==int(frame['frame_id'])))
        return targets

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.worker.close()
