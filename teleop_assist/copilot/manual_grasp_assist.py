"""Explicit, bounded learned grasp alignment on the working manual sim path.

This experiment aligns the right hand to a selected GraspGenX proposal. It is
not a general obstacle-avoidance planner and opens no hardware transport.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from .controller import rigid


def limited(vector, maximum):
    vector=np.asarray(vector,dtype=float)
    length=np.linalg.norm(vector)
    return vector*min(1.,maximum/max(length,1e-12))


def wrist_support(frame, centroid_world, *, now, clock_id, camera='right_wrist'):
    """Independent measured wrist depth near the learned head RGB-D target."""
    result=dict(accepted=False,reason='secondary_depth_unavailable',camera=camera)
    try:
        if (frame['camera_source']!=camera or frame['clock_id']!=clock_id
                or not -.01<=now-frame['timestamp']<=.25):
            return dict(result,reason='wrist_depth_stale_or_wrong_source')
        depth=np.asarray(frame['depth']);k=np.asarray(frame['intrinsics'])
        pose=rigid(frame['T_world_camera']);center=np.asarray(centroid_world,dtype=float)
        if center.shape!=(3,) or not np.isfinite(center).all():return result
        stride=max(1,round(depth.shape[1]/213))
        small=depth[::stride,::stride]
        ys,xs=np.nonzero(np.isfinite(small)&(small>.10)&(small<2.))
        ys,xs=ys*stride,xs*stride;z=depth[ys,xs]
        points=np.column_stack(((xs-k[0,2])*z/k[0,0],(ys-k[1,2])*z/k[1,1],z))@pose[:3,:3].T+pose[:3,3]
        nearby=points[np.linalg.norm(points-center,axis=1)<.05]
        result.update(frame_id=int(frame['frame_id']),capture_timestamp=float(frame['timestamp']),
            support_points=len(nearby),scope='local_depth_consistency_not_obstacle_free_path')
        if len(nearby)>=12:
            discrepancy=float(np.linalg.norm(np.median(nearby,axis=0)-center))
            result.update(accepted=discrepancy<.035,reason='wrist_depth_consistent' if discrepancy<.035 else 'wrist_head_disagreement',
                centroid_difference_m=discrepancy)
        else:result['reason']='target_not_resolved_in_wrist_depth'
    except (KeyError,ValueError,TypeError,IndexError):pass
    return result


class ManualGraspAssist:
    def __init__(self, *, simulation_mode, support_camera='right_wrist',guidance_profile='fine_alignment'):
        if simulation_mode is not True:raise ValueError('Learned grasp alignment is simulation-only')
        if support_camera not in ('head','right_wrist'):raise ValueError('Unsupported secondary depth camera')
        self.support_camera=support_camera
        self.selected=None;self.confirmed=None;self.candidates=[];self.batch={}
        self.translation=np.zeros(3);self.rotation=Rotation.identity()
        self.last_time=None;self.last_revision=None;self.wrist={};self.reason='waiting_for_learned_proposals'
        self.applied=False;self.max_translation=.03;self.max_rotation=np.deg2rad(20.)
        if guidance_profile not in ('fine_alignment','early_guidance'):raise ValueError('Unknown guidance profile')
        self.guidance_profile=guidance_profile
        self.capture_radius=.15;self.full_guidance_radius=.15
        self.translation_speed=.03;self.rotation_speed=.35
        if guidance_profile=='early_guidance':
            self.capture_radius=.40;self.full_guidance_radius=.25
            self.max_translation=.10;self.max_rotation=np.deg2rad(45.)
            self.translation_speed=.05;self.rotation_speed=.50
        self.gain=0.;self.distance=None;self.depth_mode='waiting'

    def reset_offset(self):
        self.translation[:]=0;self.rotation=Rotation.identity();self.last_time=None;self.applied=False

    def update(self,batch,ui,wrist_frame,*,now,clock_id,active):
        self.batch=batch if isinstance(batch,dict) else {}
        stamp=self.batch.get('capture_timestamp');processed=self.batch.get('processed_frame',{})
        self.fresh=(type(stamp) in (int,float) and self.batch.get('clock_id')==clock_id
            and -.01<=now-stamp<=.25 and self.batch.get('motion_static') is True
            and processed.get('rejected') is False)
        self.candidates=[{key:deepcopy(c.get(key)) for key in ('candidate_id','base_ee','world_canonical','model_score',
            'geometry_valid','ik_valid','recorded_sim_endpoint_clear','source')}
            for c in self.batch.get('candidates',[]) if isinstance(c,dict)
            and c.get('geometry_valid') is True and c.get('ik_valid') is True
            and c.get('recorded_sim_endpoint_clear') is True
            and c.get('source')=='GraspGenX_pretrained_current_Dex3'] if self.fresh else []
        self.wrist=wrist_support(wrist_frame,self.batch.get('observed_centroid_world'),now=now,clock_id=clock_id,
            camera=self.support_camera) if wrist_frame else {}
        revision=ui.get('revision');edge=type(revision) is int and revision!=self.last_revision
        if edge:
            self.last_revision=revision
            if ui.get('release'):
                self.selected=self.confirmed=None;self.reset_offset()
            # Calibration rebases the operator's hands, not the chosen object
            # grasp. Keep the frozen choice/confirmation; apply() still checks
            # fresh proposals, the matching pose and wrist depth on every use.
            if ui.get('clutch') or ui.get('start'):self.reset_offset()
            if ui.get('select_candidate_id'):
                choice=next((c for c in self.candidates if c['candidate_id']==ui['select_candidate_id']),None)
                if choice:self.selected=deepcopy(choice);self.confirmed=None
            if ui.get('confirm') and self.selected and self.fresh:
                current=next((c for c in self.candidates if c['candidate_id']==self.selected['candidate_id']),None)
                if current and np.allclose(current['base_ee'],self.selected['base_ee'],atol=1e-8,rtol=0):
                    self.confirmed=deepcopy(self.selected)
        if not active:self.reset_offset()
        self.held=ui.get('assist_held') is True
        self.reason=('learned_candidates_ready' if self.candidates else 'waiting_for_current_learned_proposals')

    def apply(self,targets,*,now):
        output=[np.array(p,copy=True) for p in targets];raw=rigid(output[1])
        dt=0. if self.last_time is None else min(.04,max(0.,now-self.last_time));self.last_time=now
        self.applied=False;self.gain=0.;self.distance=None;self.depth_mode='waiting'
        # Frozen offsets persist on release: manual motion continues from the
        # assisted hand position instead of springing back. AX rebases both.
        current=next((c for c in self.candidates if self.confirmed and c['candidate_id']==self.confirmed['candidate_id']),None)
        if self.held and self.confirmed:
            if not self.fresh or current is None:self.reason='assist_waiting_for_current_target'
            elif not np.allclose(current['base_ee'],self.confirmed['base_ee'],atol=1e-8,rtol=0):self.reason='assist_candidate_changed'
            else:
                goal=rigid(self.confirmed['base_ee'])
                self.distance=float(np.linalg.norm(goal[:3,3]-raw[:3,3]))
                # Head RGB-D can guide the early approach before the target
                # enters the wrist view. Near the grasp both views are required;
                # conflicting or stale wrist evidence never enables this fallback.
                early_head=(self.guidance_profile=='early_guidance' and self.support_camera=='right_wrist'
                    and self.distance>.15 and self.wrist.get('reason')=='target_not_resolved_in_wrist_depth'
                    and type(self.wrist.get('capture_timestamp')) in (int,float)
                    and -.01<=now-self.wrist['capture_timestamp']<=.25)
                if self.distance>=self.capture_radius:self.reason='move_hand_near_selected_grasp'
                elif not self.wrist.get('accepted') and not early_head:self.reason='assist_waiting_for_wrist_depth'
                else:
                    blend=min(1.,max(0.,(self.capture_radius-self.distance)/
                        max(self.capture_radius-self.full_guidance_radius,1e-12)))
                    self.gain=blend*blend*(3.-2.*blend)
                    self.depth_mode='head_approach' if early_head else 'head_and_wrist'
                    allowance=self.max_translation*self.gain
                    if early_head:allowance=min(allowance,max(0.,self.distance-.15))
                    desired=limited(goal[:3,3]-raw[:3,3],allowance)
                    self.translation+=limited(desired-self.translation,self.translation_speed*dt)
                    desired_rot=Rotation.from_rotvec(limited((Rotation.from_matrix(goal[:3,:3])*
                        Rotation.from_matrix(raw[:3,:3]).inv()).as_rotvec(),self.max_rotation*self.gain))
                    increment=limited((desired_rot*self.rotation.inv()).as_rotvec(),self.rotation_speed*dt)
                    self.rotation=Rotation.from_rotvec(increment)*self.rotation
                    self.applied=bool(dt>0);self.reason='learned_approach_guidance' if early_head else 'learned_grasp_alignment'
        output[1][:3,3]+=self.translation
        output[1][:3,:3]=(self.rotation*Rotation.from_matrix(raw[:3,:3])).as_matrix()
        return output

    def annotate(self,status):
        result=deepcopy(status)
        result.update(live_ai=True,assistance_applied=self.applied,candidates=self.candidates,
            ai_assist=dict(reason=self.reason,held=getattr(self,'held',False),
                translation_offset_m=float(np.linalg.norm(self.translation)),
                rotation_offset_deg=float(np.rad2deg(self.rotation.magnitude())),
                max_translation_m=self.max_translation,max_rotation_deg=float(np.rad2deg(self.max_rotation)),
                guidance_profile=self.guidance_profile,capture_radius_m=self.capture_radius,
                full_guidance_radius_m=self.full_guidance_radius,guidance_gain=self.gain,
                hand_distance_m=self.distance,depth_mode=self.depth_mode,
                wrist_depth=self.wrist,secondary_camera=self.support_camera,
                scope='bounded_grasp_alignment_not_general_obstacle_avoidance'))
        result['snapshot'].update(confirmed=self.confirmed is not None,
            selected_candidate_id=self.selected['candidate_id'] if self.selected else None,
            confirmed_candidate_id=self.confirmed['candidate_id'] if self.confirmed else None,
            translation_assistance=True,finger_assistance=False)
        return result
