"""Opt-in grasp/pull attempt driven by the production perception and input APIs.

Control uses a fresh registered CAD grasp candidate, never simulator target
coordinates. Fixture-joint telemetry is used only by observe()/report() for
evaluation, to distinguish executing a script from actually opening the
drawer. Synthetic engagement confirmation is explicit, not a grasp detector.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .stack_evaluation import StackAcceptanceDriver


class _KnownFixturePullEvidence:
    """Bounded evaluation aggregates; never consulted by the motion driver.

    The one known PullHandle rigid body includes the front plate and U-handle.
    This certifies contacts with that compound body, not a particular CAD face.
    Contact duration is accumulated only while measured hand progress increases.
    Fixed spatial bins allow final travel thirds to be evaluated without keeping
    a growing per-physics-step trace. Boundary-crossing bins are excluded from
    third durations, so their discretization cannot inflate passing evidence.
    """
    FORCE_THRESHOLD_N=1e-4
    BASELINE_DURATION_S=.2
    CONTACT_PER_THIRD_S=.2
    BIN_WIDTH_M=.0001
    MAX_HAND_TRAVEL_M=.045
    MAX_INTERVAL_S=.02
    FIXTURE_AXIS_WORLD=np.array([1.,0.,0.])
    BODY_NAMES=tuple('right_hand_'+name+'_link' for name in (
        'thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1'))

    def __init__(self):
        self.contact_bins=np.zeros(450)
        self.noise_duration=0.;self.noise_max_force=0.;self.noise_samples=0;self.noise_invalid=0
        self._noise_previous=None
        self.started=False;self.baseline_valid=False
        self.ee_start=None;self.joint_start=None;self.axis=None
        self.hand_progress=None;self.drawer_progress=None;self.max_relative_error=0.;self.max_hand_progress=0.
        self.invalid_kinematics=0;self.pair_valid=0;self.pair_invalid=0;self.pair_reasons={}
        self.pair_max_force=0.;self.invalid_intervals=0;self._previous=None
        self.negative_control_samples=0

    @staticmethod
    def _number(value):
        try:
            result=float(value)
            return result if np.isfinite(result) else None
        except (TypeError,ValueError,OverflowError):return None

    @classmethod
    def _pair(cls,state,sim_time):
        value=state.get('evaluation_hand_target_contacts')
        if not isinstance(value,dict):return False,'missing_pair_telemetry',0.
        if (value.get('schema')!='punktnull.evaluation_hand_target_contacts.v1'
                or value.get('valid') is not True or value.get('evaluation_only') is not True):
            return False,'invalid_pair_telemetry',0.
        paths=value.get('target_body_paths')
        if (value.get('target_asset_id')!='PullHandleGrasp'
                or value.get('target_rigid_body_asset_id')!='PullHandle'
                or not isinstance(paths,list) or len(paths)!=1
                or not isinstance(paths[0],str) or paths[0].rsplit('/',1)[-1]!='PullHandle'):
            return False,'wrong_contact_partner',0.
        names=value.get('sensor_body_names')
        if (not isinstance(names,list) or len(names)!=len(cls.BODY_NAMES) or any(not isinstance(n,str) for n in names)
                or set(names)!=set(cls.BODY_NAMES) or value.get('force_units')!='N'
                or value.get('measurement')!='post_step_filtered_normal_force_matrix'):
            return False,'invalid_pair_sensor_contract',0.
        sample_sim=cls._number(value.get('sim_time'));robot_sim=cls._number(state.get('sim_time'))
        sample_wall=cls._number(value.get('timestamp'));robot_wall=cls._number(state.get('timestamp'))
        if (sample_sim is None or robot_sim is None or sample_wall is None or robot_wall is None
                or abs(sample_sim-robot_sim)>1e-6 or abs(sample_sim-sim_time)>1e-6
                or abs(sample_wall-robot_wall)>1e-6):
            return False,'misaligned_pair_sample',0.
        vectors=value.get('per_body_normal_forces_W',{})
        norms=value.get('per_body_normal_force_norms_N',{})
        if not isinstance(vectors,dict) or not isinstance(norms,dict):
            return False,'invalid_pair_force_values',0.
        maximum=0.
        for body in cls.BODY_NAMES:
            try:vector=np.asarray(vectors.get(body),dtype=float)
            except (TypeError,ValueError):return False,'invalid_pair_force_values',0.
            norm=cls._number(norms.get(body))
            if (vector.shape!=(3,) or not np.isfinite(vector).all() or norm is None or norm<0.
                    or not np.isclose(np.linalg.norm(vector),norm,atol=1e-8,rtol=1e-5)):
                return False,'invalid_pair_force_values',0.
            maximum=max(maximum,norm)
        return True,None,maximum

    def observe(self,phase,state,sim_time,pull_axis,selected_cad_asset):
        if phase not in ('await_perception','align_pregrasp','bounded_pull'):return
        now=self._number(sim_time)
        if now is None:return
        valid,reason,force=self._pair(state,now)
        if phase in ('await_perception','align_pregrasp'):
            if valid:
                self.noise_samples+=1;self.noise_max_force=max(self.noise_max_force,force)
                if self._noise_previous is not None:
                    dt=now-self._noise_previous
                    if 0.<dt<=self.MAX_INTERVAL_S+1e-9:self.noise_duration+=dt
                self._noise_previous=now
            else:
                self.noise_invalid+=1;self._noise_previous=None
            return
        if selected_cad_asset!='PullHandleGrasp':
            valid=False;reason='selected_target_is_not_known_fixture_CAD'
        if valid:
            self.pair_valid+=1;self.pair_max_force=max(self.pair_max_force,force)
            self.negative_control_samples+=int(bool(state['evaluation_hand_target_contacts'].get('negative_control')))
        else:
            self.pair_invalid+=1
            self.pair_reasons[reason]=self.pair_reasons.get(reason,0)+1
        try:ee=np.asarray(state.get('ee_position'),dtype=float)
        except (TypeError,ValueError):ee=np.empty(0)
        joint=self._number((state.get('fixture_joint_states') or {}).get('PullHandle',{}).get('position'))
        if not self.started:
            self.started=True
            try:axis=np.asarray(pull_axis,dtype=float)
            except (TypeError,ValueError):axis=np.empty(0)
            if (ee.shape==(3,) and np.isfinite(ee).all() and joint is not None
                    and axis.shape==(3,) and np.isfinite(axis).all() and np.linalg.norm(axis)>1e-9):
                self.ee_start=ee.copy();self.joint_start=joint
                self.axis=axis/np.linalg.norm(axis);self.baseline_valid=True
        if not self.baseline_valid or ee.shape!=(3,) or not np.isfinite(ee).all() or joint is None:
            self.invalid_kinematics+=1;self._previous=None
            return
        hand=float((ee-self.ee_start)@self.axis)
        drawer=float((joint-self.joint_start)*(self.FIXTURE_AXIS_WORLD@self.axis))
        self.hand_progress=hand;self.drawer_progress=drawer
        self.max_hand_progress=max(self.max_hand_progress,abs(hand))
        self.max_relative_error=max(self.max_relative_error,abs(hand-drawer))
        touching=bool(valid and force>self.FORCE_THRESHOLD_N)
        if self._previous is not None:
            previous_time,previous_hand,previous_touching=self._previous
            dt=now-previous_time
            if dt<0. or dt>self.MAX_INTERVAL_S+1e-9:
                self.invalid_intervals+=1
            elif dt>0. and touching and previous_touching and hand>previous_hand:
                lo=max(0.,previous_hand);hi=min(self.MAX_HAND_TRAVEL_M,hand)
                if hi>lo:
                    first=max(0,int(np.floor(lo/self.BIN_WIDTH_M)))
                    stop=min(len(self.contact_bins),int(np.ceil(hi/self.BIN_WIDTH_M)))
                    for index in range(first,stop):
                        overlap=max(0.,min(hi,(index+1)*self.BIN_WIDTH_M)-max(lo,index*self.BIN_WIDTH_M))
                        self.contact_bins[index]+=dt*overlap/(hand-previous_hand)
        self._previous=(now,hand,touching)

    def report(self):
        noise_pass=bool(self.noise_duration>=self.BASELINE_DURATION_S-1e-9
                        and self.noise_max_force<self.FORCE_THRESHOLD_N)
        geometry_pass=bool(self.baseline_valid and self.invalid_kinematics==0
                          and self.hand_progress is not None and self.hand_progress>.025
                          and self.drawer_progress is not None and self.drawer_progress>.025
                          and self.max_hand_progress<=self.MAX_HAND_TRAVEL_M
                          and self.max_relative_error<=.005)
        thirds=[0.,0.,0.]
        if self.hand_progress is not None and self.hand_progress>0.:
            left=np.arange(len(self.contact_bins))*self.BIN_WIDTH_M
            right=left+self.BIN_WIDTH_M
            for index in range(3):
                lower=self.hand_progress*index/3.;upper=self.hand_progress*(index+1)/3.
                mask=(left>=lower-1e-12)&(right<=upper+1e-12)
                thirds[index]=float(self.contact_bins[mask].sum())
        contacts_pass=bool(noise_pass and self.pair_valid>0 and self.pair_invalid==0 and self.invalid_intervals==0
                           and all(duration>=self.CONTACT_PER_THIRD_S-1e-9 for duration in thirds))
        return {
            'functional_coupled_pull':{'passed':geometry_pass,
                'baseline_valid':self.baseline_valid,'invalid_kinematic_samples':self.invalid_kinematics,
                'hand_incremental_progress_m':self.hand_progress,'drawer_incremental_progress_along_action_axis_m':self.drawer_progress,
                'max_hand_displacement_m':self.max_hand_progress,'max_relative_hand_drawer_error_m':self.max_relative_error,
                'minimum_incremental_progress_m_strict':.025,'maximum_hand_displacement_m':self.MAX_HAND_TRAVEL_M,
                'maximum_relative_error_m':.005,'declared_action_axis_world':None if self.axis is None else self.axis.tolist(),
                'known_fixture_joint_axis_world':self.FIXTURE_AXIS_WORLD.tolist(),
                'scope':'Measured displacement from the first pull sample, projected onto the declared action axis; evaluation only'},
            'pair_contact_noise_baseline':{'passed':noise_pass,'valid_duration_s':self.noise_duration,
                'required_duration_s':self.BASELINE_DURATION_S,'maximum_normal_force_N':self.noise_max_force,
                'fixed_threshold_N':self.FORCE_THRESHOLD_N,'valid_samples':self.noise_samples,'invalid_samples':self.noise_invalid,
                'scope':'Same-run await_perception/align_pregrasp contact-free baseline; threshold fixed before the run'},
            'distributed_target_hand_contacts':{'passed':contacts_pass,'valid_samples':self.pair_valid,
                'invalid_samples':self.pair_invalid,'rejection_counts':dict(self.pair_reasons),
                'invalid_time_intervals':self.invalid_intervals,'max_sample_interval_s':self.MAX_INTERVAL_S,
                'maximum_pair_normal_force_N':self.pair_max_force,'fixed_force_threshold_N':self.FORCE_THRESHOLD_N,
                'contact_seconds_per_travel_third':thirds,'required_seconds_per_third':self.CONTACT_PER_THIRD_S,
                'positive_progress_contact_seconds':float(self.contact_bins.sum()),
                'spatial_bin_width_m':self.BIN_WIDTH_M,'fixed_spatial_bins':len(self.contact_bins),
                'spatial_contact_seconds':self.contact_bins.tolist(),
                'bin_assignment':'Only bins fully inside each final actual hand-travel third; boundary bins excluded',
                'negative_control_samples':self.negative_control_samples,
                'target_asset_id':'PullHandleGrasp','target_rigid_body_asset_id':'PullHandle',
                'scope':'Seven finger links against the compound PullHandle rigid body, including front plate and U-handle; normal force only, not CAD-face or total-wrench proof'},
        }


class VisionGraspPullDriver(StackAcceptanceDriver):
    """Same update/observe/report/done hook as StackAcceptanceDriver.

    Call only through an explicit acceptance option. This is not the operator
    default, and it does not issue robot-backend or actuator calls itself.
    """
    MAX_WALL_SECONDS=180.
    MAX_SIM_SECONDS=80.
    GRASP_POSITION_TOLERANCE_M=.0025
    GRASP_REFINEMENT_SPEED_M_S=.004
    GRASP_REFINEMENT_LIMIT_M=.004

    def __init__(self,session):
        super().__init__(session,preferred_action='pull',preferred_label='handle',require_cad_pose=True)
        self.grip=0.
        self.grasp=None
        self.orientation=None
        self.initial_orientation=None
        self.pregrasp=None
        self.pull_origin=None
        self.pull_axis=None
        self.phase_results={}
        self.fixture_observations={}
        self.finger_contact_samples=0
        self.pull_finger_contact_samples=0
        self.pull_start_observation=None
        self._pull_observation_started=False
        self._initial_observation_started=False
        self.grasp_confirmation=None
        self.max_execution_progress=0.
        self.script_completed=False
        self.abort_gate_snapshot=None
        self._pull_evidence=_KnownFixturePullEvidence()

    def _input(self,desired=None,clutch=False,quaternion=None):
        self.seq+=1
        if desired is None or self.session.hand_origin is None or self.session.robot_origin is None:
            hand=np.zeros(3)
        else:
            hand=self.XR_TO_ROBOT.T@(np.asarray(desired)-self.session.robot_origin)/self.session.config.hand_scale
            hand+=self.session.hand_origin
        hand_q=np.array([0.,0.,0.,1.])
        if quaternion is not None and self.session.robot_orientation_origin is not None:
            basis=Rotation.from_matrix(self.XR_TO_ROBOT)
            delta=Rotation.from_quat(quaternion)*Rotation.from_quat(self.session.robot_orientation_origin).inv()
            hand_q=(basis.inv()*delta*basis*Rotation.from_quat(self.session.hand_orientation_origin)).as_quat()
        self.session.command({'type':'input','sample':{
            'source':'synthetic_vision_grasp_operator','session_id':'vision_grasp_acceptance_v1','seq':self.seq,
            'right':{'valid':True,'position':hand.tolist(),'quaternion_xyzw':hand_q.tolist()},
            'orientation_mode':'pose','clutch':bool(clutch),'squeeze':1. if clutch else 0.,
            'trigger':float(self.grip),'axes':[0.,0.],'primary':False,'secondary':False,
            'frame_id':self.session.frame_id}})

    def _pose_error(self,state,point,quaternion=None):
        position=float(np.linalg.norm(np.asarray(state['ee_position'])-point))
        orientation=self.orientation if quaternion is None else quaternion
        angle=float((Rotation.from_quat(orientation)*Rotation.from_quat(state['ee_quaternion']).inv()).magnitude())
        return position,angle

    def _record_phase(self,state,target,quaternion=None):
        position,angle=self._pose_error(state,target,quaternion)
        self.phase_results[self.phase]={'measured_position_error_m':position,'measured_orientation_error_rad':angle,
                                      'measured_ee':list(state['ee_position']),'requested_grip':self.grip,
                                      'accepted_grip':state.get('accepted_grip')}

    def _finish(self,error=None):
        if error:self.results['driver']={'passed':False,'reason':str(error)}
        self.done=True
        # Preserve the current gripper hold on failure; only the explicit open
        # phase releases it. All arm motion is disarmed immediately.
        self._input(clutch=False)
        self.session.command({'type':'assist_enable','enabled':False})

    def update(self,robot_state,sim_time):
        if self.done:return
        now=self.session.clock()
        if self.start_sim is None:self.start_sim=self.phase_start=float(sim_time)
        if now-self.wall_start>self.MAX_WALL_SECONDS or float(sim_time)-self.start_sim>self.MAX_SIM_SECONDS:
            self._finish('Bounded vision grasp attempt deadline exceeded');return
        if not self.session.config.require_action:
            self._finish('Vision grasp attempt requires the production action gate');return
        elapsed=float(sim_time)-self.phase_start
        if self.phase=='await_perception':
            self._input(clutch=False)
            found=self._stable_goal(robot_state)
            if found is None:
                if now-self.phase_wall>45.:self._finish('No stable fresh handle CAD grasp candidate')
                return
            _,target,capability,self.normal,_=found
            self.target_id=target['id'];self.action_id='pull'
            if not capability.get('execution_ready') or capability.get('execution_travel_limit_m')!=.04:
                self._finish('Known calibrated 40 mm execution capability required');return
            if 'grasp_quaternion_xyzw' not in capability:
                self._finish('A full registered hand/tool orientation is required');return
            self.grasp=np.asarray(capability['grasp_position_world'])
            self.orientation=np.asarray(capability['grasp_quaternion_xyzw'])
            self.normal=np.asarray(capability['approach_axis_world'])
            self.pregrasp=self.grasp+.09*self.normal
            self.goal=self.pregrasp.copy()
            self.start_pose=np.asarray(robot_state['ee_position'])
            self.initial_orientation=np.asarray(robot_state['ee_quaternion'])
            self.target_description={'id':self.target_id,'action_id':'pull','semantic_label':'handle',
                'calibration_id':capability['calibration_id'],'cad_asset_id':target['cad_asset_id'],
                'pose_source':target['pose_source'],'pose_timestamp':target['pose_timestamp'],
                'initial_grasp_world':self.grasp.tolist(),'initial_quaternion_xyzw':self.orientation.tolist(),
                'goal_source':'registered_object_pose_times_known_local_grasp_template',
                'fresh_observations_before_lock':self._candidate_frames}
            self.session.command({'type':'assist_enable','enabled':True})
            self.session.command({'type':'eye_enable','enabled':False})
            self._select_action();self._arm(robot_state)
            self._set_phase('align_pregrasp',sim_time,robot_state)
            return
        gate_snapshot=self.session.snapshot()
        if self.selection_invalid(gate_snapshot):
            self.abort_gate_snapshot=self._abort_gate_snapshot(gate_snapshot,robot_state,sim_time)
            self._finish('Locked target/action, CAD pose or fresh-clutch gate invalid; production hold preserved');return
        if self.phase=='align_pregrasp':
            fraction=min(1.,elapsed/12.)
            desired=self.start_pose+(self.pregrasp-self.start_pose)*fraction
            if elapsed>12.:desired+=self.normal*.002*np.sin(2.*(elapsed-12.))
            quaternion=Slerp([0.,1.],Rotation.from_quat([self.initial_orientation,self.orientation]))([fraction]).as_quat()[0]
            self._input(desired,True,quaternion)
            position,angle=self._pose_error(robot_state,self.pregrasp)
            if elapsed>=12. and position<.025 and angle<.25:
                self._record_phase(robot_state,self.pregrasp)
                self._set_phase('approach_grasp',sim_time,robot_state)
            elif elapsed>18.:self._finish('Pregrasp position/orientation was not reached within bounded time')
            return
        if self.phase=='approach_grasp':
            fraction=min(1.,elapsed/6.)
            desired=self.phase_pose+(self.grasp-self.phase_pose)*fraction
            if elapsed>6.:
                # Explicit synthetic operator input, not autonomous finishing:
                # at most 4 mm beyond the original measured candidate, at a
                # speed above the whole-hand stationary deadband. Production
                # correction/stop/contact gates continue to own the reference.
                desired-=self.normal*min(self.GRASP_REFINEMENT_LIMIT_M,
                                         (elapsed-6.)*self.GRASP_REFINEMENT_SPEED_M_S)
            self._input(desired,True,self.orientation)
            capability=next(a for a in gate_snapshot['action']['available'] if a['id']==self.action_id)
            current_grasp=np.asarray(capability['grasp_position_world'])
            current_quaternion=np.asarray(capability['grasp_quaternion_xyzw'])
            position,angle=self._pose_error(robot_state,current_grasp,current_quaternion)
            if elapsed>=6. and position<=self.GRASP_POSITION_TOLERANCE_M and angle<.20:
                self._record_phase(robot_state,current_grasp,current_quaternion)
                self.phase_results[self.phase].update(
                    goal_source='current_accepted_CAD_grasp',goal_world=current_grasp.tolist(),
                    goal_quaternion_xyzw=current_quaternion.tolist(),pose_timestamp=capability.get('pose_timestamp'),
                    position_tracking_tolerance_m=self.GRASP_POSITION_TOLERANCE_M,
                    refinement_command_m=min(self.GRASP_REFINEMENT_LIMIT_M,max(0.,elapsed-6.)*self.GRASP_REFINEMENT_SPEED_M_S))
                self._set_phase('close_gripper',sim_time,robot_state)
            elif elapsed>9.:self._finish('Registered grasp region was not reached within bounded time')
            return
        if self.phase=='close_gripper':
            self.grip=.85*min(1.,elapsed/2.)
            self._input(self.phase_pose,True,self.orientation)
            if elapsed>=3.:
                accepted=robot_state.get('accepted_grip')
                if accepted is None or not np.isfinite(accepted) or abs(float(accepted)-.85)>.01:
                    self._finish('Gripper closure command was not accepted; no execution confirmation issued');return
                self._record_phase(robot_state,self.grasp)
                self.session.robot=dict(robot_state)
                self._input(clutch=False)
                try:
                    self.session.command({'type':'action_phase','phase':'execute','engagement_confirmed':True})
                except ValueError as error:
                    self._finish('Execution confirmation rejected: '+str(error));return
                self.grasp_confirmation={'kind':'synthetic_explicit_operator_confirmation',
                    'sensor_verified_grasp':False,'sim_time':float(sim_time)}
                self._event('synthetic_engagement_confirmed',sim_time,
                    kind=self.grasp_confirmation['kind'],sensor_verified_grasp=False)
                self.pull_origin=np.asarray(robot_state['ee_position'])
                self.pull_axis=np.asarray(self.session.snapshot()['action']['execution_axis_world']).copy()
                self._arm(robot_state)
                self._set_phase('bounded_pull',sim_time,robot_state)
            return
        if self.phase=='bounded_pull':
            # Six-second 40 mm operator motion, then a small continued input
            # proves the controller envelope holds while physical tracking settles.
            distance=.04*min(1.,elapsed/6.)+max(0.,elapsed-6.)*.002
            self._input(self.pull_origin+self.pull_axis*distance,True,self.orientation)
            if elapsed>=8.:
                self._record_phase(robot_state,self.pull_origin+self.pull_axis*.04)
                self._set_phase('open_gripper',sim_time,robot_state)
            return
        if self.phase=='open_gripper':
            self.grip=.85*max(0.,1.-elapsed/1.5)
            self._input(self.phase_pose,True,self.orientation)
            if elapsed>=2.5:
                self._record_phase(robot_state,self.phase_pose)
                self.script_completed=True
                self._finish()

    def selection_invalid(self,snapshot=None):
        snapshot=self.session.snapshot() if snapshot is None else snapshot
        expected_phase='execute' if self.phase in ('bounded_pull','open_gripper') else 'approach'
        return (snapshot['selection']['mode']!='confirmed' or snapshot['selection']['id']!=self.target_id
                or snapshot['action']['id']!=self.action_id or snapshot['action']['phase']!=expected_phase
                or not snapshot['action']['confirmed'] or not snapshot['action']['valid']
                or snapshot['action']['rearm_required'])

    def _abort_gate_snapshot(self,snapshot,robot_state,sim_time):
        """Freeze the rejected gate before _finish releases/disarms input.

        This is diagnostic data only. It uses the exact same session snapshot
        as the gate decision, rather than a later decimated control-log sample.
        """
        target=next((t for t in snapshot['targets'] if t['id']==self.target_id),None)
        calibration=(target or {}).get('task_calibration',{})
        action=snapshot['action']
        current=next((a for a in action['available'] if a['id']==action['id']),{})
        target_keys=('id','frame_id','timestamp','cad_asset_id','cad_grasp_status','pose_valid','pose_source',
                     'pose_timestamp','pose_registration_timestamp','pose_tracking_method',
                     'position_sigma_m','orientation_sigma_rad','pose_position_sigma_m',
                     'pose_orientation_sigma_rad','pose_uncertainty_source','pose_quality')
        robot_keys=('valid','ee_position','ee_quaternion','contact_active','collision_stop','contact_body',
                    'contact_force_max','contact_force_world','joint_limit_violation','accepted_grip','hold_reason')
        result={'captured_before_disarm':True,'sim_time':float(sim_time),'wall_time':snapshot['timestamp'],
            'driver_phase':self.phase,'selection':snapshot['selection'],
            'frame_id':snapshot['frame_id'],'frame_age_s':snapshot['frame_age'],'input_age_s':snapshot['input_age'],
            'action':{key:action.get(key) for key in ('target_id','id','phase','confirmed','valid','reason','rearm_required',
                                                    'engagement_status','execution_travel_limit_m')},
            'calibration':{'locked_id':(self.target_description or {}).get('calibration_id'),
                'current_id':current.get('calibration_id',calibration.get('id')),
                'source':calibration.get('source'),'execution_ready':current.get('execution_ready'),
                'has_calibrated_grasp_template':calibration.get('has_calibrated_grasp_template'),
                'pose_timestamp':calibration.get('pose_timestamp'),'pose_max_age_s':calibration.get('pose_max_age_s')},
            'target':{'id':self.target_id,'present':target is not None,
                      **({key:target.get(key) for key in target_keys} if target is not None else {})},
            'robot':{key:robot_state.get(key) for key in robot_keys},
            'previous_control_reason':snapshot['assist'].get('reason')}
        return copy.deepcopy(result)

    def filter_targets(self,targets):
        return targets

    def observe(self,robot_state,reference,sim_time):
        if self.done:return
        self._pull_evidence.observe(self.phase,robot_state,sim_time,self.pull_axis,
                                    (self.target_description or {}).get('cad_asset_id'))
        self.invalid_states+=int(not robot_state.get('valid',False))
        self.max_joint_violation=max(self.max_joint_violation,float(robot_state.get('joint_limit_violation',0.)))
        self.contact_stop_steps+=int(robot_state.get('collision_stop',False))
        self.max_correction_m=max(self.max_correction_m,float(self.session.control.get('correction_m',0.)))
        self.max_rotation_correction_rad=max(self.max_rotation_correction_rad,float(self.session.control.get('orientation_correction_rad',0.)))
        if self.phase in ('close_gripper','bounded_pull') and 'hand' in str(robot_state.get('contact_body','')) and float(robot_state.get('contact_force_max',0.))>1.:
            self.finger_contact_samples+=1
            if self.phase=='bounded_pull':self.pull_finger_contact_samples+=1
        # Measure displacement independently of whether the controller emitted
        # an active execution diagnostic. A hold/contact guard does not make
        # physical drift disappear from the acceptance result.
        if self.pull_origin is not None and self.pull_axis is not None:
            progress=float((np.asarray(robot_state['ee_position'])-self.pull_origin)@self.pull_axis)
            self.max_execution_progress=max(self.max_execution_progress,abs(progress))
        # Evaluation only. No value in this dictionary is read by update(),
        # selection or a robot command. It cannot influence the attempted path.
        fixture=robot_state.get('fixture_joint_states')
        if not self._initial_observation_started:
            self._initial_observation_started=True
            self.fixture_observations['initial']=copy.deepcopy(fixture) if fixture is not None else {}
        if self.phase=='bounded_pull' and not self._pull_observation_started:
            self._pull_observation_started=True
            # Latch exactly the first measured pull sample. Missing telemetry
            # here must not be backfilled later after the mechanism has moved.
            self.pull_start_observation={'sim_time':float(sim_time),
                'fixture_joint_states':copy.deepcopy(fixture) if fixture is not None else None}
        if fixture is not None:
            self.fixture_observations[self.phase]=copy.deepcopy(fixture)

    def report(self):
        checks=dict(self.results)
        checks['script_completed']={'passed':self.script_completed}
        checks['measured_robot_limits']={'passed':self.invalid_states==0 and self.max_joint_violation<=.035,
            'invalid_states':self.invalid_states,'max_joint_violation_rad':self.max_joint_violation,
            'contact_guard_steps':self.contact_stop_steps}
        checks['bounded_actual_pull']={'passed':self.script_completed and self.max_execution_progress<=.045,
            'maximum_measured_execution_displacement_m':self.max_execution_progress,
            'command_limit_m':.04,'physical_tracking_tolerance_m':.005}
        initial=self.fixture_observations.get('initial',{}).get('PullHandle',{}).get('position')
        pulled=self.fixture_observations.get('bounded_pull',{}).get('PullHandle',{}).get('position')
        released=self.fixture_observations.get('open_gripper',{}).get('PullHandle',{}).get('position')
        pull_start=((self.pull_start_observation or {}).get('fixture_joint_states') or {}).get('PullHandle',{}).get('position')
        opening=float(initial-pulled) if initial is not None and pulled is not None else None
        retained=float(initial-released) if initial is not None and released is not None else None
        incremental=float(pull_start-pulled) if pull_start is not None and pulled is not None else None
        checks['physical_drawer_opening_evidence']={'passed':opening is not None and retained is not None and retained>.025
                                                  and incremental is not None and incremental>.025,
            'opening_m':opening,'retained_after_release_m':retained,
            'pull_start_position_m':pull_start,'incremental_opening_during_pull_m':incremental,
            'required_incremental_opening_m':.025,'required_retained_total_opening_m':.025,
            'retained_total_reference':'first physical observation before closure; never backfilled',
            'source':'simulator_joint_measurements_for_evaluation_only_not_controller_input'}
        checks['measured_hand_contact']={'passed':self.finger_contact_samples>10,'measured_samples':self.finger_contact_samples,
            'required_for_success':False,'scope':'Historical operational >1 N diagnostic; not a V2 physical success gate'}
        checks['measured_hand_contact_during_pull']={'passed':self.pull_finger_contact_samples>10,
            'measured_samples':self.pull_finger_contact_samples,'force_threshold_N':1.,'required_samples_strictly_more_than':10,
            'required_for_success':False,
            'scope':'Historical operational >1 N diagnostic; not a V2 physical success gate'}
        checks.update(self._pull_evidence.report())
        retained_incremental=float(pull_start-released) if pull_start is not None and released is not None else None
        release_loss=float(released-pulled) if released is not None and pulled is not None else None
        release_quality=bool(retained_incremental is not None and retained_incremental>.025
                             and release_loss is not None and release_loss<=.005)
        checks['release_retains_incremental']={'passed':release_quality,'required_for_success':False,
            'status':'demonstrated' if release_quality else ('partial_quality' if retained_incremental is not None else 'not_demonstrated'),
            'retained_incremental_opening_m':retained_incremental,'opening_lost_during_release_m':release_loss,
            'required_retained_incremental_opening_m_strict':.025,'maximum_release_loss_m':.005,
            'scope':'Separate release quality; V2 task requires total final opening >25 mm and does not certify stable retention of all pull progress'}
        passed=bool(self.done and checks and all(check.get('passed',False) for check in checks.values()
                                                if check.get('required_for_success',True)))
        return {'schema_version':2,'acceptance_revision':'known_40mm_fixture_v2',
            'kind':'vision_conditioned_grasp_pull_attempt','complete':self.done,'passed':passed,
            'script_completed':self.script_completed,'phase':self.phase,'checks':checks,
            'abort_gate_snapshot':copy.deepcopy(self.abort_gate_snapshot),
            'target':self.target_description,'phases':self.phase_results,'events':self.events,
            'engagement_confirmation':self.grasp_confirmation,'fixture_telemetry_evaluation_only':self.fixture_observations,
            'pull_start_observation_evaluation_only':copy.deepcopy(self.pull_start_observation),
            'input_provenance':'synthetic pose/controller and explicit grip confirmation through production API',
            'simulator_target_poses_used_for_control':False,'headset_verified':False,
            'limits':['An executed script is not a successful grasp without measured contact and drawer motion.',
                      'V2 uses compound-body partner contacts, fixed noise validation and measured path coupling; historical >1 N diagnostics are not mandatory.',
                      'V2 success does not certify loss-free release; release_retains_incremental is reported separately.',
                      'Known G1 hand/CAD fixture only; requested 40 mm pull, not arbitrary objects or tasks.',
                      'The 2.5 mm approach criterion is tracking error to the current CAD estimate, not guaranteed absolute grasp accuracy.',
                      'Initial goal comes from a fresh measured CAD registration; tracking loss preserves production holds.',
                      'Explicit engagement confirmation is synthetic operator input, not a learned grasp detector.']}
