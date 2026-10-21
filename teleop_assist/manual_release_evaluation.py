"""Optional V3 evaluation: assisted pull followed by an explicit manual gesture.

Only production commands drive motion. Fixture/pair-contact telemetry below is
read exclusively by observe/report; it cannot advance or change the gesture.
The original V2 driver and its default trajectory remain unchanged.
"""
from __future__ import annotations

from collections import deque
import copy
import numpy as np

from .vision_grasp_evaluation import VisionGraspPullDriver, _KnownFixturePullEvidence


class _ReleaseHoldEvidence:
    """Bounded, conservative continuous final-second observation window."""
    DURATION_S=1.
    MAX_GAP_S=.02
    MAX_FORCE_N=1e-4
    MAX_DRAWER_DRIFT_M=.001
    MAX_OPEN_JOINT_RAD=.02
    JOINT_NAMES=tuple(name.removesuffix('_link')+'_joint' for name in _KnownFixturePullEvidence.BODY_NAMES)

    def __init__(self):
        self.samples=deque(maxlen=4096)
        self.rejections={}

    def observe(self,state,sim_time):
        valid,reason,force=_KnownFixturePullEvidence._pair(state,sim_time)
        accepted=_KnownFixturePullEvidence._number(state.get('accepted_grip'))
        if accepted is None or abs(accepted)>1e-6:
            valid=False;reason='opening_command_not_accepted'
        names=state.get('joint_names');values=state.get('joints')
        angles=[]
        if isinstance(names,list) and isinstance(values,list) and len(names)==len(values):
            for name in self.JOINT_NAMES:
                if names.count(name)!=1:break
                angle=_KnownFixturePullEvidence._number(values[names.index(name)])
                if angle is None:break
                angles.append(angle)
        if len(angles)!=7:
            valid=False;reason='missing_or_invalid_measured_finger_joints'
        elif max(abs(a) for a in angles)>self.MAX_OPEN_JOINT_RAD:
            valid=False;reason='measured_fingers_not_in_known_open_posture'
        drawer=_KnownFixturePullEvidence._number((state.get('fixture_joint_states') or {}).get('PullHandle',{}).get('position'))
        if drawer is None:valid=False;reason='missing_drawer_measurement'
        if force>self.MAX_FORCE_N:valid=False;reason='finger_target_contact_remains'
        now=float(sim_time)
        if self.samples and now<=self.samples[-1]['time']:
            self.samples.clear();valid=False;reason='nonmonotonic_sim_time'
        if not valid:self.rejections[reason]=self.rejections.get(reason,0)+1
        self.samples.append({'time':now,'valid':valid,'force':force,'drawer':drawer,
                             'max_joint':max(map(abs,angles)) if len(angles)==7 else None})
        # Retain the bracketing sample just before the final one-second window.
        # It makes all reported maxima conservative; there is no interpolation.
        while len(self.samples)>2 and self.samples[1]['time']<=now-self.DURATION_S:
            self.samples.popleft()

    def report(self):
        samples=list(self.samples)
        duration=samples[-1]['time']-samples[0]['time'] if samples else 0.
        gaps=[b['time']-a['time'] for a,b in zip(samples,samples[1:])]
        positions=[s['drawer'] for s in samples if s['drawer'] is not None]
        drift=max(positions)-min(positions) if len(positions)==len(samples) and samples else None
        maximum=max((s['force'] for s in samples),default=None)
        max_joint=max((s['max_joint'] for s in samples if s['max_joint'] is not None),default=None)
        valid=bool(samples and all(s['valid'] for s in samples))
        passed=bool(valid and duration>=self.DURATION_S-1e-9 and gaps
                    and max(gaps)<=self.MAX_GAP_S+1e-9 and drift is not None
                    and drift<=self.MAX_DRAWER_DRIFT_M)
        return {'passed':passed,'continuous_final_window_s':duration,'required_duration_s':self.DURATION_S,
                'window_samples':len(samples),'window_all_samples_valid':valid,
                'maximum_sample_gap_s':max(gaps,default=None),'allowed_sample_gap_s':self.MAX_GAP_S,
                'maximum_pair_normal_force_N':maximum,'fixed_force_threshold_N':self.MAX_FORCE_N,
                'drawer_position_range_m':drift,'maximum_drawer_drift_m':self.MAX_DRAWER_DRIFT_M,
                'maximum_absolute_measured_finger_joint_rad':max_joint,
                'maximum_open_joint_angle_rad':self.MAX_OPEN_JOINT_RAD,
                'required_joint_names':list(self.JOINT_NAMES),'accepted_grip_required':0.,
                'all_hold_rejection_counts':dict(self.rejections),'bounded_buffer_capacity':self.samples.maxlen,
                'scope':'Final continuous sim-second: seven measured G1 finger joints in known open posture and seven finger-to-compound-PullHandle normal-force channels; palm/whole-hand contact and arbitrary aperture are not measured'}


class ManualReleaseGraspDriver(VisionGraspPullDriver):
    """Explicit optional synthetic operator experiment, not autonomous release."""
    PULL_MOTION_S=4.
    PULL_HOLD_S=2.
    MANUAL_MOTION_S=3.
    MANUAL_HOLD_S=1.25
    MANUAL_DISTANCE_M=.04
    MANUAL_PHASES=('manual_release_prepare','manual_release_withdraw','manual_release_hold')

    def __init__(self,session):
        super().__init__(session)
        self.manual_origin=None
        self.manual_orientation=None
        self.manual_goal=None
        self.assisted_max_distance=0.
        self.manual_max_distance=0.
        self.total_max_distance=0.
        self.bound_invalid_samples=0
        self.manual_mode_violations=0
        self.manual_protocol=None
        self._release_hold=_ReleaseHoldEvidence()

    def _manual_gate(self,state):
        if self.session.config.assist_enabled:return 'Manual gesture unexpectedly has assistance enabled'
        if self.session.stopped:return 'Operator stop remains active during manual release'
        if not state.get('valid',False):return 'Measured robot invalid during manual release'
        if self.phase!='manual_release_prepare' and self.session.rearm_required:
            return 'Manual release fresh-clutch gate invalid; no automatic rearm'
        return None

    def update(self,robot_state,sim_time):
        if self.done:return
        if self.phase!='bounded_pull' and self.phase not in self.MANUAL_PHASES:
            return super().update(robot_state,sim_time)
        if (self.session.clock()-self.wall_start>self.MAX_WALL_SECONDS
                or self.start_sim is None or float(sim_time)-self.start_sim>self.MAX_SIM_SECONDS):
            self._finish('Bounded manual release attempt deadline exceeded');return
        if not self.session.config.require_action:
            self._finish('Manual release attempt requires the production action gate');return
        elapsed=float(sim_time)-self.phase_start
        if self.phase=='bounded_pull':
            snapshot=self.session.snapshot()
            if self.selection_invalid(snapshot):
                self.abort_gate_snapshot=self._abort_gate_snapshot(snapshot,robot_state,sim_time)
                self._finish('Locked target/action, CAD pose or fresh-clutch gate invalid; production hold preserved');return
            distance=.04*min(1.,elapsed/self.PULL_MOTION_S)
            self._input(self.pull_origin+self.pull_axis*distance,True,self.orientation)
            if elapsed>=self.PULL_MOTION_S+self.PULL_HOLD_S:
                self._record_phase(robot_state,self.pull_origin+self.pull_axis*.04)
                self.manual_origin=np.asarray(robot_state['ee_position'],dtype=float).copy()
                self.manual_orientation=np.asarray(robot_state['ee_quaternion'],dtype=float).copy()
                self.manual_goal=self.manual_origin+self.pull_axis*self.MANUAL_DISTANCE_M
                self.assisted_max_distance=max(self.assisted_max_distance,float(np.linalg.norm(self.manual_origin-self.pull_origin)))
                self._input(clutch=False)
                self.session.command({'type':'assist_enable','enabled':False})
                self.manual_protocol={'kind':'explicit_synthetic_operator_manual_release','assist_enabled':False,
                    'disarmed_sim_time':float(sim_time),'rearmed_sim_time':None,
                    'origin_world':self.manual_origin.tolist(),'axis_world':self.pull_axis.tolist(),
                    'commanded_withdrawal_m':self.MANUAL_DISTANCE_M,'motion_duration_s':self.MANUAL_MOTION_S,
                    'open_grip_start':.85,'open_grip_end':0.,'stationary_hold_s':self.MANUAL_HOLD_S}
                self._event('synthetic_manual_release_disarmed',sim_time,assist_enabled=False,
                            commanded_withdrawal_m=self.MANUAL_DISTANCE_M)
                self._set_phase('manual_release_prepare',sim_time,robot_state)
            return
        problem=self._manual_gate(robot_state)
        if problem:
            self._finish(problem);return
        if self.phase=='manual_release_prepare':
            self._input(clutch=False)
            # A separate update after a measured physics step keeps the release
            # and fresh calibration/rearm observable, not one atomic jump.
            if elapsed>0.:
                self._arm(robot_state)
                if self.session.rearm_required:
                    self._finish('Explicit manual calibration/fresh clutch was rejected');return
                self.manual_protocol['rearmed_sim_time']=float(sim_time)
                self._event('synthetic_manual_release_rearmed',sim_time,assist_enabled=False,
                            fresh_calibration=True,fresh_clutch=True)
                self._set_phase('manual_release_withdraw',sim_time,robot_state)
            return
        if self.phase=='manual_release_withdraw':
            fraction=min(1.,elapsed/self.MANUAL_MOTION_S)
            self.grip=.85*(1.-fraction)
            desired=self.manual_origin+self.pull_axis*self.MANUAL_DISTANCE_M*fraction
            self._input(desired,True,self.manual_orientation)
            if elapsed>=self.MANUAL_MOTION_S:
                self._record_phase(robot_state,self.manual_goal,self.manual_orientation)
                self._set_phase('manual_release_hold',sim_time,robot_state)
            return
        self.grip=0.
        self._input(self.manual_goal,True,self.manual_orientation)
        if elapsed>=self.MANUAL_HOLD_S:
            self._record_phase(robot_state,self.manual_goal,self.manual_orientation)
            self.script_completed=True
            self._finish()

    def observe(self,robot_state,reference,sim_time):
        if self.done:return
        super().observe(robot_state,reference,sim_time)
        if self.pull_origin is not None:
            try:position=np.asarray(robot_state['ee_position'],dtype=float)
            except (ValueError,TypeError,KeyError):position=np.empty(0)
            if position.shape!=(3,) or not np.isfinite(position).all():self.bound_invalid_samples+=1
            else:
                total=float(np.linalg.norm(position-self.pull_origin))
                self.total_max_distance=max(self.total_max_distance,total)
                if self.manual_origin is None:self.assisted_max_distance=max(self.assisted_max_distance,total)
                else:self.manual_max_distance=max(self.manual_max_distance,float(np.linalg.norm(position-self.manual_origin)))
        if self.phase in self.MANUAL_PHASES:
            self.manual_mode_violations+=int(self.session.config.assist_enabled)
        if self.phase=='manual_release_hold':self._release_hold.observe(robot_state,sim_time)

    def report(self):
        report=super().report();checks=report['checks']
        checks['bounded_actual_pull']={'passed':self.script_completed and self.bound_invalid_samples==0
            and self.assisted_max_distance<=.045,'maximum_measured_execution_displacement_m':self.assisted_max_distance,
            'command_limit_m':.04,'physical_tracking_tolerance_m':.005,
            'scope':'Euclidean displacement from pull origin through assisted phase exit, frozen before explicit manual mode'}
        checks['bounded_manual_release']={'passed':self.manual_origin is not None and self.bound_invalid_samples==0
            and self.manual_max_distance<=.045,'maximum_measured_displacement_m':self.manual_max_distance,
            'maximum_allowed_displacement_m':.045,'command_limit_m':self.MANUAL_DISTANCE_M,
            'scope':'Euclidean displacement from measured manual-mode entry, including prepare and final hold'}
        checks['bounded_combined_motion']={'passed':self.pull_origin is not None and self.bound_invalid_samples==0
            and self.total_max_distance<=.09,'maximum_measured_displacement_m':self.total_max_distance,
            'maximum_allowed_displacement_m':.09,'maximum_measured_axial_displacement_m':self.max_execution_progress,
            'invalid_bound_samples':self.bound_invalid_samples,'scope':'All measured motion since pull origin, including final hold; no silent exclusion of manual travel'}
        checks['explicit_manual_protocol']={'passed':bool(self.manual_protocol and self.manual_protocol.get('rearmed_sim_time') is not None
            and self.manual_mode_violations==0),'assistance_enabled_samples':self.manual_mode_violations,
            'protocol':copy.deepcopy(self.manual_protocol)}
        number=_KnownFixturePullEvidence._number
        def fixture(phase):return number(self.fixture_observations.get(phase,{}).get('PullHandle',{}).get('position'))
        initial=fixture('initial');pulled=fixture('bounded_pull');released=fixture('manual_release_hold')
        start=number(((self.pull_start_observation or {}).get('fixture_joint_states') or {}).get('PullHandle',{}).get('position'))
        retained=initial-released if initial is not None and released is not None else None
        incremental=start-released if start is not None and released is not None else None
        loss=released-pulled if released is not None and pulled is not None else None
        physical=checks['physical_drawer_opening_evidence']
        physical['retained_after_release_m']=retained
        physical['passed']=bool(retained is not None and retained>.025
                               and physical['incremental_opening_during_pull_m'] is not None
                               and physical['incremental_opening_during_pull_m']>.025)
        quality=bool(incremental is not None and incremental>.025 and loss is not None and loss<=.005)
        checks['release_retains_incremental']={'passed':quality,'required_for_success':True,
            'status':'demonstrated' if quality else 'not_demonstrated',
            'retained_incremental_opening_m':incremental,'opening_lost_during_release_m':loss,
            'required_retained_incremental_opening_m_strict':.025,'maximum_release_loss_m':.005,
            'scope':'Required V3 retention after explicit manual withdrawal/opening, relative to first assisted pull sample'}
        checks['final_open_finger_contact_free_hold']=self._release_hold.report()
        report.update(schema_version=3,acceptance_revision='known_40mm_fixture_manual_release_v3',
            kind='vision_conditioned_grasp_pull_with_explicit_manual_release_attempt',
            passed=bool(self.done and all(check.get('passed',False) for check in checks.values() if check.get('required_for_success',True))),
            manual_release_protocol=copy.deepcopy(self.manual_protocol),
            trajectory_contract={'assisted_pull_m':.04,'assisted_motion_sim_s':self.PULL_MOTION_S,
                'assisted_stationary_sim_s':self.PULL_HOLD_S,'manual_withdrawal_m':self.MANUAL_DISTANCE_M,
                'manual_motion_sim_s':self.MANUAL_MOTION_S,'manual_stationary_sim_s':self.MANUAL_HOLD_S,
                'difference_from_v2':'Opt-in 4 s pull plus 2 s stationary input; explicit assist-OFF 40 mm withdrawal/opening over 3 s and 1.25 s stationary input. V2 remains unchanged.'})
        report['limits']=[item for item in report['limits'] if not item.startswith('V2 success')]
        report['limits']+=['V3 manual release is a separate explicit synthetic operator gesture, not automatic assistance or a larger assisted execution cap.',
            'Final opening proof covers measured known G1 finger-joint posture; seven filtered finger contacts do not prove the palm or whole hand is contact-free.',
            'A final one-second retention observation does not certify long-term mechanism stability.']
        return report
