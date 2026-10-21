"""Bounded original-model full-q43 arm IK, without motion authorization.

The 29 non-arm coordinates remain exactly measured. Independent state, source
freshness, and collision/path validation are required after this calculation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from .controller import rigid
from .kinematics import ARM_JOINT_NAMES, sha256


@dataclass(frozen=True)
class FullStateIKConfig:
    position_tolerance_m: float = .0005
    orientation_tolerance_rad: float = float(np.deg2rad(.5))
    max_joint_delta_rad: float = .15
    max_iteration_step_rad: float = .04
    max_iterations: int = 12
    max_backtracks: int = 6
    max_elapsed_s: float = .01
    orientation_weight_m: float = .05
    damping: float = 1e-4

    def __post_init__(self):
        numerical=(self.position_tolerance_m,self.orientation_tolerance_rad,
            self.max_joint_delta_rad,self.max_iteration_step_rad,self.max_elapsed_s,
            self.orientation_weight_m,self.damping)
        if (any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) for x in numerical)
                or not 0<self.position_tolerance_m<=.0005
                or not 0<self.orientation_tolerance_rad<=np.deg2rad(.5)
                or not 0<self.max_joint_delta_rad<=.15
                or not 0<self.max_iteration_step_rad<=.04
                or type(self.max_iterations) is not int or not 1<=self.max_iterations<=12
                or type(self.max_backtracks) is not int or not 1<=self.max_backtracks<=6
                or not 0<self.max_elapsed_s<=.01
                or not .01<=self.orientation_weight_m<=.1
                or not 1e-6<=self.damping<=.01):
            raise ValueError('Invalid bounded full-state IK configuration')


class FullStateIKSolver:
    def __init__(self, model, config=None, *, clock=time.perf_counter):
        import pinocchio as pin
        self.pin,self.model,self.config,self.clock=pin,model,config or FullStateIKConfig(),clock
        if model.nq!=43 or model.nv!=43 or len(model.names)!=44:
            raise ValueError('Expected original fixed-base G1 q43 model')
        for joint in list(model.joints)[1:]:
            if joint.nq!=1 or joint.nv!=1:raise ValueError('Expected single-axis original G1 joints')
        self.data=model.createData()
        self.arm=np.array([model.joints[model.getJointId(n)].idx_q for n in ARM_JOINT_NAMES])
        self.arm_v=np.array([model.joints[model.getJointId(n)].idx_v for n in ARM_JOINT_NAMES])
        self.nonarm=np.array([i for i in range(43) if i not in self.arm])
        self.frames=[model.getFrameId(side+'_wrist_yaw_link') for side in ('left','right')]
        if any(i>=len(model.frames) for i in self.frames):raise ValueError('Original wrist frames missing')
        self.tcp=np.array([.05,0.,0.])
        self.weights=np.tile([1.,1.,1.,self.config.orientation_weight_m,
                             self.config.orientation_weight_m,self.config.orientation_weight_m],2)
        self.manifest=dict(solver_sha256=sha256(__file__),configuration=asdict(self.config),
            method='bounded_damped_least_squares_full_q43',seed_default='current_measured_arm_q14',
            fixed_coordinates='all_measured_waist_leg_and_Dex3_joints',
            jacobian='Pinocchio_LOCAL_WORLD_ALIGNED_wrist_with_original_50mm_local_X_TCP_shift',
            arm_joint_order=list(ARM_JOINT_NAMES),pinocchio=pin.__version__,
            collision_validated=False,source_freshness_validated=False,motion_authorized=False)

    def evaluate(self,q,targets,*,jacobian=False):
        """Return world-aligned two-TCP residuals and optional arm Jacobian."""
        pin,model,data=self.pin,self.model,self.data
        pin.framesForwardKinematics(model,data,q)
        if jacobian:
            pin.computeJointJacobians(model,data,q)
            pin.updateFramePlacements(model,data)
        residual=[];errors={};blocks=[]
        for side,frame,target in zip(('left','right'),self.frames,targets):
            wrist=data.oMf[frame];R=np.asarray(wrist.rotation)
            p=np.asarray(wrist.translation)+R@self.tcp
            ep=target[:3,3]-p;er=np.asarray(pin.log3(target[:3,:3]@R.T))
            residual.extend(ep);residual.extend(er)
            errors[side+'_position_error_m']=float(np.linalg.norm(ep))
            errors[side+'_orientation_error_rad']=float(np.linalg.norm(er))
            if jacobian:
                J=pin.getFrameJacobian(model,data,frame,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:,self.arm_v].copy()
                J[:3]-=pin.skew(R@self.tcp)@J[3:]
                blocks.append(J)
        return np.asarray(residual),errors,np.vstack(blocks) if jacobian else None

    def solve(self,left_base_ee,right_base_ee,current_full_q43,*,seed_q14=None):
        started=self.clock();cfg=self.config
        result=dict(ik_valid=False,collision_valid=False,trajectory_validated=False,
            joint_state_freshness_validated=False,motion_authorized=False,
            scope='independent_original_full_q43_arm_FK',
            validation_method='bounded_full_q43_DLS_plus_independent_FK',
            reason='unvalidated',solver_invoked=False,numerical_convergence_steps=0,
            seed_source='current_measured_q14' if seed_q14 is None else 'provided_numerical_seed',
            iterations=[],final=None)
        q=None
        try:
            current=np.asarray(current_full_q43,float)
            if current.shape!=(43,) or not np.isfinite(current).all():raise ValueError('invalid_current_full_q43')
            if np.any(current<self.model.lowerPositionLimit) or np.any(current>self.model.upperPositionLimit):
                raise ValueError('current_full_q43_outside_hard_joint_limits')
            seed=current[self.arm].copy() if seed_q14 is None else np.asarray(seed_q14,float)
            if seed.shape!=(14,) or not np.isfinite(seed).all():raise ValueError('invalid_numerical_q14_seed')
            left,right=rigid(left_base_ee,'left_target'),rigid(right_base_ee,'right_target');targets=(left,right)
            result.update(q_initial=current[self.arm].tolist(),q_initial_43=current.tolist(),
                left_target_base_ee=left.tolist(),right_target_base_ee=right.tolist())
            # One representable value inside the requested delta avoids an
            # outward floating-point rounding at exactly the .15-rad boundary.
            delta=np.nextafter(cfg.max_joint_delta_rad,0.)
            lo=np.maximum(self.model.lowerPositionLimit[self.arm],np.nextafter(current[self.arm]-delta,current[self.arm]))
            hi=np.minimum(self.model.upperPositionLimit[self.arm],np.nextafter(current[self.arm]+delta,current[self.arm]))
            q=current.copy();q[self.arm]=np.clip(seed,lo,hi)
            result['seed_projected_to_bounds']=bool(np.any(q[self.arm]!=seed))
            damping=cfg.damping
            for iteration in range(cfg.max_iterations+1):
                result['numerical_convergence_steps']=iteration
                if self.clock()-started>=cfg.max_elapsed_s:
                    result['reason']='full_q43_ik_deadline_exceeded';break
                e,errors,J=self.evaluate(q,targets,jacobian=True)
                success=(np.isfinite(e).all() and
                    max(errors[s+'_position_error_m'] for s in ('left','right'))<=cfg.position_tolerance_m and
                    max(errors[s+'_orientation_error_rad'] for s in ('left','right'))<=cfg.orientation_tolerance_rad)
                final=dict(iteration=iteration,solver_success=bool(success),
                    solver_status='full_q43_residual_converged' if success else 'full_q43_refinement_in_progress',
                    limits_valid=bool(np.all(q>=self.model.lowerPositionLimit) and np.all(q<=self.model.upperPositionLimit)),**errors)
                result['final']=final;result['iterations'].append(final)
                if success:
                    result.update(ik_valid=True,reason='full_q43_strict_FK_valid');break
                if iteration==cfg.max_iterations:
                    result['reason']='full_q43_bounded_iterations_exhausted';break
                Jw=J*self.weights[:,None];ew=e*self.weights
                step=Jw.T@np.linalg.solve(Jw@Jw.T+damping*damping*np.eye(12),ew)
                if not np.isfinite(step).all():raise ValueError('full_q43_nonfinite_solver_step')
                step*=min(1.,cfg.max_iteration_step_rad/max(float(np.max(abs(step))),1e-15))
                result['solver_invoked']=True
                cost=float(ew@ew);improved=False
                for backtrack in range(cfg.max_backtracks):
                    if self.clock()-started>=cfg.max_elapsed_s:break
                    trial=q.copy();trial[self.arm]=np.clip(q[self.arm]+step*(.5**backtrack),lo,hi)
                    et,_,_=self.evaluate(trial,targets)
                    trial_cost=float((et*self.weights)@(et*self.weights))
                    if np.isfinite(trial_cost) and trial_cost<cost-1e-16:
                        q=trial;improved=True;damping=max(cfg.damping,damping*.5);break
                if not improved:
                    damping*=10.
                    if damping>1.:
                        result['reason']='full_q43_no_feasible_bounded_descent';break
            result.update(q_solution=q[self.arm].tolist(),q_solution_43=q.tolist(),
                max_joint_delta_rad=float(np.max(abs(q[self.arm]-current[self.arm]))),
                fixed_nonarm_joints_unchanged=bool(np.array_equal(q[self.nonarm],current[self.nonarm])),
                limits_valid=bool(np.all(q>=self.model.lowerPositionLimit) and np.all(q<=self.model.upperPositionLimit)))
            if (not result['limits_valid'] or not result['fixed_nonarm_joints_unchanged'] or
                    result['max_joint_delta_rad']>cfg.max_joint_delta_rad):
                result.update(ik_valid=False,reason='full_q43_final_configuration_bound_failed')
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError) as exc:
            result.update(ik_valid=False,reason=str(exc))
        result['elapsed_s']=float(self.clock()-started)
        if not np.isfinite(result['elapsed_s']) or not 0<=result['elapsed_s']<cfg.max_elapsed_s:
            result.update(ik_valid=False,reason='full_q43_ik_deadline_exceeded_or_clock_invalid')
        return result
