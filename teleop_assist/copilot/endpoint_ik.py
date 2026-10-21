"""Reuse an endpoint joint hypothesis only after current full-q43 FK checking.

This numerical optimization is for candidate endpoints, never actual commands.
It has no timestamps, scene cache, collision authority or motion permission.
The caller must independently check the complete measured body state and the
entire endpoint collision geometry on every request. Original IK remains the
fallback when the hypothesis fails. ActualMotionGuard's stricter IK and joint
step bounds are not changed or imported here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time

import numpy as np

from .controller import rigid
from .kinematics import (ALREADY_AT_TARGET_ORIENTATION_TOLERANCE_RAD,
    ALREADY_AT_TARGET_POSITION_TOLERANCE_M, ARM_JOINT_NAMES, arm_vector, sha256)


class EndpointIKValidator:
    """Single-entry numerical cache; immutable target pose is the cache key."""

    def __init__(self, model, original_ik, *, clock=time.perf_counter):
        import pinocchio as pin
        if model.nq!=43 or model.nv!=43 or len(model.names)!=44:
            raise ValueError('Expected the original fixed-base G1 q43 model')
        if any(j.nq!=1 or j.nv!=1 for j in list(model.joints)[1:]):
            raise ValueError('Expected original single-axis G1 joints')
        if not callable(getattr(original_ik,'validate_pose',None)) or not callable(clock):
            raise ValueError('Original IK validator and a clock callable are required')
        self.pin,self.model,self.original,self.clock=pin,model,original_ik,clock
        self.data=model.createData()
        self.arm=np.array([model.joints[model.getJointId(n)].idx_q for n in ARM_JOINT_NAMES])
        self.nonarm=np.array([i for i in range(43) if i not in self.arm])
        self.frames=[model.getFrameId(s+'_wrist_yaw_link') for s in ('left','right')]
        if any(i>=len(model.frames) for i in self.frames):raise ValueError('Original wrist frames missing')
        self.position_limit=min(.01,float(original_ik.config.max_position_error_m))
        self.orientation_limit=min(float(np.deg2rad(10)),float(original_ik.config.max_orientation_error_rad))
        if (not np.isfinite([self.position_limit,self.orientation_limit]).all() or
                self.position_limit<=0 or self.orientation_limit<=0):raise ValueError('Invalid endpoint FK limits')
        self._cached_key=self._cached_q14=None
        self.manifest=dict(schema='punktnull.endpoint_ik_hypothesis.v1',validator_sha256=sha256(__file__),
            original_ik_manifest=deepcopy(original_ik.manifest),
            position_tolerance_m=self.position_limit,orientation_tolerance_rad=self.orientation_limit,
            cache_scope='one_right_target_pose_and_original_model; numerical_q14_only',
            candidate_key_semantics='exact_right_TCP_pose_hash; not_model_source_or_execution_identity',
            cache_trial='current_left_arm_plus_cached_right_arm_in_current_other_q43',
            joint_order=list(ARM_JOINT_NAMES),ee_frame='original_50mm_wrist_yaw_local_X',
            collision_validated=False,joint_state_freshness_validated=False,motion_authorized=False,
            actual_command_IK_or_delta_limits_changed=False)

    def clear(self):
        self._cached_key=self._cached_q14=None

    def _key(self,right):
        payload=dict(right_base_ee=right.tolist(),
            original_urdf_sha256=self.original.manifest.get('urdf_sha256'),
            ee_frame='original_50mm_wrist_yaw_local_X')
        return 'endpoint_pose:'+hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def _limits(self,q):
        return bool(q.shape==(43,) and np.isfinite(q).all() and
            np.all(q>=self.model.lowerPositionLimit) and np.all(q<=self.model.upperPositionLimit))

    def _evaluate(self,q,left,right):
        self.pin.framesForwardKinematics(self.model,self.data,q)
        residual={}
        for side,frame,target in zip(('left','right'),self.frames,(left,right)):
            wrist=self.data.oMf[frame];rotation=np.asarray(wrist.rotation)
            position=np.asarray(wrist.translation)+rotation@np.array([.05,0.,0.])
            residual[side+'_position_error_m']=float(np.linalg.norm(target[:3,3]-position))
            residual[side+'_orientation_error_rad']=float(np.linalg.norm(self.pin.log3(target[:3,:3]@rotation.T)))
        return residual

    def _residual_valid(self,residual,*,strict_noop=False):
        p=min(self.position_limit,ALREADY_AT_TARGET_POSITION_TOLERANCE_M) if strict_noop else self.position_limit
        a=min(self.orientation_limit,ALREADY_AT_TARGET_ORIENTATION_TOLERANCE_RAD) if strict_noop else self.orientation_limit
        return bool(all(np.isfinite(v) and v>=0 for v in residual.values()) and
            all(residual[s+'_position_error_m']<=p and residual[s+'_orientation_error_rad']<=a
                for s in ('left','right')))

    def validate_pose(self,right_base_ee,left_base_ee,current_full_q43):
        started=self.clock()
        result=dict(ik_valid=False,collision_valid=False,trajectory_validated=False,
            joint_state_freshness_validated=False,motion_authorized=False,cache_authority=False,
            scope='numerical_candidate_endpoint_full_q43_FK_only',reason='unvalidated',
            solver_invoked=False,original_solver_invoked=False,cache_hit=False,
            cache_revalidated_this_call=False,cache_candidate_key=None,
            numerical_convergence_steps=0,iterations=[],final=None)
        try:
            q=np.asarray(current_full_q43,dtype=float)
            if not self._limits(q):raise ValueError('current_full_q43_invalid_or_outside_hard_joint_limits')
            left,right=rigid(left_base_ee,'left_target'),rigid(right_base_ee,'right_target')
            key=self._key(right)
            result.update(cache_candidate_key=key,q_initial=q[self.arm].tolist(),q_initial_43=q.tolist(),
                left_target_base_ee=left.tolist(),right_target_base_ee=right.tolist(),
                position_tolerance_m=self.position_limit,orientation_tolerance_rad=self.orientation_limit)
            trial=None
            if self._cached_key==key and self._cached_q14 is not None:
                trial=q.copy();trial[self.arm[7:]]=self._cached_q14[7:]
                residual=self._evaluate(trial,left,right)
                valid=self._limits(trial) and self._residual_valid(residual)
                result['cached_hypothesis_check']=dict(accepted=valid,limits_valid=self._limits(trial),
                    hypothesis_source='previous_numerical_endpoint_q14',**residual)
                if valid:
                    result.update(ik_valid=True,cache_hit=True,cache_revalidated_this_call=True,
                        reason='cached_endpoint_FK',validation_method='independent_cached_endpoint_full_q43_FK',
                        final=dict(iteration=0,solver_success=None,solver_status='validated_cached_endpoint_FK',
                            limits_valid=True,**residual))
                else:trial=None
            if trial is None:
                # Failed hypotheses never suppress the unmodified original
                # solver. This result is rechecked in the complete model too.
                original=deepcopy(self.original.validate_pose(right,left,q[self.arm]))
                result.update(original_solver_invoked=original.get('solver_invoked') is True,
                    solver_invoked=original.get('solver_invoked') is True)
                # Invalid solver diagnostics must still produce a finite,
                # serializable rejection rather than break the worker pipe.
                json.dumps(original,allow_nan=False)
                result.update(original_ik_result=original,iterations=deepcopy(original.get('iterations',[])),
                    numerical_convergence_steps=original.get('numerical_convergence_steps',0))
                trial=q.copy();trial[self.arm]=arm_vector(original.get('q_solution'),'original_endpoint_q14')
                residual=self._evaluate(trial,left,right)
                original_noop=(original.get('reason')=='already_at_target' and
                    original.get('solver_invoked') is False and np.array_equal(trial,q) and
                    original.get('validation_method')=='independent_current_pose_FK' and
                    original.get('final',{}).get('solver_status')=='not_run_already_at_target' and
                    original.get('final',{}).get('solver_success') is None)
                valid=(original.get('ik_valid') is True and self._limits(trial) and
                    (original_noop or original.get('final',{}).get('solver_success') is True) and
                    self._residual_valid(residual,strict_noop=original_noop))
                result.update(ik_valid=bool(valid),
                    reason=('already_at_target' if original_noop else 'endpoint_original_IK_full_q43_FK_valid')
                        if valid else 'original_endpoint_failed_full_q43_FK_or_solver_status',
                    validation_method='independent_current_pose_FK' if original_noop else
                        'original_IK_plus_independent_full_q43_endpoint_FK',
                    final=dict(deepcopy(original.get('final') or {}),limits_valid=self._limits(trial),**residual))
            result.update(q_solution=trial[self.arm].tolist(),q_solution_43=trial.tolist(),
                limits_valid=self._limits(trial),
                fixed_nonarm_joints_unchanged=bool(np.array_equal(trial[self.nonarm],q[self.nonarm])))
            if result['ik_valid']:
                self._cached_key,self._cached_q14=key,trial[self.arm].copy()
            else:self.clear()
        except (ValueError,TypeError,KeyError,AttributeError,OverflowError) as error:
            self.clear();result.update(ik_valid=False,reason=str(error))
        result['elapsed_s']=float(self.clock()-started)
        if not np.isfinite(result['elapsed_s']) or result['elapsed_s']<0:
            self.clear();result.update(ik_valid=False,reason='endpoint_ik_clock_invalid')
        return result
