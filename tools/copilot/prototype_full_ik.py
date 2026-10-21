#!/usr/bin/env python3
"""Offline CPU experiment: bounded full-q43 IK refinement, never a command gate.

This prototype is deliberately not imported by the runtime. It solves only the
14 arm coordinates, holding the actual waist, legs and fingers exactly fixed.
It does not validate freshness, collision, or authorize simulation/hardware
motion. Original IK remains the seed provider in the recorded-data experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from teleop_assist.copilot.controller import rigid
from teleop_assist.copilot.kinematics import ARM_JOINT_NAMES, OriginalIKValidator


@dataclass(frozen=True)
class RefinementConfig:
    position_tolerance_m: float = .0005
    orientation_tolerance_rad: float = float(np.deg2rad(.5))
    max_joint_delta_rad: float = .15
    max_iteration_step_rad: float = .04
    max_iterations: int = 12
    max_backtracks: int = 6
    max_elapsed_s: float = .01
    orientation_weight_m: float = .05
    damping: float = 1e-4


class FullStateRefiner:
    def __init__(self, urdf, config=None, *, clock=time.perf_counter):
        import pinocchio as pin
        self.pin, self.config, self.clock = pin, config or RefinementConfig(), clock
        self.urdf = Path(urdf).resolve()
        self.model = pin.buildModelFromUrdf(str(self.urdf))
        self.data = self.model.createData()
        if self.model.nq != 43 or self.model.nv != 43:
            raise ValueError('Only original fixed-base q43 G1 supported')
        self.names = tuple(str(n) for n in list(self.model.names)[1:])
        self.indices = {name:self.model.joints[self.model.getJointId(name)].idx_q for name in self.names}
        self.arm = np.array([self.indices[n] for n in ARM_JOINT_NAMES])
        self.arm_v = np.array([self.model.joints[self.model.getJointId(n)].idx_v for n in ARM_JOINT_NAMES])
        self.frames = [self.model.getFrameId(s+'_wrist_yaw_link') for s in ('left','right')]
        self.tcp = np.array([.05,0.,0.])
        self.weights = np.tile([1.,1.,1.,self.config.orientation_weight_m,
                               self.config.orientation_weight_m,self.config.orientation_weight_m],2)

    def full_q(self, names, values):
        if len(names)!=43 or set(names)!=set(self.names) or len(set(names))!=43:
            raise ValueError('Complete original q43 joint names required')
        a=np.asarray(values,float)
        if a.shape!=(43,) or not np.isfinite(a).all():raise ValueError('Invalid measured q43')
        named=dict(zip(names,a))
        return np.array([named[n] for n in self.names])

    def forward(self, q):
        self.pin.framesForwardKinematics(self.model,self.data,q)
        poses=[]
        for frame in self.frames:
            pose=self.data.oMf[frame].homogeneous.copy()
            pose[:3,3]+=pose[:3,:3]@self.tcp
            poses.append(pose)
        return poses

    def evaluate(self, q, targets, *, jacobian=False):
        poses=self.forward(q)
        residual=[];errors={};blocks=[]
        if jacobian:
            self.pin.computeJointJacobians(self.model,self.data,q)
            self.pin.updateFramePlacements(self.model,self.data)
        for side,frame,pose,target in zip(('left','right'),self.frames,poses,targets):
            ep=target[:3,3]-pose[:3,3]
            er=np.asarray(self.pin.log3(target[:3,:3]@pose[:3,:3].T))
            residual.extend(ep);residual.extend(er)
            errors[side+'_position_error_m']=float(np.linalg.norm(ep))
            errors[side+'_orientation_error_rad']=float(np.linalg.norm(er))
            if jacobian:
                J=self.pin.getFrameJacobian(self.model,self.data,frame,
                    self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:,self.arm_v].copy()
                # LOCAL_WORLD_ALIGNED is evaluated at the wrist frame origin;
                # v_TCP = v_wrist + omega x (R_wrist * [.05,0,0]).
                J[:3]-=self.pin.skew(pose[:3,:3]@self.tcp)@J[3:]
                blocks.append(J)
        return np.asarray(residual),errors,np.vstack(blocks) if jacobian else None

    def refine(self, measured_q43, seed_q14, left, right):
        started=self.clock();cfg=self.config
        result=dict(accepted=False,motion_authorized=False,collision_valid=False,
            scope='offline_full_q43_kinematic_refinement_only',iterations=0,reason='unvalidated')
        try:
            current=np.asarray(measured_q43,float);seed=np.asarray(seed_q14,float)
            if current.shape!=(43,) or seed.shape!=(14,) or not np.isfinite(current).all() or not np.isfinite(seed).all():
                raise ValueError('nonfinite_or_incomplete_configuration')
            if np.any(current<self.model.lowerPositionLimit) or np.any(current>self.model.upperPositionLimit):
                raise ValueError('measured_configuration_outside_hard_limits')
            targets=(rigid(left,'left_target'),rigid(right,'right_target'))
            lo=np.maximum(self.model.lowerPositionLimit[self.arm],current[self.arm]-cfg.max_joint_delta_rad)
            hi=np.minimum(self.model.upperPositionLimit[self.arm],current[self.arm]+cfg.max_joint_delta_rad)
            q=current.copy();q[self.arm]=np.clip(seed,lo,hi)
            result['seed_projected_to_joint_and_command_bounds']=bool(np.any(q[self.arm]!=seed))
            result['q_initial_43']=current.tolist()
            damping=cfg.damping
            for iteration in range(cfg.max_iterations+1):
                result['iterations']=iteration
                if self.clock()-started>=cfg.max_elapsed_s:
                    result['reason']='refinement_deadline_exceeded';break
                e,errors,J=self.evaluate(q,targets,jacobian=True)
                result['final']=errors
                success=(max(errors[s+'_position_error_m'] for s in ('left','right'))<=cfg.position_tolerance_m and
                    max(errors[s+'_orientation_error_rad'] for s in ('left','right'))<=cfg.orientation_tolerance_rad)
                if success:
                    result.update(accepted=True,reason='both_EEs_within_refinement_tolerance');break
                if iteration==cfg.max_iterations:
                    result['reason']='bounded_iterations_exhausted';break
                Jw=J*self.weights[:,None];ew=e*self.weights
                step=Jw.T@np.linalg.solve(Jw@Jw.T+damping*damping*np.eye(12),ew)
                step*=min(1.,cfg.max_iteration_step_rad/max(float(np.max(abs(step))),1e-15))
                cost=float(ew@ew);improved=False
                for backtrack in range(cfg.max_backtracks):
                    if self.clock()-started>=cfg.max_elapsed_s:break
                    trial=q.copy();trial[self.arm]=np.clip(q[self.arm]+step*(.5**backtrack),lo,hi)
                    et,_,_=self.evaluate(trial,targets)
                    trial_cost=float((et*self.weights)@(et*self.weights))
                    if trial_cost<cost-1e-16:
                        q=trial;improved=True;damping=max(cfg.damping,damping*.5);break
                if not improved:
                    damping*=10.
                    if damping>1.:
                        result['reason']='no_feasible_bounded_descent';break
            result['q_solution']=q[self.arm].tolist()
            result['q_solution_43']=q.tolist()
            result['max_joint_delta_rad']=float(np.max(abs(q[self.arm]-current[self.arm])))
            result['fixed_nonarm_joints_unchanged']=bool(np.array_equal(np.delete(q,self.arm),np.delete(current,self.arm)))
            result['hard_joint_limits_satisfied']=bool(np.all(q>=self.model.lowerPositionLimit) and np.all(q<=self.model.upperPositionLimit))
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError) as exc:
            result.update(accepted=False,reason=str(exc))
        result['elapsed_s']=float(self.clock()-started)
        if result['elapsed_s']>=cfg.max_elapsed_s:
            result.update(accepted=False,reason='refinement_deadline_exceeded')
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows',type=Path,required=True)
    parser.add_argument('--repository',type=Path,default=Path('/home/loya2/xr_teleoperate'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=160)
    args=parser.parse_args(argv)
    if not 100<=args.samples<=1000:parser.error('Use 100..1000 samples')
    source=args.rows.read_bytes();rows=json.loads(source)
    refiner=FullStateRefiner(args.repository/'assets/g1/g1_body29_hand14.urdf')
    original=OriginalIKValidator(args.repository)
    eligible=[i for i,r in enumerate(rows) if i>0 and r['command_consumed'] and r['controller']['assistance_applied']]
    transient=[i for i in eligible if i<150]
    remaining=[i for i in eligible if i not in transient]
    selected=sorted(set(transient+([remaining[int(j)] for j in np.linspace(0,len(remaining)-1,max(0,args.samples-len(transient)))] if remaining else [])))
    records=[]
    for i in selected:
        previous=rows[i-1];c=rows[i]['controller']
        q=refiner.full_q(previous['actual_joint_names'],previous['actual_q43'])
        targets=(np.asarray(c['mapped_left_base_ee']),np.asarray(c['assisted_right_base_ee']))
        seed=original.validate_pose(targets[1],targets[0],q[refiner.arm])
        full_seed=q.copy();full_seed[refiner.arm]=seed['q_solution']
        _,seed_errors,_=refiner.evaluate(full_seed,targets)
        refined=refiner.refine(q,seed['q_solution'],*targets)
        current_seed=refiner.refine(q,q[refiner.arm],*targets)
        records.append(dict(tick=i,measured_state_from_tick=i-1,original_ik_valid=seed['ik_valid'],
            original_ik_elapsed_s=seed['elapsed_s'],original_ik_final=seed['final'],
            original_seed_full_q43_errors=seed_errors,original_seed_q14=seed['q_solution'],
            left_target=targets[0].tolist(),right_target=targets[1].tolist(),refinement=refined,
            refinement_from_current=current_seed))
    # Numerical independent derivative check at an actual recorded state.
    first=records[0];q=np.asarray(first['refinement']['q_initial_43']);targets=(np.asarray(first['left_target']),np.asarray(first['right_target']))
    _,_,analytic=refiner.evaluate(q,targets,jacobian=True);poses=refiner.forward(q);fd=np.zeros_like(analytic);eps=1e-7
    for j,k in enumerate(refiner.arm):
        trial=q.copy();trial[k]+=eps;changed=refiner.forward(trial)
        for s,(before,after) in enumerate(zip(poses,changed)):
            fd[s*6:s*6+3,j]=(after[:3,3]-before[:3,3])/eps
            fd[s*6+3:s*6+6,j]=np.asarray(refiner.pin.log3(after[:3,:3]@before[:3,:3].T))/eps
    adverse={};base=refiner.forward(q)
    unreachable=base[1].copy();unreachable[0,3]+=2.
    adverse['unreachable_two_meters']=refiner.refine(q,q[refiner.arm],base[0],unreachable)
    over=q.copy();over[refiner.arm[0]]=refiner.model.upperPositionLimit[refiner.arm[0]]+.001
    adverse['measured_hard_limit_violation']=refiner.refine(over,over[refiner.arm],*base)
    # Full straight-arm configuration is singular in some translation directions.
    near=np.clip(np.zeros(43),refiner.model.lowerPositionLimit,refiner.model.upperPositionLimit)
    near_poses=refiner.forward(near);bad=near_poses[1].copy();bad[0,3]+=.1
    adverse['straight_arm_infeasible_translation']=refiner.refine(near,near[refiner.arm],near_poses[0],bad)
    half_turn=near_poses[1].copy();half_turn[:3,:3]=refiner.pin.exp3(np.array([0.,0.,np.pi-1e-6]))@half_turn[:3,:3]
    adverse['near_pi_rotation_outside_joint_step_bound']=refiner.refine(near,near[refiner.arm],near_poses[0],half_turn)
    huge_seed=q[refiner.arm].copy();huge_seed[0]=1e5
    adverse['out_of_limits_seed_projected_not_authorized']=refiner.refine(q,huge_seed,base[0],unreachable)
    elapsed=np.array([r['refinement']['elapsed_s'] for r in records])
    current_elapsed=np.array([r['refinement_from_current']['elapsed_s'] for r in records])
    output=dict(schema='punktnull.offline_full_ik_prototype.v1',scope='no_motion_no_DDS_no_collision_authorization',
        configuration=asdict(refiner.config),rows_sha256=hashlib.sha256(source).hexdigest(),
        prototype_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_manifest=original.manifest,urdf_sha256=hashlib.sha256(refiner.urdf.read_bytes()).hexdigest(),
        samples=len(records),accepted=sum(r['refinement']['accepted'] for r in records),
        elapsed_s=dict(median=float(np.median(elapsed)),p95=float(np.quantile(elapsed,.95)),maximum=float(elapsed.max())),
        current_seed_accepted=sum(r['refinement_from_current']['accepted'] for r in records),
        current_seed_elapsed_s=dict(median=float(np.median(current_elapsed)),p95=float(np.quantile(current_elapsed,.95)),maximum=float(current_elapsed.max())),
        max_jacobian_finite_difference_error=float(np.max(abs(fd-analytic))),adverse=adverse,records=records,
        limitations=['Offline one-step replay from each measured previous physics row; refined commands were never applied.',
            'No collision or source freshness validation here; production would retain those independent gates.',
            'Timing is observed CPU wall time under this run, not a hard real-time guarantee.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:output[k] for k in ('samples','accepted','elapsed_s','current_seed_accepted','current_seed_elapsed_s','max_jacobian_finite_difference_error')},indent=2))
    print(json.dumps({n:{k:r.get(k) for k in ('accepted','reason','elapsed_s','iterations')} for n,r in adverse.items()},indent=2))


if __name__=='__main__':main()
