"""CPU-only production full-state IK checks against original Pinocchio FK."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from teleop_assist.copilot.full_ik import FullStateIKConfig


@pytest.mark.parametrize('changed',[{'position_tolerance_m':.001},{'orientation_tolerance_rad':.02},
    {'max_joint_delta_rad':.2},{'max_iterations':13},{'max_backtracks':7},{'max_elapsed_s':.02},
    {'damping':0},{'orientation_weight_m':float('nan')},{'max_iterations':True}])
def test_full_state_configuration_cannot_relax_its_contract(changed):
    with pytest.raises(ValueError):FullStateIKConfig(**changed)


@pytest.fixture(scope='module')
def cases(tmp_path_factory):
    python=Path(os.environ.get('COPILOT_ORIGINAL_IK_PYTHON','/home/loya2/miniconda3/envs/to/bin/python'))
    repo=Path(os.environ.get('COPILOT_ORIGINAL_XR_REPOSITORY','/home/loya2/xr_teleoperate'))
    if not python.is_file() or not repo.is_dir():pytest.skip('CPU Pinocchio/original URDF unavailable')
    output=tmp_path_factory.mktemp('production_full_ik')/'cases.json'
    code=r'''
import json,sys,numpy as np,pinocchio as pin
from teleop_assist.copilot.full_ik import FullStateIKSolver
model=pin.buildModelFromUrdf(sys.argv[1]+'/assets/g1/g1_body29_hand14.urdf');f=FullStateIKSolver(model)
ix={str(n):model.joints[i].idx_q for i,n in enumerate(model.names) if i>0}
q=np.zeros(43);q[ix['waist_yaw_joint']]=.04;q[ix['waist_pitch_joint']]=.03
q[ix['left_hand_index_0_joint']]=-.2;q[ix['right_hand_index_0_joint']]=.2
q[ix['left_shoulder_pitch_joint']]=.4;q[ix['right_shoulder_pitch_joint']]=.4
q[ix['left_shoulder_roll_joint']]=.3;q[ix['right_shoulder_roll_joint']]=-.3
q[ix['left_elbow_joint']]=.4;q[ix['right_elbow_joint']]=.4
data=model.createData()
def forward(values):
 pin.framesForwardKinematics(model,data,values);tcp=np.eye(4);tcp[0,3]=.05
 return [(data.oMf[model.getFrameId(s+'_wrist_yaw_link')].homogeneous@tcp).copy() for s in ('left','right')]
poses=forward(q);out={};out['unchanged']=f.solve(*poses,q)
goal=q.copy();goal[ix['right_wrist_yaw_joint']]+=.02;goal[ix['left_wrist_roll_joint']]-=.01
targets=forward(goal);out['reachable']=f.solve(*targets,q)
sol=np.asarray(out['reachable']['q_solution_43']);out['reachable']['independent_errors']=f.evaluate(sol,targets)[1]
far=poses[1].copy();far[0,3]+=2.;out['unreachable']=f.solve(poses[0],far,q)
bad=q.copy();bad[f.arm[0]]=model.upperPositionLimit[f.arm[0]]+.001
out['current_limit_violation']=f.solve(*poses,bad)
seed=q[f.arm].copy();seed[0]=100.;out['invalid_seed_unreachable']=f.solve(poses[0],far,q,seed_q14=seed)
bad=q.copy();bad[1]=np.nan;out['nonfinite']=f.solve(*poses,bad)
near=np.clip(np.zeros(43),model.lowerPositionLimit,model.upperPositionLimit);nposes=forward(near)
half=nposes[1].copy();half[:3,:3]=pin.exp3(np.array([0.,np.pi-1e-8,0.]))@half[:3,:3]
out['near_pi']=f.solve(nposes[0],half,near)
far=nposes[1].copy();far[0,3]+=.1;out['near_singular_unreachable']=f.solve(nposes[0],far,near)
_,_,J=f.evaluate(q,targets,jacobian=True);fd=np.zeros_like(J);eps=1e-7
for j,k in enumerate(f.arm):
 trial=q.copy();trial[k]+=eps;after=forward(trial)
 for s,(a,b) in enumerate(zip(poses,after)):
  fd[s*6:s*6+3,j]=(b[:3,3]-a[:3,3])/eps
  fd[s*6+3:s*6+6,j]=np.asarray(pin.log3(b[:3,:3]@a[:3,:3].T))/eps
out['jacobian_max_error']=float(np.max(abs(J-fd)))
times=iter([0.,.011,.011]);f.clock=lambda:next(times)
out['deadline']=f.solve(*poses,q)
json.dump(out,open(sys.argv[2],'w'),allow_nan=False)
'''
    result=subprocess.run([str(python),'-c',code,str(repo),str(output)],
        cwd=Path(__file__).resolve().parents[1],text=True,capture_output=True,timeout=30,
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode==0,result.stdout+result.stderr
    return json.loads(output.read_text())


def test_original_50mm_TCP_jacobian_matches_independent_numerical_FK(cases):
    assert cases['jacobian_max_error']<1e-6


def test_both_arms_solve_with_current_waist_and_fingers_exactly_fixed(cases):
    r=cases['reachable']
    assert r['ik_valid'] and r['fixed_nonarm_joints_unchanged'] and r['limits_valid']
    assert r['max_joint_delta_rad']<=.15 and r['elapsed_s']<.01
    assert r['numerical_convergence_steps']<=12
    for side in ('left','right'):
        assert r['independent_errors'][side+'_position_error_m']<=.0005
        assert r['independent_errors'][side+'_orientation_error_rad']<=.00872664626
    assert not r['motion_authorized'] and not r['collision_valid'] and not r['trajectory_validated']


def test_no_target_change_means_no_joint_change(cases):
    r=cases['unchanged']
    assert r['ik_valid'] and not r['solver_invoked']
    assert r['q_initial_43']==r['q_solution_43']


@pytest.mark.parametrize('name',['unreachable','current_limit_violation','invalid_seed_unreachable',
                                'nonfinite','near_pi','near_singular_unreachable','deadline'])
def test_unreachable_nonfinite_limits_and_deadlines_never_pass(cases,name):
    r=cases[name]
    assert not r['ik_valid'] and not r['motion_authorized']
    assert r['numerical_convergence_steps']<=12
    if name=='deadline':assert r['reason']=='full_q43_ik_deadline_exceeded_or_clock_invalid'
    if 'q_solution' in r:
        assert r['fixed_nonarm_joints_unchanged'] and r['limits_valid'] and r['max_joint_delta_rad']<=.15
