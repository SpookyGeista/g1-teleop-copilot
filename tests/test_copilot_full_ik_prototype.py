"""Independent CPU checks of the offline prototype, never motion authorization."""
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture(scope='module')
def cases(tmp_path_factory):
    python=Path(os.environ.get('COPILOT_ORIGINAL_IK_PYTHON','/home/loya2/miniconda3/envs/to/bin/python'))
    repo=Path(os.environ.get('COPILOT_ORIGINAL_XR_REPOSITORY','/home/loya2/xr_teleoperate'))
    if not python.is_file() or not repo.is_dir():pytest.skip('CPU Pinocchio/original URDF unavailable')
    output=tmp_path_factory.mktemp('full_ik_prototype')/'cases.json'
    code=r'''
import json,sys,numpy as np
from tools.copilot.prototype_full_ik import FullStateRefiner
f=FullStateRefiner(sys.argv[1]+'/assets/g1/g1_body29_hand14.urdf')
q=np.zeros(43);q[f.indices['waist_yaw_joint']]=.04;q[f.indices['waist_pitch_joint']]=.03
q[f.indices['left_hand_index_0_joint']]=-.2;q[f.indices['right_hand_index_0_joint']]=.2
q[f.indices['left_shoulder_pitch_joint']]=.4;q[f.indices['right_shoulder_pitch_joint']]=.4
q[f.indices['left_shoulder_roll_joint']]=.3;q[f.indices['right_shoulder_roll_joint']]=-.3
q[f.indices['left_elbow_joint']]=.4;q[f.indices['right_elbow_joint']]=.4
poses=f.forward(q);out={}
out['unchanged']=f.refine(q,q[f.arm],*poses)
goal=q.copy();goal[f.indices['right_wrist_yaw_joint']]+=.02;goal[f.indices['left_wrist_roll_joint']]-=.01
targets=f.forward(goal);out['reachable']=f.refine(q,q[f.arm],*targets)
sol=np.asarray(out['reachable']['q_solution_43']);out['reachable']['independent_errors']=f.evaluate(sol,targets)[1]
far=poses[1].copy();far[0,3]+=2.;out['unreachable']=f.refine(q,q[f.arm],poses[0],far)
bad=q.copy();bad[f.arm[0]]=f.model.upperPositionLimit[f.arm[0]]+.001
out['current_limit_violation']=f.refine(bad,bad[f.arm],*poses)
seed=q[f.arm].copy();seed[0]=100.;out['invalid_seed_with_unreachable_target']=f.refine(q,seed,poses[0],far)
bad=q.copy();bad[1]=np.nan;out['nonfinite']=f.refine(bad,q[f.arm],*poses)
near=np.clip(np.zeros(43),f.model.lowerPositionLimit,f.model.upperPositionLimit);nposes=f.forward(near)
half=nposes[1].copy();half[:3,:3]=f.pin.exp3(np.array([0.,np.pi-1e-8,0.]))@half[:3,:3]
out['near_pi']=f.refine(near,near[f.arm],nposes[0],half)
far=nposes[1].copy();far[0,3]+=.1;out['near_singular_unreachable']=f.refine(near,near[f.arm],nposes[0],far)
# A scheduler pause never produces an accepted late result.
times=iter([0.,.011,.011]);f.clock=lambda:next(times)
out['deadline']=f.refine(q,q[f.arm],*poses)
json.dump(out,open(sys.argv[2],'w'),allow_nan=False)
'''
    result=subprocess.run([str(python),'-c',code,str(repo),str(output)],
        cwd=Path(__file__).resolve().parents[1],text=True,capture_output=True,timeout=30,
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode==0,result.stdout+result.stderr
    return json.loads(output.read_text())


def test_independent_full_model_refinement_preserves_nonarm_state_and_bounds(cases):
    result=cases['reachable']
    assert result['accepted'] and result['fixed_nonarm_joints_unchanged']
    assert result['hard_joint_limits_satisfied'] and result['max_joint_delta_rad']<=.15+1e-12
    assert result['elapsed_s']<.01 and result['iterations']<=12
    for side in ('left','right'):
        assert result['independent_errors'][side+'_position_error_m']<=.0005
        assert result['independent_errors'][side+'_orientation_error_rad']<=.00872664626
    assert result['motion_authorized'] is False and result['collision_valid'] is False


def test_already_reached_pose_has_no_artificial_joint_change(cases):
    result=cases['unchanged']
    assert result['accepted'] and result['iterations']==0
    assert result['q_initial_43']==result['q_solution_43']


@pytest.mark.parametrize('name',['unreachable','current_limit_violation','invalid_seed_with_unreachable_target',
                                'nonfinite','near_pi','near_singular_unreachable','deadline'])
def test_adverse_targets_and_pauses_are_bounded_rejections(cases,name):
    result=cases[name]
    assert not result['accepted'] and not result['motion_authorized']
    assert result['iterations']<=12
    if name=='deadline':assert result['reason']=='refinement_deadline_exceeded'
    if 'q_solution' in result:
        assert result['fixed_nonarm_joints_unchanged'] and result['hard_joint_limits_satisfied']
        assert result['max_joint_delta_rad']<=.15+1e-12
