"""CPU full-model checks of endpoint hypotheses, with no simulator or DDS."""
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture(scope='module')
def cases(tmp_path_factory):
    python=Path(os.environ.get('COPILOT_ORIGINAL_IK_PYTHON','/home/loya2/miniconda3/envs/to/bin/python'))
    repo=Path(os.environ.get('COPILOT_ORIGINAL_XR_REPOSITORY','/home/loya2/xr_teleoperate'))
    if not python.is_file() or not repo.is_dir():pytest.skip('Original Pinocchio environment unavailable')
    output=tmp_path_factory.mktemp('endpoint_hypothesis')/'cases.json'
    code=r'''
import json,sys,numpy as np,pinocchio as pin
from types import SimpleNamespace
from teleop_assist.copilot.endpoint_ik import EndpointIKValidator
from teleop_assist.copilot.kinematics import ARM_JOINT_NAMES
model=pin.buildModelFromUrdf(sys.argv[1]+'/assets/g1/g1_body29_hand14.urdf');data=model.createData()
ix={str(n):model.joints[i].idx_q for i,n in enumerate(model.names) if i>0}
arms=np.array([ix[n] for n in ARM_JOINT_NAMES]);q=np.zeros(43)
q[ix['waist_yaw_joint']]=.04;q[ix['waist_pitch_joint']]=.03
for s in ('left','right'):
 q[ix[s+'_shoulder_pitch_joint']]=.4;q[ix[s+'_elbow_joint']]=.4
 q[ix[s+'_shoulder_roll_joint']]=.3 if s=='left' else -.3
def forward(q):
 pin.framesForwardKinematics(model,data,q);tcp=np.eye(4);tcp[0,3]=.05
 return [(data.oMf[model.getFrameId(s+'_wrist_yaw_link')].homogeneous@tcp).copy() for s in ('left','right')]
goal=q.copy();goal[arms[-1]]+=.05;left,right=forward(goal)
class FakeOriginal:
 config=SimpleNamespace(max_position_error_m=.01,max_orientation_error_rad=np.deg2rad(10))
 manifest={'urdf_sha256':'test-model-hash','scope':'test_numerical_fixture'}
 def __init__(self):self.calls=0;self.seed=goal[arms].copy();self.success=True;self.noop=False
 def validate_pose(self,r,l,current):
  self.calls+=1
  return dict(ik_valid=True,solver_invoked=not self.noop,q_initial=current.tolist(),
   q_solution=self.seed.tolist(),iterations=[],numerical_convergence_steps=0 if self.noop else 6,
   validation_method='independent_current_pose_FK' if self.noop else 'original_IK_plus_independent_FK',
   reason='already_at_target' if self.noop else 'endpoint_ik_valid_collision_unchecked',
   final=dict(solver_success=None if self.noop else self.success,limits_valid=True,
    solver_status='not_run_already_at_target' if self.noop else 'Solve_Succeeded'))
fake=FakeOriginal();f=EndpointIKValidator(model,fake);out={'first':f.validate_pose(right,left,q)}
new=q.copy();new[arms[0]]+=.03;new[ix['left_hand_index_0_joint']]=-.2;new[ix['right_hand_index_0_joint']]=.2
newleft=forward(new)[0];out['current_other_joints']=f.validate_pose(right,newleft,new);out['calls_after_hit']=fake.calls
out['current_other_joints']['expected_nonarm']=new[f.nonarm].tolist();out['current_other_joints']['expected_left']=new[arms[:7]].tolist()
out['current_other_joints']['nonarm_indices']=f.nonarm.tolist();out['current_other_joints']['arm_indices']=arms.tolist()
out['current_other_joints']['q_solution'][7]=100.
out['return_mutation_does_not_corrupt_cache']=f.validate_pose(right,newleft,new)
changed=right.copy();changed[0,3]+=.02;out['changed_pose']=f.validate_pose(changed,newleft,new)
out['calls_after_changed_pose']=fake.calls
f=EndpointIKValidator(model,FakeOriginal());f.validate_pose(right,left,q)
waist=q.copy();waist[ix['waist_yaw_joint']]+=.2;out['waist_changed']=f.validate_pose(right,forward(waist)[0],waist)
f=EndpointIKValidator(model,FakeOriginal());f.validate_pose(right,left,q)
farleft=left.copy();farleft[1,3]+=.02;out['left_target_changed']=f.validate_pose(right,farleft,q)
f=EndpointIKValidator(model,FakeOriginal());f.validate_pose(right,left,q)
f._cached_q14[7]=model.upperPositionLimit[arms[7]]+1.
out['cached_limit_violation']=f.validate_pose(right,left,q)
for name,value in [('bad_shape',q[:14]),('nonfinite',np.full(43,np.nan)),('limits',q.copy())]:
 if name=='limits':value[arms[0]]=model.upperPositionLimit[arms[0]]+.000001
 out[name]=f.validate_pose(right,left,value)
bad=right.copy();bad[:3,:3]=pin.exp3(np.array([np.pi-1e-8,0.,0.]))@right[:3,:3]
out['near_pi']=f.validate_pose(bad,left,q)
failed=FakeOriginal();failed.success=False;out['failed_optimizer']=EndpointIKValidator(model,failed).validate_pose(right,left,q)
failed=FakeOriginal();failed.seed[1]=np.nan;out['invalid_optimizer_q']=EndpointIKValidator(model,failed).validate_pose(right,left,q)
fake=FakeOriginal();fake.noop=True;fake.seed=q[arms].copy();currentposes=forward(q)
out['original_noop']=EndpointIKValidator(model,fake).validate_pose(currentposes[1],currentposes[0],q)
near=currentposes[1].copy();near[0,3]+=.00002
out['original_noop_not_strict']=EndpointIKValidator(model,fake).validate_pose(near,currentposes[0],q)
clock=iter([1.,0.]);out['clock_regression']=EndpointIKValidator(model,FakeOriginal(),clock=lambda:next(clock)).validate_pose(right,left,q)
fake=FakeOriginal();fake.config=SimpleNamespace(max_position_error_m=.02,max_orientation_error_rad=np.deg2rad(20))
out['limits_cannot_relax']=EndpointIKValidator(model,fake).manifest
json.dump(out,open(sys.argv[2],'w'),allow_nan=False)
'''
    process=subprocess.run([str(python),'-c',code,str(repo),str(output)],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30,
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    assert process.returncode==0,process.stdout+process.stderr
    return json.loads(output.read_text())


def test_current_full_model_FK_replaces_repeated_optimizer_without_permission(cases):
    first,cached=cases['first'],cases['current_other_joints']
    assert first['ik_valid'] and first['original_solver_invoked'] and not first['cache_hit']
    assert cached['ik_valid'] and cached['cache_hit'] and cached['cache_revalidated_this_call']
    assert cached['reason']=='cached_endpoint_FK'
    assert cached['validation_method']=='independent_cached_endpoint_full_q43_FK'
    assert cached['final']['solver_status']=='validated_cached_endpoint_FK'
    assert cached['final']['solver_success'] is None
    assert not cached['solver_invoked'] and not cached['original_solver_invoked']
    assert cases['calls_after_hit']==1
    assert cached['cache_candidate_key']==first['cache_candidate_key']
    assert cached['q_initial']!=first['q_initial']
    assert cached['limits_valid'] and cached['fixed_nonarm_joints_unchanged']
    q=cached['q_solution_43']
    assert [q[i] for i in cached['nonarm_indices']]==cached['expected_nonarm']
    assert [q[i] for i in cached['arm_indices'][:7]]==cached['expected_left']
    for side in ('left','right'):
        assert cached['final'][side+'_position_error_m']<1e-9
        assert cached['final'][side+'_orientation_error_rad']<1e-9
    for key in ('motion_authorized','cache_authority','collision_valid','trajectory_validated','joint_state_freshness_validated'):
        assert cached[key] is False


def test_returned_arrays_cannot_modify_the_internal_numerical_hypothesis(cases):
    r=cases['return_mutation_does_not_corrupt_cache']
    assert r['ik_valid'] and r['cache_hit'] and max(abs(x) for x in r['q_solution'])<4


@pytest.mark.parametrize('name',['changed_pose','waist_changed','left_target_changed','bad_shape',
    'nonfinite','limits','near_pi','failed_optimizer','invalid_optimizer_q','original_noop_not_strict','clock_regression'])
def test_bad_current_state_target_or_optimizer_never_bypasses_FK(name,cases):
    assert not cases[name]['ik_valid'] and not cases[name]['motion_authorized']


def test_target_identity_change_and_invalid_cache_use_original_fallback(cases):
    assert cases['calls_after_changed_pose']==2
    assert cases['changed_pose']['cache_candidate_key']!=cases['first']['cache_candidate_key']
    invalid=cases['cached_limit_violation']
    assert invalid['ik_valid'] and not invalid['cache_hit'] and invalid['original_solver_invoked']
    assert invalid['cached_hypothesis_check']['limits_valid'] is False


def test_real_current_pose_noop_remains_separate_from_cached_endpoint_FK(cases):
    r=cases['original_noop']
    assert r['ik_valid'] and r['reason']=='already_at_target' and not r['cache_hit']
    assert r['q_solution_43']==r['q_initial_43'] and r['final']['solver_success'] is None


def test_looser_original_configuration_cannot_relax_existing_endpoint_tolerances(cases):
    assert cases['limits_cannot_relax']['position_tolerance_m']==.01
    assert cases['limits_cannot_relax']['orientation_tolerance_rad']<.174533
