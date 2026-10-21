"""Persistent CPU motion guard and lease contract, without actuator interfaces."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from teleop_assist.copilot.guard import MotionGuardConfig, command_is_current


def test_command_consumer_rejects_expired_future_and_wrong_clock():
    result = {"allowed": True, "hold_required": False, "clock_id": "test", "validated_at": 10.,
              "valid_until": 10.1, "q14_command": [0.] * 14}
    assert command_is_current(result, now=10.05, clock_id="test")
    assert not command_is_current(result, now=10.1, clock_id="test")
    assert not command_is_current(result, now=9.9, clock_id="test")
    assert not command_is_current(result, now=10.05, clock_id="other")
    assert not command_is_current(dict(result, hold_required=True), now=10.05, clock_id="test")
    assert not command_is_current(dict(result, allowed=False), now=10.05, clock_id="test")


@pytest.mark.parametrize("values", [{"max_operator_age_s": 0}, {"max_robot_state_age_s": 1},
                                    {"future_tolerance_s": .1}, {"max_joint_command_change_rad": .5}])
def test_guard_configuration_has_bounded_leases_and_joint_change(values):
    with pytest.raises(ValueError):
        MotionGuardConfig(**values)


@pytest.fixture(scope="module")
def actual_guard_results(tmp_path_factory):
    repository = Path(os.environ.get("COPILOT_ORIGINAL_XR_REPOSITORY", "/home/loya2/xr_teleoperate"))
    python = Path(os.environ.get("COPILOT_ORIGINAL_IK_PYTHON", "/home/loya2/miniconda3/envs/to/bin/python"))
    if not repository.is_dir() or not python.is_file():
        pytest.skip("Original G1 geometry and its CPU Pinocchio environment are not installed")
    out = tmp_path_factory.mktemp("original_motion_guard") / "tests.json"
    code = r'''
import copy,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from teleop_assist.copilot.guard import ActualMotionGuard,MotionGuardConfig,command_is_current,JOINT_REFERENCE_SCHEMA
from teleop_assist.copilot.kinematics import ARM_JOINT_NAMES
import teleop_assist.copilot.guard as module
g=ActualMotionGuard(sys.argv[1]);v=g.collision;q=np.zeros(43);base=np.eye(4);base[2,3]=3
v.pin.framesForwardKinematics(v.model,v.data,q)
observed={f.name:(base@v.data.oMf[i].homogeneous).tolist() for i,f in enumerate(v.model.frames)}
tcp=np.eye(4);tcp[0,3]=.05
left=v.data.oMf[v.model.getFrameId('left_wrist_yaw_link')].homogeneous@tcp
right=v.data.oMf[v.model.getFrameId('right_wrist_yaw_link')].homogeneous@tcp
state={'joint_positions':{name:0. for name in v.names},'base_in_world':base.tolist(),
       'observed_body_world_poses':observed,'complete':True,'source':'synthetic_FK_fixture',
       'acquisition_timestamp':10.,'clock_id':'test'}
scene={'geometry_complete':True,'static':True,'source':'explicit_empty_above_floor_fixture',
       'acquisition_timestamp':10.,'clock_id':'test','obstacles':[{'id':'floor','type':'plane','normal':[0,0,1],'offset':0.}]}
cases={};bench=[]
def run(name, s=state, sc=scene, r=right, l=left, **kwargs):
 args={'acquisition_timestamp':10.,'clock_id':'test','now':10.,'tracking_valid':True};args.update(kwargs)
 result=g.validate(l,r,s,sc,**args);cases[name]=result;return result
for i in range(12):
 result=run('clear',now=10.);bench.append(result['elapsed_s'])
run('tracking_missing',tracking_valid=False)
run('operator_stale',acquisition_timestamp=9.)
run('operator_future',acquisition_timestamp=11.)
run('robot_stale',s=dict(state,acquisition_timestamp=9.))
run('scene_stale',sc=dict(scene,acquisition_timestamp=9.))
run('scene_clock_mismatch',sc=dict(scene,clock_id='camera'))
s=copy.deepcopy(state);s['joint_positions'].pop('left_hand_thumb_2_joint');run('finger_missing',s=s)
sc=copy.deepcopy(scene);T=np.eye(4);T[:3,3]=(base@right)[:3,3]
sc['obstacles'].append({'id':'blocked','type':'box','size_xyz':[.2,.2,.2],'world_pose':T.tolist()});run('collision',sc=sc)
g.config=MotionGuardConfig(max_joint_command_change_rad=.001)
r=right.copy();r[:3,:3]=Rotation.from_euler('x',12,degrees=True).as_matrix()@r[:3,:3]
run('joint_step',r=r)
g.config=MotionGuardConfig()
# Deterministically represent a validation calculation exceeding the lease.
ticks=iter([100.,100.2,100.2]);old=module.time.monotonic;module.time.monotonic=lambda:next(ticks)
try:run('expired_during_computation')
finally:module.time.monotonic=old
# Actual current-seeded solver and full collision path, before replacing IK
# methods with sentinels for the strict no-op/hint branch tests below.
qt=q.copy();qt[v.indices['right_wrist_yaw_joint']]=.02;qt[v.indices['left_wrist_roll_joint']]=-.01
v.pin.framesForwardKinematics(v.model,v.data,qt)
lt=v.data.oMf[v.model.getFrameId('left_wrist_yaw_link')].homogeneous@tcp
rt=v.data.oMf[v.model.getFrameId('right_wrist_yaw_link')].homogeneous@tcp
run('full_solver_direct',l=lt,r=rt)
real_solve=g.full_ik.solve;fallback_calls=[]
def forced_first_failure(*args,**kwargs):
 fallback_calls.append(bool(kwargs.get('seed_q14') is not None))
 if len(fallback_calls)==1:return dict(ik_valid=False,reason='forced_direct_failure',elapsed_s=0.,numerical_convergence_steps=0)
 return real_solve(*args,**kwargs)
g.full_ik.solve=forced_first_failure
run('full_solver_fallback',l=lt,r=rt)
cases['full_solver_fallback']['fallback_calls']=fallback_calls
g.full_ik.solve=lambda *args,**kwargs:dict(ik_valid=False,reason='forced_strict_refinement_failure')
run('full_solver_failed_fallback_cannot_release_original',l=lt,r=rt)
g.full_ik.solve=real_solve
bad=copy.deepcopy(state);bad['observed_body_world_poses']['right_wrist_yaw_link'][0][3]+=1.
run('full_solver_body_mismatch',s=bad,l=lt,r=rt)
blocked=copy.deepcopy(scene);T=np.eye(4);T[:3,3]=(base@right)[:3,3]
blocked['obstacles'].append({'id':'full_solver_blocked','type':'box','size_xyz':[.2,.2,.2],'world_pose':T.tolist()})
run('full_solver_collision',sc=blocked,l=lt,r=rt)
# The full measured waist changes both EE poses; reduced original IK locks it
# to zero. The exact full-q43 pose must hold without invoking that optimizer.
qfull=q.copy();qfull[v.indices['waist_yaw_joint']]=.00036;qfull[v.indices['waist_pitch_joint']]=.00038
v.pin.framesForwardKinematics(v.model,v.data,qfull)
full_observed={f.name:(base@v.data.oMf[i].homogeneous).tolist() for i,f in enumerate(v.model.frames)}
lf=v.data.oMf[v.model.getFrameId('left_wrist_yaw_link')].homogeneous@tcp
rf=v.data.oMf[v.model.getFrameId('right_wrist_yaw_link')].homogeneous@tcp
sf=copy.deepcopy(state);sf['joint_positions']={name:float(qfull[v.indices[name]]) for name in v.names}
sf['observed_body_world_poses']=full_observed
solver_calls=[]
def sentinel(*args,**kwargs):
 solver_calls.append(True)
 raise RuntimeError('normal_IK_invocation_sentinel')
g.ik.validate_pose=sentinel
g.full_ik.solve=sentinel
run('full_waist_null',s=sf,l=lf,r=rf)
cases['full_waist_null']['observed_optimizer_calls']=len(solver_calls)
for side in ('left','right'):
 for kind in ('position','rotation'):
  lt,rt=lf.copy(),rf.copy();target=lt if side=='left' else rt
  if kind=='position':target[0,3]+=2e-5
  else:target[:3,:3]=Rotation.from_rotvec([0,0,2e-4]).as_matrix()@target[:3,:3]
  before=len(solver_calls)
  run('full_waist_changed_'+side+'_'+kind,s=sf,l=lt,r=rt)
  cases['full_waist_changed_'+side+'_'+kind]['observed_optimizer_calls']=len(solver_calls)-before
sc=copy.deepcopy(scene);T=np.eye(4);T[:3,3]=(base@rf)[:3,3]
sc['obstacles'].append({'id':'full_waist_blocked','type':'box','size_xyz':[.2,.2,.2],'world_pose':T.tolist()})
run('full_waist_collision',s=sf,sc=sc,l=lf,r=rf)
bad=copy.deepcopy(sf);bad['observed_body_world_poses']['right_wrist_yaw_link'][0][3]+=1.
run('full_waist_body_mismatch',s=bad,l=lf,r=rf)
bad=copy.deepcopy(sf);bad['joint_positions']['waist_pitch_joint']=99.
run('full_waist_bad_limits',s=bad,l=lf,r=rf)
run('full_waist_incomplete',s=dict(sf,complete=False),l=lf,r=rf)
run('full_waist_stale',s=dict(sf,acquisition_timestamp=9.),l=lf,r=rf)
# Controlled synthetic settling beyond the strict 100-urad no-op bound.
# The clutch target and hypothesis are from qfull; every measured body pose
# and non-arm joint below instead describes the CURRENT perturbed full q43.
reference=dict(schema=JOINT_REFERENCE_SCHEMA,source='synthetic_clutch_measurement',
 acquisition_timestamp=1.,clock_id='test',calibration_id='fixture',
 joint_names=list(ARM_JOINT_NAMES),q14=qfull[v.arm_indices].tolist())
qsettled=qfull.copy();qsettled[v.indices['right_wrist_yaw_joint']]=.00015
qsettled[v.indices['left_hand_index_0_joint']]=-.001
def state_at(values):
 v.pin.framesForwardKinematics(v.model,v.data,values)
 result=copy.deepcopy(sf)
 result['joint_positions']={name:float(values[v.indices[name]]) for name in v.names}
 result['observed_body_world_poses']={f.name:(base@v.data.oMf[i].homogeneous).tolist()
    for i,f in enumerate(v.model.frames)}
 return result
settled=state_at(qsettled)
before=len(solver_calls)
run('clutch_hint',s=settled,l=lf,r=rf,joint_reference=reference)
cases['clutch_hint']['observed_optimizer_calls']=len(solver_calls)-before
for side in ('left','right'):
 for kind in ('position','rotation'):
  lt,rt=lf.copy(),rf.copy();target=lt if side=='left' else rt
  if kind=='position':target[0,3]+=2e-5
  else:target[:3,:3]=Rotation.from_rotvec([0,0,4e-4]).as_matrix()@target[:3,:3]
  before=len(solver_calls)
  run('clutch_hint_changed_'+side+'_'+kind,s=settled,l=lt,r=rt,joint_reference=reference)
  cases['clutch_hint_changed_'+side+'_'+kind]['observed_optimizer_calls']=len(solver_calls)-before
for mutation in ('schema','clock','future','joint_order','nan','boolean','length','limits','missing_source'):
 hint=copy.deepcopy(reference)
 if mutation=='schema':hint['schema']='unknown'
 elif mutation=='clock':hint['clock_id']='other'
 elif mutation=='future':hint['acquisition_timestamp']=11.
 elif mutation=='joint_order':hint['joint_names']=hint['joint_names'][::-1]
 elif mutation=='nan':hint['q14'][0]=float('nan')
 elif mutation=='boolean':hint['q14'][0]=False
 elif mutation=='length':hint['q14'].pop()
 elif mutation=='limits':hint['q14'][0]=99.
 else:hint.pop('source')
 before=len(solver_calls)
 run('clutch_hint_invalid_'+mutation,s=settled,l=lf,r=rf,joint_reference=hint)
 cases['clutch_hint_invalid_'+mutation]['observed_optimizer_calls']=len(solver_calls)-before
qw=qsettled.copy();qw[v.indices['waist_yaw_joint']]+=.001
run('clutch_hint_current_waist_changed',s=state_at(qw),l=lf,r=rf,joint_reference=reference)
bad=copy.deepcopy(settled);bad['observed_body_world_poses']['right_wrist_yaw_link'][0][3]+=1.
run('clutch_hint_body_mismatch',s=bad,l=lf,r=rf,joint_reference=reference)
run('clutch_hint_collision',s=settled,sc=sc,l=lf,r=rf,joint_reference=reference)
run('clutch_hint_stale',s=dict(settled,acquisition_timestamp=9.),l=lf,r=rf,joint_reference=reference)
bad=copy.deepcopy(settled);bad['joint_positions'].pop('left_hand_index_0_joint')
run('clutch_hint_missing_finger',s=bad,l=lf,r=rf,joint_reference=reference)
qfar=qsettled.copy();qfar[v.indices['right_wrist_yaw_joint']]=.2
run('clutch_hint_step_bound',s=state_at(qfar),l=lf,r=rf,joint_reference=reference)
json.dump({'cases':cases,'elapsed_s':bench,'manifest':g.manifest},open(sys.argv[2],'w'),allow_nan=False)
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MPLCONFIGDIR=str(out.parent / "mpl"))
    result = subprocess.run([str(python), "-c", code, str(repository), str(out)],
                            cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(out.read_text())


def test_real_original_ik_and_geometry_produce_an_expiring_command(actual_guard_results):
    result = actual_guard_results["cases"]["clear"]
    assert result["allowed"] and not result["hold_required"]
    assert len(result["q14_command"]) == 14
    assert result["ik_validation"]["ik_valid"]
    assert result["collision_validation"]["collision_valid"]
    assert result["valid_until"] == pytest.approx(10.15)
    assert result["robot_command_sent"] is False
    assert command_is_current(result, now=result["validated_at"], clock_id="test")


@pytest.mark.parametrize("name", ["tracking_missing", "operator_stale", "operator_future", "robot_stale",
                                  "scene_stale", "scene_clock_mismatch", "finger_missing", "collision", "joint_step",
                                  "expired_during_computation"])
def test_adverse_input_returns_hold_and_never_a_joint_command(actual_guard_results, name):
    result = actual_guard_results["cases"][name]
    assert not result["allowed"] and result["hold_required"]
    assert result["q14_command"] is None


def test_validation_time_is_deducted_from_command_lease(actual_guard_results):
    result = actual_guard_results["cases"]["expired_during_computation"]
    assert result["reason"] == "command_lease_expired_during_validation"
    assert result["elapsed_s"] == pytest.approx(.2)


def test_current_seeded_full_solver_still_requires_complete_short_path(actual_guard_results):
    r=actual_guard_results['cases']['full_solver_direct']
    assert r['allowed'] and r['ik_validation']['seed_source']=='current_measured_q14'
    assert r['ik_validation']['validation_method']=='bounded_full_q43_DLS_plus_independent_FK'
    assert 'original_ik_fallback_seed' not in r
    assert r['collision_validation']['path_validated'] and r['urdf_vs_observed_frames']['valid']
    assert r['reason']=='full_q43_FK_and_conservative_short_joint_path_valid'
    for side in ('left','right'):
        assert r['ik_validation']['final'][side+'_position_error_m']<=.0005
        assert r['ik_validation']['final'][side+'_orientation_error_rad']<=.00872664626


def test_original_IK_fallback_is_only_a_seed_for_same_strict_refinement(actual_guard_results):
    r=actual_guard_results['cases']['full_solver_fallback']
    assert r['allowed'] and r['fallback_calls']==[False,True]
    assert r['original_ik_fallback_seed']['ik_valid']
    assert r['ik_validation']['seed_source']=='original_reduced_IK_numerical_seed_only'
    assert r['collision_validation']['path_validated']
    failed=actual_guard_results['cases']['full_solver_failed_fallback_cannot_release_original']
    assert failed['original_ik_fallback_seed']['ik_valid']
    assert not failed['allowed'] and failed['q14_command'] is None


@pytest.mark.parametrize('fault',['body_mismatch','collision'])
def test_strict_full_solver_does_not_replace_scene_and_current_body_checks(actual_guard_results,fault):
    r=actual_guard_results['cases']['full_solver_'+fault]
    assert r['ik_validation']['ik_valid']
    assert not r['allowed'] and r['q14_command'] is None


def test_full_q43_nonzero_waist_null_target_holds_exactly_without_reduced_ik(actual_guard_results):
    result=actual_guard_results['cases']['full_waist_null']
    assert result['allowed'] and not result['hold_required']
    assert result['observed_optimizer_calls']==0
    assert result['max_joint_command_change_rad']==0.
    ik=result['ik_validation']
    assert ik['reason']=='already_at_target_full_q43' and ik['solver_invoked'] is False
    assert len(ik['q_initial_43'])==43 and ik['limits_valid'] is True
    assert result['q14_command']==ik['q_initial']==ik['q_solution']
    assert ik['collision_valid'] is False and ik['trajectory_validated'] is False
    assert result['urdf_vs_observed_frames']['valid']
    assert result['collision_validation']['path_validated'] is True
    assert result['collision_validation']['q_initial_43']==result['collision_validation']['q_final_43']


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('kind',['position','rotation'])
def test_full_state_nonzero_target_keeps_normal_ik_gate(actual_guard_results,side,kind):
    result=actual_guard_results['cases']['full_waist_changed_'+side+'_'+kind]
    assert result['observed_optimizer_calls']==1
    assert not result['allowed'] and result['q14_command'] is None
    assert result['reason']=='normal_IK_invocation_sentinel'


@pytest.mark.parametrize('name',['collision','body_mismatch','bad_limits','incomplete','stale'])
def test_full_state_null_target_never_bypasses_state_or_collision(actual_guard_results,name):
    result=actual_guard_results['cases']['full_waist_'+name]
    assert not result['allowed'] and result['hold_required'] and result['q14_command'] is None


def test_old_clutch_joint_hypothesis_is_independently_verified_at_current_full_state(actual_guard_results):
    result = actual_guard_results['cases']['clutch_hint']
    assert result['allowed'] and result['observed_optimizer_calls'] == 0
    assert result['max_joint_command_change_rad'] == pytest.approx(.00015)
    ik = result['ik_validation']
    assert ik['validation_method'] == 'independent_joint_reference_full_q43_FK'
    assert ik['solver_invoked'] is False and ik['final']['solver_success'] is None
    assert ik['q_initial'] != ik['q_solution'] == result['q14_command']
    evidence = result['joint_reference_validation']
    assert evidence['used'] and evidence['motion_authorized'] is False
    assert evidence['acquisition_timestamp'] == 1. and evidence['reference_age_s'] == 9.
    assert result['valid_until'] == pytest.approx(10.15)  # Only fresh input leases.
    collision = result['collision_validation']
    assert collision['path_validated'] and collision['manual_fingers_held_at_observed_values']
    assert collision['q_initial_43'] == ik['q_initial_43']
    assert collision['q_final_43'] == ik['q_solution_43']
    assert sum(abs(a-b) > 1e-12 for a,b in zip(ik['q_initial_43'],ik['q_solution_43'])) == 1
    assert -.001 in ik['q_initial_43'] and -.001 in ik['q_solution_43']  # Current finger state.


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('kind', ['position', 'rotation'])
def test_joint_reference_cannot_override_any_manual_or_assisted_EE_change(actual_guard_results,side,kind):
    result=actual_guard_results['cases']['clutch_hint_changed_'+side+'_'+kind]
    assert result['observed_optimizer_calls'] == 1 and not result['allowed']
    assert result['joint_reference_validation']['used'] is False
    assert result['reason'] == 'normal_IK_invocation_sentinel'


@pytest.mark.parametrize('mutation', ['schema','clock','future','joint_order','nan','boolean','length','limits','missing_source'])
def test_invalid_optional_reference_falls_back_without_authorizing_motion(actual_guard_results,mutation):
    result=actual_guard_results['cases']['clutch_hint_invalid_'+mutation]
    assert result['observed_optimizer_calls'] == 1 and not result['allowed']
    assert result['joint_reference_validation']['used'] is False
    assert result['reason'] == 'normal_IK_invocation_sentinel'


@pytest.mark.parametrize('name', ['current_waist_changed','body_mismatch','collision','stale','missing_finger','step_bound'])
def test_joint_reference_keeps_current_full_state_and_motion_gates(actual_guard_results,name):
    result=actual_guard_results['cases']['clutch_hint_'+name]
    assert not result['allowed'] and result['hold_required'] and result['q14_command'] is None
    if name=='current_waist_changed':
        assert result['joint_reference_validation']['used'] is False
        assert result['reason']=='normal_IK_invocation_sentinel'
    elif name=='step_bound':
        assert result['joint_reference_validation']['used'] is True
        assert result['reason']=='command_joint_step_exceeds_bound'
