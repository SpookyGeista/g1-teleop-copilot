"""Endpoint checks against the real Unitree solver, never robot motion tests.

Integration cases use the existing original-IK environment in a CPU child;
pytest and the main project need no Pinocchio/CasADi dependency.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from teleop_assist.copilot.kinematics import IKValidationConfig, arm_vector, isolated_validate


@pytest.mark.parametrize("value", [[], [0] * 13, [0] * 15, [[0] * 14], [float("nan")] * 14,
                                  [float("inf")] * 14])
def test_joint_state_requires_exact_finite_original_14d_order(value):
    with pytest.raises(ValueError):
        arm_vector(value)


def test_joint_state_is_copied():
    q = np.zeros(14)
    result = arm_vector(q)
    result[0] = 1
    assert q[0] == 0


@pytest.mark.parametrize("options", [dict(convergence_steps=3), dict(convergence_steps=7),
                                    dict(convergence_steps=True), dict(max_candidates=129),
                                    dict(max_position_error_m=0), dict(max_orientation_error_rad=np.inf),
                                    dict(joint_limit_tolerance_rad=.1)])
def test_configuration_cannot_weaken_gates_arbitrarily(options):
    with pytest.raises(ValueError):
        IKValidationConfig(**options)


@pytest.fixture(scope="module")
def actual_original_results(tmp_path_factory):
    repository = Path(os.environ.get("COPILOT_ORIGINAL_XR_REPOSITORY", "/home/loya2/xr_teleoperate"))
    python = Path(os.environ.get("COPILOT_ORIGINAL_IK_PYTHON", "/home/loya2/miniconda3/envs/to/bin/python"))
    if not python.is_file() or not (repository / "teleop/robot_control/robot_arm_ik.py").is_file():
        pytest.skip("Original Unitree checkout and its IK environment are not installed")
    project = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("actual_original_ik") / "evidence.json"
    code = r'''
import json, sys, copy
import numpy as np
from scipy.spatial.transform import Rotation
from teleop_assist.copilot.kinematics import OriginalIKValidator, IKValidationConfig

v = OriginalIKValidator(sys.argv[1])
q = np.array([.1,.1,0,.2,0,0,0,.1,-.1,0,.2,0,0,0])
left, right = v.forward(q)
other_q = q.copy()
other_q[10] = .45
other_q[13] = .35
_, other = v.forward(other_q)
unreachable = right.copy()
unreachable[:3,3] = [5,-3,4]
near_pi = right.copy()
near_pi[:3,:3] = Rotation.from_rotvec([0,np.pi-1e-8,0]).as_matrix() @ right[:3,:3]
poses = [('current',right), ('other_reachable',other), ('unreachable',unreachable), ('near_pi',near_pi)]
request = {'current_arm_q':q.tolist(), 'left_base_ee':left.tolist(),
           'joint_state_acquisition_timestamp':None, 'joint_state_clock_id':None,
           'joint_state_source':'offline_FK_fixture_not_a_robot_measurement',
           'batch':{'observation_id':'test', 'target_id':'test_object',
                    'candidates':[{'candidate_id':name, 'base_ee':pose.tolist(),
                                   'geometry_valid':True,'ik_valid':True,'collision_valid':True}
                                  for name,pose in poses]}}
first = v.validate_request(request)
request['batch']['candidates'].reverse()
reverse = v.validate_request(request)
malformed = dict(request)
malformed['batch'] = {'candidates':[{'candidate_id':'malformed','base_ee':np.zeros((4,4)).tolist(),
                                   'collision_valid':True, 'ik_valid':True}]}
bad = v.validate_request(malformed)
# If either target genuinely changes, the original optimizer must still run.
# A sentinel makes accidentally bypassing it observable without interpreting
# whether a regularized solution happens to move for a tiny target offset.
solver_calls=[]
def sentinel(*args,**kwargs):
 solver_calls.append(True)
 raise RuntimeError('optimizer_call_sentinel')
v.solver.solve_ik=sentinel
same=v.validate_pose(right,left,q)
same['observed_optimizer_calls']=len(solver_calls)
boundaries={}
for side in ('left','right'):
 for kind in ('position','orientation'):
  a,b=left.copy(),right.copy();target=a if side=='left' else b
  if kind=='position':target[0,3]+=2e-5
  else:target[:3,:3]=Rotation.from_rotvec([0,0,2e-4]).as_matrix()@target[:3,:3]
  before=len(solver_calls);result=v.validate_pose(b,a,q)
  result['observed_optimizer_calls']=len(solver_calls)-before
  boundaries[side+'_'+kind]=result
# Bounds are conjunctive with stricter user validation settings, never a new
# way to bypass a caller's smaller ordinary residual acceptance threshold.
v.config=IKValidationConfig(max_position_error_m=1e-7,max_orientation_error_rad=1e-6)
for kind in ('position','orientation'):
 target=right.copy()
 if kind=='position':target[0,3]+=5e-6
 else:target[:3,:3]=Rotation.from_rotvec([0,0,5e-5]).as_matrix()@target[:3,:3]
 boundaries['stricter_'+kind]=v.validate_pose(target,left,q)
invalid=q.copy();invalid[0]=99.
try:
 v.validate_pose(right,left,invalid)
 invalid_limits_rejected=False
except ValueError:
 invalid_limits_rejected=True
json.dump({'first':first, 'reverse':reverse, 'malformed':bad, 'request':request,
           'already_at_target':same, 'boundary_controls':boundaries,
           'invalid_current_limits_rejected':invalid_limits_rejected},
          open(sys.argv[2],'w'), allow_nan=False)
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLCONFIGDIR=str(output.parent / "mpl"))
    result = subprocess.run([str(python), "-c", code, str(repository), str(output)],
                            cwd=project, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(output.read_text()), repository, python


def test_original_solver_passes_known_fk_targets_with_independent_thresholds(actual_original_results):
    evidence, _, _ = actual_original_results
    rows = {row["candidate_id"]: row for row in evidence["first"]["candidates"]}
    for name in ("current", "other_reachable"):
        assert rows[name]["ik_valid"]
        diagnostics = rows[name]["kinematic_validation"]
        assert len(diagnostics["iterations"]) == (0 if name == 'current' else 6)
        assert all(row["solver_success"] and row["limits_valid"] for row in diagnostics["iterations"])
        final = diagnostics["final"]
        assert max(final[side + "_position_error_m"] for side in ("left", "right")) <= .01
        assert max(final[side + "_orientation_error_rad"] for side in ("left", "right")) <= np.deg2rad(10)


def test_current_measured_fk_pose_is_held_exactly_without_optimizer(actual_original_results):
    evidence, _, _ = actual_original_results
    result = evidence['already_at_target']
    assert result['ik_valid'] is True and result['reason'] == 'already_at_target'
    assert result['q_solution'] == result['q_initial'] == evidence['request']['current_arm_q']
    assert result['observed_optimizer_calls'] == 0 and result['solver_invoked'] is False
    assert result['numerical_convergence_steps'] == 0 and result['iterations'] == []
    assert result['final']['limits_valid'] is True and result['final']['solver_success'] is None
    for flag in ('collision_valid', 'joint_state_freshness_validated', 'trajectory_validated'):
        assert result[flag] is False


def test_either_arm_moving_requires_original_optimizer_and_regular_bounds(actual_original_results):
    evidence, _, _ = actual_original_results
    for name, result in evidence['boundary_controls'].items():
        assert result['solver_invoked'] is True, name
        assert not result['ik_valid'], name
        assert 'optimizer_call_sentinel' in result['reason'], name
        if not name.startswith('stricter'):
            assert result['observed_optimizer_calls'] == 1
    assert evidence['invalid_current_limits_rejected'] is True


def test_nonzero_reachable_target_still_solves_normally(actual_original_results):
    evidence, _, _ = actual_original_results
    result = next(c for c in evidence['first']['candidates'] if c['candidate_id']=='other_reachable')['kinematic_validation']
    assert result['ik_valid'] and result['solver_invoked'] is True
    assert result['reason'] == 'endpoint_ik_valid_collision_unchecked'
    assert len(result['iterations']) == 6
    assert np.max(np.abs(np.array(result['q_solution'])-result['q_initial'])) > .01


def test_original_solver_rejects_unreachable_endpoint_and_does_not_trust_finite_fallback(actual_original_results):
    evidence, _, _ = actual_original_results
    row = next(row for row in evidence["first"]["candidates"] if row["candidate_id"] == "unreachable")
    assert not row["ik_valid"]
    assert not row["collision_valid"]
    assert row["kinematic_validation"]["reason"] != "endpoint_ik_valid_collision_unchecked"


def test_large_rotation_is_checked_geodesically_not_by_euler_wrap(actual_original_results):
    evidence, _, _ = actual_original_results
    row = next(row for row in evidence["first"]["candidates"] if row["candidate_id"] == "near_pi")
    result = row["kinematic_validation"]
    final = result["final"]
    if final is not None:
        for side in ("left", "right"):
            assert 0 <= final[side + "_orientation_error_rad"] <= np.pi
    if row["ik_valid"]:
        assert final["right_orientation_error_rad"] <= np.deg2rad(10)
        assert final["right_position_error_m"] <= .01
    else:
        assert result["reason"] != "endpoint_ik_valid_collision_unchecked"


def test_candidates_reset_warm_start_filter_and_do_not_depend_on_order(actual_original_results):
    evidence, _, _ = actual_original_results
    reverse = {row["candidate_id"]: row for row in evidence["reverse"]["candidates"]}
    for a in evidence["first"]["candidates"]:
        b = reverse[a["candidate_id"]]
        assert a["ik_valid"] == b["ik_valid"]
        np.testing.assert_allclose(a["kinematic_validation"]["q_solution"],
                                   b["kinematic_validation"]["q_solution"], atol=1e-9, rtol=0)


def test_malformed_candidate_clears_preexisting_unearned_permissions(actual_original_results):
    evidence, _, _ = actual_original_results
    row = evidence["malformed"]["candidates"][0]
    assert not row["ik_valid"] and not row["collision_valid"]
    assert row["kinematic_validation"]["q_solution"] is None


def test_no_collision_trajectory_freshness_or_robot_claim_from_offline_ik(actual_original_results):
    evidence, _, _ = actual_original_results
    result = evidence["first"]["kinematic_validation"]
    assert result["joint_state_acquisition_timestamp"] is None
    assert result["joint_state_clock_id"] is None
    assert not result["joint_state_freshness_validated"]
    assert result["original_sources_unchanged"]
    manifest = result["manifest"]
    for field in ("collision_validated", "trajectory_validated", "joint_state_freshness_validated",
                  "dds_imported", "robot_command_sent"):
        assert manifest[field] is False
    for field in ("source_sha256", "urdf_sha256", "filter_sha256", "validator_sha256"):
        assert len(manifest[field]) == 64
    assert all(row["collision_valid"] is False for row in evidence["first"]["candidates"])


def test_isolated_cli_round_trip_uses_actual_original_solver(actual_original_results):
    evidence, repository, python = actual_original_results
    request = deepcopy(evidence["request"])
    request["batch"]["candidates"] = [next(row for row in request["batch"]["candidates"]
                                           if row["candidate_id"] == "current")]
    result = isolated_validate(request, repository=repository, python_executable=python)
    assert result["candidates"][0]["ik_valid"]
    assert result["kinematic_validation"]["manifest"]["python"] == str(python)
