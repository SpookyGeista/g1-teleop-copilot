"""CPU real-geometry contract tests; no robot actuation or physical validation."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from teleop_assist.copilot.arm_collision import CollisionConfig


@pytest.mark.parametrize("values", [{"clearance_m": -.1}, {"max_joint_step_rad": 0},
                                    {"max_joint_step_rad": .2}, {"max_samples": True},
                                    {"max_samples": 2049}, {"max_support_penetration_m": .01},
                                    {"max_snapshot_skew_s": 1}, {"clearance_m": np.nan}])
def test_invalid_collision_config_rejected(values):
    with pytest.raises(ValueError):
        CollisionConfig(**values)


@pytest.fixture(scope="module")
def actual_collision_results(tmp_path_factory):
    repository = Path(os.environ.get("COPILOT_ORIGINAL_XR_REPOSITORY", "/home/loya2/xr_teleoperate"))
    python = Path(os.environ.get("COPILOT_ORIGINAL_IK_PYTHON", "/home/loya2/miniconda3/envs/to/bin/python"))
    if not repository.is_dir() or not python.is_file():
        pytest.skip("Original G1 geometry and its CPU Pinocchio environment are not installed")
    out = tmp_path_factory.mktemp("original_arm_collision") / "tests.json"
    code = r'''
import json,sys,copy
import numpy as np
from teleop_assist.copilot.arm_collision import OriginalArmCollision
from teleop_assist.copilot.kinematics import ARM_JOINT_NAMES
v=OriginalArmCollision(sys.argv[1]);q=np.zeros(43);base=np.eye(4);base[2,3]=3
v.pin.framesForwardKinematics(v.model,v.data,q)
observed={f.name:(base@v.data.oMf[i].homogeneous).tolist() for i,f in enumerate(v.model.frames)}
tcp=np.eye(4);tcp[0,3]=.05
left=v.data.oMf[v.model.getFrameId('left_wrist_yaw_link')].homogeneous@tcp
right=v.data.oMf[v.model.getFrameId('right_wrist_yaw_link')].homogeneous@tcp
q14=q[v.arm_indices].tolist()
state={'joint_positions':{name:0. for name in v.names},'base_in_world':base.tolist(),
       'observed_body_world_poses':observed,'complete':True,'source':'synthetic_FK_fixture',
       'acquisition_timestamp':1.,'clock_id':'test'}
scene={'geometry_complete':True,'static':True,'source':'explicit_empty_above_floor_fixture',
       'acquisition_timestamp':1.,'clock_id':'test','obstacles':[{'id':'floor','type':'plane','normal':[0,0,1],'offset':0.}]}
candidate={'candidate_id':'stationary','ik_valid':True,'collision_valid':True,'base_ee':right.tolist(),
           'kinematic_validation':{'ik_valid':True,'q_solution':q14,'q_initial':q14,'left_target_base_ee':left.tolist()}}
request={'batch':{'candidates':[candidate]},'robot_state':state,'scene':scene}
cases={}
def run(name,r,endpoint_only=False):
 result=v.validate_request(r,endpoint_only=endpoint_only)
 cases[name]={'valid':result['candidates'][0]['collision_valid'],
              'reason':result['arm_collision_validation']['reason'],
              'scope':result['arm_collision_validation']['scope'],
              'candidate':result['candidates'][0]['arm_collision_validation'],
              'fk':result['arm_collision_validation'].get('urdf_vs_observed_frames')}
run('clear',request)
for name,edit in [
 ('missing_finger',lambda r:r['robot_state']['joint_positions'].pop('right_hand_thumb_2_joint')),
 ('missing_scene',lambda r:r.pop('scene')),
 ('scene_incomplete',lambda r:r['scene'].update(geometry_complete=False)),
 ('clock_mismatch',lambda r:r['scene'].update(clock_id='other')),
 ('time_skew',lambda r:r['scene'].update(acquisition_timestamp=2.)),
 ('body_mismatch',lambda r:r['robot_state']['observed_body_world_poses']['right_wrist_yaw_link'][0].__setitem__(3,99.)),
 ('camera_body_missing',lambda r:r['robot_state']['observed_body_world_poses'].pop('right_hand_camera_base_link')),
 ('camera_attachment_mismatch',lambda r:r['robot_state']['observed_body_world_poses']['right_hand_camera_base_link'][0].__setitem__(3,99.)),
 ('input_nan',lambda r:r['robot_state']['joint_positions'].update(right_hand_thumb_2_joint=float('nan'))),
 ('snapshot_joint_limit',lambda r:r['robot_state']['joint_positions'].update(right_shoulder_pitch_joint=99.)),
 ('candidate_joint_limit',lambda r:r['batch']['candidates'][0]['kinematic_validation'].update(q_solution=[99.]*14)),
 ('q_initial_mismatch',lambda r:r['batch']['candidates'][0]['kinematic_validation'].update(q_initial=[.2]*14)),
 ('endpoint_not_matching_ik_target',lambda r:r['batch']['candidates'][0]['base_ee'][0].__setitem__(3,99.)),
 ('unvalidated_ik',lambda r:r['batch']['candidates'][0].update(ik_valid=False)),
 ('illegal_support_exclusion',lambda r:r['scene'].update(allowed_static_contacts=[{'link':'right_wrist_yaw_link','obstacle':'floor','max_penetration_m':.002}]))]:
 r=copy.deepcopy(request);edit(r);run(name,r);run(name+'_endpoint_only',r,True)
# Deliberately place a solid box around the right wrist. Its volume must not be
# accepted merely because a triangle mesh shell has no surface intersection.
r=copy.deepcopy(request);obstacle=np.eye(4);obstacle[:3,3]=(base@right)[:3,3]
r['scene']['obstacles'].append({'id':'blocking_box','type':'box','size_xyz':[.2,.2,.2],'world_pose':obstacle.tolist()})
run('blocking_volume',r)
# Exercise geometry absent from the original 52-shape URDF: a small obstacle
# around an actual extreme point of the right wrist-camera CAD hull.
proxy=json.load(open(v.camera_proxy_path));camera=proxy['cameras'][1]
vertices=np.array(camera['convex_vertices_in_link']);point=vertices[np.argmax(vertices[:,2])]
v.pin.framesForwardKinematics(v.model,v.data,q)
world_camera=base@v.data.oMf[v.model.getFrameId(camera['link'])].homogeneous
T=np.eye(4);T[:3,3]=(world_camera@np.r_[point,1])[:3]
r=copy.deepcopy(request);r['scene']['obstacles'].append({'id':'camera_block','type':'sphere','radius':.002,'world_pose':T.tolist()})
run('camera_only_obstacle',r)
# The current configuration can be clear while a large swept path is not
# certifiable with the configured bounded sample budget.
v.config=type(v.config)(max_joint_step_rad=.0001,max_samples=2)
obstacles,allowed=v._obstacles(scene);final=q.copy();final[v.indices['right_shoulder_pitch_joint']]=.1
cases['bounded_budget']=v._check_path(q,final,base,obstacles,allowed)
# A genuinely changed endpoint can be checked without pretending that the
# current-to-goal trajectory fits a deliberately tiny path sample budget.
v.pin.framesForwardKinematics(v.model,v.data,final)
r=copy.deepcopy(request);c=r['batch']['candidates'][0]
c['kinematic_validation']['q_solution']=final[v.arm_indices].tolist()
c['kinematic_validation']['left_target_base_ee']=(v.data.oMf[v.model.getFrameId('left_wrist_yaw_link')].homogeneous@tcp).tolist()
c['base_ee']=(v.data.oMf[v.model.getFrameId('right_wrist_yaw_link')].homogeneous@tcp).tolist()
run('changed_default_path',r)
run('changed_endpoint_only',r,True)
json.dump({'cases':cases,'manifest':v.manifest},open(sys.argv[2],'w'),allow_nan=False)
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run([str(python), "-c", code, str(repository), str(out)],
                            cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(out.read_text())


def test_real_original_geometry_clear_stationary_pose_passes(actual_collision_results):
    result = actual_collision_results["cases"]["clear"]
    assert result["valid"]
    assert result["candidate"]["endpoint_precheck"]["collision_valid"]
    assert result["candidate"]["conservative_path_clearance_lower_bound_m"] > .001
    assert result["fk"]["valid"]


@pytest.mark.parametrize("name", ["missing_finger", "missing_scene", "scene_incomplete", "clock_mismatch",
                                  "time_skew", "body_mismatch", "camera_body_missing", "camera_attachment_mismatch",
                                  "input_nan", "snapshot_joint_limit", "candidate_joint_limit", "q_initial_mismatch",
                                  "endpoint_not_matching_ik_target", "unvalidated_ik", "illegal_support_exclusion",
                                  "blocking_volume", "camera_only_obstacle"])
def test_incomplete_invalid_or_colliding_input_clears_permissions(actual_collision_results, name):
    assert not actual_collision_results["cases"][name]["valid"]


def test_sample_budget_never_silently_coarsens_path(actual_collision_results):
    result = actual_collision_results["cases"]["bounded_budget"]
    assert not result["collision_valid"]
    assert result["reason"] == "path_sample_budget_exceeded"
    assert result["samples_checked"] == 0


def test_endpoint_only_keeps_scope_and_never_claims_approach(actual_collision_results):
    cases = actual_collision_results['cases']
    assert not cases['changed_default_path']['valid']
    assert cases['changed_default_path']['candidate']['reason'] == 'path_sample_budget_exceeded'
    result = cases['changed_endpoint_only']
    assert result['valid']
    assert result['scope'] == 'sim_candidate_endpoint_only'
    assert result['candidate']['path_validated'] is False
    assert result['candidate']['live_execution_authorized'] is False
    assert result['candidate']['manual_fingers_held_at_observed_values'] is True


def test_endpoint_only_retains_complete_state_limits_and_fk_gates(actual_collision_results):
    cases = actual_collision_results['cases']
    for name, result in cases.items():
        if name.endswith('_endpoint_only') and name != 'changed_endpoint_only':
            assert not result['valid'], name


def test_geometry_manifest_is_explicit_about_sources_exclusions_and_scope(actual_collision_results):
    manifest = actual_collision_results["manifest"]
    assert manifest["geometry_count"] == 54
    assert manifest["self_pairs_checked"] == 1344
    assert manifest["self_pairs_checked"] + len(manifest["excluded_adjacent_or_same_rigid_body_pairs"]) == 54 * 53 // 2
    assert manifest["wrist_camera_collision_coverage_verified"] is True
    assert manifest["wrist_camera_collision_source"]["source_sha256_verified"] is True
    assert manifest["entire_physical_robot_geometry_validated"] is False
    assert "d435_link" in manifest["known_geometry_coverage_exclusions"]
    assert len(manifest["mesh_sha256"]) > 20
    assert manifest["excluded_adjacent_or_same_rigid_body_pairs"]
    assert manifest["robot_command_sent"] is False
    assert manifest["live_execution_authorized"] is False


def test_actual_camera_mesh_detects_previously_unmodelled_obstacle(actual_collision_results):
    result = actual_collision_results["cases"]["camera_only_obstacle"]
    assert not result["valid"]
    assert "Current robot snapshot already fails" in result["reason"]
