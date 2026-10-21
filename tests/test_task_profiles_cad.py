"""CAD grasp transforms and fail-closed input handling; no learned model or truth."""
from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.actions import calibrated_actions
from teleop_assist.core import AssistConfig, AssistSession
from teleop_assist.stack_evaluation import StackAcceptanceDriver
from teleop_assist.task_profiles import DEFAULT_CAD_CATALOG, annotate_tasks


def measured(rotation=None, translation=(.55,-.2,1.05)):
    matrix=np.eye(4)
    matrix[:3,:3]=np.eye(3) if rotation is None else rotation
    matrix[:3,3]=translation
    return {"id":"observed_handle","label":"handle","bbox_px":[10,10,30,30],
            "center_world":(matrix[:3,:3]@np.array([-.076,0,0])+matrix[:3,3]).tolist(),
            "normal_world":(-matrix[:3,0]).tolist(),"confidence":.8,
            "T_world_object":matrix.tolist(),"pose_valid":True,"pose_source":"foundationpose",
            "pose_timestamp":10.,"cad_asset_id":"PullHandleGrasp",
            "pose_position_sigma_m":.007,"pose_orientation_sigma_rad":.08}


def action(result, name="pull"):
    return next(row for row in calibrated_actions(result) if row["id"]==name)


def test_rotated_pose_transforms_tool_quaternion_and_mechanical_axis():
    rotation=Rotation.from_euler("zyx",[.5,-.2,.1]).as_matrix()
    target=measured(rotation)
    original=deepcopy(target)
    result=annotate_tasks([target],now=10.1)[0]
    pull=action(result)
    np.testing.assert_allclose(pull["grasp_position_world"],rotation@[-.056,0,-.05]+np.array([.55,-.2,1.05]))
    np.testing.assert_allclose(pull["motion_axis_world"],rotation@[-1,0,0])
    np.testing.assert_allclose(pull["approach_axis_world"],rotation@[-1,0,0])
    expected=rotation@Rotation.from_euler("x",np.pi/2).as_matrix()
    np.testing.assert_allclose(Rotation.from_quat(pull["grasp_quaternion_xyzw"]).as_matrix(),expected,atol=1e-12)
    np.testing.assert_allclose(action(result,"push")["motion_axis_world"],rotation@[1,0,0])
    assert result["task_calibration"]["source"]=="cad_registration"
    assert pull["has_calibrated_grasp_template"] and pull["execution_ready"]
    assert pull["validated_pull_distance_m"]==.04 and pull["recommended_grip_scalar"]==.85
    assert result["cad_grasp_status"]["engagement_verified"] is False
    assert target==original


def test_model_tracking_source_preserves_same_calibration_identity():
    first=annotate_tasks([measured()],now=10.1)[0]
    target=measured();target["pose_source"]="foundationpose_rgbd_tracking";target["pose_timestamp"]=10.1
    target.update(pose_registration_timestamp=10.,pose_tracking_method='measured_point_translation_for_declared_prismatic_fixture')
    second=annotate_tasks([target],now=10.2)[0]
    assert first["task_calibration"]["id"]==second["task_calibration"]["id"]
    assert second["cad_grasp_status"]["status"]=="accepted"


def test_refreshing_tracking_timestamp_cannot_hide_old_registration():
    target=measured()
    target.update(pose_source="foundationpose_rgbd_tracking",pose_timestamp=10.1,
                  pose_registration_timestamp=9.,pose_tracking_method='measured_point_translation_for_declared_prismatic_fixture')
    result=annotate_tasks([target],now=10.1)[0]
    assert result['cad_grasp_status']['status']=='rejected'
    assert 'registration is stale' in result['cad_grasp_status']['reason']


def test_surface_execute_rejection_does_not_mutate_action_or_rearm_state():
    clock=lambda:10.1
    session=AssistSession(AssistConfig(require_action=True),clock=clock)
    target=measured();target.pop('T_world_object')
    result=annotate_tasks([target],now=10.1)[0]
    session.update_frame([result],1,clock(),{'width':100,'height':100})
    session.command({'type':'select','id':target['id']});session.command({'type':'confirm'})
    session.command({'type':'action_select','action_id':'pull'});session.command({'type':'action_confirm'})
    before=session.action.snapshot(True,session.rearm_required)
    with pytest.raises(ValueError,match='surface approach only'):
        session.command({'type':'action_phase','phase':'execute','engagement_confirmed':True})
    assert session.action.snapshot(True,session.rearm_required)==before


def test_cad_pose_expires_even_if_no_new_camera_frame_arrives():
    current=[10.1]
    session=AssistSession(AssistConfig(require_action=True),clock=lambda:current[0])
    target=annotate_tasks([measured()],now=current[0])[0]
    session.update_frame([target],1,current[0],{'width':100,'height':100})
    session.command({'type':'select','id':target['id']});session.command({'type':'confirm'})
    session.command({'type':'action_select','action_id':'pull'});session.command({'type':'action_confirm'})
    current[0]=10.51
    with pytest.raises(ValueError):
        session.command({'type':'action_phase','phase':'execute','engagement_confirmed':True})
    assert session.action.reason=='cad_pose_stale'
    assert session.action.phase=='approach'


@pytest.mark.parametrize("field,value",[
    ("pose_valid",False),("pose_valid",1),("pose_source","simulator_truth"),
    ("cad_asset_id","different_handle"),("pose_timestamp",9.4),("pose_timestamp",10.2),
    ("pose_timestamp",float("nan")),("pose_position_sigma_m",.026),
    ("pose_orientation_sigma_rad",.36),("pose_orientation_sigma_rad",float("nan")),
    ("pose_position_sigma_m",-.01),("pose_orientation_sigma_rad",None)])
def test_bad_pose_metadata_cannot_become_calibrated_grasp(field,value):
    target=measured();target[field]=value
    result=annotate_tasks([target],now=10.1)[0]
    assert result["cad_grasp_status"]["status"]=="rejected"
    assert result["task_calibration"]["source"]=="configured_fixture"
    pull=action(result)
    assert not pull["execution_ready"] and not pull["has_calibrated_grasp_template"]
    assert "grasp_quaternion_xyzw" not in pull
    assert pull["grasp_position_world"]==target["center_world"]


@pytest.mark.parametrize("fault",["reflection","scale","last_row","nan","shape","wrong_world_origin"])
def test_corrupt_transform_or_inconsistent_world_anchor_is_rejected(fault):
    target=measured();matrix=np.array(target["T_world_object"])
    if fault=="reflection":matrix[0,0]=-1
    elif fault=="scale":matrix[0,0]=2
    elif fault=="last_row":matrix[3,0]=1
    elif fault=="nan":matrix[0,3]=float("nan")
    elif fault=="shape":matrix=matrix[:3]
    else:matrix[0,3]+=1.
    target["T_world_object"]=matrix.tolist()
    result=annotate_tasks([target],now=10.1)[0]
    assert result["cad_grasp_status"]["status"]=="rejected"
    assert not action(result)["has_calibrated_grasp_template"]


def test_orientation_lever_arm_contributes_to_tool_position_uncertainty():
    target=measured();target.update(pose_position_sigma_m=.02,pose_orientation_sigma_rad=.2)
    result=annotate_tasks([target],now=10.1)[0]
    assert result["cad_grasp_status"]["status"]=="rejected"
    assert "Tool-point uncertainty" in result["cad_grasp_status"]["reason"]
    target.update(pose_position_sigma_m=.007,pose_orientation_sigma_rad=.08)
    result=annotate_tasks([target],now=10.1)[0]
    expected=.007+np.linalg.norm([-.056,0,-.05])*.08
    assert result["position_sigma_m"]>=expected
    assert result["cad_grasp_status"]["tool_position_sigma_m"]==pytest.approx(expected)


def test_surface_fallback_keeps_approach_labels_but_never_claims_tcp_pose():
    target=measured()
    for key in list(target):
        if key.startswith("pose_") or key in ("T_world_object","cad_asset_id"):target.pop(key)
    result=annotate_tasks([target],now=10.1)[0]
    assert {entry["id"] for entry in calibrated_actions(result)}=={"grasp","pull","push"}
    assert result["cad_grasp_status"]["status"]=="not_available"
    assert result["task_calibration"]["execution_ready"] is False
    assert all(not entry["has_calibrated_grasp_template"] for entry in calibrated_actions(result))
    assert all(entry["grasp_position_world"]==target["center_world"] for entry in calibrated_actions(result))


def test_missing_explicit_pose_uncertainty_is_not_replaced_by_surface_floor():
    target=measured();target.pop("pose_position_sigma_m")
    result=annotate_tasks([target],now=10.1)[0]
    assert result["cad_grasp_status"]["status"]=="rejected"
    assert "must be supplied" in result["cad_grasp_status"]["reason"]


@pytest.mark.parametrize("fault",["units","tool_point","axis","hash","template_missing"])
def test_invalid_catalog_cannot_authorize_physical_grasp(tmp_path,fault):
    catalog=json.loads(DEFAULT_CAD_CATALOG.read_text());asset=catalog["objects"][0]
    if fault=="units":catalog["units"]="millimeters"
    elif fault=="tool_point":asset["grasp_templates"][0]["tool_point_object_m"][2]=0.
    elif fault=="axis":asset["mechanical_joint"]["axis_object"]=[0,1,0]
    elif fault=="hash":asset["cad_sha256"]="incorrect"
    else:asset["grasp_templates"]=[]
    path=tmp_path/"catalog.json";path.write_text(json.dumps(catalog))
    result=annotate_tasks([measured()],now=10.1,catalog_path=path)[0]
    assert result["cad_grasp_status"]["status"]=="rejected"


def test_approach_only_profile_does_not_offer_execute_even_with_cad_pose():
    result=annotate_tasks([measured()],profile="approach_only",now=10.1)[0]
    assert [entry["id"] for entry in calibrated_actions(result)]==["grasp"]
    assert not action(result,"grasp")["execution_ready"]


def test_full_stack_driver_waits_for_registered_pose_before_lock():
    clock=lambda:10.1
    session=AssistSession(AssistConfig(require_action=True),clock=clock)
    driver=StackAcceptanceDriver(session,require_cad_pose=True)
    state={"valid":True,"ee_position":[.4,-.2,.95]}
    target=measured();target.pop("T_world_object")
    fallback=annotate_tasks([target],now=10.1)[0]
    for frame in range(3):
        session.update_frame([fallback],frame,clock(),{"width":100,"height":100})
        driver.update(state,.02*frame)
    assert driver.phase=="await_perception"
    good=annotate_tasks([measured()],now=10.1)[0]
    for frame in range(3,6):
        session.update_frame([good],frame,clock(),{"width":100,"height":100})
        driver.update(state,.02*frame)
    assert driver.phase=="baseline_approach"
    assert driver.report()["requested_task"]["measured_cad_pose_required"]


def test_cad_acceptance_approach_uses_tool_axis_instead_of_edge_normal():
    current=[10.1]
    session=AssistSession(AssistConfig(require_action=True),clock=lambda:current[0])
    driver=StackAcceptanceDriver(session,require_cad_pose=True)
    target=measured();target['normal_world']=[0,0,1.]
    annotated=annotate_tasks([target],now=current[0])[0]
    state={'valid':True,'ee_position':[.4,-.2,.95]}
    for frame in range(3):
        session.update_frame([annotated],frame,current[0],{'width':100,'height':100})
        driver.update(state,frame*.02)
    pull=action(annotated)
    expected=np.array(pull['grasp_position_world'])+np.array(pull['approach_axis_world'])*.10
    np.testing.assert_allclose(driver.goal,expected)
    np.testing.assert_allclose(driver.normal,pull['approach_axis_world'])
    assert not np.allclose(driver.normal,target['normal_world'])
