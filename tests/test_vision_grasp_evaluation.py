"""No physical success may be inferred from virtual hand/gripper commands."""
import json
import numpy as np
from scipy.spatial.transform import Rotation

from teleop_assist.core import AssistConfig,AssistSession
from teleop_assist.task_profiles import annotate_tasks
from teleop_assist.vision_grasp_evaluation import VisionGraspPullDriver


def target(now):
    transform=np.eye(4);transform[:3,3]=[.55,-.2,1.05]
    return annotate_tasks([{'id':'camera_handle','label':'handle','bbox_px':[10,10,40,40],
        'center_world':[.474,-.2,1.05],'normal_world':[-1,0,0],'confidence':.9,
        'T_world_object':transform.tolist(),'pose_valid':True,'pose_source':'foundationpose',
        'pose_timestamp':now,'cad_asset_id':'PullHandleGrasp','pose_position_sigma_m':.007,
        'pose_orientation_sigma_rad':.08}],now=now)[0]


def test_executed_script_without_measured_mechanism_opening_is_not_success():
    now=[10.]
    session=AssistSession(AssistConfig(require_action=True),clock=lambda:now[0])
    driver=VisionGraspPullDriver(session)
    state={'valid':True,'ee_position':[.404,-.2,1.],'ee_quaternion':[2**-.5,0,0,2**-.5]}
    for index in range(2500):
        sim=index*.02;now[0]=10.+sim
        session.update_frame([target(now[0])],index,now[0],{'width':100,'height':100})
        driver.update(state,sim)
        if driver.done:break
        reference=session.step(state,.02)
        state={**state,'ee_position':reference['ee_target_world'],
               'ee_quaternion':reference['ee_quat_xyzw'] or state['ee_quaternion'],
               'accepted_grip':reference['grip']}
        driver.observe(state,reference,sim)
    report=driver.report()
    assert driver.done
    assert report['script_completed'],report
    assert not report['passed']
    assert not report['checks']['physical_drawer_opening_evidence']['passed']
    assert not report['checks']['measured_hand_contact']['passed']
    assert report['checks']['physical_drawer_opening_evidence']['opening_m'] is None
    assert report['engagement_confirmation']['sensor_verified_grasp'] is False
    assert report['checks']['bounded_actual_pull']['maximum_measured_execution_displacement_m']<=.04+1e-6
    assert not session.input['clutch']


def test_target_loss_aborts_attempt_without_substituting_pose_or_grasp_confirmation():
    now=[10.]
    session=AssistSession(AssistConfig(require_action=True),clock=lambda:now[0])
    driver=VisionGraspPullDriver(session)
    state={'valid':True,'ee_position':[.404,-.2,1.],'ee_quaternion':[2**-.5,0,0,2**-.5]}
    for index in range(3):
        now[0]+=.02
        session.update_frame([target(now[0])],index,now[0],{'width':100,'height':100})
        driver.update(state,index*.02)
    assert driver.phase=='align_pregrasp'
    now[0]+=.02
    session.update_frame([],3,now[0],{'width':100,'height':100})
    driver.update(state,.06)
    assert driver.done and not driver.report()['passed']
    assert driver.grasp_confirmation is None
    assert not session.input['clutch']
    assert session.selection.id=='camera_handle'
    gate=driver.report()['abort_gate_snapshot']
    assert gate['sim_time']==.06 and gate['wall_time']==now[0]
    assert gate['selection']=={'mode':'lost','id':'camera_handle'}
    assert gate['target']=={'id':'camera_handle','present':False}
    assert gate['captured_before_disarm']


def test_current_registered_pose_after_a_fault_does_not_silently_rearm_driver():
    now=[10.]
    session=AssistSession(AssistConfig(require_action=True),clock=lambda:now[0])
    driver=VisionGraspPullDriver(session)
    state={'valid':True,'ee_position':[.404,-.2,1.],'ee_quaternion':[2**-.5,0,0,2**-.5]}
    for index in range(3):
        now[0]+=.02
        session.update_frame([target(now[0])],index,now[0],{'width':100,'height':100})
        driver.update(state,index*.02)
    now[0]+=.6
    assert not session.snapshot()['action']['valid']
    session.update_frame([target(now[0])],3,now[0],{'width':100,'height':100})
    snapshot=session.snapshot()
    assert snapshot['action']['valid'] and snapshot['action']['rearm_required']
    driver.update(state,.66)
    assert driver.done and not driver.script_completed
    assert driver.grasp_confirmation is None
    assert 'fresh-clutch' in driver.report()['checks']['driver']['reason']


def test_rejected_gripper_command_cannot_become_synthetic_grasp_confirmation():
    now=[10.]
    session=AssistSession(AssistConfig(require_action=True),clock=lambda:now[0])
    driver=VisionGraspPullDriver(session)
    state={'valid':True,'ee_position':[.404,-.2,1.],'ee_quaternion':[2**-.5,0,0,2**-.5],
           'accepted_grip':0.}
    for index in range(1800):
        sim=index*.02;now[0]=10.+sim
        session.update_frame([target(now[0])],index,now[0],{'width':100,'height':100})
        driver.update(state,sim)
        if driver.done:break
        reference=session.step(state,.02)
        state={**state,'ee_position':reference['ee_target_world'],
               'ee_quaternion':reference['ee_quat_xyzw'] or state['ee_quaternion']}
        driver.observe(state,reference,sim)
    assert driver.done and not driver.script_completed
    assert driver.grasp_confirmation is None
    assert session.action.phase=='approach'
    assert 'closure command was not accepted' in driver.report()['checks']['driver']['reason']


def test_grasp_approach_and_pose_mapping_from_recorded_robot_start_with_delayed_servo():
    now=[10.]
    session=AssistSession(AssistConfig(require_action=True),clock=lambda:now[0])
    driver=VisionGraspPullDriver(session)
    # Recorded settled G1 start; the goal remains an observed CAD transform,
    # and this CPU servo is only a controller/driver regression, not PhysX.
    state={'valid':True,'ee_position':[.3578,-.2397,.7141],
           'ee_quaternion':[-.1212,.2965,-.0292,.9469],'accepted_grip':0.}
    phases=set()
    for index in range(2500):
        sim=index*.02;now[0]=10.+sim
        observed=target(now[0])
        session.update_frame([observed],index,now[0],{'width':100,'height':100})
        driver.update(state,sim)
        phases.add(driver.phase)
        if driver.done:break
        reference=session.step(state,.02)
        measured=np.asarray(state['ee_position'])
        measured+=.06*(np.asarray(reference['ee_target_world'])-measured)
        orientation=Rotation.from_quat(state['ee_quaternion'])
        if reference['ee_quat_xyzw'] is not None:
            error=(Rotation.from_quat(reference['ee_quat_xyzw'])*orientation.inv()).as_rotvec()
            orientation=Rotation.from_rotvec(.06*error)*orientation
        state={**state,'ee_position':measured.tolist(),'ee_quaternion':orientation.as_quat().tolist(),
               'accepted_grip':reference['grip']}
        driver.observe(state,reference,sim)
    assert driver.script_completed,driver.report()
    assert {'align_pregrasp','approach_grasp','close_gripper','bounded_pull','open_gripper'}<=phases
    np.testing.assert_allclose(driver.grasp,[.494,-.2,1.])
    np.testing.assert_allclose(driver.pregrasp,[.404,-.2,1.])
    assert driver.phase_results['align_pregrasp']['measured_orientation_error_rad']<.05
    assert driver.phase_results['close_gripper']['accepted_grip']==.85
    assert not driver.report()['passed']  # No contact/opening evidence fabricated.


def test_measured_pull_drift_counts_even_when_controller_is_holding():
    session=AssistSession(AssistConfig(require_action=True))
    driver=VisionGraspPullDriver(session)
    driver.phase='bounded_pull';driver.phase_start=0.
    driver.pull_origin=np.array([.5,-.2,1.]);driver.pull_axis=np.array([-1.,0.,0.])
    session.control={'reason':'operator_stationary'}
    driver.observe({'valid':True,'ee_position':[.45,-.2,1.]},{'hold':True},1.)
    driver.script_completed=True;driver.done=True
    assert driver.max_execution_progress>.049
    assert not driver.report()['checks']['bounded_actual_pull']['passed']


def test_abort_snapshot_preserves_exact_rejected_cad_gate_before_cleanup():
    now=[10.]
    session=AssistSession(AssistConfig(require_action=True),clock=lambda:now[0])
    driver=VisionGraspPullDriver(session)
    state={'valid':True,'ee_position':[.404,-.2,1.],'ee_quaternion':[2**-.5,0,0,2**-.5],
           'contact_active':False,'collision_stop':False,'contact_body':'right_hand_index_0_link',
           'contact_force_max':1.3,'accepted_grip':.2}
    for index in range(3):
        now[0]+=.02
        session.update_frame([target(now[0])],index,now[0],{'width':100,'height':100})
        driver.update(state,index*.02)
    now[0]+=.02
    rejected=target(now[0])
    rejected.update(pose_position_sigma_m=.012,pose_orientation_sigma_rad=.18)
    rejected=annotate_tasks([rejected],now=now[0])[0]
    session.update_frame([rejected],115,now[0],{'width':100,'height':100})
    driver.update(state,11.455)
    gate=driver.report()['abort_gate_snapshot']
    assert driver.done
    assert gate['sim_time']==11.455 and gate['wall_time']==now[0] and gate['frame_id']==115
    assert gate['driver_phase']=='align_pregrasp'
    assert gate['action']['id']=='pull' and gate['action']['phase']=='approach'
    assert not gate['action']['valid'] and gate['action']['rearm_required']
    assert gate['action']['reason']=='action_calibration_lost'
    assert gate['calibration']['locked_id']!=gate['calibration']['current_id']
    assert gate['calibration']['source']=='configured_fixture'
    assert not gate['calibration']['execution_ready']
    assert gate['target']['pose_valid'] is True
    assert gate['target']['cad_grasp_status']['status']=='rejected'
    assert 'Tool-point uncertainty' in gate['target']['cad_grasp_status']['reason']
    assert gate['target']['pose_timestamp']==now[0]
    assert gate['target']['pose_position_sigma_m']==.012
    assert gate['target']['pose_orientation_sigma_rad']==.18
    assert gate['robot']['valid'] and gate['robot']['contact_force_max']==1.3
    json.dumps(gate,allow_nan=False)
    # Later cleanup, target publication or report consumers cannot rewrite
    # the exact failure evidence that preceded disarming.
    rejected['cad_grasp_status']['reason']='later mutation'
    state['ee_position'][0]=.6
    gate['action']['valid']=True
    saved=driver.report()['abort_gate_snapshot']
    assert 'Tool-point uncertainty' in saved['target']['cad_grasp_status']['reason']
    assert saved['robot']['ee_position'][0]==.404
    assert not saved['action']['valid']


def evaluation_only_pull_report(additional_pull,contact_during_pull,missing_start=False):
    session=AssistSession(AssistConfig(require_action=True))
    driver=VisionGraspPullDriver(session)
    sim=0.

    def observation(phase,position,hand_contact=False,missing=False):
        nonlocal sim
        sim+=.005;driver.phase=phase
        state={'valid':True,'ee_position':[.5,-.2,1.],
               'contact_body':'right_hand_index_0_link' if hand_contact else None,
               'contact_force_max':2. if hand_contact else 0.}
        if not missing:state['fixture_joint_states']={'PullHandle':{'position':position}}
        driver.observe(state,{'hold':False},sim)

    observation('align_pregrasp',0.)
    for _ in range(20):observation('close_gripper',-.035,True)
    observation('bounded_pull',-.035,contact_during_pull,missing_start)
    for _ in range(20):observation('bounded_pull',-.035-additional_pull,contact_during_pull)
    observation('open_gripper',-.035-additional_pull)
    driver.script_completed=True;driver.done=True
    return driver.report()


def test_closure_opening_with_free_hand_pull_cannot_pass_as_a_successful_pull():
    report=evaluation_only_pull_report(0.,False)
    assert report['checks']['physical_drawer_opening_evidence']['opening_m']>.03
    assert report['checks']['physical_drawer_opening_evidence']['incremental_opening_during_pull_m']==0.
    assert report['checks']['measured_hand_contact']['passed']  # Contact during closure alone.
    assert not report['checks']['physical_drawer_opening_evidence']['passed']
    assert not report['checks']['measured_hand_contact_during_pull']['passed']
    assert not report['passed']


def test_incremental_pull_and_legacy_force_alone_no_longer_certify_v2_success():
    report=evaluation_only_pull_report(.032,True)
    assert report['checks']['physical_drawer_opening_evidence']['incremental_opening_during_pull_m']>.025
    assert report['checks']['measured_hand_contact_during_pull']['measured_samples']==21
    assert report['checks']['measured_hand_contact']['measured_samples']==41
    assert report['pull_start_observation_evaluation_only']['fixture_joint_states']['PullHandle']['position']==-.035
    assert not report['passed']  # No partner telemetry or measured hand coupling.
    assert not report['checks']['distributed_target_hand_contacts']['passed']
    assert report['checks']['measured_hand_contact_during_pull']['required_for_success'] is False


def test_missing_first_pull_measurement_cannot_be_replaced_by_later_telemetry():
    report=evaluation_only_pull_report(.032,True,missing_start=True)
    assert report['checks']['physical_drawer_opening_evidence']['opening_m']>.03
    assert report['checks']['measured_hand_contact_during_pull']['passed']
    assert report['pull_start_observation_evaluation_only']['fixture_joint_states'] is None
    assert report['checks']['physical_drawer_opening_evidence']['incremental_opening_during_pull_m'] is None
    assert not report['passed']


def prepare_grasp_approach_driver():
    now=[10.]
    session=AssistSession(AssistConfig(require_action=True),clock=lambda:now[0])
    driver=VisionGraspPullDriver(session)
    state={'valid':True,'ee_position':[.404,-.2,1.],'ee_quaternion':[2**-.5,0,0,2**-.5],
           'accepted_grip':0.}
    for index in range(3):
        now[0]+=.02
        session.update_frame([target(now[0])],index,now[0],{'width':100,'height':100})
        driver.update(state,index*.02)
    # Restore a driver at the already aligned approach entry; no backend or
    # oracle object pose participates in the following production gate checks.
    driver._set_phase('approach_grasp',0.,state)
    return now,session,driver,state


def test_close_requires_two_point_five_mm_to_current_sensor_grasp_not_frozen_initial_goal():
    now,session,driver,state=prepare_grasp_approach_driver()
    observed=target(now[0])
    transform=np.asarray(observed['T_world_object']);transform[0,3]+=.01
    observed['T_world_object']=transform.tolist();observed['center_world'][0]+=.01
    observed=annotate_tasks([observed],now=now[0])[0]
    session.update_frame([observed],3,now[0],{'width':100,'height':100})
    state['ee_position']=[.488,-.2,1.]
    # Old fixed goal .494 is only 6 mm away and used to start closure here.
    # Current observed goal .504 remains 16 mm away: continue the approach.
    driver.update(state,6.)
    assert driver.phase=='approach_grasp'
    state['ee_position']=[.4995,-.2,1.]
    driver.update(state,6.25)
    # A 4.5 mm miss met the prior gate, but must now keep approaching.
    assert driver.phase=='approach_grasp'
    assert driver.grasp_confirmation is None
    assert driver.grip==0.
    state['ee_position']=[.5016,-.2,1.]
    driver.update(state,6.5)
    assert driver.phase=='close_gripper'
    phase=driver.phase_results['approach_grasp']
    assert phase['measured_position_error_m']<.0025
    np.testing.assert_allclose(phase['goal_world'],[.504,-.2,1.])
    assert phase['position_tracking_tolerance_m']==.0025
    assert phase['goal_source']=='current_accepted_CAD_grasp'


def test_explicit_refinement_is_four_mm_per_second_capped_at_four_mm_and_times_out():
    now,session,driver,state=prepare_grasp_approach_driver()
    commands=[]
    driver._input=lambda desired=None,clutch=False,quaternion=None:commands.append((desired,clutch))
    samples=[6.,6.25,6.5,6.75,7.,8.,8.5]
    for sim in samples:
        driver.update(state,sim)
        assert driver.phase=='approach_grasp' and not driver.done
    positions=np.asarray([row[0] for row in commands])
    progress=-(positions-driver.grasp)@driver.normal
    np.testing.assert_allclose(progress,[0.,.001,.002,.003,.004,.004,.004],atol=1e-12)
    assert all(row[1] for row in commands)
    driver.update(state,9.01)
    assert driver.done and not driver.script_completed
    assert 'within bounded time' in driver.report()['checks']['driver']['reason']
    assert driver.grasp_confirmation is None
