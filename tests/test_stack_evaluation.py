"""The synthetic stack driver itself must exercise the production gate."""
import numpy as np

from teleop_assist.core import AssistConfig, AssistSession
from teleop_assist.stack_evaluation import StackAcceptanceDriver


class Clock:
    def __init__(self): self.time=1.
    def __call__(self): return self.time


def fixture_target():
    return {"id":"observed_handle","label":"handle","bbox_px":[30,30,60,70],
            "center_world":[.5,-.2,1.08],"normal_world":[-1.,0,0],"confidence":.8,
            "task_calibration":{"id":"declared_drawer_v1","source":"configured_fixture","actions":[
                {"id":"pull","label":"Ziehen","kind":"linear","grasp_position_world":[.5,-.2,1.08],
                 "approach_axis_world":[-1.,0,0],"motion_axis_world":[-1.,0,0]}]}}


def test_driver_completes_through_real_session_api_with_ideal_measured_plant():
    clock=Clock()
    session=AssistSession(AssistConfig(require_action=True),clock=clock)
    driver=StackAcceptanceDriver(session)
    state={"valid":True,"ee_position":[.4,-.25,1.08],"ee_quaternion":[0,0,0,1]}
    for index in range(3000):
        sim_time=index*.02;clock.time=1.+sim_time
        session.update_frame(driver.filter_targets([fixture_target()]),index,clock(),{"width":100,"height":100})
        driver.update(state,sim_time)
        if driver.done:break
        reference=session.step(state,.02)
        state={**state,"ee_position":reference["ee_target_world"]}
        driver.observe(state,reference,sim_time)
    report=driver.report()
    assert driver.done
    assert report["passed"],report["checks"]
    assert report["checks"]["lost_target_hold"]["withheld_detection_frames"]>0
    assert report["target"]["action_id"]=="pull"
    assert report["target"]["fresh_observations_before_lock"]==3
    assert report["checks"]["baseline_approach"]["mean_lateral_error_m"]>.02
    assert report["checks"]["assisted_approach"]["mean_lateral_error_m"]<.01
    assert report["mechanism_telemetry_observation_only"] is None
    assert session.input["clutch"] is False
    assert session.config.assist_enabled is False
    np.testing.assert_allclose(report["goal_approach_world"],[.4,-.2,1.08])


def test_driver_timeout_is_fail_closed_and_does_not_fabricate_perception():
    clock=Clock();session=AssistSession(AssistConfig(require_action=True),clock=clock)
    driver=StackAcceptanceDriver(session)
    state={"valid":True,"ee_position":[.4,-.25,1.08]}
    driver.update(state,0)
    clock.time+=181.
    driver.update(state,1.)
    report=driver.report()
    assert driver.done and not report["passed"]
    assert report["target"] is None
    assert not session.input["clutch"]
    assert not session.config.assist_enabled


def test_driver_rejects_legacy_mode_as_production_validation():
    session=AssistSession()
    driver=StackAcceptanceDriver(session)
    driver.update({"valid":True,"ee_position":[.4,-.25,1.08]},0.)
    assert driver.done
    assert "action gate" in driver.report()["checks"]["driver"]["reason"]


def test_driver_never_substitutes_screw_or_generic_grasp_for_requested_handle():
    clock=Clock();session=AssistSession(AssistConfig(require_action=True),clock=clock)
    driver=StackAcceptanceDriver(session)
    state={"valid":True,"ee_position":[.4,-.25,1.08]}
    screw=fixture_target();screw.update(id="visible_screw",label="screw")
    screw["task_calibration"]["actions"][0].update(id="grasp",kind="grasp")
    for index in range(5):
        clock.time+=.03
        session.update_frame([screw],index,clock(),{"width":100,"height":100})
        driver.update(state,index*.03)
    assert driver.phase=="await_perception"
    assert driver.target_id is None
    assert not session.input["clutch"]
    assert not driver.report()["requested_task"]["fallback_allowed"]


def test_driver_requires_three_fresh_distinct_frames_with_same_identity():
    clock=Clock();session=AssistSession(AssistConfig(require_action=True),clock=clock)
    driver=StackAcceptanceDriver(session)
    state={"valid":True,"ee_position":[.4,-.25,1.08]}
    session.update_frame([fixture_target()],0,clock()-.3,{"width":100,"height":100})
    driver.update(state,0.)
    assert driver._candidate_frames==0
    for frame in (1,2):
        clock.time+=.03
        session.update_frame([fixture_target()],frame,clock(),{"width":100,"height":100})
        driver.update(state,frame*.03)
        assert driver.phase=="await_perception"
        for _ in range(8):driver.update(state,frame*.03)
        assert driver._candidate_frames==frame
    missing=fixture_target();missing["id"]="replacement_handle"
    clock.time+=.03
    session.update_frame([missing],3,clock(),{"width":100,"height":100});driver.update(state,.09)
    assert driver._candidate_frames==1
    for frame in (4,5):
        clock.time+=.03
        session.update_frame([missing],frame,clock(),{"width":100,"height":100});driver.update(state,frame*.03)
    assert driver.phase=="baseline_approach"
    assert driver.target_id=="replacement_handle"


def test_target_loss_aborts_comparison_without_rearming_or_substituting():
    clock=Clock();session=AssistSession(AssistConfig(require_action=True),clock=clock)
    driver=StackAcceptanceDriver(session)
    state={"valid":True,"ee_position":[.4,-.25,1.08]}
    for frame in range(3):
        clock.time+=.03
        session.update_frame([fixture_target()],frame,clock(),{"width":100,"height":100})
        driver.update(state,frame*.03)
    assert driver.phase=="baseline_approach"
    clock.time+=.03
    session.update_frame([],3,clock(),{"width":100,"height":100})
    driver.update(state,.09)
    assert driver.done and not driver.report()["passed"]
    assert "target lost" in driver.report()["checks"]["driver"]["reason"]
    assert not session.input["clutch"]
    assert session.selection.mode=="lost"
    assert session.rearm_required


def test_repeated_aged_reads_do_not_reset_fresh_delivered_observations():
    clock=Clock();session=AssistSession(AssistConfig(require_action=True),clock=clock)
    driver=StackAcceptanceDriver(session)
    state={"valid":True,"ee_position":[.4,-.25,1.08]}
    for frame in range(3):
        clock.time=1.+frame*.15
        session.update_frame([fixture_target()],frame,clock()-.1,{"width":100,"height":100})
        driver.update(state,frame*.15)
        assert driver._candidate_frames==frame+1
        if frame<2:
            clock.time+=.12
            driver.update(state,frame*.15+.12)
            driver.update(state,frame*.15+.12)
            assert driver._candidate_frames==frame+1
            assert driver.phase=="await_perception"
    assert driver.phase=="baseline_approach"
    assert driver.target_id=="observed_handle"


def test_real_measurement_gap_or_stale_new_frame_breaks_candidate_sequence():
    clock=Clock();session=AssistSession(AssistConfig(require_action=True),clock=clock)
    driver=StackAcceptanceDriver(session)
    state={"valid":True,"ee_position":[.4,-.25,1.08]}
    for frame in range(2):
        clock.time=1.+frame*.15
        session.update_frame([fixture_target()],frame,clock()-.1,{"width":100,"height":100})
        assert driver._stable_goal(state) is None
    assert driver._candidate_frames==2
    clock.time+=.6
    session.update_frame([fixture_target()],2,clock()-.1,{"width":100,"height":100})
    assert driver._stable_goal(state) is None
    assert driver._candidate_frames==1
    clock.time+=.3
    session.update_frame([fixture_target()],3,clock()-.25,{"width":100,"height":100})
    assert driver._stable_goal(state) is None
    assert driver._candidate_frames==0


def test_baseline_cannot_pass_with_good_error_while_rearm_hold_blocked_motion():
    session=AssistSession(AssistConfig(require_action=True))
    driver=StackAcceptanceDriver(session)
    driver.phase="baseline_approach"
    driver.samples=[{"elapsed_s":1.,"lateral_error_m":.04,"reason":"manual_control","hold":False},
                    {"elapsed_s":10.,"lateral_error_m":.025,"reason":"fresh_clutch_required","hold":True},
                    {"elapsed_s":11.,"lateral_error_m":.025,"reason":"manual_control","hold":False}]
    driver._complete_trial()
    check=driver.results["baseline_approach"]
    assert check["mean_lateral_error_m"]==.025
    assert not check["passed"] and not check["control_valid"]
    assert check["invalid_control_samples"]==1
    assert check["invalid_control_reasons"]==["fresh_clutch_required"]
    driver.results["assisted_approach"]={"passed":True,"control_valid":True,"mean_lateral_error_m":.005}
    assert not driver.report()["checks"]["relative_alignment_improvement"]["passed"]


def test_quality_trials_use_identical_continuous_straight_final_approach():
    clock=Clock();session=AssistSession(AssistConfig(require_action=True),clock=clock)
    driver=StackAcceptanceDriver(session)
    state={"valid":True,"ee_position":[.35,-.25,.9]}
    driver.start_sim=0.;driver.start_pose=np.asarray(state['ee_position'])
    driver.goal=np.array([.4,-.2,1.08]);driver.normal=np.array([-1.,0.,0.])
    driver.offset=np.array([0.,.025,0.])
    recorded={}
    driver._input=lambda desired=None,clutch=False:recorded.update(desired=np.asarray(desired).copy(),clutch=clutch)
    trajectories=[]
    for phase in ('baseline_approach','assisted_approach'):
        driver.phase=phase;driver.phase_start=0.
        values=[]
        for elapsed in np.arange(8.,12.,.1):
            driver.update(state,float(elapsed))
            assert recorded['clutch']
            values.append(recorded['desired'])
        trajectories.append(np.array(values))
    np.testing.assert_allclose(*trajectories)
    # Twenty millimeters are approached continuously along the declared axis,
    # while the same intentional 25 mm lateral operator miss remains present.
    steps=np.diff(trajectories[0],axis=0)
    assert np.all(steps@driver.normal<0.)
    np.testing.assert_allclose(np.linalg.norm(steps,axis=1),.0005,atol=1e-12)
    np.testing.assert_allclose(trajectories[0][:,1],-.175,atol=1e-12)
    assert np.all((trajectories[0]-driver.goal)@driver.normal>=0.)
