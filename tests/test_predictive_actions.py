"""Calibrated shared-control contracts independent of Isaac/headset hardware."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.actions import ActionSelection, calibrated_actions
from teleop_assist.core import AssistConfig, AssistSession
from teleop_assist.predictive import PredictiveServo


class Clock:
    def __init__(self): self.time = 1.
    def __call__(self): return self.time
    def advance(self, dt=.02): self.time += dt


def target():
    common = {"kind": "linear", "grasp_position_world": [.50, -.20, 1.08],
              "approach_axis_world": [1., 0., 0.], "motion_axis_world": [1., 0., 0.]}
    return {"id": "handle", "bbox_px": [20, 20, 80, 80], "center_world": [.5, -.2, 1.08],
            "normal_world": [-1., 0., 0.], "confidence": .3, "pose_confidence": .9,
            "position_sigma_m": .004, "task_calibration": {"id": "known_fixture_v1",
            "source": "configured_fixture", "actions": [dict(common, id="pull", label="Ziehen"),
                                                         dict(common, id="push", label="Drücken")]}}


def packet(sequence, position=(0, 0, 0), clutch=False, **extra):
    return {"seq": sequence, "session_id": "test", "source": "test", "clutch": clutch,
            "right": {"valid": True, "position": list(position), "quaternion_xyzw": [0, 0, 0, 1]},
            "xr_to_world_rotation": np.eye(3).tolist(), **extra}


class Rig:
    def __init__(self):
        self.clock = Clock()
        self.session = AssistSession(AssistConfig(require_action=True, assist_enabled=True), clock=self.clock)
        self.robot = {"valid": True, "ee_position": [.4, -.25, 1.08], "ee_quaternion": [0, 0, 0, 1]}
        self.sequence = 0
        self.frame = 0
        self.publish(target())
        self.session.step(self.robot, .02)

    def publish(self, t):
        self.frame += 1
        self.session.update_frame([] if t is None else [t], self.frame, self.clock(), {"width": 100, "height": 100})

    def input(self, position=(0, 0, 0), clutch=False, **extra):
        self.sequence += 1
        self.clock.advance()
        self.session.receive_input(packet(self.sequence, position, clutch, **extra))

    def select(self):
        self.input()
        self.session.command({"type": "select", "id": "handle"})
        self.session.command({"type": "confirm"})
        self.session.command({"type": "action_select", "action_id": "pull"})
        self.session.command({"type": "action_confirm"})

    def arm(self):
        self.select()
        self.input(clutch=False)
        self.input(clutch=True)

    def step(self):
        return self.session.step(self.robot, .02)


def test_no_actions_are_inferred_from_label_or_box():
    t = target(); t.pop("task_calibration"); t["label"] = "handle"
    assert calibrated_actions(t) == []
    r = Rig(); r.publish(t); r.input()
    r.session.command({"type": "confirm"})
    assert r.session.snapshot()["action"]["available"] == []
    with pytest.raises(ValueError): r.session.command({"type": "action_confirm"})


def test_bad_calibration_is_rejected_before_frame_mutation():
    r = Rig(); t = target(); t["task_calibration"]["actions"][0]["motion_axis_world"] = [0, 0, 0]
    before = r.session.frame_id
    with pytest.raises(ValueError): r.publish(t)
    assert r.session.frame_id == before
    assert r.session.snapshot()["targets"][0]["task_calibration"]["actions"][0]["motion_axis_world"] == [1, 0, 0]


def test_menu_and_confirmation_cannot_start_robot():
    r = Rig(); r.select()
    assert r.step()["hold"]
    assert r.session.control["reason"] == "fresh_clutch_required"
    r.input(clutch=True)
    assert r.step()["hold"]  # A release received before action confirmation does not count.
    r.input(clutch=False); r.input(clutch=True)
    assert r.step()["hold"]
    assert r.session.control["reason"] == "operator_stationary"
    r.input((.01, 0, 0), clutch=True)
    assert not r.step()["hold"]


def test_joystick_and_a_select_object_then_action_without_auto_execute():
    r = Rig(); r.input(primary=True)
    assert r.session.selection.mode == "confirmed"
    assert not r.session.action.confirmed
    r.input(primary=False)
    r.input(axes=[0, 1])
    assert r.session.action.id == "push"
    r.input(primary=True)
    assert r.session.action.confirmed
    r.input(primary=False); r.input(primary=True)
    assert r.session.action.phase == "approach"
    r.input(axes=[0, 0]); r.input(axes=[0, -1])
    assert r.session.action.id == "push"


def test_target_lock_and_action_selection_require_explicit_changes():
    r = Rig(); r.arm()
    other = deepcopy(target()); other["id"] = "other"
    r.session.update_frame([target(), other], 2, r.clock(), {"width": 100, "height": 100})
    with pytest.raises(ValueError): r.session.command({"type": "select", "id": "other"})
    r.session.command({"type": "pointer", "uv": [.99, .99]})
    assert r.session.selection.id == "handle"
    r.session.command({"type": "select", "id": "handle"})
    assert r.session.action.confirmed
    r.session.command({"type": "action_select", "action_id": "push"})
    assert not r.session.action.confirmed and r.session.rearm_required
    assert r.session.predictive.previous_position is None


def test_execute_requires_explicit_engagement_and_released_deadman():
    r = Rig(); r.arm()
    with pytest.raises(ValueError):
        r.session.command({"type": "action_phase", "phase": "execute", "engagement_confirmed": True})
    r.input(clutch=False)
    with pytest.raises(ValueError): r.session.command({"type": "action_phase", "phase": "execute"})
    r.session.command({"type": "action_phase", "phase": "execute", "engagement_confirmed": True})
    state = r.session.snapshot()["action"]
    assert state["phase"] == "execute"
    assert state["engagement_status"] == "operator_confirmed_not_sensor_verified"
    assert state["rearm_required"]
    r.input(clutch=True)
    assert r.step()["hold"]
    r.input(clutch=False); r.input(clutch=True)
    r.input((-.01, 0, 0), clutch=True)
    assert not r.step()["hold"]


def test_prediction_corrects_transverse_miss_and_bounds_actual_command():
    r = Rig(); r.arm(); r.input((.01, -.005, 0), clutch=True)
    result = r.step()
    correction = np.asarray(result["correction"])
    assert correction[1] > 0
    assert abs(correction[0]) < 1e-12
    assert np.linalg.norm(correction) <= .055 + 1e-12
    assert np.linalg.norm(np.asarray(result["ee_target_world"]) - r.robot["ee_position"]) <= .12 * .02 + 1e-12
    d = r.session.control["prediction"]
    assert d["predicted_operator_position"][1] < result["raw_operator_target"][1]
    assert d["model"] == "analytic_constant_velocity_visual_servo"


def test_stationary_and_reversal_cancel_pending_progress():
    r = Rig(); r.arm(); r.input((.03, 0, 0), clutch=True)
    forward = r.step()
    assert forward["ee_target_world"][0] > r.robot["ee_position"][0]
    r.input((.02, 0, 0), clutch=True)
    reverse = r.step()
    assert reverse["ee_target_world"][0] <= r.robot["ee_position"][0] + 1e-12
    assert r.session.control["prediction"]["operator_progress_velocity_m_s"] < 0
    r.input((.02, 0, 0), clutch=True)
    held = r.step()
    assert held["hold"] and r.session.control["reason"] == "operator_stationary"
    np.testing.assert_allclose(held["ee_target_world"], r.robot["ee_position"])


def test_no_packet_motion_expires_before_network_watchdog():
    r = Rig(); r.arm(); r.input((.02, 0, 0), clutch=True); r.step()
    r.clock.advance(.081)
    assert r.step()["hold"]
    assert r.session.control["reason"] == "operator_stationary"


@pytest.mark.parametrize("fault", ["tracking", "stale", "uncertain", "lost"])
def test_faults_hold_and_require_fresh_clutch(fault):
    r = Rig(); r.arm(); r.input((.02, 0, 0), clutch=True); r.step()
    if fault == "tracking": r.input(clutch=True, right={"valid": False})
    elif fault == "stale": r.clock.advance(.8); r.input((.02, 0, 0), clutch=True)
    elif fault == "uncertain":
        t = target(); t["position_sigma_m"] = .04; r.publish(t)
    else: r.publish(None)
    assert r.step()["hold"]
    assert r.session.rearm_required
    assert r.session.predictive.previous_position is None


def test_calibration_change_or_geometry_jump_invalidates_latch():
    for change in ("id", "jump"):
        r = Rig(); r.arm(); t = target()
        if change == "id": t["task_calibration"]["id"] = "different_model"
        else: t["task_calibration"]["actions"][0]["grasp_position_world"][1] += .09
        r.publish(t)
        assert not r.session.action.valid
        assert r.step()["hold"] and r.session.rearm_required


def test_stop_and_release_clear_prediction_and_resume_cannot_reuse_old_motion():
    r = Rig(); r.arm(); r.input((.02, 0, 0), clutch=True); r.step()
    r.session.command({"type": "stop"})
    assert r.step()["hold"] and r.session.predictive.previous_position is None
    with pytest.raises(ValueError): r.session.command({"type": "resume"})
    r.input(clutch=False); r.session.command({"type": "resume"})
    r.input((.03, 0, 0), clutch=True)
    assert r.step()["hold"]
    r.session.command({"type": "release"})
    assert r.session.action.id is None and r.session.predictive.previous_position is None


def test_revolute_projection_never_adds_tangential_progress():
    t = target()
    action = {"id": "rotate", "kind": "revolute", "grasp_position_world": [.1, 0, 0],
              "approach_axis_world": [0, 0, 1], "motion_axis_world": [0, 0, 1],
              "pivot_world": [0, 0, 0], "calibration_id": "wheel_v1"}
    p = PredictiveServo(); p.prime([.12, 0, .01], 1.)
    raw = np.array([.12, .005, .01])
    result = p.evaluate(raw, 1.02, 1.02, action, "execute", t)
    tangent = np.cross([0, 0, 1], raw * [1, 1, 0]); tangent /= np.linalg.norm(tangent)
    assert abs(result["correction"] @ tangent) < 1e-12
    assert result["correction"][0] < 0 and result["correction"][2] < 0
    assert np.linalg.norm(result["correction"]) <= .055 + 1e-12


def test_orientation_correction_is_bounded_and_rotation_progress_is_free():
    t = target(); action = calibrated_actions(t)[0]
    action["grasp_quaternion_xyzw"] = Rotation.from_euler("y", 1.).as_quat().tolist()
    p = PredictiveServo(); p.prime([.4, -.2, 1.08], 1., [0, 0, 0, 1])
    q = Rotation.from_euler("z", .02).as_quat()
    result = p.evaluate([.4, -.2, 1.08], 1.02, 1.02, action, "approach", t, q)
    assert 0 < np.linalg.norm(result["orientation_correction"]) <= .2 + 1e-12
    action.update(kind="revolute", motion_axis_world=[0, 0, 1], pivot_world=[.3, -.2, 1.08])
    p.prime([.5, -.2, 1.08], 1., [0, 0, 0, 1])
    result = p.evaluate([.5, -.195, 1.08], 1.02, 1.02, action, "execute", t, q)
    assert abs(result["orientation_correction"][2]) < 1e-12


def test_unavailable_orientation_does_not_block_good_translation():
    t = target(); t["orientation_sigma_rad"] = .7
    p = PredictiveServo(); p.prime([.4, -.2, 1.08], 1.)
    result = p.evaluate([.401, -.2, 1.08], 1.02, 1.02, calibrated_actions(t)[0], "approach", t, [0, 0, 0, 1])
    assert not result["hold"]
    np.testing.assert_allclose(result["orientation_correction"], [0, 0, 0])


def test_axial_rotation_uses_angular_progress_without_fake_lever_arm():
    t=target()
    entry={"id":"rotate","kind":"axial_rotation","grasp_position_world":[.5,-.2,1.08],
           "pivot_world":[.5,-.2,1.08],"approach_axis_world":[1,0,0],"motion_axis_world":[1,0,0],
           "grasp_quaternion_xyzw":[0,0,0,1]}
    t["task_calibration"]["actions"]=[entry]
    action=calibrated_actions(t)[0]
    p=PredictiveServo();p.prime([.5,-.2,1.08],1.,[0,0,0,1])
    q=Rotation.from_euler("x",.03).as_quat()
    result=p.evaluate([.5,-.2,1.08],1.02,1.02,action,"execute",t,q)
    assert not result["hold"]
    np.testing.assert_allclose(result["correction"],[0,0,0])
    assert abs(result["orientation_correction"][0])<1e-12
    assert p.diagnostics["operator_angular_progress_rad_s"]>0
    q=Rotation.from_euler("x",.02).as_quat()
    p.evaluate([.5,-.2,1.08],1.04,1.04,action,"execute",t,q)
    assert p.diagnostics["operator_angular_progress_rad_s"]<0
    assert p.evaluate([.5,-.2,1.08],1.06,1.06,action,"execute",t,q)["hold"]
    assert p.evaluate([.5,-.2,1.08],1.08,1.08,action,"execute",t)["reason"]=="pose_control_required_for_rotation"


def test_prediction_reduces_future_cross_track_error_for_synthetic_disturbances():
    t=target();action=calibrated_actions(t)[0]
    p=PredictiveServo()
    raw_before=[];corrected_after=[]
    for index in range(80):
        when=1.+.02*index
        raw=np.array([.43+.0005*index,-.2+.015+.006*np.sin(index*.15),1.08+.005*np.cos(index*.2)])
        result=p.evaluate(raw,when,when,action,"approach",t)
        if result["hold"]:continue
        predicted=np.array(p.diagnostics["predicted_operator_position"])
        raw_before.append(np.linalg.norm(predicted[1:]-np.array(action["grasp_position_world"])[1:]))
        corrected_after.append(np.linalg.norm((predicted+result["correction"])[1:]-np.array(action["grasp_position_world"])[1:]))
        assert np.linalg.norm(result["correction"])<=.055+1e-12
        assert abs(result["correction"][0])<1e-12
    assert len(raw_before)==79
    assert np.mean(corrected_after)<.2*np.mean(raw_before)


def test_bounded_execution_uses_fixed_origin_caps_40mm_and_reverses_immediately():
    r=Rig()
    calibrated=target()
    for capability in calibrated['task_calibration']['actions']:
        capability.update(execution_ready=True,execution_travel_limit_m=.04)
    r.publish(calibrated);r.arm()
    r.input(clutch=False)
    r.session.command({'type':'action_phase','phase':'execute','engagement_confirmed':True})
    origin=np.array(r.robot['ee_position'])
    r.input(clutch=False);r.input(clutch=True)
    for index in range(1,121):
        moving=deepcopy(calibrated)
        shift=r.robot['ee_position'][0]-origin[0]
        moving['center_world'][0]+=shift
        for capability in moving['task_calibration']['actions']:
            capability['grasp_position_world'][0]+=shift
        r.publish(moving)
        r.input((index*.001,0,0),clutch=True)
        reference=r.step()
        r.robot['ee_position']=reference['ee_target_world']
        assert r.robot['ee_position'][0]<=origin[0]+.04+1e-10
    assert r.robot['ee_position'][0]==pytest.approx(origin[0]+.04,abs=1e-7)
    assert r.session.control['reason']=='execution_travel_limit'
    np.testing.assert_allclose(r.session.action.execution_origin_world,origin)
    # The hand remains far beyond the envelope, but its very first reversed
    # sample must command a retreat rather than wait for old backlog to unwind.
    before=r.robot['ee_position'][0]
    r.input((.119,0,0),clutch=True)
    reference=r.step()
    assert reference['ee_target_world'][0]<before
    r.robot['ee_position']=reference['ee_target_world']
    # Releasing and re-engaging the clutch cannot silently renew 40 mm.
    r.input((.119,0,0),clutch=False);r.input((.119,0,0),clutch=True)
    np.testing.assert_allclose(r.session.action.execution_origin_world,origin)
    for index in range(20):
        r.publish(moving);r.input((.120+index*.001,0,0),clutch=True)
        reference=r.step();r.robot['ee_position']=reference['ee_target_world']
        assert r.robot['ee_position'][0]<=origin[0]+.04+1e-10


def test_bounded_action_limit_cannot_disappear_during_execution():
    r=Rig();calibrated=target()
    calibrated['task_calibration']['actions'][0].update(execution_ready=True,execution_travel_limit_m=.04)
    r.publish(calibrated);r.arm();r.input(clutch=False)
    r.session.command({'type':'action_phase','phase':'execute','engagement_confirmed':True})
    changed=deepcopy(calibrated);changed['task_calibration']['actions'][0].pop('execution_travel_limit_m')
    r.publish(changed)
    assert not r.session.action.valid
    assert r.session.action.reason=='execution_limit_changed'
    assert r.step()['hold']


def test_predictive_reference_drives_lagging_servo_and_flushes_reversal_queue():
    r=Rig();r.arm()
    for index in range(1,201):
        r.publish(target());r.input((index*.0004,0,0),clutch=True)
        reference=r.step()
        measured=np.asarray(r.robot['ee_position'])
        r.robot['ee_position']=(measured+3.*.02*(np.asarray(reference['ee_target_world'])-measured)).tolist()
    # Resolved-rate physics needs a bounded tracking error to make progress.
    # A velocity*dt budget relative to measured EE would travel only ~4 mm.
    assert r.robot['ee_position'][0]>.45
    before=r.robot['ee_position'][0]
    r.publish(target());r.input((.0796,0,0),clutch=True)
    reversed_reference=r.step()
    assert reversed_reference['reset_pending']
    assert reversed_reference['ee_target_world'][0]<before
    # The next reversed packet must keep retreating even while the raw absolute
    # hand goal is still beyond the measured robot position.
    r.robot['ee_position'][0]+=3.*.02*(reversed_reference['ee_target_world'][0]-before)
    r.publish(target());r.input((.0792,0,0),clutch=True)
    assert r.step()['ee_target_world'][0]<r.robot['ee_position'][0]
    r.input((.0792,0,0),clutch=True)
    stopped=r.step()
    assert stopped['hold']
    np.testing.assert_allclose(stopped['ee_target_world'],r.robot['ee_position'])


@pytest.mark.parametrize('axial_step',[0.,.0001])
def test_small_axial_plant_overshoot_does_not_erase_transverse_servo_queue(axial_step):
    r=Rig();r.arm()
    resets=0
    for index in range(1,201):
        r.publish(target())
        # The hand continuously moves vertically; longitudinal motion may be
        # zero. The assistance must still converge laterally onto the handle.
        r.input((index*axial_step,0,index*.0001),clutch=True)
        reference=r.step()
        assert not reference['hold']
        resets+=bool(reference.get('reset_pending'))
        measured=np.asarray(r.robot['ee_position'])
        commanded=np.asarray(reference['ee_target_world'])
        measured+=3.*.02*(commanded-measured)
        # A small measured longitudinal overshoot of the previous reference
        # reproduces PD/gravity oscillations. It is not an operator reversal.
        measured[0]=commanded[0]+.00015
        r.robot['ee_position']=measured.tolist()
    assert resets==0
    assert abs(r.robot['ee_position'][1]-(-.2))<.002
    # A deliberate reversal still flushes even with transverse backlog.
    if axial_step:
        r.input((199*axial_step,0,.0201),clutch=True)
        reference=r.step()
        assert reference['reset_pending']
        assert reference['ee_target_world'][0]<r.robot['ee_position'][0]


def test_zero_axial_velocity_flushes_once_then_allows_transverse_motion():
    r=Rig();r.arm()
    for index in range(1,20):
        r.publish(target());r.input((index*.0002,0,0),clutch=True)
        reference=r.step()
        measured=np.asarray(r.robot['ee_position'])
        r.robot['ee_position']=(measured+.06*(np.asarray(reference['ee_target_world'])-measured)).tolist()
    r.input((.0038,0,.0001),clutch=True)
    assert r.step()['reset_pending']
    for index in range(2,51):
        r.publish(target());r.input((.0038,0,index*.0001),clutch=True)
        reference=r.step()
        assert not reference['hold'] and not reference['reset_pending']
        assert reference['ee_target_world'][0]==pytest.approx(r.robot['ee_position'][0])
        measured=np.asarray(r.robot['ee_position'])
        r.robot['ee_position']=(measured+.06*(np.asarray(reference['ee_target_world'])-measured)).tolist()
    assert r.robot['ee_position'][1]>-.21


def test_diagonal_motion_with_variable_delivery_time_does_not_create_false_stops():
    r=Rig();r.arm()
    for index in range(1,161):
        r.publish(target())
        # Alternate ordinary delivery and an occasional 60 ms stall. The
        # hand still moves monotonically in both axes, but the axial speed
        # estimate alternates across the whole-hand 2 mm/s motion threshold.
        if index%8==0:r.clock.advance(.06)
        r.input((index*.0001,0,index*.00015),clutch=True)
        reference=r.step()
        assert not reference['hold']
        assert not reference['reset_pending']
        measured=np.asarray(r.robot['ee_position'])
        r.robot['ee_position']=(measured+.06*(np.asarray(reference['ee_target_world'])-measured)).tolist()
    assert r.robot['ee_position'][0]>.41
    assert abs(r.robot['ee_position'][1]-(-.2))<.003
    # True component stop while the hand keeps moving transversely still
    # cancels the pending approach exactly once.
    r.input((.016,0,.02415),clutch=True)
    reference=r.step()
    assert reference['reset_pending']
    assert reference['ee_target_world'][0]==pytest.approx(r.robot['ee_position'][0])


@pytest.mark.parametrize('rotation',[False,True])
def test_delivery_jitter_does_not_stop_motion_but_fresh_stop_and_reverse_are_immediate(rotation):
    t=target();action=calibrated_actions(t)[0]
    if rotation:
        action.update(kind='axial_rotation',pivot_world=[.4,-.2,1.08],motion_axis_world=[1.,0.,0.],
                      grasp_quaternion_xyzw=[0.,0.,0.,1.])
    p=PredictiveServo();base=np.array([.4,-.2,1.08]);when=1.
    p.prime(base,when,[0.,0.,0.,1.])
    observed_slow_delivery=False
    for index in range(1,81):
        # Eight samples represent 40 ms of steady hand motion. Scheduling
        # delivers seven quickly and one 22.5 ms later; newest velocity alone
        # falsely drops below the stillness threshold on every eighth sample.
        when+=.0225 if index%8==0 else .0025
        raw=base+([0.,0.,0.] if rotation else [index*.000025,0.,0.])
        q=Rotation.from_rotvec([index*.0001 if rotation else 0.,0.,0.]).as_quat()
        result=p.evaluate(raw,when,when,action,'execute' if rotation else 'approach',t,q)
        assert not result['hold']
        if index%8==0:
            current=np.linalg.norm(p.angular_velocity if rotation else p.velocity)
            observed_slow_delivery|=current<(.015 if rotation else .002)
    assert observed_slow_delivery
    when+=.005
    raw=base+([0.,0.,0.] if rotation else [79*.000025,0.,0.])
    q=Rotation.from_rotvec([79*.0001 if rotation else 0.,0.,0.]).as_quat()
    reverse=p.evaluate(raw,when,when,action,'execute' if rotation else 'approach',t,q)
    assert not reverse['hold']
    assert (p.angular_velocity[0] if rotation else p.velocity[0])<0
    when+=.005
    stopped=p.evaluate(raw,when,when,action,'execute' if rotation else 'approach',t,q)
    assert stopped['hold'] and stopped['reason']=='operator_stationary'
    assert p.motion_speed==0 and p.angular_motion_speed==0
    # Receipt jitter never extends the independent missing-packet watchdog.
    assert p.evaluate(raw,when,when+.081,action,'approach',t,q)['hold']


def test_stationary_arm_accepts_fresh_gripper_close_open_but_contact_blocks_closure():
    r=Rig();r.arm()
    for grip in (.2,.55,.85,.4,0.):
        r.input(clutch=True,trigger=grip)
        reference=r.step()
        assert reference['hold']
        assert r.session.control['reason']=='operator_stationary'
        assert reference['grip']==grip
        np.testing.assert_allclose(reference['ee_target_world'],r.robot['ee_position'])
    r.robot.update(contact_active=True,contact_force_world=[-20.,0,0])
    r.input(clutch=True,trigger=.85)
    assert r.step()['grip']==0.
    assert r.session.control['reason']=='collision'


@pytest.mark.parametrize('axis',[[1.,0.,0.],[0.,1.,0.],[2**-.5,2**-.5,0.]])
def test_axial_orientation_drives_lagging_servo_and_flushes_reverse_and_stop(axis):
    axis=np.asarray(axis)
    r=Rig();calibrated=target()
    calibrated['task_calibration']['actions']=[{
        'id':'pull','label':'Rotate test fixture','kind':'axial_rotation',
        'grasp_position_world':list(r.robot['ee_position']),
        'pivot_world':list(r.robot['ee_position']),
        'approach_axis_world':[-1.,0.,0.],'motion_axis_world':axis.tolist(),
        'grasp_quaternion_xyzw':[0.,0.,0.,1.]}]
    r.publish(calibrated);r.select()
    r.input(clutch=False,orientation_mode='pose')
    r.session.command({'type':'action_phase','phase':'execute','engagement_confirmed':True})
    r.input(clutch=False,orientation_mode='pose');r.input(clutch=True,orientation_mode='pose')

    def send(angle,position=(0.,0.,0.)):
        r.publish(calibrated)
        r.input(clutch=True,orientation_mode='pose',right={'valid':True,'position':list(position),
                'quaternion_xyzw':Rotation.from_rotvec(axis*angle).as_quat().tolist()})
        return r.step()

    previous_command=Rotation.from_quat(r.robot['ee_quaternion'])
    for index in range(1,201):
        reference=send(index*.004)  # 0.2 rad/s operator input at 50 Hz.
        assert not reference['hold']
        command=Rotation.from_quat(reference['ee_quat_xyzw'])
        measured=Rotation.from_quat(r.robot['ee_quaternion'])
        error=(command*measured.inv()).as_rotvec()
        assert np.linalg.norm(error)<=.25+1e-10
        assert (command*previous_command.inv()).magnitude()<=r.session.config.max_angular_speed*.02+1e-10
        # Same type of delayed resolved-rate response as the Cartesian test;
        # this is a controller regression, not a physical knob acceptance.
        measured=Rotation.from_rotvec(3.*.02*error)*measured
        r.robot['ee_quaternion']=measured.as_quat().tolist()
        previous_command=command
    assert float(measured.as_rotvec()@axis)>.60
    # The old measured-relative velocity*dt budget produced only ~0.048 rad.
    before=Rotation.from_quat(r.robot['ee_quaternion'])
    reverse=send(.796)
    assert reverse['reset_pending']
    error=(Rotation.from_quat(reverse['ee_quat_xyzw'])*before.inv()).as_rotvec()
    assert float(error@axis)<0
    r.robot['ee_quaternion']=(Rotation.from_rotvec(3.*.02*error)*before).as_quat().tolist()
    second=send(.792)
    measured=Rotation.from_quat(r.robot['ee_quaternion'])
    assert float((Rotation.from_quat(second['ee_quat_xyzw'])*measured.inv()).as_rotvec()@axis)<0
    # Translation-only fresh input still flushes pending angular progress.
    zero_twist=send(.792,(0.,.001,0.))
    assert zero_twist['reset_pending']
    assert abs(float((Rotation.from_quat(zero_twist['ee_quat_xyzw'])*measured.inv()).as_rotvec()@axis))<1e-10
    stationary=send(.792,(0.,.001,0.))
    assert stationary['hold'] and r.session.control['reason']=='operator_stationary'
    assert (Rotation.from_quat(r.session.command_quaternion)*measured.inv()).magnitude()<1e-10


@pytest.mark.parametrize('direction',[-1.,1.])
def test_disturbance_outside_execution_envelope_never_causes_auto_snapback(direction):
    r=Rig();calibrated=target()
    calibrated['task_calibration']['actions'][0].update(execution_ready=True,execution_travel_limit_m=.04)
    r.publish(calibrated);r.arm();r.input(clutch=False)
    r.session.command({'type':'action_phase','phase':'execute','engagement_confirmed':True})
    origin=np.array(r.robot['ee_position'])
    r.input(clutch=False);r.input(clutch=True)
    # A measured disturbance is not a robot command and does not move the
    # latched execution origin. It can be well outside the nominal 40 mm band.
    r.robot['ee_position'][0]=origin[0]+direction*.16
    r.input((direction*.01,0,0),clutch=True)
    held=r.step()
    assert held['reset_pending']
    assert r.session.control['execution']['outside_envelope']
    assert held['ee_target_world'][0]==pytest.approx(r.robot['ee_position'][0])
    assert np.linalg.norm(np.asarray(held['ee_target_world'])-r.robot['ee_position'])<=r.session.config.max_speed*.02+1e-10
    np.testing.assert_allclose(r.session.action.execution_origin_world,origin)
    # Recovery is initiated only by reverse operator input and remains speed
    # limited; there is no projection jump to the original envelope boundary.
    r.input((direction*.009,0,0),clutch=True)
    recovery=r.step()
    delta=np.asarray(recovery['ee_target_world'])-r.robot['ee_position']
    assert float(delta[0]*direction)<0
    assert np.linalg.norm(np.asarray(recovery['ee_target_world'])-held['ee_target_world'])<=r.session.config.max_speed*.02+1e-10
    assert np.linalg.norm(delta)<=.075+1e-10
    assert abs(recovery['ee_target_world'][0]-origin[0])>.10
