"""Behavioral contract tests: these do not establish physical robot safety."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.copilot.controller import OrientationCopilot
from teleop_assist.copilot.types import (CopilotConfig, CopilotRequest, GraspCandidate,
                                       OperatorSample, ProposalBatch)


def pose(angle=0, axis="z", translation=(0.35, -0.2, 0.9)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
    result[:3, 3] = translation
    return result


def angle_between(a, b):
    return float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())


def operator(now, angle=0, **kwargs):
    return OperatorSample(kwargs.get("left", pose(-10, "x")), pose(angle),
                          kwargs.get("acquired", now), kwargs.get("clock", "mono"),
                          kwargs.get("valid", True))


def candidate(name="a", angle=60, **kwargs):
    return GraspCandidate(name, pose(angle), kwargs.get("score", .8),
                          kwargs.get("geometry", True), kwargs.get("ik", True),
                          kwargs.get("collision", True), "fixture-model")


def batch(now, *candidates, **kwargs):
    return ProposalBatch(kwargs.get("target", "object-1"), kwargs.get("obs", "obs-1"),
                         kwargs.get("captured", now), kwargs.get("clock", "mono"),
                         kwargs.get("cal", "cal-1"), candidates or (candidate(),),
                         kwargs.get("static", True), kwargs.get("valid", True))


def tick(ctrl, now, op=None, proposals=None, request=None, **kwargs):
    return ctrl.step(op or operator(now), proposals if proposals is not None else batch(now),
                     now=now, clock_id=kwargs.get("clock", "mono"),
                     calibration_id=kwargs.get("cal", "cal-1"), request=request)


def armed(ctrl, target_angle=60):
    tick(ctrl, 0, proposals=batch(0, candidate(angle=target_angle)),
         request=CopilotRequest(select_candidate_id="a"))
    result = tick(ctrl, .02, proposals=batch(.02, candidate(angle=target_angle)),
                  request=CopilotRequest(confirm=True))
    assert result.snapshot["confirmed"]
    return result


def bound_batch(now, *, observation='validation-1', **changes):
    return replace(batch(now, obs=observation), model_source_id='frozen-inference-1',
                   model_source_capture_timestamp=0., **changes)


def test_explicit_model_binding_allows_independently_revalidated_selection_to_confirm():
    c=OrientationCopilot()
    tick(c,.1,proposals=bound_batch(.1),request=CopilotRequest(select_candidate_id='a'))
    result=tick(c,.12,proposals=bound_batch(.12,observation='validation-2'),request=CopilotRequest(confirm=True))
    assert result.snapshot['confirmed']
    assert result.snapshot['model_source_id']=='frozen-inference-1'
    assert result.snapshot['model_source_capture_timestamp']==0.
    assert result.snapshot['confirmed_observation_id']=='validation-2'
    assert result.snapshot['validation_observation_id']=='validation-2'


@pytest.mark.parametrize('fault',['model_id','model_capture','pose','calibration','geometry','ik','collision'])
def test_model_binding_does_not_allow_changed_pose_source_or_unvalidated_confirmation(fault):
    c=OrientationCopilot()
    tick(c,.1,proposals=bound_batch(.1),request=CopilotRequest(select_candidate_id='a'))
    next_batch=bound_batch(.12,observation='validation-2')
    if fault=='model_id':next_batch=replace(next_batch,model_source_id='other-inference')
    elif fault=='model_capture':next_batch=replace(next_batch,model_source_capture_timestamp=.01)
    elif fault=='calibration':next_batch=replace(next_batch,calibration_id='other-calibration')
    else:
        item=next_batch.candidates[0]
        item=replace(item,**({'base_ee':pose(61)} if fault=='pose' else {fault+'_valid':False}))
        next_batch=replace(next_batch,candidates=(item,))
    result=tick(c,.12,proposals=next_batch,request=CopilotRequest(confirm=True))
    assert not result.snapshot['confirmed'] and not result.applied


def test_bound_confirmed_choice_loses_permission_when_model_binding_changes():
    c=OrientationCopilot()
    tick(c,.1,proposals=bound_batch(.1),request=CopilotRequest(select_candidate_id='a'))
    tick(c,.12,proposals=bound_batch(.12),request=CopilotRequest(confirm=True))
    changed=replace(bound_batch(.14,observation='validation-3'),model_source_id='other-inference')
    result=tick(c,.14,proposals=changed,request=CopilotRequest(assist_held=True))
    assert not result.snapshot['confirmed'] and result.reason=='candidate_model_source_binding_changed'


@pytest.mark.parametrize('binding,stamp',[(None,0.),('',0.),('source',None),('source',float('nan')),('source',.2)])
def test_partial_or_invalid_model_binding_fails_closed(binding,stamp):
    c=OrientationCopilot();proposals=replace(batch(.1),model_source_id=binding,model_source_capture_timestamp=stamp)
    result=tick(c,.1,proposals=proposals,request=CopilotRequest(select_candidate_id='a'))
    assert result.reason=='target_model_source_binding_invalid' and not result.snapshot['confirmed']


def test_archived_model_clock_is_preserved_without_retiming_current_validation():
    c=OrientationCopilot()
    first=replace(bound_batch(.1),model_source_capture_timestamp=2191930.7,model_source_clock_id='previous-host-boot')
    tick(c,.1,proposals=first,request=CopilotRequest(select_candidate_id='a'))
    current=replace(first,observation_id='current-validation-2',capture_timestamp=.12)
    result=tick(c,.12,proposals=current,request=CopilotRequest(confirm=True))
    assert result.snapshot['confirmed']
    assert result.snapshot['model_source_capture_timestamp']==2191930.7
    assert result.snapshot['model_source_clock_id']=='previous-host-boot'
    changed=replace(current,capture_timestamp=.14,model_source_clock_id='different-host-boot')
    result=tick(c,.14,proposals=changed,request=CopilotRequest(assist_held=True))
    assert not result.snapshot['confirmed'] and not result.applied


@pytest.mark.parametrize('source_clock',['',False,3,'mono'])
def test_invalid_or_same_clock_future_model_time_is_rejected(source_clock):
    c=OrientationCopilot()
    source=replace(bound_batch(.1),model_source_capture_timestamp=200.,model_source_clock_id=source_clock)
    result=tick(c,.1,proposals=source,request=CopilotRequest(select_candidate_id='a'))
    assert result.reason=='target_model_source_binding_invalid'


def test_explicit_selection_confirmation_and_hold_are_all_required():
    c = OrientationCopilot()
    r = tick(c, 0, request=CopilotRequest(assist_held=True))
    assert not r.applied
    r = tick(c, .02, request=CopilotRequest(select_candidate_id="a", confirm=True, assist_held=True))
    assert not r.applied and r.reason == "separate_confirmation_required"
    r = tick(c, .04, request=CopilotRequest(confirm=True))
    assert r.snapshot["confirmed"] and not r.applied
    r = tick(c, .06, request=CopilotRequest(assist_held=True))
    assert r.applied and r.snapshot["state"] == "assisting"


@pytest.mark.parametrize("angle", [179.999, 180, -179.999, -180])
def test_near_pi_rotation_bounded_and_translation_left_hand_untouched(angle):
    c = OrientationCopilot()
    armed(c, angle)
    previous_offset = np.eye(4)
    for step in range(2, 62):
        now = step * .02
        op = operator(now)
        r = tick(c, now, op, batch(now, candidate(angle=angle)), CopilotRequest(assist_held=True))
        assert not r.input_hold_required
        np.testing.assert_array_equal(r.left_base_ee, op.left_base_ee)
        np.testing.assert_array_equal(r.right_base_ee[:3, 3], op.right_base_ee[:3, 3])
        assert angle_between(r.right_base_ee, op.right_base_ee) <= np.deg2rad(20) + 1e-9
        offset = np.eye(4)
        offset[:3, :3] = r.right_base_ee[:3, :3] @ op.right_base_ee[:3, :3].T
        assert angle_between(offset, previous_offset) <= np.deg2rad(30) * .02 + 1e-9
        previous_offset = offset
    assert angle_between(r.right_base_ee, op.right_base_ee) == pytest.approx(np.deg2rad(20))


def test_offset_rate_is_preserved_when_increment_budget_allows_it():
    outputs = []
    for dt in [.01, .02, .025]:
        c = OrientationCopilot()
        armed(c)
        for i in range(1, round(.5 / dt) + 1):
            now = .02 + i * dt
            result = tick(c, now, request=CopilotRequest(assist_held=True))
        outputs.append(result.snapshot["orientation_offset_rad"])
    np.testing.assert_allclose(outputs, np.deg2rad(15), atol=1e-9)


@pytest.mark.parametrize('dt',[.05,.1,.19])
def test_slow_validation_cannot_catch_up_with_a_large_assistance_jump(dt):
    c=OrientationCopilot();armed(c)
    now=.02;previous=pose(0)
    for _ in range(6):
        now+=dt
        result=tick(c,now,request=CopilotRequest(assist_held=True))
        assert angle_between(result.right_base_ee,previous)==pytest.approx(np.deg2rad(1))
        previous=result.right_base_ee
    # The same increment cap applies when the hold button is released.
    result=tick(c,now+dt,request=CopilotRequest(assist_held=False))
    assert angle_between(result.right_base_ee,previous)==pytest.approx(np.deg2rad(1))
    assert result.snapshot['returning_to_manual']
    assert result.snapshot['max_offset_increment_rad']==pytest.approx(np.deg2rad(1))


def test_model_ranking_reordering_never_switches_latched_candidate():
    c = OrientationCopilot()
    armed(c)
    for i in range(2, 8):
        now = i * .02
        r = tick(c, now, proposals=batch(now, candidate("b", -90, score=.99),
                                        candidate("a", 60, score=.01), obs=f"obs-{i}"),
                 request=CopilotRequest(assist_held=True))
        assert r.snapshot["confirmed_candidate_id"] == "a"
        assert r.snapshot["confirmed_observation_id"] == "obs-1"
        assert r.snapshot["validation_observation_id"] == f"obs-{i}"
        np.testing.assert_allclose(r.preview_right_base_ee[:3, :3], pose(60)[:3, :3])
        assert Rotation.from_matrix(r.right_base_ee[:3, :3]).as_rotvec()[2] > 0


@pytest.mark.parametrize("change,reason", [
    ({"target": "object-2"}, "target_changed"),
    ({"cal": "cal-2"}, "target_calibration_mismatch"),
    ({"static": False}, "target_not_stationary"),
    ({"valid": False}, "target_invalid"),
    ({"captured": -2.0}, "target_stale"),
    ({"captured": 1.0}, "target_acquisition_time_in_future"),
    ({"captured": None}, "target_acquisition_time_unknown"),
    ({"clock": "camera"}, "target_clock_mismatch"),
])
def test_target_invalidation_requires_new_selection_and_confirmation(change, reason):
    c = OrientationCopilot()
    armed(c)
    tick(c, .04, request=CopilotRequest(assist_held=True))
    r = tick(c, .06, proposals=batch(.06, **change), request=CopilotRequest(assist_held=True))
    assert not r.snapshot["confirmed"] and r.reason == reason
    r = tick(c, .08, request=CopilotRequest(assist_held=True))
    assert not r.snapshot["confirmed"] and not r.applied


def test_changed_pose_under_same_id_invalidates_instead_of_following_it():
    c = OrientationCopilot()
    armed(c)
    tick(c, .04, request=CopilotRequest(assist_held=True))
    changed = batch(.06, candidate("a", -90), obs="new")
    r = tick(c, .06, proposals=changed, request=CopilotRequest(assist_held=True))
    assert r.reason == "candidate_pose_or_source_changed"
    assert not r.snapshot["confirmed"]


@pytest.mark.parametrize("flag,reason", [
    ("geometry", "candidate_geometry_unvalidated"),
    ("ik", "candidate_ik_unvalidated"),
    ("collision", "candidate_collision_unvalidated"),
])
def test_unvalidated_candidates_can_preview_but_never_activate(flag, reason):
    c = OrientationCopilot()
    p = batch(0, candidate(**{flag: False}))
    r = tick(c, 0, proposals=p, request=CopilotRequest(select_candidate_id="a"))
    assert r.preview_right_base_ee is not None and not r.applied
    r = tick(c, .02, proposals=replace(p, capture_timestamp=.02),
             request=CopilotRequest(confirm=True, assist_held=True))
    assert r.reason == reason and not r.snapshot["confirmed"] and not r.applied


def test_new_unvalidated_batch_cannot_inherit_previous_validation():
    c = OrientationCopilot()
    armed(c)
    r = tick(c, .04, proposals=batch(.04, candidate(ik=False), obs="new"),
             request=CopilotRequest(assist_held=True))
    assert not r.snapshot["confirmed"] and r.reason == "candidate_ik_unvalidated"


def test_selected_observation_is_bound_until_confirmed():
    c = OrientationCopilot()
    tick(c, 0, request=CopilotRequest(select_candidate_id="a"))
    r = tick(c, .02, proposals=batch(.02, obs="new"),
             request=CopilotRequest(confirm=True, assist_held=True))
    assert not r.snapshot["confirmed"] and not r.applied


@pytest.mark.parametrize("op_kwargs,reason", [
    ({"valid": False}, "input_tracking_invalid"),
    ({"acquired": None}, "input_acquisition_time_unknown"),
    ({"acquired": -1}, "input_stale"),
    ({"acquired": 1}, "input_acquisition_time_in_future"),
    ({"acquired": float("nan")}, "input_acquisition_time_invalid"),
    ({"clock": "headset"}, "input_clock_mismatch"),
])
def test_input_dropout_requires_caller_hold_and_erases_assist_permission(op_kwargs, reason):
    c = OrientationCopilot()
    armed(c)
    r = tick(c, .04, op=operator(.04, **op_kwargs), request=CopilotRequest(assist_held=True))
    assert r.input_hold_required and not r.applied and r.reason == reason
    assert not r.snapshot["confirmed"]
    json.dumps(r.snapshot, allow_nan=False)
    r = tick(c, .06, request=CopilotRequest(assist_held=True))
    assert not r.applied and not r.input_hold_required


@pytest.mark.parametrize("now", [.02, .01, .5])
def test_nonpositive_or_large_dt_requires_hold(now):
    c = OrientationCopilot()
    armed(c)
    r = tick(c, now, request=CopilotRequest(assist_held=True))
    assert r.input_hold_required and r.reason == "step_interval_invalid"
    assert not r.snapshot["confirmed"]


def test_clock_change_requires_hold_and_new_calibration_erases_selection():
    c = OrientationCopilot()
    armed(c)
    r = tick(c, .04, operator(.04, clock="other"), batch(.04, clock="other"),
             clock="other", request=CopilotRequest(assist_held=True))
    assert r.input_hold_required and not r.snapshot["confirmed"]
    c = OrientationCopilot()
    armed(c)
    r = tick(c, .04, proposals=batch(.04, cal="cal-2"), cal="cal-2",
             request=CopilotRequest(assist_held=True))
    assert not r.snapshot["confirmed"] and r.reason == "calibration_changed"


def test_button_release_returns_offset_smoothly_to_manual_and_rehold_is_explicit():
    c = OrientationCopilot()
    armed(c)
    for i in range(2, 20):
        r = tick(c, i * .02, request=CopilotRequest(assist_held=True))
    previous = r.snapshot["orientation_offset_rad"]
    r = tick(c, .4)
    assert r.snapshot["state"] == "releasing" and r.snapshot["returning_to_manual"]
    assert previous - r.snapshot["orientation_offset_rad"] == pytest.approx(np.deg2rad(.6))
    for i in range(21, 40):
        r = tick(c, i * .02)
    assert not r.applied and r.snapshot["confirmed"]
    r = tick(c, .8, request=CopilotRequest(assist_held=True))
    assert r.applied
    r = tick(c, .82, request=CopilotRequest(release=True, assist_held=True))
    assert not r.snapshot["confirmed"] and r.reason == "explicit_release"


def test_deliberate_counterrotation_releases_but_small_alternating_noise_does_not():
    c = OrientationCopilot()
    armed(c)
    for i in range(2, 30):
        now = i * .02
        r = tick(c, now, operator(now, .15 * (-1) ** i),
                 request=CopilotRequest(assist_held=True))
        assert r.snapshot["confirmed"]
    for i in range(30, 40):
        now = i * .02
        r = tick(c, now, operator(now, -(i - 29) * 2),
                 request=CopilotRequest(assist_held=True))
        if r.reason == "manual_counterrotation":
            break
    assert r.reason == "manual_counterrotation" and not r.snapshot["confirmed"]


def test_invalid_transform_is_rejected_instead_of_fabricating_valid_manual_pose():
    c = OrientationCopilot()
    bad = pose()
    bad[0, 0] = -1
    with pytest.raises(ValueError, match="proper rigid"):
        tick(c, 0, replace(operator(0), right_base_ee=bad))


def test_no_proposals_and_duplicate_ids_do_not_arm():
    c = OrientationCopilot()
    r = c.step(operator(0), None, now=0, clock_id="mono", calibration_id="cal-1")
    assert r.reason == "no_target_proposals" and not r.applied
    r = tick(c, .02, proposals=batch(.02, candidate(), candidate()),
             request=CopilotRequest(select_candidate_id="a", confirm=True))
    assert r.reason == "candidate_identity_invalid" and not r.applied


def test_input_arrays_not_modified_and_confirmed_pose_is_a_copy():
    c = OrientationCopilot()
    p = batch(0)
    original = p.candidates[0].base_ee.copy()
    tick(c, 0, proposals=p, request=CopilotRequest(select_candidate_id="a"))
    tick(c, .02, proposals=replace(p, capture_timestamp=.02), request=CopilotRequest(confirm=True))
    np.testing.assert_array_equal(p.candidates[0].base_ee, original)
    p.candidates[0].base_ee[:3, :3] = pose(-90)[:3, :3]
    r = tick(c, .04, request=CopilotRequest(assist_held=True))
    assert r.snapshot["confirmed"]
    np.testing.assert_allclose(r.preview_right_base_ee[:3, :3], original[:3, :3])


@pytest.mark.parametrize("kwargs", [
    {"offset_rate_rad_s": 0}, {"max_step_s": 0},
    {"max_offset_increment_rad": 0}, {"max_offset_increment_rad": np.deg2rad(21)},
    {"max_orientation_correction_rad": np.pi}, {"target_max_age_s": float("nan")},
])
def test_config_rejects_unusable_bounds(kwargs):
    with pytest.raises(ValueError):
        CopilotConfig(**kwargs)
