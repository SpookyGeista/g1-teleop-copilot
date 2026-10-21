"""A stale assist pose must not interrupt calibrated manual teleoperation."""
from copy import deepcopy

import numpy as np

from teleop_assist.core import AssistConfig, AssistSession


def test_manual_teleop_survives_cad_expiry_but_assist_requires_rearming():
    clock = [1.0]
    session = AssistSession(AssistConfig(require_action=True), clock=lambda: clock[0])
    robot = {"valid": True, "ee_position": [.4, -.2, 1.], "ee_quaternion": [0, 0, 0, 1]}
    calibration = {"id": "cad_handle", "source": "cad_registration", "pose_timestamp": 1.,
                   "pose_max_age_s": .5, "actions": [{"id": "pull", "label": "Ziehen",
                   "kind": "linear", "grasp_position_world": [.5, -.2, 1.],
                   "approach_axis_world": [1, 0, 0], "motion_axis_world": [-1, 0, 0]}]}
    target = {"id": "handle", "bbox_px": [20, 20, 80, 80], "center_world": [.5, -.2, 1.],
              "normal_world": [-1, 0, 0], "pose_confidence": .9, "position_sigma_m": .004,
              "task_calibration": calibration}
    session.update_frame([target], 1, clock[0], {"width": 100, "height": 100})
    session.step(robot, .02)
    sequence = 0

    def send(clutch, x=0.):
        nonlocal sequence
        sequence += 1
        session.receive_input({"seq": sequence, "session_id": "manual", "source": "test",
                               "clutch": clutch, "right": {"valid": True,
                               "position": [x, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]},
                               "xr_to_world_rotation": np.eye(3).tolist()})

    send(False)
    for command in ({"type": "select", "id": "handle"}, {"type": "confirm"},
                    {"type": "action_select", "action_id": "pull"}, {"type": "action_confirm"}):
        session.command(command)
    send(False)
    send(True)
    assert not session.step(robot, .02)["hold"]
    clock[0] = 1.6
    session.update_frame([deepcopy(target)], 2, clock[0], {"width": 100, "height": 100})
    send(True, .03)
    reference = session.step(robot, .02)
    assert not reference["hold"]
    assert reference["ee_target_world"][0] > robot["ee_position"][0]
    assert session.action.reason == "cad_pose_stale"
    assert not session.rearm_required
    session.update_frame([], 3, clock[0], {"width": 100, "height": 100})
    send(True, .04)
    assert session.selection.mode == "lost"
    assert not session.step(robot, .02)["hold"]
    session.command({"type": "assist_enable", "enabled": True})
    assert session.rearm_required
    assert session.step(robot, .02)["hold"]
