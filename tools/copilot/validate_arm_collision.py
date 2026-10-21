#!/usr/bin/env python3
"""Original G1 full-body collision precheck for an explicit captured sim scene."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/punktnull_copilot_mpl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from teleop_assist.copilot.arm_collision import OriginalArmCollision, CollisionConfig, robot_state_from_capture
from teleop_assist.copilot.kinematics import OriginalIKValidator, ARM_JOINT_NAMES, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rerun-original-ik", action="store_true")
    parser.add_argument("--joint-step-rad", type=float, default=.01)
    parser.add_argument("--max-samples", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new path to preserve previous evidence")
    batch = json.loads(args.batch.read_text())
    scene = json.loads(args.scene.read_text())
    state = robot_state_from_capture(args.capture)
    validator = OriginalArmCollision(args.repository, CollisionConfig(
        max_joint_step_rad=args.joint_step_rad, max_samples=args.max_samples))
    if args.rerun_original_ik:
        q = validator._q(state["joint_positions"])
        validator.pin.framesForwardKinematics(validator.model, validator.data, q)
        left = validator.data.oMf[validator.model.getFrameId("left_wrist_yaw_link")].homogeneous.copy()
        tcp = np.eye(4)
        tcp[0, 3] = .05
        left = left @ tcp
        current = [state["joint_positions"][name] for name in ARM_JOINT_NAMES]
        batch = OriginalIKValidator(args.repository).validate_request({
            "batch": batch, "current_arm_q": current, "left_base_ee": left.tolist(),
            "joint_state_acquisition_timestamp": state["acquisition_timestamp"],
            "joint_state_clock_id": state["clock_id"], "joint_state_source": state["source"]})
    result = validator.validate_request({"batch": batch, "robot_state": state, "scene": scene})
    result["arm_collision_validation"]["input_files"] = {
        str(path.resolve()): sha256(path) for path in (args.batch, args.capture, args.scene)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"candidates": len(result["candidates"]),
                      "ik_valid": sum(c.get("ik_valid") is True for c in result["candidates"]),
                      "collision_valid": result["arm_collision_validation"]["accepted_count"],
                      "reason": result["arm_collision_validation"]["reason"],
                      "scope": "offline static snapshot and joint-linear path", "robot_command_sent": False}))


if __name__ == "__main__":
    main()
