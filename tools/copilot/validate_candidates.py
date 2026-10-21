#!/usr/bin/env python3
"""CPU original-Unitree-IK endpoint validation; never publishes robot commands."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from teleop_assist.copilot.kinematics import IKValidationConfig, OriginalIKValidator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--position-tolerance-m", type=float, default=.01)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=10.)
    parser.add_argument("--joint-limit-tolerance-rad", type=float, default=1e-6)
    parser.add_argument("--max-candidates", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve previous validation evidence")
    config = IKValidationConfig(args.steps, args.position_tolerance_m,
                                 np.deg2rad(args.orientation_tolerance_deg),
                                 args.joint_limit_tolerance_rad, args.max_candidates)
    validator = OriginalIKValidator(args.repository, config)
    result = validator.validate_request(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"accepted": result["kinematic_validation"]["accepted_count"],
                      "total": result["kinematic_validation"]["total_count"],
                      "collision_validated": False, "robot_command_sent": False}))


if __name__ == "__main__":
    main()
