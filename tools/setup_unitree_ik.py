#!/usr/bin/env python3
"""Audit optional local Unitree IK runtime. Never installs into an existing env."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop_assist.unitree_arm_ik import AsyncUnitreeArmIK, default_worker_python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, default=None,
                        help='Existing Python with pinocchio.casadi and casadi')
    parser.add_argument('--urdf', type=Path, default=Path(
        '/home/loya2/WholeBodyTele/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with AsyncUnitreeArmIK(args.urdf, python=args.python or default_worker_python()) as worker:
        report = {'ready': True, **worker.manifest}
    serialized = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end='')


if __name__ == '__main__':
    main()
