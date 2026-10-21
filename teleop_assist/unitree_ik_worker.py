"""Stdin/stdout-only IK subprocess; never connects to simulation or hardware."""
import argparse
import json
import sys

from .unitree_arm_ik import UnitreeArmIK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--ee-offset', nargs=3, type=float, default=[.17, -.003, 0.])
    args = parser.parse_args()
    solver = UnitreeArmIK(args.urdf, args.ee_offset)
    print(json.dumps({'event': 'ready', 'manifest': solver.manifest}, allow_nan=False), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        result = solver.solve_request(request)
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
