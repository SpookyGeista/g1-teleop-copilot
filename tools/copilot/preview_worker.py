#!/usr/bin/env python3
"""Bounded file-IPC worker: original IK and full-q43 endpoint preview only."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repository',required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output must be new')
    from teleop_assist.copilot.sim_preview import evaluate_sim_preview,read_json_file,write_json_file
    result=evaluate_sim_preview(read_json_file(args.input),args.repository)
    if any(name.startswith(('unitree_sdk2py','dds.')) for name in sys.modules):
        raise RuntimeError('Unexpected robot transport import in read-only preview worker')
    write_json_file(args.output,result)


if __name__=='__main__':main()
