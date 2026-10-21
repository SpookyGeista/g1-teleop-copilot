#!/usr/bin/env python3
"""Stage original XR plus the pre-IK copilot, or run an explicit simulator.

--stage-only and --check do not import the original launcher, DDS or Vuer.
Runtime requires --sim and is pinned to loopback/domain 1. There is deliberately
no real-robot mode in this release. Original files are never rewritten.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from tools.copilot.instrument_xr import DEFAULT_CLIENT, BROWSER_CHUNK, stage_checkout

NODE = Path('/home/loya2/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
IK_PYTHON = Path('/home/loya2/miniconda3/envs/to/bin/python')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assist-config', type=Path, required=True)
    parser.add_argument('--sim', action='store_true', help='Mandatory; no hardware mode exists')
    parser.add_argument('--repo', type=Path, default=Path('/home/loya2/xr_teleoperate'))
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--check', action='store_true', help='Offline source/browser/actual-IK check; no DDS or server')
    parser.add_argument('--ik-python', type=Path, default=IK_PYTHON)
    parser.add_argument('--vuer-client-root', type=Path, default=DEFAULT_CLIENT)
    parser.add_argument('--network-interface', default='lo')
    parser.add_argument('--arm', default='G1_29')
    return parser


def checked_arguments(argv=None):
    parser = build_parser()
    args, upstream = parser.parse_known_args(argv)
    if not args.sim:
        parser.error('--sim is mandatory; this release cannot launch a real robot')
    if args.network_interface != 'lo':
        parser.error('This release permits only --network-interface lo')
    if args.arm != 'G1_29' or '--motion' in upstream:
        parser.error('This release requires stationary G1_29 arm control')
    config = args.assist_config.resolve()
    if not config.is_file():
        parser.error('Assist config does not exist')
    loaded = json.loads(config.read_text())
    if not isinstance(loaded, dict) or loaded.get('xr', loaded).get('mode', 'shadow') not in ('shadow', 'assist'):
        parser.error('Config must contain an xr object with mode shadow or assist')
    if loaded.get('xr', loaded).get('mode', 'shadow') == 'assist' and not (args.check or args.stage_only):
        parser.error('runtime_motion_guard_unavailable: this launcher currently permits shadow only; endpoint proposal checks do not validate the current human-pose trajectory')
    args.assist_config = config
    return args, upstream


def prepare(args):
    if args.stage:
        staging = args.stage.resolve()
    else:
        # stage_checkout requires a nonexistent destination, not a half-filled
        # temporary checkout; keeping the directory makes its patch reviewable.
        staging = Path(tempfile.mkdtemp(prefix='punktnull_xr_staging_')) / 'checkout'
    manifest = stage_checkout(args.repo, staging, client_root=args.vuer_client_root)
    return staging, manifest


def check(staging, args):
    browser = staging / 'vuer_client_build' / BROWSER_CHUNK
    if not NODE.is_file():
        raise RuntimeError('Node runtime for browser syntax verification is unavailable')
    subprocess.run([str(NODE), '--check', str(browser)], check=True, timeout=30)
    # This existing standalone probe imports only the original mathematical IK,
    # never the controller/launcher. It exercises both shadow and explicit input
    # correction with 60 bimanual solves and retains all provenance hashes.
    probe = PROJECT / 'research/masters_20261003/ai_assistance/xr_bridge/probe_original_ik.py'
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                       MPLCONFIGDIR=str(staging / 'matplotlib'),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    subprocess.run([str(args.ik_python), str(probe), '--repo', str(args.repo.resolve()),
                    '--output', str(staging / 'original_ik_check.json')],
                   check=True, env=environment, timeout=60)
    report = {'schema': 'punktnull.xr_offline_check.v1', 'browser_syntax_passed': True,
              'actual_original_bimanual_ik_probe_passed': True,
              'dds_imported': False, 'server_started': False,
              'live_browser_verified': False, 'stage': str(staging)}
    (staging / 'offline_check.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def launch_command(staging, args, upstream):
    return [str(args.ik_python), str(staging / 'teleop/teleop_hand_and_arm.py'),
            '--sim', '--network-interface', 'lo', '--arm', 'G1_29',
            '--assist-config', str(args.assist_config), *upstream]


def main(argv=None):
    args, upstream = checked_arguments(argv)
    staging, manifest = prepare(args)
    print(json.dumps({'staging': str(staging), 'source_manifest': manifest}, indent=2), flush=True)
    if args.check:
        print(json.dumps(check(staging, args), indent=2))
        return 0
    if args.stage_only:
        return 0
    # Import resolution selects staged original source/submodules before any
    # editable install; no original file needs to be changed to add the hook.
    python_paths = [str(staging), str(staging / 'teleop/televuer/src'),
                    str(staging / 'teleop/teleimager/src'), str(PROJECT)]
    environment = dict(os.environ,
        PYTHONPATH=os.pathsep.join(python_paths + [os.environ.get('PYTHONPATH', '')]),
        PUNKTNULL_XR_CLIENT_ROOT=str(staging / 'vuer_client_build'),
        PYTHONDONTWRITEBYTECODE='1')
    return subprocess.call(launch_command(staging, args, upstream), cwd=staging / 'teleop', env=environment)


if __name__ == '__main__':
    raise SystemExit(main())
