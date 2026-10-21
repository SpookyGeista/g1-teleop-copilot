#!/usr/bin/env python3
"""Bounded real PhysX/camera probe; no robot hardware or network commands."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--steps', type=int, default=900)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.device = 'cpu'
if args.output.exists():
    parser.error('Probe output must be a fresh directory')
args.output.mkdir(parents=True)
app = AppLauncher(args).app

try:
    import numpy as np
    from PIL import Image
    from teleop_assist.isaac_backend import IsaacBackend
    backend = IsaacBackend(args.output, headless=args.headless)
    initial = backend.capture()
    Image.fromarray(initial['rgb']).save(args.output/'initial_rgb.png')
    np.savez_compressed(args.output/'initial_rgbd.npz', **{k: v for k, v in initial.items() if isinstance(v, np.ndarray)},
                        metadata=json.dumps({k: v for k, v in initial.items() if not isinstance(v, np.ndarray)}))
    print('ASSIST_INITIAL', json.dumps(backend.last_state), flush=True)
    print('ASSIST_CAMERA', initial['T_world_camera'].tolist(), flush=True)
    start = np.array(backend.last_state['ee_position'])
    # A declared test command toward the visible lower selector's approach
    # region. Production controller must instead use RGB-D detections.
    target = np.array([.48, -.26, .985])
    trace = []
    begun = time.monotonic()
    with (args.output/'physics_trace.jsonl').open('w') as log:
        for i in range(args.steps):
            state = backend.step(target, grip=.5 if i > args.steps*.7 else 0.)
            if i % 10 == 0:
                log.write(json.dumps(state)+'\n')
                trace.append(state)
            if i % 40 == 0:
                frame = backend.capture()
            if not state['valid']:
                raise RuntimeError('Invalid real PhysX state')
    final = backend.capture()
    Image.fromarray(final['rgb']).save(args.output/'final_rgb.png')
    np.savez_compressed(args.output/'final_rgbd.npz', **{k: v for k, v in final.items() if isinstance(v, np.ndarray)},
                        metadata=json.dumps({k: v for k, v in final.items() if not isinstance(v, np.ndarray)}))
    summary = {'initial_ee': start.tolist(), 'target': target.tolist(), 'final_state': backend.last_state,
               'motion_m': float(np.linalg.norm(start-np.array(state['ee_position']))),
               'final_error_m': float(np.linalg.norm(target-np.array(state['ee_position']))),
               'wall_seconds': time.monotonic()-begun, 'steps': args.steps,
               'depth_finite_fraction': float(np.isfinite(final['depth']).mean()),
               'real_isaac_render': True, 'real_physx': True,
               'input_kind': 'explicit automated probe target, not headset/perception input'}
    (args.output/'probe_summary.json').write_text(json.dumps(summary, indent=2))
    (args.output/'evaluation_truth.json').write_text(json.dumps(backend.evaluation_truth(), indent=2))
    print('ASSIST_PROBE_RESULT', json.dumps(summary), flush=True)
    backend.close()
except BaseException:
    import traceback
    error = traceback.format_exc()
    (args.output/'probe_error.txt').write_text(error)
    print(error, flush=True)
    raise
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
