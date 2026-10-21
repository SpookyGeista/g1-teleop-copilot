#!/usr/bin/env python3
"""Local copilot: reviewed recording or timestamped live RGB-D mailbox.

This process never opens a robot transport. Live frames must be written as a
complete temporary NPZ then atomically renamed to frame.npz by their producer.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import signal
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from teleop_assist.copilot.dashboard import Dashboard
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--review', type=Path, help='Existing analyze_capture output')
    source.add_argument('--live-frames', type=Path, help='Producer directory with atomic frame.npz')
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--duration', type=float, default=0.)
    parser.add_argument('--no-dashboard', action='store_true', help='Live proposal worker only; another local process owns the dashboard')
    parser.add_argument('--prompt', default='red cylinder')
    parser.add_argument('--recover-transient-tracking',action='store_true',
        help='Manual-sim experiment: revalidate fresh unchanged target geometry after a visual gap')
    parser.add_argument('--complete-static-sim-preview',action='store_true',
        help='Rank 512 model candidates against same-frame complete simulator geometry; preview only')
    parser.add_argument('--xr-repo', default='/home/loya2/xr_teleoperate')
    parser.add_argument('--ik-python', default='/home/loya2/miniconda3/envs/to/bin/python')
    parser.add_argument('--model-python', default=str(ROOT/'.venv-models/bin/python'))
    parser.add_argument('--grasp-python', default=str(ROOT/'third_party/GraspGenX/.venv/bin/python'))
    parser.add_argument('--grasp-repo', type=Path, default=ROOT/'third_party/GraspGenX')
    parser.add_argument('--model-root', type=Path, default=ROOT/'models')
    args = parser.parse_args()
    if args.duration < 0 or not 0 <= args.port <= 65535:
        parser.error('Invalid duration or port')
    if args.live_frames and args.runtime is None:
        parser.error('--live-frames requires --runtime')
    if args.review and not (args.review/'latest_proposals.json').is_file():
        parser.error('Review directory has no proposals')
    if args.no_dashboard and not args.live_frames:
        parser.error('--no-dashboard requires live frames')
    if args.complete_static_sim_preview and not args.live_frames:
        parser.error('--complete-static-sim-preview requires live simulator frames')
    directory = args.review or args.runtime
    stop = False
    def interrupt(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    dashboard = None if args.no_dashboard else Dashboard(directory, port=args.port)
    if dashboard is not None:
        dashboard.start()
        print(f'Copilot dashboard: http://127.0.0.1:{dashboard.address[1]}', flush=True)
    else:
        print('Copilot proposal service: live frames, external dashboard owner', flush=True)
    perception = model = session = grasp = None
    started = time.monotonic()
    try:
        if args.live_frames:
            import cv2
            # Small tracking images do not need a machine-wide worker pool;
            # leave CPU time for the simulator and controller event receiver.
            cv2.setNumThreads(1)
            from teleop_assist.model_bridge import ModelProcess
            from teleop_assist.perception_runtime import AsyncPerception
            from teleop_assist.copilot.process import GraspProcess
            from teleop_assist.copilot.service import ProposalEngine, ProposalSession
            from teleop_assist.copilot.frames import load_frame
            from teleop_assist.copilot.mailbox import read_json
            model = ModelProcess(python=args.model_python, prompts=[args.prompt],
                                 model_root=str(args.model_root.expanduser().resolve()))
            perception = AsyncPerception(model,selected_surface_primary=args.complete_static_sim_preview)
            descriptor = ROOT/'teleop_assist/copilot/assets/dex3_right/config.json'
            grasp = GraspProcess(python=args.grasp_python, repository=str(args.grasp_repo.expanduser().resolve()),
                                 descriptor=str(descriptor), num_grasps=512 if args.complete_static_sim_preview else 256,
                                 top_k=512 if args.complete_static_sim_preview else 64)
            session = ProposalSession(directory, ProposalEngine(grasp, descriptor,
                xr_repository=args.xr_repo, ik_python=args.ik_python,audit_directory=directory,
                complete_static_sim_preview=args.complete_static_sim_preview),
                **({'recover_transient_tracking':True} if args.recover_transient_tracking else {}))
            from teleop_assist.copilot.source_evidence import source_manifest
            from teleop_assist.copilot.mailbox import atomic_json
            source=source_manifest(ROOT,'learned_proposal_service')
            source.update(prompt=args.prompt,complete_static_sim_preview=args.complete_static_sim_preview,
                          opencv_threads=cv2.getNumThreads(),
                          selected_surface_primary=args.complete_static_sim_preview)
            source['configured_model_paths'] = dict(grasp_repository=str(args.grasp_repo.expanduser().resolve()),
                model_root=str(args.model_root.expanduser().resolve()))
            # Retain the actual ready workers' identities, including the
            # loaded grasp checkpoint hashes, separately from motion authority.
            source['learned_model_manifest'] = dict(grasp=grasp.manifest, perception=model.manifest)
            atomic_json(directory/'proposal_source_manifest.json',source)
        last_signature = None
        while not stop and (not args.duration or time.monotonic()-started < args.duration):
            if session is not None:
                path = args.live_frames/'frame.npz'
                try:
                    signature = (path.stat().st_mtime_ns, path.stat().st_size)
                    if signature != last_signature:
                        frame = load_frame(path)
                        frame_received = time.monotonic()
                        selected = read_json(directory/'target_selection.json', {}).get('target_id')
                        result = perception.process({'frames': {'head': frame}}, selected_target_id=selected)
                        session.frame_diagnostics = {'received_age_s': frame_received-frame['timestamp'],
                            'perception_s': time.monotonic()-frame_received,
                            'post_perception_age_s': time.monotonic()-frame['timestamp'],
                            'perception_diagnostics': result.get('diagnostics', {})}
                        session.update(frame, result['targets'], perception.tracker.last_masks,
                            tracking_failure=perception.selected_tracking_failure,
                            feature_failure=perception.selected_feature_failure)
                        last_signature = signature
                    else:
                        session.poll()
                except FileNotFoundError:
                    session.poll()
                except (ValueError, KeyError, RuntimeError) as exc:
                    session.invalidate('camera_or_perception_failed', f'{type(exc).__name__}: {exc}')
                    session.publish()
            # Current-frame tracking must not inherit a gratuitous 30-ms
            # polling delay. Same-frame polls are cached but still revoke
            # stale/changed selections. No acquisition time is changed here.
            time.sleep(.005)
    finally:
        if dashboard is not None:dashboard.close()
        if session is not None: session.close()
        elif grasp is not None: grasp.close()
        if perception is not None: perception.close()
        if model is not None: model.close()


if __name__ == '__main__':
    main()
