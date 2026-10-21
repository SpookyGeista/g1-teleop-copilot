"""Private JSON-lines FoundationPose worker; stdout contains protocol only."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--mesh',type=Path,default=None)
    p.add_argument('--config',default='{}')
    p.add_argument('--grasp-template',default=None)
    args=p.parse_args()
    directory=args.directory.resolve()
    # Native C++ libraries can print directly to fd1, so redirect the process
    # file descriptor as well as Python stdout, retaining a protocol duplicate.
    import os
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    def send(value):
        protocol.write(json.dumps(value,allow_nan=False,
            default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
        protocol.flush()
    try:
        import numpy as np
        from teleop_assist.foundation_pose import FoundationPoseEstimator
        from teleop_assist.model_registry import PROJECT_ROOT
        config=json.loads(args.config)
        mesh=Path(args.mesh or config.pop('mesh_path',PROJECT_ROOT/'assets/cad/handle_pose/PullHandleGrasp.obj')).resolve()
        estimator=FoundationPoseEstimator(mesh,config=config,debug_dir=directory/'debug')
        template=json.loads(args.grasp_template) if args.grasp_template else None
        send({'status':'ready','backend':'foundationpose','mesh':str(mesh)})
    except BaseException:
        send({'status':'error','error':traceback.format_exc()})
        return 1
    target_identity=None
    for line in sys.stdin:
        stem=None
        try:
            packet=json.loads(line);stem=packet['stem']
            if not isinstance(stem,str) or not stem.isdecimal() or len(stem)!=8:
                raise ValueError('Invalid private frame key')
            operation=packet.get('operation') or ('register' if packet.get('reset') or
                       target_identity is None or packet.get('target_id')!=target_identity else 'track')
            if operation not in ('register','track'):
                raise ValueError('operation must be register or track')
            # A tracking state may never silently transfer to another object.
            if operation=='track' and packet.get('target_id')!=target_identity:
                raise ValueError('Tracking target changed; a fresh registration is required')
            with np.load(directory/(stem+'.npz'),allow_pickle=False) as a:
                recovery=bool(packet.get('recover_current_depth',False))
                if recovery:
                    if operation!='track':raise ValueError('Current-depth recovery requires the same registered target')
                    result=estimator.track_visible_depth(a['rgb'],a['depth'],a['intrinsics'],a['T_world_camera'],
                        timestamp=float(a['timestamp']))
                    if result.get('accepted'):
                        np.savez(directory/(stem+'.recovered-mask.npz'),mask=estimator.last_recovery_mask)
                else:
                    method=estimator.register if operation=='register' else estimator.track
                    result=method(a['rgb'],a['depth'],a['intrinsics'],a['mask'],a['T_world_camera'],template,
                                  timestamp=float(a['timestamp']))
                result.update(frame_id=int(a['frame_id']),timestamp=float(a['timestamp']),target_id=packet.get('target_id'))
            if result['accepted'] or result.get('retained_tracking_prior',False):
                target_identity=packet.get('target_id')
            else:
                target_identity=None
            send({'status':'ok','stem':stem,'result':result})
        except BaseException:
            target_identity=None
            send({'status':'error','stem':stem,'error':traceback.format_exc()})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
