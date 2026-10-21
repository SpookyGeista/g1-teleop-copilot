"""Isolated pretrained inference process; one bounded request at a time.

Isaac's Python environment is never modified to satisfy model dependencies.
Frames cross the process boundary through private local files, not a network.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading
import time

import numpy as np


class ModelProcess:
    def __init__(self, python=None, timeout=30., startup_timeout=240., worker='model_worker.py', **config):
        self.root = Path(__file__).resolve().parents[1]
        interpreter = Path(python or self.root/'.venv-models/bin/python')
        if not interpreter.is_file():
            raise RuntimeError('Model environment missing. Run tools/setup_models.py first.')
        self.directory = tempfile.TemporaryDirectory(prefix='teleop-model-')
        self.work = Path(self.directory.name)
        self.timeout = float(timeout)
        self.lock = threading.Lock()
        self.counter = 0
        self.closed = False
        self.stderr = (self.work/'worker.log').open('w+')
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        self.process = subprocess.Popen(
            [str(interpreter), str(self.root/'teleop_assist'/worker),
             '--directory', str(self.work), '--config', json.dumps(config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, bufsize=1, cwd=self.root, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            ready = self._read(float(startup_timeout))
            if ready.get('status') != 'ready':
                raise RuntimeError('Model worker initialization failed: '+str(ready))
            self.manifest = ready
        except BaseException:
            self.close()
            raise

    def _read(self, timeout):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if self.selector.select(max(0., deadline-time.monotonic())):
                line = self.process.stdout.readline()
                if not line:
                    self.stderr.flush()
                    details = (self.work/'worker.log').read_text()[-2000:]
                    raise RuntimeError('Model worker exited: '+details)
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    continue
        raise TimeoutError('Model inference deadline exceeded; no substitute detections were produced')

    def process_frame(self, frame, bundle=None, selected_target_id=None):
        with self.lock:
            if self.closed:
                raise RuntimeError('Model worker is closed')
            self.counter += 1
            stem = f'{self.counter:08d}'
            request = self.work/(stem+'.npz')
            k = frame['intrinsics']
            if isinstance(k, dict):
                k = [[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]]
            arrays=dict(rgb=frame['rgb'], depth=frame['depth'], intrinsics=k,
                        T_world_camera=frame['T_world_camera'], frame_id=frame['frame_id'],
                        timestamp=frame['timestamp'])
            for name, sensor in (bundle or {}).get('frames',{}).items():
                if name not in ('left_wrist','right_wrist'):
                    continue
                sk=sensor['intrinsics']
                if isinstance(sk,dict):sk=[[sk['fx'],0,sk['cx']],[0,sk['fy'],sk['cy']],[0,0,1]]
                arrays.update({name+'_rgb':sensor['rgb'],name+'_depth':sensor['depth'],
                    name+'_intrinsics':sk,name+'_T_world_camera':sensor['T_world_camera'],
                    name+'_timestamp':sensor['timestamp'],name+'_frame_id':sensor.get('frame_id',frame['frame_id'])})
            scan=(bundle or {}).get('lidar')
            if scan is not None:
                payload={key:scan[key] for key in ('timestamp','sim_time','source','physics_step',
                         'T_world_lidar','points_lidar','deskew_required') if key in scan}
                arrays['lidar_json']=json.dumps(payload,allow_nan=False,
                         default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))
            np.savez(request, **arrays)
            try:
                self.process.stdin.write(json.dumps({'stem':stem,'selected_target_id':selected_target_id})+'\n')
                self.process.stdin.flush()
                result = self._read(self.timeout)
                if result.get('stem') != stem:
                    raise RuntimeError('Model result/frame mismatch')
                if result.get('status') != 'ok':
                    raise RuntimeError('Model inference failed: '+str(result.get('error')))
                value=result['result']
                mask_file=self.work/(stem+'.masks.npz')
                with np.load(mask_file,allow_pickle=False) as data:
                    value['_masks']={key:data[key].copy() for key in data.files if not key.startswith('obs__')}
                    camera_observations={}
                    for observation in value.pop('source_observations',[]):
                        camera=observation['camera_id'];identity=observation['target_id']
                        source=frame if camera=='head' else (bundle or {}).get('frames',{}).get(camera)
                        if source is None or abs(float(source['timestamp'])-observation['timestamp'])>1e-6:
                            raise RuntimeError('Observation does not match its submitted camera frame')
                        mask=np.asarray(data[observation['mask_key']],dtype=bool).copy()
                        if mask.shape!=source['depth'].shape:
                            raise RuntimeError('Observation mask/camera shape mismatch')
                        group=camera_observations.setdefault(camera,{'frame':source,'targets':[],
                            'masks':{},'head_targets':{}})
                        group['targets'].append(observation['target']);group['masks'][identity]=mask
                        group['head_targets'][identity]=observation['head_target']
                    value['_camera_observations']=camera_observations
                return value
            except BaseException:
                # A late result must never become the response to another frame.
                self.close()
                raise
            finally:
                request.unlink(missing_ok=True)
                (self.work/(stem+'.masks.npz')).unlink(missing_ok=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()
        self.directory.cleanup()
