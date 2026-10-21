#!/usr/bin/env python3
"""Read-only browser QA of real archived pixels with frozen offline timing.

No Isaac, inference, controller tick, or hardware connection is started. A
common documented timestamp offset puts one recorded acquisition pair into a
frozen browser-fixture clock. Original evidence is retained in evidence.json.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from teleop_assist.core import AssistConfig,AssistSession
from teleop_assist.server import AssistHTTPServer


def remap_timestamps(value,offset):
    if isinstance(value,dict):
        return {key:(float(item)+offset if (key=='timestamp' or key.endswith('_timestamp'))
                     and isinstance(item,(int,float)) and np.isfinite(item)
                     else remap_timestamps(item,offset)) for key,item in value.items()}
    if isinstance(value,list):return [remap_timestamps(item,offset) for item in value]
    return copy.deepcopy(value)


def load_frame(path,offset):
    with np.load(path,allow_pickle=False) as archive:
        frame={key:archive[key].copy() for key in archive.files}
    for key in ('timestamp','sim_time'):frame[key]=float(frame[key])
    frame['frame_id']=int(frame['frame_id']);frame['timestamp']+=offset
    frame.update(camera_id='head',depth_convention='optical_z_meters',
                 source='Actual archived Isaac RGB-D; offline visual fixture')
    return frame


class VisualFixtureSession(AssistSession):
    def snapshot(self):
        state=super().snapshot()
        state['mode']='offline_visual_fixture'
        state['offline_visual_fixture']=self.fixture_metadata
        return state

    def command(self,data):
        raise RuntimeError('Read-only visual fixture rejects every command')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--measurement-frame',type=int,required=True)
    p.add_argument('--raw-frame',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--node',default='/home/loya2/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
    a=p.parse_args();recording=a.recording.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    records=[json.loads(line) for line in (recording/'frames.jsonl').read_text().splitlines()]
    measured=next(row for row in records if row['frame_id']==a.measurement_frame)
    paths={name:recording/'rgbd'/f'{identifier:06d}.npz' for name,identifier in
           [('measurement',a.measurement_frame),('raw',a.raw_frame)]}
    with np.load(paths['raw'],allow_pickle=False) as data:raw_time=float(data['timestamp'])
    if a.raw_frame<=a.measurement_frame:raise ValueError('Raw frame must follow the measurement')
    now=time.monotonic();offset=now-(raw_time+.025)
    frames={name:load_frame(path,offset) for name,path in paths.items()}
    transformed=remap_timestamps(measured,offset)
    camera=transformed['camera'];raw=frames['raw'];reference=frames['measurement']
    if np.array_equal(raw['T_world_camera'],reference['T_world_camera']):
        raise ValueError('This reprojection probe needs actually different archived extrinsics')
    fixture={'kind':'recorded_multirate_visual_fixture','recording':str(recording),
             'measurement_frame_id':a.measurement_frame,'raw_frame_id':a.raw_frame,
             'original_acquisition_timestamp':measured['timestamp'],
             'original_publication_timestamp':measured.get('publish_timestamp'),
             'original_raw_timestamp':raw_time,'common_timestamp_offset_s':offset,
             'frozen_fixture_clock':now,'timestamp_remapping_scope':'offline browser rendering only',
             'robot_controller_started':False,'model_inference_run':False,
             'headset_verified':False,'archived_model_targets':True,
             'measurement_age_s':now-reference['timestamp'],'raw_age_s':now-raw['timestamp'],
             'files':{name:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                      for name,path in paths.items()}}
    session=VisualFixtureSession(AssistConfig(assist_enabled=False,eye_enabled=False,require_action=True),clock=lambda:now)
    session.fixture_metadata=fixture
    session.update_frame(transformed['targets'],a.measurement_frame,reference['timestamp'],camera,transformed['diagnostics'])
    controls=[json.loads(line) for line in (recording/'control.jsonl').read_text().splitlines()]
    control=controls[max(0,bisect.bisect_right([c['timestamp'] for c in controls],measured.get('publish_timestamp',measured['timestamp']))-1)]
    if control['selection']['id'] in session.selection.targets:
        session.selection.id=control['selection']['id'];session.selection.mode=control['selection']['mode']
    session.robot={'valid':False,'source':'offline fixture; no robot or controller running',
                   'ee_position':control['robot'].get('ee_position')}
    service=AssistHTTPServer(session,host='127.0.0.1',port=0)
    service.control_lease.read_only=True
    requests=[];parent_handler=service.httpd.RequestHandlerClass
    class FixtureHandler(parent_handler):
        def do_GET(self):
            requests.append({'method':'GET','path':self.path});return super().do_GET()
        def do_POST(self):
            requests.append({'method':'POST','path':self.path})
            return self.error(403,'Read-only offline visual fixture')
    service.httpd.RequestHandlerClass=FixtureHandler
    service.publish_frame(reference)
    service.publish_live_bundle({'frames':{'head':raw}})
    optical=np.diag([1.,-1.,-1.,1.])
    relative=optical@np.linalg.inv(reference['T_world_camera'])@raw['T_world_camera']@optical
    fixture['expected_surface_to_measurement_column_major']=relative.T.flatten().tolist()
    fixture['measurement_camera']=camera
    fixture['raw_camera']={'intrinsics':raw['intrinsics'].tolist(),'T_world_camera':raw['T_world_camera'].tolist()}
    fixture['measurement_target_ids']=[target['id'] for target in measured['targets']]
    (out/'evidence.json').write_text(json.dumps(fixture,indent=2))
    (out/'original_measurement.json').write_text(json.dumps(measured,indent=2))
    service.start()
    try:
        url=f'http://127.0.0.1:{service.address[1]}'
        completed=subprocess.run([a.node,str(ROOT/'tools/probe_recorded_multirate_ui.cjs'),url,str(out)],check=False)
    finally:
        service.close()
        (out/'http_requests.json').write_text(json.dumps(requests,indent=2))
    if any(request['method']!='GET' for request in requests):raise AssertionError('Browser attempted a write')
    return completed.returncode


if __name__=='__main__':raise SystemExit(main())
