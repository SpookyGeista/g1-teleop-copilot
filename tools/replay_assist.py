#!/usr/bin/env python3
"""Read-only browser/VR replay of recorded RGB-D and measured robot state.

No Isaac, robot model, controller, simulator ground truth or hardware endpoint is
loaded. Replayed images/depth are the original stored samples, not regenerated.
Only perception frames are replayed; additional raw camera captures are omitted.
"""
from __future__ import annotations
import argparse
import bisect
import json
import math
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from teleop_assist.core import AssistSession
from teleop_assist.server import AssistHTTPServer

class ReplaySession(AssistSession):
    def __init__(self):
        super().__init__()
        self.recorded={}
        self.recorded_action=None
    def command(self,data):
        return {'accepted':False,'reason':'Read-only recorded session; no control commands are executed'}
    def _propose(self,now):
        # Preserve the operator's recorded preview as well as confirmed IDs.
        return
    def snapshot(self):
        with self.lock:
            value=super().snapshot()
            value.update(viewer_only=True,mode='recorded_replay',input_source='recorded_session',
                         recorded=dict(self.recorded),eye_status='recorded',hand_calibrated=False)
            if self.recorded_action is not None:value['action']=dict(self.recorded_action)
            return value

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def replay_timing(frame, timebase='publication'):
    """Keep acquisition evidence intact while choosing the viewer timeline."""
    if timebase not in ('publication', 'acquisition'):
        raise ValueError('Replay timebase must be publication or acquisition')
    acquisition=float(frame['timestamp'])
    publication=frame.get('publish_timestamp')
    publication=None if publication is None else float(publication)
    if not math.isfinite(acquisition) or acquisition<0:
        raise ValueError('Recording contains an invalid acquisition timestamp')
    if publication is not None and (not math.isfinite(publication) or publication<acquisition):
        raise ValueError('Recording contains an invalid publication timestamp')
    use_publication=timebase=='publication' and publication is not None
    return {'timebase':timebase,
            'timeline_timestamp':publication if use_publication else acquisition,
            'timeline_source':'publication' if use_publication else
                              ('acquisition_fallback' if timebase=='publication' else 'acquisition'),
            'acquisition_timestamp':acquisition,'publication_timestamp':publication,
            'capture_to_publish_s':None if publication is None else publication-acquisition,
            'scope':'perception_frames_only'}

def prepare_replay_timeline(frames, timebase='publication', start=0.):
    """Frame-ID gaps are allowed; the selected timeline must not run backwards."""
    timeline=[(frame,replay_timing(frame,timebase)) for frame in frames]
    stamps=[timing['timeline_timestamp'] for _,timing in timeline]
    if any(new<old for old,new in zip(stamps,stamps[1:])):
        raise ValueError('Recording timestamps run backwards in the selected replay timebase')
    if not timeline:return []
    return [(frame,timing) for frame,timing in timeline if timing['timeline_timestamp']>=stamps[0]+start]

def replay_control_sample(controls, stamps, timing):
    """Use the same timeline as playback, including explicit historical fallback."""
    index=bisect.bisect_right(stamps,timing['timeline_timestamp'])-1
    return controls[index] if index>=0 else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8766)
    p.add_argument('--cert',type=Path);p.add_argument('--key',type=Path)
    p.add_argument('--rate',type=float,default=1.);p.add_argument('--loop',action='store_true')
    p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--timebase',choices=('publication','acquisition'),default='publication',
                   help='Publication: replay when perception became available to the operator (default; acquisition fallback for old recordings). Acquisition: sensor-time research view.')
    p.add_argument('--start',type=float,default=0.,help='Skip initial seconds in the selected --timebase')
    a=p.parse_args()
    if not .05<=a.rate<=10:p.error('--rate must be .05..10')
    if not all(math.isfinite(x) and x>=0 for x in (a.duration,a.start)):
        p.error('--duration and --start must be finite and nonnegative')
    root=a.recording.resolve()
    frames=read_jsonl(root/'frames.jsonl');controls=read_jsonl(root/'control.jsonl')
    if not frames:p.error('Recording has no frames')
    try:timeline=prepare_replay_timeline(frames,a.timebase,a.start)
    except (ValueError,KeyError,TypeError) as error:p.error(str(error))
    if not timeline:p.error('--start is past the end of the recording')
    for f,_ in timeline:
        stem=f"{int(f['frame_id']):06d}"
        if not (root/'rgbd'/(stem+'.npz')).is_file() or not (root/'rgbd'/(stem+'.jpg')).is_file():
            p.error('Recording lacks per-frame RGB-D archives; rerun without --no-record-rgbd')
    stamps=[c['timestamp'] for c in controls]
    session=ReplaySession();service=AssistHTTPServer(session,a.host,a.port,a.cert,a.key)
    stop=False
    def request_stop(*unused):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    service.start();begun=time.monotonic();sequence=0
    print(f'ASSIST_REPLAY_READY {service.scheme}://{a.host}:{service.address[1]} recording={root}',flush=True)
    try:
        while not stop:
            loop_start=time.monotonic()
            for frame_index,(f,timing) in enumerate(timeline):
                due=loop_start+(timing['timeline_timestamp']-timeline[0][1]['timeline_timestamp'])/a.rate
                while not stop and time.monotonic()<due:
                    if a.duration and time.monotonic()-begun>=a.duration:stop=True;break
                    time.sleep(min(.03,max(0.,due-time.monotonic())))
                if stop:break
                sequence+=1;stem=f"{int(f['frame_id']):06d}"
                bgr=cv2.imread(str(root/'rgbd'/(stem+'.jpg')))
                with np.load(root/'rgbd'/(stem+'.npz'),allow_pickle=False) as data:
                    depth=data['depth'].copy()
                    rgb=data['rgb'].copy() if 'rgb' in data else cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                current=time.monotonic()
                frame={'frame_id':sequence,'rgb':rgb,'depth':depth}
                service.publish_frame(frame)
                for camera_id in ('left_wrist','right_wrist'):
                    sensor_path=root/'sensors'/camera_id/(stem+'.npz')
                    if sensor_path.is_file():
                        with np.load(sensor_path,allow_pickle=False) as sensor:
                            service.publish_frame({'frame_id':sequence,'rgb':sensor['rgb'],'depth':sensor['depth']},camera_id=camera_id)
                with session.lock:
                    session.update_frame(f['targets'],sequence,current,f['camera'],f.get('diagnostics'))
                    c=replay_control_sample(controls,stamps,timing)
                    if c is not None:
                        session.robot=c['robot'];session.control=c['assist']
                        session.recorded_action=c.get('action')
                        session.selection.id=c['selection']['id'];session.selection.mode=c['selection']['mode']
                    session.recorded={'recording':root.name,'frame_id':f['frame_id'],
                                      'timestamp':f['timestamp'],'sim_time':f.get('sim_time'),
                                      **timing,'rate':a.rate,'progress':(frame_index+1)/len(timeline)}
                if a.duration and time.monotonic()-begun>=a.duration:stop=True;break
            if not a.loop:break
    finally:service.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
