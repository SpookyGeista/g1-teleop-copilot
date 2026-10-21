#!/usr/bin/env python3
"""Explicit SYNTHETIC gaze input for exercising the optional eye switch.

This sends simulated samples; it neither reads eyes nor measures eye accuracy.
The user still enables eyes and confirms targets in the operator interface.
"""
import argparse
import json
import math
import time
import urllib.request

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8765')
    p.add_argument('--mode',choices=('uv','head-ray'),default='uv')
    p.add_argument('--duration',type=float,default=30.)
    a=p.parse_args()
    if not 0<a.duration<=3600:p.error('Use a bounded positive duration up to 3600 seconds')
    print('SYNTHETIC GAZE: no eye hardware is used. Toggle eyes in the UI to test.',flush=True)
    start=time.monotonic();seq=0;session='synthetic-demo-'+str(time.time_ns())
    while time.monotonic()-start<a.duration:
        elapsed=time.monotonic()-start
        with urllib.request.urlopen(a.url+'/api/state',timeout=3) as r:state=json.load(r)
        sample={'source':'synthetic_gaze_demo','session_id':session,'seq':seq,
                'valid':True,'calibrated':True,'confidence':.95,'frame_id':state['frame_id']}
        if a.mode=='head-ray':
            x=.25*math.sin(elapsed*.6);y=.12*math.cos(elapsed*.5);length=math.sqrt(1+x*x+y*y)
            sample['head_ray']={'origin':[0,0,0],'direction':[x/length,y/length,-1/length]}
        elif state.get('targets'):
            target=state['targets'][int(elapsed//3)%len(state['targets'])];b=target['bbox_px'];c=state['camera']
            sample['uv']=[(b[0]+b[2])/(2*c['width']),(b[1]+b[3])/(2*c['height'])]
        else:time.sleep(.1);continue
        request=urllib.request.Request(a.url+'/api/gaze',data=json.dumps(sample).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=3) as response:response.read()
        seq+=1;time.sleep(.04)
    print(f'Sent {seq} synthetic samples; input will expire automatically.',flush=True)

if __name__=='__main__':main()
