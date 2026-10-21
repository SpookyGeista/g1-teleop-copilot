#!/usr/bin/env python3
"""CPU-only measurement of eager versus lazy RGB-D publication; no sockets/GPU.

Reads an actual recorded frame. This compares producer work only; compression
still costs time on the first HTTP request and is reused by later clients.
"""
import argparse
from collections import OrderedDict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from teleop_assist.server import AssistHTTPServer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgbd',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=24)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh report path')
    if not 3<=args.iterations<=100:parser.error('iterations must be 3..100')
    with np.load(args.rgbd,allow_pickle=False) as archive:
        rgb=archive['rgb'].copy();depth=archive['depth'].copy()
    cv2.setNumThreads(4)
    # Exercise the production publish method without constructing/binding HTTP.
    service=object.__new__(AssistHTTPServer)
    service.frames=OrderedDict();service.sensor_frames={};service.frame_lock=threading.Lock()
    def eager(frame):
        ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(np.asarray(frame['rgb'],dtype=np.uint8),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:raise RuntimeError('JPEG encoding failed')
        packed=gzip.compress(np.asarray(frame['depth'],dtype='<f4').tobytes(),compresslevel=1)
        with service.frame_lock:
            service.frames[frame['frame_id']]=(jpeg.tobytes(),packed)
            while len(service.frames)>8:service.frames.popitem(last=False)
    measurements={'previous_eager_publish':[],'lazy_publish':[],'lazy_first_encode_both':[],'cached_encode_both':[]}
    for n in range(args.iterations+2):
        frame={'frame_id':n,'rgb':rgb,'depth':depth}
        for name,method in [('previous_eager_publish',eager),('lazy_publish',service.publish_frame)]:
            start=time.perf_counter();method(frame);elapsed=(time.perf_counter()-start)*1000
            if n>=2:measurements[name].append(elapsed)
        published=service.frames[n]
        for name in ['lazy_first_encode_both','cached_encode_both']:
            start=time.perf_counter();jpeg=published.encoded('jpeg');packed=published.encoded('depth');elapsed=(time.perf_counter()-start)*1000
            if n>=2:measurements[name].append(elapsed)
    assert np.array_equal(np.frombuffer(gzip.decompress(packed),'<f4').reshape(depth.shape),depth,equal_nan=True)
    stats={name:{'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95))} for name,v in measurements.items()}
    report={'kind':'CPU_producer_publication_comparison','rgbd':str(args.rgbd.resolve()),
            'rgb_shape':list(rgb.shape),'iterations':args.iterations,'opencv_threads':cv2.getNumThreads(),
            'measurements':stats,'median_producer_ms_saved':stats['previous_eager_publish']['median_ms']-stats['lazy_publish']['median_ms'],
            'depth_roundtrip_exact':True,'gpu_used':False,'http_network_benchmarked':False,
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['teleop_assist/server.py','tools/benchmark_frame_publication.py']},
            'scope':'Actual captured RGBD, same JPEG85/gzip1; producer latency only. First HTTP consumer still pays encoding, later consumers reuse cached bytes.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
