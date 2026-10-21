#!/usr/bin/env python3
"""CPU-only replay benchmark of unchanged tracking on archived neural keyframes."""
from __future__ import annotations
import argparse,copy,cProfile,io,json,pstats,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from teleop_assist.visual_tracker import RGBDMaskTracker
from teleop_assist.perception_runtime import AsyncPerception


def load(path):
    with np.load(path,allow_pickle=False) as data:frame={key:data[key].copy() for key in data.files}
    frame['timestamp']=float(frame['timestamp']);frame['frame_id']=int(frame['frame_id'])
    return frame


class NoInference:
    def process_frame(self,*args):raise AssertionError('Benchmark must never invoke a model')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keyframes',type=Path,default=Path('runs/learned_moved_drawer_validated_001'))
    parser.add_argument('--output',type=Path,default=Path('runs/cpu_tracking_profile_001'))
    parser.add_argument('--rounds',type=int,default=8)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.keyframes/'report.json').read_text());scenarios=[]
    for record in manifest['frames']:
        path=Path(record['file']);frame=load(path)
        result=json.loads((args.keyframes/(path.stem+'.json')).read_text())
        with np.load(args.keyframes/(path.stem+'_masks.npz')) as data:masks={key:data[key].copy() for key in data.files}
        current=[load(path.with_name(f'{frame["frame_id"]+offset:06d}.npz')) for offset in (1,2,3)]
        scenarios.append((frame,result,masks,current))
    default=cv2.getNumThreads();counts=list(dict.fromkeys([default,1,2,4]));samples={};outputs={};rng=random.Random(199)
    def replay(mode):
        times=[];checks=[]
        for frame,result,masks,following in scenarios:
            if mode=='tracker':
                tracker=RGBDMaskTracker();tracker.initialize(frame,result['targets'],masks)
                process=tracker.track
            else:
                runtime=AsyncPerception(NoInference(),semantic_interval=float('inf'))
                selected=next(t['id'] for t in result['targets'] if t['label']=='handle')
                runtime.selected_target_id=selected
                runtime._initialize(frame,{**copy.deepcopy(result),'_masks':masks})
                runtime.last_submit=time.monotonic()
                process=lambda current:runtime.process({'frames':{'head':current}},selected)
            for current in following:
                start=time.perf_counter();out=process(current);times.append((time.perf_counter()-start)*1000)
                checks.append({'frame':current['frame_id'],'ids':[t['id'] for t in out['targets']],
                               'centers':[t['center_world'] for t in out['targets']]})
            if mode=='async_head_confirmed':runtime.close()
        return times,checks
    for round_no in range(args.rounds+1):
        order=counts.copy();rng.shuffle(order)
        for count in order:
            cv2.setNumThreads(count)
            for mode in ('tracker','async_head_confirmed'):
                duration,checks=replay(mode)
                if round_no:
                    samples.setdefault((count,mode),[]).extend(duration)
                    outputs.setdefault(mode,{}).setdefault(count,[]).append(checks)
    result={'kind':'CPU replay without model inference','default_opencv_threads':default,
            'opencv_version':cv2.__version__,'parallel_backend':'pthreads',
            'source_keyframes':str(args.keyframes),'frame_scenarios':len(scenarios),'frames_per_scenario':3,
            'rounds':args.rounds,'gates_changed':False,'measurements':[]}
    for (count,mode),values in samples.items():
        reference=outputs[mode][counts[0]][0]
        equal=all(v==reference for v in outputs[mode][count])
        record={'threads':count,'mode':mode,'samples':len(values),'median_ms':float(np.median(values)),
                'p90_ms':float(np.percentile(values,90)),'mean_ms':float(np.mean(values)),
                'exact_output_match':equal}
        result['measurements'].append(record);print(json.dumps(record),flush=True)
    for count in (default,1):
        cv2.setNumThreads(count);profiler=cProfile.Profile();profiler.enable();replay('async_head_confirmed');profiler.disable()
        stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(25)
        (args.output/f'profile_threads_{count}.txt').write_text(stream.getvalue())
    cv2.setNumThreads(default)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
