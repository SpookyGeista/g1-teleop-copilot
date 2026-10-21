#!/usr/bin/env python3
"""Summarize measured CAD freshness and recorded control interruptions."""
import argparse,json
from pathlib import Path
import numpy as np


def quantiles(values):
    if not values:return None
    return dict(zip(('min','p50','p90','p95','p99','max'),map(float,np.percentile(values,[0,50,90,95,99,100]))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    frames=[json.loads(line) for line in (args.recording/'frames.jsonl').read_text().splitlines()]
    controls=[json.loads(line) for line in (args.recording/'control.jsonl').read_text().splitlines()]
    unique={};accepted_frames=0;result_frames=0;pose_ages=[];publish_ages=[];valid_target_frames=0;handle_frames=0
    for frame in frames:
        pose=frame['diagnostics'].get('cad_pose',{}).get('result',{})
        if pose:
            result_frames+=1
            if pose.get('accepted'):accepted_frames+=1
            unique[(pose['frame_id'],pose['timestamp'])]=pose
        for target in frame['targets']:
            if target.get('label')!='handle':continue
            handle_frames+=1
            if target.get('pose_valid'):valid_target_frames+=1
            if 'pose_timestamp' in target:
                pose_ages.append(frame['timestamp']-target['pose_timestamp'])
                if 'publish_timestamp' in frame:publish_ages.append(frame['publish_timestamp']-target['pose_timestamp'])
    accepted=sorted((p for p in unique.values() if p.get('accepted')),key=lambda p:p['timestamp'])
    stale=[row for row in controls if row['assist'].get('reason')=='cad_pose_stale']
    first=stale[0] if stale else None
    context=None
    if first:
        recent=[frame for frame in frames if frame['timestamp']<=first['timestamp']][-5:]
        context={'control_timestamp':first['timestamp'],'sim_time':first['robot'].get('sim_time'),
                 'selection':first['selection'],'recent_acquired_frames':[
                     {'frame_id':frame['frame_id'],'timestamp':frame['timestamp'],
                      'pose_frame_id':frame['diagnostics'].get('cad_pose',{}).get('result',{}).get('frame_id'),
                      'pose_timestamp':frame['diagnostics'].get('cad_pose',{}).get('result',{}).get('timestamp'),
                      'pose_age_at_control_s':first['timestamp']-frame['diagnostics'].get('cad_pose',{}).get('result',{}).get('timestamp',first['timestamp']),
                      'semantic_source_frame':frame['diagnostics'].get('semantic_keyframe',{}).get('frame_id'),
                      'publish_timestamp':frame.get('publish_timestamp')} for frame in recent]}
    report={'recording':str(args.recording),'camera_frames':len(frames),'frames_with_pose_result':result_frames,
        'frames_with_accepted_pose_result':accepted_frames,
        'accepted_result_fraction_when_result_present':accepted_frames/max(1,result_frames),
        'unique_actual_cad_measurements':len(unique),'unique_accepted_cad_measurements':len(accepted),
        'unique_actual_cad_acceptance_fraction':len(accepted)/max(1,len(unique)),
        'handle_frames':handle_frames,'handle_frames_with_valid_pose':valid_target_frames,
        'valid_pose_fraction_when_handle_present':valid_target_frames/max(1,handle_frames),
        'accepted_pose_acquisition_gap_s':quantiles(np.diff([p['timestamp'] for p in accepted]).tolist()),
        'pose_age_at_camera_acquisition_s':quantiles(pose_ages),
        'pose_age_at_publication_s':quantiles(publish_ages),
        'warm_model_compute_seconds':quantiles([p['seconds'] for p in unique.values() if p.get('operation')=='track']),
        'decimated_control_stale_records':len(stale),'first_stale_context':context,
        'limitations':['Control log is decimated; counts are not physics sample counts.',
                      'Acquisition age is not control-time age. Missing publication times are unavailable, not zero.']}
    report.update(capture_to_publication_s=quantiles([f['capture_to_publish_s'] for f in frames if 'capture_to_publish_s' in f]),
        process_start_to_publication_s=quantiles([f['publish_timestamp']-f['processing_started_timestamp'] for f in frames
            if 'publish_timestamp' in f and 'processing_started_timestamp' in f]),
        same_frame_pose_result_count=sum(bool(f['diagnostics'].get('cad_pose',{}).get('pose_result_consumed_in_same_frame')) for f in frames),
        warm_pose_deadline_timeout_frames=sum(bool(f['diagnostics'].get('cad_pose',{}).get('current_frame_wait_timed_out')) for f in frames),
        pose_wait_ms=quantiles([f['diagnostics']['cad_pose']['current_frame_pose_wait_ms'] for f in frames
            if 'current_frame_pose_wait_ms' in f['diagnostics'].get('cad_pose',{})]),
        publication_timestamp_coverage=sum('publish_timestamp' in f for f in frames)/len(frames))
    acceptance_path=args.recording/'acceptance.json'
    if acceptance_path.exists():
        checks=json.loads(acceptance_path.read_text()).get('checks',{})
        report['approach_control_validity']={name:{key:value[key] for key in ('control_valid','all_samples','invalid_control_samples','invalid_control_reasons') if key in value}
            for name,value in checks.items() if name in ('baseline_approach','assisted_approach')}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('first_stale_context','limitations')},indent=2))


if __name__=='__main__':main()
