#!/usr/bin/env python3
"""CPU-only archived-state endpoint IK/FK and complete-collision timing probe.

Run in the original Pinocchio environment. Old captures are never retimestamped
or used as live evidence. No transport, physics application or GPU is opened.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repository',default='/home/loya2/xr_teleoperate')
    parser.add_argument('--samples',type=int,default=100)
    parser.add_argument('--collision-repeats',type=int,default=40)
    args=parser.parse_args(argv)
    if not 1<=args.samples<=300 or not 1<=args.collision_repeats<=100:
        parser.error('Bounded CPU probe requires 1..300 FK samples and 1..100 collision repeats')
    import numpy as np
    from teleop_assist.copilot.arm_collision import OriginalArmCollision
    from teleop_assist.copilot.endpoint_ik import EndpointIKValidator
    from teleop_assist.copilot.kinematics import OriginalIKValidator,sha256
    paths=[args.run/'rows.json',args.run/'live_endpoint_evidence.json']
    rows=json.loads(paths[0].read_text());evidence=json.loads(paths[1].read_text())
    original=OriginalIKValidator(args.repository)
    collision=OriginalArmCollision(args.repository)
    optimized=EndpointIKValidator(collision.model,original)
    state=evidence['state_evidence']['robot_state'];scene=evidence['state_evidence']['scene']
    source_digest=hashlib.sha256(json.dumps({'robot_state':state,'scene':scene},sort_keys=True).encode()).hexdigest()
    full=collision._q(state['joint_positions'])
    right=np.asarray(evidence['ik']['right_target_base_ee']);left=np.asarray(evidence['ik']['left_target_base_ee'])
    cold=optimized.validate_pose(right,left,full)
    if not cold['ik_valid']:raise ValueError('Recorded endpoint did not reproduce valid initial IK')
    def summary(values):
        a=np.asarray(values,float)
        return dict(n=len(a),median_ms=float(np.median(a)*1000),p95_ms=float(np.percentile(a,95)*1000),
            maximum_ms=float(a.max()*1000),minimum_ms=float(a.min()*1000))
    candidates=[r for r in rows if r.get('endpoint_accepted') is True and r.get('scenario_tick') is not None
                and r['scenario_tick']>=0 and len(r.get('actual_q43',[]))==43]
    chosen=[candidates[i] for i in np.linspace(0,len(candidates)-1,min(args.samples,len(candidates)),dtype=int)]
    samples=[]
    for row in chosen:
        named=dict(zip(row['actual_joint_names'],row['actual_q43']));current=collision._q(named)
        measured_left=np.asarray(row['actual_left_base_ee'])
        started=time.perf_counter();baseline=original.validate_pose(right,measured_left,current[collision.arm_indices]);baseline_s=time.perf_counter()-started
        started=time.perf_counter();checked=optimized.validate_pose(right,measured_left,current);optimized_s=time.perf_counter()-started
        samples.append(dict(runtime_tick=row['tick'],scenario_tick=row['scenario_tick'],
            original_ik_s=baseline_s,optimized_ik_s=optimized_s,original_ik_valid=baseline['ik_valid'],
            optimized_ik_valid=checked['ik_valid'],cache_hit=checked['cache_hit'],
            original_solver_invoked_by_optimized=checked['original_solver_invoked'],
            original_final=baseline['final'],optimized_final=checked['final'],
            reason=checked['reason'],q_initial_43=checked.get('q_initial_43'),q_solution_43=checked.get('q_solution_43')))
    optimized.clear();optimized.validate_pose(right,left,full)
    def endpoint(ik):
        return collision.validate_request(dict(batch=dict(candidates=[dict(candidate_id='archived_endpoint',
            base_ee=right.tolist(),ik_valid=ik['ik_valid'],kinematic_validation=ik)]),
            robot_state=state,scene=scene),endpoint_only=True)
    pairs=[]
    for index in range(args.collision_repeats):
        started=time.perf_counter();ik=original.validate_pose(right,left,full[collision.arm_indices]);ik_done=time.perf_counter()
        checked=endpoint(ik);done=time.perf_counter()
        baseline=dict(ik_s=ik_done-started,full_collision_s=done-ik_done,total_s=done-started,
            collision_valid=checked['candidates'][0]['collision_valid'],
            snapshot_sha256=checked['arm_collision_validation']['snapshot_sha256'])
        started=time.perf_counter();ik=optimized.validate_pose(right,left,full);ik_done=time.perf_counter()
        checked=endpoint(ik);done=time.perf_counter()
        paired=dict(ik_s=ik_done-started,full_collision_s=done-ik_done,total_s=done-started,
            collision_valid=checked['candidates'][0]['collision_valid'],cache_hit=ik['cache_hit'],
            snapshot_sha256=checked['arm_collision_validation']['snapshot_sha256'])
        pairs.append(dict(repeat=index,original=baseline,optimized=paired))
    # A new obstacle must still reject a cached numerical solution; the cache
    # does not store or skip any part of the endpoint/current-state precheck.
    obstructed=deepcopy(scene);obstacle=deepcopy(scene['obstacles'][0])
    obstacle.update(id='probe_blocking_volume',size_xyz=[4.,4.,4.],world_pose=np.eye(4).tolist())
    obstructed['obstacles'].append(obstacle)
    ik=optimized.validate_pose(right,left,full)
    blocked=collision.validate_request(dict(batch=dict(candidates=[dict(candidate_id='archived_endpoint',
        base_ee=right.tolist(),ik_valid=ik['ik_valid'],kinematic_validation=ik)]),
        robot_state=state,scene=obstructed),endpoint_only=True)
    unchanged=source_digest==hashlib.sha256(json.dumps({'robot_state':state,'scene':scene},sort_keys=True).encode()).hexdigest()
    result=dict(schema='punktnull.archived_endpoint_ik_profile.v1',scope='offline_CPU_numerical_profile_only',
        motion_authorized=False,hardware_tested=False,GPU_used=False,timestamps_refreshed=False,
        original_input_snapshots_unchanged=unchanged,manifest=optimized.manifest,
        artifact_sha256={str(p.resolve()):sha256(p) for p in paths+[Path(__file__)]},
        cold_original_fallback=cold,kinematic_samples=samples,full_endpoint_repeats=pairs,
        metrics=dict(kinematic_samples=len(samples),optimized_cache_hits=sum(r['cache_hit'] for r in samples),
            optimized_valid=sum(r['optimized_ik_valid'] for r in samples),
            original_valid=sum(r['original_ik_valid'] for r in samples),
            original_ik=summary([r['original_ik_s'] for r in samples]),
            cached_ik=summary([r['optimized_ik_s'] for r in samples]),
            original_endpoint_total=summary([r['original']['total_s'] for r in pairs]),
            cached_endpoint_total=summary([r['optimized']['total_s'] for r in pairs]),
            original_collision=summary([r['original']['full_collision_s'] for r in pairs]),
            cached_collision=summary([r['optimized']['full_collision_s'] for r in pairs])),
        fresh_obstacle_test=dict(cache_hit=ik['cache_hit'],ik_valid=ik['ik_valid'],
            collision_valid=blocked['candidates'][0]['collision_valid'],
            reason=blocked['arm_collision_validation']['reason']),
        limitations=['Archived q43/left-TCP rows support numerical FK profiling only; complete measured body poses are not present in every row.',
            'Complete-body FK/current collision/endpoint collision repeats use the exact single archived full snapshot, never synthetic fresh measurements.',
            'Endpoint residual limits remain 10 mm/10 degrees; actual-command IK remains separately bounded to 0.5 mm/0.5 degrees and 0.15 rad joint change.',
            'CPU timing is observed on this machine, not a worst-case execution-time guarantee; no live scene/source lease is granted.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['metrics'],indent=2))
    print(json.dumps(result['fresh_obstacle_test']))
    return 0 if unchanged and all(r['optimized_ik_valid'] for r in samples) and all(
        r['original']['collision_valid'] and r['optimized']['collision_valid'] for r in pairs
        ) and not result['fresh_obstacle_test']['collision_valid'] else 1


if __name__=='__main__':raise SystemExit(main())
