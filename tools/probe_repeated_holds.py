#!/usr/bin/env python3
"""Actual PhysX repeated-hold audit; explicit free-space commands, no hardware."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--awake-articulation',action='store_true',help='Diagnostic: disable robot sleeping/stabilization before spawn')
p.add_argument('--sleep-threshold',type=float)
p.add_argument('--stabilization-threshold',type=float)
p.add_argument('--gravity-feedforward',action='store_true',help='Diagnostic continuous arm gravity torque, with remaining drive-force budget')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.output.exists():p.error('Choose a fresh output directory')
a.output.mkdir(parents=True)
a.enable_cameras=True;a.device='cpu'
app=AppLauncher(a).app
try:
    import numpy as np
    if a.awake_articulation or a.sleep_threshold is not None or a.stabilization_threshold is not None:
        import importlib
        assets_module=importlib.import_module('isaaclab.assets')
        original_articulation=assets_module.Articulation
        def awake(cfg):
            if a.awake_articulation or a.sleep_threshold is not None:
                cfg.spawn.articulation_props.sleep_threshold=0. if a.awake_articulation else a.sleep_threshold
            if a.awake_articulation or a.stabilization_threshold is not None:
                cfg.spawn.articulation_props.stabilization_threshold=0. if a.awake_articulation else a.stabilization_threshold
            return original_articulation(cfg)
        assets_module.Articulation=awake
    from teleop_assist.isaac_backend import IsaacBackend
    b=IsaacBackend(a.output/'backend',headless=a.headless,width=160,height=120,
                   wrist_cameras=False,scene_variant='interaction')
    if a.gravity_feedforward:
        gravity_getter=b.robot.root_physx_view.get_gravity_compensation_forces
        original_flush=b._flush_arm_reference
        def flush_without_position_offset(q,ee,mode):
            view=b.robot.root_physx_view
            view.get_gravity_compensation_forces=lambda: b.torch.zeros_like(gravity_getter())
            try:return original_flush(q,ee,mode)
            finally:view.get_gravity_compensation_forces=gravity_getter
        b._flush_arm_reference=flush_without_position_offset
    from pxr import PhysxSchema
    physics_attributes=[]
    for prim in b.stage.Traverse():
        if str(prim.GetPath()).startswith('/World/AssistG1') and prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            api=PhysxSchema.PhysxArticulationAPI(prim)
            physics_attributes.append({'path':str(prim.GetPath()),'sleep_threshold':api.GetSleepThresholdAttr().Get(),
                                       'stabilization_threshold':api.GetStabilizationThresholdAttr().Get()})
    trace=[];summaries={}
    def step(phase,target=None,hold=False):
        if a.gravity_feedforward:
            effort=gravity_getter()[0,b.arm_ids].clone()
            limits=b.torch.tensor(b.effort_limits[b.arm_ids],dtype=b.torch.float32)
            effort=b.torch.clamp(effort,-.8*limits,.8*limits)
            # PhysX clips drive and external torques separately. Reserve enough
            # drive budget that their sum cannot exceed the original rating.
            b.robot.write_joint_effort_limit_to_sim((limits-b.torch.abs(effort)).unsqueeze(0),joint_ids=b.arm_ids)
            b.robot.set_joint_effort_target(effort.unsqueeze(0),joint_ids=b.arm_ids)
        if hold:b.hold()
        state=b.step(b._measured_ee()[0] if target is None else target)
        trace.append({'phase':phase,'robot':state})
        return state
    def pose():return b._measured_ee()[0].copy()
    # Normal IK to a free-space pose near the fixture's operating height.
    for _ in range(1800):step('align',np.array([.42,-.26,.94]))
    for phase,cycles,active_steps,offset in [('continuous_hold',1,0,0.),
                                          ('null_resume_hold',50,1,0.),
                                          ('tiny_resume_hold',50,1,.00005),
                                          ('mm_resume_hold',20,20,.001)]:
        origin=pose();first=len(trace);before=b.arm_reference_flushes
        if phase=='continuous_hold':
            for _ in range(1000):step(phase,hold=True)
        else:
            for cycle in range(cycles):
                target=pose()+[offset*(-1 if cycle%2 else 1),0,0]
                for _ in range(active_steps):step(phase,target)
                for _ in range(80 if phase=='mm_resume_hold' else 19):step(phase,hold=True)
        rows=trace[first:]
        displacement=np.asarray([r['robot']['ee_position'] for r in rows])-origin
        summaries[phase]={'cycles':cycles,'new_flushes':b.arm_reference_flushes-before,
                          'origin_ee':origin.tolist(),'final_displacement_m':displacement[-1].tolist(),
                          'peak_displacement_m':float(np.linalg.norm(displacement,axis=1).max()),
                          'final_displacement_norm_m':float(np.linalg.norm(displacement[-1])),
                          'valid':all(r['robot']['valid'] for r in rows),
                          'max_contact_N':max(r['robot']['contact_force_max'] for r in rows)}
    report={'real_physx':True,'scope':'Repeated zero/tiny command resumes followed by explicit holds; no perception claim.',
            'phases':summaries,'physics_dt':b.dt,'joint_pose_writes':False,'external_forces':False,
            'diagnostic_sleep_and_stabilization_disabled':a.awake_articulation,
            'diagnostic_sleep_threshold':a.sleep_threshold,
            'diagnostic_stabilization_threshold':a.stabilization_threshold}
    report['diagnostic_continuous_gravity_feedforward']=a.gravity_feedforward
    report['actual_robot_articulation_attributes']=physics_attributes
    (a.output/'trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in trace))
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print('REPEATED_HOLD_REPORT',json.dumps(report),flush=True)
    b.close()
except BaseException:
    (a.output/'error.txt').write_text(traceback.format_exc());raise
finally:app.close(wait_for_replicator=False,skip_cleanup=True)
