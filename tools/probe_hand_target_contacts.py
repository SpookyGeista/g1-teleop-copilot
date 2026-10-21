#!/usr/bin/env python3
"""Real PhysX partner-force audit with declared oracle trajectory, no models.

Positive and collision-disabled negative controls must use separate processes.
Only the normal G1 IK and finger actuators move. Known scene coordinates choose
the evaluation trajectory and are never reported as learned perception.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--negative-control',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.output.exists():parser.error('Use a new output directory')
args.output.mkdir(parents=True)
args.enable_cameras=True;args.device='cpu'
app=AppLauncher(args).app
result={'kind':'real_physx_filtered_hand_target_contact_probe','passed':False,
        'negative_control':args.negative_control,'scope':'Oracle kinematics/sensor evaluation, no perception or headset claim'}
exit_code=1;backend=None
try:
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from teleop_assist.isaac_backend import IsaacBackend,EVALUATION_HAND_BODY_NAMES
    backend=IsaacBackend(args.output/'backend',headless=args.headless,width=640,height=480,
        scene_variant='interaction',evaluation_contact_pairs=True,
        evaluation_disable_hand_target_collisions=args.negative_control)
    target=next(t for t in backend.scene['targets'] if t['id']=='PullHandle')
    grasp=np.asarray(target['center'])+[-.056,0.,-.050]
    pregrasp=grasp+[-.09,0.,0.];pull=grasp+[-.04,0.,0.]
    quat=Rotation.from_euler('x',90,degrees=True).as_quat()
    phases=[('free',np.asarray(backend.last_state['ee_position']),None,0.,200),
            ('align',pregrasp,quat,0.,1200),('approach',grasp,quat,0.,1000),
            ('close',grasp,quat,.85,700),('pull',pull,quat,.85,1600),('open',pull,quat,0.,400)]
    initial_joint=backend.fixture_joint_states()['PullHandle']['position']
    summaries={};invalid=0;timestamp_mismatch=0
    with (args.output/'contacts.jsonl').open('w',buffering=1) as log:
        for phase,goal,orientation,grip,steps in phases:
            start=np.asarray(backend.last_state['ee_position']);start_joint=backend.fixture_joint_states()['PullHandle']['position']
            norms=[];drawer=[];per_body={name:[] for name in EVALUATION_HAND_BODY_NAMES};net_norms=[]
            for step in range(steps):
                command=start+(goal-start)*min(1.,(step+1)/(steps*.75)) if phase in ('approach','pull') else goal
                if phase=='free':backend.hold()
                state=backend.step(command,orientation,grip)
                contact=state['evaluation_hand_target_contacts']
                invalid+=int(not state['valid'])
                timestamp_mismatch+=int(contact['timestamp']!=state['timestamp'] or contact['sim_time']!=state['sim_time'])
                values=contact['per_body_normal_force_norms_N'];norms.append(max(values.values()))
                for name in per_body:per_body[name].append(values[name])
                drawer.append(state['fixture_joint_states']['PullHandle']['position'])
                net=backend.contact.data.net_forces_w.detach().cpu().numpy()[0]
                bodies=list(backend.contact.body_names)
                net_hand={name:float(np.linalg.norm(net[bodies.index(name)])) for name in EVALUATION_HAND_BODY_NAMES}
                net_norms.append(max(net_hand.values()))
                log.write(json.dumps({'phase':phase,'phase_step':step,'ee_position':state['ee_position'],
                    'ee_quaternion':state['ee_quaternion'],'command_target':command.tolist(),'requested_grip':grip,
                    'drawer_position_m_evaluation_only':drawer[-1],'evaluation_hand_target_contacts':contact,
                    'unfiltered_hand_normal_force_norms_N_comparison_only':net_hand})+'\n')
            summaries[phase]={'steps':steps,'start_drawer_position_m':start_joint,'end_drawer_position_m':drawer[-1],
                'drawer_displacement_m':start_joint-drawer[-1],'peak_partner_normal_force_N':max(norms),
                'samples_above_1e_minus4_N':sum(v>1e-4 for v in norms),
                'samples_above_1_N':sum(v>1. for v in norms),
                'peak_unfiltered_hand_normal_force_N':max(net_norms),
                'per_body_peak_partner_normal_force_N':{name:max(v) for name,v in per_body.items()},
                'per_body_normal_force_norm_integral_Ns':{name:float(sum(v)*backend.dt) for name,v in per_body.items()},
                'final_ee':state['ee_position'],'final_ee_error_m':float(np.linalg.norm(goal-state['ee_position']))}
            if phase in ('free','close','pull'):
                bundle=backend.capture_bundle()
                for name in ('head','right_wrist'):
                    frame=bundle['frames'][name];Image.fromarray(frame['rgb']).save(args.output/f'{phase}_{name}.png')
            print('PARTNER_CONTACT_PHASE',phase,json.dumps(summaries[phase]),flush=True)
    free_max=summaries['free']['peak_partner_normal_force_N']
    checks={'valid_robot_state':invalid==0,'same_step_timestamps':timestamp_mismatch==0,
            'free_space_partner_channel_zero':free_max<=1e-7,
            'seven_individually_bound_sensor_bodies':len(backend.evaluation_contact_sensors)==7}
    if args.negative_control:
        checks.update({'precise_collision_filter_declared':backend.evaluation_contact_collision_filtering['disabled']
                       and len(backend.evaluation_contact_collision_filtering['pairs'])==8,
                       'filtered_contacts_absent_entire_trajectory':max(p['peak_partner_normal_force_N'] for p in summaries.values())<=1e-7,
                       'drawer_not_opened_without_hand_contacts':abs(initial_joint-summaries['open']['end_drawer_position_m'])<.001})
    else:
        checks.update({'positive_collisions_unchanged':not backend.evaluation_contact_collision_filtering['disabled'],
                       'closing_partner_contact_measured':summaries['close']['samples_above_1e_minus4_N']>=25,
                       'pull_partner_contact_measured':summaries['pull']['samples_above_1e_minus4_N']>=25,
                       'drawer_moved_during_actual_pull':summaries['pull']['drawer_displacement_m']>.015})
    result.update(passed=all(checks.values()),checks=checks,phases=summaries,
        free_space_noise_baseline_max_N=free_max,invalid_steps=invalid,timestamp_mismatches=timestamp_mismatch,
        collision_filtering=backend.evaluation_contact_collision_filtering,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'teleop_assist/isaac_backend.py']},
        scope_limits=['Filtered partner is compound PullHandle drawer front plus U-handle, not CAD-only surface.',
            'Normal-force vectors may sum opposing contact normals; no tangential/friction measurement.',
            'Thresholds here only test the measurement channel; production controller does not consume them.',
            'Oracle trajectory, no model or headset validation, no attachments or external forces.'])
    exit_code=0 if result['passed'] else 2
except BaseException:
    result['error']=traceback.format_exc();print(result['error'],flush=True)
finally:
    (args.output/'report.json').write_text(json.dumps(result,indent=2))
    (args.output/'exit_status').write_text(str(exit_code))
    print('PARTNER_CONTACT_RESULT',json.dumps(result),flush=True)
    if backend:backend.close()
    app.close(wait_for_replicator=False,skip_cleanup=True)
raise SystemExit(exit_code)
