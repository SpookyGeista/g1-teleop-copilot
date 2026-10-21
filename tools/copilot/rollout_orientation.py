#!/usr/bin/env python3
"""Controlled PhysX comparison from an explicitly initialized learned-grasp pose.

The human input is a scripted 12-degree orientation error, not a user study.
A fresh original-IK/full-robot guard validates each emitted joint command.
This tool never starts DDS, XR, or any real-robot transport.
"""
from __future__ import annotations
import argparse, importlib.util, json, os
from pathlib import Path
import sys,time
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialization',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ticks',type=int,default=70)
    parser.add_argument('--repo',type=Path,default=Path('/home/loya2/unitree_sim_isaaclab'))
    parser.add_argument('--xr-repo',default='/home/loya2/xr_teleoperate')
    parser.add_argument('--ik-python',default='/home/loya2/miniconda3/envs/to/bin/python')
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    if args.output.exists() or not 20<=args.ticks<=300:parser.error('New output and 20..300 ticks required')
    setup=json.loads(args.initialization.read_text());candidate=json.loads(args.candidate.read_text())
    if setup.get('accepted') is not True:parser.error('Reviewed numerical test initialization required')
    args.output.mkdir(parents=True)
    args.enable_cameras=True;args.device='cpu'
    os.environ['PROJECT_ROOT']=str(args.repo.resolve())
    app=AppLauncher(args).app
    import numpy as np
    import torch
    from PIL import Image
    from scipy.spatial.transform import Rotation
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation,RigidObject,RigidObjectCfg
    from isaaclab.sensors import Camera,ContactSensor,ContactSensorCfg
    from isaaclab.utils.math import matrix_from_quat
    from teleop_assist.copilot.motion_process import MotionProcess
    from teleop_assist.copilot.guard import command_is_current
    from teleop_assist.copilot.frames import robot_request
    from teleop_assist.copilot.kinematics import ARM_JOINT_NAMES
    from teleop_assist.copilot.controller import OrientationCopilot
    from teleop_assist.copilot.types import OperatorSample,ProposalBatch,GraspCandidate,CopilotRequest
    from teleop_assist.copilot.mailbox import atomic_json,host_clock_id
    from teleop_assist.isaac_backend import gravity_effort_budget
    worker=None
    try:
        module=load_file('orientation_original_robot_cfg',args.repo/'robots/unitree.py')
        cameras=load_file('orientation_original_camera_cfg',args.repo/'tasks/common_config/camera_configs.py')
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device='cpu'))
        cfg=module.G129_CFG_WITH_DEX3_BASE_FIX.copy();cfg.prim_path='/World/Robot'
        robot=Articulation(cfg)
        sim_utils.GroundPlaneCfg().func('/World/Ground',sim_utils.GroundPlaneCfg())
        light=sim_utils.DomeLightCfg(intensity=2200);light.func('/World/Light',light)
        table=sim_utils.CuboidCfg(size=(.65,.8,.04),collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.22,.25,.30)))
        table.func('/World/Table',table,translation=(.48,0,.84))
        cylinder=RigidObject(RigidObjectCfg(prim_path='/World/Object',spawn=sim_utils.CylinderCfg(radius=.02,height=.08,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),mass_props=sim_utils.MassPropertiesCfg(mass=.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.8,.12,.07))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(.4,-.18,.9))))
        cam_cfg=cameras.CameraBaseCfg.get_camera_config(prim_path='/World/Robot/d435_link/copilot_front',data_types=['rgb'])
        cam_cfg.update_latest_camera_pose=True;camera=Camera(cam_cfg)
        contacts={name:ContactSensor(ContactSensorCfg(prim_path='/World/Robot/'+name,update_period=0.,history_length=1,
            filter_prim_paths_expr=['/World/Table','/World/Object'])) for name in (
                'right_shoulder_pitch_link','right_shoulder_roll_link','right_shoulder_yaw_link','right_elbow_link',
                'right_wrist_roll_link','right_wrist_pitch_link','right_wrist_yaw_link','right_hand_palm_link','right_hand_camera_base_link',
                'right_hand_thumb_0_link','right_hand_thumb_1_link','right_hand_thumb_2_link',
                'right_hand_index_0_link','right_hand_index_1_link','right_hand_middle_0_link','right_hand_middle_1_link')}
        sim.reset()
        initial=torch.tensor([[setup['joint_positions'][n] for n in robot.joint_names]],dtype=torch.float32)
        arm_ids=[robot.joint_names.index(n) for n in ARM_JOINT_NAMES]
        right_id=robot.body_names.index('right_wrist_yaw_link')
        clock=host_clock_id();cal='unitree_usd_scene_observed_mount_v1'
        raw=np.asarray(setup['raw_right_base_ee']);left=np.asarray(setup['left_base_ee']);goal=np.asarray(setup['goal_base_ee'])
        finger_ids=[i for i,n in enumerate(robot.joint_names) if '_hand_' in n]
        limit_xml=ET.parse(Path(args.xr_repo)/'assets/g1/g1_body29_hand14.urdf')
        effort_by_name={j.attrib['name']:float(j.find('limit').attrib['effort']) for j in limit_xml.getroot().findall('joint') if j.find('limit') is not None}
        original_efforts=robot.data.joint_effort_limits[0].cpu().numpy().copy()
        total_efforts=np.minimum(original_efforts,np.array([effort_by_name[n] for n in robot.joint_names]))
        if not np.isfinite(total_efforts).all() or np.any(total_efforts<=0):raise RuntimeError('Missing physical effort limits')
        ff_saturated_steps=0
        worker=MotionProcess(args.ik_python,args.xr_repo)
        atomic_json(args.output/'motion_guard_manifest.json',worker.manifest)
        def step(n,render=False):
            nonlocal ff_saturated_steps
            maximum=0.
            for _ in range(n):
                gravity=robot.root_physx_view.get_gravity_compensation_forces()[0].cpu().numpy()
                feedforward,budget=gravity_effort_budget(gravity,total_efforts)
                ff_saturated_steps+=int(np.any(np.abs(gravity-feedforward)>1e-5))
                robot.write_joint_effort_limit_to_sim(torch.tensor(budget[None],dtype=torch.float32))
                robot.set_joint_effort_target(torch.tensor(feedforward[None],dtype=torch.float32))
                robot.write_data_to_sim();sim.step(render=render);robot.update(.005);cylinder.update(.005)
                for sensor in contacts.values():
                    sensor.update(.005)
                    forces=sensor.data.force_matrix_w
                    if forces is None or not torch.isfinite(forces).all():
                        raise RuntimeError('Required right-arm contact measurement is missing or nonfinite')
                    maximum=max(maximum,float(torch.linalg.vector_norm(forces,dim=-1).max()))
            return maximum
        def pose_from_body(body):
            T=np.eye(4);T[:3,:3]=matrix_from_quat(body[3:7]).cpu().numpy();T[:3,3]=body[:3].cpu().numpy();return T
        def snapshot():
            stamp=time.monotonic()
            base=pose_from_body(robot.data.root_state_w[0,:7])
            frame={'joint_names':np.asarray(robot.joint_names),'joint_positions':robot.data.joint_pos[0].cpu().numpy().copy(),
                   'body_names':np.asarray(robot.body_names),'body_poses':robot.data.body_state_w[0,:,:7].cpu().numpy().copy(),
                   'T_world_base':base,'timestamp':stamp,'clock_id':clock}
            state=robot_request(frame)['robot_state']
            table_pose=np.eye(4);table_pose[:3,3]=[.48,0,.84]
            object_pose=pose_from_body(cylinder.data.root_state_w[0,:7])
            scene={'scene_id':'original_unitree_usd_minimal_table','geometry_complete':True,'static':True,
                'source':'current_simulator_complete_collision_scene','acquisition_timestamp':stamp,'clock_id':clock,
                'obstacles':[{'id':'table','type':'box','size_xyz':[.65,.8,.04],'world_pose':table_pose.tolist()},
                    {'id':'target','type':'cylinder','radius':.02,'length':.08,'world_pose':object_pose.tolist()},
                    {'id':'ground','type':'plane','normal':[0,0,1],'offset':0}],
                'allowed_static_contacts':[{'link':side+'_ankle_roll_link','obstacle':'ground','max_penetration_m':.002} for side in ('left','right')]}
            wrist=pose_from_body(robot.data.body_state_w[0,right_id,:7]);tcp=np.eye(4);tcp[0,3]=.05
            ee=np.linalg.inv(base)@wrist@tcp
            return stamp,state,scene,ee
        def photograph(name):
            sim.render();camera.update(.005,force_recompute=True)
            Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy().astype(np.uint8)).save(args.output/name)
        summary=[]
        for mode in ('manual','assist'):
            object_initial=cylinder.data.default_root_state.clone()
            cylinder.write_root_pose_to_sim(object_initial[:,:7]);cylinder.write_root_velocity_to_sim(torch.zeros_like(object_initial[:,7:13]))
            robot.write_joint_state_to_sim(initial,torch.zeros_like(initial));robot.set_joint_position_target(initial)
            settling_contact=step(100,render=True);photograph(mode+'_initial.png')
            controller=OrientationCopilot();rows=[]
            chosen=GraspCandidate(candidate['candidate_id'],goal,candidate['model_score'],True,True,True,candidate['source'])
            for i in range(args.ticks):
                stamp,state,scene,ee=snapshot()
                # This experiment uses the freshly observed simulator object pose
                # solely to check stationarity/collision, not to generate grasps.
                delta=float(np.linalg.norm(np.asarray(scene['obstacles'][1]['world_pose'])[:3,3]-[.4,-.18,.9]))
                static=delta<.001
                batch=ProposalBatch('evaluation_object','learned_pipeline_003_frozen_candidate',stamp,clock,cal,(chosen,),static,static)
                request=(CopilotRequest(select_candidate_id=chosen.candidate_id) if i==0 else
                         CopilotRequest(confirm=True) if i==1 else CopilotRequest(assist_held=mode=='assist'))
                outcome=controller.step(OperatorSample(left,raw,stamp,clock,True),batch,now=stamp,clock_id=clock,calibration_id=cal,request=request)
                checked=worker.call('validate',left_base_ee=outcome.left_base_ee.tolist(),right_base_ee=outcome.right_base_ee.tolist(),
                    robot_state=state,scene=scene,acquisition_timestamp=stamp,clock_id=clock,tracking_valid=True)
                accepted=command_is_current(checked,now=time.monotonic(),clock_id=clock) and not outcome.input_hold_required
                target=robot.data.joint_pos.clone()
                measured_before_command=target.clone()
                if accepted:target[0,arm_ids]=torch.tensor(checked['q14_command'],dtype=torch.float32)
                robot.set_joint_position_target(target)
                contact=step(10)
                _,_,_,actual=snapshot()
                error=float(np.rad2deg(Rotation.from_matrix(actual[:3,:3]@goal[:3,:3].T).magnitude()))
                rows.append({'tick':i,'mode':mode,'guard_allowed':accepted,'guard_reason':checked['reason'],
                    'guard_elapsed_s':checked['elapsed_s'],'controller_reason':outcome.reason,
                    'orientation_error_deg':error,'translation_error_m':float(np.linalg.norm(actual[:3,3]-raw[:3,3])),
                    'assistance_applied':outcome.applied,'max_table_target_contact_force_n':contact,
                    'max_arm_speed_rad_s':float(torch.max(torch.abs(robot.data.joint_vel[0,arm_ids]))),
                    'finger_command_matches_checked_state':bool(torch.equal(target[0,finger_ids],measured_before_command[0,finger_ids])),
                    'max_finger_drift_from_initial_rad':float(torch.max(torch.abs(robot.data.joint_pos[0,finger_ids]-initial[0,finger_ids]))),
                    'observed_object_displacement_m':delta})
                if not torch.isfinite(robot.data.joint_pos).all():raise RuntimeError('Nonfinite physical state')
                if contact>2.:break
            photograph(mode+'_final.png')
            atomic_json(args.output/(mode+'_rows.json'),rows)
            summary.append({'mode':mode,'ticks':len(rows),'accepted_commands':sum(r['guard_allowed'] for r in rows),
                'initial_error_deg':rows[0]['orientation_error_deg'],'final_error_deg':rows[-1]['orientation_error_deg'],
                'final_median_error_deg':float(np.median([r['orientation_error_deg'] for r in rows[-10:]])),
                'max_contact_force_n':max(r['max_table_target_contact_force_n'] for r in rows),
                'settling_max_right_arm_contact_force_n':settling_contact,
                'max_translation_error_m':max(r['translation_error_m'] for r in rows),
                'guard_median_s':float(np.median([r['guard_elapsed_s'] for r in rows])),
                'all_finger_commands_match_checked_state':all(r['finger_command_matches_checked_state'] for r in rows),
                'max_finger_drift_from_initial_rad':max(r['max_finger_drift_from_initial_rad'] for r in rows)})
        report={'status':'complete','modes':summary,'candidate_id':candidate['candidate_id'],
            'dds_imported':any(n.startswith('unitree_sdk2py') or n.startswith('dds.') for n in sys.modules),
            'real_robot_tested':False,'scripted_operator':True,'scene_collision_geometry':'simulator_known',
            'physical_contact_sensor_scope':'16 right-arm/hand/wrist-camera rigid bodies versus table and target only',
            'upstream_PhysX_self_collisions_enabled':bool(cfg.spawn.articulation_props.enabled_self_collisions),
            'actuation':'original position drives plus native PhysX gravity feedforward on every physics step',
            'combined_effort_budget':'min(original_sim_limit, original_URDF_limit); PD budget reduced by abs(feedforward)',
            'gravity_saturation_physics_steps':ff_saturated_steps,
            'initialization':'numerically_checked_pose_written_explicitly; no_home_to_target_approach_claim',
            'source_candidate':str(args.candidate.resolve()),'source_initialization':str(args.initialization.resolve())}
        atomic_json(args.output/'report.json',report);print(json.dumps(report,indent=2),flush=True)
    finally:
        if worker:worker.close()
        app.close(wait_for_replicator=False,skip_cleanup=True)

if __name__=='__main__':main()
