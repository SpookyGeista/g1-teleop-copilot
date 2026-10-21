#!/usr/bin/env python3
"""Bounded RGB-D/kinematic capture using the user's original Unitree Isaac asset.

No original sim_main import: it starts DDS/image servers. This isolated scene
reuses its actual robot USD and actuator/camera configurations directly.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('/home/loya2/unitree_sim_isaaclab'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--stream-seconds', type=float, default=0., help='Write fresh frame.npz atomically for the live copilot')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.frames <= 100 or not 0 <= args.stream_seconds <= 3600:
        parser.error('Use new output and 1..100 frames')
    args.enable_cameras = True
    args.device = 'cpu'
    args.output.mkdir(parents=True)
    os.environ['PROJECT_ROOT'] = str(args.repo.resolve())
    app = AppLauncher(args).app
    stop_requested = False
    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.sensors import Camera
    from isaaclab.utils.math import matrix_from_quat
    from teleop_assist.copilot.mailbox import atomic_json, host_clock_id
    from teleop_assist.copilot.grasp_model import sha256
    from pxr import UsdGeom
    import omni.usd
    from PIL import Image
    try:
        robot_module = load_file('copilot_original_robot_cfg', args.repo / 'robots/unitree.py')
        cameras = load_file('copilot_original_camera_cfg', args.repo / 'tasks/common_config/camera_configs.py')
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device='cpu'))
        sim_utils.GroundPlaneCfg().func('/World/Ground', sim_utils.GroundPlaneCfg())
        light = sim_utils.DomeLightCfg(intensity=2200.)
        light.func('/World/Light', light)
        cfg = robot_module.G129_CFG_WITH_DEX3_BASE_FIX.copy()
        cfg.prim_path = '/World/Robot'
        robot = Articulation(cfg)
        # Simple bounded first task with actual contacts: rigid cylinder on a
        # table. Object pose belongs only to the simulator/evaluator, not the
        # model request or target selection logic.
        table = sim_utils.CuboidCfg(size=(.65, .8, .04), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.22, .25, .30)))
        table.func('/World/Table', table, translation=(.48, 0, .84))
        cylinder = RigidObject(RigidObjectCfg(prim_path='/World/Object',
            spawn=sim_utils.CylinderCfg(radius=.02, height=.08,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=.08),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.8, .12, .07))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(.40, -.18, .90))))
        cam_cfg = cameras.CameraBaseCfg.get_camera_config(
            prim_path='/World/Robot/d435_link/copilot_front', data_types=['rgb', 'distance_to_image_plane'])
        cam_cfg.update_latest_camera_pose = True
        camera = Camera(cam_cfg)
        sim.reset()
        q = robot.data.default_joint_pos.clone()
        # Keep the observed object clear for the first task. This explicitly
        # configured setup pose is not an AI output or a physical robot command.
        q[0, robot.joint_names.index('right_shoulder_roll_joint')] = -.35
        q[0, robot.joint_names.index('left_shoulder_roll_joint')] = .35
        q[0, robot.joint_names.index('right_shoulder_pitch_joint')] = .65
        q[0, robot.joint_names.index('left_shoulder_pitch_joint')] = .65
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        for _ in range(100):
            robot.write_data_to_sim(); sim.step(); robot.update(.005); cylinder.update(.005); camera.update(.005)
        frames = []
        stream_start = time.monotonic()
        for index in range(1000000 if args.stream_seconds else args.frames):
            if stop_requested or (args.stream_seconds and time.monotonic()-stream_start >= args.stream_seconds):
                break
            for _ in range(1 if args.stream_seconds else 10):
                robot.write_data_to_sim(); sim.step(); robot.update(.005); cylinder.update(.005); camera.update(.005)
            camera.update(.005, force_recompute=True)
            stamp = time.monotonic()
            data = camera.data
            rgb = data.output['rgb'][0, ..., :3].cpu().numpy().astype(np.uint8)
            depth = data.output['distance_to_image_plane'][0].cpu().numpy().squeeze()
            T = np.eye(4)
            T[:3, :3] = matrix_from_quat(data.quat_w_ros[0]).cpu().numpy()
            T[:3, 3] = data.pos_w[0].cpu().numpy()
            base = np.eye(4)
            base[:3, :3] = matrix_from_quat(robot.data.root_quat_w[0]).cpu().numpy()
            base[:3, 3] = robot.data.root_pos_w[0].cpu().numpy()
            obj = np.eye(4)
            obj[:3, :3] = matrix_from_quat(cylinder.data.root_quat_w[0]).cpu().numpy()
            obj[:3, 3] = cylinder.data.root_pos_w[0].cpu().numpy()
            table_pose = np.eye(4); table_pose[:3, 3] = [.48, 0, .84]
            scene = {'scene_id': 'original_unitree_usd_minimal_table', 'geometry_complete': True,
                'static': True, 'source': 'isolated_simulator_complete_collision_scene',
                'acquisition_timestamp': stamp, 'clock_id': host_clock_id(),
                'obstacles': [
                    {'id': 'table', 'type': 'box', 'size_xyz': [.65, .8, .04], 'world_pose': table_pose.tolist()},
                    {'id': 'target', 'type': 'cylinder', 'radius': .02, 'length': .08, 'world_pose': obj.tolist()},
                    {'id': 'ground', 'type': 'plane', 'normal': [0, 0, 1], 'offset': 0}],
                'allowed_static_contacts': [
                    {'link': 'left_ankle_roll_link', 'obstacle': 'ground', 'max_penetration_m': .002},
                    {'link': 'right_ankle_roll_link', 'obstacle': 'ground', 'max_penetration_m': .002}]}
            path = args.output / ('frame.next.npz' if args.stream_seconds else f'frame_{index:04d}.npz')
            np.savez_compressed(path, rgb=rgb, depth=depth, intrinsics=data.intrinsic_matrices[0].cpu().numpy(),
                T_world_camera=T, T_world_base=base, timestamp=stamp, frame_id=index,
                clock_id=host_clock_id(), calibration_id='unitree_usd_scene_observed_mount_v1',
                joint_names=np.asarray(robot.joint_names), joint_positions=robot.data.joint_pos[0].cpu().numpy(),
                joint_velocities=robot.data.joint_vel[0].cpu().numpy(),
                body_names=np.asarray(robot.body_names), body_poses=robot.data.body_state_w[0, :, :7].cpu().numpy())
            atomic_json(args.output / ('scene.json' if args.stream_seconds else f'scene_{index:04d}.json'), scene)
            if args.stream_seconds:
                path.replace(args.output/'frame.npz')
                if index % 50 == 0:
                    Image.fromarray(rgb).save(args.output / 'preview.png')
                frames = [{'path': str((args.output/'frame.npz').resolve()), 'capture_timestamp': stamp,
                           'frame_id': index}]
            else:
                Image.fromarray(rgb).save(args.output / f'frame_{index:04d}.png')
                frames.append({'path': str(path.resolve()), 'sha256': sha256(path), 'capture_timestamp': stamp})
        report = dict(status='captured' if frames else 'cancelled_before_capture', source='original_unitree_USD_and_camera_cfg_isolated_scene',
            frames=frames, robot_usd=cfg.spawn.usd_path, robot_usd_sha256=sha256(cfg.spawn.usd_path),
            joint_names=robot.joint_names, body_names=robot.body_names,
            root_world_position=robot.data.root_pos_w[0].cpu().numpy().tolist(),
            camera_world_position=camera.data.pos_w[0].cpu().numpy().tolist(),
            joint_state_finite=bool(torch.isfinite(robot.data.joint_pos).all()),
            dds_imported=any(n.startswith('unitree_sdk2py') or n.startswith('dds.') for n in sys.modules),
            limitations=['Original robot asset and camera mount, custom minimal table scene',
                         'Camera data only; no claimed XR/hardware session',
                         'Default upstream hand effort limits are not a hardware safety configuration'])
        atomic_json(args.output / 'capture_report.json', report)
        print(json.dumps(report, indent=2))
        if report['dds_imported'] or not report['joint_state_finite']:
            raise RuntimeError('Isolation or joint-state check failed')
    finally:
        # No asynchronous Replicator writer is used; waiting on its orchestrator
        # here hangs on this Isaac 5.1 installation after all outputs are saved.
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == '__main__':
    main()
