"""Isolated original-asset orientation bench. Import after Isaac AppLauncher.

There is no DDS transport and no automatic approach or finger controller.
The initial pose is an explicit experiment initialization, never a robot move.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import uuid
import xml.etree.ElementTree as ET
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, ContactSensor, ContactSensorCfg
from isaaclab.utils.math import matrix_from_quat
from teleop_assist.isaac_backend import gravity_effort_budget
from .frames import robot_request
from .kinematics import ARM_JOINT_NAMES
from .mailbox import host_clock_id


def load_file(name, path, *, source_manifest=None):
    """Load the source bytes whose initialization-time fingerprint we record.

    Compile the captured bytes directly: hashing a .py file around exec_module
    would not prove that a timestamp-valid cached .pyc used those same bytes.
    This fingerprints only this file, not its imports or environment/assets.
    """
    path = Path(path).resolve(strict=True)
    read_at = time.monotonic()
    content = path.read_bytes()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(content, str(path), 'exec', dont_inherit=True), module.__dict__)
    if source_manifest is not None:
        source_manifest['files'][str(path)] = dict(
            module_name=name, sha256=hashlib.sha256(content).hexdigest(),
            source_bytes=len(content), read_at_host_monotonic=read_at,
            loaded_at_host_monotonic=time.monotonic(),
            load_method='compile_and_exec_captured_source_bytes')
    return module


class PhysXOrientationBench:
    def __init__(self, repository, xr_repository, initialization, *, camera_source='head',
                 operator_camera_source=None, support_camera_resolution=None):
        if camera_source not in ('head','right_wrist','left_wrist'):
            raise ValueError('Unknown original Unitree camera source')
        if operator_camera_source is not None and operator_camera_source not in ('head','right_wrist','left_wrist'):
            raise ValueError('Unknown operator camera source')
        self.camera_source = camera_source
        self.operator_camera_source = operator_camera_source
        import carb
        from .render_sync import verify_render_sync
        self.render_sync = verify_render_sync(carb.settings.get_settings())
        repository = Path(repository)
        self.external_source_manifest = dict(
            schema='punktnull.PhysX_external_configuration_sources.v1',
            captured_at_host_monotonic=time.monotonic(), files={},
            scope='Exact directly loaded configuration source bytes at initialization; '
                  'not transitive imports, environment, assets, or historical run verification.')
        module = load_file('copilot_bench_robot_cfg', repository/'robots/unitree.py',
            source_manifest=self.external_source_manifest)
        cameras = load_file('copilot_bench_camera_cfg', repository/'tasks/common_config/camera_configs.py',
            source_manifest=self.external_source_manifest)
        self.open_geometry_limitations = [
            'Archived copilot_sim_original_003 logged a non-existent '
            '/World/Robot/right_shoulder_roll_link/collisions/mesh_0 path. Its cause '
            'and collision-shape coverage remain unresolved; source fingerprints '
            'and zero measured contacts do not resolve this warning.']
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device='cpu'))
        cfg = module.G129_CFG_WITH_DEX3_BASE_FIX.copy(); cfg.prim_path = '/World/Robot'
        self.external_source_manifest.update(
            selected_robot_configuration='G129_CFG_WITH_DEX3_BASE_FIX',
            selected_camera_source=camera_source,
            resolved_robot_usd_path=str(cfg.spawn.usd_path))
        self.robot = Articulation(cfg)
        sim_utils.GroundPlaneCfg().func('/World/Ground', sim_utils.GroundPlaneCfg())
        light = sim_utils.DomeLightCfg(intensity=2200); light.func('/World/Light', light)
        table = sim_utils.CuboidCfg(size=(.65,.8,.04), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.22,.25,.30)))
        table.func('/World/Table', table, translation=(.48,0,.84))
        self.object = RigidObject(RigidObjectCfg(prim_path='/World/Object', spawn=sim_utils.CylinderCfg(
            radius=.02, height=.08, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.08), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.8,.12,.07))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(.4,-.18,.9))))
        self.camera_sensors = {}
        sources = tuple(dict.fromkeys((camera_source, operator_camera_source or camera_source)))
        for camera_name in sources:
            if camera_name == 'head':
                camera_cfg = cameras.CameraBaseCfg.get_camera_config(prim_path='/World/Robot/d435_link/copilot_front',
                    data_types=['rgb','distance_to_image_plane'])
                parent_link = 'd435_link'
            else:
                side = camera_name.split('_')[0]
                camera_cfg = getattr(cameras.CameraPresets,side+'_dex3_wrist_camera')()
                camera_cfg.prim_path = f'/World/Robot/{side}_hand_camera_base_link/copilot_wrist'
                camera_cfg.data_types = ['rgb','distance_to_image_plane']
                parent_link = side+'_hand_camera_base_link'
            camera_cfg.update_latest_camera_pose = True
            if support_camera_resolution is not None and camera_name != operator_camera_source:
                width,height=support_camera_resolution
                if not (type(width) is int and type(height) is int and width>=160 and height>=120):
                    raise ValueError('Bounded support camera dimensions required')
                camera_cfg.width=width;camera_cfg.height=height
            # Every requested render must acquire a new sensor buffer.
            original_period = float(camera_cfg.update_period)
            camera_cfg.update_period = 0.
            if camera_cfg.offset.convention != 'ros':
                raise ValueError('Original camera mount must use ROS optical axes')
            self.camera_sensors[camera_name] = dict(sensor=Camera(camera_cfg), parent_link=parent_link,
                mount_position=tuple(camera_cfg.offset.pos), mount_quaternion=tuple(camera_cfg.offset.rot),
                original_period=original_period, last_physics_time_s=0.)
        primary = self.camera_sensors[camera_source]
        self.camera = primary['sensor']
        self.original_camera_update_period = primary['original_period']
        self.camera_parent_link = primary['parent_link']
        self.camera_mount_position = primary['mount_position']
        self.camera_mount_quaternion = primary['mount_quaternion']
        self.external_source_manifest['operator_camera_source'] = operator_camera_source
        self.external_source_manifest['support_camera_resolution'] = support_camera_resolution
        links = ['right_'+name+'_link' for name in ('shoulder_pitch','shoulder_roll','shoulder_yaw',
            'elbow','wrist_roll','wrist_pitch','wrist_yaw','hand_palm','hand_camera_base','hand_thumb_0',
            'hand_thumb_1','hand_thumb_2','hand_index_0','hand_index_1','hand_middle_0','hand_middle_1')]
        self.contacts = [ContactSensor(ContactSensorCfg(prim_path='/World/Robot/'+name,
            update_period=0., history_length=1, filter_prim_paths_expr=['/World/Table','/World/Object'])) for name in links]
        self.sim.reset()
        self.clock_id = host_clock_id()
        self.calibration_id = 'unitree_usd_'+camera_source+'_measured_link_mount_sync_v3'
        self.arm_ids = [self.robot.joint_names.index(n) for n in ARM_JOINT_NAMES]
        self.finger_ids = [i for i,n in enumerate(self.robot.joint_names) if '_hand_' in n]
        self.initial = torch.tensor([[initialization['joint_positions'][n] for n in self.robot.joint_names]], dtype=torch.float32)
        self.robot.write_joint_state_to_sim(self.initial, torch.zeros_like(self.initial))
        self.robot.set_joint_position_target(self.initial)
        effort_xml = ET.parse(Path(xr_repository)/'assets/g1/g1_body29_hand14.urdf')
        efforts = {j.attrib['name']:float(j.find('limit').attrib['effort']) for j in effort_xml.getroot().findall('joint') if j.find('limit') is not None}
        self.total_efforts = np.minimum(self.robot.data.joint_effort_limits[0].cpu().numpy(),
            np.array([efforts[n] for n in self.robot.joint_names]))
        if not np.isfinite(self.total_efforts).all() or np.any(self.total_efforts<=0):
            raise ValueError('Missing physical effort limits')
        self.gravity_saturation_steps = 0
        self.physics_time_s = 0.
        self.physics_step_index = 0
        self.source_instance_id = 'isolated_PhysX:'+uuid.uuid4().hex
        self._command_sample = None
        self.last_camera_physics_time_s = 0.
        self.settling_contact_n = self.step(100, render=True)
        self.frame_id = 0

    def step(self, count=10, *, render=False, before_step=None):
        self._command_sample = None
        maximum = 0.
        for _ in range(count):
            gravity = self.robot.root_physx_view.get_gravity_compensation_forces()[0].cpu().numpy()
            ff, budget = gravity_effort_budget(gravity, self.total_efforts)
            self.gravity_saturation_steps += int(np.any(np.abs(gravity-ff)>1e-5))
            self.robot.write_joint_effort_limit_to_sim(torch.tensor(budget[None],dtype=torch.float32))
            self.robot.set_joint_effort_target(torch.tensor(ff[None],dtype=torch.float32))
            if before_step is not None:
                before_step()
            self.robot.write_data_to_sim(); self.sim.step(render=render)
            self.physics_time_s += .005
            self.robot.update(.005); self.object.update(.005)
            self.physics_step_index += 1
            for sensor in self.contacts:
                sensor.update(.005)
                forces = sensor.data.force_matrix_w
                if forces is None or not torch.isfinite(forces).all():
                    raise RuntimeError('Required contact measurement unavailable')
                maximum = max(maximum,float(torch.linalg.vector_norm(forces,dim=-1).max()))
        if not torch.isfinite(self.robot.data.joint_pos).all():
            raise RuntimeError('Nonfinite physical state')
        return maximum

    @staticmethod
    def pose(body):
        transform = np.eye(4)
        transform[:3,:3] = matrix_from_quat(body[3:7]).cpu().numpy()
        transform[:3,3] = body[:3].cpu().numpy()
        return transform

    def snapshot(self):
        stamp = time.monotonic()
        base = self.pose(self.robot.data.root_state_w[0,:7])
        frame = dict(joint_names=np.asarray(self.robot.joint_names),
            joint_positions=self.robot.data.joint_pos[0].cpu().numpy().copy(),
            body_names=np.asarray(self.robot.body_names),
            body_poses=self.robot.data.body_state_w[0,:,:7].cpu().numpy().copy(),
            T_world_base=base, timestamp=stamp, clock_id=self.clock_id,
            calibration_id=self.calibration_id, frame_id=self.frame_id)
        scene = self.scene_snapshot(stamp)
        tcp = np.eye(4); tcp[0,3] = .05
        state=robot_request(frame)['robot_state']
        # Derive control TCPs from the exact same normalized body matrices
        # sent to the independent checker. A second float32 conversion differs
        # by ~1e-7 and must not be hidden by weakening identity checks.
        poses = {s:np.linalg.inv(base) @ np.asarray(state['observed_body_world_poses'][s+'_wrist_yaw_link']) @ tcp
            for s in ('left','right')}
        return frame, state, scene, poses

    def scene_snapshot(self, acquisition_timestamp):
        """Read actual scene geometry in the owner's paused capture context.

        The original adapter separately copies every joint and body. Reading
        only this scene avoids a second complete robot normalization while
        retaining fresh target pose/velocity and the original capture time.
        This method neither advances physics nor produces command authority.
        """
        if (type(acquisition_timestamp) not in (int,float)
                or not np.isfinite(acquisition_timestamp)):
            raise ValueError('Finite scene acquisition timestamp required')
        table = np.eye(4); table[:3,3] = [.48,0,.84]
        target = self.pose(self.object.data.root_state_w[0,:7])
        scene = dict(scene_id='original_unitree_usd_minimal_table', geometry_complete=True, static=True,
            source='current_simulator_complete_collision_scene', acquisition_timestamp=acquisition_timestamp, clock_id=self.clock_id,
            obstacles=[dict(id='table',type='box',size_xyz=[.65,.8,.04],world_pose=table.tolist()),
                dict(id='target',type='cylinder',radius=.02,length=.08,world_pose=target.tolist()),
                dict(id='ground',type='plane',normal=[0,0,1],offset=0)],
            allowed_static_contacts=[dict(link=s+'_ankle_roll_link',obstacle='ground',max_penetration_m=.002) for s in ('left','right')])
        scene['observed_target_velocity'] = self.object.data.root_state_w[0,7:13].cpu().numpy().tolist()
        return scene

    def capture_command_sample(self):
        """Begin one sampled command transaction after camera/endpoint work."""
        from .sim_snapshot import SimSnapshotTransaction
        frame, state, scene, poses = self.snapshot()
        self._command_sample = SimSnapshotTransaction.capture(state, scene,
            physics_step=self.physics_step_index, source_instance_id=self.source_instance_id)
        return frame, state, scene, poses

    def read_command_sample(self):
        """Prove state is unchanged, retaining the original acquisition times."""
        if self._command_sample is None:
            raise ValueError('command_sample_not_captured_or_physics_advanced')
        frame, state, scene, poses = self.snapshot()
        state, scene = self._command_sample.read(state, scene,
            physics_step=self.physics_step_index, source_instance_id=self.source_instance_id)
        frame['timestamp'] = state['acquisition_timestamp']
        return frame, state, scene, poses

    def command(self, q14, *, expires_at=None, clock_id=None):
        self.last_command_checked_at = None
        # Fingers and every uncommanded joint use the state actually checked.
        target = self.robot.data.joint_pos.clone()
        if q14 is not None:
            q14 = np.asarray(q14,dtype=float)
            if q14.shape != (14,) or not np.isfinite(q14).all():
                raise ValueError('Invalid q14 command')
            target[0,self.arm_ids] = torch.tensor(q14,dtype=torch.float32)
            # Check after tensor allocation, immediately before the simulator
            # target handoff. The consumer's deadline is never extended here.
            checked_at = time.monotonic()
            if (clock_id != self.clock_id or type(expires_at) not in (int,float)
                    or not np.isfinite(expires_at) or checked_at >= expires_at):
                self.robot.set_joint_position_target(self.robot.data.joint_pos.clone())
                return False
            self.last_command_checked_at = checked_at
        self.robot.set_joint_position_target(target)
        return q14 is not None

    def camera_frame(self, *, source=None):
        source = source or self.camera_source
        return self.camera_frames((source,))[source]

    def camera_frames(self, sources):
        """Read requested sensors after one render of the same physics state."""
        sources=tuple(dict.fromkeys(sources))
        if not sources or any(source not in self.camera_sensors for source in sources):
            raise ValueError('Existing camera sources required')
        previous={source:int(self.camera_sensors[source]['sensor'].frame[0]) for source in sources}
        capture_started=time.monotonic()
        self.sim.render()
        return {source:self._camera_frame_after_render(source,capture_started,previous[source]) for source in sources}

    def _camera_frame_after_render(self, source, capture_started, previous_sensor_frame):
        selected = self.camera_sensors[source]
        camera = selected['sensor']
        camera.update(self.physics_time_s-selected['last_physics_time_s'],force_recompute=True)
        selected['last_physics_time_s']=self.physics_time_s
        if source == self.camera_source:
            self.last_camera_physics_time_s=self.physics_time_s
        sensor_frame=int(camera.frame[0])
        if sensor_frame!=previous_sensor_frame+1:
            raise RuntimeError('Camera capture reused a cached sensor buffer')
        frame, _, scene, _ = self.snapshot()
        frame['calibration_id'] = 'unitree_usd_'+source+'_measured_link_mount_sync_v3'
        frame['camera_source'] = source
        frame['rgb'] = camera.data.output['rgb'][0,:,:,:3].cpu().numpy().astype(np.uint8)
        frame['depth'] = camera.data.output['distance_to_image_plane'][0,:,:,0].cpu().numpy().copy()
        frame['intrinsics'] = camera.data.intrinsic_matrices[0].cpu().numpy().copy()
        pose = torch.cat((camera.data.pos_w[0],camera.data.quat_w_ros[0]))
        frame['camera_view_reported_T_world_camera'] = self.pose(pose)
        from .frames import mounted_camera_pose
        frame['T_world_camera'] = mounted_camera_pose(frame,selected['parent_link'],
            selected['mount_position'],selected['mount_quaternion'])
        frame['camera_extrinsics_source'] = 'measured_PhysX_parent_body_and_original_Unitree_ROS_mount'
        frame['camera_render_started_at'] = capture_started
        frame['camera_capture_time_kind'] = 'simulator_render_read_completion_not_hardware_sensor_time'
        frame['camera_render_settings_json'] = json.dumps(self.render_sync,allow_nan=False)
        frame['camera_render_synchronization'] = 'Kit_zero_delay_current_physics_state'
        frame['camera_original_update_period_s'] = selected['original_period']
        frame['camera_effective_update_period_s'] = float(camera.cfg.update_period)
        frame['camera_physics_time_s'] = self.physics_time_s
        frame['camera_sensor_frame'] = sensor_frame
        # Archived geometry for reproducible *preview* evaluation. Runtime
        # permission still requires a separate current snapshot and guard.
        frame['simulation_scene_json'] = json.dumps(scene,allow_nan=False)
        self.frame_id += 1
        return frame
