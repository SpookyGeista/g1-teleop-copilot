"""Real Isaac/PhysX backend. Instantiate only AFTER Isaac AppLauncher.

No stage coordinates are supplied to perception. ``evaluation_truth`` is a
separate, explicitly test-only endpoint. Public end-effector quaternions use
XYZW; Isaac internally uses WXYZ. RGB-D follows the optical +Z convention.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import numpy as np

RETAINED = Path(os.environ.get('PUNKTNULL_G1_REPO', '/home/loya2/WholeBodyTele/GR00T-WholeBodyControl')).expanduser().resolve()
HAND_URDF = RETAINED / 'decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf'
EVALUATION_HAND_BODY_NAMES = tuple('right_hand_'+name+'_link' for name in (
    'thumb_0', 'thumb_1', 'thumb_2', 'middle_0', 'middle_1', 'index_0', 'index_1'))


def filtered_normal_force_vector(matrix):
    """One environment, one sensor body, one partner filter; already newtons."""
    value = np.asarray(matrix, dtype=float)
    if value.shape != (1, 1, 1, 3) or not np.isfinite(value).all():
        raise RuntimeError(f'Expected finite (1,1,1,3) partner force matrix, got {value.shape}')
    return value[0, 0, 0].copy()


def gravity_effort_budget(gravity, ratings):
    """Reserve physical drive capacity so |drive + gravity FF| <= rating."""
    gravity, ratings = np.asarray(gravity, float), np.asarray(ratings, float)
    if (gravity.shape != ratings.shape or not np.isfinite(gravity).all()
            or not np.isfinite(ratings).all() or np.any(ratings <= 0)):
        raise ValueError('Invalid arm gravity forces or total effort ratings')
    feedforward = np.clip(gravity, -.8*ratings, .8*ratings)
    return feedforward, ratings-np.abs(feedforward)


class IsaacBackend:
    dt = .005
    ee_offset = np.array([.17, -.003, 0.])

    def __init__(self, output_dir, headless=True, seed=0, width=960, height=720,
                 scene_variant='cabinet', wrist_cameras=True, wrist_width=480, wrist_height=360,
                 evaluation_contact_pairs=False, evaluation_disable_hand_target_collisions=False,
                 start_pose='fixture', arm_solver='dls', arm_solver_sync=False):
        if start_pose not in ('fixture', 'reference'):
            raise ValueError('Unknown G1 start pose')
        self.start_pose = start_pose
        self.reference_definition = None
        if arm_solver not in ('dls', 'unitree'):
            raise ValueError('Unknown arm solver')
        self.arm_solver = arm_solver
        self.arm_solver_sync = bool(arm_solver_sync)
        self._unitree_ik = None
        self._unitree_goal = None
        self._unitree_result_at = -float('inf')
        self._unitree_submit_step = -1000
        self._arm_solver_diagnostics = {'name': arm_solver, 'status': 'starting',
                                        'accepted': False, 'solve_ms': 0., 'request_id': None}
        if evaluation_disable_hand_target_collisions and not evaluation_contact_pairs:
            raise ValueError('Negative contact control requires evaluation_contact_pairs=True')
        import torch
        import omni.usd
        import omni.replicator.core as rep
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from pxr import Gf, UsdGeom
        from .scene import build_cabinet_scene

        self.torch, self.rep, self.Gf = torch, rep, Gf
        torch.set_num_threads(1)
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.width, self.height = int(width), int(height)
        self.frame_id, self.step_index = 0, 0
        self.closed = False
        self.last_capture = None
        self.last_state = None
        self.command_rejections = 0
        self.hold_reason = None
        self._arm_reference_mode = 'active'
        self._external_hold_requested = False
        self.arm_reference_flushes = 0
        self._arm_hold_gravity_offset = np.zeros(7)
        self._arm_hold_gravity_forces = np.zeros(7)
        self._arm_hold_gravity_saturated = False
        self._arm_gravity_feedforward = np.zeros(7)
        self._arm_drive_effort_budget = np.zeros(7)
        self.last_grip = 0.
        self.evaluation_contact_pairs = bool(evaluation_contact_pairs)
        self.evaluation_disable_hand_target_collisions = bool(evaluation_disable_hand_target_collisions)
        self.evaluation_contact_sensors = {}
        self.evaluation_contact_collision_filtering = {'disabled': False, 'method': 'none', 'pairs': []}
        rep.orchestrator.set_capture_on_play(False)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.dt, render_interval=10, device='cpu', enable_scene_query_support=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.85, dynamic_friction=.7)))
        self.sim.set_setting('/app/runLoops/main/rateLimitEnabled', False)
        self.stage = omni.usd.get_context().get_stage()
        self.scene = build_cabinet_scene(self.stage, seed, variant=scene_variant,
                                         asset_dir=self.output/'cad_assets')
        self.scene_variant = scene_variant
        self.camera_streams = {}
        self.wrist_cameras = bool(wrist_cameras)
        self.last_bundle = None
        spec = importlib.util.spec_from_file_location('_assist_original_g1_cfg', RETAINED/'gear_sonic/envs/manager_env/robots/g1.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.G1_CYLINDER_MODEL_12_DEX_CFG.copy()
        cfg.prim_path = '/World/AssistG1'
        cfg.spawn.asset_path = str(HAND_URDF)
        cfg.spawn.usd_dir = str(self.output/'robot_cache')
        cfg.spawn.usd_file_name = 'g1_hands_fixed_fixture.usd'
        cfg.spawn.fix_base = True
        cfg.spawn.force_usd_conversion = True
        cfg.spawn.activate_contact_sensors = True
        cfg.spawn.articulation_props.enabled_self_collisions = True
        cfg.spawn.articulation_props.solver_position_iteration_count = 12
        cfg.spawn.articulation_props.solver_velocity_iteration_count = 4
        cfg.init_state.pos = (0., 0., .80)
        cfg.init_state.joint_pos.pop('.*_elbow_joint', None)
        cfg.init_state.joint_pos['left_elbow_joint'] = .6
        cfg.init_state.joint_pos.update({'right_shoulder_pitch_joint': -.35,
                                         'right_shoulder_roll_joint': -.25,
                                         'right_elbow_joint': .95})
        if start_pose == 'reference':
            from .retained_reference import retained_zero_reference, zero_reference_root_height
            self.reference_definition = retained_zero_reference(RETAINED, HAND_URDF)
            cfg.init_state.joint_pos = {**self.reference_definition['joint_positions'],
                                       **self.reference_definition['finger_positions']}
            plate = next(box for box in self.scene['collision_boxes'] if box['name'] == 'PelvisMount/Base')
            support_top = plate['center'][2]+plate['size'][2]/2
            height = zero_reference_root_height(HAND_URDF, support_top)
            cfg.init_state.pos = (0., 0., height)
            self.reference_definition.update(root_height_m=height, support_top_m=support_top,
                                             foot_collision_clearance_m=.003)
        # Dedicated manipulation gains, not the SONIC policy gains. The same
        # physical effort/velocity/position limits remain in force.
        cfg.actuators['arms'].stiffness = 100.
        cfg.actuators['arms'].damping = 8.
        cfg.actuators['waist'].stiffness = 180.
        cfg.actuators['waist'].damping = 12.
        cfg.actuators['waist_yaw'].stiffness = 180.
        cfg.actuators['waist_yaw'].damping = 12.
        limits = {}
        for j in ET.parse(HAND_URDF).getroot().findall('joint'):
            if j.attrib['type'] == 'revolute':
                l = j.find('limit')
                limits[j.attrib['name']] = {k: float(l.attrib[k]) for k in ('lower', 'upper', 'effort', 'velocity')}
        finger_names = [n for n in limits if '_hand_' in n]
        cfg.actuators['fingers'] = ImplicitActuatorCfg(
            joint_names_expr=finger_names,
            effort_limit_sim={n: limits[n]['effort'] for n in finger_names},
            velocity_limit_sim={n: limits[n]['velocity'] for n in finger_names},
            stiffness=6., damping=.3, armature=.00015)
        # SONIC's cylinder model has different hip ratings than this hand
        # URDF. Resolve every actuator against the actually imported asset.
        for actuator in cfg.actuators.values():
            matched = [n for n in limits if any(re.fullmatch(pattern, n) for pattern in actuator.joint_names_expr)]
            actuator.effort_limit_sim = {n: limits[n]['effort'] for n in matched}
            actuator.velocity_limit_sim = {n: limits[n]['velocity'] for n in matched}
        self.robot = Articulation(cfg)
        self.contact = ContactSensor(ContactSensorCfg(
            prim_path='/World/AssistG1/right_.*', update_period=0., history_length=1,
            force_threshold=1., track_pose=False))
        if self.evaluation_contact_pairs:
            self._configure_evaluation_contact_pairs(ContactSensor, ContactSensorCfg)
        self._add_camera('head', '/World/AssistEyeCamera', self.width, self.height)
        # Preserve public single-camera fields used by previous recordings.
        head = self.camera_streams['head']
        self.camera_path, self.camera_op = head['path'], head['op']
        self.intrinsics = head['intrinsics']
        self.product = head['product']
        self.rgb_annotator, self.depth_annotator = head['rgb'], head['depth']
        if self.wrist_cameras:
            from .sensor_geometry import wrist_camera_extrinsic
            for side in ('left', 'right'):
                self._add_camera(side+'_wrist', '/World/Assist'+side.title()+'WristCamera',
                                 int(wrist_width), int(wrist_height),
                                 source_link=side+'_wrist_yaw_link',
                                 T_link_camera=wrist_camera_extrinsic(side))
        # Show robot and working side of the panel; a view from +X hides the
        # operator-facing side behind the solid cabinet back.
        self.sim.set_camera_view((-1.2, -2., 1.65), (.25, -.03, 1.))
        self.sim.reset()
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos.clone(),
                                           self.robot.data.default_joint_vel.clone())
        self.robot.reset()
        self.robot.update(self.dt)
        self.names = list(self.robot.joint_names)
        if len(self.names) != 43 or not self.robot.is_fixed_base:
            raise RuntimeError('Expected the explicit fixed-base 43-joint G1 hand fixture')
        self.arm_names = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
                          'right_shoulder_yaw_joint', 'right_elbow_joint',
                          'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        self.arm_ids = [self.names.index(n) for n in self.arm_names]
        self.finger_ids = [self.names.index(n) for n in finger_names if n.startswith('right_')]
        self.body_id = list(self.robot.body_names).index('right_wrist_yaw_link')
        self.torso_id = list(self.robot.body_names).index('torso_link')
        for stream in self.camera_streams.values():
            if stream.get('source_link'):
                stream['body_id'] = list(self.robot.body_names).index(stream['source_link'])
        import omni.physics.tensors as physics_tensors
        self.fixture_physics = physics_tensors.create_simulation_view('numpy')
        self.fixture_physics.set_subspace_roots('/')
        self.fixture_body_views = {joint['id']: self.fixture_physics.create_rigid_body_view(joint['body_path'])
                                   for joint in self.scene['fixture_joints']}
        if any(view.count != 1 for view in self.fixture_body_views.values()):
            raise RuntimeError('Fixture rigid-body telemetry view did not bind exactly one body')
        self.joint_target = self.robot.data.default_joint_pos.clone()
        self.lower = np.array([limits[n]['lower'] for n in self.names])
        self.upper = np.array([limits[n]['upper'] for n in self.names])
        self.effort_limits = np.array([limits[n]['effort'] for n in self.names])
        self.arm_total_effort_limits = np.minimum(self.effort_limits[self.arm_ids],
            self.robot.data.joint_effort_limits[0, self.arm_ids].detach().cpu().numpy())
        self.velocity_limits = np.array([limits[n]['velocity'] for n in self.names])
        self._target_ee = self._measured_ee()[0].copy()
        if self.arm_solver == 'unitree':
            from .unitree_arm_ik import AsyncUnitreeArmIK
            self._unitree_ik = AsyncUnitreeArmIK(HAND_URDF, ee_offset=self.ee_offset,
                                              log_path=self.output/'unitree_ik.log')
        self._update_camera()
        for _ in range(12):
            self.sim.render()
        for _ in range(100):
            self.step()
        self._target_ee = self._measured_ee()[0].copy()
        self.manifest = {'simulator': 'Isaac Sim 5.1 / PhysX', 'physics_dt': self.dt,
                         'start_pose': self.start_pose,
                         'arm_solver': self.arm_solver,
                         'arm_solver_sync': self.arm_solver_sync,
                         'arm_solver_provenance': self._unitree_ik.manifest if self._unitree_ik else None,
                         'reference_definition': self.reference_definition,
                         'robot_urdf': str(HAND_URDF), 'fixed_base': True,
                         'joint_names': self.names, 'effort_limits': self.effort_limits.tolist(),
                         'velocity_limits': self.velocity_limits.tolist(),
                         'ee_offset_wrist_m': self.ee_offset.tolist(), 'camera_intrinsics': self.intrinsics,
                         'camera_streams': self.camera_calibration(),
                         'scene_variant': self.scene_variant,
                         'known_asset_catalog': str(self.output/'cad_assets/catalog.json'),
                         'sensor_timebase': 'shared simulation step and monotonic seconds; one render per camera bundle',
                         'lidar': {'type': 'PhysX scene raycast', 'mount_link': 'torso_link',
                                   'scan_model': 'instantaneous 21x9 forward grid, 0.02 to 5m; not a modeled MID360'},
                         'ee_definition': 'Wrist-local tool point 0.17m along +X, -0.003m alongY; same measured point as IK goal',
                         'recommended_approach_standoff_m': .08,
                         'control': ('Measured PhysX Jacobian integrated damped least-squares IK' if self.arm_solver == 'dls'
                                     else 'Unitree-objective seven-joint IK adapted to measured torso and same tool frame')+'; force-limited implicit drives',
                         'solver_scheduling': ('Every physics tick, synchronous comparison only' if self.arm_solver_sync
                                               else 'DLS every physics tick; Unitree latest-only worker submitted at 50 Hz simulation time'),
                         'arm_gravity_support': 'Continuous measured PhysX gravity feedforward; drive budget = original rating minus |feedforward|',
                         'arm_total_effort_limits_Nm': self.arm_total_effort_limits.tolist(),
                         'effort_telemetry': 'efforts is IsaacLab clipped PD estimate for implicit drives; projected_joint_forces is measured PhysX joint reaction projection',
                         'self_collisions': True, 'environment_collisions': True,
                         'safety': 'Cartesian and joint target speed limits, joint limits, contact hold; no complete motion planner',
                         'fixture_disclosures': self.scene['disclosures']}
        self.manifest['evaluation_hand_target_contacts'] = {
            'enabled': self.evaluation_contact_pairs,
            'evaluation_only': True,
            'target_body_path': getattr(self, 'evaluation_contact_target_path', None),
            'sensor_body_names': list(self.evaluation_contact_sensors),
            'collision_filtering': self.evaluation_contact_collision_filtering,
            'scope': 'Filtered normal forces against compound drawer front plus U-handle; not CAD-only contact faces'}
        (self.output/'backend_manifest.json').write_text(json.dumps(self.manifest, indent=2))

    def _configure_evaluation_contact_pairs(self, ContactSensor, ContactSensorCfg):
        """Optional evaluation instruments, never consumed by the controller."""
        from pxr import Sdf, UsdPhysics
        matches = [j['body_path'] for j in self.scene['fixture_joints'] if j['id'] == 'PullHandle']
        if len(matches) != 1:
            raise ValueError('Partner contact evaluation requires exactly one interaction PullHandle body')
        target_path = matches[0]
        target = self.stage.GetPrimAtPath(target_path)
        if not target.IsValid() or not target.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError('The evaluated PullHandle partner is not a rigid body')
        self.evaluation_contact_target_path = target_path
        prefix = '/World/AssistG1/'
        for name in EVALUATION_HAND_BODY_NAMES:
            path = prefix+name
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f'Expected separate physical finger body: {path}')
            # IsaacLab filtered contact reporting supports one-to-many, not
            # multiple sensor bodies in one filtered sensor.
            self.evaluation_contact_sensors[name] = ContactSensor(ContactSensorCfg(
                prim_path=path, filter_prim_paths_expr=[target_path],
                update_period=0., history_length=0, track_pose=False,
                track_air_time=False, max_contact_data_count_per_prim=32))
        if self.evaluation_disable_hand_target_collisions:
            # Resolve the palm through actual USD rigid bodies and the URDF's
            # fixed parent chain. The importer normally merges it into wrist_yaw.
            joints = {j.find('child').attrib['link']: j for j in ET.parse(HAND_URDF).getroot().findall('joint')}
            palm_owner = 'right_hand_palm_link'; chain = [palm_owner]
            while True:
                candidate = self.stage.GetPrimAtPath(prefix+palm_owner)
                if candidate.IsValid() and candidate.HasAPI(UsdPhysics.RigidBodyAPI):break
                joint = joints.get(palm_owner)
                if joint is None or joint.attrib['type'] != 'fixed':
                    raise RuntimeError('Cannot resolve the actual fixed palm rigid body for negative control')
                palm_owner = joint.find('parent').attrib['link']; chain.append(palm_owner)
            bodies = [*EVALUATION_HAND_BODY_NAMES]
            if palm_owner not in bodies:bodies.append(palm_owner)
            pairs = []
            for name in bodies:
                prim = self.stage.GetPrimAtPath(prefix+name)
                relation = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
                relation.AddTarget(Sdf.Path(target_path))
                if Sdf.Path(target_path) not in relation.GetTargets():
                    raise RuntimeError('Failed to author precise negative-control collision pair')
                pairs.append({'hand_body_path': str(prim.GetPath()), 'target_body_path': target_path})
            self.evaluation_contact_collision_filtering = {
                'disabled': True, 'method': 'UsdPhysics.FilteredPairsAPI before simulation reset',
                'pairs': pairs, 'palm_fixed_chain': chain,
                'scope': 'Only listed hand/palm bodies against compound PullHandle; other collisions retained'}

    def _evaluation_contact_observation(self, timestamp):
        """Measure forces from exactly the step represented by last_state."""
        forces = {}
        for name, sensor in self.evaluation_contact_sensors.items():
            if (not sensor.is_initialized or list(sensor.body_names) != [name]
                    or sensor.num_bodies != 1 or sensor.contact_physx_view.filter_count != 1):
                raise RuntimeError(f'Invalid one-body/one-partner contact sensor binding: {name}')
            sensor.update(self.dt, force_recompute=True)
            matrix = sensor.data.force_matrix_w
            if matrix is None:raise RuntimeError(f'Missing partner force matrix: {name}')
            force = filtered_normal_force_vector(matrix.detach().cpu().numpy())
            forces[name] = force.tolist()
        if tuple(forces) != EVALUATION_HAND_BODY_NAMES:
            raise RuntimeError('Evaluation must report all seven physical finger bodies')
        return {'schema': 'punktnull.evaluation_hand_target_contacts.v1', 'evaluation_only': True,
                'valid': True, 'target_asset_id': 'PullHandleGrasp', 'target_rigid_body_asset_id': 'PullHandle',
                'target_body_paths': [self.evaluation_contact_target_path],
                'partner_scope': 'Compound drawer front plus U-handle, not individual CAD contact faces',
                'sensor_body_names': list(forces), 'timestamp': float(timestamp),
                'sim_time': self.step_index*self.dt, 'physics_step': self.step_index,
                'measurement': 'post_step_filtered_normal_force_matrix', 'force_units': 'N',
                'force_semantics': 'PhysX normal impulses divided by physics_dt; excludes friction; summed per finger/partner',
                'per_body_normal_forces_W': forces,
                'per_body_normal_force_norms_N': {name: float(np.linalg.norm(value)) for name, value in forces.items()},
                'negative_control': self.evaluation_disable_hand_target_collisions,
                'collision_filtering': self.evaluation_contact_collision_filtering}

    def _add_camera(self, name, path, width, height, source_link=None, T_link_camera=None):
        from pxr import UsdGeom
        if width < 16 or height < 16:
            raise ValueError('Camera dimensions must be at least 16 pixels')
        camera = UsdGeom.Camera.Define(self.stage, path)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(36.*height/width)
        camera.CreateFocalLengthAttr(18.)
        camera.CreateClippingRangeAttr(self.Gf.Vec2f(.02, 20.))
        product = self.rep.create.render_product(path, (width, height))
        rgb = self.rep.AnnotatorRegistry.get_annotator('rgb')
        depth = self.rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
        rgb.attach(product)
        depth.attach(product)
        self.camera_streams[name] = {
            'path': path, 'width': width, 'height': height,
            'op': UsdGeom.Xformable(camera.GetPrim()).AddTransformOp(),
            'product': product, 'rgb': rgb, 'depth': depth,
            'intrinsics': {'fx': width*.5, 'fy': width*.5, 'cx': width/2., 'cy': height/2.,
                           'width': width, 'height': height},
            'source_link': source_link or 'torso_link',
            'T_link_camera': T_link_camera,
        }

    def camera_calibration(self):
        return {name: {'intrinsics': dict(stream['intrinsics']),
                       'source_link': stream['source_link'],
                       'T_link_camera': (stream['T_link_camera'].tolist()
                                         if stream['T_link_camera'] is not None else None),
                       'convention': 'optical_x_right_y_down_z_forward',
                       'distortion': 'none (ideal simulated pinhole)',
                       'mount_source': 'configured rigid extrinsic, transformed by measured PhysX link pose'}
                for name, stream in self.camera_streams.items()}

    @staticmethod
    def _rotation(wxyz):
        from scipy.spatial.transform import Rotation
        q = np.asarray(wxyz)
        return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()

    def _measured_ee(self):
        pose = self.robot.data.body_link_pose_w[0, self.body_id].detach().cpu().numpy()
        rot = self._rotation(pose[3:7])
        return pose[:3]+rot@self.ee_offset, pose[3:7][[1, 2, 3, 0]], rot

    def _update_camera(self):
        torso = self.robot.data.body_link_pose_w[0, self.torso_id].detach().cpu().numpy()
        rot = self._rotation(torso[3:7])
        eye = torso[:3]+rot@np.array([.10, 0., .426])
        forward = rot@np.array([1., -.10, -.40])
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, rot[:, 2]); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        T = np.eye(4); T[:3, :3] = np.column_stack((right, down, forward)); T[:3, 3] = eye
        self.T_world_camera = T
        view = self.Gf.Matrix4d().SetLookAt(self.Gf.Vec3d(*eye), self.Gf.Vec3d(*(eye+forward)), self.Gf.Vec3d(*(-down)))
        self.camera_op.Set(view.GetInverse())
        self.camera_streams['head']['T_world_camera'] = T
        # Explicitly expose the unchanged head calibration relative to torso.
        T_world_torso = np.eye(4)
        T_world_torso[:3, :3] = rot
        T_world_torso[:3, 3] = torso[:3]
        self.camera_streams['head']['T_link_camera'] = np.linalg.inv(T_world_torso) @ T
        from .sensor_geometry import camera_from_link_pose
        for name, stream in self.camera_streams.items():
            if name == 'head':
                continue
            pose = self.robot.data.body_link_pose_w[0, stream['body_id']].detach().cpu().numpy()
            transform = camera_from_link_pose(pose[:3], pose[3:7][[1, 2, 3, 0]], stream['T_link_camera'])
            stream['T_world_camera'] = transform
            eye = transform[:3, 3]
            forward, down = transform[:3, 2], transform[:3, 1]
            view = self.Gf.Matrix4d().SetLookAt(self.Gf.Vec3d(*eye), self.Gf.Vec3d(*(eye+forward)),
                                             self.Gf.Vec3d(*(-down)))
            stream['op'].Set(view.GetInverse())

    def step(self, ee_target_world=None, ee_quat_xyzw=None, grip=0.0):
        if self.closed:
            raise RuntimeError('Isaac backend already closed')
        q = self.robot.data.joint_pos[0].detach().cpu().numpy().copy()
        ee, quat, rot = self._measured_ee()
        self._update_arm_gravity_support()
        external_hold = self._external_hold_requested
        self._external_hold_requested = False
        self.hold_reason = None
        contact_norm = 0.
        contact_force_world = np.zeros(3)
        contact_body = None
        contact_retreat = False
        if self.contact.is_initialized:
            forces = self.contact.data.net_forces_w.detach().cpu().numpy()
            # Ignore leg/foot contacts; only right arm and hand are a movement guard.
            names = list(self.contact.body_names)
            relevant = [i for i, n in enumerate(names) if any(s in n for s in ('shoulder', 'elbow', 'wrist', 'hand'))]
            if relevant:
                maximum = relevant[int(np.argmax(np.linalg.norm(forces[0, relevant], axis=-1)))]
                contact_force_world = np.asarray(forces[0, maximum], dtype=float).copy()
                contact_norm = float(np.linalg.norm(contact_force_world))
                contact_body = names[maximum]
        target = np.asarray(ee_target_world if ee_target_world is not None else self._target_ee, dtype=float)
        target_valid = target.shape == (3,) and np.isfinite(target).all()
        if not target_valid:
            self.command_rejections += 1
            target = ee.copy()
            self.hold_reason = 'invalid_cartesian_target'
        if not np.isfinite(grip) or not 0. <= grip <= 1.:
            self.command_rejections += 1
            grip = 0.
            self.hold_reason = 'invalid_grip'
        # Broad physical workspace limit; individual attainability is monitored
        # from measured error rather than asserted from the workspace box.
        if np.any(target < np.array([.05, -.65, .60])) or np.any(target > np.array([.68, .30, 1.45])):
            target = ee.copy(); self.hold_reason = 'workspace_limit'
        if ee_quat_xyzw is not None:
            incoming_quat = np.asarray(ee_quat_xyzw, float)
            if incoming_quat.shape != (4,) or not np.isfinite(incoming_quat).all() or np.linalg.norm(incoming_quat) < 1e-8:
                self.command_rejections += 1
                self.hold_reason = 'invalid_orientation'
                target = ee.copy()
                ee_quat_xyzw = None
        if contact_norm > 18.:
            # Contact is a unilateral constraint. Permit only an explicit
            # retreat along the measured force on the robot, and remove any
            # old inward/tangential trajectory. This is not contact repair or
            # a general collision planner; multiple opposing contacts can
            # require a separately designed recovery procedure.
            outward = contact_force_world/contact_norm
            retreat_distance = float((target-ee)@outward)
            grip = min(float(grip), self.last_grip)
            ee_quat_xyzw = None
            if self.hold_reason is None and retreat_distance > 1e-6:
                # A Cartesian projection alone cannot cancel an older inward
                # joint-drive reference. Flush once on entry into retreat.
                self._flush_arm_reference(q, ee, 'contact_retreat')
                target = ee+outward*retreat_distance
                pending_outward = max(0., float((self._target_ee-ee)@outward))
                self._target_ee = ee+outward*min(pending_outward, retreat_distance)
                contact_retreat = True
            else:
                target = ee.copy()
                self.hold_reason = self.hold_reason or 'arm_contact_force'
        if self.hold_reason:
            # Both reference integrators must stop. Also replace target: an
            # invalid grip must not leave an otherwise valid motion pending.
            self._flush_arm_reference(q, ee, 'hold')
            target = ee.copy()
            self._target_ee = ee.copy()
            ee_quat_xyzw = None
        # hold() is also used for a one-cycle reversal flush. A new nonzero
        # target in that same step resumes integration from the flushed q;
        # persistent deadman holds retain the first fixed PD hold reference.
        stationary_hold = (external_hold and np.linalg.norm(target-ee) < 1e-9
                           and ee_quat_xyzw is None and not contact_retreat)
        arm_holding = bool(self.hold_reason) or stationary_hold
        if not arm_holding and not contact_retreat:
            self._arm_reference_mode = 'active'
        delta = target-self._target_ee
        reference_speed = .06 if contact_retreat else .16
        self._target_ee += delta*min(1., reference_speed*self.dt/max(np.linalg.norm(delta), 1e-9))
        old_target = self.joint_target[0, self.arm_ids].cpu().numpy()
        # Integrate the resolved-rate reference. Replacing it with measured q
        # every cycle would erase the drive's gravity-holding position offset.
        if not arm_holding:
            proposed = self._arm_joint_proposal(q, ee, quat, rot, ee_quat_xyzw, old_target)
            if proposed is not None:
                proposed = old_target+np.clip(proposed-old_target, -.8*self.dt, .8*self.dt)
                proposed = np.clip(proposed, self.lower[self.arm_ids]+.015, self.upper[self.arm_ids]-.015)
                self.joint_target[0, self.arm_ids] = self.torch.tensor(proposed, dtype=self.torch.float32)
        else:
            self._arm_solver_diagnostics = {**getattr(self, '_arm_solver_diagnostics', {}),
                                            'name': getattr(self, 'arm_solver', 'dls'),
                                            'status': 'hold', 'accepted': False,
                                            'solve_ms': None, 'request_id': None}
        for index in self.finger_ids:
            name = self.names[index]
            closed = -.72 if ('thumb_1' in name or 'thumb_2' in name) else (.4 if 'thumb_0' in name else .9)
            wanted = float(np.clip(grip*closed, self.lower[index]+.005, self.upper[index]-.005))
            old = float(self.joint_target[0, index])
            if contact_norm > 18.:
                # Discard already queued finger closure as well as rejecting
                # a larger grip scalar. Opening remains available under load.
                direction = np.sign(closed)
                if (wanted-q[index])*direction > 0.:
                    wanted = float(q[index])
                if (old-q[index])*direction > 0.:
                    old = float(q[index])
            self.joint_target[0, index] = old+float(np.clip(wanted-old, -1.0*self.dt, 1.0*self.dt))
        self.last_grip = float(grip)
        self.robot.set_joint_position_target(self.joint_target)
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.step_index += 1
        self.robot.update(self.dt)
        self.contact.update(self.dt)
        ee, quat, _ = self._measured_ee()
        measured_q = self.robot.data.joint_pos[0].detach().cpu().numpy().copy()
        velocity = self.robot.data.joint_vel[0].detach().cpu().numpy().copy()
        effort = self.robot.data.applied_torque[0].detach().cpu().numpy().copy()
        reaction = self.robot.root_physx_view.get_dof_projected_joint_forces()[0].detach().cpu().numpy().copy()
        velocity_violation = float(max(0., (np.abs(velocity)-self.velocity_limits).max()))
        valid = bool(np.isfinite(measured_q).all() and np.isfinite(ee).all()
                     and np.isfinite(velocity).all() and np.isfinite(reaction).all()
                     and velocity_violation < .05
                     and np.all(measured_q >= self.lower-.035) and np.all(measured_q <= self.upper+.035))
        violation = np.maximum(self.lower-measured_q, measured_q-self.upper)
        self.last_state = {'ee_position': ee.tolist(), 'ee_quaternion': quat.tolist(),
                           'joints': measured_q.tolist(), 'velocities': velocity.tolist(),
                           'efforts': effort.tolist(), 'effort_source': 'clipped_implicit_pd_estimate',
                           'projected_joint_forces': reaction.tolist(),
                           'joint_targets': self.joint_target[0].cpu().tolist(),
                           'joint_names': self.names, 'sim_time': self.step_index*self.dt,
                           'timestamp': time.monotonic(), 'valid': valid, 'fixed_base': True,
                           'start_pose': self.start_pose,
                           'arm_solver': dict(self._arm_solver_diagnostics),
                           'contact_force_max': contact_norm, 'hold_reason': self.hold_reason,
                           'contact_force_world': contact_force_world.tolist(), 'contact_body': contact_body,
                           'contact_retreat_active': contact_retreat, 'accepted_grip': self.last_grip,
                           'joint_limit_violation': float(max(0., violation.max())),
                           'joint_velocity_limit_violation': velocity_violation,
                           'contact_present': bool(contact_norm > 1.),
                           'contact_active': bool(contact_norm > 18.),
                           'collision_stop': self.hold_reason == 'arm_contact_force',
                           'command_accepted': self.hold_reason is None,
                           'command_rejections': self.command_rejections,
                           'arm_reference_mode': self._arm_reference_mode,
                           'arm_reference_flushes': self.arm_reference_flushes,
                           'arm_hold_gravity_offset_rad': self._arm_hold_gravity_offset.tolist(),
                           'arm_hold_gravity_forces_Nm': self._arm_hold_gravity_forces.tolist(),
                           'arm_hold_gravity_saturated': self._arm_hold_gravity_saturated,
                           'arm_gravity_feedforward_Nm': self._arm_gravity_feedforward.tolist(),
                           'arm_drive_effort_budget_Nm': self._arm_drive_effort_budget.tolist(),
                           'arm_total_effort_limits_Nm': self.arm_total_effort_limits.tolist(),
                           'tracking_error': float(np.linalg.norm(target-ee)),
                           'servo_target': self._target_ee.tolist(),
                           'fixture_joint_states': self.fixture_joint_states()}
        if getattr(self, 'evaluation_contact_pairs', False):
            self.last_state['evaluation_hand_target_contacts'] = self._evaluation_contact_observation(self.last_state['timestamp'])
        return self.last_state

    def _arm_joint_proposal(self, measured_q, ee, quat, rot, target_quaternion, old_target):
        """One solver choice; both share the physical drives and motion guards."""
        if getattr(self, 'arm_solver', 'dls') == 'unitree':
            return self._unitree_joint_proposal(measured_q, target_quaternion)
        started = time.perf_counter()
        error = self._target_ee-ee
        error *= min(1., .08/max(np.linalg.norm(error), 1e-9))
        jac = self.robot.root_physx_view.get_jacobians()[0, self.body_id-1].detach().cpu().numpy()[:, self.arm_ids].copy()
        offset = rot@self.ee_offset
        skew = np.array([[0., -offset[2], offset[1]], [offset[2], 0., -offset[0]], [-offset[1], offset[0], 0.]])
        jac[:3] -= skew@jac[3:]
        J, error_all = jac[:3], error
        if target_quaternion is not None:
            from scipy.spatial.transform import Rotation
            angular = (Rotation.from_quat(target_quaternion)*Rotation.from_quat(quat).inv()).as_rotvec()
            angular *= min(1., .12/max(np.linalg.norm(angular), 1e-9))
            J = np.vstack((jac[:3], jac[3:]*.30))
            error_all = np.r_[error, angular*.30]
        dq = J.T@np.linalg.solve(J@J.T+np.eye(J.shape[0])*.018**2, error_all)
        self._arm_solver_diagnostics = {'name': 'dls', 'status': 'solved', 'accepted': True,
                                        'solve_ms': (time.perf_counter()-started)*1000.,
                                        'request_id': self.step_index}
        return old_target+dq*3.*self.dt

    def _unitree_joint_proposal(self, measured_q, target_quaternion):
        client = self._unitree_ik
        torso = self.robot.data.body_link_pose_w[0, self.torso_id].detach().cpu().numpy()
        args = (dict(zip(self.names, map(float, measured_q))), self._target_ee.tolist(),
                None if target_quaternion is None else np.asarray(target_quaternion).tolist(),
                torso[:3].tolist(), torso[3:7][[1, 2, 3, 0]].tolist())
        try:
            if self.arm_solver_sync:
                result = client.solve(*args, timeout=2.)
            else:
                result = client.poll()
                if self.step_index-self._unitree_submit_step >= round(.02/self.dt):
                    client.submit(*args)
                    self._unitree_submit_step = self.step_index
            if result is not None:
                self._arm_solver_diagnostics = {**result, 'name': 'unitree',
                                                'status': 'solved' if result.get('accepted') else 'rejected'}
                if not self.arm_solver_sync and float(result.get('result_age_s', float('inf'))) > .15:
                    raise TimeoutError('IK result was computed from stale input')
                if not result.get('accepted', False):
                    raise ValueError(result.get('solver_status', result.get('status', 'IK rejected target')))
                values = result['joint_targets']
                candidate = np.asarray([values[name] for name in self.arm_names], dtype=float)
                if not np.isfinite(candidate).all() or np.any(candidate < self.lower[self.arm_ids]) or np.any(candidate > self.upper[self.arm_ids]):
                    raise ValueError('IK returned invalid joint target')
                self._unitree_goal = candidate
                self._unitree_result_at = time.monotonic()-float(result.get('result_age_s', 0.))
            if self._unitree_goal is None:
                self._arm_solver_diagnostics = {**self._arm_solver_diagnostics, 'status': 'waiting', 'accepted': False}
                return None
            if not self.arm_solver_sync and time.monotonic()-self._unitree_result_at > .15:
                raise TimeoutError('IK result stream stale')
            return self._unitree_goal.copy()
        except (ValueError, RuntimeError, TimeoutError, KeyError, TypeError, OSError) as exc:
            self.command_rejections += 1
            self.hold_reason = 'arm_ik_rejected'
            self._arm_solver_diagnostics = {**self._arm_solver_diagnostics, 'accepted': False,
                                            'status': 'rejected', 'error': str(exc)[:180]}
            self._flush_arm_reference(measured_q, self._measured_ee()[0], 'hold')
            return None

    def capture_bundle(self):
        """Render RGB-D from all measured mounts at the same physics step.

        Physics does not advance while this bundle is rendered. The timestamp
        is the shared monotonic capture-completion time, not wall-clock UTC.
        """
        if self.closed:
            raise RuntimeError('Isaac backend already closed')
        self._update_camera()
        self.sim.render()
        timestamp = time.monotonic()
        self.frame_id += 1
        frames = {}
        for name, stream in self.camera_streams.items():
            rgb = np.asarray(stream['rgb'].get_data())
            depth = np.asarray(stream['depth'].get_data())
            expected = (stream['height'], stream['width'])
            if rgb.shape[:2] != expected or depth.shape != expected:
                raise RuntimeError(f'Invalid {name} rendered RGB-D shapes: {rgb.shape}, {depth.shape}')
            frames[name] = {
                'rgb': np.ascontiguousarray(rgb[..., :3], dtype=np.uint8).copy(),
                'depth': np.ascontiguousarray(depth, dtype=np.float32).copy(),
                'intrinsics': dict(stream['intrinsics']),
                'T_world_camera': stream['T_world_camera'].copy(),
                'T_link_camera': stream['T_link_camera'].copy(),
                'source_link': stream['source_link'], 'camera_id': name,
                'frame_id': self.frame_id, 'timestamp': timestamp,
                'sim_time': self.step_index*self.dt, 'physics_step': self.step_index,
                'depth_convention': 'optical_z_meters', 'source': 'Isaac rendered RGB-D'}
        self.last_capture = frames['head']
        self.last_bundle = {'frames': frames, 'frame_id': self.frame_id, 'timestamp': timestamp,
                            'sim_time': self.step_index*self.dt, 'synchronized': True,
                            'synchronization': 'one render of stationary physics step; measured link extrinsics'}
        return self.last_bundle

    def capture(self):
        """Backward-compatible robot head stream. Full bundle is retained."""
        return self.capture_bundle()['frames']['head']

    def known_task_metadata(self):
        """Known CAD/affordances in object coordinates, never observed poses."""
        return json.loads(json.dumps(self.scene['known_asset_catalog']))

    def fixture_joint_states(self):
        from .sensor_geometry import measure_fixed_joint
        return {joint['id']: measure_fixed_joint(
                    self.fixture_body_views[joint['id']].get_transforms()[0],
                    self.fixture_body_views[joint['id']].get_velocities()[0], joint)
                for joint in self.scene['fixture_joints']}

    def capture_lidar(self, horizontal_beams=21, vertical_beams=9, max_range=5.):
        """Actual PhysX scene-raycast scan from a fixed torso LiDAR mount.

        This ideal, instantaneous forward scan is additional sensor evidence;
        it is not re-labelled camera depth and does not emulate a MID360.
        Self hits are excluded and retained as an explicit beam validity mask.
        """
        from omni.physx import get_physx_scene_query_interface
        from .sensor_geometry import camera_from_link_pose
        if not (2 <= horizontal_beams <= 256 and 2 <= vertical_beams <= 128 and .02 < max_range <= 20.):
            raise ValueError('Invalid LiDAR scan dimensions/range')
        torso = self.robot.data.body_link_pose_w[0, self.torso_id].detach().cpu().numpy()
        mount = np.eye(4)
        mount[:3, 3] = [.10, 0., .27]
        transform = camera_from_link_pose(torso[:3], torso[3:7][[1, 2, 3, 0]], mount)
        azimuth, elevation = np.meshgrid(np.linspace(-np.deg2rad(65), np.deg2rad(65), horizontal_beams),
                                       np.linspace(-np.deg2rad(45), np.deg2rad(15), vertical_beams))
        directions = np.column_stack((np.cos(elevation.ravel())*np.cos(azimuth.ravel()),
                                      np.cos(elevation.ravel())*np.sin(azimuth.ravel()),
                                      np.sin(elevation.ravel())))
        origin = transform[:3, 3]
        ranges = np.full(len(directions), np.nan, dtype=np.float32)
        self_hit = np.zeros(len(directions), dtype=bool)
        interface = get_physx_scene_query_interface()
        for i, local_direction in enumerate(directions):
            direction = transform[:3, :3] @ local_direction
            hit = interface.raycast_closest(tuple(float(v) for v in origin),
                                            tuple(float(v) for v in direction), float(max_range))
            if not hit.get('hit'):
                continue
            self_hit[i] = str(hit.get('rigidBody', hit.get('collision', ''))).startswith('/World/AssistG1')
            if not self_hit[i] and float(hit['distance']) >= .02:
                ranges[i] = float(hit['distance'])
        valid = np.isfinite(ranges)
        return {'source': 'PhysX scene raycasts', 'sensor_id': 'torso_lidar',
                'source_link': 'torso_link', 'T_link_lidar': mount,
                'T_world_lidar': transform, 'frame_id': self.frame_id,
                'timestamp': time.monotonic(), 'sim_time': self.step_index*self.dt,
                'physics_step': self.step_index, 'points_sensor': (directions[valid]*ranges[valid, None]).astype(np.float32),
                'points_lidar': (directions[valid]*ranges[valid, None]).astype(np.float32), 'deskew_required': False,
                'directions_sensor': directions.astype(np.float32), 'ranges': ranges,
                'valid': valid, 'self_hit': self_hit, 'range_limits_m': [.02, float(max_range)],
                'scan_shape': [vertical_beams, horizontal_beams],
                'convention': 'link_x_forward_y_left_z_up',
                'scan_model': 'ideal instantaneous forward grid; no real hardware timing/noise model'}

    def _update_arm_gravity_support(self):
        """Apply gravity through bounded actuator torques on active and hold ticks.

        PhysX clips implicit drive and applied torque separately. Reducing the
        drive budget by |FF| preserves the original combined effort bound.
        """
        gravity = self.robot.root_physx_view.get_gravity_compensation_forces()[0].detach().cpu().numpy()[self.arm_ids]
        feedforward, budget = gravity_effort_budget(gravity, self.arm_total_effort_limits)
        self.robot.write_joint_effort_limit_to_sim(
            self.torch.tensor(budget[None], dtype=self.torch.float32), joint_ids=self.arm_ids)
        self.robot.set_joint_effort_target(
            self.torch.tensor(feedforward[None], dtype=self.torch.float32), joint_ids=self.arm_ids)
        self._arm_gravity_feedforward = feedforward
        self._arm_drive_effort_budget = budget
        self._arm_hold_gravity_forces = gravity.copy()
        self._arm_hold_gravity_saturated = bool(np.any(np.abs(gravity-feedforward) > 1e-5))

    def _flush_arm_reference(self, measured_q, measured_ee, mode):
        """Discard both integrator queues once per hold/retreat transition.

        Gravity is continuously supplied as bounded actuator feedforward.
        Repeatedly reconstructing a gravity/Kp position offset produced a
        measured 80 mm drift over 50 stop/resume cycles in this PhysX fixture.
        The hold therefore latches measured q alone and retains it thereafter.
        """
        if self._arm_reference_mode != mode:
            if getattr(self, '_unitree_ik', None) is not None:
                self._unitree_ik.reset()
                self._unitree_goal = None
                self._unitree_result_at = -float('inf')
                self._unitree_submit_step = -1000
            q = np.asarray(measured_q)[self.arm_ids]
            arm_q = np.clip(q, self.lower[self.arm_ids], self.upper[self.arm_ids])
            self._arm_hold_gravity_offset = np.zeros_like(arm_q)
            self.joint_target[0, self.arm_ids] = self.torch.tensor(arm_q, dtype=self.torch.float32)
            self._target_ee = np.asarray(measured_ee).copy()
            self.arm_reference_flushes += 1
        self._arm_reference_mode = mode

    def hold(self):
        """Discard pending Cartesian and joint references on hold entry.

        The once-latched PD reference discards the old motion queue; continuous
        gravity feedforward supports the arm. Contact and inertia can
        still move the arm. This is not an instantaneous plant stop.
        A following nonzero step target permits motion from this new reference.
        """
        ee = self._measured_ee()[0]
        q = self.robot.data.joint_pos[0].detach().cpu().numpy()
        self._flush_arm_reference(q, ee, 'hold')
        self._target_ee = ee.copy()
        self._external_hold_requested = True

    def evaluation_truth(self):
        from pxr import UsdGeom
        result = json.loads(json.dumps(self.scene))
        prim = self.stage.GetPrimAtPath(self.scene['root']+'/Parts/GreenButton')
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        result['green_button_position'] = list(world.ExtractTranslation())
        result['sim_time'] = self.step_index*self.dt
        result['fixture_joint_states'] = self.fixture_joint_states()
        return result

    def close(self):
        if not self.closed:
            # SimulationApp owns graph teardown. Destroying a render product
            # here leaves Replicator's shutdown callback with an invalid graph.
            self.closed = True
            if self._unitree_ik is not None:
                self._unitree_ik.close()
