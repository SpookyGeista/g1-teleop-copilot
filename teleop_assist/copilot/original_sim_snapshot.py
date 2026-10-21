"""Single-owner, paused-PhysX state adapter for an external original-sim shim.

No Isaac, SDK, DDS, camera or robot is initialized/imported here. The caller
supplies the original InteractiveScene (or a scene containing its actual
Articulation) and owns ALL stepping/resetting. ``read_generation`` must change
on every physics update/reset; a new simulator instance needs a new adapter.

The collision reader is receiver-owned code, not a packet/JSON flag. It must
read the complete known scene inside the supplied capture context. This module
does not extract arbitrary warehouse USD collision geometry or infer coverage.
Mock-object unit tests establish this adapter contract, not Isaac integration.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


ARM_JOINT_NAMES = tuple(f'{s}_{j}_joint' for s in ('left','right') for j in
    ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw'))
G1_JOINT_NAMES = frozenset(ARM_JOINT_NAMES) | frozenset(
    f'{s}_{j}_joint' for s in ('left','right') for j in
    ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll',
     'hand_index_0','hand_index_1','hand_middle_0','hand_middle_1','hand_thumb_0','hand_thumb_1','hand_thumb_2')) | frozenset(
    ('waist_yaw_joint','waist_roll_joint','waist_pitch_joint'))
# The reviewed original 29+14 fixed-base USD has 55 measured bodies. Requiring
# its actual named manifest avoids silently accepting only the two wrists.
G1_BODY_NAMES = frozenset('''pelvis imu_in_pelvis pelvis_contour_link torso_link
    waist_yaw_link waist_roll_link d435_link head_link imu_in_torso logo_link mid360_link'''.split()) | frozenset(
    f'{s}_{j}_link' for s in ('left','right') for j in
    ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll',
     'shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw',
     'hand_camera_base','hand_palm','hand_index_0','hand_index_1','hand_middle_0','hand_middle_1',
     'hand_thumb_0','hand_thumb_1','hand_thumb_2'))


def _number(value):
    try:return type(value) in (int,float) and math.isfinite(value)
    except OverflowError:return False


def _json(value):
    return json.dumps(value,sort_keys=True,allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _name(value):
    return isinstance(value,str) and 0<len(value)<=512


def _file_sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _array(value,shape,label):
    # Copy immediately; never retain aliased Isaac buffers. Torch is supplied
    # by the owner and deliberately not imported by this module.
    if hasattr(value,'detach'): value=value.detach()
    if hasattr(value,'cpu'): value=value.cpu()
    if hasattr(value,'numpy'): value=value.numpy()
    raw=np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.shape!=shape:
        raise ValueError(label+'_shape_or_numeric_type_invalid')
    copied=np.array(raw,dtype=np.float64,copy=True)
    if not np.isfinite(copied).all():raise ValueError(label+'_nonfinite')
    return copied


def _names(value,expected,label):
    if not isinstance(value,(list,tuple)) or not all(_name(n) for n in value):
        raise ValueError(label+'_names_required')
    names=tuple(value)
    if len(names)!=len(set(names)) or set(names)!=expected:
        raise ValueError(label+'_names_incomplete_duplicate_or_unexpected')
    return names


def _pose(values,label):
    values=_array(values,(7,),label)
    if abs(float(np.linalg.norm(values[3:]))-1.)>1e-4:
        raise ValueError(label+'_quaternion_not_unit_wxyz')
    matrix=np.eye(4);matrix[:3,3]=values[:3]
    # Same wxyz -> xyzw and normalization as frames.robot_request.
    matrix[:3,:3]=Rotation.from_quat(values[[4,5,6,3]]).as_matrix()
    return matrix.tolist()


def _rigid(value,label):
    a=_array(value,(4,4),label)
    if (not np.allclose(a[3],[0,0,0,1],rtol=0,atol=1e-8) or
            not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),rtol=0,atol=1e-6) or
            abs(float(np.linalg.det(a[:3,:3]))-1)>1e-6):
        raise ValueError(label+'_not_rigid')
    return a


@dataclass(frozen=True,slots=True)
class CaptureContext:
    acquisition_timestamp: float
    clock_id: str
    source_instance_id: str
    physics_generation: int
    calibration_id: str


@dataclass(frozen=True,slots=True)
class OriginalSimSnapshot:
    robot_state_json: str
    scene_json: str
    snapshot_sha256: str
    acquisition_timestamp: float
    capture_finished_at: float
    clock_id: str
    source_instance_id: str
    physics_generation: int
    sample_sequence: int
    calibration_id: str
    scope: str = 'original_USD_paused_simulator_snapshot_only'

    @property
    def robot_state(self):return json.loads(self.robot_state_json)

    @property
    def scene(self):return json.loads(self.scene_json)

    @property
    def left_base_ee(self):return self._tcp('left')

    @property
    def right_base_ee(self):return self._tcp('right')

    def _tcp(self,side):
        state=self.robot_state;tcp=np.eye(4);tcp[0,3]=.05
        return np.linalg.inv(state['base_in_world'])@np.asarray(
            state['observed_body_world_poses'][side+'_wrist_yaw_link'])@tcp


_SNAPSHOT_FIELDS=tuple(f.name for f in fields(OriginalSimSnapshot))


class OriginalSimSnapshotAdapter:
    """Copy actual original Articulation data; keep one private capture binding.

    The required receiver-owned ``collision_scene_reader(scene, context)`` must
    perform its reads synchronously under the same paused-generation ownership.
    It returns the existing complete-static-scene schema plus every context
    field verbatim. Old DDS/SHM/JSON observations cannot be made fresh by adding
    that metadata. Supply an audited reader of current simulator geometry.

    Same-generation capture calls only revalidate the existing sample. They
    never refresh its time. No method accepts a received timestamp or grants
    command permission. Guard, private receipt and final sink remain necessary.
    """
    def __init__(self,scene,*,read_generation,clock_id,source_instance_id,calibration_id,
                 robot_asset_sha256,collision_scene_reader,simulation_mode=False,
                 clock=time.monotonic):
        if simulation_mode is not True:raise ValueError('explicit_simulator_scope_required')
        if not all(_name(x) for x in (clock_id,source_instance_id,calibration_id)):
            raise ValueError('named_clock_source_and_calibration_required')
        if (not isinstance(robot_asset_sha256,str) or len(robot_asset_sha256)!=64 or
                any(c not in '0123456789abcdef' for c in robot_asset_sha256)):
            raise ValueError('reviewed_robot_asset_hash_required')
        if not all(callable(x) for x in (read_generation,collision_scene_reader,clock)):
            raise ValueError('receiver_owned_scene_and_generation_readers_required')
        self._scene=scene;self._robot=scene['robot'];self._data=self._robot.data
        # Bind to the actual configured local asset, not a caller's unattached
        # hash claim. This happens once and never reads a URL or loads Isaac.
        asset_path=self._robot.cfg.spawn.usd_path
        if not isinstance(asset_path,str) or '://' in asset_path:
            raise ValueError('local_configured_robot_asset_required')
        self._asset_path=Path(asset_path).resolve(strict=True)
        if not self._asset_path.is_file() or _file_sha256(self._asset_path)!=robot_asset_sha256:
            raise ValueError('configured_robot_asset_hash_mismatch')
        self._generation_reader=read_generation;self._scene_reader=collision_scene_reader;self._clock=clock
        self._clock_id=clock_id;self._source_id=source_instance_id;self._calibration=calibration_id
        self._asset_hash=robot_asset_sha256;self._owner=threading.get_ident()
        self._active=None;self._last_generation=None;self._sequence=0;self._last_time=None

    @property
    def sim_scene(self):return self._scene

    @property
    def robot(self):return self._robot

    @property
    def clock_id(self):return self._clock_id

    @property
    def source_instance_id(self):return self._source_id

    @property
    def calibration_id(self):return self._calibration

    def assert_owner(self):
        """Verify the articulation owner and return its current generation.

        This is an identity check only: it neither acquires a sample nor renews
        an existing one. A measured hold may therefore use it after physics has
        advanced, when the earlier command snapshot is no longer reusable.
        """
        return self._identity()

    def _identity(self):
        if threading.get_ident()!=self._owner:raise ValueError('single_simulator_owner_thread_required')
        if self._scene['robot'] is not self._robot or self._robot.data is not self._data:
            raise ValueError('simulator_articulation_instance_changed')
        if Path(self._robot.cfg.spawn.usd_path).resolve()!=self._asset_path:
            raise ValueError('configured_robot_asset_path_changed')
        generation=self._generation_reader()
        if type(generation) is not int or generation<0:raise ValueError('physics_generation_invalid')
        if self._last_generation is not None and generation<self._last_generation:
            raise ValueError('physics_generation_regressed_new_source_required')
        return generation

    def _now(self):
        now=self._clock()
        if not _number(now) or now<0 or (self._last_time is not None and now<self._last_time):
            raise ValueError('capture_clock_invalid_or_regressed')
        self._last_time=now
        return float(now)

    def invalidate(self):
        """Discard the active transaction; this never resets sequence/clock."""
        if threading.get_ident()!=self._owner:raise ValueError('single_simulator_owner_thread_required')
        self._active=None

    def _robot_values(self):
        data=self._data
        joints=_names(data.joint_names,G1_JOINT_NAMES,'joint')
        bodies=_names(data.body_names,G1_BODY_NAMES,'body')
        q=_array(data.joint_pos,(1,43),'joint_pos')[0]
        dq=_array(data.joint_vel,(1,43),'joint_vel')[0]
        root=_array(data.root_pose_w,(1,7),'root_pose_w')[0]
        body=_array(data.body_link_pose_w,(1,len(bodies),7),'body_link_pose_w')[0]
        state=dict(joint_positions=dict(zip(joints,q.tolist())),joint_velocities=dict(zip(joints,dq.tolist())),
            joint_names=list(joints),body_names=list(bodies),base_in_world=_pose(root,'root_pose_w'),
            observed_body_world_poses={name:_pose(pose,'body_'+name) for name,pose in zip(bodies,body)},
            body_pose_quaternion_convention='wxyz',robot_asset_sha256=self._asset_hash,
            robot_asset_source=str(self._asset_path),
            source='original_Unitree_USD_simulator_measured_state',complete=True)
        if state['base_in_world']!=state['observed_body_world_poses']['pelvis']:
            raise ValueError('measured_root_and_pelvis_frame_mismatch')
        return state

    def _scene_values(self,context):
        supplied=self._scene_reader(self._scene,context)
        if not isinstance(supplied,dict):raise ValueError('complete_scene_mapping_required')
        # Copy before inspecting; caller-owned geometry cannot mutate a capture.
        scene=json.loads(_json(supplied))
        if not _number(scene.get('acquisition_timestamp')) or type(scene.get('physics_generation')) is not int:
            raise ValueError('scene_capture_context_numeric_type_invalid')
        for field in fields(context):
            if scene.get(field.name)!=getattr(context,field.name):
                raise ValueError('scene_capture_context_mismatch:'+field.name)
        if (scene.get('geometry_complete') is not True or scene.get('static') is not True or
                scene.get('source')!='current_simulator_complete_collision_scene' or not _name(scene.get('scene_id'))):
            raise ValueError('complete_static_receiver_observed_scene_required')
        obstacles=scene.get('obstacles')
        if not isinstance(obstacles,list) or not 1<=len(obstacles)<=64:
            raise ValueError('complete_obstacles_required')
        ids=set();planes=set()
        for item in obstacles:
            if not isinstance(item,dict) or not _name(item.get('id')) or item['id'] in ids:
                raise ValueError('obstacle_identity_missing_or_duplicate')
            ids.add(item['id']);kind=item.get('type')
            if kind=='plane':
                if item.get('normal')!=[0,0,1] or not _number(item.get('offset')):
                    raise ValueError('world_z_plane_required')
                planes.add(item['id'])
            else:
                _rigid(item.get('world_pose'),'obstacle_pose')
                if kind=='box':
                    if np.any(_array(item.get('size_xyz'),(3,),'box_size')<=0):raise ValueError('positive_box_dimensions_required')
                elif kind in ('sphere','cylinder'):
                    sizes=[item.get('radius_m',item.get('radius'))]
                    if kind=='cylinder':sizes.append(item.get('length_m',item.get('length')))
                    if not all(_number(x) and x>0 for x in sizes):raise ValueError('positive_obstacle_dimensions_required')
                else:raise ValueError('unsupported_complete_scene_geometry')
        contacts=scene.get('allowed_static_contacts',[])
        if not isinstance(contacts,list):raise ValueError('contact_policy_list_required')
        for c in contacts:
            if (not isinstance(c,dict) or c.get('link',c.get('robot_link')) not in
                    ('left_ankle_roll_link','right_ankle_roll_link') or
                    c.get('obstacle',c.get('obstacle_id')) not in planes or
                    not _number(c.get('max_penetration_m')) or not 0<=c['max_penetration_m']<=.002):
                raise ValueError('support_contact_policy_invalid')
        if 'target' in ids:
            velocity=_array(scene.get('observed_target_velocity'),(6,),'observed_target_velocity')
            # Existing complete-static orientation preview contract; observing
            # one low velocity is not a general dynamic-scene certificate.
            if np.linalg.norm(velocity[:3])>.01 or np.linalg.norm(velocity[3:])>.02:
                raise ValueError('target_observed_velocity_not_stationary')
        return scene

    def _collect(self,context):
        if self._identity()!=context.physics_generation:raise ValueError('physics_changed_before_snapshot')
        state=self._robot_values()
        scene=self._scene_values(context)
        if self._robot_values()!=state:raise ValueError('robot_changed_during_snapshot')
        if self._identity()!=context.physics_generation:raise ValueError('physics_changed_during_snapshot')
        state.update({f.name:getattr(context,f.name) for f in fields(context)})
        return state,scene

    def _fresh_copy(self):
        return OriginalSimSnapshot(**dict(zip(_SNAPSHOT_FIELDS,self._active)))

    def capture(self):
        """Begin at most one acquisition per physics generation."""
        try:
            generation=self._identity()
            if generation==self._last_generation:
                if self._active is None:raise ValueError('generation_already_captured_requires_new_physics_sample')
                return self.read_same()
            self._active=None
            started=self._now()
            # Burn this acquisition generation even on failure; retries must
            # not turn the same buffered physics sample into a newer lease.
            self._last_generation=generation
            context=CaptureContext(started,self._clock_id,self._source_id,generation,self._calibration)
            state,scene=self._collect(context)
            encoded_state,encoded_scene=_json(state),_json(scene)
            digest=_digest(dict(robot_state=state,scene=scene))
            finished=self._now()
            if finished-started>=.15:raise ValueError('snapshot_capture_exceeded_state_lease')
            if self._identity()!=generation:raise ValueError('physics_changed_before_capture_commit')
            self._sequence+=1
            snapshot=OriginalSimSnapshot(encoded_state,encoded_scene,digest,started,finished,self._clock_id,
                self._source_id,generation,self._sequence,self._calibration)
            self._active=tuple(getattr(snapshot,name) for name in _SNAPSHOT_FIELDS)
            return self._fresh_copy()
        except Exception:
            self._active=None
            raise

    def read_same(self):
        """Read current geometry, return the private capture with its OLD time."""
        try:
            if self._active is None:raise ValueError('no_active_simulator_capture')
            saved=self._fresh_copy()
            if self._identity()!=saved.physics_generation:raise ValueError('physics_changed_after_capture')
            started=self._now()
            if started-saved.acquisition_timestamp>=.15:raise ValueError('captured_state_expired')
            context=CaptureContext(saved.acquisition_timestamp,saved.clock_id,saved.source_instance_id,
                saved.physics_generation,saved.calibration_id)
            state,scene=self._collect(context)
            if _json(state)!=saved.robot_state_json or _json(scene)!=saved.scene_json:
                raise ValueError('simulator_snapshot_changed_after_capture')
            finished=self._now()
            if finished-saved.acquisition_timestamp>=.15:raise ValueError('captured_state_expired_during_recheck')
            if self._identity()!=saved.physics_generation:raise ValueError('physics_changed_before_snapshot_return')
            return self._fresh_copy()
        except Exception:
            self._active=None
            raise
