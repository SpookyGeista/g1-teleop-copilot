"""CPU preview checks for an OPEN Dex3 hand against observed metric points.

This is deliberately not a full-robot collision certificate. Occluded surfaces,
point-cloud gaps, arm motion, finger closure and execution uncertainty are not
checked. Each link's convex hull encloses its actual CAD surface: concavities
can therefore cause conservative rejection, not spurious extra free space.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from .observations import rigid


DEFAULT_DESCRIPTOR = Path(__file__).parent / 'assets/dex3_right'
SCOPE = 'observed_points_vs_open_right_Dex3_and_straight_approach_only'


@dataclass(frozen=True)
class CollisionSettings:
    clearance_m: float = .003
    approach_m: float = .080
    approach_samples: int = 9
    pad_contact_distance_m: float = .002
    pad_penetration_m: float = .001

    def __post_init__(self):
        if not 5 <= self.approach_samples <= 10:
            raise ValueError('approach_samples must be between 5 and 10')
        if not all(np.isfinite(v) and v >= 0 for v in
                   [self.clearance_m, self.approach_m,
                    self.pad_contact_distance_m, self.pad_penetration_m]):
            raise ValueError('collision distances must be finite and nonnegative')
        if self.approach_m <= 0 or self.pad_penetration_m > self.pad_contact_distance_m:
            raise ValueError('invalid approach or pad contact allowance')


def _origin(element):
    node = element.find('origin')
    result = np.eye(4)
    if node is not None:
        result[:3, :3] = Rotation.from_euler(
            'xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        result[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
    return result


def _open_fk(root, positions):
    children = {j.find('child').get('link') for j in root.findall('joint')}
    roots = {link.get('name') for link in root.findall('link')} - children
    if roots != {'right_wrist_yaw_link'}:
        raise ValueError('descriptor must be rooted at right_wrist_yaw_link')
    frames = {'right_wrist_yaw_link': np.eye(4)}
    remaining = list(root.findall('joint'))
    while remaining:
        progressed = False
        for joint in remaining[:]:
            parent = joint.find('parent').get('link')
            if parent not in frames:
                continue
            local = _origin(joint)
            if joint.get('type') == 'revolute':
                q = float(positions[joint.get('name')])
                limits = joint.find('limit')
                if not float(limits.get('lower')) <= q <= float(limits.get('upper')):
                    raise ValueError('OPEN joint target exceeds a URDF limit')
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                if not np.isclose(np.linalg.norm(axis), 1.):
                    raise ValueError('joint axis must have unit length')
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
                local = local @ motion
            elif joint.get('type') != 'fixed':
                raise ValueError('unsupported hand joint')
            frames[joint.find('child').get('link')] = frames[parent] @ local
            remaining.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError('disconnected or cyclic hand URDF')
    return frames


def _points(value, name):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f'{name} must be finite Nx3 metric world points')
    if len(points) > 500_000:
        raise ValueError(f'{name} exceeds the 500000-point CPU preview bound')
    return points


class GraspCollisionFilter:
    """Load and hash-check assets once; reuse this object for a proposal batch.

    ``world_canonical`` is the GraspGenX output G = T_world_C. The descriptor
    stores B = T_C_wrist. A source mesh point reaches world as
    G @ B @ T_wrist_link(open) @ T_link_visual @ p_mesh. Here we precompute
    everything after G once, then transform the observed points into C.
    """

    def __init__(self, descriptor_path=DEFAULT_DESCRIPTOR, settings=None):
        path = Path(descriptor_path).resolve()
        self.path = path.parent if path.is_file() else path
        self.settings = settings or CollisionSettings()
        validation = json.loads((self.path / 'validation.json').read_text())
        if (validation.get('validation_schema') != 'punktnull.copilot.descriptor_validation.v1'
                or validation.get('geometry_valid') is not True):
            raise ValueError('descriptor has no matching geometry validation')
        self._hashes = validation.get('asset_sha256', {})
        for required in ['config.json', 'gripper.urdf', 'contact_patches.json']:
            if required not in self._hashes:
                raise ValueError(f'geometry validation does not cover {required}')
        for filename, expected in self._hashes.items():
            asset = (self.path / filename).resolve()
            if not asset.is_relative_to(self.path):
                raise ValueError('descriptor asset path escapes package')
            if hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
                raise ValueError(f'descriptor asset hash mismatch: {filename}')
        self.config_sha256 = self._hashes['config.json']
        self.config = json.loads((self.path / 'config.json').read_text())
        self.base_rotation = rigid(self.config['base_rotation'], 'T_C_wrist')
        root = ET.parse(self.path / 'gripper.urdf').getroot()
        frames = _open_fk(root, self.config['open'])
        annotations = json.loads((self.path / 'contact_patches.json').read_text())
        self.canonical_meshes = {}
        self.pad_meshes = {}
        self._hulls = {}
        for link in root.findall('link'):
            name = link.get('name')
            parts = []
            for visual in link.findall('visual'):
                node = visual.find('geometry/mesh')
                if node is None:
                    raise ValueError('preview requires actual mesh visual geometry')
                filename = node.get('filename')
                if filename not in self._hashes:
                    raise ValueError(f'unvalidated mesh: {filename}')
                mesh = trimesh.load(self.path / filename, force='mesh', process=True)
                mesh.apply_scale(np.fromstring(node.get('scale', '1 1 1'), sep=' '))
                mesh.apply_transform(_origin(visual))
                parts.append(mesh)
            if not parts:
                continue  # empty wrist root
            mesh = trimesh.util.concatenate(parts)
            mesh.apply_transform(self.base_rotation @ frames[name])
            self.canonical_meshes[name] = mesh
            hull = ConvexHull(mesh.vertices)
            # Qhull plane normals are unit length. Keep all original planes;
            # dropping rounded near-duplicate planes would enlarge the hull.
            self._hulls[name] = (mesh.bounds.copy(), hull.equations.copy())
            if name in annotations:
                ids = np.asarray(annotations[name]['triangle_ids'], dtype=int)
                if not len(ids) or ids.min() < 0 or ids.max() >= len(mesh.faces):
                    raise ValueError('invalid annotated contact triangle indices')
                self.pad_meshes[name] = trimesh.Trimesh(
                    vertices=mesh.vertices.copy(), faces=mesh.faces[ids].copy(), process=False)
        if len(self.canonical_meshes) != 8 or len(self.pad_meshes) != 3:
            raise ValueError('expected eight Dex3 mesh links and three annotated pads')
        self._bounds = np.stack([
            np.min([m.bounds[0] for m in self.canonical_meshes.values()], axis=0),
            np.max([m.bounds[1] for m in self.canonical_meshes.values()], axis=0)])

    def _hits(self, name, points, margin):
        bounds, equations = self._hulls[name]
        indices = np.flatnonzero(np.all(points >= bounds[0] - margin, axis=1)
                                 & np.all(points <= bounds[1] + margin, axis=1))
        hits = []
        for first in range(0, len(indices), 256):
            selected = indices[first:first + 256]
            # einsum avoids repeated BLAS thread-pool work for thin matrices.
            distances = np.einsum('ij,kj->ik', points[selected], equations[:, :3]) + equations[:, 3]
            hits.extend(selected[np.all(distances <= margin + 1e-10, axis=1)].tolist())
        return np.asarray(hits, dtype=int)

    def _allowed_pad_contacts(self, name, points):
        allowed = np.zeros(len(points), dtype=bool)
        if name not in self.pad_meshes or not len(points):
            return allowed
        patch = self.pad_meshes[name]
        closest, distance, triangle = trimesh.proximity.closest_point(patch, points)
        signed = np.einsum('ij,ij->i', points - closest, patch.face_normals[triangle])
        allowed = ((distance <= self.settings.pad_contact_distance_m + 1e-10)
                   & (signed >= -self.settings.pad_penetration_m - 1e-10))
        return allowed

    def _pose_check(self, object_points, scene_points, margin, allow_pad_contact,
                    scene_indices=None):
        links = {}
        rejected_object, rejected_scene, allowed_count = set(), set(), set()
        for name in self._hulls:
            object_hits = self._hits(name, object_points, margin)
            allowed = (self._allowed_pad_contacts(name, object_points[object_hits])
                       if allow_pad_contact else np.zeros(len(object_hits), dtype=bool))
            forbidden = object_hits[~allowed]
            scene_hits = self._hits(name, scene_points, margin)
            if scene_indices is not None:
                scene_hits = scene_indices[scene_hits]
            if len(object_hits) or len(scene_hits):
                links[name] = {
                    'forbidden_object_points': int(len(forbidden)),
                    'scene_points': int(len(scene_hits)),
                    'allowed_target_pad_contacts': int(allowed.sum()),
                    'object_point_examples': forbidden[:5].tolist(),
                    'scene_point_examples': scene_hits[:5].tolist(),
                }
            rejected_object.update(forbidden.tolist())
            rejected_scene.update(scene_hits.tolist())
            allowed_count.update(object_hits[allowed].tolist())
        return {
            'observed_points_clear': not (rejected_object or rejected_scene),
            'forbidden_object_points': len(rejected_object),
            'scene_collision_points': len(rejected_scene),
            'allowed_target_pad_contacts': len(allowed_count - rejected_object),
            'links': links,
        }

    def assess_grasp(self, world_canonical, object_points, scene_points):
        g = rigid(world_canonical, 'T_world_canonical')
        obj = _points(object_points, 'object_points')
        scene = _points(scene_points, 'scene_points')
        if not len(obj):
            raise ValueError('target object point cloud is empty')
        obj = (obj - g[:3, 3]) @ g[:3, :3]
        scene = (scene - g[:3, 3]) @ g[:3, :3]
        cfg = self.settings
        scene_count = len(scene)
        # At any translation on this fixed-orientation line, the nearest
        # sampled pose is at most half a step away. Plane inflation by that
        # distance encloses the unsampled swept hull for every observed point.
        half_step = cfg.approach_m / (cfg.approach_samples - 1) / 2
        # Exact broad phase for this OPEN-hand translation family. Retain all
        # scene points that could approach any link; never randomly thin them.
        broad_lower = self._bounds[0] - cfg.clearance_m - half_step
        broad_lower = broad_lower - [0, 0, cfg.approach_m]
        broad_upper = self._bounds[1] + cfg.clearance_m + half_step
        scene_indices = np.flatnonzero(np.all(scene >= broad_lower, axis=1)
                                       & np.all(scene <= broad_upper, axis=1))
        scene = scene[scene_indices]
        final = self._pose_check(obj, scene, cfg.clearance_m, allow_pad_contact=True,
                                 scene_indices=scene_indices)
        path = []
        for distance in np.linspace(cfg.approach_m, 0., cfg.approach_samples):
            shift = np.array([0., 0., distance])
            check = self._pose_check(obj + shift, scene + shift,
                                     cfg.clearance_m + half_step,
                                     allow_pad_contact=bool(distance == 0.),
                                     scene_indices=scene_indices)
            path.append({'distance_behind_grasp_m': float(distance), **check})
        complete = bool(scene_count)
        hand_free = complete and final['observed_points_clear']
        approach_free = complete and all(p['observed_points_clear'] for p in path)
        return {
            'status': ('incomplete_observation' if not complete else
                       'observed_hand_and_approach_clear' if hand_free and approach_free
                       else 'observed_collision_rejected'),
            'hand_collision_free': bool(hand_free),
            'pregrasp_collision_free': bool(approach_free),
            'collision_valid': False,
            'collision_scope': SCOPE,
            'scene_coverage_unknown': True,
            'full_robot_collision_checked': False,
            'finger_closure_checked': False,
            'object_point_count': len(obj), 'scene_point_count': scene_count,
            'scene_broad_phase_point_count': len(scene),
            'point_example_index_space': 'original_input_point_cloud',
            'descriptor_config_sha256': self.config_sha256,
            'hand_configuration': 'descriptor_OPEN',
            'geometry': 'per-link convex hulls of hash-verified real Dex3 visual meshes',
            'transform_contract': 'T_world_link = T_world_canonical @ T_C_wrist @ FK_OPEN(wrist, link)',
            'clearance_m': cfg.clearance_m,
            'approach_m': cfg.approach_m,
            'approach_samples': cfg.approach_samples,
            'approach_extra_inflation_m': half_step,
            'pad_contact_distance_m': cfg.pad_contact_distance_m,
            'pad_penetration_m': cfg.pad_penetration_m,
            'final': final, 'approach': path,
            'limitations': [
                'Observed input points only; gaps, hidden surfaces and scene changes are unknown.',
                'Arm, torso, self collision, reachability, rotation and path to pregrasp are unchecked.',
                'Fixed OPEN fingers only; no closure sweep or grasp stability certificate.',
                'Convex hull and clearance inflation can reject geometrically feasible grasps.',
                'Only target points near annotated distal pads may touch in the final pose.',
                'Scene points are never excluded, including duplicated target points.',
            ],
        }


def assess_grasp(world_canonical, object_points, scene_points, descriptor_path=DEFAULT_DESCRIPTOR):
    """Convenience API; reuse GraspCollisionFilter directly for multiple grasps."""
    return GraspCollisionFilter(descriptor_path).assess_grasp(
        world_canonical, object_points, scene_points)
