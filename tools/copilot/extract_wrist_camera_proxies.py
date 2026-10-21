#!/usr/bin/env python3
"""CPU-only export of conservative wrist-camera boxes from actual USD meshes.

Run with the USD Python libraries available; no SimulationApp or GPU is used.
Instance proxies must be traversed to find the collision Mesh descendants.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from teleop_assist.copilot.asset_paths import CAMERA_SOURCE_RELATIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--usd', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--usd-python-library', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing evidence; choose a new output')
    if args.usd_python_library:
        sys.path.insert(0, str(args.usd_python_library))
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(args.usd))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError('Expected explicitly metre-based robot USD')
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    cameras = []
    for side in ('left', 'right'):
        link_name = side + '_hand_camera_base_link'
        parent_name = side + '_wrist_yaw_link'
        matches = [prim for prim in stage.Traverse() if prim.GetName() == link_name
                   and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(matches) != 1:
            raise ValueError('Unique rigid camera link required')
        link = matches[0]
        joints = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.FixedJoint):
                continue
            joint = UsdPhysics.Joint(prim)
            body0, body1 = list(joint.GetBody0Rel().GetTargets()), list(joint.GetBody1Rel().GetTargets())
            if len(body0) == len(body1) == 1 and str(body1[0]) == str(link.GetPath()) and body0[0].name == parent_name:
                joints.append(joint)
        if len(joints) != 1:
            raise ValueError('Camera must have an explicit fixed joint to its wrist_yaw link')
        joint = joints[0]
        def pose(pos, quat):
            result = np.eye(4)
            result[:3, 3] = pos
            result[:3, :3] = Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
            return result
        parent_to_link = (pose(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()) @
                          np.linalg.inv(pose(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())))
        parent = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        authored = (np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(parent)).T) @
                    np.asarray(cache.GetLocalToWorldTransform(link)).T)
        rest_translation_error = float(np.linalg.norm(parent_to_link[:3, 3] - authored[:3, 3]))
        if (rest_translation_error > 1e-4 or
                not np.allclose(parent_to_link[:3, :3], authored[:3, :3], atol=1e-6, rtol=0)):
            raise ValueError('USD rest transform disagrees with fixed joint attachment')
        all_points, meshes = [], []
        for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or '/collisions/' not in str(prim.GetPath()):
                continue
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            transform = np.asarray(cache.ComputeRelativeTransform(prim, link)[0])
            transformed = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
            if not np.isfinite(transformed).all() or not len(points):
                raise ValueError('Invalid source mesh vertices')
            all_points.append(transformed)
            meshes.append({'path': str(prim.GetPath()), 'vertices': len(points),
                           'transformed_vertices_sha256': hashlib.sha256(transformed.tobytes()).hexdigest()})
        if not all_points:
            raise ValueError('No actual collision Mesh found for ' + link_name)
        points = np.concatenate(all_points)
        lower, upper = points.min(0), points.max(0)
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        index_map = {int(old): new for new, old in enumerate(hull.vertices)}
        faces = []
        for triangle, equation in zip(hull.simplices, hull.equations):
            a, b, c = points[triangle]
            if np.dot(np.cross(b-a, c-a), equation[:3]) < 0:
                triangle = triangle[[0, 2, 1]]
            faces.append([index_map[int(vertex)] for vertex in triangle])
        maximum_plane_violation = max(float(np.max(points @ eq[:3] + eq[3])) for eq in hull.equations)
        if maximum_plane_violation > 1e-9:
            raise ValueError('Exported convex hull does not enclose every source vertex')
        cameras.append({'link': link_name, 'parent_link': parent_name,
                        'parent_to_link': parent_to_link.tolist(),
                        'authored_rest_vs_fixed_joint_translation_error_m': rest_translation_error,
                        'attachment_contract': 'USD fixed joint; runtime observed body-frame agreement required',
                        'box_center_in_link': ((lower + upper) / 2).tolist(),
                        'box_size_xyz': (upper - lower).tolist(),
                        'minimum_in_link': lower.tolist(), 'maximum_in_link': upper.tolist(),
                        'source_meshes': meshes, 'all_source_vertices_inside_box': bool(
                            np.all(points >= lower) and np.all(points <= upper)),
                        'convex_vertices_in_link': hull_points.tolist(), 'convex_triangles': faces,
                        'all_source_vertices_inside_hull_tolerance_m': 1e-9,
                        'maximum_source_hull_plane_violation_m': maximum_plane_violation,
                        'source_fixed_joint': str(joint.GetPrim().GetPath())})
    result = {'schema_version': 2, 'source_usd': str(args.usd.resolve()),
              'source_usd_relative': CAMERA_SOURCE_RELATIVE,
              'source_usd_sha256': hashlib.sha256(args.usd.read_bytes()).hexdigest(),
              'source_urdf_sha256': hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
              'extractor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'method': 'convex hulls and diagnostic boxes enclosing every transformed USD collision mesh vertex',
              'purpose': 'conservative static-simulation collision proxies; hardware mounting not verified',
              'cameras': cameras}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'cameras': [
        {'link': camera['link'], 'source_vertices': sum(m['vertices'] for m in camera['source_meshes']),
         'convex_vertices': len(camera['convex_vertices_in_link']), 'triangles': len(camera['convex_triangles'])}
        for camera in cameras]}))


if __name__ == '__main__':
    main()
