"""Conservative offline collision precheck for the original G1 URDF.

This validates a static, fully specified snapshot and a *joint-linear* path.
It does not authorize a robot command or certify the XR/IK controller's actual
path. Mesh collision shapes use their convex hulls (conservative overestimates).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .controller import rigid
from .kinematics import ARM_JOINT_NAMES, arm_vector, sha256
from .asset_paths import camera_source_usd


@dataclass(frozen=True)
class CollisionConfig:
    clearance_m: float = .001
    max_joint_step_rad: float = .01
    max_samples: int = 512
    joint_limit_tolerance_rad: float = 1e-6
    max_snapshot_skew_s: float = .05
    max_fk_position_error_m: float = .002
    max_fk_orientation_error_rad: float = np.deg2rad(.5)
    max_support_penetration_m: float = .002

    def __post_init__(self):
        values = [self.clearance_m, self.max_joint_step_rad, self.joint_limit_tolerance_rad,
                  self.max_snapshot_skew_s, self.max_fk_position_error_m,
                  self.max_fk_orientation_error_rad, self.max_support_penetration_m]
        if (not np.isfinite(values).all() or not 0 <= self.clearance_m <= .02
                or not 0 < self.max_joint_step_rad <= .1
                or type(self.max_samples) is not int or not 2 <= self.max_samples <= 2048
                or not 0 <= self.joint_limit_tolerance_rad <= 1e-4
                or not 0 <= self.max_snapshot_skew_s <= .1
                or not 0 < self.max_fk_position_error_m <= .005
                or not 0 < self.max_fk_orientation_error_rad <= np.deg2rad(1)
                or not 0 <= self.max_support_penetration_m <= .003):
            raise ValueError("Invalid collision precheck configuration")


def _provenance(mapping, name):
    if not isinstance(mapping, dict) or not isinstance(mapping.get("source"), str) or not mapping["source"]:
        raise ValueError(f"{name} requires nonempty source provenance")
    stamp = mapping.get("acquisition_timestamp")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not np.isfinite(stamp):
        raise ValueError(f"{name} requires finite acquisition_timestamp")
    if not isinstance(mapping.get("clock_id"), str) or not mapping["clock_id"]:
        raise ValueError(f"{name} requires clock_id")


class OriginalArmCollision:
    """CPU Pinocchio/HPP-FCL checker; contains no robot or DDS interface."""

    def __init__(self, repository, config: CollisionConfig | None = None, *, camera_proxy_path=None):
        import pinocchio as pin
        import hppfcl as fcl
        self.pin, self.fcl = pin, fcl
        self.config = config or CollisionConfig()
        self.repository = Path(repository).resolve()
        self.urdf = self.repository / "assets/g1/g1_body29_hand14.urdf"
        self.model = pin.buildModelFromUrdf(str(self.urdf))
        self.geometry = pin.buildGeomFromUrdf(self.model, str(self.urdf), pin.GeometryType.COLLISION,
                                             package_dirs=[str(self.urdf.parent)])
        self.camera_proxy_path = (Path(camera_proxy_path) if camera_proxy_path is not None else
                                  Path(__file__).parent / "assets/g1_wrist_camera_collision_v2.json")
        camera_evidence = self._add_camera_proxies()
        self.data, self.geometry_data = self.model.createData(), pin.GeometryData(self.geometry)
        self.names = tuple(str(n) for n in list(self.model.names)[1:])
        if self.model.nq != 43 or self.model.nv != 43 or len(self.names) != 43:
            raise ValueError("Expected original fixed-base G1 29+14 single-axis joints")
        self.indices = {}
        for name in self.names:
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError("Collision displacement bound supports only single-axis revolute joints")
            self.indices[name] = joint.idx_q
        self.arm_indices = np.array([self.indices[n] for n in ARM_JOINT_NAMES])
        self.objects = list(self.geometry.geometryObjects)
        self.shapes, hashes, self.geometry_links = [], {}, []
        self.levers = np.zeros((len(self.objects), self.model.nq))
        for index, obj in enumerate(self.objects):
            self.geometry_links.append(str(self.model.frames[obj.parentFrame].name))
            shape = obj.geometry
            if hasattr(shape, "buildConvexHull"):
                shape.buildConvexHull(True, "Qt")
                if shape.convex is None:
                    raise ValueError(f"Cannot conservatively convexify {obj.name}")
                shape = shape.convex
            shape.computeLocalAABB()
            low, high = np.asarray(shape.aabb_local.min_), np.asarray(shape.aabb_local.max_)
            if not np.isfinite([low, high]).all() or np.any(low > high):
                raise ValueError(f"Invalid collision shape bounds for {obj.name}")
            self.shapes.append(shape)
            # All-pose upper bounds on distance from an ancestor joint to any
            # point in this shape: sum of rigid translation lengths + radius.
            distance = float(np.linalg.norm(np.maximum(abs(low), abs(high))))
            distance += float(np.linalg.norm(obj.placement.translation))
            joint_id = obj.parentJoint
            while joint_id:
                joint = self.model.joints[joint_id]
                self.levers[index, joint.idx_q] = distance
                distance += float(np.linalg.norm(self.model.jointPlacements[joint_id].translation))
                joint_id = int(self.model.parents[joint_id])
            path = Path(obj.meshPath)
            if path.is_file():
                hashes[str(path)] = sha256(path)
        self.pairs, excluded, self.pair_levers = [], [], []
        for a in range(len(self.objects)):
            for b in range(a + 1, len(self.objects)):
                ja, jb = self.objects[a].parentJoint, self.objects[b].parentJoint
                if ja == jb or self.model.parents[ja] == jb or self.model.parents[jb] == ja:
                    excluded.append([self.objects[a].name, self.objects[b].name])
                    continue
                self.pairs.append((a, b))
                # A common ancestor rotates both shapes rigidly and cannot
                # change their mutual separation; retain only unique ancestors.
                shared = (self.levers[a] > 0) & (self.levers[b] > 0)
                self.pair_levers.append(np.where(shared, 0, self.levers[a] + self.levers[b]))
        self.pair_levers = np.array(self.pair_levers)
        self.manifest = {
            "kind": "original_G1_URDF_conservative_static_snapshot_precheck",
            "urdf": str(self.urdf), "urdf_sha256": sha256(self.urdf),
            "checker_sha256": sha256(__file__), "mesh_sha256": hashes,
            "pinocchio": pin.__version__, "hppfcl": fcl.__version__,
            "joint_names": list(self.names), "geometry_count": len(self.objects),
            "original_urdf_geometry_count": 52, "additional_wrist_camera_geometry_count": 2,
            "geometry_links": self.geometry_links, "self_pairs_checked": len(self.pairs),
            "excluded_adjacent_or_same_rigid_body_pairs": excluded,
            "mesh_treatment": "convex_hulls_of_original_collision_meshes; original primitives retained",
            "path": "joint-linear interpolation of both arms; remaining 29 joints and base held fixed",
            "continuous_bound": "sum of ancestor lever bounds times half interval joint displacement; shared ancestor motion cancels for self pairs",
            "configuration": asdict(self.config), "robot_command_sent": False,
            "live_execution_authorized": False,
            "entire_physical_robot_geometry_validated": False,
            "wrist_camera_collision_coverage_verified": True,
            "wrist_camera_collision_source": camera_evidence,
            "known_geometry_coverage_exclusions": ["d435_link", "mid360_link"],
            "limitations": ["URDF collision geometry is not proven identical to USD or hardware geometry",
                            "Wrist-camera mount meshes are added from the pinned USD; actual hardware mounts remain unverified",
                            "D435/Mid360 accessories have no explicit collision geometry in this URDF/USD",
                            "Adjacent link/rigid-body contacts are excluded by structural rule, listed explicitly",
                            "No guarantee for the actual XR/IK trajectory, future finger motion, or moving obstacles",
                            "Snapshot provenance is retained; no live freshness is inferred"],
        }

    def _add_camera_proxies(self):
        """Add source-verified USD CAD hulls and fixed frames missing in URDF."""
        source = json.loads(self.camera_proxy_path.read_text())
        if source.get("schema_version") != 2 or source.get("source_urdf_sha256") != sha256(self.urdf):
            raise ValueError("Camera proxy schema or original URDF hash mismatch")
        usd = camera_source_usd(source)
        entries = source.get("cameras", [])
        required = {side + "_hand_camera_base_link": side + "_wrist_yaw_link" for side in ("left", "right")}
        if len(entries) != 2 or {item.get("link") for item in entries} != set(required):
            raise ValueError("Both explicitly named wrist-camera proxies are required")
        records = []
        for item in entries:
            name = item["link"]
            if item.get("parent_link") != required[name] or not self.model.existFrame(required[name]):
                raise ValueError("Camera proxy parent is not the matching wrist_yaw link")
            if self.model.existFrame(name):
                raise ValueError("Unexpected existing wrist-camera URDF frame; re-audit source models")
            if (item.get("all_source_vertices_inside_box") is not True or
                    not np.isfinite(item.get("maximum_source_hull_plane_violation_m", float("nan"))) or
                    item["maximum_source_hull_plane_violation_m"] > 1e-9):
                raise ValueError("Missing source-vertex hull containment evidence")
            vertices = np.asarray(item.get("convex_vertices_in_link"), dtype=float)
            triangles = np.asarray(item.get("convex_triangles"))
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or not 4 <= len(vertices) <= 5000 or
                    not np.isfinite(vertices).all() or triangles.ndim != 2 or triangles.shape[1] != 3 or
                    not np.issubdtype(triangles.dtype, np.integer) or np.any(triangles < 0) or
                    np.any(triangles >= len(vertices))):
                raise ValueError("Malformed camera collision hull")
            fcl_vertices, fcl_triangles = self.fcl.StdVec_Vec3f(), self.fcl.StdVec_Triangle()
            for vertex in vertices:
                fcl_vertices.append(vertex)
            for triangle in triangles:
                fcl_triangles.append(self.fcl.Triangle(*[int(i) for i in triangle]))
            shape = self.fcl.Convex(fcl_vertices, fcl_triangles)
            parent_id = self.model.getFrameId(required[name])
            parent = self.model.frames[parent_id]
            joint_id = parent.parentJoint
            placement = parent.placement * self.pin.SE3(rigid(item["parent_to_link"], "camera_fixed_joint"))
            frame_id = self.model.addFrame(self.pin.Frame(name, joint_id, parent_id, placement, self.pin.BODY))
            self.geometry.addGeometryObject(self.pin.GeometryObject(
                name + "_usd_convex", joint_id, frame_id, placement, shape, str(self.camera_proxy_path)))
            records.append({"link": name, "parent_link": required[name],
                            "convex_vertices": len(vertices), "convex_triangles": len(triangles),
                            "source_meshes": item["source_meshes"],
                            "fixed_joint": item["source_fixed_joint"],
                            "parent_to_link": item["parent_to_link"]})
        return {"asset": str(self.camera_proxy_path), "asset_sha256": sha256(self.camera_proxy_path),
                "source_usd": str(usd), "source_usd_sha256": source["source_usd_sha256"],
                "method": source["method"], "source_sha256_verified": True, "cameras": records,
                "hardware_geometry_verified": False}

    def _q(self, named):
        if not isinstance(named, dict) or set(named) != set(self.names):
            missing = sorted(set(self.names) - set(named or {})) if isinstance(named, dict) else list(self.names)
            raise ValueError(f"All 43 joint positions are required by name; missing={missing}")
        q = np.zeros(self.model.nq)
        for name in self.names:
            value = named[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Nonfinite or invalid joint value for {name}")
            q[self.indices[name]] = value
        tol = self.config.joint_limit_tolerance_rad
        if np.any(q < self.model.lowerPositionLimit - tol) or np.any(q > self.model.upperPositionLimit + tol):
            raise ValueError("Robot snapshot violates original URDF joint limits")
        return q

    def compare_observed_frames(self, q, base, observed):
        """Independent URDF FK ↔ captured USD body transforms, including +50 mm TCP."""
        if not isinstance(observed, dict):
            return {"valid": False, "reason": "observed_body_world_poses_missing", "frames": []}
        self.pin.framesForwardKinematics(self.model, self.data, q)
        required = set(self.geometry_links) | {"left_wrist_yaw_link", "right_wrist_yaw_link"}
        missing = sorted(required - set(observed))
        rows, relative_rotations = [], []
        for name, matrix in observed.items():
            if not self.model.existFrame(name):
                continue
            measured = rigid(matrix, "observed_body_world_pose")
            predicted = base @ self.data.oMf[self.model.getFrameId(name)].homogeneous
            row = {"body": name,
                   "position_error_m": float(np.linalg.norm(predicted[:3, 3] - measured[:3, 3])),
                   "orientation_error_rad": None}
            relative_rotations.append(predicted[:3, :3] @ measured[:3, :3].T)
            if name in {"left_wrist_yaw_link", "right_wrist_yaw_link"}:
                tcp = np.eye(4)
                tcp[0, 3] = .05
                row["original_50mm_tcp_world"] = (predicted @ tcp).tolist()
                row["observed_50mm_tcp_world"] = (measured @ tcp).tolist()
                row["original_50mm_tcp_base"] = (np.linalg.inv(base) @ predicted @ tcp).tolist()
            rows.append(row)
        if relative_rotations:
            # Same SciPy orthogonalization and angle metric, including inputs
            # within rigid()'s nonorthogonality tolerance; no trace shortcut.
            angles = Rotation.from_matrix(np.asarray(relative_rotations)).magnitude()
            for row, angle in zip(rows, angles):
                row["orientation_error_rad"] = float(angle)
        max_pos = max((r["position_error_m"] for r in rows), default=float("inf"))
        max_rot = max((r["orientation_error_rad"] for r in rows), default=float("inf"))
        valid = bool(not missing and max_pos <= self.config.max_fk_position_error_m
                     and max_rot <= self.config.max_fk_orientation_error_rad)
        return {"valid": valid, "reason": "within_tolerance" if valid else "frame_missing_or_mismatch",
                "missing_required_bodies": missing, "compared_count": len(rows),
                "max_position_error_m": max_pos if rows else None,
                "max_orientation_error_rad": max_rot if rows else None, "frames": rows}

    def _obstacles(self, scene):
        if scene.get("geometry_complete") is not True or scene.get("static") is not True:
            raise ValueError("Scene must explicitly describe complete, static collision geometry")
        entries = scene.get("obstacles")
        if not isinstance(entries, list) or not entries or len(entries) > 64:
            raise ValueError("Scene requires 1..64 explicit obstacles (including ground/target when present)")
        obstacles, ids = [], set()
        for item in entries:
            name = item.get("id")
            if not isinstance(name, str) or not name or name in ids:
                raise ValueError("Unique nonempty obstacle IDs required")
            ids.add(name)
            kind = item.get("type")
            if kind == "plane":
                if item.get("normal") != [0, 0, 1] or not np.isfinite(item.get("offset", float("nan"))):
                    raise ValueError("Only an explicitly world-Z-up ground plane is supported")
                matrix = np.eye(4)
                matrix[2, 3] = float(item["offset"])
                kind = "ground_halfspace"
            else:
                matrix = rigid(item.get("world_pose"), "obstacle_world_pose")
            if kind == "box":
                sizes = np.asarray(item.get("size_xyz"), dtype=float)
                if sizes.shape != (3,) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
                    raise ValueError("Box dimensions must be three finite positive values")
                shape = self.fcl.Box(*sizes)
            elif kind in {"sphere", "cylinder"}:
                radius = float(item.get("radius_m", item.get("radius", 0)))
                length = float(item.get("length_m", item.get("length", 0))) if kind == "cylinder" else 1.
                if not np.isfinite([radius, length]).all() or min(radius, length) <= 0:
                    raise ValueError("Sphere/cylinder dimensions must be finite and positive")
                shape = self.fcl.Sphere(radius) if kind == "sphere" else self.fcl.Cylinder(radius, length)
            elif kind == "ground_halfspace":
                if not np.allclose(matrix[:3, :3], np.eye(3), atol=1e-9):
                    raise ValueError("Ground halfspace requires world Z-up orientation")
                shape = self.fcl.Halfspace(np.array([0., 0., 1.]), 0.)
            else:
                raise ValueError(f"Unsupported obstacle geometry: {kind}")
            obstacles.append((name, kind, shape, self.fcl.Transform3f(matrix[:3, :3], matrix[:3, 3])))
        allowed = {}
        for item in scene.get("allowed_static_contacts", []):
            key = (item.get("robot_link", item.get("link")), item.get("obstacle_id", item.get("obstacle")))
            match = [o for o in obstacles if o[0] == key[1]]
            if (key[0] not in {"left_ankle_roll_link", "right_ankle_roll_link"} or not match
                    or match[0][1] != "ground_halfspace"):
                raise ValueError("Only explicit fixed supporting-foot ↔ ground contacts may be permitted")
            requested = item.get("max_penetration_m", self.config.max_support_penetration_m)
            if not np.isfinite(requested) or not 0 <= requested <= self.config.max_support_penetration_m:
                raise ValueError("Requested support penetration exceeds configured bound")
            allowed[key] = float(requested)
        return obstacles, allowed

    def _distance(self, shape_a, pose_a, shape_b, pose_b):
        value = self.fcl.distance(shape_a, pose_a, shape_b, pose_b,
                                  self.fcl.DistanceRequest(), self.fcl.DistanceResult())
        if not math.isfinite(value):
            raise ValueError("Nonfinite FCL distance")
        return float(value)

    def _path_distance(self, distances, key, shape_a, pose_a, shape_b, pose_b):
        # These dictionaries are private to one validated request and one
        # bit-identical evaluated q. Never memoize a verdict or motion bound.
        if distances is not None and key in distances:
            value = distances[key]
        else:
            value = self._distance(shape_a, pose_a, shape_b, pose_b)
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Nonfinite or invalid cached FCL distance")
        if distances is not None:
            distances[key] = value
        return value

    def _check_path(self, initial, final, base, obstacles, allowed, *, _distance_cache=None):
        delta = final - initial
        intervals = int(np.ceil(np.max(np.abs(delta)) / self.config.max_joint_step_rad))
        if intervals + 1 > self.config.max_samples:
            return {"collision_valid": False, "reason": "path_sample_budget_exceeded", "samples_checked": 0,
                    "required_samples": intervals + 1}
        half_step = abs(delta) / (2 * max(1, intervals))
        environment_motion = self.levers @ half_step
        pair_motion = self.pair_levers @ half_step
        minimum_distance = float("inf")
        minimum_bound = float("inf")
        worst = None
        support_contacts = []
        samples_checked = 0
        for sample, alpha in enumerate(np.linspace(0., 1., intervals + 1)):
            q = initial + alpha * delta
            distances = None
            if _distance_cache is not None:
                # Key the actually evaluated full state, including fingers
                # and signed zero. No tolerance, rounding or alpha=1 shortcut.
                key = (q.dtype.str, q.shape, q.tobytes())
                if key in _distance_cache:
                    distances = _distance_cache[key]
                elif len(_distance_cache) < 2:
                    distances = _distance_cache[key] = {}
            self.pin.updateGeometryPlacements(self.model, self.data, self.geometry, self.geometry_data, q)
            poses = []
            for placement in self.geometry_data.oMg:
                matrix = base @ placement.homogeneous
                poses.append(self.fcl.Transform3f(matrix[:3, :3], matrix[:3, 3]))
            samples_checked += 1
            for index, (a, b) in enumerate(self.pairs):
                key = ('self', index)
                distance = self._path_distance(distances,key,
                    self.shapes[a], poses[a], self.shapes[b], poses[b])
                lower = distance - pair_motion[index]
                minimum_distance = min(minimum_distance, distance)
                if lower < minimum_bound:
                    minimum_bound = lower
                    worst = {"type": "self", "a": self.objects[a].name, "b": self.objects[b].name,
                             "sample": sample, "alpha": float(alpha), "distance_m": distance,
                             "motion_bound_m": float(pair_motion[index])}
                if distance <= self.config.clearance_m:
                    witness = {"type": "self", "a": self.objects[a].name, "b": self.objects[b].name,
                               "sample": sample, "alpha": float(alpha), "distance_m": distance,
                               "motion_bound_m": float(pair_motion[index])}
                    return {"collision_valid": False, "reason": "self_collision_or_clearance_violation",
                            "samples_checked": samples_checked, "total_samples": intervals + 1, "witness": witness,
                            "minimum_sample_distance_m": minimum_distance,
                            "conservative_path_clearance_lower_bound_m": minimum_bound}
            for a, link in enumerate(self.geometry_links):
                for obstacle_index, (name, kind, shape, pose) in enumerate(obstacles):
                    key = ('environment', a, obstacle_index)
                    distance = self._path_distance(distances,key,
                        self.shapes[a], poses[a], shape, pose)
                    if (link, name) in allowed:
                        if environment_motion[a] > 0:
                            raise ValueError("An allowed support contact would move along this path")
                        if sample == 0:
                            support_contacts.append({"robot_geometry": self.objects[a].name,
                                                     "obstacle": name, "distance_m": distance})
                        if distance >= -allowed[(link, name)]:
                            continue
                    lower = distance - environment_motion[a]
                    minimum_distance = min(minimum_distance, distance)
                    if lower < minimum_bound:
                        minimum_bound = lower
                        worst = {"type": "environment", "a": self.objects[a].name, "b": name,
                                 "sample": sample, "alpha": float(alpha), "distance_m": distance,
                                 "motion_bound_m": float(environment_motion[a])}
                    if distance <= self.config.clearance_m:
                        witness = {"type": "environment", "a": self.objects[a].name, "b": name,
                                   "sample": sample, "alpha": float(alpha), "distance_m": distance,
                                   "motion_bound_m": float(environment_motion[a])}
                        return {"collision_valid": False, "reason": "environment_collision_or_clearance_violation",
                                "samples_checked": samples_checked, "total_samples": intervals + 1, "witness": witness,
                                "minimum_sample_distance_m": minimum_distance,
                                "conservative_path_clearance_lower_bound_m": minimum_bound,
                                "allowed_support_contacts": support_contacts}
        clear = bool(minimum_bound > self.config.clearance_m)
        return {"collision_valid": clear,
                "reason": "conservative_joint_linear_path_clear" if clear else "insufficient_clearance_for_between_sample_bound",
                "samples_checked": samples_checked, "total_samples": intervals + 1, "witness": worst,
                "minimum_sample_distance_m": minimum_distance,
                "conservative_path_clearance_lower_bound_m": minimum_bound,
                "allowed_support_contacts": support_contacts}

    def validate_request(self, request, *, endpoint_only=False):
        """Return an enriched batch; missing state/scene always clears permissions."""
        if type(endpoint_only) is not bool:
            raise ValueError('endpoint_only must be an explicit boolean')
        if not isinstance(request, dict) or not isinstance(request.get("batch"), dict):
            raise ValueError("A candidate batch is required")
        batch = deepcopy(request["batch"])
        candidates = batch.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 128:
            raise ValueError("Candidate list must contain 1..128 items")
        if any(not isinstance(c, dict) for c in candidates):
            raise ValueError("Candidates must be mappings")
        for candidate in candidates:
            candidate["collision_valid"] = False
        started = time.perf_counter()
        metadata = {"manifest": self.manifest,
                    "scope": ("sim_candidate_endpoint_only" if endpoint_only else
                              "offline_static_snapshot_joint_linear_path_only"),
                    "path_check_requested": not endpoint_only,
                    "path_validated": False,
                    "live_execution_authorized": False}
        try:
            state, scene = request.get("robot_state"), request.get("scene")
            _provenance(state, "robot_state")
            _provenance(scene, "scene")
            if state.get("complete") is not True:
                raise ValueError("Robot state must explicitly be complete")
            if (state["clock_id"] != scene["clock_id"] or
                    abs(state["acquisition_timestamp"] - scene["acquisition_timestamp"]) > self.config.max_snapshot_skew_s):
                raise ValueError("Robot and scene snapshots have different clocks or excessive acquisition skew")
            q = self._q(state.get("joint_positions"))
            base = rigid(state.get("base_in_world"), "base_in_world")
            obstacles, allowed = self._obstacles(scene)
            comparison = self.compare_observed_frames(q, base, state.get("observed_body_world_poses"))
            metadata["urdf_vs_observed_frames"] = comparison
            if not comparison["valid"]:
                raise ValueError("URDF/current simulation body-frame mapping not verified")
            metadata["robot_state"] = deepcopy(state)
            metadata["scene"] = deepcopy(scene)
            # Base, converted scene geometry and robot geometry remain fixed
            # throughout this synchronous request. Nothing is kept on self or
            # accepted from a caller. Each candidate inherits only initial q.
            initial_distances = {}
            initial_check = self._check_path(q, q, base, obstacles, allowed,
                                             _distance_cache=initial_distances)
            metadata["current_state_precheck"] = initial_check
            if not initial_check["collision_valid"]:
                raise ValueError("Current robot snapshot already fails collision/clearance precheck")
            for candidate in candidates:
                result = {"collision_valid": False, "reason": "candidate_ik_unvalidated"}
                try:
                    kin = candidate.get("kinematic_validation", {})
                    if candidate.get("ik_valid") is True and kin.get("ik_valid") is True:
                        q14 = arm_vector(kin.get("q_solution"), "candidate_q_solution")
                        if not np.allclose(arm_vector(kin.get("q_initial")), q[self.arm_indices], atol=1e-6, rtol=0):
                            raise ValueError("IK solution and robot snapshot use different initial joint states")
                        final = q.copy()
                        final[self.arm_indices] = q14
                        tol = self.config.joint_limit_tolerance_rad
                        if np.any(final < self.model.lowerPositionLimit - tol) or np.any(final > self.model.upperPositionLimit + tol):
                            raise ValueError("Candidate solution violates joint limits")
                        self.pin.framesForwardKinematics(self.model, self.data, final)
                        tcp = np.eye(4)
                        tcp[0, 3] = .05
                        full_fk = {}
                        for side, target in (("left", kin.get("left_target_base_ee")),
                                             ("right", candidate.get("base_ee"))):
                            expected = rigid(target, side + "_candidate_target")
                            measured = self.data.oMf[self.model.getFrameId(side + "_wrist_yaw_link")].homogeneous @ tcp
                            position_error = float(np.linalg.norm(measured[:3, 3] - expected[:3, 3]))
                            rotation_error = float(Rotation.from_matrix(measured[:3, :3] @ expected[:3, :3].T).magnitude())
                            full_fk[side] = {"position_error_m": position_error, "orientation_error_rad": rotation_error}
                            if position_error > .01 or rotation_error > np.deg2rad(10):
                                raise ValueError("Full q43 FK does not reproduce reduced original IK endpoint")
                        distances = {key: dict(values) for key, values in initial_distances.items()}
                        endpoint = self._check_path(final, final, base, obstacles, allowed,
                                                    _distance_cache=distances)
                        result = dict(endpoint) if endpoint_only else self._check_path(
                            q, final, base, obstacles, allowed, _distance_cache=distances)
                        result["scope"] = metadata["scope"]
                        result["path_validated"] = not endpoint_only and bool(result["collision_valid"])
                        result["live_execution_authorized"] = False
                        result["endpoint_precheck"] = endpoint
                        result["full_q43_endpoint_fk"] = full_fk
                        result["q_initial_43"] = q.tolist()
                        result["q_final_43"] = final.tolist()
                        result["manual_fingers_held_at_observed_values"] = True
                        result["path_is_not_actual_XR_controller_trajectory"] = True
                except (TypeError, ValueError, KeyError) as exc:
                    result = {"collision_valid": False, "reason": f"invalid_candidate:{exc}"}
                candidate["collision_valid"] = result["collision_valid"]
                candidate["arm_collision_validation"] = result
            metadata["reason"] = "evaluated"
        except (TypeError, ValueError, KeyError) as exc:
            metadata["reason"] = f"incomplete_or_invalid_precheck:{exc}"
            for candidate in candidates:
                candidate["arm_collision_validation"] = {"collision_valid": False, "reason": metadata["reason"]}
        metadata["accepted_count"] = sum(c["collision_valid"] for c in candidates)
        metadata["path_validated"] = not endpoint_only and metadata["accepted_count"] == len(candidates)
        metadata["elapsed_s"] = float(time.perf_counter() - started)
        try:
            metadata["snapshot_sha256"] = hashlib.sha256(json.dumps(
                {"robot_state": request.get("robot_state"), "scene": request.get("scene")},
                sort_keys=True, allow_nan=False).encode()).hexdigest()
        except (TypeError, ValueError):
            metadata["snapshot_sha256"] = None
        batch["arm_collision_validation"] = metadata
        return batch


def robot_state_from_capture(path):
    """Read actual Isaac body state; quaternion convention is explicitly wxyz."""
    data = np.load(path, allow_pickle=False)
    names = [str(x) for x in data["body_names"]]
    poses = np.asarray(data["body_poses"])
    if poses.shape != (len(names), 7):
        raise ValueError("Capture body_poses must be position + wxyz quaternion")
    matrices = {}
    for name, pose in zip(names, poses):
        matrix = np.eye(4)
        matrix[:3, 3] = pose[:3]
        matrix[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        matrices[name] = matrix.tolist()
    return {"joint_positions": {str(n): float(q) for n, q in zip(data["joint_names"], data["joint_positions"])},
            "base_in_world": data["T_world_base"].tolist(),
            "observed_body_world_poses": matrices, "complete": True,
            "acquisition_timestamp": float(data["timestamp"]), "clock_id": str(data["clock_id"]),
            "source": str(Path(path).resolve()), "source_sha256": sha256(path),
            "body_pose_quaternion_convention": "wxyz"}
