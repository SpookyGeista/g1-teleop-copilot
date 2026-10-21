"""Offline candidate validation through the unmodified original G1_29 IK.

Instantiate in a dedicated CPU subprocess with Unitree's Pinocchio/CasADi
environment. No robot, DDS, image server or teleoperation launcher is imported.
The six convergence iterations are numerical IK refinement, not measured robot
motion. A successful endpoint does not validate the intervening trajectory.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .controller import rigid


ARM_JOINT_NAMES = tuple(f"{side}_{joint}_joint" for side in ("left", "right")
                        for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                                      "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))

# Numerical pose-resolution bounds, not relaxed IK or collision tolerances.
ALREADY_AT_TARGET_POSITION_TOLERANCE_M = 1e-5
ALREADY_AT_TARGET_ORIENTATION_TOLERANCE_RAD = 1e-4


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_vector(value, name="current_arm_q"):
    value = np.asarray(value, dtype=float)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain 14 finite joint angles in radians")
    return value.copy()


@dataclass(frozen=True)
class IKValidationConfig:
    convergence_steps: int = 6
    max_position_error_m: float = .01
    max_orientation_error_rad: float = np.deg2rad(10)
    joint_limit_tolerance_rad: float = 1e-6
    max_candidates: int = 128

    def __post_init__(self):
        if (type(self.convergence_steps) is not int or not 4 <= self.convergence_steps <= 6
                or type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 128
                or not np.isfinite([self.max_position_error_m,
                                    self.max_orientation_error_rad,
                                    self.joint_limit_tolerance_rad]).all()
                or not 0 < self.max_position_error_m <= .02
                or not 0 < self.max_orientation_error_rad <= np.deg2rad(20)
                or not 0 <= self.joint_limit_tolerance_rad <= 1e-4):
            raise ValueError("Invalid candidate IK validation configuration")


@contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class OriginalIKValidator:
    """Actual original solver plus independent FK and finite/limit gates.

    This is deliberately synchronous; use the CLI or a dedicated worker, never
    invoke it inside an XR event loop. Original cache files are confined to a
    fresh temporary directory and are never loaded from the user's checkout.
    """
    def __init__(self, repository, config: IKValidationConfig | None = None):
        self.config = config or IKValidationConfig()
        self.repository = Path(repository).resolve()
        self.source_path = self.repository / "teleop/robot_control/robot_arm_ik.py"
        self.urdf_path = self.repository / "assets/g1/g1_body29_hand14.urdf"
        self.filter_path = self.repository / "teleop/utils/weighted_moving_filter.py"
        for path in (self.source_path, self.urdf_path, self.filter_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self._source_hashes = {str(path): sha256(path) for path in
                               (self.source_path, self.urdf_path, self.filter_path)}
        import pinocchio as pin
        import casadi
        self.pin = pin
        sys.path.insert(0, str(self.repository))
        module_name = "_copilot_original_g1_ik_" + self._source_hashes[str(self.source_path)][:12]
        spec = importlib.util.spec_from_file_location(module_name, self.source_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if Path(inspect.getfile(module.WeightedMovingFilter)).resolve() != self.filter_path:
            raise RuntimeError("Original IK imported a smoothing filter from a different repository")
        self._filter_type = module.WeightedMovingFilter
        # Original relative URDF paths and pickle cache are satisfied in /tmp.
        # The original checkout and original source are never rewritten.
        with tempfile.TemporaryDirectory(prefix="punktnull_copilot_ik_") as directory:
            stage = Path(directory)
            (stage / "assets").symlink_to(self.repository / "assets", target_is_directory=True)
            (stage / "teleop").mkdir()
            with _working_directory(stage / "teleop"):
                self.solver = module.G1_29_ArmIK(Unit_Test=False, Visualization=False)
        self.model = self.solver.reduced_robot.model
        actual_names = tuple(str(name) for name in list(self.model.names)[1:])
        if self.model.nq != 14 or self.model.nv != 14 or actual_names != ARM_JOINT_NAMES:
            raise ValueError(f"Unexpected original reduced arm joint order: {actual_names}")
        # Original code adds EE frames after creating Data. Use new Data for FK.
        self.data = self.model.createData()
        self.left_id, self.right_id = self.model.getFrameId("L_ee"), self.model.getFrameId("R_ee")
        self.lower, self.upper = (np.asarray(self.model.lowerPositionLimit).copy(),
                                  np.asarray(self.model.upperPositionLimit).copy())
        for name, identifier in (("L_ee", self.left_id), ("R_ee", self.right_id)):
            frame = self.model.frames[identifier]
            if (not np.allclose(frame.placement.translation, [.05, 0, 0], atol=1e-12)
                    or not np.allclose(frame.placement.rotation, np.eye(3), atol=1e-12)):
                raise ValueError(f"{name} is not the original 50 mm wrist TCP")
        try:
            revision = subprocess.check_output(["git", "-C", str(self.repository), "rev-parse", "HEAD"],
                                                text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        reduced_collision = getattr(self.solver.reduced_robot, "collision_model", None)
        self.manifest = {
            "kind": "original_Unitree_G1_29_IK_endpoint_validation",
            "repository": str(self.repository), "revision": revision,
            "source_sha256": self._source_hashes[str(self.source_path)],
            "urdf_sha256": self._source_hashes[str(self.urdf_path)],
            "filter_sha256": self._source_hashes[str(self.filter_path)],
            "validator_sha256": sha256(__file__),
            "python": sys.executable, "pinocchio": pin.__version__, "casadi": casadi.__version__,
            "joint_order": list(ARM_JOINT_NAMES), "configuration": asdict(self.config),
            "ee_frame": "original L_ee/R_ee, 0.05 m along wrist_yaw local X",
            "per_candidate_reset": "current arm q, new original weighted filter, zero initial constraint multipliers",
            "already_at_target": {
                "method": "independent_original_50mm_EE_FK_of_both_measured_arms",
                "maximum_position_tolerance_m": ALREADY_AT_TARGET_POSITION_TOLERANCE_M,
                "maximum_orientation_tolerance_rad": ALREADY_AT_TARGET_ORIENTATION_TOLERANCE_RAD,
                "regular_residual_limits_also_apply": True,
                "optimizer_skipped_only_for_verified_current_pose": True},
            "reduced_collision_geometry_count": (len(reduced_collision.geometryObjects)
                                                   if reduced_collision is not None else 0),
            "collision_validated": False,
            "collision_reason": "No scene or current finger configuration supplied; original IK locks fingers/waist/legs at zero",
            "trajectory_validated": False, "joint_state_freshness_validated": False,
            "dds_imported": any(name.startswith("unitree_sdk2py") for name in sys.modules),
            "robot_command_sent": False,
        }
        if self.manifest["dds_imported"]:
            raise RuntimeError("IK validation process unexpectedly contains a DDS SDK import")

    def forward(self, q):
        q = arm_vector(q)
        self.pin.framesForwardKinematics(self.model, self.data, q)
        return (self.data.oMf[self.left_id].homogeneous.copy(),
                self.data.oMf[self.right_id].homogeneous.copy())

    def _within_limits(self, q):
        tolerance = self.config.joint_limit_tolerance_rad
        return bool(np.all(q >= self.lower - tolerance) and np.all(q <= self.upper + tolerance))

    def validate_pose(self, right_base_ee, left_base_ee, current_arm_q):
        started = time.perf_counter()
        left = rigid(left_base_ee, "left_base_ee")
        right = rigid(right_base_ee, "candidate_base_ee")
        initial = arm_vector(current_arm_q)
        if not self._within_limits(initial):
            raise ValueError("Current arm joint positions violate original URDF limits")
        # Original IK includes regularization and smoothing, so re-solving an
        # already reached pose can move the arm. A measured pose that passes
        # independent FK needs no optimizer. Full q43 collision/freshness and
        # the actual zero-length path are still checked by the motion guard.
        left_fk, right_fk = self.forward(initial)
        current = {
            "iteration": 0, "solver_success": None,
            "solver_status": "not_run_already_at_target", "limits_valid": True,
            "left_position_error_m": float(np.linalg.norm(left_fk[:3, 3] - left[:3, 3])),
            "right_position_error_m": float(np.linalg.norm(right_fk[:3, 3] - right[:3, 3])),
            "left_orientation_error_rad": float(Rotation.from_matrix(left_fk[:3, :3] @ left[:3, :3].T).magnitude()),
            "right_orientation_error_rad": float(Rotation.from_matrix(right_fk[:3, :3] @ right[:3, :3].T).magnitude()),
        }
        position_tolerance = min(ALREADY_AT_TARGET_POSITION_TOLERANCE_M, self.config.max_position_error_m)
        orientation_tolerance = min(ALREADY_AT_TARGET_ORIENTATION_TOLERANCE_RAD, self.config.max_orientation_error_rad)
        if (max(current['left_position_error_m'], current['right_position_error_m']) <= position_tolerance
                and max(current['left_orientation_error_rad'], current['right_orientation_error_rad']) <= orientation_tolerance):
            return {
                "ik_valid": True, "collision_valid": False, "reason": "already_at_target",
                "solver_invoked": False, "validation_method": "independent_current_pose_FK",
                "q_solution": initial.tolist(), "q_initial": initial.tolist(),
                "left_target_base_ee": left.tolist(), "right_target_base_ee": right.tolist(),
                "iterations": [], "final": current, "numerical_convergence_steps": 0,
                "elapsed_s": float(time.perf_counter() - started),
                "joint_state_freshness_validated": False, "trajectory_validated": False,
            }
        self.solver.init_data = initial.copy()
        self.solver.smooth_filter = self._filter_type(np.array([.4, .3, .2, .1]), 14)
        self.solver.opti.set_initial(self.solver.var_q, initial)
        self.solver.opti.set_initial(self.solver.opti.lam_g,
                                     np.zeros(self.solver.opti.lam_g.numel()))
        cursor = initial.copy()
        rows, error = [], None
        for index in range(self.config.convergence_steps):
            try:
                q, torque = self.solver.solve_ik(left.copy(), right.copy(), cursor.copy(), np.zeros(14))
                q = arm_vector(q, "solver_q")
                torque = arm_vector(torque, "solver_feedforward")
                stats = self.solver.opti.stats()
                success = bool(stats.get("success", False))
                left_fk, right_fk = self.forward(q)
                row = {
                    "iteration": index + 1,
                    "solver_success": success,
                    "solver_status": str(stats.get("return_status", "unknown")),
                    "limits_valid": self._within_limits(q),
                    "left_position_error_m": float(np.linalg.norm(left_fk[:3, 3] - left[:3, 3])),
                    "right_position_error_m": float(np.linalg.norm(right_fk[:3, 3] - right[:3, 3])),
                    "left_orientation_error_rad": float(Rotation.from_matrix(left_fk[:3, :3] @ left[:3, :3].T).magnitude()),
                    "right_orientation_error_rad": float(Rotation.from_matrix(right_fk[:3, :3] @ right[:3, :3].T).magnitude()),
                }
                rows.append(row)
                cursor = q.copy()
                if not success or not row["limits_valid"]:
                    error = "original_solver_failed" if not success else "solver_joint_limit_violation"
                    break
            except Exception as exc:
                # Original IK can return fallback/debug iterates after failure.
                # Any exception or failed status remains an explicit rejection.
                error = f"solver_exception:{type(exc).__name__}:{exc}"
                break
        final = rows[-1] if rows else None
        residual_ok = bool(final and
                           max(final["left_position_error_m"], final["right_position_error_m"])
                           <= self.config.max_position_error_m and
                           max(final["left_orientation_error_rad"], final["right_orientation_error_rad"])
                           <= self.config.max_orientation_error_rad)
        valid = bool(error is None and len(rows) == self.config.convergence_steps and residual_ok)
        return {
            "ik_valid": valid, "collision_valid": False,
            "solver_invoked": True, "validation_method": "original_IK_plus_independent_FK",
            "reason": "endpoint_ik_valid_collision_unchecked" if valid else error or "fk_residual_exceeds_threshold",
            "q_solution": cursor.tolist(), "q_initial": initial.tolist(),
            "left_target_base_ee": left.tolist(), "right_target_base_ee": right.tolist(),
            "iterations": rows, "final": final,
            "numerical_convergence_steps": self.config.convergence_steps,
            "elapsed_s": float(time.perf_counter() - started),
            "joint_state_freshness_validated": False, "trajectory_validated": False,
        }

    def validate_request(self, request):
        if not isinstance(request, dict) or not isinstance(request.get("batch"), dict):
            raise ValueError("Request requires a batch mapping")
        batch = deepcopy(request["batch"])
        candidates = batch.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= self.config.max_candidates:
            raise ValueError("Candidate list missing or outside configured size limit")
        if any(not isinstance(item, dict) for item in candidates):
            raise ValueError("Each candidate must be a mapping")
        ids = [item.get("candidate_id") for item in candidates]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("Candidate IDs must be nonempty and unique")
        current = arm_vector(request["current_arm_q"])
        left = rigid(request["left_base_ee"], "left_base_ee")
        for candidate in candidates:
            try:
                result = self.validate_pose(candidate.get("base_ee"), left, current)
            except (TypeError, ValueError) as exc:
                result = {"ik_valid": False, "collision_valid": False,
                          "reason": f"invalid_candidate_input:{exc}", "q_solution": None,
                          "joint_state_freshness_validated": False, "trajectory_validated": False}
            candidate["ik_valid"] = result["ik_valid"]
            candidate["collision_valid"] = False
            candidate["kinematic_validation"] = result
        unchanged = all(sha256(path) == value for path, value in self._source_hashes.items())
        if not unchanged:
            for candidate in candidates:
                candidate["ik_valid"] = False
            raise RuntimeError("Original IK sources changed during validation")
        batch["kinematic_validation"] = {
            "manifest": self.manifest,
            "accepted_count": sum(candidate["ik_valid"] for candidate in candidates),
            "total_count": len(candidates), "original_sources_unchanged": unchanged,
            "current_arm_q": current.tolist(),
            "joint_state_acquisition_timestamp": request.get("joint_state_acquisition_timestamp"),
            "joint_state_clock_id": request.get("joint_state_clock_id"),
            "joint_state_source": request.get("joint_state_source", "unspecified"),
            "joint_state_freshness_validated": False,
        }
        return batch


def isolated_validate(request, *, repository, python_executable,
                      timeout_s=60.0, config: IKValidationConfig | None = None):
    """Run the real original IK in another environment, with bounded wait.

    Stderr contains solver logs. The result is a JSON batch; a failed child or
    timeout raises rather than manufacturing successful validation flags.
    """
    config = config or IKValidationConfig()
    if not np.isfinite(timeout_s) or not 0 < timeout_s <= 120:
        raise ValueError("timeout_s must be in (0, 120]")
    project = Path(__file__).resolve().parents[2]
    script = project / "tools/copilot/validate_candidates.py"
    with tempfile.TemporaryDirectory(prefix="punktnull_copilot_validation_") as folder:
        input_path, output_path = Path(folder) / "input.json", Path(folder) / "result.json"
        input_path.write_text(json.dumps(request, allow_nan=False))
        command = [str(python_executable), str(script), "--input", str(input_path),
                   "--output", str(output_path), "--repository", str(repository),
                   "--steps", str(config.convergence_steps),
                   "--position-tolerance-m", str(config.max_position_error_m),
                   "--orientation-tolerance-deg", str(np.rad2deg(config.max_orientation_error_rad)),
                   "--joint-limit-tolerance-rad", str(config.joint_limit_tolerance_rad),
                   "--max-candidates", str(config.max_candidates)]
        env = dict(os.environ)
        env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        subprocess.run(command, cwd=project, env=env, check=True, timeout=timeout_s,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return json.loads(output_path.read_text())
