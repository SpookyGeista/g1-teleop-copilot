#!/usr/bin/env python3
"""Offline preflight for the assisted teleoperation stack; never launches Isaac.

Checks declared files, installed environments, pinned model hashes, public
defaults and an RGB-D schema sample. It does not load model weights, open robot
connections, start a server, or establish headset/hardware readiness.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def child_json(interpreter, source, *arguments):
    completed = subprocess.run([str(interpreter), "-B", "-c", source, *map(str, arguments)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-1800:] or f"Interpreter exited {completed.returncode}")
    return json.loads(completed.stdout)


def literal_constants(path, names):
    """Read version pins without importing model code or a CUDA library."""
    result = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    missing = set(names)-set(result)
    if missing:
        raise ValueError('Missing declared version pins: '+', '.join(sorted(missing)))
    return result


def verify_foundationpose_files(source, manifest_path, revision, weight_names, native_sources, verify_hashes=True):
    """Verify every declared pose-model asset, including same-size corruption."""
    source = Path(source).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('source') != 'https://github.com/NVlabs/FoundationPose' or manifest.get('revision') != revision:
        raise ValueError('FoundationPose source/revision manifest mismatch')
    if manifest.get('native_sources') != native_sources:
        raise ValueError('FoundationPose native source pins do not match setup script')
    expected_files = {f'weights/{name}/{filename}' for name in weight_names
                      for filename in ('config.yml', 'model_best.pth')}
    files = manifest.get('weights', {})
    if not expected_files.issubset(files):
        raise ValueError('FoundationPose manifest omits required weights or configuration')
    total = 0
    for relative, expected in files.items():
        path = (source/relative).resolve()
        if not path.is_relative_to(source/'weights'):
            raise ValueError('FoundationPose manifest path escapes weights directory')
        size = expected.get('bytes')
        sha256 = expected.get('sha256')
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError('Invalid FoundationPose file size in manifest')
        if not isinstance(sha256, str) or len(sha256) != 64 or any(c not in '0123456789abcdef' for c in sha256):
            raise ValueError('Invalid FoundationPose SHA256 in manifest')
        if path.stat().st_size != size:
            raise ValueError('FoundationPose size mismatch: '+relative)
        if verify_hashes and digest(path) != sha256:
            raise ValueError('FoundationPose SHA256 mismatch: '+relative)
        total += size
    return {'source_root': str(source), 'manifest': str(Path(manifest_path).resolve()),
            'revision': revision, 'native_sources': native_sources,
            'files': len(files), 'bytes': total, 'sha256_checked': bool(verify_hashes),
            'manifest_claims_inference_verified': bool(manifest.get('inference_verified', False)),
            'inference_retested_by_preflight': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-python", type=Path, default=Path(os.environ.get(
        "PUNKTNULL_ISAAC_PYTHON", "/home/loya2/miniconda3/envs/unitree_sim_env/bin/python")))
    parser.add_argument("--model-python", type=Path, default=ROOT / ".venv-models/bin/python")
    parser.add_argument("--pose-python", type=Path, default=ROOT / ".venv-pose/bin/python")
    parser.add_argument("--pose-root", type=Path, default=ROOT / "third_party/FoundationPose")
    parser.add_argument("--pose-manifest", type=Path, default=ROOT / "models/foundationpose_manifest.json")
    parser.add_argument("--pose-backend", choices=("foundationpose", "surface"), default="foundationpose")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/perception.interaction.json")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--rgbd", type=Path, help="Optionally validate an actual recorded RGB-D .npz")
    hashes = parser.add_mutually_exclusive_group()
    hashes.add_argument("--verify-models", action="store_true",
                        help="Explicitly verify SHA256 of all detector, segmentation and pose weights (also the default)")
    hashes.add_argument("--skip-hashes", action="store_true", help="Fast size-only model check; reported as a warning")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path")
    args = parser.parse_args(argv)
    checks = []
    warnings = ["Headset streaming, TLS trust, hand-eye calibration, live control latency and physical task success still require their own tests."]

    def check(name, operation):
        try:
            detail = operation()
            checks.append({"name": name, "passed": True, "detail": detail})
            return detail
        except Exception as exc:
            checks.append({"name": name, "passed": False, "error": str(exc)})
            return None

    def source_files():
        required = ["run_assist.py", "tools/assist_sim.py", "teleop_assist/core.py", "teleop_assist/actions.py",
                    "teleop_assist/predictive.py", "teleop_assist/task_profiles.py", "teleop_assist/model_bridge.py",
                    "teleop_assist/model_worker.py", "teleop_assist/perception_runtime.py", "teleop_assist/visual_tracker.py",
                    "teleop_assist/sensor_fusion.py", "teleop_assist/learned_perception.py", "teleop_assist/stack_evaluation.py",
                    "teleop_assist/foundation_pose.py", "teleop_assist/pose_worker.py", "teleop_assist/pose_bridge.py",
                    "tools/setup_foundation_pose.py", "requirements-pose.txt",
                    "teleop_assist/static/index.html", "teleop_assist/static/app.mjs", "teleop_assist/static/vendor/three.module.min.js"]
        missing = [name for name in required if not (ROOT / name).is_file()]
        if missing:
            raise FileNotFoundError("Missing stack files: " + ", ".join(missing))
        for name in required:
            if name.endswith(".py"):
                ast.parse((ROOT / name).read_text(), filename=name)
        return {"required_files": len(required), "python_syntax": "parsed_without_import_or_Isaac_start"}

    check("source_files_and_syntax", source_files)
    registry = check("model_registry", lambda: runpy.run_path(str(ROOT / "teleop_assist/model_registry.py"))["MODELS"])

    def config_check():
        config = json.loads(args.config.read_text())
        if not isinstance(config, dict):
            raise ValueError("Model config must be a JSON object")
        tree = ast.parse((ROOT / "teleop_assist/learned_perception.py").read_text())
        dataclass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LearnedPerceptionConfig")
        fields = {node.target.id for node in dataclass.body if isinstance(node, ast.AnnAssign)}
        unknown = sorted(set(config) - fields - {"timeout", "startup_timeout"})
        if unknown:
            raise ValueError("Unknown model configuration keys: " + ", ".join(unknown))
        for field in ("box_threshold", "text_threshold", "min_mask_iou", "max_mask_fraction", "min_valid_depth_fraction"):
            if field in config and (not isinstance(config[field], (int, float)) or not math.isfinite(config[field]) or not 0 <= config[field] <= 1):
                raise ValueError(f"{field} must be a finite value between zero and one")
        for field in ("max_mask_fraction", "min_valid_depth_fraction", "min_depth_m", "max_depth_m",
                      "max_object_extent_m", "association_distance_m", "track_retention_s", "max_surface_residual_m",
                      "ambiguity_margin_m", "timeout", "startup_timeout"):
            if field in config and (isinstance(config[field], bool) or not isinstance(config[field], (int, float)) or
                                    not math.isfinite(config[field]) or config[field] <= 0):
                raise ValueError(f"{field} must be finite and positive")
        for field in ("max_targets", "detector_short_edge", "detector_long_edge", "min_mask_pixels"):
            if field in config and (isinstance(config[field], bool) or not isinstance(config[field], int) or config[field] <= 0):
                raise ValueError(f"{field} must be a positive integer")
        if config.get("min_mask_pixels", 24) < 12:
            raise ValueError("min_mask_pixels must be at least 12")
        if not config.get("min_depth_m", .08) < config.get("max_depth_m", 3.):
            raise ValueError("min_depth_m must be less than max_depth_m")
        prompts = config.get("prompts", ["handle"])
        if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError("prompts must contain nonempty text prompts")
        name = config.get("detector_model", "grounding_dino")
        if registry is None or name not in registry or name == "sam2":
            raise ValueError("The configured detector is not registered")
        return {"path": str(args.config.resolve()), "config": config, "detector": name,
                "allowed_fields_checked": len(fields)}

    configuration = check("model_configuration", config_check)

    def model_files():
        if registry is None or configuration is None:
            raise RuntimeError("A valid registry and model configuration are required")
        root = Path(args.model_root or configuration["config"].get("model_root", ROOT / "models")).expanduser().resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        verified = {}
        for name in (configuration["detector"], "sam2"):
            known = registry[name]
            actual = manifest.get("models", {}).get(name)
            if actual is None or any(actual.get(key) != known[key] for key in ("repo_id", "revision", "directory")):
                raise ValueError(f"Pinned model metadata mismatch or absence: {name}")
            required = {"model.safetensors", "config.json", "preprocessor_config.json"}
            if not required.issubset(actual.get("files", {})):
                raise ValueError(f"Manifest missing required model assets: {name}")
            total = 0
            directory = (root / actual["directory"]).resolve()
            for relative, expected in actual["files"].items():
                path = (directory / relative).resolve()
                if not path.is_relative_to(directory):
                    raise ValueError("Model manifest path escapes its directory")
                if path.stat().st_size != expected["bytes"]:
                    raise ValueError(f"Model size mismatch: {name}/{relative}")
                if not args.skip_hashes and digest(path) != expected["sha256"]:
                    raise ValueError(f"Model SHA256 mismatch: {name}/{relative}")
                total += path.stat().st_size
            verified[name] = {"revision": actual["revision"], "files": len(actual["files"]), "bytes": total}
        return {"root": str(root), "sha256_checked": not args.skip_hashes, "models": verified}

    if args.skip_hashes:
        warnings.append("Model SHA256 verification was explicitly skipped; only file sizes were checked.")
    check("pinned_model_files", model_files)

    package_code = """
import importlib.util, importlib.metadata, json, sys
names=json.loads(sys.argv[1]); modules=json.loads(sys.argv[2]); versions={};missing=[]
for name in names:
 try: versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError: missing.append(name)
module_status={name:importlib.util.find_spec(name) is not None for name in modules}
print(json.dumps({'python':sys.version,'executable':sys.executable,'versions':versions,'missing_distributions':missing,'modules':module_status}))
"""

    def environment(interpreter, distributions, modules):
        if not interpreter.is_file():
            raise FileNotFoundError(f"Python interpreter missing: {interpreter}")
        value = child_json(interpreter, package_code, json.dumps(distributions), json.dumps(modules))
        if value["missing_distributions"] or not all(value["modules"].values()):
            raise RuntimeError("Missing runtime dependencies: " + json.dumps(value))
        return value

    check("isaac_environment_discovery", lambda: environment(args.isaac_python, ["numpy", "scipy"], ["isaaclab", "cv2"]))
    model_env = check("model_environment_discovery", lambda: environment(args.model_python,
        ["torch", "transformers", "huggingface-hub", "safetensors", "numpy", "scipy", "Pillow"], ["cv2"]))
    if model_env:
        for name, expected in (("transformers", "4.57.6"), ("huggingface-hub", "0.36.2")):
            if model_env["versions"][name] != expected:
                checks.append({"name": "pinned_runtime_" + name, "passed": False,
                               "error": f"Expected {expected}, found {model_env['versions'][name]}"})

    if args.pose_backend == 'foundationpose':
        def pose_files():
            pins = literal_constants(ROOT/'teleop_assist/foundation_pose.py',
                                     {'FOUNDATIONPOSE_REVISION', 'FOUNDATIONPOSE_WEIGHTS'})
            native = literal_constants(ROOT/'tools/setup_foundation_pose.py',
                                       {'PYTORCH3D_REV', 'NVDIFFRAST_REV'})
            return verify_foundationpose_files(args.pose_root, args.pose_manifest,
                pins['FOUNDATIONPOSE_REVISION'], pins['FOUNDATIONPOSE_WEIGHTS'],
                {'pytorch3d': native['PYTORCH3D_REV'], 'nvdiffrast': native['NVDIFFRAST_REV']},
                verify_hashes=not args.skip_hashes)

        pose_assets = check('pinned_foundationpose_files', pose_files)

        def pose_environment():
            detail = environment(args.pose_python,
                ['torch', 'numpy', 'scipy', 'pytorch3d', 'nvdiffrast', 'open3d', 'kornia',
                 'omegaconf', 'warp-lang', 'transformations', 'pybind11', 'ninja'], ['cv2', 'trimesh'])
            ready = child_json(args.pose_python, """
import json,sys
from teleop_assist.foundation_pose import readiness
print(json.dumps(readiness(sys.argv[1])))
""", args.pose_root.resolve())
            if not ready['ready_for_import']:
                raise RuntimeError('FoundationPose import-light discovery failed: '+json.dumps(ready))
            detail['import_light_readiness'] = ready
            native_installs = child_json(args.pose_python, """
import importlib.metadata,json
print(json.dumps({name:json.loads(importlib.metadata.distribution(name).read_text('direct_url.json') or '{}')
                  for name in ('pytorch3d','nvdiffrast')}))
""")
            if pose_assets is None or any(native_installs[name].get('vcs_info', {}).get('commit_id') != pin
                                         for name, pin in pose_assets['native_sources'].items()):
                raise RuntimeError('Installed native pose packages do not match pinned Git provenance')
            detail['native_source_provenance'] = native_installs
            # --help exits in argparse before importing estimator/native GPU code.
            help_result = subprocess.run([str(args.pose_python), '-B', str(ROOT/'teleop_assist/pose_worker.py'), '--help'],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
            if help_result.returncode or '--directory' not in help_result.stdout:
                raise RuntimeError('FoundationPose worker CLI is unavailable: '+help_result.stderr[-1200:])
            detail['worker_help_passed'] = True
            detail['cuda_initialized'] = False
            return detail

        check('foundationpose_environment_discovery', pose_environment)

        def pose_setup():
            if pose_assets is None:
                raise RuntimeError('Valid FoundationPose manifest required before checking checkout')
            revision = subprocess.run(['git', '-C', str(args.pose_root), 'rev-parse', 'HEAD'],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
            if revision.returncode or revision.stdout.strip() != pose_assets['revision']:
                raise RuntimeError('FoundationPose checkout does not match the pinned source revision')
            required = ['estimater.py', 'Utils.py', 'mycpp/CMakeLists.txt']
            missing = [name for name in required if not (args.pose_root/name).is_file()]
            if missing:
                raise FileNotFoundError('FoundationPose source files missing: '+', '.join(missing))
            binaries = list((args.pose_root/'mycpp/build').glob('mycpp*.so'))
            if not binaries or any(path.stat().st_size == 0 for path in binaries):
                raise RuntimeError('FoundationPose native mycpp build is missing or empty')
            if not (ROOT/'.cuda-toolkit/bin/nvcc').is_file():
                raise FileNotFoundError('Local CUDA toolchain needed for native renderer setup is missing')
            catalog_path = ROOT/'assets/cad/handle_pose/catalog.json'
            catalog = json.loads(catalog_path.read_text())
            handle = next(obj for obj in catalog['objects'] if obj['id'] == 'PullHandleGrasp')
            mesh = ROOT/'assets/cad/handle_pose/PullHandleGrasp.obj'
            if digest(mesh) != handle['cad_sha256']:
                raise ValueError('Handle CAD mesh no longer matches catalog SHA256')
            if not handle.get('grasp_templates'):
                raise ValueError('Known CAD handle has no declared grasp template')
            return {'source_revision': revision.stdout.strip(), 'native_extensions': [str(p) for p in binaries],
                    'local_cuda_toolchain': str(ROOT/'.cuda-toolkit'), 'cad_asset': str(mesh),
                    'cad_sha256_checked': True, 'native_code_executed': False}

        check('foundationpose_setup_and_cad', pose_setup)
    else:
        warnings.append('FoundationPose checks deliberately disabled with --pose-backend surface; this is not the default live configuration.')

    def robot_assets():
        repo = Path(os.environ.get("PUNKTNULL_G1_REPO", "/home/loya2/WholeBodyTele/GR00T-WholeBodyControl")).expanduser()
        files = [repo / "gear_sonic/envs/manager_env/robots/g1.py",
                 repo / "decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf"]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required retained G1 assets missing: " + ", ".join(missing))
        return {"repo": str(repo.resolve()), "files_checked": len(files)}

    check("robot_assets", robot_assets)

    def public_parameters():
        tree = ast.parse((ROOT / "tools/assist_sim.py").read_text())
        params = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            flags = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            for flag in flags:
                values = {}
                for keyword in node.keywords:
                    try: values[keyword.arg] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError): pass
                params[flag] = values
        required = {"--perception", "--require-action", "--wrist-cameras", "--lidar", "--model-config", "--model-python",
                    "--eye-enabled", "--assist-enabled", "--stack-acceptance", "--frame-timeout", "--scene", "--pose-backend", "--pose-python"}
        if not required.issubset(params):
            raise ValueError("Missing public parameters: " + ", ".join(sorted(required - set(params))))
        expected = {"--perception": "learned", "--require-action": True, "--wrist-cameras": True, "--lidar": True,
                    "--scene": "interaction", "--pose-backend": "foundationpose"}
        for flag, default in expected.items():
            if params[flag].get("default") != default:
                raise ValueError(f"Unexpected production default for {flag}")
        for flag in ("--eye-enabled", "--assist-enabled"):
            if params[flag].get("action") != "store_true" or params[flag].get("default", False):
                raise ValueError(f"{flag} must be an explicit opt-in")
        return {flag: params[flag] for flag in sorted(required)}

    check("public_parameters_and_defaults", public_parameters)

    rgbd_code = """
import json,sys,numpy as np
from teleop_assist.perception import _calibration
path=sys.argv[1]
if path:
 with np.load(path,allow_pickle=False) as archive: data={k:archive[k] for k in archive.files}
else:
 data={'rgb':np.zeros((12,16,3),np.uint8),'depth':np.full((12,16),.6,np.float32),'intrinsics':np.array([[20.,0,8],[0,20.,6],[0,0,1]]),'T_world_camera':np.eye(4),'frame_id':np.asarray(1),'timestamp':np.asarray(1.)}
required={'rgb','depth','intrinsics','T_world_camera','frame_id','timestamp'}
if not required.issubset(data):raise ValueError('Missing RGB-D fields: '+str(required-set(data)))
rgb,depth=data['rgb'],data['depth']
if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or depth.shape!=rgb.shape[:2]:raise ValueError('Invalid RGB/depth dimensions or dtype')
_calibration(data['intrinsics'],data['T_world_camera'])
if not np.isfinite(float(data['timestamp'])) or int(data['frame_id'])<0:raise ValueError('Invalid frame identity/time')
valid=np.isfinite(depth)&(depth>0)
if not valid.any():raise ValueError('RGB-D sample contains no measured positive depth')
print(json.dumps({'source':'recorded_sample' if path else 'synthetic_schema_only','path':path or None,'rgb_shape':list(rgb.shape),'depth_valid_fraction':float(valid.mean()),'depth_convention':'optical +Z required by production API; source calibration must establish this'}))
"""
    check("rgbd_schema", lambda: child_json(args.isaac_python, rgbd_code, args.rgbd.resolve() if args.rgbd else ""))
    if not args.rgbd:
        warnings.append("RGB-D validation used a synthetic schema sample. Use --rgbd with a captured .npz to check an actual archive.")
    warnings.append("Package discovery does not load model checkpoints or establish CUDA/GPU memory capacity; run the model probe and stack acceptance separately.")
    blocking = [row["name"] for row in checks if not row["passed"]]
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "project_root": str(ROOT),
              "ready_for_simulation_attempt": not blocking, "headset_verified": False, "hardware_verified": False,
              "isaac_started": False, "models_loaded": False, "pose_backend": args.pose_backend,
              "sha256_verification_requested": not args.skip_hashes,
              "blocking": blocking, "warnings": warnings, "checks": checks}
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0 if not blocking else 2


if __name__ == "__main__":
    raise SystemExit(main())
