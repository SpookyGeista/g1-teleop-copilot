#!/usr/bin/env python3
"""Read-only copilot installation inventory; never loads a model or robot SDK."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LOCK = Path(__file__).with_name('sources.lock.json')
PATH_KEYS = ('grasp_repository', 'grasp_python', 'perception_python', 'xr_repository',
             'xr_python', 'isaac_repository', 'isaac_python', 'descriptor', 'model_root')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_config(path, project_root=ROOT):
    raw = json.loads(Path(path).read_text())
    if raw.get('schema') != 'punktnull.copilot.local.v1':
        raise ValueError('expected punktnull.copilot.local.v1 config schema')
    paths = {}
    for key in PATH_KEYS:
        value = raw.get('paths', {}).get(key)
        if not isinstance(value, str) or not value or '\x00' in value:
            raise ValueError(f'missing or invalid path: {key}')
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = Path(project_root) / p
        # DO NOT resolve the final Python symlink: doing so bypasses a venv.
        paths[key] = Path(os.path.abspath(p))
    return paths


def command_output(argv, *, timeout=20):
    """Return bounded stdout only on success; never echo arbitrary stderr/env."""
    try:
        completed = subprocess.run([str(a) for a in argv], capture_output=True, text=True,
                                   timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode or len(completed.stdout) > 100_000:
        return None
    return completed.stdout.strip()


def probe_python(python, distributions):
    # Metadata discovery does not import torch/graspgenx/Isaac/DDS, and thus
    # cannot trigger GraspGenX's upstream implicit-download import hook.
    code = '''import importlib.metadata as m,json,sys,urllib.parse
names=json.loads(sys.argv[1]);versions={};origins={}
for name in names:
 try: versions[name]=m.version(name)
 except m.PackageNotFoundError: versions[name]=None
 if name=='graspgenx' and versions[name]:
  data=json.loads(m.distribution(name).read_text('direct_url.json') or '{}');url=urllib.parse.urlsplit(data.get('url',''))
  origins[name]={'editable':bool(data.get('dir_info',{}).get('editable')),'local_project_path':urllib.parse.unquote(url.path) if url.scheme=='file' and url.netloc in ('','localhost') else None}
print(json.dumps({'python':sys.version.split()[0],'prefix':sys.prefix,'base_prefix':sys.base_prefix,'versions':versions,'origins':origins}))
'''
    output = command_output([python, '-I', '-c', code, json.dumps(list(distributions))])
    if output is None:
        return None
    try:
        result = json.loads(output)
        if not isinstance(result.get('versions'), dict):
            return None
        return result
    except (ValueError, AttributeError):
        return None


def audit(config_path, *, project_root=ROOT, lock_path=LOCK, gpu=False):
    paths = load_config(config_path, project_root)
    lock = json.loads(Path(lock_path).read_text())
    checks = []

    def add(name, ok, detail, *, severity='error', **values):
        checks.append(dict(name=name, status='pass' if ok else severity, detail=detail, **values))

    for name, path in paths.items():
        expected_file = name.endswith('_python') or name == 'descriptor'
        ok = path.is_file() if expected_file else path.is_dir()
        add('path.' + name, ok, 'Present' if ok else 'Configure an existing local path', path=str(path))
    repositories = [('graspgenx', paths['grasp_repository']),
                    ('gripper_descriptions', paths['grasp_repository'] / 'ext/gripper_descriptions'),
                    ('checkpoints', paths['grasp_repository'] / 'ext/graspgenx_checkpoints')]
    for name, path in repositories:
        head = command_output(['git', '-C', path, 'rev-parse', 'HEAD'])
        # Content comparison matters: freshly hydrated LFS files can retain
        # pointer-size stat entries until a later Git index refresh.
        dirty = command_output(['git', '-C', path, 'diff', '--no-ext-diff', '--exit-code', 'HEAD', '--'])
        add('revision.' + name, head == lock[name]['revision'],
            'Pinned revision required', expected=lock[name]['revision'], observed=head)
        add('tracked_files.' + name, dirty == '', 'Tracked source files must match the pinned checkout',
            tracked_modifications=None if dirty is None else bool(dirty))
    base = paths['grasp_repository'] / 'ext/graspgenx_checkpoints'
    for relative, expected in lock['checkpoints']['files'].items():
        path = base / relative
        observed = sha256(path) if path.is_file() else None
        add('checkpoint.' + relative, observed == expected['sha256'],
            'Pinned checkpoint/config content hash', expected=expected['sha256'], observed=observed)
    descriptor = paths['descriptor']
    evidence = descriptor.parent / 'validation.json'
    add('descriptor.config_pin', descriptor.is_file() and sha256(descriptor) == lock['dex3']['config_sha256'],
        'Current configured Dex3 descriptor content, not the older upstream G1 hand')
    add('descriptor.validation_pin', evidence.is_file() and sha256(evidence) == lock['dex3']['validation_sha256'],
        'Geometry evidence must match the reviewed descriptor release')
    if evidence.is_file():
        try:
            validation = json.loads(evidence.read_text())
            hashes = validation.get('asset_sha256', {})
            valid = (validation.get('geometry_valid') is True and len(hashes) == 14
                     and validation.get('validation_schema') == 'punktnull.copilot.descriptor_validation.v1')
            failed = []
            for relative, expected in hashes.items():
                path = (descriptor.parent / relative).resolve()
                if not path.is_relative_to(descriptor.parent.resolve()) or not path.is_file() or sha256(path) != expected:
                    failed.append(relative)
            add('descriptor.assets', valid and not failed,
                'URDF, eight meshes, canonical meshes, contact annotations and license',
                asset_count=len(hashes), failed_assets=failed,
                scope=validation.get('geometry_valid_scope', 'unknown'))
        except (ValueError, OSError, TypeError):
            add('descriptor.assets', False, 'Cannot parse descriptor validation evidence')
    constraints = Path(lock_path).with_name('graspgenx_constraints.txt')
    add('environment.constraints_pin', constraints.is_file() and sha256(constraints) == lock['environment']['constraints_sha256'],
        'Version constraints from the tested inference environment')
    expected_versions = dict(lock['environment']['required_versions'], graspgenx='1.0.0')
    info = probe_python(paths['grasp_python'], expected_versions)
    add('environment.grasp_python', info is not None, 'Isolated inference interpreter responds', probe=info)
    if info:
        mismatches = {name: {'expected': expected, 'observed': info['versions'].get(name)}
                      for name, expected in expected_versions.items() if info['versions'].get(name) != expected}
        add('environment.grasp_versions', not mismatches, 'Pinned package versions (metadata only)', mismatches=mismatches)
        add('environment.grasp_isolated', info['prefix'] != info['base_prefix'], 'Inference must use its own virtual environment')
        origin = info.get('origins', {}).get('graspgenx', {})
        add('environment.grasp_import_source', origin.get('editable') is True
            and origin.get('local_project_path') == str(paths['grasp_repository'].resolve()),
            'Installed editable package must refer to the configured pinned checkout')
        venv_cfg = paths['grasp_python'].parent.parent / 'pyvenv.cfg'
        isolated = venv_cfg.is_file() and any(
            line.lower().strip().replace(' ', '') == 'include-system-site-packages=false'
            for line in venv_cfg.read_text().splitlines())
        add('environment.grasp_no_system_packages', isolated,
            'Grasp virtual environment must not inherit unrelated system packages')
        add('environment.grasp_python_version', info['python'].startswith(lock['environment']['python_major_minor'] + '.'),
            'Tested Python major/minor', observed=info['python'])
    for name, distributions in [('perception_python', list(lock['perception_environment']['required_versions'])),
                                ('xr_python', ['numpy', 'pin', 'casadi']),
                                ('isaac_python', ['numpy', 'torch', 'isaaclab'])]:
        info = probe_python(paths[name], distributions)
        add('environment.' + name, info is not None,
            'Metadata inventory only; original environment is never modified', probe=info)
        if info:
            if name == 'perception_python':
                expected = lock['perception_environment']['required_versions']
                mismatch = {n: {'expected': version, 'observed': info['versions'].get(n)}
                            for n, version in expected.items() if info['versions'].get(n) != version}
                add('environment.perception_versions', not mismatch,
                    'Pinned direct dependencies of the dedicated perception process', mismatches=mismatch)
            # Isaac Lab can be installed via its launcher rather than Python
            # distribution metadata; that one missing entry is informational.
            missing = [n for n, v in info['versions'].items() if v is None and n != 'isaaclab']
            add('environment.' + name + '.modules', not missing, 'Required installed distributions', missing=missing)
    for name, relative in [('xr_repository', 'teleop/robot_control/robot_arm_ik.py'),
                           ('xr_repository', 'assets/g1/g1_body29_hand14.urdf'),
                           ('isaac_repository', 'robots/unitree.py'),
                           ('isaac_repository', 'tasks/common_config/camera_configs.py')]:
        path = paths[name] / relative
        add('original.' + relative, path.is_file(), 'Original integration file exists; no source edits performed')
    try:
        sys.path.insert(0, str(project_root))
        from teleop_assist.copilot.asset_paths import camera_source_usd
        proxy = project_root / 'teleop_assist/copilot/assets/g1_wrist_camera_collision_v2.json'
        source = json.loads(proxy.read_text())
        usd = camera_source_usd(source, isaac_repository=paths['isaac_repository'])
        add('original.camera_proxy_usd', True, 'Configured original USD path and immutable reviewed geometry hash',
            path=str(usd), sha256=source['source_usd_sha256'])
    except (ImportError, KeyError, OSError, ValueError, TypeError):
        add('original.camera_proxy_usd', False,
            'Configured Isaac repository must contain the pinned wrist-camera source USD')
    # Read the existing perception manifest; do not import/load the models.
    manifest_path = paths['model_root'] / 'manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text())
        sys.path.insert(0, str(project_root))
        from teleop_assist.model_registry import MODELS
        for name in ['grounding_dino', 'sam2']:
            entry = manifest.get('models', {}).get(name, {})
            pinned = lock['perception'][name]
            valid = entry.get('revision') == MODELS[name]['revision'] == pinned['revision']
            files = entry.get('files', {})
            valid &= files == pinned['files']
            valid &= all(f in files for f in ['config.json', 'model.safetensors', 'preprocessor_config.json'])
            failed = []
            for relative, expected in files.items():
                path = (paths['model_root'] / MODELS[name]['directory'] / relative).resolve()
                if not path.is_relative_to(paths['model_root'].resolve()) or not path.is_file() or sha256(path) != expected['sha256']:
                    failed.append(relative)
            add('perception.' + name, valid and not failed,
                'Pinned revision and all recorded local file hashes', failed_files=failed,
                revision=entry.get('revision'), license=MODELS[name]['license'])
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        add('perception.manifest', False, 'Missing or invalid pinned perception model manifest; use tools/setup_models.py')
    if gpu:
        result = command_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'])
        add('gpu.inventory', result is not None, 'Optional driver inventory; no model or CUDA kernel executed',
            severity='warning', inventory=result)
    else:
        checks.append({'name': 'gpu.inventory', 'status': 'not_checked', 'detail': 'Use --gpu for read-only driver inventory'})
    errors = sum(c['status'] == 'error' for c in checks)
    return {'schema': 'punktnull.copilot.doctor.v1', 'status': 'pass' if not errors else 'issues_found',
            'error_count': errors, 'motion_ready': False, 'hardware_validated': False,
            'scope': 'Installation provenance and local file inventory only; no model/robot operation',
            'checks': checks,
            'licenses': {name: {k: v for k, v in lock[name].items() if k.startswith('license')}
                         for name in ['graspgenx', 'gripper_descriptions', 'checkpoints', 'dex3']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/copilot.local.json')
    parser.add_argument('--json', type=Path, dest='output')
    parser.add_argument('--gpu', action='store_true', help='Read driver metadata; does not load a model')
    args = parser.parse_args()
    try:
        report = audit(args.config, gpu=args.gpu)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {'schema': 'punktnull.copilot.doctor.v1', 'status': 'configuration_error',
                  'error_count': 1, 'motion_ready': False,
                  'checks': [{'name': 'configuration', 'status': 'error', 'detail': type(error).__name__}]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Copilot doctor: {report['status']} ({report['error_count']} errors). Motion readiness is not certified.")
    for check in report['checks']:
        if check['status'] in ('error', 'warning'):
            print(f"  {check['status'].upper()} {check['name']}: {check['detail']}")
    if args.output:
        print(f'JSON report: {args.output}')
    return 0 if not report['error_count'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
