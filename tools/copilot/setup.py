#!/usr/bin/env python3
"""Print the copilot setup plan by default; install only with explicit --install.

Existing mismatched/modified Git checkouts are never reset. All installation
destinations must be inside this project. Original XR/Isaac environments remain
external prerequisites and are never modified by this script.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import doctor

ROOT = doctor.ROOT


def build_plan(config_path, *, project_root=ROOT, lock_path=doctor.LOCK, reuse_cache_from=None, offline=False, wheel_cache=None):
    paths = doctor.load_config(config_path, project_root)
    lock = json.loads(Path(lock_path).read_text())
    project = Path(project_root).resolve()
    cached = doctor.load_config(reuse_cache_from, project_root) if reuse_cache_from else None
    repository = paths['grasp_repository']
    # Installation must not be redirected into an original robot checkout,
    # arbitrary shared environment or symlink outside the project.
    for name in ('grasp_repository', 'model_root'):
        if not paths[name].resolve().is_relative_to(project):
            raise ValueError(f'installation destination must be inside project: {name}')
    expected_python = repository / '.venv/bin/python'
    if paths['grasp_python'] != expected_python:
        raise ValueError('grasp_python must be grasp_repository/.venv/bin/python')
    if not expected_python.parent.parent.resolve().is_relative_to(project):
        raise ValueError('grasp virtual environment must not be a symlink outside the project')
    perception_venv = paths['perception_python'].parent.parent
    if (paths['perception_python'].name != 'python'
            or paths['perception_python'].parent.name != 'bin'
            or not perception_venv.resolve().is_relative_to(project)):
        raise ValueError('perception_python must be a project-local virtual environment bin/python')
    if perception_venv.resolve() == (repository / '.venv').resolve():
        raise ValueError('perception and grasp models require separate environments')
    steps = []
    for name, target in [('graspgenx', repository),
                         ('gripper_descriptions', repository / 'ext/gripper_descriptions'),
                         ('checkpoints', repository / 'ext/graspgenx_checkpoints')]:
        steps.append({'action': 'pinned_checkout', 'name': name, 'destination': str(target),
                      'url': lock[name]['url'], 'revision': lock[name]['revision']})
        if cached:
            source = cached['grasp_repository']
            if name != 'graspgenx':
                source = source / ('ext/gripper_descriptions' if name == 'gripper_descriptions'
                                   else 'ext/graspgenx_checkpoints')
            if source.resolve() == target.resolve():
                raise ValueError('cache source and new checkout destination must differ')
            steps[-1]['cache_source'] = str(source)
    steps.append({'action': 'venv', 'name': 'grasp_environment', 'destination': str(repository / '.venv'),
                  'python': str(paths['grasp_python']), 'base_python': str(paths['isaac_python'])})
    steps.append({'action': 'packages', 'name': 'grasp_packages', 'python': str(paths['grasp_python']),
                  'torch_backend': lock['environment']['torch_backend'],
                  'requirements_file': str(Path(lock_path).with_name('graspgenx_constraints.txt')),
                  'editable_repository': str(repository),
                  'required_versions': dict(lock['environment']['required_versions'], graspgenx='1.0.0')})
    for name in ('gripper_descriptions', 'checkpoints'):
        steps.append({'action': 'git_lfs', 'name': name + '_data',
                      'destination': str(repository / ('ext/gripper_descriptions' if name == 'gripper_descriptions'
                                                        else 'ext/graspgenx_checkpoints')),
                      'required': name == 'checkpoints',
                      'include': '' if name == 'gripper_descriptions' else ','.join(lock['checkpoints']['files'])})
        if name == 'checkpoints':
            steps[-1]['expected_files'] = lock['checkpoints']['files']
    steps.append({'action': 'venv', 'name': 'perception_environment', 'destination': str(perception_venv),
                  'python': str(paths['perception_python']), 'base_python': str(paths['isaac_python'])})
    steps.append({'action': 'packages', 'name': 'perception_packages', 'python': str(paths['perception_python']),
                  'torch_backend': lock['perception_environment']['torch_backend'],
                  'required_versions': lock['perception_environment']['required_versions']})
    steps.append({'action': 'perception_weights', 'name': 'perception_weights',
                  'python': str(paths['perception_python']), 'destination': str(paths['model_root']),
                  'script': str(project / 'tools/setup_models.py'),
                  'models': {name: {k: entry[k] for k in ('repo_id', 'revision')}
                             for name, entry in lock['perception'].items()},
                  'expected_files': {name: {'directory': entry['directory'], 'files': entry['files']}
                                     for name, entry in lock['perception'].items()}})
    if cached:
        steps[-1]['cache_source'] = str(cached['model_root'])
    return {'schema': 'punktnull.copilot.setup_plan.v1', 'mode': 'plan',
            'offline': bool(offline), 'wheel_cache': str(Path(wheel_cache).resolve()) if wheel_cache else None,
            'network_and_install_performed': False, 'motion_ready': False,
            'steps': steps,
            'known_grasp_checkpoint_bytes': sum(x['bytes'] for x in lock['checkpoints']['files'].values()),
            'download_note': 'Grasp checkpoints alone total about 1.69 GB; CUDA wheels, perception weights and gripper assets add further downloads.',
            'external_prerequisites': ['git with Git LFS', 'uv', 'configured Python 3.11 interpreter',
                                       'existing original XR and Isaac installations', 'suitable NVIDIA driver for CUDA 12.8/12.4 wheels'],
            'licenses': {name: {k: v for k, v in lock[name].items() if k.startswith('license')}
                         for name in ['graspgenx', 'gripper_descriptions', 'checkpoints', 'dex3']},
            'perception_licenses': {name: {'license': entry['license'],
                'model_card': f"https://huggingface.co/{entry['repo_id']}/tree/{entry['revision']}"}
                                   for name, entry in lock['perception'].items()},
            'reproducibility_limit': 'Pins code, model content and tested package versions; does not reproduce driver/OS/Isaac or wheel-content hashes.'}


def run_command(argv, *, timeout=1800, environment=None):
    """Do not print subprocess output, which can contain repository credentials."""
    try:
        completed = subprocess.run([str(a) for a in argv], capture_output=True,
                                   timeout=timeout, env=environment, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('setup command exceeded its time bound') from error
    if completed.returncode:
        raise RuntimeError(f'setup command failed with exit code {completed.returncode}; raw subprocess output suppressed')


def lfs_pointers(repository):
    tracked = doctor.command_output(['git', '-C', repository, 'ls-files', '-z'])
    if tracked is None:
        raise ValueError('cannot inventory tracked files for pinned LFS checkout')
    pointers = []
    for relative in tracked.split('\x00'):
        if not relative:
            continue
        path = Path(repository) / relative
        if not path.is_file():
            raise ValueError('pinned checkout is missing a tracked file')
        with path.open('rb') as stream:
            header = stream.read(300)
        if header.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            digest = re.search(rb'\noid sha256:([a-f0-9]{64})\n', header)
            size = re.search(rb'\nsize ([0-9]+)', header)
            if not digest or not size:
                raise ValueError('malformed Git LFS pointer')
            pointers.append((relative, digest.group(1).decode(), int(size.group(1))))
    return pointers


def reuse_lfs_files(destination, source, *, allow_missing=False):
    """Materialize tracked pointers only from SHA256-verified local bytes."""
    count, reused_bytes, optional_missing = 0, 0, 0
    for relative, expected, size in lfs_pointers(destination):
        original = Path(source) / relative
        target = Path(destination) / relative
        if allow_missing and original.is_file() and original.stat().st_size < 300:
            original_pointer = original.read_bytes()
            if (original_pointer.startswith(b'version https://git-lfs.github.com/spec/v1\n')
                    and ('oid sha256:' + expected).encode() in original_pointer):
                optional_missing += 1
                continue
        if (not original.resolve().is_relative_to(Path(source).resolve())
                or not original.is_file() or original.stat().st_size != size
                or doctor.sha256(original) != expected):
            raise ValueError('local cache LFS file does not match pinned content hash')
        target.unlink()  # only the just-created small LFS pointer, never source
        try:
            os.link(original, target)
        except OSError:
            shutil.copy2(original, target)
        count += 1
        reused_bytes += size
    return {'verified_files': count, 'reused_bytes': reused_bytes,
            'unmaterialized_optional_vendor_files': optional_missing}


def ensure_checkout(step, *, runner=run_command):
    destination = Path(step['destination'])
    if destination.exists():
        head = doctor.command_output(['git', '-C', destination, 'rev-parse', 'HEAD'])
        dirty = doctor.command_output(['git', '-C', destination, 'diff', '--no-ext-diff', '--exit-code', 'HEAD', '--'])
        if head != step['revision'] or dirty != '':
            raise ValueError(f"{step['name']}: existing checkout differs; choose a new empty project-local destination")
        return 'already_pinned'
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A killed clone never becomes the final checkout. Only our own marked
    # partial staging directory can be discarded/restarted on the next run.
    staging = destination.with_name('.' + destination.name + '.copilot-' + step['revision'][:12] + '.partial')
    marker = staging.with_name(staging.name + '.owner.json')
    ownership = {'schema': 'punktnull.setup.partial.v1', 'destination': str(destination),
                 'revision': step['revision'], 'url': step['url']}
    if staging.exists() or staging.is_symlink():
        if staging.is_symlink() or not marker.is_file() or json.loads(marker.read_text()) != ownership:
            raise ValueError('unowned partial checkout exists; refusing to overwrite')
        shutil.rmtree(staging)
    marker.write_text(json.dumps(ownership, indent=2) + '\n')
    environment = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')
    source = step.get('cache_source')
    if source:
        head = doctor.command_output(['git', '-C', source, 'rev-parse', 'HEAD'])
        dirty = doctor.command_output(['git', '-C', source, 'diff', '--no-ext-diff', '--exit-code', 'HEAD', '--'])
        if head != step['revision'] or dirty != '':
            raise ValueError('local Git cache is not the clean pinned revision')
        runner(['git', 'clone', '--local', '--no-checkout', source, staging], environment=environment)
    else:
        runner(['git', 'clone', '--no-checkout', '--filter=blob:none', step['url'], staging], environment=environment)
        runner(['git', '-C', staging, 'fetch', '--depth', '1', 'origin', step['revision']], environment=environment)
    runner(['git', '-C', staging, 'checkout', '--detach', step['revision']], environment=environment)
    if source:
        reused = reuse_lfs_files(staging, source, allow_missing=step['name'] == 'gripper_descriptions')
        (staging / '.copilot-cache-receipt.json').write_text(json.dumps(reused, indent=2) + '\n')
        runner(['git', '-C', staging, 'remote', 'set-url', 'origin', step['url']], environment=environment)
    staging.rename(destination)
    marker.unlink()
    return 'installed_pinned_checkout_from_verified_local_cache' if source else 'installed_pinned_checkout'


def ensure_venv(step, *, runner=run_command):
    target = Path(step['destination'])
    marker = target / '.copilot-managed-venv.json'
    ownership = {'schema': 'punktnull.setup.venv.v1', 'base_python': step['base_python']}
    if target.exists():
        info = doctor.probe_python(step['python'], [])
        if (target / 'pyvenv.cfg').is_file() and info and Path(info['prefix']).resolve() == target.resolve():
            return 'existing_venv_preserved'
        if not marker.is_file() or json.loads(marker.read_text()) != ownership:
            raise ValueError(f"{step['name']}: existing directory is not the expected virtual environment")
        status = 'resumed_owned_partial_venv'
    else:
        target.mkdir(parents=True)
        marker.write_text(json.dumps(ownership, indent=2) + '\n')
        status = 'created_isolated_venv'
    runner([step['base_python'], '-m', 'venv', target])
    return status


def verify_expected_files(base, files):
    for relative, expected in files.items():
        path = (Path(base) / relative).resolve()
        if not path.is_relative_to(Path(base).resolve()) or not path.is_file():
            return False
        if doctor.sha256(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            return False
    return True


def reuse_perception_cache(step):
    source, destination = Path(step['cache_source']), Path(step['destination'])
    manifest = json.loads((source / 'manifest.json').read_text())
    selected = {}
    # Validate the complete source before creating target artifacts.
    for name, item in step['expected_files'].items():
        entry = manifest.get('models', {}).get(name, {})
        if (entry.get('revision') != step['models'][name]['revision']
                or entry.get('repo_id') != step['models'][name]['repo_id']
                or entry.get('files') != item['files']
                or not verify_expected_files(source / item['directory'], item['files'])):
            raise ValueError('local perception cache does not match pinned model content')
        selected[name] = entry
    destination.mkdir(parents=True, exist_ok=True)
    for item in step['expected_files'].values():
        for relative, expected in item['files'].items():
            original = source / item['directory'] / relative
            target = destination / item['directory'] / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('perception cache target escapes configured model directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if doctor.sha256(target) != expected['sha256']:
                    raise ValueError('existing perception target file differs; refusing to overwrite')
                continue
            try:
                os.link(original, target)
            except OSError:
                shutil.copy2(original, target)
    manifest_path = destination / 'manifest.json'
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({'format': 1, 'models': selected,
                                            'source': 'verified_local_cache'}, indent=2) + '\n')


def install_plan(plan, *, runner=run_command):
    if not shutil.which('git') or not shutil.which('uv') or doctor.command_output(['git', 'lfs', 'version']) is None:
        raise ValueError('install requires existing git, Git LFS and uv commands')
    # Verify base interpreter before creating or downloading anything.
    for step in plan['steps']:
        if step['action'] == 'venv':
            info = doctor.probe_python(step['base_python'], [])
            if not info or not info['python'].startswith('3.11.'):
                raise ValueError('configured base interpreter must be Python 3.11')
    outcomes = []
    for step in plan['steps']:
        print(f"Setup: {step['name']}", flush=True)
        action, status = step['action'], 'complete'
        if action == 'pinned_checkout':
            status = ensure_checkout(step, runner=runner)
        elif action == 'venv':
            status = ensure_venv(step, runner=runner)
        elif action == 'packages':
            expected = step['required_versions']
            info = doctor.probe_python(step['python'], expected)
            source_matches = True
            if 'editable_repository' in step:
                origin = (info or {}).get('origins', {}).get('graspgenx', {})
                source_matches = (origin.get('editable') is True
                                  and origin.get('local_project_path') == str(Path(step['editable_repository']).resolve()))
            if info and source_matches and all(info['versions'].get(k) == v for k, v in expected.items()):
                status = 'already_matching_versions'
            else:
                args = ['uv', 'pip', 'install', '--python', step['python'], '--torch-backend',
                        step['torch_backend'], '--prerelease', 'allow', '--link-mode', 'hardlink']
                if plan.get('offline'):
                    args += ['--offline']
                if plan.get('wheel_cache'):
                    args += ['--find-links', plan['wheel_cache']]
                if 'requirements_file' in step:
                    args += ['--requirement', step['requirements_file'], '-e', step['editable_repository']]
                else:
                    args += [f'{name}=={version}' for name, version in expected.items()]
                runner(args)
        elif action == 'git_lfs':
            pointers = lfs_pointers(step['destination'])
            if pointers and not step.get('required', True):
                # The production descriptor is our hash-verified Dex3 asset
                # package. Other manufacturers' meshes are not runtime input.
                status = 'optional_vendor_meshes_not_downloaded'
            elif pointers:
                if plan.get('offline'):
                    raise ValueError('LFS cache is incomplete in offline mode')
                args = ['git', '-C', step['destination'], 'lfs', 'pull']
                if step['include']:
                    args += ['--include', step['include']]
                runner(args, environment=dict(os.environ, GIT_TERMINAL_PROMPT='0'))
            else:
                status = 'hydrated_local_files_preserved'
            if 'expected_files' in step and not verify_expected_files(step['destination'], step['expected_files']):
                raise ValueError('checkpoint content hash does not match release lock')
        elif action == 'perception_weights':
            complete = all(verify_expected_files(Path(step['destination']) / item['directory'], item['files'])
                           for item in step['expected_files'].values())
            if not complete and step.get('cache_source'):
                reuse_perception_cache(step)
                complete = True
            if complete:
                status = 'verified_local_weights_preserved'
            elif plan.get('offline'):
                raise ValueError('perception weights are absent or hash-mismatched in offline mode')
            else:
                runner([step['python'], step['script'], '--model-root', step['destination'],
                        '--download', '--verify', '--only', *step['models']])
                if not all(verify_expected_files(Path(step['destination']) / item['directory'], item['files'])
                           for item in step['expected_files'].values()):
                    raise ValueError('downloaded perception content hash does not match release lock')
        else:
            raise ValueError('unknown setup action')
        outcomes.append({'name': step['name'], 'status': status})
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/copilot.local.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='Default: inspect the plan, no installation/network')
    mode.add_argument('--install', action='store_true', help='Explicitly download and install the pinned isolated model stacks')
    parser.add_argument('--json', type=Path, dest='output')
    parser.add_argument('--reuse-cache-from', type=Path, help='Reuse verified local source/model bytes from another copilot config')
    parser.add_argument('--offline', action='store_true', help='Require all packages/models in local caches; no network fallback')
    parser.add_argument('--wheel-cache', type=Path, help='Optional local wheel directory for uv --find-links')
    args = parser.parse_args()
    try:
        plan = build_plan(args.config, reuse_cache_from=args.reuse_cache_from,
                          offline=args.offline, wheel_cache=args.wheel_cache)
        if args.offline and not args.reuse_cache_from:
            raise ValueError('offline setup requires --reuse-cache-from for pinned source checkouts')
        if args.install:
            # Validate the locked requirements before allowing downloads.
            lock = json.loads(doctor.LOCK.read_text())
            constraints = doctor.LOCK.with_name('graspgenx_constraints.txt')
            if doctor.sha256(constraints) != lock['environment']['constraints_sha256']:
                raise ValueError('GraspGenX constraints hash differs from release lock')
            plan['steps_completed'] = install_plan(plan)
            plan.update(mode='install', network_and_install_performed=True)
            plan['doctor'] = doctor.audit(args.config)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
        print(f"Copilot setup: {plan['mode']}. Motion readiness is not certified.")
        for step in plan['steps']:
            print(f"  {step['action']}: {step['name']}")
        print(plan['download_note'])
        if not args.install:
            print('Nothing installed. Use --install only when you intend these downloads and local environment changes.')
        if args.output:
            print(f'JSON report: {args.output}')
        return 2 if args.install and plan['doctor']['error_count'] else 0
    except (OSError, ValueError, RuntimeError) as error:
        # Errors are our controlled messages; OS errors may reveal arbitrary
        # configured path data, so print only their class.
        detail = type(error).__name__ if isinstance(error, OSError) else str(error)
        print(f'Setup stopped: {detail}', file=sys.stderr)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'status': 'setup_failed', 'detail': detail,
                                              'motion_ready': False}, indent=2) + '\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
