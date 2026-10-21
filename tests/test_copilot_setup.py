"""Installer guard and doctor provenance regressions; no network/GPU/installs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DOCTOR = load('doctor', 'tools/copilot/doctor.py')
SETUP = load('copilot_setup_test', 'tools/copilot/setup.py')


@pytest.fixture
def config(tmp_path):
    data = json.loads((ROOT / 'configs/copilot.local.json').read_text())
    for name in ('xr_repository', 'isaac_repository'):
        data['paths'][name] = 'external_test/' + name
    for name in ('xr_python', 'isaac_python'):
        data['paths'][name] = 'external_test/' + name + '/bin/python'
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(data))
    return path


def test_default_plan_has_no_subprocess_network_or_mutations(config, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('plan must never execute commands')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))
    plan = SETUP.build_plan(config, project_root=tmp_path)
    assert plan['mode'] == 'plan' and plan['network_and_install_performed'] is False
    assert plan['motion_ready'] is False
    assert before == sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))
    checkouts = [s for s in plan['steps'] if s['action'] == 'pinned_checkout']
    assert len(checkouts) == 3 and all(len(s['revision']) == 40 for s in checkouts)
    assert plan['known_grasp_checkpoint_bytes'] > 1_000_000_000


def test_python_symlink_is_not_resolved_out_of_virtual_environment(config, tmp_path):
    path = tmp_path / 'third_party/GraspGenX/.venv/bin/python'
    path.parent.mkdir(parents=True)
    path.symlink_to(sys.executable)
    paths = DOCTOR.load_config(config, tmp_path)
    assert paths['grasp_python'] == path
    assert paths['grasp_python'] != path.resolve()


@pytest.mark.parametrize('name,value', [
    ('grasp_repository', '/tmp/foreign-grasp'),
    ('model_root', '/tmp/foreign-models'),
    ('grasp_python', '/usr/bin/python'),
    ('perception_python', '/usr/bin/python'),
])
def test_install_destination_escape_is_rejected(config, tmp_path, name, value):
    data = json.loads(config.read_text())
    data['paths'][name] = value
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        SETUP.build_plan(config, project_root=tmp_path)


def test_symlinked_environment_escape_is_rejected(config, tmp_path):
    repo = tmp_path / 'third_party/GraspGenX'
    repo.mkdir(parents=True)
    (repo / '.venv').symlink_to('/usr', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        SETUP.build_plan(config, project_root=tmp_path)


def test_existing_different_checkout_is_never_reset(tmp_path, monkeypatch):
    destination = tmp_path / 'repo'
    destination.mkdir()
    (destination / 'user-work.txt').write_text('keep my work')
    monkeypatch.setattr(DOCTOR, 'command_output', lambda args, **kwargs: 'different revision')
    calls = []
    with pytest.raises(ValueError, match='existing checkout differs'):
        SETUP.ensure_checkout({'name': 'test', 'destination': str(destination), 'revision': 'a' * 40},
                              runner=lambda *args, **kwargs: calls.append(args))
    assert not calls and (destination / 'user-work.txt').read_text() == 'keep my work'


def test_new_checkout_uses_exact_revision_without_destructive_reset(tmp_path):
    calls = []
    revision = 'b' * 40
    def runner(args, **kwargs):
        calls.append((args, kwargs))
        if args[1] == 'clone':
            Path(args[-1]).mkdir()
    SETUP.ensure_checkout({'name': 'test', 'destination': str(tmp_path / 'new'),
                           'revision': revision, 'url': 'https://example.invalid/repo.git'},
                          runner=runner)
    assert len(calls) == 3
    assert calls[-1][0][-3:] == ['checkout', '--detach', revision]
    assert all('reset' not in args for args, _ in calls)
    assert calls[0][1]['environment']['GIT_LFS_SKIP_SMUDGE'] == '1'
    assert calls[0][1]['environment']['GIT_TERMINAL_PROMPT'] == '0'


def test_interrupted_owned_checkout_restarts_without_touching_other_files(tmp_path):
    destination = tmp_path / 'new'
    untouched = tmp_path / 'user-work.txt'
    untouched.write_text('keep')
    step = {'name': 'test', 'destination': str(destination), 'revision': 'c' * 40,
            'url': 'https://example.invalid/repo.git'}
    def interrupted(args, **kwargs):
        if args[1] == 'clone':
            stage = Path(args[-1]);stage.mkdir();(stage / 'incomplete').write_text('partial')
        raise RuntimeError('simulated process interruption')
    with pytest.raises(RuntimeError):
        SETUP.ensure_checkout(step, runner=interrupted)
    assert not destination.exists()
    def completed(args, **kwargs):
        if args[1] == 'clone':
            Path(args[-1]).mkdir()
    SETUP.ensure_checkout(step, runner=completed)
    assert destination.is_dir() and not (destination / 'incomplete').exists()
    assert untouched.read_text() == 'keep'


def test_unowned_partial_checkout_is_never_deleted(tmp_path):
    destination = tmp_path / 'new'
    stage = tmp_path / '.new.copilot-dddddddddddd.partial'
    stage.mkdir();(stage / 'user-work').write_text('keep')
    with pytest.raises(ValueError, match='unowned'):
        SETUP.ensure_checkout({'name': 'test', 'destination': str(destination), 'revision': 'd' * 40,
                               'url': 'https://example.invalid/repo.git'}, runner=lambda *a, **kw: None)
    assert (stage / 'user-work').read_text() == 'keep'


def test_bad_local_cache_hash_is_rejected_before_replacing_pointer(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.mkdir();destination.mkdir()
    (source / 'weight').write_bytes(b'bad')
    (destination / 'weight').write_bytes(b'original pointer')
    monkeypatch.setattr(SETUP, 'lfs_pointers', lambda path: [('weight', '0' * 64, 3)])
    with pytest.raises(ValueError, match='content hash'):
        SETUP.reuse_lfs_files(destination, source)
    assert (destination / 'weight').read_bytes() == b'original pointer'


def test_valid_cache_bytes_are_verified_and_hardlinked(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.mkdir();destination.mkdir()
    (source / 'weight').write_bytes(b'known weights')
    (destination / 'weight').write_bytes(b'pointer')
    expected = DOCTOR.sha256(source / 'weight')
    monkeypatch.setattr(SETUP, 'lfs_pointers', lambda path: [('weight', expected, 13)])
    receipt = SETUP.reuse_lfs_files(destination, source)
    assert receipt['verified_files'] == 1
    assert (source / 'weight').stat().st_ino == (destination / 'weight').stat().st_ino


def test_interrupted_owned_venv_can_resume(tmp_path, monkeypatch):
    target = tmp_path / 'venv'
    step = {'name': 'test', 'destination': str(target), 'python': str(target / 'bin/python'),
            'base_python': sys.executable}
    monkeypatch.setattr(DOCTOR, 'probe_python', lambda *a, **kw: None)
    def stop(*args, **kwargs):
        raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError):
        SETUP.ensure_venv(step, runner=stop)
    calls = []
    status = SETUP.ensure_venv(step, runner=lambda args, **kwargs: calls.append(args))
    assert status == 'resumed_owned_partial_venv'
    assert calls == [[sys.executable, '-m', 'venv', target]]


def test_content_verifier_detects_tampered_bytes(tmp_path):
    path = tmp_path / 'weights.bin';path.write_bytes(b'original')
    expected = {'weights.bin': {'sha256': DOCTOR.sha256(path), 'bytes': 8}}
    assert SETUP.verify_expected_files(tmp_path, expected)
    path.write_bytes(b'modified')
    assert not SETUP.verify_expected_files(tmp_path, expected)


def test_optional_vendor_pointer_can_remain_unmaterialized(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.mkdir();destination.mkdir()
    pointer = b'version https://git-lfs.github.com/spec/v1\noid sha256:' + b'a' * 64 + b'\nsize 9999\n'
    (source / 'unused-mesh').write_bytes(pointer)
    (destination / 'unused-mesh').write_bytes(pointer)
    monkeypatch.setattr(SETUP, 'lfs_pointers', lambda path: [('unused-mesh', 'a' * 64, 9999)])
    receipt = SETUP.reuse_lfs_files(destination, source, allow_missing=True)
    assert receipt['unmaterialized_optional_vendor_files'] == 1
    assert (destination / 'unused-mesh').read_bytes() == pointer
    with pytest.raises(ValueError):
        SETUP.reuse_lfs_files(destination, source)


def test_perception_cache_preserves_pinned_bytes_and_writes_subset_manifest(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    (source / 'model').mkdir(parents=True)
    weight = source / 'model/model.safetensors';weight.write_bytes(b'pinned tensor bytes')
    files = {'model.safetensors': {'sha256': DOCTOR.sha256(weight), 'bytes': weight.stat().st_size}}
    entry = {'repo_id': 'example/model', 'revision': 'f' * 40, 'directory': 'model', 'files': files}
    (source / 'manifest.json').write_text(json.dumps({'format': 1, 'models': {'test': entry}}))
    step = {'cache_source': str(source), 'destination': str(target),
            'models': {'test': {'repo_id': entry['repo_id'], 'revision': entry['revision']}},
            'expected_files': {'test': {'directory': 'model', 'files': files}}}
    SETUP.reuse_perception_cache(step)
    assert SETUP.verify_expected_files(target / 'model', files)
    assert json.loads((target / 'manifest.json').read_text())['models']['test'] == entry
    # Alter only the manifest fixture, avoiding any mutation of shared bytes.
    entry['revision'] = 'e' * 40
    (source / 'manifest.json').write_text(json.dumps({'format': 1, 'models': {'test': entry}}))
    with pytest.raises(ValueError, match='pinned model content'):
        SETUP.reuse_perception_cache(step)


def test_metadata_probe_does_not_import_model_or_device_modules(monkeypatch):
    captured = []
    def answer(args, **kwargs):
        captured.append(args)
        return json.dumps({'python': '3.11.15', 'prefix': '/test/venv', 'base_prefix': '/test/base',
                           'versions': {'torch': '2.6.0+cu124'}})
    monkeypatch.setattr(DOCTOR, 'command_output', answer)
    assert DOCTOR.probe_python('/test/venv/bin/python', ['torch'])['versions']['torch'] == '2.6.0+cu124'
    code = captured[0][3]
    assert 'importlib.metadata' in code
    assert all(text not in code for text in ['import torch', 'import graspgenx', 'import isaac', 'import unitree'])
    assert '-I' in captured[0]


def test_subprocess_failure_does_not_expose_stderr_credentials(monkeypatch, capsys):
    class Failure:
        returncode = 9
        stderr = b'password=NEVER_PRINT_THIS_SECRET'
        stdout = b'token=ALSO_NEVER_PRINT'
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: Failure())
    with pytest.raises(RuntimeError) as caught:
        SETUP.run_command(['dummy'])
    assert 'SECRET' not in str(caught.value)
    assert 'NEVER_PRINT' not in capsys.readouterr().out + capsys.readouterr().err
    assert DOCTOR.command_output(['dummy']) is None


def test_source_lock_includes_distinct_code_weights_assets_and_descriptor_evidence():
    lock = json.loads(DOCTOR.LOCK.read_text())
    assert lock['graspgenx']['license'] == 'Apache-2.0'
    assert lock['checkpoints']['license'] == 'NVIDIA Open Model License'
    assert lock['dex3']['config_sha256'] == DOCTOR.sha256(ROOT / 'teleop_assist/copilot/assets/dex3_right/config.json')
    assert lock['dex3']['validation_sha256'] == DOCTOR.sha256(ROOT / 'teleop_assist/copilot/assets/dex3_right/validation.json')
    assert lock['environment']['constraints_sha256'] == DOCTOR.sha256(DOCTOR.LOCK.with_name('graspgenx_constraints.txt'))
    assert all(len(v['sha256']) == 64 for v in lock['checkpoints']['files'].values())
    assert set(lock['perception']) == {'grounding_dino', 'sam2'}


def test_doctor_bad_config_is_not_reported_as_installation_or_motion_success(tmp_path):
    invalid = tmp_path / 'invalid.json'
    invalid.write_text('{"schema":"wrong", "token":"MUST_NOT_APPEAR"}')
    result = subprocess.run([sys.executable, str(ROOT / 'tools/copilot/doctor.py'), '--config', str(invalid)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'configuration_error' in result.stdout
    assert 'MUST_NOT_APPEAR' not in result.stdout + result.stderr
