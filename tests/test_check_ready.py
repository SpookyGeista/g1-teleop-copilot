import hashlib
import json

import pytest

from tools.check_ready import literal_constants, main, verify_foundationpose_files


@pytest.fixture
def pose_manifest(tmp_path):
    source = tmp_path/'FoundationPose'
    names = ('refiner', 'scorer')
    manifest = {'source': 'https://github.com/NVlabs/FoundationPose', 'revision': 'pinned',
                'native_sources': {'pytorch3d': 'p3d_pin', 'nvdiffrast': 'nvd_pin'}, 'weights': {}}
    for name in names:
        for filename in ('config.yml', 'model_best.pth'):
            relative = f'weights/{name}/{filename}'
            path = source/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = (name+filename).encode()
            path.write_bytes(content)
            manifest['weights'][relative] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    return source, path, manifest, names


def verify(fixture, hashes=True):
    source, path, manifest, names = fixture
    return verify_foundationpose_files(source, path, 'pinned', names, manifest['native_sources'], hashes)


def test_pose_manifest_hash_verification_catches_same_size_corruption(pose_manifest):
    assert verify(pose_manifest)['sha256_checked']
    source = pose_manifest[0]
    weight = source/'weights/refiner/model_best.pth'
    weight.write_bytes(b'x'*weight.stat().st_size)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verify(pose_manifest)
    assert not verify(pose_manifest, hashes=False)['sha256_checked']


def test_pose_manifest_requires_both_model_weights_and_configs(pose_manifest):
    _, path, manifest, _ = pose_manifest
    del manifest['weights']['weights/scorer/config.yml']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='omits required'):
        verify(pose_manifest)


def test_pose_manifest_refuses_paths_outside_weights(pose_manifest):
    _, path, manifest, _ = pose_manifest
    manifest['weights']['../escape.bin'] = {'bytes': 1, 'sha256': '0'*64}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='escapes weights'):
        verify(pose_manifest)


def test_pose_manifest_rejects_wrong_pinned_source(pose_manifest):
    _, path, manifest, _ = pose_manifest
    manifest['revision'] = 'different'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source/revision'):
        verify(pose_manifest)


def test_pin_reader_never_executes_source(tmp_path):
    path = tmp_path/'pins.py'
    path.write_text("raise RuntimeError('must never execute')\nPIN='revision'\nWEIGHTS=('a','b')\n")
    assert literal_constants(path, {'PIN', 'WEIGHTS'}) == {'PIN': 'revision', 'WEIGHTS': ('a', 'b')}


def test_explicit_hash_verify_and_skip_cannot_be_combined():
    with pytest.raises(SystemExit) as failure:
        main(['--verify-models', '--skip-hashes'])
    assert failure.value.code == 2
