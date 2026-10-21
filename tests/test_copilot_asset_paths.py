"""Repository relocation must retain exact reviewed source geometry."""
import hashlib

import pytest

from teleop_assist.copilot import asset_paths


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    data = b'test-only reviewed USD bytes'
    digest = hashlib.sha256(data).hexdigest()
    # Fixture pin replaces the production constant for these small CPU tests.
    monkeypatch.setattr(asset_paths, 'CAMERA_SOURCE_SHA256', digest)
    monkeypatch.delenv(asset_paths.ISAAC_REPOSITORY_ENV, raising=False)
    old = tmp_path / 'old.usd'
    old.write_bytes(data)
    root = tmp_path / 'other_checkout'
    relocated = root / asset_paths.CAMERA_SOURCE_RELATIVE
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(data)
    source = dict(source_usd=str(old), source_usd_relative=asset_paths.CAMERA_SOURCE_RELATIVE,
                  source_usd_sha256=digest)
    return source, root, relocated, old


def test_environment_relocates_same_hashed_asset(fixture, monkeypatch):
    source, root, relocated, old = fixture
    old.unlink()
    monkeypatch.setenv(asset_paths.ISAAC_REPOSITORY_ENV, str(root))
    assert asset_paths.camera_source_usd(source) == relocated


def test_explicit_config_wins_over_unrelated_environment(fixture, monkeypatch):
    source, root, relocated, _ = fixture
    monkeypatch.setenv(asset_paths.ISAAC_REPOSITORY_ENV, '/missing/other/install')
    assert asset_paths.camera_source_usd(source, isaac_repository=root) == relocated


def test_mismatched_override_never_falls_back_to_valid_old_asset(fixture, monkeypatch):
    source, root, relocated, old = fixture
    relocated.write_bytes(b'different geometry')
    assert old.is_file()
    monkeypatch.setenv(asset_paths.ISAAC_REPOSITORY_ENV, str(root))
    with pytest.raises(ValueError, match='changed'):
        asset_paths.camera_source_usd(source)


@pytest.mark.parametrize('field,value', [('source_usd_relative', '../unreviewed.usd'),
                                         ('source_usd_sha256', '0'*64)])
def test_metadata_cannot_change_expected_path_or_hash(fixture, field, value):
    source, root, _, _ = fixture
    source[field] = value
    with pytest.raises(ValueError, match='differs'):
        asset_paths.camera_source_usd(source, isaac_repository=root)


def test_missing_override_does_not_use_valid_absolute_fallback(fixture, monkeypatch):
    source, root, relocated, _ = fixture
    relocated.unlink()
    monkeypatch.setenv(asset_paths.ISAAC_REPOSITORY_ENV, str(root))
    with pytest.raises(ValueError, match='missing'):
        asset_paths.camera_source_usd(source)


def test_empty_override_is_rejected(fixture, monkeypatch):
    monkeypatch.setenv(asset_paths.ISAAC_REPOSITORY_ENV, '')
    with pytest.raises(ValueError, match='Empty'):
        asset_paths.camera_source_usd(fixture[0])


def test_same_host_fallback_still_verifies_pin(fixture):
    source, _, _, old = fixture
    assert asset_paths.camera_source_usd(source) == old
    old.write_bytes(b'changed original')
    with pytest.raises(ValueError, match='changed'):
        asset_paths.camera_source_usd(source)
