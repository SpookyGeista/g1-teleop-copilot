"""Stdlib-only resolution of reviewed original simulator assets."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path

ISAAC_REPOSITORY_ENV = 'PUNKTNULL_ISAAC_REPOSITORY'
CAMERA_SOURCE_RELATIVE = 'assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd'
CAMERA_SOURCE_SHA256 = 'd23b9bbc8b59cf20fa1983f55874f855a76e038b9e22eae39542e62be5b4ba3d'


def camera_source_usd(source, *, isaac_repository=None):
    """Relocate the known asset, never substitute a different geometry hash.

    Explicit configured repository wins over the environment. The recorded
    absolute source remains a same-host fallback only when no override exists.
    An invalid override never silently falls back to a different installation.
    """
    if (source.get('source_usd_relative') != CAMERA_SOURCE_RELATIVE
            or source.get('source_usd_sha256') != CAMERA_SOURCE_SHA256):
        raise ValueError('Camera proxy source path/hash differs from reviewed asset')
    repository = isaac_repository if isaac_repository is not None else os.environ.get(ISAAC_REPOSITORY_ENV)
    if repository is not None:
        if not str(repository).strip():
            raise ValueError('Empty configured Isaac repository')
        path = Path(repository).expanduser() / CAMERA_SOURCE_RELATIVE
    else:
        path = Path(source['source_usd']).expanduser()
    path = path.resolve()
    if not path.is_file():
        raise ValueError('Pinned source USD for camera proxies is missing')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != CAMERA_SOURCE_SHA256:
        raise ValueError('Pinned source USD for camera proxies changed')
    return path
