"""Pinned official checkpoints. Loading never implicitly contacts a model hub."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / 'models'
MODELS = {
    'grounding_dino': {
        'repo_id': 'IDEA-Research/grounding-dino-tiny',
        'revision': 'a2bb814dd30d776dcf7e30523b00659f4f141c71',
        'directory': 'grounding-dino-tiny',
        'license': 'Apache-2.0',
        'purpose': 'text-conditioned object detection',
    },
    'sam2': {
        'repo_id': 'facebook/sam2.1-hiera-tiny',
        'revision': 'de431c4043854a71d8101e17995dfe596bf101a5',
        'directory': 'sam2.1-hiera-tiny',
        'license': 'Apache-2.0',
        'purpose': 'box-prompted instance segmentation',
    },
    'grounding_dino_base': {
        'repo_id': 'IDEA-Research/grounding-dino-base',
        'revision': '12bdfa3120f3e7ec7b434d90674b3396eccf88eb',
        'directory': 'grounding-dino-base',
        'license': 'Apache-2.0',
        'purpose': 'optional stronger text-conditioned object detection',
    },
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def model_paths(model_root=None, names=('grounding_dino', 'sam2')):
    """Require downloaded safetensors and configuration; fail closed otherwise."""
    root = Path(model_root or DEFAULT_MODEL_ROOT).expanduser().resolve()
    paths = {name: root / MODELS[name]['directory'] for name in names}
    missing = [str(path / item) for path in paths.values()
               for item in ('model.safetensors', 'config.json', 'preprocessor_config.json')
               if not (path / item).is_file()]
    if missing:
        raise FileNotFoundError('Pretrained models are unavailable. Run '
                                'tools/setup_models.py --download. Missing: ' + ', '.join(missing))
    return paths


def model_manifest(model_root=None):
    root = Path(model_root or DEFAULT_MODEL_ROOT)
    path = root / 'manifest.json'
    return json.loads(path.read_text()) if path.is_file() else None
