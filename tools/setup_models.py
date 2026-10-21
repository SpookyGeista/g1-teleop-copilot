#!/usr/bin/env python3
"""Download fixed official checkpoints without touching the Isaac environment."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop_assist.model_registry import DEFAULT_MODEL_ROOT, MODELS, model_paths, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument('--download', action='store_true', help='Explicitly download pinned weights')
    parser.add_argument('--verify', action='store_true', help='Verify every file against recorded SHA256')
    parser.add_argument('--only', nargs='+', choices=list(MODELS), default=None)
    args = parser.parse_args()
    root = args.model_root.resolve()
    if args.download:
        from huggingface_hub import snapshot_download
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'format': 1, 'models': {}}
        manifest['created_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        for name in args.only or list(MODELS):
            info = MODELS[name]
            destination = root / info['directory']
            snapshot_download(repo_id=info['repo_id'], revision=info['revision'],
                              local_dir=str(destination),
                              allow_patterns=['*.json', '*.txt', '*.md', '*.safetensors'])
            files = {str(p.relative_to(destination)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                     for p in sorted(destination.iterdir()) if p.is_file()}
            manifest['models'][name] = {**info, 'files': files}
            print(f"Downloaded {info['repo_id']} at {info['revision']}", flush=True)
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    paths = model_paths(root, args.only or list(MODELS))
    if args.verify:
        manifest = json.loads((root / 'manifest.json').read_text())
        for name in args.only or list(manifest['models']):
            info = manifest['models'][name]
            if info['revision'] != MODELS[name]['revision']:
                raise RuntimeError(f'Revision mismatch: {name}')
            for relative, expected in info['files'].items():
                path = paths[name] / relative
                if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
                    raise RuntimeError(f'Model integrity mismatch: {path}')
        print('All model file SHA256 values match the manifest.')
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == '__main__':
    main()
