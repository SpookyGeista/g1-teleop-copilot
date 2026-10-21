"""Read-only source-file fingerprints captured at process initialization."""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
import time


def source_manifest(root,role):
    root=Path(root).resolve()
    # Perception workers import modules outside copilot (model_worker,
    # learned_perception, multiview_tracker, sensor geometry). Keep the full
    # owned package in this initialization fingerprint, including those paths.
    paths=set((root/'teleop_assist').rglob('*.py'))
    paths.update((root/'tools/copilot').glob('*.py'))
    paths.update(root/name for name in ('run_copilot.py','teleop_assist/perception_runtime.py',
        'teleop_assist/visual_tracker.py','teleop_assist/learned_perception.py','teleop_assist/model_bridge.py'))
    return dict(schema='punktnull.runtime_source_files.v1',role=role,
        captured_at_host_monotonic=time.monotonic(),python_version=sys.version,
        files={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()},
        scope='Source files at initialization; not a statement about files changed later.')
