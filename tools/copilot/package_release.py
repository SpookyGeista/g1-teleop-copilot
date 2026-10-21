#!/usr/bin/env python3
"""Build a local source/evidence bundle, excluding environments and credentials.

This does not publish, upload, license project-owned code, or authorize hardware.
Original dependencies and model weights are installed separately from pinned
sources. Bundled third-party geometry retains its redistribution notices.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[2]
TREES=('teleop_assist','tools/copilot','research/masters_20261003/ai_assistance/dex3_descriptor')
FILES=('run_copilot.py','configs/copilot.local.json','requirements-models.txt','requirements-pose.txt',
    'pytest.ini','docs/COPILOT.md','tools/setup_models.py','tools/assist_tls.py',
    'research/masters_20261003/ai_assistance/README.md',
    'research/masters_20261003/ai_assistance/inference_summary.json',
    'research/masters_20261003/ai_assistance/shared_control_sources.json',
    'research/masters_20261003/ai_assistance/model_dataset_evidence/model_dataset_reuse_recommendation.json',
    # --xr-check launches this standalone script. Its sibling is imported by
    # the child interpreter, so ordinary package import collection misses it.
    'research/masters_20261003/ai_assistance/xr_bridge/probe_original_ik.py',
    'research/masters_20261003/ai_assistance/xr_bridge/shadow_bridge.py',
    # Owned browser/PhysX trial and offline acceptance tools. Original Vuer,
    # robot sources, model weights and TLS keys remain external prerequisites.
    'runs/copilot_xr_physx_live_001/run_experiment.py',
    'runs/copilot_xr_physx_live_001/source_guard.py',
    'runs/copilot_xr_physx_live_001/browser_driver.js',
    'runs/copilot_xr_physx_live_001/browser_target_ui.js',
    'runs/copilot_xr_physx_live_001/camera_observer.js',
    'runs/copilot_xr_physx_live_001/acceptance_contract.json',
    'runs/copilot_xr_physx_live_001/fixture_source_manifest.json',
    'runs/copilot_xr_physx_live_001/upstream_sources.json',
    'runs/copilot_xr_physx_live_001/check_live_run.py',
    'runs/copilot_xr_physx_live_001/test_live_run.py',
    'runs/copilot_xr_physx_live_001/checker_contract.json',
    'runs/copilot_xr_physx_001/check_browser_sim_run.py',
    'runs/copilot_xr_physx_001/check_operability.py',
    'runs/copilot_xr_physx_001/acceptance_contract.json',
    'runs/copilot_xr_physx_001/operability_contract_v1.json',
    'runs/copilot_xr_physx_001/browser_driver.js')
# Retain the failed software-renderer fixture as well as its separately
# versioned hardware-renderer comparison; earlier evidence stays reproducible.
FILES += tuple(name.replace('copilot_xr_physx_live_001/', 'copilot_xr_physx_live_002/')
               for name in FILES if name.startswith('runs/copilot_xr_physx_live_001/'))
MAX_EVIDENCE_BYTES=64*1024*1024  # Same bound as the recorded-run validator.
EXCLUDED={'__pycache__','models','weights','checkpoints','venv','venvs','node_modules',
          'certs','certificates','credentials','secrets','private'}
ALLOWED_SUFFIXES={'.py','.json','.html','.js','.css','.txt','.md','.urdf','.stl','.obj','.yaml','.yml','.toml','.ini'}
HARDWARE_FLAGS={'hardware_tested','hardware_ready','hardware_authorized','hardware_validated',
                'real_robot_tested','real_robot_ready','motion_ready','hardware_motion_authorized'}


def excluded(relative):
    if any(part.startswith('.') or part.lower() in EXCLUDED for part in relative.parts):
        return True
    if relative.suffix.lower() not in {'.py','.js','.html','.css','.md'}:
        words=set(re.split(r'[^a-z0-9]+',relative.stem.lower()))
        if words & {'credential','credentials','secret','secrets','certificate','certificates',
                    'cert','token','tokens','private','password','passwords'}:
            return True
    return False


def validate_acceptance(value):
    if not isinstance(value,dict) or value.get('hardware_tested') is not False:
        raise ValueError('This source bundle requires hardware_tested=false')
    def inspect(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key.lower() in HARDWARE_FLAGS and item is not False and item is not None:
                    raise ValueError('No positive hardware or motion-readiness claim is supported')
                inspect(item)
        elif isinstance(value,list):
            for item in value:inspect(item)
    inspect(value)
    json.dumps(value,allow_nan=False)
    return dict(value,packaging_scope='simulation_source_and_evidence_only',hardware_authorized=False,
                hardware_ready=False)


def verify_descriptor(entries):
    """The bundle must contain the exact geometry its validation file names."""
    lookup=dict(entries);prefix='teleop_assist/copilot/assets/dex3_right/'
    evidence=json.loads(lookup[prefix+'validation.json'])
    hashes=evidence.get('asset_sha256',{})
    if len(hashes)!=14 or evidence.get('geometry_valid') is not True:
        raise ValueError('Expected complete reviewed descriptor evidence')
    for relative,expected in hashes.items():
        data=lookup.get(prefix+relative)
        if data is None or hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError('Required descriptor asset missing or changed: '+relative)


def sources(root=ROOT):
    paths={root/name for name in FILES}
    paths.update(root.glob('tests/test_copilot*.py'))
    for tree in TREES:
        paths.update((root/tree).rglob('*'))
    result=[]
    for path in sorted(paths):
        relative=path.relative_to(root)
        if (not path.is_file() or path.is_symlink() or excluded(relative)
                or path.suffix.lower() not in ALLOWED_SUFFIXES):continue
        path.resolve().relative_to(root.resolve())
        result.append((str(relative),path.read_bytes()))
    selected={name for name,_ in result}
    missing=[name for name in FILES if name not in selected]
    if missing:raise ValueError('Missing release sources: '+', '.join(missing))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--acceptance',type=Path,required=True,help='Explicit factual acceptance JSON')
    parser.add_argument('--evidence',type=Path,action='append',default=[],help='Explicit JSON/JSONL/NPZ/PNG/JPG/TXT evidence, at most 64 MiB per file, inside this project')
    args=parser.parse_args()
    if args.output.exists():parser.error('Refusing to replace existing release')
    try:acceptance=validate_acceptance(json.loads(args.acceptance.read_text()))
    except (ValueError,TypeError) as error:parser.error(str(error))
    entries=sources()
    verify_descriptor(entries)
    for path in args.evidence:
        if path.is_symlink():parser.error('Evidence symlinks are not bundled')
        path=path.absolute();relative=path.relative_to(ROOT.resolve())
        path.resolve().relative_to(ROOT.resolve())
        if (excluded(relative) or path.suffix.lower() not in {'.json','.jsonl','.npz','.png','.jpg','.txt'}
                or path.stat().st_size>MAX_EVIDENCE_BYTES):
            parser.error('Evidence must be JSON, JSONL, NPZ, PNG, JPG or TXT, at most 64 MiB per file')
        entries.append((str(relative),path.read_bytes()))
    entries.append(('acceptance.json',json.dumps(acceptance,indent=2,allow_nan=False).encode()))
    entries.append(('START_HERE.txt',b'G1/Dex3 Copilot simulation release candidate\n\n'
        b'Start with docs/COPILOT.md and acceptance.json.\n'
        b'Configure paths in configs/copilot.local.json, then run:\n'
        b'  python3 run_copilot.py --doctor\n'
        b'  python3 run_copilot.py --demo\n'
        b'  python3 run_copilot.py --assist-sim --scripted-test\n\n'
        b'Live model pipeline in isolated simulation, with labelled synthetic input:\n'
        b'  python3 run_copilot.py --assist-sim --live-ai --camera-source right_wrist --scripted-test\n\n'
        b'Original Unitree XR/Isaac environments, model weights and a GPU are prerequisites.\n'
        b'Pinned model setup: tools/copilot/README_SETUP.md\n'
        b'No hardware authorization or successful human/headset trial is implied.\n'
        b'Third-party notices are retained beside their assets. No license for\n'
        b'project-owned code is assigned by this packaging operation.\n'
        b'SHA256_MANIFEST.json lists every other ZIP entry and excludes itself.\n'))
    unique={}
    for name,data in entries:
        if name in unique and unique[name]!=data:raise ValueError('Conflicting duplicate ZIP entry: '+name)
        unique[name]=data
    entries=list(unique.items())
    manifest={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)) for name,data in entries}
    entries.append(('SHA256_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True).encode()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent,prefix='.'+args.output.name+'.',suffix='.partial',delete=False) as stage:
        temporary=Path(stage.name)
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name,data in sorted(entries):
                item=zipfile.ZipInfo(name,date_time=(2026,10,4,0,0,0))
                item.compress_type=zipfile.ZIP_DEFLATED;item.external_attr=0o100644<<16
                archive.writestr(item,data)
        with zipfile.ZipFile(temporary) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)) or set(names)!=set(manifest)|{'SHA256_MANIFEST.json'}:
                raise ValueError('Archive entry list differs from manifest')
            for name,info in manifest.items():
                data=archive.read(name)
                if len(data)!=info['bytes'] or hashlib.sha256(data).hexdigest()!=info['sha256']:
                    raise ValueError('Archive verification failed: '+name)
        temporary.replace(args.output)
    finally:temporary.unlink(missing_ok=True)
    print(json.dumps(dict(output=str(args.output),files=len(entries),bytes=args.output.stat().st_size,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),hardware_authorized=False),indent=2))


if __name__=='__main__':main()
