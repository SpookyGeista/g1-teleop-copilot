"""Validate the reviewed descriptor bundle before using its geometric claims."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def descriptor_evidence(descriptor):
    path = Path(descriptor).resolve()
    report = json.loads(path.with_name('validation.json').read_text())
    if (report.get('validation_schema') != 'punktnull.copilot.descriptor_validation.v1'
            or report.get('geometry_valid') is not True
            or report.get('limits_passed') is not True):
        raise ValueError('Descriptor has no completed geometry validation')
    hashes = report.get('asset_sha256', {})
    required = {'config.json', 'gripper.urdf', 'contact_patches.json', 'coll_mesh.obj', 'vis_mesh.obj'}
    if path.name != 'config.json' or not required <= hashes.keys():
        raise ValueError('Incomplete descriptor validation manifest')
    for relative, expected in hashes.items():
        asset = (path.parent / relative).resolve()
        if not asset.is_relative_to(path.parent) or not asset.is_file():
            raise ValueError('Missing or invalid descriptor asset')
        if hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Descriptor asset changed after validation: {relative}')
    return {'geometry_valid': True, 'scope': report['geometry_valid_scope'],
            'descriptor_sha256': hashes['config.json'], 'asset_count': len(hashes)}
