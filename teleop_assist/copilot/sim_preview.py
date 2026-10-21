"""Read-only whole-body reranking of a recorded complete PhysX snapshot.

Every retained model proposal is evaluated before truncation. The independent
original IK and q43 endpoint checks are preview hints only: all returned
collision_valid flags stay false. This module never creates motion authority,
retimes capture evidence, runs a GPU model or imports a robot transport.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from .assets import descriptor_evidence
from .kinematics import ARM_JOINT_NAMES
from .observations import rigid

POLICY = 'complete_static_sim_orientation'
SCENE_SOURCE = 'current_simulator_complete_collision_scene'
SCHEMA = 'punktnull.recorded_sim_preview.v1'
MAX_JSON_BYTES = 64 * 1024 * 1024
DESCRIPTOR = Path(__file__).parent / 'assets/dex3_right/config.json'


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _identity(value):
    return isinstance(value, str) and 0 < len(value) <= 512


def _bytes(value):
    result = json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
    if len(result) > MAX_JSON_BYTES:
        raise ValueError('Simulation preview JSON exceeds the 64 MiB file-IPC limit')
    return result


def _hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def read_json_file(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('Simulation preview JSON exceeds the 64 MiB file-IPC limit')
    with path.open('rb') as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise ValueError('Simulation preview JSON exceeds the 64 MiB file-IPC limit')
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(data, parse_constant=invalid_constant)


def write_json_file(path, value):
    path = Path(path)
    data = _bytes(value)
    temporary = path.with_name(path.name + '.partial')
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_request(request):
    if not isinstance(request, dict) or request.get('policy') != POLICY:
        raise ValueError('Explicit complete_static_sim_orientation policy required')
    batch, state, scene = (request.get(key) for key in ('batch', 'robot_state', 'scene'))
    if not all(isinstance(value, dict) for value in (batch, state, scene)):
        raise ValueError('Raw batch, complete state and decoded same-camera scene are required')
    if (state.get('complete') is not True or not _identity(state.get('source')) or
            scene.get('source') != SCENE_SOURCE or scene.get('geometry_complete') is not True or
            scene.get('static') is not True):
        raise ValueError('Independent complete static PhysX state/scene required')
    velocity = np.asarray(scene.get('observed_target_velocity'), dtype=float)
    if (velocity.shape != (6,) or not np.isfinite(velocity).all() or
            np.linalg.norm(velocity[:3]) > .01 or np.linalg.norm(velocity[3:]) > .02):
        raise ValueError('Recorded target must have independently measured stationary velocity')
    stamp, clock = batch.get('capture_timestamp'), batch.get('clock_id')
    if (not _number(stamp) or stamp < 0 or not _identity(clock) or
            any(not _identity(batch.get(key)) for key in ('target_id', 'observation_id', 'calibration_id')) or
            not _number(state.get('acquisition_timestamp')) or not _number(scene.get('acquisition_timestamp')) or
            state.get('clock_id') != clock or scene.get('clock_id') != clock or
            abs(state['acquisition_timestamp']-stamp) > 1e-8 or
            abs(scene['acquisition_timestamp']-stamp) > .02):
        raise ValueError('Model/state must share the original camera capture; scene clock/skew must match')
    if not isinstance(state.get('joint_positions'), dict) or len(state['joint_positions']) != 43:
        raise ValueError('Complete named q43 state required')
    rigid(state.get('base_in_world'), 'base_in_world')
    bodies = state.get('observed_body_world_poses')
    if not isinstance(bodies, dict):
        raise ValueError('Measured original body frames required')
    rigid(bodies.get('left_wrist_yaw_link'), 'measured_left_wrist')
    candidates = batch.get('candidates')
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 512:
        raise ValueError('Require 1..512 retained model candidates')
    ids = set()
    for candidate in candidates:
        if (not isinstance(candidate, dict) or not _identity(candidate.get('candidate_id')) or
                candidate['candidate_id'] in ids or not _identity(candidate.get('source')) or
                not _number(candidate.get('model_score')) or candidate.get('geometry_valid') is not True):
            raise ValueError('Unique finite descriptor-validated model proposals required')
        ids.add(candidate['candidate_id'])
        rigid(candidate.get('base_ee'), 'candidate_base_ee')
        rigid(candidate.get('world_canonical'), 'candidate_world_canonical')
    _bytes(request)
    return batch, state, scene


def _check_snapshot_evidence(evidence, state, scene):
    expected = hashlib.sha256(json.dumps(dict(robot_state=state, scene=scene),
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    if (evidence.get('reason') != 'evaluated' or evidence.get('scope') != 'sim_candidate_endpoint_only' or
            evidence.get('path_check_requested') is not False or evidence.get('live_execution_authorized') is not False or
            evidence.get('current_state_precheck', {}).get('collision_valid') is not True or
            evidence.get('urdf_vs_observed_frames', {}).get('valid') is not True or
            evidence.get('snapshot_sha256') != expected or evidence.get('robot_state') != state or evidence.get('scene') != scene):
        raise ValueError('Recorded full robot snapshot failed exact independent FK/collision evidence binding: ' + str(evidence.get('reason')))


def evaluate_sim_preview(request, repository, *, ik_factory=None, collision_factory=None,
                         descriptor_validator=descriptor_evidence):
    """Worker-side synchronous CPU implementation; injected factories are tests only."""
    batch, state, scene = _validate_request(request)
    descriptor = descriptor_validator(DESCRIPTOR)
    if descriptor.get('geometry_valid') is not True or batch.get('descriptor_sha256') != descriptor['descriptor_sha256']:
        raise ValueError('Model descriptor does not match the hash-verified Dex3 geometry')
    if ik_factory is None:
        from .kinematics import OriginalIKValidator
        ik_factory = OriginalIKValidator
    if collision_factory is None:
        from .arm_collision import OriginalArmCollision
        collision_factory = OriginalArmCollision
    started = time.perf_counter()
    collision = collision_factory(repository)
    # Establish q43 limits, measured-body FK and current collision consistency
    # before spending work on any proposed endpoint.
    preflight = collision.validate_request(dict(batch={'candidates':[
        dict(candidate_id='snapshot_preflight', base_ee=np.eye(4).tolist(), ik_valid=False)]},
        robot_state=deepcopy(state), scene=deepcopy(scene)), endpoint_only=True)['arm_collision_validation']
    _check_snapshot_evidence(preflight, state, scene)
    ik = ik_factory(repository)
    current = [state['joint_positions'][name] for name in ARM_JOINT_NAMES]
    tcp = np.eye(4); tcp[0, 3] = .05
    left = np.linalg.inv(rigid(state['base_in_world'])) @ rigid(state['observed_body_world_poses']['left_wrist_yaw_link']) @ tcp
    output = []; chunk_counts = []
    for offset in range(0, len(batch['candidates']), 128):
        part = deepcopy(batch); part['candidates'] = part['candidates'][offset:offset+128]
        checked_ik = ik.validate_request(dict(batch=part, current_arm_q=current,
            left_base_ee=left.tolist(), joint_state_acquisition_timestamp=state['acquisition_timestamp'],
            joint_state_clock_id=state['clock_id'], joint_state_source=state['source']))
        checked = collision.validate_request(dict(batch=checked_ik, robot_state=deepcopy(state),
            scene=deepcopy(scene)), endpoint_only=True)
        evidence = checked['arm_collision_validation']
        _check_snapshot_evidence(evidence, state, scene)
        chunk_counts.append(len(checked['candidates']))
        for source, candidate in zip(part['candidates'], checked['candidates'], strict=True):
            if candidate['candidate_id'] != source['candidate_id']:
                raise ValueError('Independent validator changed candidate identity/order')
            kin = candidate['kinematic_validation']; endpoint = candidate['arm_collision_validation']
            clear = candidate.get('ik_valid') is True and candidate.get('collision_valid') is True
            row = {key: deepcopy(source[key]) for key in
                ('candidate_id', 'base_ee', 'world_canonical', 'model_score', 'geometry_valid', 'source')}
            row.update(ik_valid=candidate.get('ik_valid') is True, collision_valid=False,
                recorded_sim_endpoint_clear=bool(clear), preview_only=True, motion_authorized=False,
                validation_reason='recorded_sim_endpoint_clear_preview_only' if clear else 'recorded_sim_endpoint_rejected',
                kinematic_validation={key:deepcopy(kin[key]) for key in
                    ('ik_valid','reason','q_initial','q_solution','left_target_base_ee','right_target_base_ee',
                     'final','solver_invoked','validation_method') if key in kin},
                recorded_sim_endpoint_validation={key:deepcopy(endpoint[key]) for key in
                    ('reason','witness','minimum_sample_distance_m','conservative_path_clearance_lower_bound_m',
                     'full_q43_endpoint_fk') if key in endpoint})
            row['recorded_sim_endpoint_validation'].update(scope='recorded_sim_endpoint_only_preview',
                path_validated=False, motion_authorized=False, live_execution_authorized=False)
            output.append(row)
    # Python's stable sort preserves the original order when scores tie.
    output.sort(key=lambda item:(not item['recorded_sim_endpoint_clear'], -item['model_score']))
    result = {key:deepcopy(value) for key,value in batch.items() if key != 'candidates'}
    result.update(candidates=output, valid=False, motion_static=False, motion_authorized=False, path_validated=False,
        sim_preview_validation=dict(schema=SCHEMA, policy=POLICY, recorded_snapshot_only=True,
            motion_authorized=False, hardware_authorized=False, path_validated=False,
            source_timestamps_preserved=True, capture_timestamp=batch['capture_timestamp'], clock_id=batch['clock_id'],
            state_capture_timestamp=state['acquisition_timestamp'], scene_capture_timestamp=scene['acquisition_timestamp'],
            evaluated_count=len(output), ik_accepted_count=sum(c['ik_valid'] for c in output),
            recorded_sim_endpoint_clear_count=sum(c['recorded_sim_endpoint_clear'] for c in output),
            chunk_counts=chunk_counts, ordering='recorded_endpoint_clear_first_then_descending_uncalibrated_model_score',
            elapsed_s=time.perf_counter()-started,
            source_sha256=dict(batch=_hash(batch), robot_state=_hash(state), scene=_hash(scene),
                descriptor=batch['descriptor_sha256'], module=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
            manifests=dict(ik=ik.manifest, collision=collision.manifest),
            limitations=['Recorded endpoint feasibility is only a ranking hint, never a live execution permission',
                'All actual commands still require current full-body state/scene and short-path validation',
                'Model score is uncalibrated; no grasp stability, finger closure or autonomous approach checked']))
    _bytes(result)
    return result


def isolated_sim_preview(request, repository, python_executable, timeout_s=60):
    """Evaluate the entire retained batch in an isolated CPU environment.

    File IPC is bounded to 64 MiB; the 2 MiB live control mailbox is not used.
    Original acquisition times are carried through without freshness renewal.
    """
    batch, state, scene = _validate_request(request)
    if not _number(timeout_s) or not 0 < timeout_s <= 120:
        raise ValueError('Preview worker timeout must be in (0,120] seconds')
    script = Path(__file__).resolve().parents[2] / 'tools/copilot/preview_worker.py'
    with tempfile.TemporaryDirectory(prefix='punktnull-sim-preview-') as temporary:
        directory = Path(temporary); source=directory/'input.json'; target=directory/'output.json'; log=directory/'worker.log'
        write_json_file(source, request)
        # Preserve a venv's executable symlink instead of resolving to Python's
        # base interpreter and silently bypassing the isolated environment.
        command = [os.path.abspath(os.fspath(python_executable)), str(script), '--input', str(source),
            '--output', str(target), '--repository', str(repository)]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with log.open('wb') as stream:
            try:
                subprocess.run(command, cwd=script.parents[2], env=environment, timeout=timeout_s,
                    check=True, stdout=stream, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as error:
                with log.open('rb') as failure:
                    failure.seek(max(0,log.stat().st_size-4096))
                    detail=failure.read().decode(errors='replace')
                raise RuntimeError('Isolated simulation preview failed: '+detail) from error
        result=read_json_file(target)
        if not isinstance(result,dict) or not isinstance(result.get('sim_preview_validation'),dict):
            raise ValueError('Malformed isolated simulation preview result')
        evidence=result['sim_preview_validation']
        if (evidence.get('schema')!=SCHEMA or evidence.get('policy')!=POLICY or
                any(evidence.get('source_sha256',{}).get(key)!=_hash(value) for key,value in
                    (('batch',batch),('robot_state',state),('scene',scene))) or result.get('valid') is not False or
                result.get('motion_authorized') is not False or
                any(result.get(key)!=batch.get(key) for key in
                    ('capture_timestamp','clock_id','calibration_id','target_id','observation_id','descriptor_sha256')) or
                len(result.get('candidates',[]))!=len(batch['candidates']) or
                {c.get('candidate_id') for c in result['candidates']}!={c['candidate_id'] for c in batch['candidates']} or
                any(c.get('collision_valid') is not False or c.get('motion_authorized') is not False
                    or type(c.get('recorded_sim_endpoint_clear')) is not bool for c in result['candidates'])):
            raise ValueError('Isolated preview result lost provenance or violated its read-only contract')
        return result
