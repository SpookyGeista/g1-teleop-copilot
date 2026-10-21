"""Bind untrusted learned previews to independently checked simulator evidence.

This module does not run perception, IK, physics, or transport. The runtime must
record actual published camera frames and their same-state complete simulator
scene, then pass a separate validate_sim_candidate worker result. A preview's
valid/collision flags are never an execution permission. The returned one-item
batch permits controller selection only; ActualMotionGuard still checks every
concrete human-controlled command and its short path.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from .controller import rigid
from .kinematics import (ALREADY_AT_TARGET_ORIENTATION_TOLERANCE_RAD,
                         ALREADY_AT_TARGET_POSITION_TOLERANCE_M, ARM_JOINT_NAMES, arm_vector)
from .types import GraspCandidate, ProposalBatch


OBSERVED_PREGRASP_REQUIRED = 'observed_pregrasp_required'
COMPLETE_STATIC_SIM_ORIENTATION = 'complete_static_sim_orientation'
COMPLETE_SIM_SCENE_SOURCE = 'current_simulator_complete_collision_scene'


def _number(value):
    try:
        return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number)) and math.isfinite(value)
    except (TypeError, OverflowError):
        return False


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _identity(value):
    return isinstance(value, str) and 0 < len(value) <= 512


def _pose_close(a, b, position=.001, rotation=.005):
    a, b = rigid(a), rigid(b)
    return bool(np.linalg.norm(a[:3, 3]-b[:3, 3]) <= position and
        Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude() <= rotation)


def _fresh(stamp, clock, now, expected, maximum):
    if not _number(stamp) or clock != expected or not -.01 <= now-stamp <= maximum:
        raise ValueError('evidence_clock_stale_or_future')


def _scene_geometry(scene):
    if (not isinstance(scene, dict) or scene.get('geometry_complete') is not True or
            scene.get('static') is not True or not _identity(scene.get('scene_id')) or
            not _identity(scene.get('source'))):
        raise ValueError('complete_known_static_sim_scene_required')
    obstacles = scene.get('obstacles')
    if not isinstance(obstacles, list) or not obstacles or any(not isinstance(x, dict) for x in obstacles):
        raise ValueError('scene_obstacles_required')
    ids = [x.get('id') for x in obstacles]
    if any(not _identity(x) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('scene_obstacle_identity_invalid')
    geometry = {'scene_id': scene['scene_id'], 'obstacles': {x['id']: deepcopy(x) for x in obstacles},
                'allowed_static_contacts': deepcopy(scene.get('allowed_static_contacts', []))}
    _hash(geometry)
    for obstacle in geometry['obstacles'].values():
        if obstacle.get('type') != 'plane':
            rigid(obstacle.get('world_pose'))
    return geometry


def _stationary_velocity(scene):
    velocity = np.asarray(scene.get('observed_target_velocity'), dtype=float)
    if (velocity.shape != (6,) or not np.isfinite(velocity).all() or
            np.linalg.norm(velocity[:3]) > .01 or np.linalg.norm(velocity[3:]) > .02):
        raise ValueError('sim_target_motion_or_velocity_unknown')


def _same_geometry(a, b):
    a, b = deepcopy(a), deepcopy(b)
    if a['scene_id'] != b['scene_id'] or a['allowed_static_contacts'] != b['allowed_static_contacts']:
        return False
    if set(a['obstacles']) != set(b['obstacles']):
        return False
    for key in a['obstacles']:
        first, second = a['obstacles'][key], b['obstacles'][key]
        p, q = first.pop('world_pose', None), second.pop('world_pose', None)
        if first != second or (p is None) != (q is None):
            return False
        if p is not None and not _pose_close(p, q):
            return False
    return True


def _recorded_geometry_profile(record):
    """Validate poses once and retain exact non-pose scene parameters.

    Sorted obstacle IDs make the stacked pose order independent of input-list
    order. A missing/None plane pose remains distinct from a present pose, as
    in _same_geometry. The base pose is the final entry and uses the same
    position/rotation tolerances as each obstacle.
    """
    geometry = record['geometry']
    metadata = dict(scene_id=geometry['scene_id'],
        allowed_static_contacts=deepcopy(geometry['allowed_static_contacts']), obstacles={}, pose_ids=[])
    poses = []
    for identity in sorted(geometry['obstacles']):
        obstacle = deepcopy(geometry['obstacles'][identity])
        pose = obstacle.pop('world_pose', None)
        metadata['obstacles'][identity] = obstacle
        if pose is not None:
            metadata['pose_ids'].append(identity)
            poses.append(rigid(pose))
    poses.append(rigid(record['base']))
    stacked = np.stack(poses)
    stacked.setflags(write=False)
    return metadata, stacked


def _recorded_geometry_matches(anchor_profile, profiles):
    """Vectorized equivalent of comparing every recorded scene to its anchor.

    This never chains adjacent-frame tolerances: every transform is compared
    to the original anchor, so slow accumulated drift and movement followed by
    a return remain visible. Exact metadata equality precedes numeric checks.
    All transforms were independently validated at record_frame acquisition.
    """
    metadata, anchor = anchor_profile
    if any(profile[0] != metadata for profile in profiles):
        return False
    poses = np.stack([profile[1] for profile in profiles])
    if not np.all(np.linalg.norm(poses[..., :3, 3]-anchor[None, :, :3, 3], axis=-1) <= .001):
        return False
    relative = np.matmul(anchor[None, :, :3, :3], np.swapaxes(poses[..., :3, :3], -1, -2))
    return bool(np.all(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude() <= .005))


@dataclass(frozen=True)
class PreparedSimProposal:
    ready: bool
    reason: str
    candidate: dict | None = None
    model_source_id: str | None = None
    validation_observation_id: str | None = None
    validation_capture_timestamp: float | None = None
    token: str | None = None


@dataclass(frozen=True)
class SimProposalRecheck:
    """Current binding status only; never an endpoint authorization input."""
    ready: bool
    reason: str
    model_source_id: str | None = None
    validation_observation_id: str | None = None
    validation_capture_timestamp: float | None = None


@dataclass(frozen=True)
class SimProposalAuthorization:
    accepted: bool
    reason: str
    batch: ProposalBatch | None
    evidence: dict


class LiveSimProposals:
    """Single deliberate selection, frozen until reset(), never auto-ranked.

    Call reset only following an explicit target/grasp reselection while the
    control session is held. It keeps actual frame history, but cannot revive a
    failed old selection or manufacture a new observation. The simulator has
    one explicitly identified target obstacle; centroid containment only checks
    consistency of the learned object's association with that known fixture.

    Default eligibility requires the observed pregrasp check. The explicit
    complete_static_sim_orientation policy instead requires the independent
    PhysX scene source at every acquisition and endpoint authorization. It may
    offer a negatively flagged observed pregrasp because that hypothetical
    approach is not executed; full endpoint and actual-command checks remain
    mandatory. Neither policy authorizes hardware or an autonomous approach.
    """

    def __init__(self, *, descriptor_sha256, target_obstacle_id='target', policy=OBSERVED_PREGRASP_REQUIRED):
        if (not isinstance(descriptor_sha256, str) or len(descriptor_sha256) != 64 or
                any(x not in '0123456789abcdef' for x in descriptor_sha256)):
            raise ValueError('Pinned descriptor SHA256 required')
        if not _identity(target_obstacle_id):
            raise ValueError('Explicit simulator target obstacle required')
        if policy not in (OBSERVED_PREGRASP_REQUIRED, COMPLETE_STATIC_SIM_ORIENTATION):
            raise ValueError('Unknown simulator preview collision policy')
        self.policy = policy
        self.descriptor_sha256, self.target_obstacle_id = descriptor_sha256, target_obstacle_id
        self.frames = OrderedDict()
        self._geometry_profiles = OrderedDict()
        self._last_stamp = None
        self._last_context = None
        self._record_sequence = 0
        self._invalid_record_barrier = 0
        self._frozen = self._prepared = None
        self.invalid_reason = None

    def reset(self):
        """Revoke the binding after deliberate selection; no motion authority."""
        self._frozen = self._prepared = None
        self.invalid_reason = None

    def _fail(self, reason, *, prepare=True):
        if self._frozen is not None and self.invalid_reason is None:
            self.invalid_reason = reason
        self._prepared = None
        status = PreparedSimProposal if prepare else SimProposalRecheck
        return status(False, self.invalid_reason or reason)

    def _require_scene_policy(self, scene):
        if self.policy == COMPLETE_STATIC_SIM_ORIENTATION and (
                not isinstance(scene, dict) or scene.get('source') != COMPLETE_SIM_SCENE_SOURCE):
            raise ValueError('complete_static_sim_orientation_requires_independent_PhysX_scene')

    def record_frame(self, frame, scene):
        """Record source time, base and complete scene of a real camera frame.

        No images are retained. The caller must take scene before another
        physics step; timestamp skew is additionally bounded to 20 ms. Repeated
        frame IDs never count as independent stationary observations.
        """
        self._record_sequence += 1
        try:
            stamp, clock, calibration = frame['timestamp'], frame['clock_id'], frame['calibration_id']
            frame_id = frame['frame_id']
            if (not _number(stamp) or not _identity(clock) or not _identity(calibration) or
                    isinstance(frame_id, bool) or not isinstance(frame_id, (int, np.integer)) or frame_id < 0):
                raise ValueError('frame_source_identity_invalid')
            if scene.get('clock_id') != clock or not _number(scene.get('acquisition_timestamp')) or abs(scene['acquisition_timestamp']-stamp) > .02:
                raise ValueError('frame_scene_source_snapshot_mismatch')
            self._require_scene_policy(scene)
            geometry = _scene_geometry(scene)
            if self.target_obstacle_id not in geometry['obstacles']:
                raise ValueError('known_sim_target_missing')
            _stationary_velocity(scene)
            observation_id = f'{calibration}:{frame_id}'
            record = {'observation_id': observation_id, 'timestamp': float(stamp), 'clock_id': clock,
                      'calibration_id': calibration, 'geometry': geometry,
                      'base': rigid(frame['T_world_base']).tolist()}
            old = self.frames.get(observation_id)
            if old is not None:
                if _hash({k: v for k, v in old.items() if k != 'record_sequence'}) != _hash(record):
                    raise ValueError('recorded_frame_identity_reused_with_changed_content')
                return {'recorded': False, 'reason': 'same_source_frame_no_new_evidence'}
            record['record_sequence'] = self._record_sequence
            context = (clock, calibration)
            if self._last_context is not None and context != self._last_context:
                self.frames.clear()
                self._geometry_profiles.clear()
                self._last_stamp = None
                if self._frozen is not None:
                    self.invalid_reason = 'recorded_clock_or_calibration_changed'
            if self._last_stamp is not None and stamp <= self._last_stamp:
                raise ValueError('recorded_acquisition_nonincreasing')
            if self._frozen is not None:
                anchor = self._frozen['anchor']
                if (self._last_stamp is not None and stamp-self._last_stamp > .25) or not (
                        _same_geometry(anchor['geometry'], geometry) and _pose_close(anchor['base'], record['base'])):
                    self.invalid_reason = 'recorded_sim_scene_moved_or_observation_gap'
            profile = _recorded_geometry_profile(record)
            self.frames[observation_id] = record
            self._geometry_profiles[observation_id] = profile
            self._last_stamp, self._last_context = float(stamp), context
            while len(self.frames) > 1024:
                removed, _ = self.frames.popitem(last=False)
                self._geometry_profiles.pop(removed)
            return {'recorded': True, 'observation_id': observation_id, 'reason': self.invalid_reason or 'actual_sim_frame_recorded'}
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            # A rejected observation cannot disappear from the stationarity
            # audit merely because the model result arrives afterwards. Only a
            # new model anchor observed after this barrier can start anew.
            self._invalid_record_barrier = self._record_sequence
            self._fail(str(exc))
            return {'recorded': False, 'reason': str(exc)}

    def _association(self, centroid, anchor):
        point = np.asarray(centroid, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('model_observed_centroid_unknown')
        obstacle = anchor['geometry']['obstacles'][self.target_obstacle_id]
        local = (np.linalg.inv(rigid(obstacle['world_pose'])) @ np.r_[point, 1.])[:3]
        kind = obstacle.get('type')
        if kind == 'cylinder':
            contained = np.linalg.norm(local[:2]) <= obstacle['radius']+.01 and abs(local[2]) <= obstacle['length']/2+.01
        elif kind == 'sphere':
            contained = np.linalg.norm(local) <= obstacle['radius']+.01
        elif kind == 'box':
            contained = np.all(abs(local) <= np.asarray(obstacle['size_xyz'])/2+.01)
        else:
            raise ValueError('sim_target_shape_unsupported')
        if not contained:
            raise ValueError('learned_target_not_consistent_with_known_sim_fixture')

    def inspect(self, preview, *, selected_candidate_id, confirmed_candidate_id=None,
                now, clock_id, calibration_id):
        """Validate preview identity and actual scene history, without elevating flags."""
        return self._inspect(preview, selected_candidate_id=selected_candidate_id,
            confirmed_candidate_id=confirmed_candidate_id, now=now, clock_id=clock_id,
            calibration_id=calibration_id, prepare=True)

    def recheck(self, preview, *, selected_candidate_id, confirmed_candidate_id=None,
                now, clock_id, calibration_id):
        """Check an existing binding without creating or retaining a capability.

        The caller's pending batch keeps its original timestamps and deadlines.
        This read-only preview check cannot establish a new frozen selection.
        """
        return self._inspect(preview, selected_candidate_id=selected_candidate_id,
            confirmed_candidate_id=confirmed_candidate_id, now=now, clock_id=clock_id,
            calibration_id=calibration_id, prepare=False)

    def _inspect(self, preview, *, selected_candidate_id, confirmed_candidate_id,
                 now, clock_id, calibration_id, prepare):
        self._prepared = None
        try:
            if self.invalid_reason:
                raise ValueError(self.invalid_reason)
            if not prepare and self._frozen is None:
                raise ValueError('frozen_candidate_not_bound_for_recheck')
            if not _number(now) or not isinstance(preview, dict):
                raise ValueError('preview_or_current_time_invalid')
            if not _identity(selected_candidate_id):
                raise ValueError('explicit_grasp_selection_required')
            if confirmed_candidate_id is not None and confirmed_candidate_id != selected_candidate_id:
                raise ValueError('confirmed_candidate_cannot_switch_without_deliberate_reset')
            if preview.get('valid') is not False or preview.get('motion_static') is not True:
                raise ValueError('preview_invalid_or_stationarity_unavailable')
            if preview.get('validation_reason') != 'preview_only_actual_command_path_not_validated':
                raise ValueError('preview_service_unhealthy_or_unknown')
            if preview.get('clock_id') != clock_id or preview.get('calibration_id') != calibration_id:
                raise ValueError('preview_clock_or_calibration_mismatch')
            _fresh(preview.get('capture_timestamp'), preview.get('clock_id'), now, clock_id, .25)
            if preview.get('descriptor_sha256') != self.descriptor_sha256:
                raise ValueError('preview_descriptor_not_pinned_geometry')
            model_obs = preview.get('model_source_observation_id')
            validation_obs = preview.get('validation_observation_id')
            if preview.get('observation_id') != model_obs or not _identity(model_obs) or not _identity(validation_obs):
                raise ValueError('model_and_validation_observation_identity_required')
            source_stamp = preview.get('model_source_capture_timestamp')
            if not _number(source_stamp) or source_stamp > preview['capture_timestamp']:
                raise ValueError('model_source_timestamp_invalid')
            source = dict(target_id=preview.get('target_id'), model_source_observation_id=model_obs,
                model_source_capture_timestamp=source_stamp, clock_id=clock_id,
                calibration_id=calibration_id, descriptor_sha256=self.descriptor_sha256)
            if not _identity(source['target_id']):
                raise ValueError('preview_target_identity_required')
            source_id = 'learned_sim_source:' + _hash(source)
            candidates = preview.get('candidates')
            if not isinstance(candidates, list) or not 1 <= len(candidates) <= 128 or any(not isinstance(c, dict) for c in candidates):
                raise ValueError('preview_candidates_lost')
            ids = [c.get('candidate_id') for c in candidates]
            if any(not _identity(i) for i in ids) or len(set(ids)) != len(ids):
                raise ValueError('preview_candidate_identity_invalid')
            candidate = next((c for c in candidates if c['candidate_id'] == selected_candidate_id), None)
            if candidate is None:
                raise ValueError('selected_preview_candidate_lost')
            observed = candidate.get('observed_hand_collision')
            observed_clear = isinstance(observed, dict) and observed.get('pregrasp_collision_free') is True
            if (candidate.get('geometry_valid') is not True or candidate.get('ik_valid') is not True or
                    candidate.get('collision_valid') is not False or
                    (self.policy == OBSERVED_PREGRASP_REQUIRED and not observed_clear)):
                raise ValueError('selected_preview_geometry_or_hand_precheck_unavailable')
            pose = rigid(candidate.get('base_ee'))
            if not _number(candidate.get('model_score')) or not _identity(candidate.get('source')):
                raise ValueError('selected_preview_model_provenance_invalid')
            identity = dict(candidate_id=selected_candidate_id, base_ee=pose.tolist(),
                            source=candidate['source'], model_source_id=source_id)
            if self._frozen is not None:
                if _hash(identity) != self._frozen['identity_digest']:
                    raise ValueError('frozen_candidate_identity_pose_or_model_source_changed')
                anchor = self._frozen['anchor']
            else:
                anchor = self.frames.get(model_obs)
                if anchor is None:
                    raise ValueError('model_source_frame_was_not_recorded_by_this_simulator')
            validation = self.frames.get(validation_obs)
            if validation is None or abs(validation['timestamp']-preview['capture_timestamp']) > 1e-8:
                raise ValueError('validation_frame_identity_timestamp_not_recorded')
            if abs(anchor['timestamp']-source_stamp) > 1e-8:
                raise ValueError('model_source_timestamp_does_not_match_recorded_frame')
            if anchor['record_sequence'] <= self._invalid_record_barrier:
                raise ValueError('model_source_predates_invalid_sim_observation')
            if anchor['clock_id'] != clock_id or anchor['calibration_id'] != calibration_id:
                raise ValueError('recorded_model_source_context_mismatch')
            if self._frozen is None:
                interval = [r for r in self.frames.values() if anchor['timestamp'] <= r['timestamp'] <= validation['timestamp']]
                if len(interval) < 3 or validation['timestamp']-anchor['timestamp'] < .10:
                    raise ValueError('independent_stationary_sim_observations_insufficient')
                for previous, current in zip(interval, interval[1:]):
                    if current['timestamp']-previous['timestamp'] > .25:
                        raise ValueError('recorded_stationarity_observation_gap')
                # Include observations newer than the still-fresh preview:
                # motion followed by a return must never disappear.
                profiles = [self._geometry_profiles[r['observation_id']] for r in self.frames.values()
                    if r['timestamp'] >= anchor['timestamp']]
                if not _recorded_geometry_matches(self._geometry_profiles[model_obs], profiles):
                    raise ValueError('actual_sim_target_scene_or_base_moved')
            elif validation['record_sequence'] < self._frozen['minimum_validation_sequence']:
                raise ValueError('validation_predates_bound_stationarity_evidence')
            # Once frozen, record_frame checks every subsequent observation
            # against this exact anchor, including gaps, with a failure latch.
            # Rechecking the whole bounded history here adds hundreds of ms
            # but no new evidence. Keep the original proven interval's lower
            # bound; never replace or retime the model anchor.
            self._association(preview.get('observed_centroid_world'), anchor)
            if not prepare:
                return SimProposalRecheck(True, 'bound_preview_still_current', source_id,
                    validation_obs, validation['timestamp'])
            if self._frozen is None:
                self._frozen = {'identity_digest': _hash(identity), 'anchor': deepcopy(anchor),
                                'candidate': deepcopy(candidate), 'source': deepcopy(source), 'model_source_id': source_id,
                                'minimum_validation_sequence': validation['record_sequence']}
            prepared = PreparedSimProposal(True, 'awaiting_independent_current_sim_endpoint_check',
                deepcopy(self._frozen['candidate']), source_id, validation_obs,
                validation['timestamp'], uuid.uuid4().hex)
            self._prepared = prepared
            return deepcopy(prepared)
        except (ValueError, TypeError, KeyError, OverflowError, AttributeError) as exc:
            return self._fail(str(exc), prepare=prepare)

    def authorize(self, prepared, endpoint_result, *, robot_state, scene, now, clock_id):
        """Bind a trusted worker response to its exact current-state request.

        Pass the same state/scene used for validate_sim_candidate. Its response
        must contain matching snapshot hashes and actual IK/FK/collision results.
        Fresh command validation and consumer lease checks remain mandatory.
        """
        try:
            if (self.invalid_reason or not isinstance(prepared, PreparedSimProposal) or not prepared.ready or
                    self._prepared is None or prepared != self._prepared):
                raise ValueError('prepared_selection_missing_changed_or_revoked')
            if not _number(now):
                raise ValueError('current_time_invalid')
            _fresh(prepared.validation_capture_timestamp, self._frozen['source']['clock_id'], now, clock_id, .25)
            _fresh(robot_state.get('acquisition_timestamp'), robot_state.get('clock_id'), now, clock_id, .15)
            _fresh(scene.get('acquisition_timestamp'), scene.get('clock_id'), now, clock_id, .15)
            if (robot_state.get('complete') is not True or not _identity(robot_state.get('source')) or
                    abs(robot_state['acquisition_timestamp']-scene['acquisition_timestamp']) > .02):
                raise ValueError('current_complete_same_state_sim_evidence_required')
            self._require_scene_policy(scene)
            geometry = _scene_geometry(scene)
            _stationary_velocity(scene)
            if not _same_geometry(self._frozen['anchor']['geometry'], geometry) or not _pose_close(
                    self._frozen['anchor']['base'], robot_state.get('base_in_world')):
                raise ValueError('current_sim_scene_or_base_changed_since_model')
            if not isinstance(endpoint_result, dict) or endpoint_result.get('accepted') is not True or (
                    endpoint_result.get('scope') != 'sim_candidate_endpoint_only' or endpoint_result.get('motion_authorized') is not False):
                raise ValueError('separate_sim_endpoint_evidence_required')
            ik, endpoint, evidence = (endpoint_result.get(key, {}) for key in ('ik', 'endpoint', 'state_evidence'))
            if (ik.get('ik_valid') is not True or endpoint.get('collision_valid') is not True or
                    evidence.get('scope') != 'sim_candidate_endpoint_only' or evidence.get('reason') != 'evaluated' or
                    evidence.get('path_check_requested') is not False or evidence.get('live_execution_authorized') is not False or
                    evidence.get('current_state_precheck', {}).get('collision_valid') is not True or
                    evidence.get('urdf_vs_observed_frames', {}).get('valid') is not True):
                raise ValueError('endpoint_ik_full_scene_or_current_state_rejected')
            expected = _hash({'robot_state': robot_state, 'scene': scene})
            if evidence.get('snapshot_sha256') != expected or _hash({'robot_state': evidence.get('robot_state'), 'scene': evidence.get('scene')}) != expected:
                raise ValueError('endpoint_evidence_does_not_match_current_request')
            if not _pose_close(ik.get('right_target_base_ee'), prepared.candidate['base_ee'], 1e-8, 1e-8):
                raise ValueError('endpoint_evidence_checked_different_candidate')
            tcp = np.eye(4); tcp[0, 3] = .05
            left = np.linalg.inv(rigid(robot_state['base_in_world'])) @ rigid(
                robot_state['observed_body_world_poses']['left_wrist_yaw_link']) @ tcp
            if not _pose_close(ik.get('left_target_base_ee'), left, 1e-8, 1e-8):
                raise ValueError('endpoint_evidence_did_not_hold_measured_left_TCP')
            if not np.allclose(arm_vector(ik.get('q_initial')), [robot_state['joint_positions'][j] for j in ARM_JOINT_NAMES], atol=1e-8, rtol=0):
                raise ValueError('endpoint_ik_initial_state_mismatch')
            solution = arm_vector(ik.get('q_solution'))
            final = ik.get('final', {})
            no_op = (ik.get('reason') == 'already_at_target' and ik.get('solver_invoked') is False and
                ik.get('validation_method') == 'independent_current_pose_FK' and
                final.get('solver_status') == 'not_run_already_at_target' and final.get('solver_success') is None and
                np.array_equal(solution, arm_vector(ik.get('q_initial'))))
            cached_fk = (ik.get('reason') == 'cached_endpoint_FK' and
                ik.get('validation_method') == 'independent_cached_endpoint_full_q43_FK' and
                ik.get('scope') == 'numerical_candidate_endpoint_full_q43_FK_only' and
                ik.get('solver_invoked') is False and ik.get('original_solver_invoked') is False and
                ik.get('cache_hit') is True and ik.get('cache_revalidated_this_call') is True and
                ik.get('cache_authority') is False and ik.get('motion_authorized') is False and
                ik.get('fixed_nonarm_joints_unchanged') is True and ik.get('limits_valid') is True and
                final.get('solver_status') == 'validated_cached_endpoint_FK' and
                final.get('solver_success') is None)
            # A reused numerical seed is not a successful optimization call.
            # Fresh full-model FK may prove it instead; endpoint collisions and
            # the exact current request binding above remain mandatory.
            if (not no_op and not cached_fk and final.get('solver_success') is not True) or final.get('limits_valid') is not True:
                raise ValueError('endpoint_solver_or_joint_limits_unverified')
            limits = (('position_error_m', ALREADY_AT_TARGET_POSITION_TOLERANCE_M if no_op else .01),
                      ('orientation_error_rad', ALREADY_AT_TARGET_ORIENTATION_TOLERANCE_RAD if no_op else np.deg2rad(10)))
            for side in ('left', 'right'):
                for suffix, limit in limits:
                    value = final.get(side+'_'+suffix)
                    if not _number(value) or not 0 <= value <= limit:
                        raise ValueError('endpoint_independent_fk_tolerance_failed')
            frozen = self._frozen
            candidate = GraspCandidate(prepared.candidate['candidate_id'], rigid(prepared.candidate['base_ee']),
                frozen['candidate']['model_score'], True, True, True,
                frozen['candidate']['source']+'|'+prepared.model_source_id)
            batch = ProposalBatch(target_id=frozen['source']['target_id'],
                observation_id=prepared.validation_observation_id,
                capture_timestamp=prepared.validation_capture_timestamp, clock_id=clock_id,
                calibration_id=frozen['source']['calibration_id'], candidates=(candidate,),
                motion_static=True, valid=True, model_source_id=prepared.model_source_id,
                model_source_capture_timestamp=frozen['source']['model_source_capture_timestamp'],
                model_source_clock_id=frozen['source']['clock_id'])
            audit = dict(scope='known_static_sim_scene_endpoint_only', model_source_id=prepared.model_source_id,
                preview_collision_policy=self.policy,
                model_source_observation_id=frozen['source']['model_source_observation_id'],
                model_source_capture_timestamp=frozen['source']['model_source_capture_timestamp'],
                model_source_clock_id=frozen['source']['clock_id'],
                validation_observation_id=prepared.validation_observation_id,
                validation_capture_timestamp=prepared.validation_capture_timestamp,
                endpoint_snapshot_sha256=expected, endpoint_state_timestamp=robot_state['acquisition_timestamp'],
                endpoint_scene_timestamp=scene['acquisition_timestamp'], candidate_id=candidate.candidate_id,
                motion_authorized=False, hardware_authorized=False, requires_actual_command_guard=True,
                object_association='observed_centroid_inside_explicit_known_sim_target_bounds')
            return SimProposalAuthorization(True, 'selected_endpoint_independently_checked_for_simulation', batch, audit)
        except (ValueError, TypeError, KeyError, OverflowError, AttributeError) as exc:
            self._prepared = None
            return SimProposalAuthorization(False, str(exc), None,
                {'motion_authorized': False, 'hardware_authorized': False, 'requires_actual_command_guard': True})
