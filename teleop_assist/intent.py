"""Preview-only geometric intent evidence from operator motion and image pointing.

This estimates which currently perceived component a motion approaches. It does
not infer repair verbs, tool use, semantics, task completion or permission to move.
The caller must keep manual/confirmed selections latched and require confirmation.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class IntentConfig:
    target_timeout_s: float = .70
    motion_gap_s: float = .45
    min_motion_speed_m_s: float = .012
    full_motion_speed_m_s: float = .12
    max_motion_speed_m_s: float = 2.0
    velocity_filter_s: float = .08
    maximum_distance_m: float = .65
    approach_corridor_m: float = .065
    pointer_sigma_uv: float = .14
    standoff_m: float = .08
    probability_temperature: float = .12
    minimum_probability: float = .60
    minimum_probability_margin: float = .20
    minimum_evidence: float = .30


class IntentEstimator:
    def __init__(self, config: IntentConfig | None = None):
        self.config = config or IntentConfig()
        c = self.config
        numeric = vars(c)
        if not all(math.isfinite(v) for v in numeric.values()):
            raise ValueError('Intent configuration must be finite')
        if (not 0 < c.min_motion_speed_m_s < c.full_motion_speed_m_s < c.max_motion_speed_m_s
                or min(c.target_timeout_s, c.motion_gap_s, c.velocity_filter_s,
                       c.maximum_distance_m, c.approach_corridor_m, c.pointer_sigma_uv,
                       c.probability_temperature) <= 0 or not 0 <= c.standoff_m < .3
                or not .5 < c.minimum_probability <= 1
                or not 0 < c.minimum_probability_margin < 1 or not 0 < c.minimum_evidence <= 1):
            raise ValueError('Invalid geometric intent configuration')
        self._previous_position = None
        self._previous_timestamp = None
        self._velocity = np.zeros(3)
        self.diagnostics = {'reason':'not_initialized', 'semantic_inference':False, 'automatic_confirmation':False}

    def reset(self):
        self._previous_position = self._previous_timestamp = None
        self._velocity = np.zeros(3)
        self.diagnostics = {'reason':'reset', 'semantic_inference':False, 'automatic_confirmation':False}

    @staticmethod
    def _vector(value, length):
        try:
            v = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        return v.copy() if v.shape == (length,) and np.isfinite(v).all() else None

    def _pointer(self, pointer, camera):
        if pointer is None:
            return None, 0., None
        confidence = 1.
        if isinstance(pointer, dict):
            if pointer.get('valid', True) is not True:
                return None, 0., None
            confidence = pointer.get('confidence', 1.)
            pointer = pointer.get('uv')
        p = self._vector(pointer, 2)
        try:
            dimensions = self._vector([camera['width'], camera['height']], 2)
            confidence = float(confidence)
        except (TypeError, KeyError, ValueError):
            return None, 0., None
        if (p is None or np.any(p < 0) or np.any(p > 1)
                or dimensions is None or np.any(dimensions <= 0)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            return None, 0., None
        return p, confidence, dimensions

    def update(self, targets, hand_position_world, timestamp, pointer_uv, camera):
        """Return ranked current IDs with bounded scores and suggestion eligibility.

        ``pointer_uv`` is either normalized [u,v], None, or
        {'uv':[u,v], 'confidence':0..1, 'valid':bool}. Coordinates must already
        refer to the same displayed camera image as ``targets``. Raw eye angles
        must be calibrated upstream. Times are host monotonic acquisition times.
        Invalid/stale operator samples return []; stale targets never appear.
        """
        c = self.config
        self.diagnostics = dict(reason='evaluating', semantic_inference=False,
                                automatic_confirmation=False, method='temporal_geometry_and_image_pointing')
        position = self._vector(hand_position_world, 3)
        try:
            stamp = float(timestamp)
        except (TypeError, ValueError):
            stamp = math.nan
        if position is None or isinstance(timestamp, bool) or not math.isfinite(stamp) or stamp < 0:
            self.reset(); self.diagnostics['reason'] = 'invalid_operator_sample'; return []
        if self._previous_timestamp is not None and stamp <= self._previous_timestamp:
            self.diagnostics['reason'] = 'nonadvancing_operator_time'; return []
        motion_ready = False
        if self._previous_timestamp is not None:
            dt = stamp-self._previous_timestamp
            if dt <= c.motion_gap_s:
                observed_velocity = (position-self._previous_position)/dt
                speed = float(np.linalg.norm(observed_velocity))
                if speed > c.max_motion_speed_m_s:
                    self.reset()
                    self._previous_position, self._previous_timestamp = position.copy(), stamp
                    self.diagnostics['reason'] = 'operator_jump_rejected'; return []
                alpha = 1-math.exp(-dt/c.velocity_filter_s)
                self._velocity += alpha*(observed_velocity-self._velocity)
                motion_ready = True
            else:
                self._velocity[:] = 0
        self._previous_position, self._previous_timestamp = position.copy(), stamp
        speed = float(np.linalg.norm(self._velocity)) if motion_ready else 0.
        motion_strength = float(np.clip((speed-c.min_motion_speed_m_s)
                                     /(c.full_motion_speed_m_s-c.min_motion_speed_m_s), 0, 1))
        direction = self._velocity/max(speed, 1e-12)
        pointer, pointer_confidence, dimensions = self._pointer(pointer_uv, camera)
        # At rest, pointing carries all evidence. Motion can override a mistaken
        # coarse gaze proposal, but never a caller's manual or confirmed latch.
        hand_weight = .88*motion_strength
        pointer_weight = 1-hand_weight if pointer is not None else 0.
        if pointer is None and motion_strength > 0:
            hand_weight = motion_strength
        ranked = []
        if not isinstance(targets, (list, tuple)):
            self.diagnostics['reason'] = 'invalid_targets'; return []
        ignored = 0
        ids = set()
        for target in targets:
            if not isinstance(target, dict):
                ignored += 1; continue
            identity = target.get('id')
            center = self._vector(target.get('center_world'), 3)
            normal = self._vector(target.get('normal_world'), 3)
            try:
                age = stamp-float(target['timestamp'])
                quality = float(target.get('confidence', 0))
            except (TypeError, KeyError, ValueError):
                ignored += 1; continue
            if (not isinstance(identity, str) or not identity or identity in ids or center is None
                    or normal is None or np.linalg.norm(normal) < 1e-6 or target.get('stale', False)
                    or not math.isfinite(age) or age < -.05 or age > c.target_timeout_s
                    or not math.isfinite(quality) or not 0 <= quality <= 1):
                ignored += 1; continue
            ids.add(identity)
            normal /= np.linalg.norm(normal)
            goal = center+c.standoff_m*normal
            delta = goal-position
            distance = float(np.linalg.norm(delta))
            if distance > c.maximum_distance_m:
                ignored += 1; continue
            alignment = float(delta @ direction / max(distance, 1e-12)) if motion_ready else 0.
            along = float(delta @ direction) if motion_ready else 0.
            lateral = float(np.linalg.norm(delta-direction*max(0., along))) if motion_ready else distance
            approach = (max(0., alignment) * math.exp(-.5*(lateral/c.approach_corridor_m)**2)
                        * math.exp(-distance/.8)) if motion_strength > 0 else 0.
            # Only weak proximity evidence at zero distance; stationarity alone
            # must never manufacture a confident intent selection.
            pointed = 0.
            box = self._vector(target.get('bbox_px'), 4)
            if pointer is not None and box is not None and box[2] > box[0] and box[3] > box[1]:
                lo, hi = box[:2]/dimensions, box[2:]/dimensions
                outside = np.maximum(np.maximum(lo-pointer, pointer-hi), 0.)
                center_uv = (lo+hi)/2
                pointed = math.exp(-.5*(np.linalg.norm(outside)/c.pointer_sigma_uv)**2
                                   -.5*(np.linalg.norm(pointer-center_uv)/(c.pointer_sigma_uv*2))**2)
                pointed *= pointer_confidence
            score = float(np.clip((hand_weight*approach+pointer_weight*pointed)*quality, 0, 1))
            ranked.append(dict(id=identity, score=score, probability=0.,
                hand_approach_score=float(approach), pointer_score=float(pointed),
                hand_weight=float(hand_weight), pointer_weight=float(pointer_weight),
                distance_m=distance, approach_alignment=alignment, approach_lateral_error_m=lateral,
                suggestion_eligible=False, automatic_confirmation=False,
                probability_kind='normalized_heuristic_scores_not_calibrated_probability',
                explanation='Geometric approach direction and calibrated image-pointing proximity; no task semantics'))
        ranked.sort(key=lambda row:(-row['score'], row['id']))
        if ranked:
            logits=np.asarray([r['score'] for r in ranked])/c.probability_temperature
            likelihood=np.exp(logits-logits.max()); probabilities=likelihood/likelihood.sum()
            for row, probability in zip(ranked, probabilities):row['probability']=float(probability)
            margin=ranked[0]['probability']-(ranked[1]['probability'] if len(ranked)>1 else 0.)
            eligible=(ranked[0]['score'] >= c.minimum_evidence
                      and ranked[0]['probability'] >= c.minimum_probability
                      and margin >= c.minimum_probability_margin)
            ranked[0]['suggestion_eligible']=bool(eligible)
            ranked[0]['probability_margin']=float(margin)
            reason='confident_preview_suggestion' if eligible else 'ambiguous_or_weak_evidence'
        else:
            reason='no_fresh_nearby_targets'
        self.diagnostics.update(reason=reason, velocity_world_m_s=self._velocity.tolist(),
            speed_m_s=speed, motion_ready=motion_ready, pointer_available=pointer is not None,
            pointer_confidence=pointer_confidence, ignored_targets=ignored, ranked_targets=len(ranked))
        return ranked
