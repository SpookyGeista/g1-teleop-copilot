"""Observed surface consistency for a frozen, operator-selected grasp batch.

The gate can reject measured changes; it cannot certify unseen object motion.
It never advances a capture timestamp without another independent observation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .observations import GeometryObservation


@dataclass(frozen=True)
class StaticConfig:
    max_age_s: float = .25
    max_gap_s: float = .25
    min_duration_s: float = .10
    min_observations: int = 3
    centroid_limit_m: float = .010
    surface_p95_limit_m: float = .012
    base_position_limit_m: float = .001
    base_rotation_limit_rad: float = .005


class ObservedStationarity:
    def __init__(self, anchor: GeometryObservation, config=None):
        self.config = config or StaticConfig()
        points, _ = anchor.point_clouds(max_object_points=2000)
        self.anchor = anchor
        self.anchor_points = points.copy()
        self.tree = cKDTree(points)
        self.last_timestamp = anchor.capture_timestamp
        self.last_observation = anchor.observation_id
        self.seen_observations = {anchor.observation_id}
        self.count = 1
        self.validation_started_at = anchor.capture_timestamp
        self.invalid_reason = None
        self.failure_metrics = {}

    def recover_current(self, current, *, now, clock_id):
        """Explicit simulation experiment: compare a new view to the OLD anchor.

        A visual gap never refreshes a grasp. Recovery needs fresh measured
        geometry at the original location, then a new multi-frame validation
        window. Model/anchor acquisition timestamps are not changed.
        """
        if self.invalid_reason not in (None,'observation_sequence_gap','selected_object_not_observed',
                'camera_stream_expired','camera_clock_unknown_or_stale'):
            return False
        from dataclasses import replace
        probe=ObservedStationarity(self.anchor,replace(self.config,max_gap_s=float('inf')))
        report=probe.observe(current,now=now,clock_id=clock_id)
        if (probe.invalid_reason is not None or current.capture_timestamp<=self.last_timestamp
                or current.observation_id in self.seen_observations):
            return False
        self.invalid_reason=None;self.count=1
        self.last_timestamp=current.capture_timestamp;self.last_observation=current.observation_id
        self.seen_observations.add(current.observation_id)
        self.validation_started_at=current.capture_timestamp
        return True

    def invalidate(self, reason):
        self.invalid_reason = reason
        return {'motion_static': False, 'reason': reason, 'observations': self.count,
                **self.failure_metrics}

    def observe(self, current: GeometryObservation, *, now, clock_id):
        if self.invalid_reason:
            return self.invalidate(self.invalid_reason)
        c = self.config
        if (clock_id != current.clock_id or current.clock_id != self.anchor.clock_id
                or current.calibration_id != self.anchor.calibration_id
                or current.target_id != self.anchor.target_id):
            return self.invalidate('identity_clock_or_calibration_changed')
        age = float(now) - current.capture_timestamp
        if not np.isfinite(age) or not -.01 <= age <= c.max_age_s:
            return self.invalidate('observed_target_stale')
        gap = current.capture_timestamp - self.last_timestamp
        if gap < 0 or gap > c.max_gap_s:
            return self.invalidate('observation_sequence_gap')
        if current.observation_id == self.last_observation:
            if gap != 0:
                return self.invalidate('observation_id_reused')
            return {'motion_static': self.count >= c.min_observations
                    and current.capture_timestamp-self.validation_started_at >= c.min_duration_s,
                    'reason': 'same_observation_no_new_evidence', 'observations': self.count}
        if gap <= 0:
            return self.invalidate('nonincreasing_acquisition_time')
        if current.observation_id in self.seen_observations:
            return self.invalidate('observation_id_reused')
        A, B = np.asarray(self.anchor.T_world_base), np.asarray(current.T_world_base)
        cosine = (np.trace(A[:3, :3].T @ B[:3, :3])-1)/2
        if (np.linalg.norm(A[:3, 3]-B[:3, 3]) > c.base_position_limit_m
                or np.arccos(np.clip(cosine, -1, 1)) > c.base_rotation_limit_rad):
            return self.invalidate('robot_base_moved')
        points, _ = current.point_clouds(max_object_points=2000)
        centroid = float(np.linalg.norm(np.median(points, axis=0)-np.median(self.anchor_points, axis=0)))
        distances = np.r_[self.tree.query(points, k=1)[0], cKDTree(points).query(self.anchor_points, k=1)[0]]
        p95 = float(np.quantile(distances, .95))
        if centroid > c.centroid_limit_m or p95 > c.surface_p95_limit_m:
            self.failure_metrics = {'centroid_change_m': centroid, 'surface_p95_m': p95,
                'centroid_limit_m': c.centroid_limit_m, 'surface_p95_limit_m': c.surface_p95_limit_m,
                'model_source_capture_timestamp': self.anchor.capture_timestamp,
                'validation_capture_timestamp': current.capture_timestamp,
                'validation_observation_id': current.observation_id,
                'scope': 'visible_depth_surface_relative_to_frozen_anchor_only'}
            return self.invalidate('observed_target_surface_changed')
        self.count += 1
        self.last_timestamp = current.capture_timestamp
        self.last_observation = current.observation_id
        self.seen_observations.add(current.observation_id)
        enough = self.count >= c.min_observations and current.capture_timestamp-self.validation_started_at >= c.min_duration_s
        return {'motion_static': enough, 'reason': 'observed_surface_stable' if enough else 'collecting_stationary_observations',
                'observations': self.count, 'centroid_change_m': centroid, 'surface_p95_m': p95,
                'model_source_capture_timestamp': self.anchor.capture_timestamp,
                'validation_capture_timestamp': current.capture_timestamp,
                'scope': 'visible_depth_surface_relative_to_frozen_anchor_only'}
