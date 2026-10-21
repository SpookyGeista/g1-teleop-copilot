"""Separate measured depth discontinuities inside a semantic target mask.

The retained target is the dominant component of an eight-neighbour image
graph whose edges also satisfy a metric 3D distance bound. This uses no known
object pose, table height, category shape or simulator truth. It cannot remove
foreign surfaces physically connected to the target, or repair missing depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .observations import rigid


@dataclass(frozen=True)
class DepthCleanupSettings:
    # Explicit engineering scale, not an Intel depth-noise calibration.
    neighbour_distance_m: float = .008
    max_removed_fraction: float = .15
    max_runner_up_ratio: float = .25
    min_retained_pixels: int = 100
    max_mask_pixels: int = 100_000
    min_depth_m: float = .03
    max_depth_m: float = 3.

    def __post_init__(self):
        if not (np.isfinite(self.neighbour_distance_m) and self.neighbour_distance_m > 0
                and 0 <= self.max_removed_fraction < .5
                and 0 <= self.max_runner_up_ratio < 1
                and 1 <= self.min_retained_pixels <= self.max_mask_pixels
                and 0 < self.min_depth_m < self.max_depth_m < np.inf):
            raise ValueError('invalid depth cleanup settings')


def clean_object_mask(depth, mask, K, T_world_camera, *, settings=None):
    """Return (clean boolean mask, report), preserving image pixel alignment.

    On ambiguity/excessive loss return an EMPTY mask and accepted=False: callers
    must abstain rather than silently switch targets. On success pass the clean
    mask into GeometryObservation. Its complementary scene cloud then includes
    every removed pixel that has valid measured depth.

    The 8 mm default needs camera/task calibration before hardware use. No
    erosion, object resizing, random downsampling or depth-value editing occurs.
    """
    cfg = settings or DepthCleanupSettings()
    d = np.asarray(depth, dtype=float)
    m = np.asarray(mask)
    k = np.asarray(K, dtype=float)
    if d.ndim != 2 or d.shape != m.shape or m.dtype != bool or d.size > 4096 * 2160:
        raise ValueError('matching metric depth image and boolean mask required')
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError('rectified finite pinhole intrinsics required')
    world_camera = rigid(T_world_camera, 'T_world_camera')
    original_count = int(m.sum())
    report = {
        'method': '8_neighbour_image_graph_with_metric_3D_edge_bound',
        'neighbour_distance_m': cfg.neighbour_distance_m,
        'max_removed_fraction': cfg.max_removed_fraction,
        'max_runner_up_ratio': cfg.max_runner_up_ratio,
        'input_mask_pixels': original_count,
        'scene_truth_used': False,
        'removed_valid_pixels_return_to_scene': True,
        'scope': 'observed depth connectivity only; semantic identity is inherited from input mask',
        'limitations': [
            'Connected foreign surfaces cannot be separated by this test.',
            'Occlusion may split one real object; ambiguity deliberately abstains.',
            'The metric neighbourhood needs calibration for real camera noise and range.',
        ],
    }

    def abstain(reason):
        return np.zeros_like(m), dict(report, status='abstained', accepted=False,
                                      reason=reason, returned_mask_pixels=0)

    if original_count < cfg.min_retained_pixels:
        return abstain('insufficient_input_mask_pixels')
    if original_count > cfg.max_mask_pixels:
        return abstain('mask_exceeds_bounded_CPU_budget')
    valid = m & np.isfinite(d) & (d > cfg.min_depth_m) & (d < cfg.max_depth_m)
    y, x = np.nonzero(valid)
    report['valid_mask_pixels'] = len(y)
    report['invalid_depth_mask_pixels'] = original_count - len(y)
    if len(y) < cfg.min_retained_pixels:
        return abstain('insufficient_valid_mask_depth')
    z = d[y, x]
    camera = np.column_stack(((x - k[0, 2]) * z / k[0, 0],
                              (y - k[1, 2]) * z / k[1, 1], z))
    # Rigid transforms preserve distances. Build edges in camera coordinates
    # to avoid subtraction of large world-coordinate offsets.
    lookup = np.full(m.shape, -1, dtype=np.int32)
    lookup[y, x] = np.arange(len(y), dtype=np.int32)
    source_edges, target_edges = [], []
    for dy, dx in [(0, 1), (1, -1), (1, 0), (1, 1)]:
        yy, xx = y + dy, x + dx
        in_image = ((yy >= 0) & (yy < m.shape[0]) & (xx >= 0) & (xx < m.shape[1]))
        source = np.flatnonzero(in_image)
        target = lookup[yy[in_image], xx[in_image]]
        has_depth = target >= 0
        source, target = source[has_depth], target[has_depth]
        metric_edge = np.linalg.norm(camera[source] - camera[target], axis=1) <= cfg.neighbour_distance_m
        source_edges.append(source[metric_edge])
        target_edges.append(target[metric_edge])
    source, target = np.concatenate(source_edges), np.concatenate(target_edges)
    graph = coo_matrix((np.ones(len(source), dtype=np.uint8), (source, target)),
                       shape=(len(camera), len(camera)))
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=count)
    order = np.argsort(-sizes, kind='stable')
    largest = int(sizes[order[0]])
    runner_up = int(sizes[order[1]]) if len(order) > 1 else 0
    removed_fraction = (original_count - largest) / original_count
    report.update(
        component_count=int(count), largest_component_pixels=largest,
        component_sizes_descending=sizes[order[:20]].tolist(),
        runner_up_pixels=runner_up, runner_up_ratio=runner_up / largest,
        proposed_removed_fraction=removed_fraction,
    )
    if runner_up / largest >= cfg.max_runner_up_ratio and runner_up:
        return abstain('multiple_competing_depth_components')
    if removed_fraction > cfg.max_removed_fraction:
        return abstain('excessive_mask_loss')
    if largest < cfg.min_retained_pixels:
        return abstain('insufficient_dominant_component_pixels')
    keep = labels == order[0]
    clean = np.zeros_like(m)
    clean[y[keep], x[keep]] = True
    world = camera[keep] @ world_camera[:3, :3].T + world_camera[:3, 3]
    report.update(
        status='accepted', accepted=True, reason='dominant_observed_depth_component',
        returned_mask_pixels=largest,
        removed_valid_mask_pixels=int(len(y) - largest),
        retained_world_bounds_m=np.stack([world.min(0), world.max(0)]).tolist(),
        retained_world_centroid_m=world.mean(0).tolist(),
    )
    return clean, report
