"""Measured RGB-D panel components; no simulator, semantic labels or scene truth.

This bounded detector fits a visible backing plane and finds separated raised
parts. Labels describe geometry, not electrical function. Flush screws, touching
parts and partly hidden components are deliberately not promised recognition.
Depth uses OpenCV optical axes (+x right, +y down, +z forward), in metres.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numbers

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PerceptionConfig:
    min_depth_m: float = 0.12
    max_depth_m: float = 3.0
    plane_tolerance_m: float = 0.0035
    min_plane_fraction: float = 0.22
    min_protrusion_m: float = 0.009
    max_protrusion_m: float = 0.10
    split_depth_jump_m: float = .015
    min_component_pixels: int = 24
    min_component_extent_m: float = 0.009
    max_component_extent_m: float = 0.36
    max_component_fraction: float = 0.16
    association_distance_m: float = 0.065
    track_retention_s: float = 0.8
    ransac_samples: int = 2200
    ransac_iterations: int = 96


def _calibration(intrinsics, transform):
    if isinstance(intrinsics, dict):
        k = {n: float(intrinsics[n]) for n in ('fx', 'fy', 'cx', 'cy')}
    else:
        a = np.asarray(intrinsics, dtype=float)
        if a.shape != (3, 3):
            raise ValueError('Intrinsics require fx/fy/cx/cy or a 3x3 matrix')
        k = dict(fx=float(a[0, 0]), fy=float(a[1, 1]), cx=float(a[0, 2]), cy=float(a[1, 2]))
    if not np.isfinite(list(k.values())).all() or min(k['fx'], k['fy']) <= 0:
        raise ValueError('Finite camera intrinsics with positive focal lengths required')
    t = np.asarray(transform, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('T_world_camera must be a finite 4x4 transform')
    r = t[:3, :3]
    if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-4)):
        raise ValueError('T_world_camera must be a proper rigid transform')
    return k, t


def backproject(pixels_xy, z_m, intrinsics):
    """Backproject measured image-plane depth; invalid depth is never filled."""
    p = np.asarray(pixels_xy, dtype=float)
    z = np.asarray(z_m, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or z.shape != (len(p),):
        raise ValueError('Expected Nx2 pixels and N depths')
    if not np.isfinite(p).all() or not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError('Backprojection requires finite pixels and positive depths')
    k, _ = _calibration(intrinsics, np.eye(4))
    return np.column_stack(((p[:, 0] - k['cx']) * z / k['fx'],
                            (p[:, 1] - k['cy']) * z / k['fy'], z))


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / max(union, 1)


class CabinetPerception:
    """Stateful depth-plane component detection and short-term association.

    ``process`` returns current measurements only. Missing tracks are retained
    internally for association, never returned as targets that could command an
    arm. T_world_camera is calibration/robot-state input; scene truth is forbidden.
    """
    def __init__(self, config: PerceptionConfig | None = None):
        self.config = config or PerceptionConfig()
        c = self.config
        if not (0 < c.min_depth_m < c.max_depth_m and 0 < c.min_plane_fraction < 1
                and 0 < c.plane_tolerance_m < c.min_protrusion_m < c.max_protrusion_m
                and 0 < c.min_component_extent_m < c.max_component_extent_m
                and 0 < c.split_depth_jump_m < c.max_protrusion_m
                and c.min_component_pixels >= 4 and c.ransac_samples >= 100
                and c.ransac_iterations >= 8 and c.association_distance_m > 0
                and c.track_retention_s > 0):
            raise ValueError('Invalid perception configuration')
        self._last_frame = None
        self._last_time = None
        self._tracks = {}
        self._next_id = 1

    def reset(self):
        """Explicitly clear identity when a scene or sensor clock restarts."""
        self._last_frame = self._last_time = None
        self._tracks.clear()
        self._next_id = 1

    def _plane(self, z, valid, k, frame_id):
        h, w = z.shape
        # Camera-facing backing must dominate this central region. Peripheral
        # floors/walls are excluded using image coordinates, not scene geometry.
        region = valid.copy()
        border_y, border_x = int(h * .08), int(w * .08)
        if border_y:
            region[:border_y] = False
            region[h-border_y:] = False
        if border_x:
            region[:, :border_x] = False
            region[:, w-border_x:] = False
        ys, xs = np.nonzero(region)
        if len(xs) < 100:
            return None, {'reason': 'insufficient_valid_depth_for_plane'}
        rng = np.random.default_rng(int(frame_id) % (2**32))
        chosen = rng.choice(len(xs), min(self.config.ransac_samples, len(xs)), replace=False)
        points = backproject(np.column_stack((xs[chosen], ys[chosen])), z[ys[chosen], xs[chosen]], k)
        triples = rng.integers(0, len(points), size=(self.config.ransac_iterations, 3))
        a, b, c = points[triples[:, 0]], points[triples[:, 1]], points[triples[:, 2]]
        normals = np.cross(b-a, c-a)
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:, None], 1e-12)
        usable = (lengths > 1e-8) & (abs(normals[:, 2]) > .45)
        normals = normals[usable]
        a = a[usable]
        if not len(normals):
            return None, {'reason': 'no_camera_facing_plane'}
        offsets = -np.einsum('ij,ij->i', normals, a)
        # Explicit arithmetic avoids spawning large BLAS thread pools for the
        # small matrix multiply inside an interactive render loop.
        residual = abs(points[:, 0, None] * normals[None, :, 0]
                       + points[:, 1, None] * normals[None, :, 1]
                       + points[:, 2, None] * normals[None, :, 2] + offsets)
        counts = (residual < self.config.plane_tolerance_m).sum(axis=0)
        best = int(np.argmax(counts))
        inliers = residual[:, best] < self.config.plane_tolerance_m
        fraction = float(np.mean(inliers))
        if fraction < self.config.min_plane_fraction or int(inliers.sum()) < 100:
            return None, {'reason': 'backing_plane_not_observed', 'plane_fraction': fraction}
        support = points[inliers]
        center = np.mean(support, axis=0)
        _, _, v = np.linalg.svd(support-center, full_matrices=False)
        normal = v[-1]
        if normal @ center > 0:
            normal = -normal
        offset = -float(np.median(support @ normal))
        residual_m = np.abs(support @ normal + offset)
        return (normal, offset), dict(plane_fraction=fraction,
            plane_residual_m=float(np.median(residual_m)),
            plane_normal_camera=normal.tolist(), plane_offset_m=offset,
            plane_sample_count=len(points), plane_inlier_count=len(support))

    def _component(self, rgb, depth, labels, label_id, stats, signed, normal, k, t):
        c = self.config
        x, y, w, h, count = (int(v) for v in stats[label_id])
        if count < c.min_component_pixels or count > depth.size * c.max_component_fraction:
            return None, 'component_pixel_extent'
        patch_mask = (labels[y:y+h, x:x+w] == label_id).astype(np.uint8)
        ys, xs = np.nonzero(patch_mask)
        xs, ys = xs+x, ys+y
        measured_protrusion = signed[ys, xs]
        if float(np.quantile(measured_protrusion, .95)) > c.max_protrusion_m:
            return None, 'foreground_outside_panel_part_depth'
        points = backproject(np.column_stack((xs, ys)), depth[ys, xs], k)
        lo, hi = np.percentile(points, [1, 99], axis=0)
        extent = hi-lo
        # Perspective projection of even a slanted front is measured here. Tiny
        # marks and large cabinet frames are excluded, without target templates.
        if min(extent[:2]) < c.min_component_extent_m or max(extent) > c.max_component_extent_m:
            return None, 'component_metric_extent'
        eroded = cv2.erode(patch_mask, np.ones((3, 3), np.uint8), iterations=1)
        ey, ex = np.nonzero(eroded)
        if len(ex) < max(9, c.min_component_pixels // 3):
            return None, 'insufficient_interior_depth'
        ex, ey = ex+x, ey+y
        interior = backproject(np.column_stack((ex, ey)), depth[ey, ex], k)
        front = signed[ey, ex]
        # Select an observed front surface, not the backing or the inferred
        # centre of a whole object. Curved/side surfaces remain in the cuboid.
        front_level = float(np.quantile(front, .75))
        front_mask = front >= front_level - max(.003, c.plane_tolerance_m)
        surface = interior[front_mask]
        if len(surface) < 9:
            return None, 'insufficient_contact_surface'
        surface_median = np.median(surface, axis=0)
        # Choose an actual observed surface sample nearest the robust centre.
        # The arithmetic centre can lie inside a hole or occluded image region.
        squared_distance = np.sum((surface-surface_median)**2, axis=1)
        sample_index = int(np.flatnonzero(squared_distance <= squared_distance.min()+1e-12)[0])
        center = surface[sample_index].copy()
        surface_pixels = np.column_stack((ex[front_mask], ey[front_mask]))
        contact_pixel = surface_pixels[sample_index].astype(int).tolist()
        _, singular, vectors = np.linalg.svd(surface-center, full_matrices=False)
        surface_normal = vectors[-1]
        if surface_normal @ center > 0:
            surface_normal = -surface_normal
        rms = float(np.sqrt(np.mean(((surface-center) @ surface_normal)**2)))
        normal_source = 'local_depth_surface'
        if (len(singular) < 3 or singular[1] < .006 or rms > .004
                or surface_normal @ normal < .55):
            surface_normal = normal.copy()
            normal_source = 'backing_plane_approach_normal'
        center_world = t[:3, :3] @ center + t[:3, 3]
        normal_world = t[:3, :3] @ surface_normal
        contour_list, _ = cv2.findContours(patch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contour_list, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        area = max(float(cv2.contourArea(contour)), 1.)
        circularity = min(1., 4*math.pi*area/max(perimeter*perimeter, 1.))
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 2.0:
            label = 'elongated_component'
        elif circularity > .84 and aspect < 1.4:
            label = 'round_component'
        else:
            label = 'raised_component'
        # RGB actually contributes boundary contrast evidence, without a color
        # code or assumed electrical semantics. Geometry handles lighting changes.
        padded = np.pad(patch_mask, 2)
        ring = cv2.dilate(padded, np.ones((5, 5), np.uint8)) - padded
        py, px = np.nonzero(ring)
        gy, gx = py+y-2, px+x-2
        ok = (gy >= 0) & (gy < rgb.shape[0]) & (gx >= 0) & (gx < rgb.shape[1])
        mean_part = np.median(rgb[ys, xs].astype(float), axis=0)
        contrast = 0.
        if np.any(ok):
            mean_surround = np.median(rgb[gy[ok], gx[ok]].astype(float), axis=0)
            contrast = float(np.linalg.norm(mean_part-mean_surround)/(255*math.sqrt(3)))
        fill = count / max(w*h, 1)
        confidence = float(np.clip(.64 + .12*min(count/180, 1) + .10*min(contrast/.25, 1)
                           + .10*min(fill/.7, 1) - .15*min(rms/.005, 1), .1, .96))
        hull = cv2.convexHull(contour).reshape(-1, 2) + np.array([x, y])
        return dict(label=label, bbox_px=[x, y, x+w, y+h],
            center_camera=center.tolist(), center_world=center_world.tolist(),
            normal_world=normal_world.tolist(),
            box_camera={'center':((lo+hi)/2).tolist(), 'size':np.maximum(extent, .001).tolist(),
                        'quaternion_xyzw':[0., 0., 0., 1.]},
            confidence=confidence, stale=False, observed=True,
            polygon_px=hull.astype(int).tolist(), pixel_area=count,
            surface_residual_m=rms, normal_source=normal_source, surface_pixel_px=contact_pixel,
            rgb_boundary_contrast=contrast, protrusion_m=front_level,
            geometry_scope='visible_depth_surface',
            major_axis_world=(t[:3, :3] @ vectors[0]).tolist()), None

    def _associate(self, detections, timestamp):
        c = self.config
        expired = [i for i, tr in self._tracks.items() if timestamp-tr['timestamp'] > c.track_retention_s]
        for i in expired:
            del self._tracks[i]
        ids = list(self._tracks)
        # A merged visible blob must not silently steal one of two selected IDs.
        # Detect it using previous measured projections, only at a similar view.
        ambiguous = []
        filtered = []
        for det in detections:
            compatible = []
            for i in ids:
                tr = self._tracks[i]
                b = tr['bbox_px']
                cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                d = det['bbox_px']
                close = np.linalg.norm(np.asarray(tr['center_world']) - det['center_world']) < c.association_distance_m*2
                larger = (d[2]-d[0])*(d[3]-d[1]) > 1.25*(b[2]-b[0])*(b[3]-b[1])
                if close and larger and d[0] <= cx <= d[2] and d[1] <= cy <= d[3]:
                    compatible.append(i)
            if len(compatible) > 1:
                ambiguous.append({'track_ids':compatible, 'bbox_px':det['bbox_px'], 'reason':'merged_visible_components'})
            else:
                filtered.append(det)
        detections = filtered
        assigned = {}
        if ids and detections:
            cost = np.full((len(ids), len(detections)), 1e6)
            for row, i in enumerate(ids):
                tr = self._tracks[i]
                for col, det in enumerate(detections):
                    distance = float(np.linalg.norm(np.asarray(tr['center_world']) - det['center_world']))
                    if distance <= c.association_distance_m:
                        cost[row, col] = distance/c.association_distance_m + .2*(1-_iou(tr['bbox_px'], det['bbox_px']))
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if cost[row, col] < 1e5:
                    assigned[int(col)] = ids[int(row)]
        observed_ids = []
        for index, det in enumerate(detections):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = f'component_{self._next_id:03d}'
                self._next_id += 1
            det['id'] = track_id
            self._tracks[track_id] = {**det, 'timestamp':timestamp}
            observed_ids.append(track_id)
        stale = [i for i in self._tracks if i not in observed_ids]
        return detections, dict(stale_track_ids=stale, expired_track_ids=expired,
            ambiguous_components=ambiguous, tracked_count=len(self._tracks))

    def process(self, rgb, depth, intrinsics, T_world_camera, frame_id, timestamp):
        """Return observed targets and JSON-compatible sensor diagnostics.

        Invalid shapes/calibration raise ValueError. Empty depth, missing parts,
        and repeated/stale frames return no targets. There is no pose/scene query.
        """
        rgb, z = np.asarray(rgb), np.asarray(depth)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 8:
            raise ValueError('RGB must be HxWx3 uint8 with nonempty dimensions')
        if z.shape != rgb.shape[:2] or z.dtype.kind not in 'fiu':
            raise ValueError('Metric depth must match RGB image dimensions')
        k, t = _calibration(intrinsics, T_world_camera)
        if isinstance(frame_id, bool) or not isinstance(frame_id, numbers.Integral) or frame_id < 0:
            raise ValueError('Nonnegative integer frame_id required')
        if isinstance(timestamp, bool) or not isinstance(timestamp, numbers.Real) or not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Finite nonnegative acquisition timestamp required')
        frame_id, timestamp = int(frame_id), float(timestamp)
        diagnostics = dict(frame_id=frame_id, timestamp=timestamp,
            method='measured_RGB_D_raised_panel_components', semantic_model=False,
            source='rgb_and_image_plane_depth_only', stale=False,
            constraints='dominant backing plane; separated raised visible parts',
            config=asdict(self.config))
        if self._last_frame is not None and (frame_id <= self._last_frame or timestamp <= self._last_time):
            diagnostics.update(reason='nonadvancing_sensor_frame', stale=True, stale_track_ids=list(self._tracks))
            return {'targets':[], 'diagnostics':diagnostics}
        self._last_frame, self._last_time = frame_id, timestamp
        z = z.astype(np.float64, copy=False)
        valid = np.isfinite(z) & (z > self.config.min_depth_m) & (z < self.config.max_depth_m)
        diagnostics['valid_depth_fraction'] = float(np.mean(valid))
        plane, plane_diag = self._plane(z, valid, k, frame_id)
        diagnostics.update(plane_diag)
        candidates, rejected = [], {}
        if plane is not None:
            normal, offset = plane
            yy, xx = np.indices(z.shape, dtype=float)
            with np.errstate(invalid='ignore'):
                signed = z * (normal[0]*(xx-k['cx'])/k['fx']
                              + normal[1]*(yy-k['cy'])/k['fy'] + normal[2]) + offset
            # Keep foreground connected before applying the depth limit.
            # Clipping an occluding hand at a maximum depth would manufacture
            # small apparently-valid fragments of that same hand.
            foreground = (valid & (signed >= self.config.min_protrusion_m)).astype(np.uint8)
            # No closing or depth filling: those would connect adjacent parts or
            # invent surfaces across a missing-depth hole.
            # Separate surfaces at an observed range discontinuity, for example
            # a hand crossing a knob in the image with air between their depths.
            # This removes boundary pixels only; it neither fills hidden surfaces
            # nor clips near objects into a synthetic permitted-depth slab.
            discontinuity = np.zeros(z.shape, dtype=bool)
            with np.errstate(invalid='ignore'):
                for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    ya = slice(max(0, dy), z.shape[0]+min(0, dy))
                    yb = slice(max(0, -dy), z.shape[0]-max(0, dy))
                    xa = slice(max(0, dx), z.shape[1]+min(0, dx))
                    xb = slice(max(0, -dx), z.shape[1]-max(0, dx))
                    jump = ((foreground[ya, xa] > 0) & (foreground[yb, xb] > 0)
                            & (abs(z[ya, xa]-z[yb, xb]) > self.config.split_depth_jump_m))
                    discontinuity[ya, xa] |= jump
                    discontinuity[yb, xb] |= jump
            foreground[discontinuity] = 0
            diagnostics['depth_discontinuity_pixels_removed'] = int(discontinuity.sum())
            n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
            diagnostics['raw_component_count'] = n-1
            for component_id in range(1, n):
                candidate, reason = self._component(rgb, z, labels, component_id, stats, signed, normal, k, t)
                if candidate is not None:
                    candidate.update(frame_id=frame_id, timestamp=timestamp)
                    candidates.append(candidate)
                else:
                    rejected[reason] = rejected.get(reason, 0)+1
        targets, tracking = self._associate(candidates, timestamp)
        diagnostics.update(tracking)
        diagnostics.update(rejected_components=rejected, detected_count=len(targets))
        if plane is None:
            diagnostics['stale'] = True
        elif not targets:
            diagnostics.setdefault('reason', 'no_separated_raised_components')
        else:
            diagnostics['reason'] = 'observed_components'
        return {'targets':targets, 'diagnostics':diagnostics}
