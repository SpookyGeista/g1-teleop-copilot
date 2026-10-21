"""Camera-projected actual OPEN Dex3 geometry for operator review."""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from scipy.spatial import ConvexHull
from .observations import rigid


class HandPreview:
    def __init__(self, collision_filter):
        self.vertices = {}
        self.cache = OrderedDict()
        for name, mesh in collision_filter.canonical_meshes.items():
            vertices = np.asarray(mesh.vertices)
            self.vertices[name] = vertices[ConvexHull(vertices).vertices].copy()

    def project(self, world_canonical, frame):
        camera_hand = np.linalg.inv(rigid(frame['T_world_camera'])) @ rigid(world_canonical)
        K = np.asarray(frame['intrinsics'], float)
        key = camera_hand.tobytes()+K.tobytes()
        if key in self.cache:
            self.cache.move_to_end(key)
            return deepcopy(self.cache[key])
        shapes = []
        for name, vertices in self.vertices.items():
            camera = vertices @ camera_hand[:3, :3].T + camera_hand[:3, 3]
            if np.any(camera[:, 2] <= .03):
                continue
            uv = camera @ K.T; uv = uv[:, :2]/uv[:, 2:]
            if not np.isfinite(uv).all():
                continue
            silhouette = uv[ConvexHull(uv).vertices]
            shapes.append({'link': name, 'uv': silhouette.tolist(),
                           'mean_depth_m': float(camera[:, 2].mean())})
        shapes = sorted(shapes, key=lambda p: -p['mean_depth_m'])
        self.cache[key] = shapes
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        return deepcopy(shapes)
