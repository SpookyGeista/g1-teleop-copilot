from pathlib import Path
import numpy as np
from teleop_assist.copilot.collision import GraspCollisionFilter
from teleop_assist.copilot.preview import HandPreview


def test_actual_open_hand_projection_has_eight_valid_link_silhouettes():
    geometry = GraspCollisionFilter()
    preview = HandPreview(geometry)
    G = np.eye(4); G[2, 3] = 1.
    frame = {'T_world_camera': np.eye(4), 'intrinsics': np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]])}
    shapes = preview.project(G, frame)
    assert len(shapes) == 8
    assert {p['link'] for p in shapes} == set(geometry.canonical_meshes)
    for shape in shapes:
        uv = np.asarray(shape['uv'])
        assert len(uv) >= 3 and uv.shape[1] == 2 and np.isfinite(uv).all()
        assert 260 < uv[:, 0].min() < uv[:, 0].max() < 380
        assert 210 < uv[:, 1].min() < uv[:, 1].max() < 270
    moved = dict(frame, T_world_camera=G)
    assert preview.project(np.eye(4), moved) == []


def test_world_and_camera_motion_together_do_not_change_projected_hand():
    preview = HandPreview(GraspCollisionFilter())
    G = np.eye(4); G[2, 3] = 1.
    K = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]])
    a = preview.project(G, {'T_world_camera': np.eye(4), 'intrinsics': K})
    shift = np.eye(4); shift[:3, 3] = [.3, -.1, .5]
    b = preview.project(shift@G, {'T_world_camera': shift, 'intrinsics': K})
    for first, second in zip(a, b):
        assert first['link'] == second['link']
        assert np.allclose(first['uv'], second['uv'])
