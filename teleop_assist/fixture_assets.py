"""Local CAD assets for known test objects, without simulator world poses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def drawer_handle_components():
    """Drawer front then U-handle parts, all in the drawer rigid-body frame."""
    return [
        {'shape': 'box', 'center': [0., 0., 0.], 'size': [.024, .30, .18]},
        {'shape': 'box', 'center': [-.065, 0., 0.], 'size': [.022, .14, .022]},
        {'shape': 'box', 'center': [-.033, -.059, 0.], 'size': [.065, .022, .022]},
        {'shape': 'box', 'center': [-.033, .059, 0.], 'size': [.065, .022, .022]},
    ]


def pull_handle_grasp_asset(grasp_templates=None):
    """Mask-compatible U-handle CAD, retaining the parent drawer's origin."""
    return {'id': 'PullHandleGrasp', 'kind': 'drawer_handle', 'parent_asset_id': 'PullHandle',
            'object_frame': 'Same frame as PullHandle drawer root; origin is drawer front centre',
            'T_parent_object': np.eye(4).tolist(),
            'components': drawer_handle_components()[1:], 'articulated': True,
            'actions': ['approach', 'grasp', 'pull', 'push'],
            'contact_point_object_m': [-.076, 0., 0.],
            'approach_direction_object': [1., 0., 0.],
            'mechanical_joint': {'type': 'prismatic', 'axis_object': [1., 0., 0.], 'limits_m': [-.16, 0.]},
            'grasp_templates': list(grasp_templates or []),
            'mask_scope': 'U-handle bar and two stands only; excludes drawer front plate'}


def _component_mesh(component):
    center = np.asarray(component.get('center', [0., 0., 0.]), dtype=float)
    size = np.asarray(component['size'], dtype=float)
    if component['shape'] == 'box':
        vertices = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5)
                             for z in (-.5, .5)]) * size + center
        faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
                 (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
                 (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    elif component['shape'] == 'cylinder':
        # Cylinders in this fixture have their symmetry axis along object X.
        segments = 64
        vertices = [[x, size[1] / 2 * np.cos(2*np.pi*i/segments),
                     size[2] / 2 * np.sin(2*np.pi*i/segments)]
                    for x in (-size[0]/2, size[0]/2) for i in range(segments)]
        vertices.extend([[-size[0]/2, 0., 0.], [size[0]/2, 0., 0.]])
        vertices = np.asarray(vertices) + center
        faces = []
        for i in range(segments):
            j = (i+1) % segments
            faces.extend([(i, j, segments+j), (i, segments+j, segments+i),
                          (2*segments, j, i), (2*segments+1, segments+i, segments+j)])
    else:
        raise ValueError('Unsupported CAD primitive')
    return vertices, faces


def write_cad_assets(objects, output):
    """Write deterministic meter-scale triangle OBJ files and a known catalog."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    catalog = {'schema': 'punktnull.known_fixture_assets.v1', 'units': 'meters',
               'pose_source': 'No world poses: CAD-local geometry and mechanical priors only',
               'objects': []}
    for obj in objects:
        path = output / (obj['id'] + '.obj')
        rows, offset = ['# CAD-local geometry; meters; not a measured world pose'], 0
        for component in obj['components']:
            vertices, faces = _component_mesh(component)
            rows.extend('v ' + ' '.join(f'{float(v):.9f}' for v in vertex) for vertex in vertices)
            rows.extend('f ' + ' '.join(str(i+offset+1) for i in face) for face in faces)
            offset += len(vertices)
        path.write_text('\n'.join(rows) + '\n')
        item = {key: value for key, value in obj.items() if key != 'components'}
        item.update({'cad_mesh': str(path.resolve()), 'mesh_units': 'meters',
                     'cad_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'pose_estimation_required': True})
        catalog['objects'].append(item)
    (output/'catalog.json').write_text(json.dumps(catalog, indent=2))
    return catalog
