"""Offline contracts for the packaged Dex3 grasp descriptor and its geometry QA."""
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'teleop_assist/copilot/assets/dex3_right'
SPEC=importlib.util.spec_from_file_location('descriptor_builder',ROOT/'tools/copilot/build_dex3_descriptor.py')
BUILDER=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def config():
    return json.loads((ASSETS/'config.json').read_text())


def test_packaged_meshes_are_real_dex3_assets():
    root=ET.parse(ASSETS/'gripper.urdf').getroot()
    assert len(root.findall("joint[@type='revolute']"))==7
    sources=ROOT/'research/masters_20261003/ai_assistance/dex3_descriptor'
    mesh_names={m.get('filename') for m in root.findall('.//mesh')}
    assert len(mesh_names)==8
    for name in mesh_names:
        assert hashlib.sha256((ASSETS/name).read_bytes()).digest()==hashlib.sha256((sources/name).read_bytes()).digest()


def test_open_mid_closed_joint_targets_respect_local_limits():
    root=ET.parse(ASSETS/'gripper.urdf').getroot();c=config()
    for q in [c['open'],BUILDER.interpolate(c['open'],c['close'],.5),c['close']]:
        for joint in root.findall("joint[@type='revolute']"):
            limit=joint.find('limit');v=q[joint.get('name')]
            assert float(limit.get('lower'))<=v<=float(limit.get('upper'))
    assert c['close']['right_hand_thumb_0_joint']==0.


def test_canonical_axes_origin_and_determinant():
    b=np.array(config()['base_rotation'])
    assert np.allclose(b[:3,:3]@b[:3,:3].T,np.eye(3))
    assert np.isclose(np.linalg.det(b[:3,:3]),1.)
    assert np.allclose(b@np.array([.140,0,0,1]),[0,0,0,1])
    assert np.allclose(b[:3,:3]@[0,1,0],[0,0,1])
    assert np.allclose(b[:3,:3]@[-1,0,0],[1,0,0])


def test_grasp_to_xr_chain_places_same_physical_points():
    # Independent vector identity: wrist points transformed through canonical
    # geometry must agree with composition of the canonical grasp and B.
    b=np.array(config()['base_rotation']);g=np.eye(4)
    g[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];g[:3,3]=[.45,-.18,.32]
    t_ue=np.eye(4);t_ue[:3,3]=[.05,0,0]
    result=g@b@t_ue
    assert np.allclose(result[:3,3],(g@(b@np.array([.05,0,0,1])))[:3])
    p_e=np.array([.013,.004,-.011,1.])
    assert np.allclose(result@p_e,g@(b@(t_ue@p_e)))
    assert np.allclose(result[:3,3],[.45,-.09,.32])
    assert not np.allclose(result,b@g@t_ue)
    # This particular canonical rotation plus origin is an involution; an
    # inverse-B negative control would be ineffective for this descriptor.


@pytest.mark.parametrize('triangle,intersects',[
    ([[-2,0,0],[2,0,0],[0,2,0]],True),
    ([[.1,.1,.1],[.2,.1,.1],[.1,.2,.1]],True),
    ([[2,0,0],[2,1,0],[2,0,1]],False),
    ([[.4,.8,.8],[.8,.4,.8],[.8,.8,.4]],False),
    ([[-.5,-.5,.5],[.5,-.5,.5],[0,.5,.5]],True),
])
def test_triangle_aabb_sat_including_corner_separation(triangle,intersects):
    hits=BUILDER.triangle_box_overlap(np.array([triangle]),[0,0,0],[1,1,1])
    assert bool(len(hits)) is intersects


def test_box_entirely_inside_solid_is_rejected_even_without_surface_crossing():
    cube=trimesh.creation.box(extents=[1,1,1])
    r=BUILDER.check_inner_box({'test':cube},[0,0,0],[.1,.1,.1])
    assert not r['passed']
    assert r['links']['test']['center_inside_mesh']


def test_nonwatertight_surface_disjoint_by_bounds_is_not_falsely_blocked():
    plane=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],process=False)
    r=BUILDER.check_inner_box({'open_surface':plane},[0,0,1],[.1,.1,.1])
    assert r['passed']
    assert r['links']['open_surface']['center_outside_mesh_aabb']


def test_artifact_validation_matches_config_and_collision_checks_pass():
    c=config();v=json.loads((ASSETS/'validation.json').read_text())
    assert hashlib.sha256((ASSETS/'config.json').read_bytes()).hexdigest()==v['config_sha256']
    assert v['validation_schema']=='punktnull.copilot.descriptor_validation.v1'
    assert v['geometry_valid'] is True
    required={'config.json','gripper.urdf','vis_mesh.obj','coll_mesh.obj'}
    assert required.issubset(v['asset_sha256'])
    assert len([k for k in v['asset_sha256'] if k.startswith('meshes/')])==8
    for name,expected in v['asset_sha256'].items():
        assert hashlib.sha256((ASSETS/name).read_bytes()).hexdigest()==expected
    assert v['limits_passed']
    assert all(item['passed'] for item in v['mesh_inner_boxes'].values())
    assert all(item['passed'] for item in v['mujoco_inner_boxes'].values())
    assert v['mujoco_contact_controls']['box_inside_palm_detected']
    assert v['mujoco_contact_controls']['far_box_collision_free']
    assert c['type']=='revolute_3f' and c['symmetric'] is False
    assert 'not_grasp_success_certificate' in c['status']


def test_contact_patches_reference_actual_distal_mesh_triangles():
    root=ET.parse(ASSETS/'gripper.urdf').getroot();meshes=BUILDER.load_meshes(root,ASSETS)
    patches=json.loads((ASSETS/'contact_patches.json').read_text())
    assert set(patches)==set(BUILDER.PAD_LINKS)
    for name,patch in patches.items():
        ids=np.array(patch['triangle_ids']);mesh=meshes[name]
        assert ids.min()>=0 and ids.max()<len(mesh.faces)
        assert patch['area_m2']>0
        expected=np.average(mesh.triangles_center[ids],axis=0,weights=mesh.area_faces[ids])
        assert np.allclose(expected,patch['center_local_m'])
