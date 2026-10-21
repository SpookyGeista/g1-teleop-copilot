"""Synthetic numerical/identity contracts; actual-model reports are separate."""
import hashlib
import json

import numpy as np
import pytest

from teleop_assist.cad_surface import current_cad_surface_support,point_triangle_surface_distance,point_triangle_correspondences
from teleop_assist.visual_tracker import RGBDMaskTracker
from test_visual_tracker import frames,cad_evidence


@pytest.fixture
def plane_catalog(tmp_path,monkeypatch):
    mesh=tmp_path/'plane.obj'
    mesh.write_text('v -0.2 -0.2 1\nv 0.2 -0.2 1\nv 0.2 0.2 1\nv -0.2 0.2 1\nf 1 2 3\nf 1 3 4\n')
    digest=hashlib.sha256(mesh.read_bytes()).hexdigest()
    catalog=tmp_path/'catalog.json'
    catalog.write_text(json.dumps({'units':'meters','objects':[{'id':'plane','mesh_units':'meters',
        'cad_mesh':mesh.name,'cad_sha256':digest}]}))
    monkeypatch.setattr('teleop_assist.cad_surface.CATALOG',catalog)
    return {'cad_asset_id':'plane','mesh_sha256':digest,'T_world_object':np.eye(4).tolist()}


def test_point_triangle_distance_handles_face_edges_and_vertices():
    triangle=np.array([[[0,0,0],[1,0,0],[0,1,0]]],float)
    points=np.array([[.2,.2,.1],[-1,0,0],[1,1,0],[0,0,0]],float)
    assert np.allclose(point_triangle_surface_distance(points,triangle),[.1,1,np.sqrt(.5),0])


def test_distance_only_kernel_matches_full_correspondences_and_threshold_decisions():
    rng=np.random.default_rng(713)
    triangles=rng.uniform(-.1,.1,(36,3,3));points=rng.uniform(-.25,.25,(1024,3))
    points=np.vstack((points,triangles.reshape(-1,3)))
    original=point_triangle_correspondences(points,triangles)[1]
    optimized=point_triangle_surface_distance(points,triangles)
    assert np.allclose(optimized,original,rtol=1e-11,atol=1e-9)
    assert np.array_equal(original<=.012,optimized<=.012)
    triangle=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    depths=np.array([0.,.012-1e-10,.012,.012+1e-10,.01,.02])
    boundary=np.column_stack((np.full(len(depths),.2),np.full(len(depths),.2),depths))
    assert np.array_equal(point_triangle_surface_distance(boundary,triangle)<=.012,
                          point_triangle_correspondences(boundary,triangle)[1]<=.012)
    with pytest.raises(ValueError,match='Degenerate'):
        point_triangle_surface_distance(points,np.zeros((1,3,3)))


def test_surface_support_requires_fresh_matching_pose_and_actual_depth(plane_catalog):
    f,t,m=frames();e=cad_evidence(t,1.,1,**plane_catalog)
    assert current_cad_surface_support(f,m,e)['accepted']
    assert not current_cad_surface_support({**f,'timestamp':1.501},m,e)['accepted']
    assert not current_cad_surface_support(f,m,{**e,'mesh_sha256':'0'*64})['accepted']
    depth=f['depth'].copy();depth[m]=.97
    assert not current_cad_surface_support({**f,'depth':depth},m,e)['accepted']
    assert e['timestamp']==1.  # surface check never renews CAD evidence


def test_local_normal_switch_requires_current_CAD_surface_not_only_old_quality(plane_catalog):
    f,t,m=frames();t['normal_world']=[1.,0.,0.]
    current={**f,'frame_id':2,'timestamp':1.1}
    tracker=RGBDMaskTracker();tracker.initialize(f,[t],{t['id']:m})
    assert tracker.track(current)['targets']==[]
    tracker.initialize(f,[t],{t['id']:m})
    assert tracker.confirm_identity(cad_evidence(t,1.,1,**plane_catalog))
    result=tracker.track(current)
    assert result['targets'][0]['id']==t['id']
    quality=result['targets'][0]['tracking_quality']
    assert quality['local_normal_change_deg']==90.
    assert quality['normal_change_CAD_surface_validation']['accepted']
    # High old CAD confidence is insufficient when current depth fails CAD.
    tracker.initialize(f,[t],{t['id']:m})
    misplaced=np.eye(4);misplaced[2,3]=.04
    assert tracker.confirm_identity(cad_evidence(t,1.,1,**{**plane_catalog,'T_world_object':misplaced.tolist()}))
    assert tracker.track(current)['targets']==[]


def test_cad_surface_alternative_never_bypasses_other_tracking_gates(plane_catalog):
    f,t,m=frames();t['normal_world']=[1.,0.,0.]
    tracker=RGBDMaskTracker();tracker.initialize(f,[t],{t['id']:m})
    assert tracker.confirm_identity(cad_evidence(t,1.,1,**plane_catalog))
    depth=f['depth'].copy();depth[m]=.7
    result=tracker.track({**f,'depth':depth,'frame_id':2,'timestamp':1.1})
    assert result['targets']==[]
    assert result['diagnostics']['lost_tracks'][t['id']]=='inconsistent_measured_world_motion'
