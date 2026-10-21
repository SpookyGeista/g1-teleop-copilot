import numpy as np
from teleop_assist.pose_observability import point_plane_operating_bounds
from teleop_assist.cad_surface import point_triangle_correspondences


def surfaces():
    grid=np.linspace(-.05,.05,12);a,b=np.meshgrid(grid,grid);q=[];n=[]
    for axis in range(3):
        p=np.zeros((a.size,3));p[:,axis]=.025;p[:,(axis+1)%3]=a.ravel();p[:,(axis+2)%3]=b.ravel()
        normal=np.zeros_like(p);normal[:,axis]=1.;q.append(p);n.append(normal)
    return np.concatenate(q),np.concatenate(n)


def test_full_surface_fit_retains_bias_and_conservative_floors():
    q,n=surfaces();translation=np.array([.002,-.001,.003]);rotation=np.array([.015,-.025,.01])
    p=q+translation+np.cross(rotation,q)
    result=point_plane_operating_bounds(p,q,n)
    assert result['observable'] and result['scaled_rank']==6
    assert np.allclose(result['local_fit_correction_twist'],[*translation,*rotation])
    assert result['pose_orientation_sigma_rad']>=.10
    assert result['pose_position_sigma_m']>=np.linalg.norm(translation)
    assert result['probabilistically_calibrated'] is False


def test_plane_is_rejected_instead_of_claiming_rotation_certainty():
    q,n=surfaces();keep=n[:,2]>0
    result=point_plane_operating_bounds(q[keep],q[keep],n[keep])
    assert not result['observable'] and result['scaled_rank']==3


def test_duplicate_pixels_cannot_artificially_reduce_uncertainty():
    q,n=surfaces();p=q+[.001,0,0]
    first=point_plane_operating_bounds(p,q,n)
    duplicated=point_plane_operating_bounds(np.repeat(p,5,axis=0),np.repeat(q,5,axis=0),np.repeat(n,5,axis=0))
    assert first==duplicated


def test_large_unapplied_refinement_does_not_pass_as_well_localized():
    q,n=surfaces();result=point_plane_operating_bounds(q+[.04,0,0],q,n)
    assert not result['observable'] and result['reason']=='large_unapplied_local_pose_correction'


def test_triangle_correspondence_is_on_real_face_or_edge():
    triangles=np.array([[[0,0,0],[1,0,0],[0,1,0]]],float)
    q,distance,face=point_triangle_correspondences(np.array([[.2,.2,.1],[1,1,0],[-1,0,0]]),triangles)
    assert np.allclose(q,[[.2,.2,0],[.5,.5,0],[0,0,0]])
    assert np.allclose(distance,[.1,np.sqrt(.5),1]) and (face==0).all()
