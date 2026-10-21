"""Local CAD/RGBD observability and conservative operating uncertainty.

The point-to-plane Jacobian captures geometric degeneracy. Its white-noise
least-squares spread is only one heuristic term, never a calibrated confidence
probability. Voxel reduction, a noise floor, retained systematic-fit correction
and explicit pose floors prevent dense duplicate pixels from claiming certainty.
"""
import numpy as np
from .cad_surface import point_triangle_correspondences


def point_plane_operating_bounds(points,correspondences,normals,*,voxel_m=.005,noise_floor_m=.001):
    points=np.asarray(points,float);q=np.asarray(correspondences,float);n=np.asarray(normals,float)
    failure={'observable':False,'reason':'insufficient_surface_correspondences',
             'uncertainty_source':'local_point_plane_operating_bound_not_calibrated_probability'}
    if points.shape!=q.shape or n.shape!=q.shape or points.ndim!=2 or points.shape[1]!=3 or len(points)<30:return failure
    if not np.isfinite(points).all() or not np.isfinite(q).all() or not np.isfinite(n).all():return failure
    # Keep spatially separate samples per face-normal direction. Increasing
    # image resolution or repeating the same samples cannot multiply evidence.
    keys=np.column_stack((np.floor(q/voxel_m),np.round(n,4)))
    _,take=np.unique(keys,axis=0,return_index=True);points,q,n=points[take],q[take],n[take]
    if len(points)<30:return {**failure,'independent_voxel_samples':len(points)}
    residual=np.sum((points-q)*n,axis=1)
    J=np.column_stack((n,np.cross(q,n)))
    # Normalize translation/rotation units for rank and condition diagnostics.
    scale=max(.01,float(np.sqrt(np.mean(np.sum((q-q.mean(axis=0))**2,axis=1)))))
    scaled=J.copy();scaled[:,3:]/=scale
    singular=np.linalg.svd(scaled,compute_uv=False)
    rank=int(np.count_nonzero(singular>singular[0]*1e-4))
    condition=float(singular[0]/max(1e-15,singular[-1]))
    diagnostic={**failure,'independent_voxel_samples':len(points),'voxel_m':voxel_m,
        'scaled_rank':rank,'scaled_condition_number':condition,'scaled_singular_values':singular.tolist(),
        'normalization_radius_m':scale,'noise_floor_m':noise_floor_m}
    if rank<6 or condition>1e3:
        return {**diagnostic,'reason':'unobservable_or_ill_conditioned_CAD_surface'}
    correction=np.linalg.lstsq(J,residual,rcond=None)[0]
    remaining=residual-J@correction
    scale_residual=max(noise_floor_m,float(1.4826*np.median(np.abs(remaining-np.median(remaining)))))
    # Inverse full 6D information retains translation/rotation coupling. Never
    # use the unconditioned rotation block as if translation were known exactly.
    covariance=np.linalg.inv(J.T@J)*scale_residual**2
    translation_spread=3*float(np.sqrt(max(0,np.linalg.eigvalsh(covariance[:3,:3]).max())))
    rotation_spread=3*float(np.sqrt(max(0,np.linalg.eigvalsh(covariance[3:,3:]).max())))
    translation_bias=float(np.linalg.norm(correction[:3]));rotation_bias=float(np.linalg.norm(correction[3:]))
    diagnostic.update(local_fit_correction_translation_m=translation_bias,
        local_fit_correction_rotation_rad=rotation_bias,local_fit_correction_twist=correction.tolist(),
        residual_scale_m=scale_residual,translation_spread_3x_m=translation_spread,
        rotation_spread_3x_rad=rotation_spread,
        pose_position_sigma_m=max(.005,translation_bias+translation_spread),
        pose_orientation_sigma_rad=max(.10,rotation_bias+rotation_spread),
        position_floor_m=.005,orientation_floor_rad=.10,
        uses_independent_white_noise_approximation=True,probabilistically_calibrated=False)
    if translation_bias>.020 or rotation_bias>.25:
        return {**diagnostic,'reason':'large_unapplied_local_pose_correction'}
    return {**diagnostic,'observable':True,'reason':'full_rank_current_CAD_surface'}


def observed_pose_bounds(depth,mask,K,T_camera_object,triangles,*,voxel_m=.005,noise_floor_m=.001):
    yy,xx=np.nonzero(np.asarray(mask,bool)&np.isfinite(depth)&(depth>.05))
    if len(xx)<80:return {'observable':False,'reason':'insufficient_observed_mask_depth'}
    take=np.linspace(0,len(xx)-1,min(1536,len(xx)),dtype=int);xx=xx[take];yy=yy[take];z=depth[yy,xx]
    points=np.column_stack(((xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z))
    pose=np.asarray(T_camera_object);local=(points-pose[:3,3])@pose[:3,:3]
    q,distance,face=point_triangle_correspondences(local,triangles)
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    consistent=distance<=.012
    if consistent.mean()<.7:return {'observable':False,'reason':'insufficient_current_CAD_surface_support',
        'surface_support_fraction':float(consistent.mean())}
    result=point_plane_operating_bounds(local[consistent],q[consistent],normal[face[consistent]],
        voxel_m=voxel_m,noise_floor_m=noise_floor_m)
    result.update(sampled_mask_points=len(xx),surface_support_fraction=float(consistent.mean()))
    return result
