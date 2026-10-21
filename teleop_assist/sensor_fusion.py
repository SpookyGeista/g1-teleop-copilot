"""Conservative camera/LiDAR measurement association, independent of scene truth.

Wrist masks refine a visible surface point. They do not establish a complete
CAD pose or a grasp. LiDAR only supports depth where the head mask and measured
depth agree; absent rays never invalidate RGB-D.
"""
from __future__ import annotations

from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation

from .perception import _calibration, backproject


def project(points_world, frame):
    k, t = _calibration(frame['intrinsics'], frame['T_world_camera'])
    points = (np.asarray(points_world)-t[:3,3])@t[:3,:3]
    z = points[:,2]
    uv = np.column_stack((points[:,0]*k['fx']/np.maximum(z,1e-9)+k['cx'],
                          points[:,1]*k['fy']/np.maximum(z,1e-9)+k['cy']))
    return uv, z


def target_corners_world(target, head):
    box = target['box_camera']
    corners = np.array(list(product((-.5,.5),repeat=3)))*np.asarray(box['size'])
    corners = corners@Rotation.from_quat(box.get('quaternion_xyzw',[0,0,0,1])).as_matrix().T+box['center']
    t = np.asarray(head['T_world_camera'])
    return corners@t[:3,:3].T+t[:3,3]


def wrist_proposals(targets, head, wrist, max_skew=.025):
    if abs(float(head['timestamp'])-float(wrist['timestamp'])) > max_skew:
        return []
    h,w = wrist['depth'].shape
    proposals = []
    for target in targets:
        if target.get('stale') or not target.get('observed',True):
            continue
        uv,z = project(target_corners_world(target,head),wrist)
        if not np.isfinite(uv).all() or min(z)<.08 or max(z)>1.5:
            continue
        lo,hi = uv.min(axis=0)-4,uv.max(axis=0)+4
        # Fully visible bounds avoid pretending a partial surface is its centre.
        if lo[0]<1 or lo[1]<1 or hi[0]>=w-1 or hi[1]>=h-1 or min(hi-lo)<7:
            continue
        proposals.append({'id':target['id'],'bbox_px':[*lo.tolist(),*hi.tolist()]})
    return proposals


def refine_with_wrist(targets, head, wrist, proposals, segmentations):
    """Refine only unambiguous, depth-consistent target points from a wrist view."""
    by_id = {t['id']:t for t in targets}
    k,t = _calibration(wrist['intrinsics'],wrist['T_world_camera'])
    records=[]
    for proposal,segmentation in zip(proposals,segmentations):
        target=by_id[proposal['id']]
        mask=np.asarray(segmentation['mask'],dtype=bool)
        if mask.shape!=wrist['depth'].shape or float(segmentation.get('confidence',0))<.6:
            records.append({'id':target['id'],'status':'mask_rejected'});continue
        valid=mask&np.isfinite(wrist['depth'])&(wrist['depth']>.08)&(wrist['depth']<1.5)
        ys,xs=np.nonzero(valid)
        if len(xs)<25:
            records.append({'id':target['id'],'status':'insufficient_depth'});continue
        points=backproject(np.column_stack((xs,ys)),wrist['depth'][ys,xs],k)@t[:3,:3].T+t[:3,3]
        old=np.asarray(target['center_world'])
        corners=target_corners_world(target,head)
        lo,hi=corners.min(axis=0)-.018,corners.max(axis=0)+.018
        inside=np.all((points>=lo)&(points<=hi),axis=1)
        if inside.mean()<.85:
            records.append({'id':target['id'],'status':'foreground_or_background_conflict'});continue
        points=points[inside]
        measured=np.median(points,axis=0)
        delta=measured-old
        # Reject view-dependent centroid shifts rather than pulling towards them.
        if np.linalg.norm(delta)>.025:
            records.append({'id':target['id'],'status':'anchor_disagreement'});continue
        neighbors=[np.linalg.norm(np.asarray(other['center_world'])-measured)
                   for other in targets if other['id']!=target['id']]
        if neighbors and min(neighbors)<np.linalg.norm(delta)+.025:
            records.append({'id':target['id'],'status':'ambiguous_neighbor'});continue
        correction=delta*.35
        if np.linalg.norm(correction)>.006:
            correction*=.006/np.linalg.norm(correction)
        target.setdefault('head_center_world',old.tolist())
        target['center_world']=(old+correction).tolist()
        target.setdefault('measurement_sources',['head_rgbd']).append(wrist.get('camera_id','wrist_rgbd'))
        target['wrist_refinement']={'source':wrist.get('camera_id','wrist'),
            'timestamp':float(wrist['timestamp']),'sample_count':len(points),
            'anchor_difference_m':float(np.linalg.norm(delta)),
            'applied_m':float(np.linalg.norm(correction)),
            'scope':'bounded_visible_surface_point_not_full_6d_pose'}
        records.append({'id':target['id'],'status':'surface_refined',**target['wrist_refinement']})
    return records


def lidar_support(targets, head, masks, scan, max_skew=.025):
    if scan is None:
        return {'status':'not_connected','accepted_points':0}
    if abs(float(scan['timestamp'])-float(head['timestamp']))>max_skew:
        return {'status':'stale_scan','accepted_points':0}
    points=np.asarray(scan.get('points_lidar',[]),float)
    transform=np.asarray(scan['T_world_lidar'],float)
    if points.ndim!=2 or points.shape[1]!=3 or transform.shape!=(4,4):
        raise ValueError('LiDAR points must be Nx3 with calibrated T_world_lidar')
    # Time-aligned snapshots only: asynchronous moving scans need deskew first.
    if scan.get('deskew_required',False):
        return {'status':'deskew_required','accepted_points':0}
    points=points[np.isfinite(points).all(axis=1)]@transform[:3,:3].T+transform[:3,3]
    uv,z=project(points,head)
    h,w=head['depth'].shape
    visible=(z>.08)&np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    uv,z,points=uv[visible].astype(int),z[visible],points[visible]
    total=0
    for target in targets:
        mask=masks.get(target['id'])
        if mask is None:
            continue
        onmask=np.asarray(mask,dtype=bool)[uv[:,1],uv[:,0]]
        depth=head['depth'][uv[:,1],uv[:,0]]
        match=onmask&np.isfinite(depth)&(abs(z-depth)<.015)
        count=int(match.sum());total+=count
        target['lidar_support']={'points':count,'timestamp':float(scan['timestamp']),
                                 'scope':'depth_consistency_only'}
        if count:
            target.setdefault('measurement_sources',['head_rgbd']).append('lidar')
    return {'status':'aligned','accepted_points':total,'input_points':len(points),
            'source':scan.get('source','external_lidar'),'changes_target_identity':False}
