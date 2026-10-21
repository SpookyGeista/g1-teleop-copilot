"""Per-frame neural proposal reconciliation against a current measured CAD pose.

No persistent ID aliases are created. The comparison mask contains only current
depth samples supported by the projected mesh; it is not published as a sensor
observation or used to manufacture a head-camera tracking mask.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import cad_surface
from .perception import _calibration


def _independent_current_depth_tracking(evidence,selected_id):
    """Only an independently measured registered track can outrank a proposal."""
    q=evidence.get('quality',{});prior=evidence.get('mask_prior_filter',{});obs=evidence.get('pose_observability',{})
    return bool(evidence.get('accepted') is True and evidence.get('target_id')==selected_id
        and evidence.get('method')=='FoundationPose' and evidence.get('operation')=='track'
        and evidence.get('recovered_from_current_depth') is True
        and prior.get('applied') is True and 0<=prior.get('prior_age_s',float('inf'))<=.5
        and obs.get('observable') is True and obs.get('scaled_rank')==6
        and q.get('minimum_observed_extent_passed') is True
        and q.get('silhouette_iou',0)>=.45 and q.get('measured_depth_support',0)>=.7
        and 0<=q.get('median_depth_error_m',float('inf'))<=.01
        and 0<=q.get('p90_depth_error_m',float('inf'))<=.02)


def _disproved_secondary_proposal(item,evidence,selected_id):
    """Current overlapping pixels fail association, not the independent CAD ID.

    Missing/invalid sensor evidence and geometrically compatible but ambiguous
    masks do not qualify. The proposal stays visible with its own original ID.
    """
    return bool(_independent_current_depth_tracking(evidence,selected_id)
        and item['current_proposal'] and item['semantic_compatible']
        and item['intersection_pixels']>=80
        and item['surface'].get('accepted') is False
        and item['surface'].get('reason')=='current_depth_not_supported_by_CAD')


def _project_cad_silhouette(frame,evidence):
    """Cheap geometric upper bound on the depth-supported comparison mask."""
    if (not evidence.get('accepted') or int(evidence.get('frame_id',-1))!=int(frame['frame_id'])
            or abs(float(evidence['timestamp'])-float(frame['timestamp']))>.04):
        raise ValueError('Same-frame accepted CAD measurement required')
    catalog=cad_surface.CATALOG
    triangles=cad_surface._triangles(evidence['cad_asset_id'],evidence['mesh_sha256'],str(catalog),catalog.stat().st_mtime_ns)
    k,transform=_calibration(frame['intrinsics'],frame['T_world_camera'])
    _,pose=_calibration(frame['intrinsics'],evidence['T_world_object'])
    camera_object=np.linalg.inv(transform)@pose
    camera_points=triangles@camera_object[:3,:3].T+camera_object[:3,3]
    if np.min(camera_points[...,2])<=.05:raise ValueError('CAD crosses camera near plane')
    uv=np.stack((camera_points[...,0]*k['fx']/camera_points[...,2]+k['cx'],
                 camera_points[...,1]*k['fy']/camera_points[...,2]+k['cy']),axis=-1)
    depth=np.asarray(frame['depth']);silhouette=np.zeros(depth.shape,np.uint8)
    for triangle in uv:cv2.fillConvexPoly(silhouette,np.rint(triangle).astype(np.int32),1)
    return silhouette,triangles,k,transform,pose


def current_cad_comparison_mask(frame,evidence,*,_projection=None):
    """Current depth-supported silhouette, used only for identity comparison."""
    silhouette,triangles,k,transform,pose=(_project_cad_silhouette(frame,evidence)
        if _projection is None else _projection)
    depth=np.asarray(frame['depth'])
    yy,xx=np.nonzero((silhouette>0)&np.isfinite(depth)&(depth>.05)&(depth<3.))
    observed=np.zeros(depth.shape,bool)
    object_camera=np.linalg.inv(pose)@transform
    for start in range(0,len(xx),1024):
        x,y=xx[start:start+1024],yy[start:start+1024];z=depth[y,x]
        points=np.column_stack(((x-k['cx'])*z/k['fx'],(y-k['cy'])*z/k['fy'],z))
        local=points@object_camera[:3,:3].T+object_camera[:3,3]
        support=cad_surface.point_triangle_surface_distance(local,triangles)<=.012
        observed[y[support],x[support]]=True
    return observed


def reconcile_current_proposals(frame,targets,masks,selected_id,evidence,center_world):
    """Return one measured duplicate ID, or ambiguity; never mutate the inputs."""
    diagnostic={'status':'no_duplicate','scope':'this_frame_only','selected_id':selected_id,
                'frame_id':int(frame['frame_id']),'cad_timestamp':evidence.get('timestamp'),'proposals':[]}
    others=[t for t in targets if t['id']!=selected_id]
    if not others:return diagnostic
    try:projection=_project_cad_silhouette(frame,evidence)
    except (KeyError,TypeError,ValueError,OSError) as error:
        diagnostic.update(status='unavailable',reason=str(error));return diagnostic
    silhouette=projection[0]>0
    # Every final comparison pixel is a subset of this projected silhouette.
    # Thus a non-near mask with fewer than 80 projected intersections cannot
    # reach the existing 80 observed intersections, irrespective of depth.
    # Missing/invalid masks and the 4-cm neighbor gate retain their old rules.
    reference=None
    matches=[];conflicts=[];non_vetoing=[]
    for target in others:
        identity=target['id'];near=np.linalg.norm(np.asarray(target['center_world'])-center_world)<.04
        mask=masks.get(identity)
        if mask is None:
            if near:conflicts.append(identity)
            continue
        mask=np.asarray(mask,bool)
        if mask.shape!=silhouette.shape:
            if near:conflicts.append(identity)
            continue
        if not near and np.count_nonzero(mask&silhouette)<80:continue
        if reference is None:
            reference=current_cad_comparison_mask(frame,evidence,_projection=projection)
        intersection=int((mask&reference).sum());union=int((mask|reference).sum())
        overlap=intersection/max(1,union);precision=intersection/max(1,int(mask.sum()))
        # Overlap selects a possible duplicate even when partial views move its
        # centroid away. Proximity alone never establishes identity.
        if intersection<80 and not near:continue
        item={'proposal_id':identity,'intersection_pixels':intersection,'mask_iou':overlap,
              'proposal_supported_fraction':precision,'near_current_surface_center':bool(near)}
        fresh=(int(target.get('frame_id',-1))==int(frame['frame_id'])
               and abs(float(target.get('timestamp',-1))-float(frame['timestamp']))<=1e-6)
        semantic=target.get('semantic_label',target.get('label'))=='handle'
        item['current_proposal']=fresh;item['semantic_compatible']=semantic
        surface=cad_surface.current_cad_surface_support(frame,mask,evidence)
        item['surface']=surface
        matches_pose=fresh and semantic and overlap>=.45 and precision>=.70 and surface['accepted']
        item['matches_current_CAD']=bool(matches_pose);diagnostic['proposals'].append(item)
        if matches_pose:matches.append(target)
        elif _disproved_secondary_proposal(item,evidence,selected_id):
            item['association_status']='current_depth_disproves_association_to_confirmed_CAD'
            non_vetoing.append(identity)
        elif near or intersection>=80:conflicts.append(identity)
    if len(matches)>1 or conflicts:
        diagnostic.update(status='ambiguous',matching_ids=[t['id'] for t in matches],conflicting_ids=conflicts)
    elif len(matches)==1:
        proposal=matches[0]
        diagnostic.update(status='unique_duplicate',proposal_id=proposal['id'],
            proposal_timestamp=proposal['timestamp'],
            proposal_semantic_keyframe_timestamp=proposal.get('semantic_keyframe_timestamp',proposal['timestamp']),
            proposal_detection_confidence=proposal.get('detection_confidence'),
            proposal_segmentation_confidence=proposal.get('segmentation_confidence'))
    if non_vetoing:
        diagnostic['non_vetoing_unassigned_proposal_ids']=non_vetoing
        if diagnostic['status']=='no_duplicate':diagnostic['status']='unassigned_secondary_proposals'
    diagnostic['comparison_depth_evaluated']=reference is not None
    return diagnostic
