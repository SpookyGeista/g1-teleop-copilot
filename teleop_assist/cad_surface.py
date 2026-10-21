"""Current RGBD surface consistency against a previously observed CAD pose.

This is an identity check for multi-surface known objects, never a pose update.
It does not access simulator world poses, create masks, or renew CAD timestamps.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from .perception import _calibration

CATALOG=Path(__file__).resolve().parents[1]/'assets/cad/handle_pose/catalog.json'


@lru_cache(maxsize=8)
def _triangles(asset_id,expected_sha,catalog_path,mtime_ns):
    catalog_file=Path(catalog_path);catalog=json.loads(catalog_file.read_text())
    if catalog.get('units')!='meters':raise ValueError('CAD meter units required')
    assets=[a for a in catalog['objects'] if a['id']==asset_id]
    if len(assets)!=1:raise ValueError('Unambiguous known CAD asset required')
    asset=assets[0]
    if asset.get('mesh_units')!='meters' or asset['cad_sha256']!=expected_sha:
        raise ValueError('Accepted pose and CAD asset differ')
    mesh=Path(asset['cad_mesh'])
    if not mesh.is_absolute():mesh=catalog_file.parent/mesh
    data=mesh.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_sha:raise ValueError('CAD hash mismatch')
    vertices=[];faces=[]
    for line in data.decode().splitlines():
        fields=line.split()
        if fields and fields[0]=='v':vertices.append([float(v) for v in fields[1:4]])
        elif fields and fields[0]=='f':
            if len(fields)!=4:raise ValueError('Triangular CAD faces required')
            faces.append([int(v.split('/')[0])-1 for v in fields[1:]])
    points=np.asarray(vertices,float);indices=np.asarray(faces,int)
    if points.ndim!=2 or indices.ndim!=2 or not np.isfinite(points).all():raise ValueError('Finite CAD mesh required')
    if indices.min()<0 or indices.max()>=len(points):raise ValueError('Invalid CAD indices')
    return points[indices]


def point_triangle_surface_distance(points,triangles):
    """Exact face/edge distance without allocating unused 3D correspondences.

    The correspondence-returning implementation below remains the observability
    path. Identity/visibility checks need only distances; einsum avoids its large
    coordinate arrays and temporary products without approximate nearest points.
    """
    points=np.asarray(points,float);triangles=np.asarray(triangles,float)
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    ab=b-a;ac=c-a;relative=points[:,None,:]-a[None,:,:]
    d00=np.einsum('fi,fi->f',ab,ab);d01=np.einsum('fi,fi->f',ab,ac);d11=np.einsum('fi,fi->f',ac,ac)
    denominator=d00*d11-d01*d01
    if np.any(denominator<=1e-18):raise ValueError('Degenerate CAD triangle')
    d20=np.einsum('nfi,fi->nf',relative,ab);d21=np.einsum('nfi,fi->nf',relative,ac)
    v=(d11*d20-d01*d21)/denominator;w=(d00*d21-d01*d20)/denominator
    normal=np.cross(ab,ac)
    projection=np.einsum('nfi,fi->nf',relative,normal)
    plane=projection**2/np.einsum('fi,fi->f',normal,normal)
    squared=np.where((v>=0)&(w>=0)&(v+w<=1),plane,np.inf)
    for offset,edge in ((None,ab),(ab,c-b),(ac,a-c)):
        delta=relative if offset is None else relative-offset
        dot=np.einsum('nfi,fi->nf',delta,edge);length=np.einsum('fi,fi->f',edge,edge)
        fraction=np.clip(dot/length,0,1)
        edge_squared=np.einsum('nfi,nfi->nf',delta,delta)-2*fraction*dot+fraction**2*length
        squared=np.minimum(squared,edge_squared)
    return np.sqrt(np.maximum(0,np.min(squared,axis=1)))


def point_triangle_correspondences(points,triangles):
    """Return nearest CAD surface points, distances and triangle indices."""
    points=np.asarray(points,float);triangles=np.asarray(triangles,float)
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    ab=b-a;ac=c-a;relative=points[:,None,:]-a[None,:,:]
    d00=np.sum(ab*ab,axis=-1);d01=np.sum(ab*ac,axis=-1);d11=np.sum(ac*ac,axis=-1)
    d20=np.sum(relative*ab,axis=-1);d21=np.sum(relative*ac,axis=-1)
    denominator=d00*d11-d01*d01
    if np.any(denominator<=1e-18):raise ValueError('Degenerate CAD triangle')
    v=(d11*d20-d01*d21)/denominator;w=(d00*d21-d01*d20)/denominator
    normal=np.cross(ab,ac)
    plane=np.sum(relative*normal,axis=-1)**2/np.sum(normal*normal,axis=-1)
    squared=np.where((v>=0)&(w>=0)&(v+w<=1),plane,np.inf)
    closest=points[:,None,:]-np.sum(relative*normal,axis=-1)[:,:,None]*normal/np.sum(normal*normal,axis=-1)[:,None]
    for start,end in ((a,b),(b,c),(c,a)):
        edge=end-start;delta=points[:,None,:]-start[None,:,:]
        fraction=np.clip(np.sum(delta*edge,axis=-1)/np.sum(edge*edge,axis=-1),0,1)
        candidate=start+fraction[:,:,None]*edge
        edge_squared=np.sum((points[:,None,:]-candidate)**2,axis=-1)
        closest=np.where((edge_squared<squared)[:,:,None],candidate,closest)
        squared=np.minimum(squared,edge_squared)
    face=np.argmin(squared,axis=1);index=np.arange(len(points))
    return closest[index,face],np.sqrt(np.maximum(0,squared[index,face])),face


def current_cad_surface_support(frame,mask,evidence,catalog_path=None):
    diagnostic={'accepted':False,'source':'current_depth_against_previous_accepted_CAD_surface',
                'updates_pose_or_timestamp':False}
    try:
        if not evidence or evidence.get('accepted') is not True:raise ValueError('Accepted CAD evidence required')
        age=float(frame['timestamp'])-float(evidence['timestamp'])
        if not np.isfinite(age) or not 0<=age<=.5:raise ValueError('CAD evidence expired or future')
        catalog_file=Path(catalog_path or CATALOG)
        triangles=_triangles(evidence['cad_asset_id'],evidence['mesh_sha256'],str(catalog_file),catalog_file.stat().st_mtime_ns)
        k,camera=_calibration(frame['intrinsics'],frame['T_world_camera'])
        _,pose=_calibration(frame['intrinsics'],evidence['T_world_object'])
        mask=np.asarray(mask,bool);depth=np.asarray(frame['depth'])
        valid=mask&np.isfinite(depth)&(depth>.05)&(depth<3.)
        yy,xx=np.nonzero(valid)
        if len(xx)<80 or len(xx)/max(1,mask.sum())<.7:raise ValueError('Insufficient current mask depth')
        take=np.linspace(0,len(xx)-1,min(1024,len(xx)),dtype=int);xx=xx[take];yy=yy[take];z=depth[yy,xx]
        points=np.column_stack(((xx-k['cx'])*z/k['fx'],(yy-k['cy'])*z/k['fy'],z))
        object_camera=np.linalg.inv(pose)@camera
        local=points@object_camera[:3,:3].T+object_camera[:3,3]
        distance=point_triangle_surface_distance(local,triangles)
        inliers=distance<=.012;support=float(inliers.mean());median=float(np.median(distance))
        p90=float(np.quantile(distance[inliers],.9)) if inliers.any() else None
        accepted=support>=.7 and median<=.010 and p90 is not None and p90<=.010
        diagnostic.update(accepted=bool(accepted),reason='observed_CAD_surface' if accepted else 'current_depth_not_supported_by_CAD',
            cad_asset_id=evidence['cad_asset_id'],cad_timestamp=evidence['timestamp'],cad_age_s=age,
            sampled_current_depth_points=len(distance),surface_support_fraction=support,
            median_surface_distance_m=median,inlier_p90_surface_distance_m=p90,
            surface_tolerance_m=.012,min_surface_support_fraction=.7)
    except (KeyError,ValueError,TypeError,OSError,IndexError) as error:
        diagnostic['reason']=str(error)
    return diagnostic
