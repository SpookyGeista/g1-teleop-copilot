"""CPU selected-surface measurement; no semantic identity or motion authority.

Uses every valid previous mask pixel, forward reprojection and a nearest-surface
z buffer. It emits only pixels that are actually measured in the current RGBD.
No GFTT, SAM call, filling, dilation, inferred hidden surface or timestamp renewal.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
import numbers

import numpy as np

from .learned_perception import LearnedPerceptionConfig, mask_to_target
from .copilot.observations import rigid
from .perception import _calibration, backproject
from .sensor_fusion import project, target_corners_world

CONTRACT={
    'semantic_ttl_s':1.0,'max_frame_gap_s':.65,'max_current_age_s':.25,
    'min_depth_m':.08,'max_depth_m':1.5,'depth_tolerance_m':.012,
    'minimum_native_support_fraction':.80,'maximum_mean_rgb_error':35.,
    'volume_slack_m':.025,'minimum_volume_fraction':.80,
    'max_anchor_change_m':.040,'neighbour_margin_m':.025,
    'scope':'stationary pregrasp visible-surface continuity only',
    'new_exploratory_metrics':[
        '80% support is accepted z-buffer pixels divided by ALL valid prior-mask pixels, including losses/collisions.',
        'Mean RGB L1<35 is a fixed exploratory metric; it is not equivalent to LK patch photometric error<35.',
        'Neither threshold is an empirically calibrated identity bound.'],
}


def _finite(value):
    return (isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value))


def _name(value):
    return isinstance(value, str) and 0 < len(value) <= 512


def _vector(value, shape, reason):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(reason)
    return array.astype(float, copy=False)


def _intrinsics(value):
    if isinstance(value, Mapping):
        if not all(_finite(value.get(key)) for key in ('fx', 'fy', 'cx', 'cy')):
            raise ValueError('invalid_intrinsics')
        matrix = np.array([[value['fx'], 0., value['cx']],
                           [0., value['fy'], value['cy']], [0., 0., 1.]])
    else:
        matrix = _vector(value, (3, 3), 'invalid_intrinsics')
    if (min(matrix[0, 0], matrix[1, 1]) <= 0 or matrix[0, 1] != 0 or matrix[1, 0] != 0
            or not np.array_equal(matrix[2], [0., 0., 1.])):
        raise ValueError('invalid_intrinsics')
    return matrix


def _previous_geometry(target, frame):
    center = _vector(target['center_world'], (3,), 'invalid_previous_center')
    normal = _vector(target['normal_world'], (3,), 'invalid_previous_normal')
    if abs(np.linalg.norm(normal)-1.) > 1e-5:
        raise ValueError('invalid_previous_normal')
    for name in ('detection_confidence', 'segmentation_confidence', 'pose_confidence'):
        value = target.get(name)
        if not _finite(value) or not 0 <= value <= 1:
            raise ValueError('invalid_previous_quality:'+name)
    sigma = target.get('position_sigma_m')
    if not _finite(sigma) or sigma < 0:
        raise ValueError('invalid_previous_quality:position_sigma_m')
    if not _name(target.get('label')) or ('semantic_phrase' in target and not _name(target['semantic_phrase'])):
        raise ValueError('invalid_previous_label')
    box = target['box_camera']
    if not isinstance(box, Mapping):
        raise ValueError('invalid_previous_box')
    _vector(box['center'], (3,), 'invalid_previous_box_center')
    size = _vector(box['size'], (3,), 'invalid_previous_box_size')
    quaternion = _vector(box['quaternion_xyzw'], (4,), 'invalid_previous_box_quaternion')
    if np.any(size <= 0) or abs(np.linalg.norm(quaternion)-1.) > 1e-5:
        raise ValueError('invalid_previous_box')
    corners = target_corners_world(target, frame)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError('invalid_previous_box_corners')
    return center, normal, corners


def _config(value):
    config = LearnedPerceptionConfig() if value is None else value
    if not isinstance(config, LearnedPerceptionConfig):
        raise ValueError('invalid_geometry_config')
    positive = ('min_depth_m', 'max_depth_m', 'max_mask_fraction',
                'min_valid_depth_fraction', 'max_object_extent_m', 'max_surface_residual_m')
    if (not all(_finite(getattr(config, key)) and getattr(config, key) > 0 for key in positive)
            or config.min_depth_m >= config.max_depth_m or config.max_mask_fraction > 1
            or config.min_valid_depth_fraction > 1 or type(config.min_mask_pixels) is not int
            or config.min_mask_pixels < 1 or not isinstance(config.label_min_extent_m, dict)
            or len(config.label_min_extent_m) > 256
            or any(not _name(k) or not _finite(v) or v < 0 for k, v in config.label_min_extent_m.items())):
        raise ValueError('invalid_geometry_config')
    return config


def _frame(frame):
    if not isinstance(frame, Mapping):
        raise ValueError('invalid_frame')
    if (not isinstance(frame.get('frame_id'), numbers.Integral)
            or isinstance(frame['frame_id'], (bool, np.bool_)) or frame['frame_id']<0
            or not _finite(frame.get('timestamp')) or frame['timestamp']<0):
        raise ValueError('invalid_frame_identity')
    for key in ('clock_id','calibration_id','camera_id'):
        if not _name(frame.get(key)):raise ValueError('missing_'+key)
    if 'source_instance_id' in frame and not _name(frame['source_instance_id']):
        raise ValueError('invalid_source_instance_id')
    rgb=np.asarray(frame['rgb']);depth=np.asarray(frame['depth'])
    if (rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3
            or depth.ndim!=2 or depth.shape!=rgb.shape[:2] or depth.size>4096*2160
            or min(depth.shape) < 1 or depth.dtype.kind not in 'fiu' or depth.dtype.itemsize > 8):
        raise ValueError('invalid_rgbd_shape_or_type')
    _intrinsics(frame['intrinsics'])
    rigid(frame['T_world_camera']);rigid(frame['T_world_base'])
    return rgb,depth


def measure(previous_frame,previous_mask,previous_target,current_frame,semantic_timestamp,
            *,now=None,selected_target_id=None,other_targets=(),geometry_config=None):
    """Return {target, mask, quality, reason}; target=None means withhold.

    ``camera_id`` is an explicit caller-owned channel label, not a discovered
    raw-NPZ camera identity. A source epoch, when supplied, must match exactly;
    absent epochs are not invented. Caller retains its common normal/world-
    motion checks, quality caps, loss latch, frozen stationarity and command
    gates. This function never verifies semantic identity. Default ``now`` is
    real host monotonic time. Offline harness passes the recorded-time evaluation
    clock explicitly; acquisition times themselves are never changed.
    """
    started=time.perf_counter();quality={'method':'measured_rgbd_surface_reprojection',
        'semantic_identity_proven':False,'motion_authorized':False,'scene_truth_used':False}
    def result(reason,target=None,mask=None):
        quality['elapsed_s']=time.perf_counter()-started
        return dict(target=target,mask=mask,quality=quality,reason=reason)
    c=CONTRACT
    try:
        if not isinstance(previous_target, Mapping):return result('invalid_previous_target')
        identity=previous_target.get('id')
        if not _name(identity) or not _name(selected_target_id) or selected_target_id != identity:
            return result('selected_identity_mismatch')
        if not isinstance(other_targets, (tuple, list)) or len(other_targets) > 256:
            return result('invalid_neighbour_evidence')
        geometry_config = _config(geometry_config)
        old_rgb,old_depth=_frame(previous_frame);rgb,depth=_frame(current_frame)
        stamp=current_frame['timestamp'];old_stamp=previous_frame['timestamp']
        checked_at=time.monotonic() if now is None else now
        if not _finite(checked_at) or checked_at < 0:return result('invalid_consumer_time')
        if (any(previous_frame[k]!=current_frame[k] for k in ('camera_id','clock_id','calibration_id'))
                or previous_frame.get('source_instance_id')!=current_frame.get('source_instance_id')):
            return result('camera_clock_calibration_or_source_changed')
        if (rgb.shape!=old_rgb.shape or not np.array_equal(_intrinsics(previous_frame['intrinsics']),
                                                         _intrinsics(current_frame['intrinsics']))):
            return result('image_or_intrinsic_calibration_changed')
        if stamp<=old_stamp or current_frame['frame_id']<=previous_frame['frame_id']:
            return result('nonadvancing_frame')
        dt=stamp-old_stamp
        if dt>c['max_frame_gap_s']:return result('frame_gap_exceeded')
        if not -.01<=checked_at-stamp<=c['max_current_age_s']:return result('current_observation_stale')
        if (not _finite(semantic_timestamp) or not 0<=semantic_timestamp<=old_stamp
                or stamp-semantic_timestamp>c['semantic_ttl_s']
                or checked_at-semantic_timestamp>c['semantic_ttl_s']):
            return result('original_semantic_evidence_expired')
        if (previous_target.get('frame_id')!=previous_frame['frame_id']
                or isinstance(previous_target.get('frame_id'), (bool, np.bool_))
                or not _finite(previous_target.get('timestamp'))
                or not _finite(previous_target.get('semantic_keyframe_timestamp'))
                or previous_target.get('timestamp')!=old_stamp
                or previous_target.get('semantic_keyframe_timestamp')!=semantic_timestamp
                or previous_target.get('stale') or previous_target.get('observed') is not True):
            return result('previous_target_provenance_mismatch')
        old_center, old_normal, corners = _previous_geometry(previous_target, previous_frame)
        original=np.asarray(previous_mask)
        if original.dtype!=bool or original.shape!=old_depth.shape:return result('invalid_previous_mask')
        valid=original&np.isfinite(old_depth)&(old_depth>c['min_depth_m'])&(old_depth<c['max_depth_m'])
        n=int(valid.sum())
        if n<100 or n<.70*max(1,int(original.sum())):return result('insufficient_previous_depth')
        y,x=np.nonzero(valid);old_z=old_depth[y,x]
        k,T=_calibration(previous_frame['intrinsics'],previous_frame['T_world_camera'])
        camera=backproject(np.column_stack((x,y)),old_z,k)
        world=camera@T[:3,:3].T+T[:3,3]
        uv,z=project(world,current_frame);h,w=depth.shape
        visible=np.isfinite(uv).all(1)&np.isfinite(z)&(z>c['min_depth_m'])&(z<c['max_depth_m'])
        visible&=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        indices=np.flatnonzero(visible)
        if not len(indices):return result('no_projected_pixels')
        xy=np.rint(uv[indices]).astype(np.int64)
        good=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        indices=indices[good];xy=xy[good]
        if not len(indices):return result('no_projected_pixels')
        linear=xy[:,1]*w+xy[:,0]
        order=np.lexsort((z[indices],linear))  # pixel primary; nearest projected surface secondary.
        pixel_sorted=linear[order]
        keep=np.r_[True,pixel_sorted[1:]!=pixel_sorted[:-1]]
        winners=order[keep];indices=indices[winners];xy=xy[winners];linear=linear[winners]
        current_z=depth[xy[:,1],xy[:,0]]
        delta_depth=np.abs(current_z-z[indices])
        depth_ok=np.isfinite(current_z)&(current_z>c['min_depth_m'])&(current_z<c['max_depth_m'])&(delta_depth<c['depth_tolerance_m'])
        color_error=np.mean(np.abs(rgb[xy[:,1],xy[:,0]].astype(float)-old_rgb[y[indices],x[indices]].astype(float)),axis=1)
        consistent=depth_ok&(color_error<c['maximum_mean_rgb_error'])
        supported=int(consistent.sum());coverage=supported/n
        quality.update(previous_valid_pixels=n,projected_in_bounds_pixels=int(good.sum()),
            unique_zbuffer_pixels=len(indices),current_depth_supported_pixels=int(depth_ok.sum()),
            current_rgb_and_depth_supported_pixels=supported,native_support_fraction=coverage,
            original_semantic_timestamp=semantic_timestamp,semantic_age_s=stamp-semantic_timestamp,
            previous_frame_id=previous_frame['frame_id'],current_frame_id=current_frame['frame_id'],
            timestamp=stamp,camera_id=current_frame['camera_id'],clock_id=current_frame['clock_id'],
            calibration_id=current_frame['calibration_id'],camera_id_source='caller_owned_channel')
        if 'source_instance_id' in current_frame:
            quality['source_instance_id']=current_frame['source_instance_id']
        if coverage<c['minimum_native_support_fraction']:return result('insufficient_native_surface_support')
        mask=np.zeros(depth.shape,bool);mask.flat[linear[consistent]]=True
        # Geometry is recomputed from CURRENT measured depth at the supported
        # pixels only. Sparse holes remain holes; no old depth or RGB is filled.
        target,reason=mask_to_target(mask,depth,current_frame['intrinsics'],current_frame['T_world_camera'],
            label=previous_target.get('semantic_phrase',previous_target['label']),
            detection_score=previous_target.get('detection_confidence',.3),
            mask_score=previous_target.get('segmentation_confidence',.7),config=geometry_config)
        if target is None:return result('current_geometry_'+str(reason))
        k,T=_calibration(current_frame['intrinsics'],current_frame['T_world_camera'])
        yy,xx=np.nonzero(mask);points=backproject(np.column_stack((xx,yy)),depth[yy,xx],k)@T[:3,:3].T+T[:3,3]
        inside=np.all((points>=corners.min(0)-c['volume_slack_m'])&(points<=corners.max(0)+c['volume_slack_m']),axis=1)
        fraction=float(inside.mean());quality['depth_volume_fraction']=fraction
        if fraction<c['minimum_volume_fraction']:return result('current_surface_outside_previous_volume')
        delta=np.asarray(target['center_world'])-old_center;displacement=float(np.linalg.norm(delta))
        quality['visible_world_displacement_m']=displacement
        if displacement>c['max_anchor_change_m']:return result('previous_anchor_disagreement')
        angle=float(np.degrees(np.arccos(np.clip(np.asarray(target['normal_world'])@old_normal,-1,1))))
        quality['local_normal_change_deg']=angle
        for other in other_targets:
            if not isinstance(other, Mapping) or not _name(other.get('id')):
                return result('invalid_neighbour_evidence')
            if other.get('id')==identity:continue
            center=np.asarray(other.get('center_world'),float)
            if center.shape!=(3,) or not np.isfinite(center).all():return result('invalid_neighbour_evidence')
            if np.linalg.norm(center-np.asarray(target['center_world']))<displacement+c['neighbour_margin_m']:
                return result('ambiguous_neighbour')
        target.update(id=identity,frame_id=current_frame['frame_id'],timestamp=stamp,observed=True,stale=False,
            semantic_keyframe_timestamp=semantic_timestamp,semantic_keyframe_age_s=stamp-semantic_timestamp,
            perception_method='measured_rgbd_surface_reprojection',
            normal_source='supported_current_measured_depth',
            segmentation_confidence_source='previous_SAM_only_not_current_semantic_inference',
            identity_proven=False,motion_authorized=False)
        return result(None,target=target,mask=mask)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,np.linalg.LinAlgError) as exc:
        return result('invalid_surface_input:'+type(exc).__name__+':'+str(exc))
