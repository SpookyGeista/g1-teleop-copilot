"""Actual optional FoundationPose CAD registration in the isolated pose process.

Input: observed RGB-D, a neural/explicit mask, calibrated camera, known CAD.
Output: object pose plus rendered-depth/silhouette consistency diagnostics.
No pose is returned as usable unless the current sensor consistency gates pass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

from .model_registry import PROJECT_ROOT, sha256
from .perception import _calibration
from .pose_observability import observed_pose_bounds

FOUNDATIONPOSE_REVISION = 'a1b694b83e633c2cb6115b9063d940a687759392'
FOUNDATIONPOSE_WEIGHTS = ('2023-10-28-18-33-37', '2024-01-11-20-02-45')


@dataclass(frozen=True)
class FoundationPoseConfig:
    source_root: str = str(PROJECT_ROOT/'third_party/FoundationPose')
    registration_iterations: int = 5
    tracking_iterations: int = 2
    min_mask_pixels: int = 80
    min_silhouette_iou: float = .45
    min_depth_support: float = .70
    max_median_depth_error_m: float = .010
    max_p90_depth_error_m: float = .020
    prior_max_age_s: float = .5
    prior_mask_margin_px: float = 3.
    prior_surface_tolerance_m: float = .012
    observability_voxel_m: float = .005
    depth_noise_floor_m: float = .001
    foreground_occlusion_tolerance_m: float = .012
    min_observed_cad_fraction: float = .20
    # This handles an explicitly known CAD symmetry/mounting prior, not a
    # guessed scene orientation. Disable for objects without this symmetry.
    resolve_x_half_turn_with_world_up: bool = False


def restrict_mask_to_observed_cad(mask, depth, prior_rendered_depth, margin_px=3., tolerance_m=.012):
    """Intersect a sensor mask with a nearby, previously measured CAD surface.

    No CAD pixels are added to the observation. The prior render must already
    be transported into the current calibrated camera; temporal validity is
    checked by the caller. Nearest-surface lookup is bounded to the CAD ROI.
    """
    mask=np.asarray(mask,dtype=bool);depth=np.asarray(depth)
    rendered=np.asarray(prior_rendered_depth)
    if mask.shape!=depth.shape or rendered.shape!=depth.shape:
        raise ValueError('Mask, sensor depth and CAD depth must have equal shapes')
    silhouette=np.isfinite(rendered)&(rendered>.001)
    filtered=np.zeros_like(mask)
    yy,xx=np.nonzero(silhouette)
    if len(xx):
        pad=int(np.ceil(margin_px))+1
        y0=max(0,int(yy.min())-pad);y1=min(mask.shape[0],int(yy.max())+pad+1)
        x0=max(0,int(xx.min())-pad);x1=min(mask.shape[1],int(xx.max())+pad+1)
        roi=np.s_[y0:y1,x0:x1]
        distance,indices=distance_transform_edt(~silhouette[roi],return_indices=True)
        nearest=rendered[roi][indices[0],indices[1]]
        measured=depth[roi]
        filtered[roi]=(mask[roi] & (distance<=margin_px) & np.isfinite(measured)
                       & (measured>.05) & (np.abs(measured-nearest)<=tolerance_m))
    return filtered,{'raw_mask_pixels':int(mask.sum()),'filtered_mask_pixels':int(filtered.sum()),
                     'removed_mask_pixels':int(mask.sum()-filtered.sum()),
                     'image_margin_px':float(margin_px),'surface_tolerance_m':float(tolerance_m),
                     'added_pixels':0,'source':'previous_accepted_CAD_pose_and_current_measured_depth'}


def measured_visible_cad_mask(depth,prior_rendered_depth,tolerance_m=.012):
    """A CAD-conditioned segmentation of current depth, never a CAD-only mask."""
    z=np.asarray(depth);predicted=np.asarray(prior_rendered_depth)
    if z.shape!=predicted.shape:raise ValueError('Sensor and prior-render shapes must agree')
    projected=np.isfinite(predicted)&(predicted>.001)
    mask=projected&np.isfinite(z)&(z>.05)&(z<3.)&(np.abs(z-predicted)<=tolerance_m)
    return mask,{'source':'current_measured_depth_near_previous_accepted_CAD_surface',
        'current_visible_pixels':int(mask.sum()),'prior_projected_pixels':int(projected.sum()),
        'current_visible_fraction':float(mask.sum()/max(1,projected.sum())),
        'depth_tolerance_m':tolerance_m,'fills_missing_depth':False,'claims_new_neural_segmentation':False}


def rendered_sensor_consistency(depth,mask,predicted,config=None,*,allow_measured_occlusion=False):
    """Compare an observation with visible CAD, retaining raw silhouette IoU.

    Only actual nearer depth explains occlusion. Missing depth and CAD rendered
    in front of the measured scene remain mismatches. The visibility adjustment
    is permitted only by the caller's fresh registered tracking prior; initial
    registration continues to compare the complete rendered silhouette.
    """
    cfg=config or FoundationPoseConfig();z=np.asarray(depth);mask=np.asarray(mask,bool)
    predicted=np.asarray(predicted)
    if z.shape!=mask.shape or z.shape!=predicted.shape:raise ValueError('Equal sensor/mask/render shapes required')
    silhouette=np.isfinite(predicted)&(predicted>.001)
    sensor_valid=np.isfinite(z)&(z>.05)&(z<3.)
    foreground=silhouette&sensor_valid&(z<predicted-cfg.foreground_occlusion_tolerance_m)
    visible=silhouette&~foreground if allow_measured_occlusion else silhouette
    raw_iou=float(np.count_nonzero(silhouette&mask)/max(1,np.count_nonzero(silhouette|mask)))
    iou=float(np.count_nonzero(visible&mask)/max(1,np.count_nonzero(visible|mask)))
    overlap=visible&mask&sensor_valid
    depth_support=float(overlap.sum()/max(1,mask.sum()))
    residual=np.abs(predicted[overlap]-z[overlap])
    median=float(np.median(residual)) if len(residual) else float('inf')
    p90=float(np.quantile(residual,.9)) if len(residual) else float('inf')
    supported_pixels=int(np.count_nonzero(residual<=cfg.prior_surface_tolerance_m))
    observed_fraction=float(supported_pixels/max(1,int(silhouette.sum())))
    extent_ok=(supported_pixels>=cfg.min_mask_pixels
        and (not allow_measured_occlusion or observed_fraction>=cfg.min_observed_cad_fraction))
    accepted=(iou>=cfg.min_silhouette_iou and depth_support>=cfg.min_depth_support and extent_ok
              and median<=cfg.max_median_depth_error_m and p90<=cfg.max_p90_depth_error_m)
    quality={'silhouette_iou':iou,'raw_silhouette_iou':raw_iou,
        'silhouette_iou_definition':'measured_foreground_occlusion_excluded' if allow_measured_occlusion else 'complete_CAD_silhouette',
        'measured_depth_support':depth_support,'median_depth_error_m':median if np.isfinite(median) else None,
        'p90_depth_error_m':p90 if np.isfinite(p90) else None,
        'foreground_occlusion_pixels':int(foreground.sum()),'rendered_CAD_pixels':int(silhouette.sum()),
        'foreground_occlusion_fraction':float(foreground.sum()/max(1,silhouette.sum())),
        'foreground_occlusion_tolerance_m':cfg.foreground_occlusion_tolerance_m,
        'observed_CAD_surface_pixels':supported_pixels,'observed_CAD_surface_fraction':observed_fraction,
        'minimum_observed_extent_passed':bool(extent_ok),'missing_depth_explains_occlusion':False}
    return bool(accepted),quality,overlap


def readiness(source_root=None):
    root = Path(source_root or FoundationPoseConfig().source_root)
    modules = {}
    for name in ('torch','pytorch3d','nvdiffrast','open3d','kornia','omegaconf','warp','transformations'):
        try:
            modules[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            modules[name] = False
    weights = {name: (root/'weights'/name/'model_best.pth').is_file() and
               (root/'weights'/name/'config.yml').is_file() for name in FOUNDATIONPOSE_WEIGHTS}
    cpp = list((root/'mycpp/build').glob('mycpp*.so'))
    return {'ready_for_import':all(modules.values()) and all(weights.values()) and bool(cpp)
            and (root/'estimater.py').is_file(), 'modules':modules, 'weights':weights,
            'native_extension':bool(cpp), 'source_root':str(root),
            'source_revision':FOUNDATIONPOSE_REVISION, 'inference_verified':False}


def apply_grasp_template(T_world_object, template):
    """Transform a declared CAD-local template; never reads scene world poses."""
    _, pose = _calibration(np.eye(3), T_world_object)
    point = np.asarray(template['tool_point_object_m'],dtype=float)
    quaternion = np.asarray(template['wrist_quaternion_object_xyzw'],dtype=float)
    if point.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(point).all() or not np.isfinite(quaternion).all():
        raise ValueError('Finite CAD-local grasp point/quaternion required')
    if not np.isclose(np.linalg.norm(quaternion),1.,atol=1e-4):
        raise ValueError('Unit grasp quaternion required')
    rotation = pose[:3,:3] @ Rotation.from_quat(quaternion).as_matrix()
    return {'position_world':(pose[:3,:3] @ point+pose[:3,3]).tolist(),
            'quaternion_world_xyzw':Rotation.from_matrix(rotation).as_quat().tolist(),
            'template_id':template['id'], 'source':'known_CAD_template_and_observed_FoundationPose',
            'grip_scalar':float(template.get('grip_scalar',.85)),
            'approach_standoff_m':float(template.get('approach_standoff_m',.09)),
            'validated_pull_distance_m':float(template.get('validated_pull_distance_m',0.)),
            'template_validation_scope':template.get('scope','unspecified')}


class FoundationPoseEstimator:
    def __init__(self, mesh_path, config=None, debug_dir=None):
        self.config = FoundationPoseConfig(**config) if isinstance(config,dict) else config or FoundationPoseConfig()
        self.mesh_path = Path(mesh_path).resolve()
        if not self.mesh_path.is_file():
            raise FileNotFoundError(self.mesh_path)
        status = readiness(self.config.source_root)
        if not status['ready_for_import']:
            raise RuntimeError(f'FoundationPose dependencies unavailable: {status}')
        source = Path(self.config.source_root).resolve()
        sys.path.insert(0,str(source))
        import torch
        import trimesh
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        import nvdiffrast.torch as dr
        from Utils import nvdiffrast_render, make_mesh_tensors
        if not torch.cuda.is_available():
            raise RuntimeError('FoundationPose requires a CUDA device')
        self.torch = torch
        torch.set_num_threads(min(8,torch.get_num_threads()))
        self.mesh = trimesh.load(str(self.mesh_path), force='mesh', process=False)
        if not len(self.mesh.vertices) or not np.isfinite(self.mesh.vertices).all():
            raise ValueError('A finite nonempty CAD mesh in metres is required')
        if not .005 < float(np.max(self.mesh.extents)) < 2.:
            raise ValueError('CAD extent outside expected metre-scale workspace')
        self.glctx = dr.RasterizeCudaContext()
        self.render = nvdiffrast_render
        self.mesh_tensors = make_mesh_tensors(self.mesh)
        debug = Path(debug_dir or PROJECT_ROOT/'runs/foundation_pose_runtime')
        debug.mkdir(parents=True,exist_ok=True)
        self.estimator = FoundationPose(model_pts=self.mesh.vertices,model_normals=self.mesh.vertex_normals,
             mesh=self.mesh,scorer=ScorePredictor(),refiner=PoseRefinePredictor(),
             glctx=self.glctx,debug=0,debug_dir=str(debug))
        # Upstream compute_mesh_diameter returns a numpy.float64. In current
        # PyTorch a mixed list containing that scalar creates float64 crop
        # offsets despite the float32 default, conflicting with camera K.
        self.estimator.diameter=float(self.estimator.diameter)
        self._registered = False
        self._last_T_world_camera = None
        self._accepted_prior = None

    def register(self,rgb,depth,intrinsics,mask,T_world_camera=None,grasp_template=None,timestamp=None):
        # Explicit registration resets identity; an old object's CAD prior may
        # never filter a newly selected object's neural mask.
        self._accepted_prior=None
        return self._infer(rgb,depth,intrinsics,mask,T_world_camera,grasp_template,register=True,timestamp=timestamp)

    def track(self,rgb,depth,intrinsics,mask,T_world_camera=None,grasp_template=None,timestamp=None):
        if not self._registered:
            raise RuntimeError('A verified registration is required before CAD tracking')
        return self._infer(rgb,depth,intrinsics,mask,T_world_camera,grasp_template,register=False,timestamp=timestamp)

    def track_visible_depth(self,rgb,depth,intrinsics,T_world_camera,timestamp):
        """Try genuine current-frame CAD tracking after an image-track dropout.

        Requires the same registered identity and a fresh accepted pose. A
        prior silhouette only proposes pixels; current valid depth must agree.
        The ordinary pose gates and local observability still decide acceptance.
        """
        self.last_recovery_mask=None
        prior=self._accepted_prior
        age=float(timestamp)-prior['timestamp'] if prior and prior['timestamp'] is not None else float('inf')
        if not self._registered or not 0<=age<=self.config.prior_max_age_s:
            return {'accepted':False,'reason':'fresh_registered_CAD_identity_required',
                    'method':'FoundationPose','retained_tracking_prior':False}
        k,t=_calibration(intrinsics,T_world_camera)
        K=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]],dtype=np.float32)
        prior_depth=self._render_depth(np.linalg.inv(t)@prior['T_world_object'],K,np.asarray(depth).shape)
        mask,diagnostic=measured_visible_cad_mask(depth,prior_depth,self.config.prior_surface_tolerance_m)
        diagnostic.update(prior_timestamp=prior['timestamp'],prior_age_s=age)
        result=self._infer(rgb,depth,K,mask,t,None,register=False,timestamp=timestamp)
        result['depth_identity_recovery']=diagnostic
        result['recovered_from_current_depth']=bool(result['accepted'])
        if result['accepted']:self.last_recovery_mask=self.last_mask.copy()
        return result

    def _render_depth(self,pose,K,shape):
        pose_tensor=self.torch.as_tensor(pose[None],device='cuda',dtype=self.torch.float32)
        with self.torch.inference_mode():
            _,rendered_depth,_=self.render(K=K,H=shape[0],W=shape[1],ob_in_cams=pose_tensor,
                    context='cuda',glctx=self.glctx,mesh_tensors=self.mesh_tensors)
        return rendered_depth[0].detach().cpu().numpy()

    def _restore_prior(self,usable):
        self._registered=bool(usable)
        if usable:
            self.estimator.pose_last=self._accepted_prior['native_pose'].clone()
            self._last_T_world_camera=self._accepted_prior['T_world_camera'].copy()
        else:
            self._last_T_world_camera=None

    def _infer(self,rgb,depth,intrinsics,mask,T_world_camera,template,register,timestamp):
        image = np.asarray(rgb)
        z = np.asarray(depth,dtype=np.float32)
        mask = np.asarray(mask,dtype=bool)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or z.shape != image.shape[:2] or mask.shape != z.shape:
            raise ValueError('RGB, depth and object mask shapes must agree')
        k,t = _calibration(intrinsics,np.eye(4) if T_world_camera is None else T_world_camera)
        K = np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]],dtype=np.float32)
        started=time.perf_counter()
        prior=self._accepted_prior
        prior_age=(float(timestamp)-prior['timestamp'] if prior and timestamp is not None
                   and prior['timestamp'] is not None else float('inf'))
        usable_prior=not register and 0<=prior_age<=self.config.prior_max_age_s
        raw_mask=mask.copy()
        filtering={'applied':False,'reason':'no_fresh_accepted_CAD_prior'}
        if usable_prior:
            predicted_prior=self._render_depth(np.linalg.inv(t)@prior['T_world_object'],K,z.shape)
            mask,filtering=restrict_mask_to_observed_cad(mask,z,predicted_prior,
                self.config.prior_mask_margin_px,self.config.prior_surface_tolerance_m)
            filtering.update(applied=True,prior_timestamp=prior['timestamp'],prior_age_s=prior_age)
        valid = mask & np.isfinite(z) & (z>.05) & (z<3.)
        if valid.sum()<self.config.min_mask_pixels or valid.sum()/max(1,mask.sum())<self.config.min_depth_support:
            self._restore_prior(usable_prior)
            return {'accepted':False,'reason':'insufficient_current_mask_depth','method':'FoundationPose',
                    'retained_tracking_prior':bool(usable_prior),'mask_prior_filter':filtering}
        z=np.where(np.isfinite(z)&(z>.001),z,0).astype(np.float32)
        with self.torch.inference_mode():
            if register:
                pose=self.estimator.register(K=K,rgb=image,depth=z,ob_mask=mask,
                       iteration=self.config.registration_iterations)
            else:
                # Transport the previous object hypothesis into the current
                # optical frame before refinement, including a head/wrist switch.
                # Camera extrinsics come from calibration+robot FK, not scene truth.
                if self._last_T_world_camera is not None:
                    relative=np.linalg.inv(t)@self._last_T_world_camera
                    self.estimator.pose_last=self.torch.as_tensor(relative,device='cuda',dtype=self.torch.float32)@self.estimator.pose_last
                pose=self.estimator.track_one(rgb=image,depth=z,K=K,
                       iteration=self.config.tracking_iterations)
        # A model can lose its rigid pose (including after long tracking).
        # Reject that measurement and reset identity for a fresh registration;
        # input calibration and worker/protocol failures remain fatal.
        try:
            pose=np.asarray(pose,dtype=float).reshape(4,4)
            _calibration(K,pose)
        except ValueError as error:
            self._accepted_prior=None
            self._restore_prior(False)
            self.estimator.pose_last=None
            return {'accepted':False,'reason':'invalid_model_rigid_transform',
                    'method':'FoundationPose','operation':'register' if register else 'track',
                    'retained_tracking_prior':False,'requires_registration':True,
                    'validation_error':str(error),'seconds':time.perf_counter()-started}
        symmetry_resolved=False
        if self.config.resolve_x_half_turn_with_world_up:
            world_pose=t@pose
            if world_pose[2,2]<0:
                pose=pose@np.diag([1.,-1.,-1.,1.])
                symmetry_resolved=True
        predicted=self._render_depth(pose,K,z.shape)
        silhouette=predicted>.001
        sensor_consistent,quality,overlap=rendered_sensor_consistency(z,mask,predicted,self.config,
            allow_measured_occlusion=usable_prior)
        observability=observed_pose_bounds(z,overlap,K,pose,np.asarray(self.mesh.triangles),
            voxel_m=self.config.observability_voxel_m,noise_floor_m=self.config.depth_noise_floor_m)
        accepted=sensor_consistent and observability.get('observable',False)
        world=t@pose
        if accepted:
            self._registered=True;self._last_T_world_camera=t.copy()
            self._accepted_prior={'T_world_object':world.copy(),'T_world_camera':t.copy(),
                'timestamp':float(timestamp) if timestamp is not None else None,
                'native_pose':self.estimator.pose_last.clone()}
        else:
            self._restore_prior(usable_prior)
        result={'accepted':bool(accepted),'reason':('observed_CAD_pose' if accepted else
                    ('unobservable_current_CAD_geometry' if sensor_consistent else 'rendered_sensor_consistency_failed')),
                'method':'FoundationPose','operation':'register' if register else 'track',
                'source_revision':FOUNDATIONPOSE_REVISION,'mesh_path':str(self.mesh_path),
                'mesh_sha256':sha256(self.mesh_path),'T_camera_object':pose.tolist(),
                'T_world_object':world.tolist(),'quality':quality,'seconds':time.perf_counter()-started,
                'retained_tracking_prior':bool(not accepted and usable_prior),'mask_prior_filter':filtering,
                'raw_mask_quality':{'silhouette_iou':float(np.count_nonzero(silhouette & raw_mask)/max(1,np.count_nonzero(silhouette | raw_mask))),
                    'measured_depth_support':float(np.count_nonzero(silhouette & raw_mask & (z>.001))/max(1,raw_mask.sum()))},
                'pose_observability':observability,
                'config':asdict(self.config),'scene_truth_used':False,
                'symmetry_resolved_with_declared_up_prior':symmetry_resolved,
                'symmetry_prior_enabled':self.config.resolve_x_half_turn_with_world_up,
                'pose_position_sigma_m':observability.get('pose_position_sigma_m'),
                'pose_orientation_sigma_rad':observability.get('pose_orientation_sigma_rad'),
                'uncertainty_source':'local_point_plane_operating_bound_not_calibrated_probability',
                'pose_scope':'known_CAD_registration_not_general_grasp_validation'}
        if accepted and template is not None:
            result['grasp_candidate']=apply_grasp_template(world,template)
        self.last_rendered_depth=predicted
        self.last_mask=mask.copy()
        return result
