import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.foundation_pose import apply_grasp_template, readiness, restrict_mask_to_observed_cad,measured_visible_cad_mask,rendered_sensor_consistency


def test_declared_template_uses_observed_frame_transform():
    pose=np.eye(4)
    pose[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    pose[:3,3]=[1,2,3]
    template={'id':'test','tool_point_object_m':[.1,0,-.05],
              'wrist_quaternion_object_xyzw':[0,0,0,1]}
    out=apply_grasp_template(pose,template)
    assert np.allclose(out['position_world'],[1,2.1,2.95])
    assert np.allclose(Rotation.from_quat(out['quaternion_world_xyzw']).as_matrix(),pose[:3,:3])


def test_invalid_template_or_transform_rejected():
    template={'id':'test','tool_point_object_m':[.1,0,-.05],
              'wrist_quaternion_object_xyzw':[0,0,0,2]}
    with pytest.raises(ValueError):
        apply_grasp_template(np.eye(4),template)


def test_readiness_never_claims_actual_inference(tmp_path):
    status=readiness(tmp_path)
    assert not status['ready_for_import']
    assert not status['inference_verified']


def test_cad_prior_removes_neighbor_and_wrong_depth_without_inventing_pixels():
    rendered=np.zeros((40,50),np.float32);rendered[10:25,10:25]=.5
    mask=rendered>0;mask[20:30,24:40]=True
    depth=np.full(mask.shape,.5,np.float32);depth[12:17,12:17]=.46
    filtered,diagnostics=restrict_mask_to_observed_cad(mask,depth,rendered)
    assert np.all(~filtered|mask)
    assert not filtered[:,29:].any()
    assert not filtered[12:17,12:17].any()
    assert filtered[18:24,12:24].all()
    assert diagnostics['added_pixels']==0
    assert diagnostics['removed_mask_pixels']>100


def test_cad_prior_without_projected_surface_is_fail_closed():
    mask=np.ones((20,20),bool)
    filtered,_=restrict_mask_to_observed_cad(mask,np.ones(mask.shape),np.zeros(mask.shape))
    assert not filtered.any()


def test_current_CAD_depth_segmentation_never_fills_missing_or_occluded_pixels():
    predicted=np.ones((20,20),np.float32)*.5;depth=predicted.copy()
    depth[:5]=np.nan;depth[5:10]=0.;depth[10:15]=.40
    mask,diagnostic=measured_visible_cad_mask(depth,predicted)
    assert not mask[:15].any() and mask[15:].all()
    assert diagnostic['current_visible_fraction']==.25
    assert not diagnostic['fills_missing_depth']
    assert not measured_visible_cad_mask(np.full_like(depth,.4),predicted)[0].any()


def occluded_surface():
    predicted=np.zeros((64,64),np.float32);predicted[20:44,20:44]=1.
    depth=np.full_like(predicted,2.);depth[predicted>0]=.9
    mask=np.zeros(predicted.shape,bool);mask[20:27,20:44]=True;depth[mask]=1.
    return depth,mask,predicted


def test_tracking_visibility_uses_measured_foreground_and_retains_raw_iou():
    depth,mask,predicted=occluded_surface()
    okay,quality,overlap=rendered_sensor_consistency(depth,mask,predicted,allow_measured_occlusion=True)
    assert okay and quality['raw_silhouette_iou']<.3 and quality['silhouette_iou']==1.
    assert quality['observed_CAD_surface_pixels']==168
    assert np.array_equal(overlap,mask)
    # A fresh registered prior is mandatory: registration uses full CAD.
    assert not rendered_sensor_consistency(depth,mask,predicted,allow_measured_occlusion=False)[0]


def test_unknown_depth_and_small_residual_cannot_excuse_missing_silhouette():
    depth,mask,predicted=occluded_surface();missing=(predicted>0)&~mask
    for value in (0.,np.nan,np.inf,.99,1.1):
        changed=depth.copy();changed[missing]=value
        okay,quality,_=rendered_sensor_consistency(changed,mask,predicted,allow_measured_occlusion=True)
        assert not okay and quality['foreground_occlusion_pixels']==0
        assert quality['silhouette_iou']==quality['raw_silhouette_iou']


def test_wrong_pose_full_occlusion_and_tiny_visible_patch_are_withheld():
    depth,mask,predicted=occluded_surface()
    wrong=np.where(predicted>0,predicted+.05,0)
    okay,quality,_=rendered_sensor_consistency(depth,mask,wrong,allow_measured_occlusion=True)
    assert not okay and not quality['minimum_observed_extent_passed']
    fully=depth.copy();fully[predicted>0]=.9
    assert not rendered_sensor_consistency(fully,mask,predicted,allow_measured_occlusion=True)[0]
    tiny=mask.copy();tiny[24:]=False  # 96 valid pixels but less than 20% of CAD.
    depth[mask&~tiny]=.9
    okay,quality,_=rendered_sensor_consistency(depth,tiny,predicted,allow_measured_occlusion=True)
    assert not okay and quality['observed_CAD_surface_pixels']==96
    assert quality['silhouette_iou']==1. and not quality['minimum_observed_extent_passed']


def test_visibility_consistency_does_not_replace_full_6D_observability():
    from teleop_assist.pose_observability import observed_pose_bounds
    depth,mask,predicted=occluded_surface()
    okay,_,overlap=rendered_sensor_consistency(depth,mask,predicted,allow_measured_occlusion=True)
    assert okay
    K=np.array([[100,0,32],[0,100,32],[0,0,1]],float)
    triangles=np.array([[[-.12,-.12,1],[.12,-.12,1],[.12,.12,1]],
                        [[-.12,-.12,1],[.12,.12,1],[-.12,.12,1]]])
    bounds=observed_pose_bounds(depth,overlap,K,np.eye(4),triangles)
    assert not bounds['observable'] and bounds.get('scaled_rank',0)<6
