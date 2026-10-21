import numpy as np
import cv2

from teleop_assist.learned_perception import mask_to_target
from teleop_assist.visual_tracker import RGBDMaskTracker


def frames():
    rng=np.random.default_rng(11)
    rgb=np.zeros((100,120,3),np.uint8)
    rgb[30:70,40:80]=rng.integers(20,220,(40,40,3),dtype=np.uint8)
    mask=np.zeros((100,120),bool);mask[30:70,40:80]=True
    f={'rgb':rgb,'depth':np.ones((100,120),np.float32),
       'intrinsics':{'fx':200.,'fy':200.,'cx':60.,'cy':50.},'T_world_camera':np.eye(4),
       'frame_id':1,'timestamp':1.}
    target,_=mask_to_target(mask,f['depth'],f['intrinsics'],f['T_world_camera'],label='handle',detection_score=.6,mask_score=.9)
    target['id']='learned_007'
    return f,target,mask


def test_translation_keeps_identity_and_remeasures_depth():
    f,t,m=frames();tracker=RGBDMaskTracker();tracker.initialize(f,[t],{t['id']:m})
    g={**f,'rgb':cv2.warpAffine(f['rgb'],np.float32([[1,0,4],[0,1,2]]),(120,100)),
       'frame_id':2,'timestamp':1.1}
    result=tracker.track(g)
    assert len(result['targets'])==1
    new=result['targets'][0]
    assert new['id']==t['id'] and new['frame_id']==2
    assert np.allclose(np.array(new['center_world'])-t['center_world'],[.020,.010,0],atol=.008)
    assert new['semantic_keyframe_age_s']<.11
    assert new['perception_method']=='lk_ransac_sam2_mask_current_rgbd'


def test_missing_texture_omits_target():
    f,t,m=frames();tracker=RGBDMaskTracker();tracker.initialize(f,[t],{t['id']:m})
    result=tracker.track({**f,'rgb':np.zeros_like(f['rgb']),'frame_id':2,'timestamp':1.1})
    assert result['targets']==[] and result['diagnostics']['lost_tracks']


def test_depth_occluder_omits_target():
    f,t,m=frames();tracker=RGBDMaskTracker();tracker.initialize(f,[t],{t['id']:m})
    depth=f['depth'].copy();depth[m]=.7
    result=tracker.track({**f,'depth':depth,'frame_id':2,'timestamp':1.1})
    assert result['targets']==[]


def test_expired_semantics_no_extrapolation():
    f,t,m=frames();tracker=RGBDMaskTracker();tracker.initialize(f,[t],{t['id']:m})
    result=tracker.track({**f,'frame_id':2,'timestamp':2.5})
    assert result['targets']==[] and result['diagnostics']['stale']


def test_no_implicit_initialization():
    f,_,_=frames();tracker=RGBDMaskTracker()
    assert tracker.track(f)['targets']==[]


def cad_evidence(target,timestamp,frame_id,**changes):
    return {'kind':'foundationpose_rgbd','accepted':True,'target_id':target['id'],
            'timestamp':timestamp,'frame_id':frame_id,'camera_id':'head','center_world':target['center_world'],
            'quality':{'silhouette_iou':.9,'measured_depth_support':.95,
                       'median_depth_error_m':.002,'p90_depth_error_m':.004},
            'pose_position_sigma_m':.012,'pose_orientation_sigma_rad':.15,**changes}


def test_fresh_cad_measurement_extends_identity_not_sam_timestamp():
    f,t,m=frames();tracker=RGBDMaskTracker();tracker.initialize(f,[t],{t['id']:m})
    for index in range(1,9):
        frame={**f,'frame_id':index+1,'timestamp':1.+index*.2}
        result=tracker.track(frame)
        assert len(result['targets'])==1
        target=result['targets'][0]
        assert target['semantic_keyframe_timestamp']==1.
        assert tracker.confirm_identity(cad_evidence(target,frame['timestamp'],frame['frame_id']))
    assert result['targets'][0]['semantic_keyframe_age_s']>1.5
    # Without another actual CAD observation the same valid optical image
    # expires after the separate 0.5 second evidence interval.
    result=tracker.track({**f,'frame_id':10,'timestamp':3.11})
    assert result['targets']==[]


def test_cad_evidence_cannot_revive_lost_or_wrong_identity_or_replace_rgbd():
    f,t,m=frames();tracker=RGBDMaskTracker();tracker.initialize(f,[t],{t['id']:m})
    assert not tracker.confirm_identity(cad_evidence(t,1.,1,target_id='other'))
    assert not tracker.confirm_identity(cad_evidence(t,.4,1))
    assert not tracker.confirm_identity(cad_evidence(t,1.,1,pose_orientation_sigma_rad=.5))
    assert tracker.confirm_identity(cad_evidence(t,1.,1))
    depth=f['depth'].copy();depth[m]=.7
    assert tracker.track({**f,'depth':depth,'frame_id':2,'timestamp':1.1})['targets']==[]
    assert not tracker.confirm_identity(cad_evidence(t,1.1,2))
