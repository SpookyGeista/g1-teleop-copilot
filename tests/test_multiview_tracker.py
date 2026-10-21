"""Numerical recovery gates with synthetic masks, not neural inference claims."""
import numpy as np

from teleop_assist.learned_perception import mask_to_target
from teleop_assist.multiview_tracker import LockedTargetWristRecovery


def sample():
    rgb=np.zeros((100,120,3),np.uint8);z=np.ones((100,120),np.float32)
    m=np.zeros(z.shape,bool);m[30:70,40:80]=True
    k={'fx':200.,'fy':200.,'cx':60.,'cy':50.}
    frame={'rgb':rgb,'depth':z,'intrinsics':k,'T_world_camera':np.eye(4),'timestamp':1.,'frame_id':1}
    target,_=mask_to_target(m,z,k,np.eye(4),label='handle',detection_score=.7,mask_score=.9)
    target['id']='chosen_1'
    return frame,target,m


def test_fresh_wrist_can_recover_only_confirmed_existing_identity():
    f,t,m=sample();memory=LockedTargetWristRecovery();memory.observe(f,[t])
    current={**f,'timestamp':1.1,'frame_id':2}
    result=memory.recover(t['id'],current,{'right_wrist':current},lambda rgb,boxes:[{'mask':m,'confidence':.95}])
    assert len(result['targets'])==1
    target=result['targets'][0]
    assert target['id']==t['id'] and not target['head_visible']
    assert target['timestamp']==1.1 and target['observed']
    assert 'surface_pixel_px' not in target  # no fabricated head contact pixel
    assert result['observations'][t['id']]['camera_id']=='right_wrist'


def test_new_identity_and_stale_sensor_never_recovered():
    f,t,m=sample();memory=LockedTargetWristRecovery();memory.observe(f,[t])
    current={**f,'timestamp':1.1,'frame_id':2}
    segment=lambda rgb,boxes:[{'mask':m,'confidence':.95}]
    assert memory.recover('different',current,{'right_wrist':current},segment)['targets']==[]
    assert memory.recover(t['id'],current,{'right_wrist':f},segment)['targets']==[]


def test_occluding_foreground_rejected_even_with_good_mask_score():
    f,t,m=sample();memory=LockedTargetWristRecovery();memory.observe(f,[t])
    current={**f,'timestamp':1.1,'frame_id':2};wrong={**current,'depth':np.full(m.shape,.7,np.float32)}
    result=memory.recover(t['id'],current,{'right_wrist':wrong},lambda rgb,boxes:[{'mask':m,'confidence':.99}])
    assert result['targets']==[]
    assert result['diagnostics']['views']['right_wrist']['reason']=='wrist_mask_conflicts_with_known_target_volume'


def test_point_prompts_only_reuse_current_depth_supported_surface_pixels():
    f,t,m=sample();memory=LockedTargetWristRecovery();memory.observe(f,[t],{t['id']:m})
    prior=memory._memory[t['id']]
    positive,negative=memory._point_prompts(prior,f,np.array([30,20]),np.array([90,80]))
    assert len(positive)>=3
    for x,y in positive:assert m[int(y),int(x)]
    occluded={**f,'depth':np.full(m.shape,.7,np.float32)}
    assert memory._point_prompts(prior,occluded,np.array([30,20]),np.array([90,80]))==([],[])


def test_point_guided_hypothesis_must_pass_original_depth_volume_gate():
    f,t,m=sample();memory=LockedTargetWristRecovery();memory.observe(f,[t],{t['id']:m})
    current={**f,'timestamp':1.1,'frame_id':2};calls=[]
    wrong=np.ones(m.shape,bool)
    def guided(rgb,box,positive,negative):
        calls.append((positive,negative))
        return [{'mask':wrong,'confidence':.99},{'mask':m,'confidence':.95}]
    result=memory.recover(t['id'],current,{'head':current},
        lambda rgb,boxes:[{'mask':wrong,'confidence':.99}],segment_points=guided)
    assert calls and len(result['targets'])==1
    assert np.array_equal(result['observations'][t['id']]['mask'],m)
    assert result['diagnostics']['views']['head']['prompt_mode']=='measured_rgbd_points'


def test_disconnected_neighbor_component_is_removed_without_filling_pixels():
    f,t,m=sample();memory=LockedTargetWristRecovery();memory.observe(f,[t],{t['id']:m})
    merged=m.copy();merged[75:90,45:65]=True
    cleaned,diagnostic=memory.filter_confirmed_mask(t['id'],{**f,'timestamp':1.1},merged)
    assert np.array_equal(cleaned,m)
    assert diagnostic['removed_pixels']==300
    assert not np.any(cleaned&~merged)
