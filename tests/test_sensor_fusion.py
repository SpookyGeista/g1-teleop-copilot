"""Measurement-gating contracts, deliberately synthetic and without models."""
import copy
import numpy as np
from teleop_assist.sensor_fusion import wrist_proposals, refine_with_wrist, lidar_support
from teleop_assist.task_profiles import annotate_tasks


def fixture():
    frame={'depth':np.ones((100,100),np.float32),'rgb':np.zeros((100,100,3),np.uint8),
           'intrinsics':{'fx':100.,'fy':100.,'cx':50.,'cy':50.},
           'T_world_camera':np.eye(4),'timestamp':1.,'camera_id':'right_wrist'}
    target={'id':'a','label':'handle','center_world':[0,0,1.], 'normal_world':[0,0,-1.],
            'box_camera':{'center':[0,0,1.],'size':[.20,.20,.015]},'confidence':.8}
    mask=np.zeros((100,100),bool);mask[41:60,41:60]=True
    return frame,target,mask


def test_stale_or_cropped_wrist_cannot_refine():
    frame,t,mask=fixture();wrist=copy.deepcopy(frame);wrist['timestamp']=1.04
    assert wrist_proposals([t],frame,wrist)==[]
    wrist['timestamp']=1.;wrist['T_world_camera'][0,3]=1.
    assert wrist_proposals([t],frame,wrist)==[]


def test_wrist_consistent_observed_surface_refines_bounded_point():
    frame,t,mask=fixture();wrist=copy.deepcopy(frame);wrist['depth'][mask]=.99
    proposals=wrist_proposals([t],frame,wrist)
    records=refine_with_wrist([t],frame,wrist,proposals,[{'mask':mask,'confidence':.9}])
    assert records[0]['status']=='surface_refined'
    assert 0<np.linalg.norm(np.array(t['center_world'])-[0,0,1])<=.006
    assert t['id']=='a'


def test_foreground_occlusion_does_not_pull_target_to_robot_hand():
    frame,t,mask=fixture();wrist=copy.deepcopy(frame);wrist['depth'][mask]=.80
    proposals=wrist_proposals([t],frame,wrist)
    records=refine_with_wrist([t],frame,wrist,proposals,[{'mask':mask,'confidence':.95}])
    assert records[0]['status']=='foreground_or_background_conflict'
    assert t['center_world']==[0,0,1.]


def test_lidar_background_and_stale_scan_do_not_support_mask():
    frame,t,mask=fixture()
    scan={'timestamp':1.,'T_world_lidar':np.eye(4),'points_lidar':[[0,0,1.],[0,0,1.4],[.4,0,1.]]}
    result=lidar_support([t],frame,{'a':mask},scan)
    assert result['accepted_points']==1
    assert t['center_world']==[0,0,1.]  # consistency evidence never a centroid move
    scan['timestamp']=.9
    assert lidar_support([t],frame,{'a':mask},scan)['status']=='stale_scan'
    scan['timestamp']=1.;scan['deskew_required']=True
    assert lidar_support([t],frame,{'a':mask},scan)['status']=='deskew_required'


def test_absent_lidar_keeps_rgbd_and_task_catalog_uses_observation():
    frame,t,mask=fixture()
    assert lidar_support([t],frame,{'a':mask},None)['status']=='not_connected'
    result=annotate_tasks([t])[0]
    assert t.get('task_calibration') is None
    actions=result['task_calibration']['actions']
    assert {a['id'] for a in actions}=={'grasp','pull','push'}
    assert all(a['grasp_position_world']==t['center_world'] for a in actions)
    assert result['task_calibration']['mechanism_source'].startswith('declared_fixture')
