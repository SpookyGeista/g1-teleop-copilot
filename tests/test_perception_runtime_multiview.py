"""Synthetic runtime/protocol contracts; real inference has separate run artifacts."""
import copy
from concurrent.futures import Future

import numpy as np

from teleop_assist.perception_runtime import AsyncPerception
from teleop_assist.pose_bridge import PoseRuntime
from test_visual_tracker import frames


class NoWorker:
    def close(self):pass
    def estimate(self,*args):raise AssertionError('Unexpected inference in this unit test')


def accepted_source(target,age=0.,orientation_sigma=.12):
    normal=np.asarray(target['normal_world'],float)
    return {'id':target['id'],'timestamp':1.-age,'center_world':np.asarray(target['center_world']),
            'normal_world':normal,'result':{'accepted':True,'T_world_object':np.eye(4).tolist(),
            'quality':{'p90_depth_error_m':.003},'pose_position_sigma_m':.006,
            'pose_orientation_sigma_rad':orientation_sigma}}


def test_pose_is_withheld_after_expiry_and_on_orientation_uncertainty():
    f,t,m=frames();runtime=PoseRuntime(NoWorker(),interval=float('inf'))
    try:
        runtime.accepted=accepted_source(t,age=.71)
        runtime.update(f,[t],{})
        assert not t.get('pose_valid',False)
        runtime.accepted=accepted_source(t,orientation_sigma=.36)
        runtime.update(f,[t],{})
        assert t['pose_valid'] is False
        runtime.accepted=accepted_source(t)
        runtime.update(f,[t],{})
        assert t['pose_valid'] is True
        expired={**f,'timestamp':2.,'frame_id':2}
        runtime.update(expired,[t],{})
        assert t['pose_valid'] is False  # copied target cannot retain old flag
    finally:runtime.close()


def test_pose_source_uses_actual_wrist_frame_and_mask():
    f,t,m=frames();wrist={**f,'camera_id':'right_wrist','rgb':f['rgb'].copy()}
    calls=[]
    class Worker(NoWorker):
        def estimate(self,*args):calls.append(args);return {'accepted':False}
    runtime=PoseRuntime(Worker(),interval=0.)
    try:
        runtime.update(f,[t],{},t['id'],observations={t['id']:{'camera_id':'right_wrist','frame':wrist,'mask':m}})
        if runtime.future is not None:runtime.future.result(timeout=2)
        assert calls[0][0] is wrist
        assert np.array_equal(calls[0][1],m)
        assert runtime.source['camera_id']=='right_wrist'
    finally:runtime.close()


def test_stale_wrist_observation_cannot_start_pose():
    f,t,m=frames();runtime=PoseRuntime(NoWorker(),interval=0.)
    try:
        runtime.update({**f,'timestamp':1.1},[t],{},t['id'],
            observations={t['id']:{'camera_id':'right_wrist','frame':f,'mask':m}})
        assert runtime.future is None
    finally:runtime.close()


def test_current_wrist_retains_confirmed_id_without_head_mask():
    f,t,m=frames();runtime=AsyncPerception(NoWorker())
    try:
        result={'targets':[],'_masks':{},'diagnostics':{},'_camera_observations':{
            'right_wrist':{'frame':f,'targets':[t],'masks':{t['id']:m},'head_targets':{t['id']:copy.deepcopy(t)}}}}
        runtime._initialize(f,result)
        current={**f,'timestamp':1.1,'frame_id':2}
        tracked=runtime.tracker.track(current)
        runtime._current_views({'frames':{'head':current,'right_wrist':current}},tracked,t['id'])
        assert len(tracked['targets'])==1 and tracked['targets'][0]['id']==t['id']
        assert tracked['targets'][0]['head_visible'] is False
        assert t['id'] not in runtime.tracker.last_masks
        assert runtime.observations[t['id']]['camera_id']=='right_wrist'
        assert runtime.observations[t['id']]['frame'] is current
        # A different selected identity cannot inherit this observation.
        result={'targets':[],'diagnostics':{}}
        runtime._current_views({'frames':{'head':current,'right_wrist':current}},result,'different')
        assert not result['targets']
    finally:runtime.close()


def test_missing_semantic_proposal_does_not_erase_fresh_confirmed_head_flow():
    f,t,m=frames();runtime=AsyncPerception(NoWorker());runtime.selected_target_id=t['id']
    try:
        runtime._initialize(f,{'targets':[t],'_masks':{t['id']:m},'diagnostics':{}})
        # New detector keyframe has no proposal. The independent confirmed
        # target is still measured in a current image until its finite expiry.
        current={**f,'timestamp':1.1,'frame_id':2}
        runtime._initialize(current,{'targets':[],'_masks':{},'diagnostics':{}})
        result=runtime.tracker.track(current)
        runtime._current_views({'frames':{'head':current}},result,t['id'])
        assert [v['id'] for v in result['targets']]==[t['id']]
        assert runtime.observations[t['id']]['camera_id']=='head'
        assert t['id'] in runtime.tracker.last_masks
        expired={**f,'timestamp':2.1,'frame_id':3}
        result=runtime.tracker.track(expired)
        runtime._current_views({'frames':{'head':expired}},result,t['id'])
        assert result['targets']==[]
    finally:runtime.close()


def test_pose_hold_keeps_actual_transform_and_acquisition_timestamp():
    f,t,m=frames();runtime=PoseRuntime(NoWorker(),interval=float('inf'))
    runtime.accepted=accepted_source(t)
    try:
        original=np.asarray(runtime.accepted['result']['T_world_object']).copy()
        shifted={**t,'center_world':(np.asarray(t['center_world'])+[.023,0,0]).tolist()}
        runtime.update({**f,'timestamp':1.4,'frame_id':2},[shifted],{})
        assert shifted['pose_valid'] is True
        assert np.array_equal(shifted['T_world_object'],original)
        assert shifted['pose_timestamp']==1.
        assert shifted['pose_source']=='foundationpose'
        assert runtime.identity_confirmation['timestamp']==1.
        runtime.update({**f,'timestamp':1.501,'frame_id':3},[shifted],{})
        assert shifted['pose_valid'] is False
        assert runtime.identity_confirmation is None
    finally:runtime.close()


def test_rejected_pose_update_keeps_original_measurement_only_until_existing_expiry():
    f,t,m=frames();runtime=PoseRuntime(NoWorker(),interval=float('inf'))
    runtime.accepted=accepted_source(t)
    runtime.source={**runtime.accepted,'timestamp':1.2}
    runtime.future=Future();runtime.future.set_result({'result':{'accepted':False,'retained_tracking_prior':True},
        'roundtrip_seconds':.02,'completed_monotonic':1.3})
    try:
        runtime.update({**f,'timestamp':1.4},[t],{})
        assert t['pose_valid'] and t['pose_timestamp']==1.
        assert runtime.diagnostics['status']=='rejected'
        assert runtime.diagnostics['retained_last_accepted_measurement']
        assert not runtime.reset_next
        runtime.update({**f,'timestamp':1.501},[t],{})
        assert t['pose_valid'] is False and runtime.identity_confirmation is None
    finally:runtime.close()


def test_rejected_other_identity_cannot_inherit_previous_pose():
    f,t,m=frames();runtime=PoseRuntime(NoWorker(),interval=float('inf'))
    runtime.accepted=accepted_source(t)
    runtime.source={**runtime.accepted,'id':'other','timestamp':1.2}
    runtime.future=Future();runtime.future.set_result({'result':{'accepted':False,'retained_tracking_prior':True},
        'roundtrip_seconds':.02,'completed_monotonic':1.3})
    try:
        runtime.update({**f,'timestamp':1.4},[t],{})
        assert runtime.accepted is None and runtime.reset_next
        assert not t.get('pose_valid',False)
    finally:runtime.close()


def test_warm_pose_result_is_consumed_in_current_frame_without_retimestamping():
    import time
    f,t,m=frames()
    class Worker(NoWorker):
        def estimate(self,frame,*args):
            time.sleep(.01)
            return {**accepted_source(t)['result'],'frame_id':frame['frame_id'],'timestamp':frame['timestamp']}
    runtime=PoseRuntime(Worker(),interval=0.,current_frame_wait_s=.08)
    runtime.accepted=accepted_source(t);runtime.reset_next=False
    try:
        current={**f,'frame_id':2,'timestamp':1.1}
        runtime.update(current,[t],{t['id']:m},t['id'])
        assert t['pose_timestamp']==1.1
        assert runtime.diagnostics['pose_result_consumed_in_same_frame']
        assert not runtime.diagnostics['current_frame_wait_timed_out']
    finally:runtime.close()


def test_warm_pose_deadline_does_not_relabel_old_measurement_as_current():
    import time
    f,t,m=frames()
    class Worker(NoWorker):
        def estimate(self,frame,*args):
            time.sleep(.03)
            return {**accepted_source(t)['result'],'frame_id':frame['frame_id'],'timestamp':frame['timestamp']}
    runtime=PoseRuntime(Worker(),interval=0.,current_frame_wait_s=.001)
    runtime.accepted=accepted_source(t);runtime.reset_next=False
    try:
        runtime.update({**f,'frame_id':2,'timestamp':1.1},[t],{t['id']:m},t['id'])
        assert t['pose_timestamp']==1.
        assert runtime.diagnostics['current_frame_wait_timed_out']
        assert not runtime.diagnostics['pose_result_consumed_in_same_frame']
    finally:runtime.close()


def test_warm_pose_concurrent_scheduling_tail_is_consumed_without_next_frame_delay():
    import time
    f,t,m=frames()
    class Worker(NoWorker):
        def estimate(self,frame,*args):
            time.sleep(.12)
            return {**accepted_source(t)['result'],'frame_id':frame['frame_id'],'timestamp':frame['timestamp']}
    runtime=PoseRuntime(Worker(),interval=0.)
    runtime.accepted=accepted_source(t);runtime.reset_next=False
    try:
        runtime.update({**f,'frame_id':2,'timestamp':1.1},[t],{t['id']:m},t['id'])
        assert t['pose_timestamp']==1.1
        assert runtime.diagnostics['pose_result_consumed_in_same_frame']
        assert not runtime.diagnostics['current_frame_wait_timed_out']
        assert runtime.diagnostics['current_frame_wait_budget_ms']==180.
    finally:runtime.close()


def test_two_completed_pose_results_keep_both_original_timing_events():
    f,t,m=frames()
    class Worker(NoWorker):
        def estimate(self,frame,*args):
            return {**accepted_source(t)['result'],'frame_id':frame['frame_id'],'timestamp':frame['timestamp']}
    runtime=PoseRuntime(Worker(),interval=0.)
    runtime.source={**accepted_source(t),'frame_id':1}
    runtime.future=Future();runtime.future.set_result({'result':{**accepted_source(t)['result'],'frame_id':1,'timestamp':1.},
        'roundtrip_seconds':.12,'completed_monotonic':1.2})
    try:
        runtime.update({**f,'frame_id':2,'timestamp':1.3},[t],{t['id']:m},t['id'])
        records=runtime.diagnostics['completed_results_this_update']
        assert [r['frame_id'] for r in records]==[1,2]
        assert records[0]['worker_roundtrip_ms']==120.
        assert records[0]['timestamp']==1. and records[1]['timestamp']==1.3
    finally:runtime.close()


def recovery_source(target):
    return {**accepted_source(target),'target':{**target,'timestamp':1.,'semantic_keyframe_timestamp':.7},
            'frame_id':1,'camera_id':'head','reset':False}


def test_current_accepted_CAD_can_restore_confirmed_ID_independent_of_LK():
    f,t,m=frames();calls=[]
    class Worker(NoWorker):
        def estimate(self,frame,mask,identity,reset):
            calls.append((frame,mask,identity,reset))
            return {**accepted_source(t)['result'],'frame_id':frame['frame_id'],'timestamp':frame['timestamp'],
                    'quality':{'silhouette_iou':.9,'measured_depth_support':.95,'median_depth_error_m':.002,'p90_depth_error_m':.003},
                    'recovered_from_current_depth':True,'_recovered_mask':m.copy()}
    runtime=PoseRuntime(Worker(),interval=0.);runtime.accepted=recovery_source(t);runtime.reset_next=False
    perception=AsyncPerception(NoWorker());perception.selected_target_id=t['id']
    try:
        targets=[];current={**f,'frame_id':2,'timestamp':1.1}
        result=runtime.update(current,targets,{},t['id'])
        assert result is targets and targets[0]['id']==t['id']
        assert calls[0][1] is None and calls[0][3] is False
        assert targets[0]['pose_timestamp']==1.1 and targets[0]['semantic_keyframe_timestamp']==.7
        assert targets[0]['perception_method']=='foundationpose_current_depth_CAD_recovery'
        assert targets[0]['normal_source']=='current_CAD_surface_mask_local_measured_depth'
        assert perception.confirm_cad_identity(runtime.identity_confirmation)
        observed=perception.confirmed_head_tracker.track({**f,'frame_id':3,'timestamp':1.2})
        assert observed['targets'][0]['id']==t['id']
        assert observed['targets'][0]['semantic_keyframe_timestamp']==.7
        assert observed['targets'][0]['perception_method']=='lk_ransac_CAD_depth_mask_current_rgbd'
        assert observed['targets'][0]['normal_source']=='current_CAD_surface_mask_local_measured_depth'
        assert '_recovered_mask' not in runtime.diagnostics['result']
    finally:runtime.close();perception.close()


def test_CAD_recovery_cannot_start_without_confirmation_or_fresh_prior():
    f,t,m=frames()
    for selected,now in ((None,1.1),('different',1.1),(t['id'],1.501)):
        runtime=PoseRuntime(NoWorker(),interval=0.);runtime.accepted=recovery_source(t);runtime.reset_next=False
        try:
            targets=[];runtime.update({**f,'frame_id':2,'timestamp':now},targets,{},selected)
            assert targets==[] and runtime.future is None
        finally:runtime.close()


def test_rejected_or_late_CAD_recovery_cannot_publish_missing_target():
    f,t,m=frames()
    class Rejected(NoWorker):
        def estimate(self,*args):return {'accepted':False,'reason':'occluded','retained_tracking_prior':True}
    runtime=PoseRuntime(Rejected(),interval=0.);runtime.accepted=recovery_source(t);runtime.reset_next=False
    try:
        targets=[];runtime.update({**f,'frame_id':2,'timestamp':1.1},targets,{},t['id'])
        assert not targets and runtime.identity_confirmation is None
        # A previously completed accepted frame cannot masquerade as frame3.
        runtime.accepted={**recovery_source(t),'frame_id':2,'timestamp':1.1,'recovery':True,
            'recovered_mask':m,'result':{**accepted_source(t)['result'],'recovered_from_current_depth':True}}
        runtime.interval=float('inf')
        runtime.update({**f,'frame_id':3,'timestamp':1.2},targets,{},t['id'])
        assert not targets and runtime.identity_confirmation is None
    finally:runtime.close()


def test_worker_recovery_with_wrong_acquisition_cannot_be_retimestamped():
    f,t,m=frames()
    class WrongFrame(NoWorker):
        def estimate(self,*args):
            return {**accepted_source(t)['result'],'frame_id':1,'timestamp':1.,
                    'recovered_from_current_depth':True,'_recovered_mask':m}
    runtime=PoseRuntime(WrongFrame(),interval=0.);runtime.accepted=recovery_source(t);runtime.reset_next=False
    try:
        targets=[];runtime.update({**f,'frame_id':2,'timestamp':1.1},targets,{},t['id'])
        assert not targets and runtime.identity_confirmation is None
        assert runtime.diagnostics['result']['reason']=='recovery_result_acquisition_mismatch'
        assert runtime.accepted['timestamp']==1.
    finally:runtime.close()


def test_same_frame_head_rejection_uses_actual_wrist_pose_and_projects_head_UI():
    f,t,m=frames();current={**f,'frame_id':2,'timestamp':1.1,'camera_id':'head'}
    transform=np.eye(4);transform[0,3]=.05
    wrist={**current,'camera_id':'right_wrist','T_world_camera':transform}
    calls=[]
    class Worker(NoWorker):
        def estimate(self,frame,mask,*args):
            calls.append((frame,mask))
            if frame['camera_id']=='head':return {'accepted':False,'reason':'occluded','retained_tracking_prior':True}
            return {**accepted_source(t)['result'],'frame_id':2,'timestamp':1.1,
                'quality':{'silhouette_iou':.9,'measured_depth_support':.95,'median_depth_error_m':.002,'p90_depth_error_m':.003},
                'recovered_from_current_depth':True,'_recovered_mask':m.copy()}
    runtime=PoseRuntime(Worker(),interval=0.);runtime.accepted=recovery_source(t);runtime.reset_next=False
    runtime._rank_recovery_views=lambda *args:[wrist]
    perception=AsyncPerception(NoWorker());perception.selected_target_id=t['id']
    try:
        targets=[copy.deepcopy(t)];runtime.update(current,targets,{t['id']:m},t['id'],recovery_frames={'right_wrist':wrist})
        assert len(calls)==2 and calls[1][0] is wrist and calls[1][1] is None
        assert len(targets)==1 and targets[0]['observation_camera']=='right_wrist'
        assert targets[0]['head_visible'] is False
        assert np.isclose(targets[0]['center_camera'][0],t['center_camera'][0]+.05)
        assert targets[0]['bbox_px'][0]>t['bbox_px'][0]+8
        assert runtime.diagnostics['observation_camera']=='right_wrist'
        assert runtime.identity_confirmation['camera_id']=='right_wrist'
        assert perception.confirm_cad_identity(runtime.identity_confirmation)
        assert t['id'] in perception.wrist_trackers['right_wrist'].last_masks
        assert t['id'] not in perception.tracker.last_masks
    finally:runtime.close();perception.close()


def test_current_head_can_be_ranked_after_wrist_failure_but_stale_or_outside_views_cannot():
    import json
    from teleop_assist.cad_surface import CATALOG
    f,t,m=frames();runtime=PoseRuntime(NoWorker(),interval=0.)
    runtime.accepted=recovery_source(t)
    asset=next(a for a in json.loads(CATALOG.read_text())['objects'] if a['id']=='PullHandleGrasp')
    pose=np.eye(4);pose[2,3]=1.
    runtime.accepted['result'].update(T_world_object=pose.tolist(),mesh_sha256=asset['cad_sha256'])
    head={**f,'frame_id':2,'timestamp':1.1}
    outside=np.eye(4);outside[0,3]=5.
    views={'head':head,'left_wrist':{**head,'T_world_camera':outside},
           'right_wrist':{**head,'timestamp':.8}}
    try:
        ranked=runtime._rank_recovery_views(head,views,'right_wrist')
        assert [v['camera_id'] for v in ranked]==['head']
        assert runtime._rank_recovery_views(head,views,'head')==[]
    finally:runtime.close()


def test_delayed_semantic_candidate_cannot_rewind_and_erase_current_confirmed_track():
    f,t,m=frames();runtime=AsyncPerception(NoWorker());runtime.selected_target_id=t['id']
    try:
        recent={**f,'timestamp':1.9,'frame_id':19}
        runtime._initialize(recent,{'targets':[t],'_masks':{t['id']:m},'diagnostics':{}})
        previous=runtime.confirmed_head_tracker
        delayed={**f,'timestamp':1.2,'frame_id':12}
        current={**f,'timestamp':2.,'frame_id':20}
        runtime._initialize(delayed,{'targets':[t],'_masks':{t['id']:m},'diagnostics':{}},
                            current_bundle={'frames':{'head':current}})
        # Candidate's direct time gap exceeds the existing gate. Preserve the
        # recent valid tracker, rather than clearing it before validation.
        assert runtime.confirmed_head_tracker is previous
        result=runtime.tracker.track(current)
        runtime._current_views({'frames':{'head':current}},result,t['id'])
        assert [v['id'] for v in result['targets']]==[t['id']]
        assert result['targets'][0]['semantic_keyframe_timestamp']==1.9
    finally:runtime.close()
