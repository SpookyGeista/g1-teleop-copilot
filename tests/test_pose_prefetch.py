"""CPU synchronization/identity contracts; no pretrained-inference claims."""
import copy
import threading
from concurrent.futures import Future

import numpy as np
import pytest

from teleop_assist.pose_bridge import PoseRuntime
from test_perception_runtime_multiview import NoWorker,recovery_source
from test_visual_tracker import frames


def registered(target):
    source=recovery_source(target)
    source['result']['mesh_sha256']='synthetic_test_mesh'
    return source


def result(frame,target,mask,**changes):
    return {**registered(target)['result'],'frame_id':frame['frame_id'],'timestamp':frame['timestamp'],
        'target_id':target['id'],'recovered_from_current_depth':True,'_recovered_mask':mask.copy(),
        'quality':{'silhouette_iou':.8,'measured_depth_support':.95,'median_depth_error_m':.002,'p90_depth_error_m':.003},
        **changes}


def test_prefetch_runs_before_CPU_finishes_and_update_does_not_submit_twice():
    f,t,m=frames();current={**f,'frame_id':2,'timestamp':1.1}
    entered=threading.Event();release=threading.Event();calls=[]
    class Worker(NoWorker):
        def estimate(self,frame,mask,identity,reset):
            calls.append((frame,mask,identity,reset));entered.set()
            assert release.wait(2)
            return result(frame,t,m)
    runtime=PoseRuntime(Worker(),interval=0.);runtime.accepted=registered(t);runtime.reset_next=False
    runtime._rank_recovery_views=lambda *a,**k:pytest.fail('Accepted primary must not rank other views')
    try:
        assert runtime.prefetch(current,t['id'])
        assert entered.wait(1)  # Native work started before ordinary CPU targets exist.
        assert calls[0][1] is None and not calls[0][3]
        assert not runtime.prefetch(current,t['id'])  # pending native state has one owner
        release.set();runtime.future.result(timeout=2)
        targets=[copy.deepcopy(t)]
        runtime.update(current,targets,{t['id']:m},t['id'])
        assert len(calls)==1 and runtime.future is None
        assert targets[0]['id']==t['id'] and targets[0]['pose_timestamp']==1.1
        assert targets[0]['semantic_keyframe_timestamp']==.7
        assert targets[0]['normal_source']=='current_CAD_surface_mask_local_measured_depth'
        assert runtime.recovered_observation['frame']['timestamp']==1.1
        assert runtime.diagnostics['prefetched_before_image_tracking']
        assert runtime.diagnostics['pose_result_consumed_in_same_frame']
        assert not runtime.prefetch(current,t['id'])
        runtime.update(current,targets,{t['id']:m},t['id'])
        assert len(calls)==1
    finally:release.set();runtime.close()


@pytest.mark.parametrize('selected,timestamp,reset,semantic,has_mesh',[
    (None,1.1,False,'handle',True),('another',1.1,False,'handle',True),
    ('learned_007',1.501,False,'handle',True),('learned_007',.99,False,'handle',True),
    ('learned_007',1.1,True,'handle',True),('learned_007',1.1,False,'screw',True),
    ('learned_007',1.1,False,'handle',False)])
def test_prefetch_requires_same_confirmed_fresh_registered_CAD(selected,timestamp,reset,semantic,has_mesh):
    f,t,m=frames();runtime=PoseRuntime(NoWorker(),interval=0.)
    runtime.accepted=registered(t);runtime.reset_next=reset
    runtime.accepted['target']['semantic_label']=semantic
    if not has_mesh:runtime.accepted['result'].pop('mesh_sha256')
    try:
        assert not runtime.prefetch({**f,'frame_id':2,'timestamp':timestamp},selected)
        assert runtime.future is None
    finally:runtime.close()


def test_initial_registration_still_requires_semantic_mask():
    f,t,m=frames();runtime=PoseRuntime(NoWorker(),interval=0.)
    try:
        assert not runtime.prefetch(f,t['id'])
        assert runtime.future is None and runtime.accepted is None
    finally:runtime.close()


def test_pending_previous_frame_is_not_parallelized_or_relabelled_current():
    f,t,m=frames();runtime=PoseRuntime(NoWorker(),interval=0.,current_frame_wait_s=0.)
    runtime.accepted=registered(t);runtime.reset_next=False
    previous={**f,'frame_id':2,'timestamp':1.1};current={**f,'frame_id':3,'timestamp':1.2}
    runtime.source={**registered(t),'frame_id':2,'timestamp':1.1,'recovery':True,'prefetched':True,
                    'observation_frame':previous}
    runtime.future=Future()
    try:
        assert not runtime.prefetch(current,t['id'])
        # Finish the old request only after the new frame's CPU work has begun.
        runtime.future.set_result({'result':result(previous,t,m),'roundtrip_seconds':.2,'completed_monotonic':1.2})
        runtime.last_submit=float('inf')
        targets=[];runtime.update(current,targets,{},t['id'])
        assert not targets and runtime.accepted['timestamp']==1.1
        assert runtime.identity_confirmation is None
    finally:runtime.close()


@pytest.mark.parametrize('changed', [{'frame_id':1},{'timestamp':1.0},{'target_id':'other'}])
def test_prefetched_result_requires_exact_acquisition_and_identity(changed):
    f,t,m=frames();current={**f,'frame_id':2,'timestamp':1.1}
    class Worker(NoWorker):
        def estimate(self,frame,*args):return result(frame,t,m,**changed)
    runtime=PoseRuntime(Worker(),interval=0.);runtime.accepted=registered(t);runtime.reset_next=False
    try:
        assert runtime.prefetch(current,t['id'])
        runtime.future.result(timeout=2)
        targets=[];runtime.update(current,targets,{},t['id'])
        assert not targets and runtime.accepted['timestamp']==1.
        assert runtime.diagnostics['result']['reason']=='recovery_result_acquisition_mismatch'
    finally:runtime.close()


def test_selection_change_cannot_recover_previous_ID():
    f,t,m=frames();current={**f,'frame_id':2,'timestamp':1.1}
    class Worker(NoWorker):
        def estimate(self,frame,*args):return result(frame,t,m)
    runtime=PoseRuntime(Worker(),interval=0.);runtime.accepted=registered(t);runtime.reset_next=False
    try:
        assert runtime.prefetch(current,t['id']);runtime.future.result(timeout=2)
        targets=[];runtime.update(current,targets,{},'different_confirmed_ID')
        assert not targets and runtime.identity_confirmation is None
    finally:runtime.close()


def test_prefetch_keeps_current_wrist_camera_and_ranks_fallback_only_in_background():
    f,t,m=frames();head={**f,'frame_id':2,'timestamp':1.1,'camera_id':'head'}
    wrist={**head,'camera_id':'right_wrist'};calls=[];rank_threads=[]
    class Worker(NoWorker):
        def estimate(self,frame,*args):
            calls.append(frame['camera_id'])
            if frame['camera_id']=='right_wrist':return {'accepted':False,'retained_tracking_prior':True}
            return result(frame,t,m)
    runtime=PoseRuntime(Worker(),interval=0.);runtime.accepted=registered(t);runtime.reset_next=False
    runtime.accepted['camera_id']='right_wrist'
    def rank(*args,**kwargs):
        rank_threads.append(threading.current_thread().name)
        assert kwargs['prior']['timestamp']==1.
        return [head]
    runtime._rank_recovery_views=rank
    try:
        assert runtime.prefetch(head,t['id'],recovery_frames={'head':head,'right_wrist':wrist})
        runtime.future.result(timeout=2)
        targets=[];runtime.update(head,targets,{},t['id'])
        assert calls==['right_wrist','head'] and all(name.startswith('cad-pose') for name in rank_threads)
        assert targets[0]['pose_timestamp']==1.1 and targets[0]['observation_camera']=='head'
        assert runtime.diagnostics['observation_camera']=='head'
    finally:runtime.close()
