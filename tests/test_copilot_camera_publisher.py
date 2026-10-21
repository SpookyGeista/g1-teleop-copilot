import json
import time
import numpy as np
import pytest
from teleop_assist.copilot.camera_publisher import CameraPublisher


def frame():
    return dict(rgb=np.zeros((120,160,3),dtype=np.uint8),frame_id=8,timestamp=100.,
        clock_id='host_monotonic:worker_test',calibration_id='recorded_test',
        T_world_camera=np.eye(4),intrinsics=np.eye(3),camera_source='head')


def test_spawned_publisher_preserves_source_and_rejects_queueing_old_frames(tmp_path):
    publisher=CameraPublisher(tmp_path,assistance=False)
    try:
        source=frame();publisher.submit(source,source,[],None)
        source['timestamp']=500.  # The worker already owns its serialized copy.
        with pytest.raises(RuntimeError,match='in-flight'):publisher.submit(source,source,[],None)
        deadline=time.monotonic()+5
        while publisher.busy and time.monotonic()<deadline:
            publisher.poll();time.sleep(.005)
        assert not publisher.busy and publisher.last_work_s>0
        state=json.loads((tmp_path/'operator_camera/ui_state.json').read_text())
        assert state['display_frame']['capture_timestamp']==100.
        assert state['display_frame']['frame_id']==8
        assert not state['movement_authorized']
        assert (tmp_path/'frame.jpg').read_bytes()==(tmp_path/'operator_camera/frame.jpg').read_bytes()
    finally:publisher.close()
    assert not publisher.process.is_alive()


def test_worker_failure_is_reported_to_owner(tmp_path):
    publisher=CameraPublisher(tmp_path,assistance=False)
    try:
        publisher.submit({},None,[],None)
        deadline=time.monotonic()+5
        with pytest.raises(RuntimeError,match='KeyError'):
            while time.monotonic()<deadline:
                publisher.poll();time.sleep(.005)
    finally:publisher.close()
    assert not publisher.process.is_alive()


def test_head_publishes_before_mesh_work_and_mesh_failure_keeps_both_streams_alive(tmp_path,monkeypatch):
    from teleop_assist.copilot import camera_publisher, xr_grasp_overlay, preview, collision
    monkeypatch.setattr(preview,'HandPreview',lambda _:object())
    monkeypatch.setattr(collision,'GraspCollisionFilter',lambda:object())
    def fail_preview(*args):
        state=json.loads((tmp_path/'operator_camera/ui_state.json').read_text())
        assert state['display_frame']['frame_id']==8
        assert (tmp_path/'operator_camera/display_frame.bin').exists()
        raise RuntimeError('display-only mesh failure')
    monkeypatch.setattr(xr_grasp_overlay,'build_operator_grasp_preview',fail_preview)
    class Connection:
        jobs=iter([(frame(),frame(),[],None),None])
        sent=[]
        def recv(self):return next(self.jobs)
        def send(self,value):self.sent.append(value)
        def close(self):pass
    connection=Connection()
    camera_publisher._serve(connection,tmp_path,True)
    assert [v[0] for v in connection.sent]==['ready','done']
    assert (tmp_path/'ai_frames/frame.npz').exists()
    other=json.loads((tmp_path/'assist_camera/grasp_preview.json').read_text())
    assert not other['preview_current'] and other['error']=='RuntimeError'
