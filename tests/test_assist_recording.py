import numpy as np
import pytest
from teleop_assist.recording import write_rgbd

def test_rgbd_archive_is_lossless_calibrated_and_never_overwrites(tmp_path):
    random=np.random.default_rng(123)
    frame={'rgb':random.integers(0,256,(12,16,3),dtype=np.uint8),
           'depth':random.uniform(.2,3.,(12,16)).astype(np.float32),
           'intrinsics':{'fx':8.,'fy':8.,'cx':8.,'cy':6.},
           'T_world_camera':np.eye(4),'timestamp':100.,'frame_id':7,'sim_time':.25}
    path=tmp_path/'frame.npz';write_rgbd(path,frame)
    with np.load(path,allow_pickle=False) as replay:
        np.testing.assert_array_equal(replay['rgb'],frame['rgb'])
        np.testing.assert_array_equal(replay['depth'],frame['depth'])
        np.testing.assert_array_equal(replay['intrinsics'],[[8,0,8],[0,8,6],[0,0,1]])
        assert replay['frame_id']==7 and replay['timestamp']==100.
    with pytest.raises(FileExistsError):write_rgbd(path,frame)

def test_replay_preserves_recorded_preview_and_rejects_input():
    from tools.replay_assist import ReplaySession
    s=ReplaySession()
    target={'id':'recorded_choice','bbox_px':[1,1,10,10],
            'center_world':[.5,0,1.],'normal_world':[-1,0,0]}
    s.update_frame([target],1,s.clock(),{'width':100,'height':100})
    s.selection.id='recorded_choice';s.selection.mode='preview'
    assert s.snapshot()['selection']=={'id':'recorded_choice','mode':'preview'}
    assert s.command({'type':'select','id':'anything'})['accepted'] is False
    assert s.snapshot()['viewer_only'] is True and s.input is None
