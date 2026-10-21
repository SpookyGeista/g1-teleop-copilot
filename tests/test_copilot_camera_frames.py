import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.copilot.frames import mounted_camera_pose, publish_frame, load_frame


def test_articulated_parent_rotation_is_applied_to_mount_position_and_optical_axes():
    s=2**-.5
    frame={'body_names':['wrist'], 'body_poses':[[1,2,3,s,0,0,s]]}
    result=mounted_camera_pose(frame,'wrist',[.1,.2,.3],[s,0,s,0])
    assert np.allclose(result[:3,3],[.8,2.1,3.3])
    # Mount +Z looks along parent +X, which the measured wrist rotates to +Y.
    assert np.allclose(result[:3,:3]@[0,0,1],[0,1,0])
    assert np.allclose(result[:3,:3].T@result[:3,:3],np.eye(3))


@pytest.mark.parametrize('names,poses,link',[
    (['other'],[[0,0,0,1,0,0,0]],'wrist'),
    (['wrist','wrist'],[[0,0,0,1,0,0,0]]*2,'wrist'),
    (['wrist'],[[0,0,0,0,0,0,0]],'wrist'),
    (['wrist'],[[0,0,float('nan'),1,0,0,0]],'wrist'),
])
def test_missing_ambiguous_or_invalid_camera_parent_is_rejected(names,poses,link):
    with pytest.raises(ValueError):
        mounted_camera_pose(dict(body_names=names,body_poses=poses),link,[0,0,0],[1,0,0,0])


def test_frame_publication_preserves_camera_pose_and_source_clock(tmp_path):
    pose=np.eye(4);pose[:3,3]=[.2,-.1,1.]
    frame=dict(rgb=np.zeros((2,3,3),dtype=np.uint8),depth=np.ones((2,3)),
        intrinsics=np.eye(3),T_world_base=np.eye(4),T_world_camera=pose,
        timestamp=123.456,clock_id='original-boot-clock',calibration_id='measured-mount',frame_id=77)
    publish_frame(tmp_path,frame)
    result=load_frame(tmp_path/'frame.npz')
    assert result['timestamp']==123.456 and result['clock_id']=='original-boot-clock'
    assert result['frame_id']==77 and np.array_equal(result['T_world_camera'],pose)
    assert len(list(tmp_path.iterdir()))==1
