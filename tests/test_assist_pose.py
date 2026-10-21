"""Calibrated XR stage rotations and bounded 6D operator control."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from teleop_assist.core import AssistSession

def setup_session(basis):
    s=AssistSession(clock=lambda:1.)
    robot={'valid':True,'ee_position':[.4,-.2,1.],'ee_quaternion':[0,0,0,1]}
    packet={'seq':1,'session_id':'pose','source':'test','orientation_mode':'pose',
            'right':{'valid':True,'position':[0,0,0],'quaternion_xyzw':[0,0,0,1]},
            'xr_to_world_rotation':basis,'clutch':True}
    s.step(robot,.01);s.receive_input(packet);s.calibrate_hand()
    return s,robot,packet

def test_nonidentity_anchor_maps_displacement_and_rotation_in_same_basis():
    basis=Rotation.from_euler('z',90,degrees=True).as_matrix()
    s,robot,p=setup_session(basis)
    p['seq']=2;p['right']['position']=[.05,0,0]
    p['right']['quaternion_xyzw']=Rotation.from_euler('x',30,degrees=True).as_quat().tolist()
    s.receive_input(p);r=s.step(robot,.02)
    np.testing.assert_allclose(r['raw_operator_target'],[.4,-.15,1.],atol=1e-9)
    rv=Rotation.from_quat(r['ee_quat_xyzw']).as_rotvec()
    np.testing.assert_allclose(rv,[0,s.config.max_angular_speed*.02,0],atol=1e-9)
    assert np.linalg.norm(np.asarray(r['ee_target_world'])-robot['ee_position'])<=s.config.max_speed*.02+1e-9

def test_pose_mode_change_requires_calibration_and_reflections_rejected_atomically():
    s,robot,p=setup_session(np.eye(3))
    p['seq']=2;p['xr_to_world_rotation']=np.diag([1,1,-1]).tolist()
    with pytest.raises(ValueError):s.receive_input(p)
    assert s.input['seq']==1
    p['xr_to_world_rotation']=np.eye(3).tolist();p['orientation_mode']='position'
    s.receive_input(p);assert s.step(robot,.01)['hold']
    assert s.control['reason']=='calibration_required'

def test_zero_controller_rotation_preserves_measured_wrist_orientation():
    s,robot,p=setup_session(np.eye(3))
    robot['ee_quaternion']=Rotation.from_euler('xyz',[15,-20,30],degrees=True).as_quat().tolist()
    s.robot=robot;s.calibrate_hand()
    r=s.step(robot,.01)
    assert (Rotation.from_quat(r['ee_quat_xyzw'])*Rotation.from_quat(robot['ee_quaternion']).inv()).magnitude()<1e-8
