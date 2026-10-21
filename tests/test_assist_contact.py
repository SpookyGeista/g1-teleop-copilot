import numpy as np
from teleop_assist.core import AssistSession

def ready():
    s=AssistSession(clock=lambda:1.)
    robot={'valid':True,'ee_position':[.47,-.25,1.],'ee_quaternion':[0,0,0,1]}
    p={'source':'test','seq':1,'right':{'valid':True,'position':[0,0,0]},'clutch':True,'trigger':0.}
    s.step(robot,.005);s.receive_input(p);s.calibrate_hand()
    robot.update(contact_active=True,collision_stop=True,contact_force_world=[-30.,0,0])
    return s,robot,p

def test_contact_blocks_push_but_accepts_only_projected_outward_withdrawal():
    s,robot,p=ready();p['seq']=2;p['right']['position']=[0,0,-.05]
    s.receive_input(p);assert s.step(robot,.005)['hold']
    assert s.control['reason']=='collision'
    p['seq']=3;p['right']['position']=[-.03,.02,.05];p['trigger']=1.
    s.receive_input(p);result=s.step(robot,.005)
    assert not result['hold'] and s.control['reason']=='contact_retreat'
    assert result['ee_target_world'][0]<robot['ee_position'][0]
    np.testing.assert_allclose(result['ee_target_world'][1:],robot['ee_position'][1:])
    assert result['grip']==0 and result['ee_quat_xyzw'] is None
    assert not s.control['active']

def test_contact_allows_opening_but_stop_and_deadman_still_gate_motion():
    s,robot,p=ready();s.last_grip=1.
    p['seq']=2;p['trigger']=0.;s.receive_input(p)
    result=s.step(robot,.005);assert result['hold'] and result['grip']==0.
    p['seq']=3;p['right']['position']=[0,0,.05];p['clutch']=False;s.receive_input(p)
    assert s.step(robot,.005)['hold'] and s.control['reason']=='deadman_released'
    p['seq']=4;p['clutch']=True;s.receive_input(p);s.command({'type':'stop'})
    assert s.step(robot,.005)['hold'] and s.control['reason']=='stopped'

def test_unknown_contact_direction_never_allows_retreat():
    s,robot,p=ready();robot['contact_force_world']=[0,0,0]
    p['seq']=2;p['right']['position']=[0,0,.05];s.receive_input(p)
    assert s.step(robot,.005)['hold'] and s.control['reason']=='collision'
