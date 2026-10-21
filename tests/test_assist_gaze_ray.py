import pytest
from teleop_assist.core import AssistSession

def ray(seq=1,**extra):
    return {'seq':seq,'source':'test_eye_ray','session_id':'device',
            'head_ray':{'origin':[0,0,0],'direction':[0,0,-1]},
            'valid':True,'calibrated':True,'confidence':.8,**extra}

def projected(seq=1,device_seq=1,**extra):
    return {'seq':seq,'source':'webxr_gaze_projection','session_id':'projection',
            'projection_of_seq':device_seq,'projection_of_session':'device',
            'uv':[.5,.5],'valid':True,'calibrated':True,'confidence':1.,'frame_id':1,**extra}

def session():
    clock=[1.];s=AssistSession(clock=lambda:clock[0]);s.config.eye_enabled=True
    s.update_frame([{'id':'part','bbox_px':[40,40,60,60],'center_world':[.5,0,1.],
                     'normal_world':[-1,0,0]}],1,1.,{'width':100,'height':100})
    return s,clock

def test_head_ray_does_not_invent_image_coordinates_before_projection():
    s,clock=session();s.set_gaze(ray())
    assert s.gaze is None and s.eye_status(clock[0])=='projection_required'
    s.set_gaze(projected())
    assert s.eye_status(clock[0])=='active'
    assert s.gaze['confidence']==.8

def test_high_rate_device_allows_recent_linked_projection_not_arbitrary_sequence():
    s,clock=session();s.set_gaze(ray(1));clock[0]+=.01;s.set_gaze(ray(2))
    assert s.set_gaze(projected(device_seq=1))
    with pytest.raises(ValueError):s.set_gaze(projected(seq=2,device_seq=100))
    assert s.set_gaze(projected(seq=2,device_seq=2))
    clock[0]+=.26
    assert s.eye_status(clock[0])=='stale'
    with pytest.raises(ValueError):s.set_gaze(projected(seq=3,device_seq=2))

def test_invalid_or_replaced_device_disables_prior_projection():
    s,clock=session();s.set_gaze(ray());s.set_gaze(projected())
    s.set_gaze(ray(2,valid=False));assert s.eye_status(clock[0])=='projection_required'
    with pytest.raises(ValueError):s.set_gaze(projected(seq=2))
    s.set_gaze(ray(1,session_id='replacement'))
    with pytest.raises(ValueError):s.set_gaze(projected(seq=2))

def test_nonunit_or_implausible_origin_rejected_without_consuming_sequence():
    s,clock=session()
    with pytest.raises(ValueError):s.set_gaze(ray(head_ray={'origin':[2,0,0],'direction':[0,0,-1]}))
    with pytest.raises(ValueError):s.set_gaze(ray(head_ray={'origin':[0,0,0],'direction':[0,0,0]}))
    assert s.set_gaze(ray())
