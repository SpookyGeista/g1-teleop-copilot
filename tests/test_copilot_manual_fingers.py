from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from teleop_assist.copilot.manual_fingers import Dex3TriggerControl, trigger_values
from test_copilot_vr_buttons import Source

URDF=Path('/home/loya2/xr_teleoperate/assets/g1/g1_body29_hand14.urdf')


@pytest.fixture
def source(tmp_path):return Source(tmp_path/'operator.json')


def make():
    import xml.etree.ElementTree as ET
    names=[j.attrib['name'] for j in ET.parse(URDF).getroot().findall('joint') if j.find('limit') is not None]
    # Deliberately reversed articulation order catches positional DDS assumptions.
    names.reverse()
    return names,Dex3TriggerControl(names,URDF,simulation_mode=True)


def test_original_event_publication_keeps_both_analog_values(source):
    packet=source.sample(triggers=(.27,.83))
    assert trigger_values(packet,now=source.now,clock_id=source.clock_id)==dict(left=.27,right=.83)
    source.now+=.16
    assert trigger_values(packet,now=source.now,clock_id=source.clock_id)==dict(left=None,right=None)


@pytest.mark.parametrize('value',[None,True,'1',-1.,1.01,float('nan'),float('inf')])
def test_bad_trigger_holds_only_its_own_hand(source,value):
    packet=source.sample(triggers=(value,.6))
    assert trigger_values(packet,now=source.now,clock_id=source.clock_id)==dict(left=None,right=.6)


@pytest.mark.parametrize('field,value',[('event_sequence',999),('source_instance_id','foreign'),
    ('source_event_clock_id','foreign'),('clock_id','foreign'),('valid',False),('expires_at',100.),
    ('clock_uncertainty_s',.019),('received_at_host_monotonic',9.)])
def test_trigger_must_be_bound_to_this_pose_event(source,field,value):
    packet=source.sample(triggers=(1.,1.));packet['vr_buttons'][field]=value
    assert trigger_values(packet,now=source.now,clock_id=source.clock_id)==dict(left=None,right=None)


def test_independent_named_hands_proportional_closure_and_opening(source):
    names,control=make();q=np.zeros(len(names))
    for triggers in ((0.,1.),(1.,0.),(.5,.5),(0.,0.)):
        for _ in range(120):
            packet=source.sample(triggers=triggers)
            next_q=control.update(q,packet,enabled=True,now=source.now,clock_id=source.clock_id,dt=.005)
            assert np.max(np.abs(next_q-q))<=.010000001
            q=next_q
        for side,value in zip(('left','right'),triggers):
            sign=-1. if side=='left' else 1.
            assert q[names.index(side+'_hand_index_0_joint')]==pytest.approx(sign*(.005+.895*value))
            assert q[names.index(side+'_hand_thumb_1_joint')]==pytest.approx(-sign*.72*value)
        assert all(q[i]==0. for i,name in enumerate(names) if '_hand_' not in name)


def test_contact_does_not_build_up_closure_and_pause_or_loss_holds(source):
    names,control=make();measured=np.zeros(len(names))
    for _ in range(150):
        packet=source.sample(triggers=(1.,1.))
        target=control.update(measured,packet,enabled=True,now=source.now,clock_id=source.clock_id,dt=.005)
    assert np.max(np.abs(target-measured))<=.150000001
    opened=source.sample(triggers=(0.,0.))
    released=control.update(measured,opened,enabled=True,now=source.now,clock_id=source.clock_id,dt=.005)
    assert released[names.index('right_hand_index_0_joint')]<target[names.index('right_hand_index_0_joint')]
    for enabled,invalid in ((False,False),(True,True)):
        packet=source.sample(triggers=(1.,1.))
        if invalid:packet['tracking_valid']=False
        hold=control.update(measured,packet,enabled=enabled,now=source.now,clock_id=source.clock_id,dt=.005)
        np.testing.assert_array_equal(hold,measured)
        assert all(not hand['command_applied'] for hand in control.status['hands'].values())
    with pytest.raises(ValueError,match='isolated simulation'):
        Dex3TriggerControl(names,URDF,simulation_mode=False)
