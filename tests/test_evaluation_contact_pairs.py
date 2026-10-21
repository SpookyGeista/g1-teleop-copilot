"""Filtered force channel contract without importing Isaac or using a GPU."""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from teleop_assist.isaac_backend import IsaacBackend, EVALUATION_HAND_BODY_NAMES, filtered_normal_force_vector


def sensor(name, force):
    calls=[]
    value=SimpleNamespace(is_initialized=True,body_names=[name],num_bodies=1,
        contact_physx_view=SimpleNamespace(filter_count=1),
        data=SimpleNamespace(force_matrix_w=torch.tensor(force,dtype=torch.float32).reshape(1,1,1,3)),
        update=lambda dt,force_recompute=False:calls.append((dt,force_recompute)))
    value.calls=calls
    return value


def backend():
    b=IsaacBackend.__new__(IsaacBackend)
    b.step_index=123
    b.evaluation_contact_target_path='/World/AssistCabinet/Parts/PullHandle'
    b.evaluation_disable_hand_target_collisions=False
    b.evaluation_contact_collision_filtering={'disabled':False,'method':'none','pairs':[]}
    b.evaluation_contact_sensors={name:sensor(name,[0.,0.,0.]) for name in EVALUATION_HAND_BODY_NAMES}
    return b


def test_default_backend_adds_no_evaluation_or_collision_filter():
    parameters=inspect.signature(IsaacBackend).parameters
    assert parameters['evaluation_contact_pairs'].default is False
    assert parameters['evaluation_disable_hand_target_collisions'].default is False
    with pytest.raises(ValueError,match='requires evaluation_contact_pairs'):
        IsaacBackend('/unused',evaluation_disable_hand_target_collisions=True)


def test_seven_separate_filtered_channels_have_same_step_and_units():
    b=backend();name=EVALUATION_HAND_BODY_NAMES[3]
    b.evaluation_contact_sensors[name]=sensor(name,[.002,-.003,.004])
    result=b._evaluation_contact_observation(17.125)
    assert result['schema']=='punktnull.evaluation_hand_target_contacts.v1'
    assert result['evaluation_only'] and result['valid']
    assert result['target_asset_id']=='PullHandleGrasp' and result['target_rigid_body_asset_id']=='PullHandle'
    assert result['timestamp']==17.125 and result['sim_time']==123*b.dt and result['physics_step']==123
    assert result['force_units']=='N' and not result['negative_control']
    assert len(result['per_body_normal_forces_W'])==7
    np.testing.assert_allclose(result['per_body_normal_forces_W'][name],[.002,-.003,.004])
    assert result['per_body_normal_force_norms_N'][name]==pytest.approx(np.sqrt(29)*.001)
    for item in b.evaluation_contact_sensors.values():assert item.calls==[(b.dt,True)]


@pytest.mark.parametrize('shape',[(1,7,1,3),(1,1,2,3),(1,1,3),(3,)])
def test_wrong_force_matrix_shape_fails_instead_of_silent_zero(shape):
    with pytest.raises(RuntimeError,match='force matrix'):filtered_normal_force_vector(np.zeros(shape))


def test_nonfinite_force_fails_closed():
    with pytest.raises(RuntimeError,match='finite'):filtered_normal_force_vector(np.full((1,1,1,3),np.nan))


@pytest.mark.parametrize('change',[
    lambda s:setattr(s,'is_initialized',False),
    lambda s:setattr(s,'body_names',['wrong_partner']),
    lambda s:setattr(s.contact_physx_view,'filter_count',2),
    lambda s:setattr(s.data,'force_matrix_w',None),
])
def test_bad_sensor_binding_is_not_hidden(change):
    b=backend();change(next(iter(b.evaluation_contact_sensors.values())))
    with pytest.raises(RuntimeError):b._evaluation_contact_observation(1.)


def test_negative_control_is_explicit_and_zero_vectors_are_not_thresholded():
    b=backend();b.evaluation_disable_hand_target_collisions=True
    b.evaluation_contact_collision_filtering={'disabled':True,'method':'UsdPhysics.FilteredPairsAPI','pairs':[{'hand_body_path':'hand','target_body_path':'drawer'}]}
    result=b._evaluation_contact_observation(1.)
    assert result['negative_control'] and result['collision_filtering']['disabled']
    assert all(value==[0.,0.,0.] for value in result['per_body_normal_forces_W'].values())


def test_missing_finger_channel_fails_closed():
    b=backend();b.evaluation_contact_sensors.pop(EVALUATION_HAND_BODY_NAMES[-1])
    with pytest.raises(RuntimeError,match='seven'):b._evaluation_contact_observation(1.)
