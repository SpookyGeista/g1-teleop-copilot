"""Exact absolute-tolerance behavior at rigid-transform acceptance boundaries."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.copilot.controller import rigid


def previous_rigid(value, name='transform'):
    """Pinned previous predicate; an equivalence oracle, never a production path."""
    pose=np.array(value,dtype=float,copy=True)
    if (pose.shape!=(4,4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3],[0,0,0,1],atol=1e-8,rtol=0)
            or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6,rtol=0)
            or abs(np.linalg.det(pose[:3,:3])-1)>1e-6):
        raise ValueError(f'{name} must be a finite proper rigid 4x4 transform')
    return pose


def outcome(function,value):
    try:
        result=function(value,'boundary fixture')
        return 'accepted',result.shape,result.dtype.str,result.tobytes()
    except (ValueError,TypeError,OverflowError) as error:
        return type(error).__name__,str(error)


def equivalence_cases():
    rng=np.random.default_rng(9004)
    # 56 valid poses stand in for the real pose corpus without coupling unit
    # tests to a historical runtime directory or its files.
    for i in range(56):
        pose=np.eye(4);pose[:3,:3]=Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        pose[:3,3]=rng.normal(size=3)
        yield pytest.param(pose,id=f'valid-pose-{i}')
    for field in [(3,0),(3,3),(0,0),(0,1)]:
        values=[0.,1e-8,np.nextafter(1e-8,0),np.nextafter(1e-8,np.inf),-1e-8,
                1e-6,np.nextafter(1e-6,np.inf),float('nan'),float('inf'),float('-inf')]
        for i,value in enumerate(values):
            pose=np.eye(4);pose[field]+=value
            yield pytest.param(pose,id=f'boundary-{field[0]}-{field[1]}-{i}')
    for name,value in [('none',None),('empty',[]),('wrong-shape',np.zeros((3,3))),
                       ('singular',np.zeros((4,4))),('reflection',np.diag([-1.,1.,1.,1.])),
                       ('nonnumeric',[['bad']*4]*4)]:
        yield pytest.param(value,id=name)
    for i in range(500):
        pose=np.eye(4)
        pose[:3,:3]+=rng.normal(0,10**rng.uniform(-10,-3),(3,3))
        pose[3]+=rng.normal(0,10**rng.uniform(-12,-5),4)
        yield pytest.param(pose,id=f'near-boundary-{i}')


@pytest.mark.parametrize('value',list(equivalence_cases()))
def test_same_values_and_failures_as_previous_predicate(value):
    assert outcome(rigid,value)==outcome(previous_rigid,value)


@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('sign',[-1.,1.])
@pytest.mark.parametrize('value,accepted',[(np.nextafter(1e-8,0),True),
    (1e-8,True),(np.nextafter(1e-8,np.inf),False)])
def test_last_row_absolute_tolerance_is_inclusive_without_relative_slack(axis,sign,value,accepted):
    pose=np.eye(4);pose[3,axis]=sign*value
    if accepted:
        assert np.array_equal(rigid(pose),pose)
    else:
        with pytest.raises(ValueError,match='finite proper rigid'):rigid(pose)


@pytest.mark.parametrize('scale,accepted',[(1.+.3e-6,True),(1.+.4e-6,False)])
def test_determinant_gate_remains_even_when_orthogonality_passes(scale,accepted):
    pose=np.eye(4);pose[:3,:3]*=scale
    assert np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),rtol=0,atol=1e-6)
    if accepted:rigid(pose)
    else:
        with pytest.raises(ValueError,match='finite proper rigid'):rigid(pose)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
@pytest.mark.parametrize('index',[(0,3),(3,0),(0,0)])
def test_nonfinite_translation_last_row_or_rotation_always_rejected(value,index):
    pose=np.eye(4);pose[index]=value
    with pytest.raises(ValueError,match='finite proper rigid'):rigid(pose)


def test_result_is_independent_copy_and_never_normalizes_input():
    source=np.eye(4);source[3,0]=1e-8
    result=rigid(source)
    assert result[3,0]==1e-8 and not np.shares_memory(source,result)
    result[0,3]=99.
    assert source[0,3]==0.
    source[1,3]=88.
    assert result[1,3]==0.


@pytest.mark.parametrize('value',[np.eye(4).astype(bool),np.eye(4).astype(str),np.eye(4,dtype=np.float32)])
def test_existing_float_conversion_contract_is_unchanged(value):
    assert outcome(rigid,value)==outcome(previous_rigid,value)
