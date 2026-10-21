"""Request-local raw-distance reuse, with every path decision still evaluated."""
from types import SimpleNamespace
import math
import numpy as np
import pytest
from teleop_assist.copilot.arm_collision import OriginalArmCollision, CollisionConfig


def toy(distance=.2):
    checker=object.__new__(OriginalArmCollision)
    checker.config=CollisionConfig()
    checker.model=checker.data=checker.geometry=None
    checker.geometry_data=SimpleNamespace(oMg=[SimpleNamespace(homogeneous=np.eye(4)),SimpleNamespace(homogeneous=np.eye(4))])
    checker.placements=[]
    checker.pin=SimpleNamespace(updateGeometryPlacements=lambda *args:checker.placements.append(args[-1].copy()))
    checker.fcl=SimpleNamespace(Transform3f=lambda r,t:None)
    checker.pairs=[(0,1)];checker.shapes=[0,1]
    checker.objects=[SimpleNamespace(name='a'),SimpleNamespace(name='b')]
    checker.geometry_links=['left_ankle_roll_link','right_arm']
    checker.levers=np.zeros((2,43));checker.pair_levers=np.zeros((1,43))
    checker.calls=[]
    def measure(*args):
        checker.calls.append(args)
        return distance
    checker._distance=measure
    return checker


def path(c,q,r,cache=None,allowed=None):
    return c._check_path(q,r,np.eye(4),[('floor','ground_halfspace',2,None)],allowed or {},_distance_cache=cache)


def test_identical_full_state_reuses_only_distances_and_repeats_placements():
    c=toy();q=np.zeros(43);cache={}
    first=path(c,q,q,cache)
    assert len(c.calls)==3
    assert path(c,q,q,cache)==first
    assert len(c.calls)==3
    assert len(c.placements)==2
    assert len(cache)==1
    assert not any('cache' in key for key in vars(c))


@pytest.mark.parametrize('index',[0,29,42])
def test_bit_different_joint_including_fingers_misses(index):
    c=toy();q=np.ones(43)*.01;r=q.copy();r[index]=np.nextafter(r[index],math.inf);cache={}
    path(c,q,q,cache);path(c,r,r,cache)
    assert len(cache)==2
    assert len(c.calls)==6


def test_cached_float_type_and_finiteness_fail_closed():
    c=toy();q=np.zeros(43);cache={};path(c,q,q,cache)
    for value in [float('nan'),float('inf'),float('-inf'),True,1,np.float64(.2),None]:
        next(iter(cache.values()))[('self',0)]=value
        with pytest.raises(ValueError,match='invalid cached FCL'):
            path(c,q,q,cache)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf'),True,1,None])
def test_nonfinite_or_wrong_type_new_distance_is_never_saved(value):
    c=toy(value);q=np.zeros(43);cache={}
    with pytest.raises(ValueError,match='invalid cached FCL'):
        path(c,q,q,cache)
    assert list(cache.values())==[{}]


def test_two_configurations_max_and_intermediates_recomputed():
    c=toy();q=np.zeros(43);r=q.copy();r[0]=.05;cache={}
    c.pair_levers[0,0]=1.
    c.levers[:,0]=1.
    path(c,q,q,cache);path(c,r,r,cache)
    frozen_keys=set(cache)
    first=path(c,q,r,cache);calls=len(c.calls)
    second=path(c,q,r,cache)
    assert first==second
    assert len(cache)==2 and set(cache)==frozen_keys
    assert len(c.calls)-calls==12 # four uncached intermediate samples * three distances
    assert len(c.placements)==14 # geometry placements still happen at all samples


def test_same_raw_distance_different_motion_bound_changes_verdict():
    c=toy(.005);c.pair_levers[0,0]=1.
    q=np.zeros(43);r=q.copy();r[0]=.01;cache={}
    assert path(c,q,q,cache)['collision_valid']
    assert path(c,r,r,cache)['collision_valid']
    actual=path(c,q,r,cache)
    uncached=path(c,q,r)
    assert actual==uncached
    assert not actual['collision_valid']
    assert actual['reason']=='insufficient_clearance_for_between_sample_bound'
    assert actual['witness']['motion_bound_m']==.005


def test_cached_support_distance_never_skips_movement_rejection():
    c=toy();c.levers[0,0]=1.
    q=np.zeros(43);r=q.copy();r[0]=.01;cache={};allowed={('left_ankle_roll_link','floor'):.002}
    assert path(c,q,q,cache,allowed)['collision_valid']
    assert path(c,r,r,cache,allowed)['collision_valid']
    for memo in [cache,None]:
        with pytest.raises(ValueError,match='support contact would move'):
            path(c,q,r,memo,allowed)


def test_lazy_first_collision_preserves_witness_and_does_not_evaluate_later_pairs():
    c=toy(-.1);q=np.zeros(43);cache={}
    first=path(c,q,q,cache)
    assert first['reason']=='self_collision_or_clearance_violation'
    assert len(c.calls)==1
    assert list(next(iter(cache.values())))==[('self',0)]
    assert path(c,q,q,cache)==first
    assert len(c.calls)==1


def test_distance_exception_is_not_memoized_or_converted_to_success():
    c=toy();q=np.zeros(43);cache={};calls=[]
    def fail(*args):calls.append(1);raise ValueError('FCL failure')
    c._distance=fail
    for _ in range(2):
        with pytest.raises(ValueError,match='FCL failure'):path(c,q,q,cache)
    assert len(calls)==2
    assert list(cache.values())==[{}]


def test_support_contacts_and_endpoint_witness_equal_uncached():
    c=toy(.002);q=np.zeros(43);r=q.copy();r[1]=.025;cache={};allowed={('left_ankle_roll_link','floor'):.002}
    path(c,q,q,cache,allowed);path(c,r,r,cache,allowed)
    assert path(c,q,r,cache,allowed)==path(c,q,r,None,allowed)
