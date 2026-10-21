"""Numerical equivalence to the former scalar SciPy pose-comparison path."""
from copy import deepcopy
import json
import inspect
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.copilot.controller import rigid
from teleop_assist.copilot.sim_control import SimControlSession,SimControlConfig
from teleop_assist.copilot.arm_collision import OriginalArmCollision,CollisionConfig


def scalar_compare_observed_frames(self,q,base,observed):
    """Frozen pre-batch algorithm; intentionally retain the old per-row metric."""
    if not isinstance(observed,dict):
        return {'valid':False,'reason':'observed_body_world_poses_missing','frames':[]}
    self.pin.framesForwardKinematics(self.model,self.data,q)
    required=set(self.geometry_links)|{'left_wrist_yaw_link','right_wrist_yaw_link'}
    missing=sorted(required-set(observed));rows=[]
    for name,matrix in observed.items():
        if not self.model.existFrame(name):continue
        measured=rigid(matrix,'observed_body_world_pose')
        predicted=base@self.data.oMf[self.model.getFrameId(name)].homogeneous
        row={'body':name,'position_error_m':float(np.linalg.norm(predicted[:3,3]-measured[:3,3])),
            'orientation_error_rad':float(Rotation.from_matrix(predicted[:3,:3]@measured[:3,:3].T).magnitude())}
        if name in {'left_wrist_yaw_link','right_wrist_yaw_link'}:
            tcp=np.eye(4);tcp[0,3]=.05
            row['original_50mm_tcp_world']=(predicted@tcp).tolist()
            row['observed_50mm_tcp_world']=(measured@tcp).tolist()
            row['original_50mm_tcp_base']=(np.linalg.inv(base)@predicted@tcp).tolist()
        rows.append(row)
    max_pos=max((r['position_error_m'] for r in rows),default=float('inf'))
    max_rot=max((r['orientation_error_rad'] for r in rows),default=float('inf'))
    valid=bool(not missing and max_pos<=self.config.max_fk_position_error_m
        and max_rot<=self.config.max_fk_orientation_error_rad)
    return {'valid':valid,'reason':'within_tolerance' if valid else 'frame_missing_or_mismatch',
        'missing_required_bodies':missing,'compared_count':len(rows),
        'max_position_error_m':max_pos if rows else None,
        'max_orientation_error_rad':max_rot if rows else None,'frames':rows}


def session():
    value=object.__new__(SimControlSession);value.config=SimControlConfig();return value


def pose(angle=0.):
    value=np.eye(4);value[:3,:3]=Rotation.from_rotvec([angle,0,0]).as_matrix();return value


def outcome(function):
    try:return ('value',function())
    except Exception as error:return ('error',type(error).__name__,str(error))


def assert_body_equivalent(before,current):
    s=session()
    scalar=lambda:all(s._pose_matches(before[n],current[n],position=1e-8,rotation=1e-8) for n in before)
    assert outcome(scalar)==outcome(lambda:s._body_poses_match(before,current))


@pytest.mark.parametrize('angle',[0.,1e-8,np.nextafter(1e-8,0.),np.nextafter(1e-8,np.inf),
    1e-8-1e-12,1e-8+1e-12,2e-8,.005,np.pi-1e-12,np.pi])
@pytest.mark.parametrize('perturbation',['none','scale','shear'])
def test_body_exact_and_near_rigid_rotation_boundaries(angle,perturbation):
    rng=np.random.default_rng(2729);before={};current={}
    distortion=np.eye(3)
    if perturbation=='scale':distortion=np.diag([1+2e-7,1-2e-7,1])
    if perturbation=='shear':distortion+=np.array([[0,4e-7,0],[4e-7,0,0],[0,0,0]])
    for i in range(55):
        a=pose();a[:3,:3]=Rotation.random(random_state=rng).as_matrix()@distortion
        b=a.copy();b[:3,:3]=pose(angle)[:3,:3]@b[:3,:3]
        rigid(a);rigid(b)
        before[str(i)]=a.tolist();current[str(i)]=b.tolist()
    a,b=deepcopy(before),deepcopy(current)
    assert_body_equivalent(before,current)
    assert before==a and current==b


@pytest.mark.parametrize('delta',[0.,1e-8,np.nextafter(1e-8,0.),np.nextafter(1e-8,np.inf),-1e-8,-2e-8])
def test_position_metric_and_threshold_unchanged(delta):
    before={'body':pose().tolist()};current=deepcopy(before);current['body'][0][3]=float(delta)
    assert_body_equivalent(before,current)


@pytest.mark.parametrize('first',['valid','rotation','position','invalid'])
@pytest.mark.parametrize('later',['missing','shape','nan','reflection','string'])
def test_first_failure_short_circuit_and_error_identity(first,later):
    before={'first':pose(),'later':pose()};current=deepcopy(before)
    if first=='rotation':current['first']=pose(2e-8)
    elif first=='position':current['first'][0,3]=2e-8
    elif first=='invalid':current['first'][0,0]=2
    if later=='missing':current.pop('later')
    elif later=='shape':current['later']=[1,2,3]
    elif later=='nan':current['later'][0,0]=float('nan')
    elif later=='reflection':current['later'][0,0]=-1
    elif later=='string':current['later']='not a matrix'
    assert_body_equivalent(before,current)


def test_empty_body_sequence_preserves_vacuous_scalar_result():
    assert_body_equivalent({},{})


def test_raw_trace_cancellation_would_be_an_actual_false_accept():
    a=pose(2e-8);rigid(a)
    assert np.arccos(np.clip((np.trace(a[:3,:3])-1)/2,-1,1))==0
    assert session()._body_poses_match({'a':pose()},{'a':a}) is False


def fake_fk():
    rng=np.random.default_rng(781)
    names=['left_wrist_yaw_link','right_wrist_yaw_link']+[f'body_{i}' for i in range(53)]
    transforms=[]
    for name in names:
        p=pose();p[:3,:3]=Rotation.random(random_state=rng).as_matrix();p[:3,3]=rng.normal(size=3)
        transforms.append(SimpleNamespace(homogeneous=p))
    value=object.__new__(OriginalArmCollision)
    value.pin=SimpleNamespace(framesForwardKinematics=lambda *_:None)
    value.model=SimpleNamespace(existFrame=lambda name:name in names,getFrameId=names.index)
    value.data=SimpleNamespace(oMf=transforms);value.geometry_links=names;value.config=CollisionConfig()
    base=pose(.9);base[:3,3]=[.1,.5,.9]
    observed={n:(base@transforms[i].homogeneous).tolist() for i,n in enumerate(names)}
    return value,base,observed


@pytest.mark.parametrize('fault',['none','empty','nondict','missing','unknown','nan','reflection',
    'scale','shear','translation_below','translation_at','translation_above',
    'rotation_below','rotation_at','rotation_above','near_pi'])
def test_complete_fk_report_rows_TCPs_order_errors_and_thresholds_unchanged(fault):
    v,base,observed=fake_fk()
    if fault=='empty':observed={}
    elif fault=='nondict':observed=None
    elif fault=='missing':observed.pop('right_wrist_yaw_link')
    elif fault=='unknown':observed['unknown_ignored']=None
    elif fault!='none':
        name='body_52';a=np.array(observed[name])
        if fault=='nan':a[0,0]=float('nan')
        elif fault=='reflection':a[:3,0]*=-1
        elif fault=='scale':a[:3,:3]=a[:3,:3]@np.diag([1+2e-7,1-2e-7,1])
        elif fault=='shear':a[:3,:3]=a[:3,:3]@(np.eye(3)+np.array([[0,4e-7,0],[4e-7,0,0],[0,0,0]]))
        elif fault.startswith('translation'):
            delta=v.config.max_fk_position_error_m
            if fault.endswith('below'):delta=np.nextafter(delta,0.)
            elif fault.endswith('above'):delta=np.nextafter(delta,np.inf)
            a[0,3]+=delta
        else:
            delta=v.config.max_fk_orientation_error_rad
            if fault.endswith('below'):delta=np.nextafter(delta,0.)
            elif fault.endswith('above'):delta=np.nextafter(delta,np.inf)
            elif fault=='near_pi':delta=np.pi-1e-12
            a[:3,:3]=a[:3,:3]@pose(delta)[:3,:3]
        observed[name]=a.tolist()
    old=outcome(lambda:scalar_compare_observed_frames(v,None,base,observed))
    new=outcome(lambda:v.compare_observed_frames(None,base,observed))
    assert old==new
    # Also preserve field insertion order in JSON reports, not only dict equality.
    assert json.dumps(old)==json.dumps(new)


def test_native_Pinocchio_full_report_matches_old_scalar_algorithm():
    root=Path(__file__).resolve().parents[1]
    repository=Path(os.environ.get('COPILOT_ORIGINAL_XR_REPOSITORY','/home/loya2/xr_teleoperate'))
    python=Path(os.environ.get('COPILOT_ORIGINAL_IK_PYTHON','/home/loya2/miniconda3/envs/to/bin/python'))
    if not repository.is_dir() or not python.is_file():pytest.skip('Original CPU geometry environment not installed')
    code="""import sys,json,numpy as np
from scipy.spatial.transform import Rotation
from teleop_assist.copilot.controller import rigid
from teleop_assist.copilot.arm_collision import OriginalArmCollision
"""+inspect.getsource(scalar_compare_observed_frames)+"""
def audit(event,args):
 if event.startswith('socket.') or event in {'subprocess.Popen','os.fork','os.system'}:raise RuntimeError(event)
sys.addaudithook(audit)
v=OriginalArmCollision(sys.argv[1]);base=np.eye(4);base[:3,3]=[.1,-.2,3.]
for value in (0.,.01):
 q=np.full(v.model.nq,value)
 v.pin.framesForwardKinematics(v.model,v.data,q)
 observed={f.name:(base@v.data.oMf[i].homogeneous).tolist() for i,f in enumerate(v.model.frames)}
 old=scalar_compare_observed_frames(v,q,base,observed);new=v.compare_observed_frames(q,base,observed)
 assert json.dumps(old)==json.dumps(new)
assert not any(n.startswith(('isaaclab','isaacsim','cyclonedds','unitree_sdk2py')) for n in sys.modules)
print('native FK reports equal')
"""
    result=subprocess.run([str(python),'-c',code,str(repository)],cwd=root,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
