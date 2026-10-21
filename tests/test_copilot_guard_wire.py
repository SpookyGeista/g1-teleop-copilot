"""Quantized arm targets retain all numerical gates, without SDK/DDS imports."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import pytest


@pytest.fixture(scope='module')
def wire_cases(tmp_path_factory):
    python = Path(os.environ.get('COPILOT_ORIGINAL_IK_PYTHON', '/home/loya2/miniconda3/envs/to/bin/python'))
    repo = Path(os.environ.get('COPILOT_ORIGINAL_XR_REPOSITORY', '/home/loya2/xr_teleoperate'))
    if not python.is_file() or not repo.is_dir():
        pytest.skip('Original URDF / CPU Pinocchio environment unavailable')
    out = tmp_path_factory.mktemp('quantized_guard') / 'cases.json'
    code = r'''
import copy,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from teleop_assist.copilot.guard import ActualMotionGuard
import teleop_assist.copilot.guard as module
g=ActualMotionGuard(sys.argv[1]);v=g.collision;cases={}
q=np.zeros(43)
for name,value in [('waist_yaw_joint',.013333333333),('right_shoulder_pitch_joint',.123456789),
                   ('left_shoulder_pitch_joint',.134567891),('right_wrist_yaw_joint',.023456789),
                   ('left_hand_index_0_joint',-.0123456789)]:q[v.indices[name]]=value
base=np.eye(4);base[2,3]=3.;tcp=np.eye(4);tcp[0,3]=.05
def poses(values):
 v.pin.framesForwardKinematics(v.model,v.data,values)
 return [(v.data.oMf[v.model.getFrameId(s+'_wrist_yaw_link')].homogeneous@tcp).copy() for s in ('left','right')]
left,right=poses(q)
state=dict(joint_positions={n:float(q[v.indices[n]]) for n in v.names},base_in_world=base.tolist(),
 observed_body_world_poses={f.name:(base@v.data.oMf[i].homogeneous).tolist() for i,f in enumerate(v.model.frames)},
 complete=True,source='synthetic_full_q43_FK_fixture',acquisition_timestamp=10.,clock_id='test')
scene=dict(geometry_complete=True,static=True,source='synthetic_above_ground_scene',acquisition_timestamp=10.,
 clock_id='test',obstacles=[dict(id='floor',type='plane',normal=[0,0,1],offset=0.)])
def run(name,s=state,sc=scene,l=left,r=right,**kw):
 args=dict(acquisition_timestamp=10.,clock_id='test',now=10.,tracking_valid=True,
           command_format='unitree_float32');args.update(kw)
 cases[name]=g.validate(l,r,s,sc,**args);return cases[name]
run('clear')
run('explicit_float64',command_format='float64')
cases['default']=g.validate(left,right,state,scene,acquisition_timestamp=10.,clock_id='test',now=10.,tracking_valid=True)
for index,fmt in enumerate(('float16','float32',None,False,{},[])):
 run('invalid_format_'+str(index),command_format=fmt)
run('operator_stale',acquisition_timestamp=9.)
run('state_stale',s=dict(state,acquisition_timestamp=9.))
run('scene_stale',sc=dict(scene,acquisition_timestamp=9.))
run('wrong_clock',s=dict(state,clock_id='other'))
run('tracking_missing',tracking_valid=False)
bad=copy.deepcopy(state);bad['joint_positions'].pop('left_hand_index_0_joint');run('missing_finger',s=bad)
bad=copy.deepcopy(state);bad['observed_body_world_poses']['right_wrist_yaw_link'][0][3]+=.1;run('body_mismatch',s=bad)
blocked=copy.deepcopy(scene);T=np.eye(4);T[:3,3]=(base@right)[:3,3]
blocked['obstacles'].append(dict(id='block',type='box',size_xyz=[.2,.2,.2],world_pose=T.tolist()))
run('collision',sc=blocked)
goal=q.copy();goal[v.indices['right_wrist_yaw_joint']]+=.02
lt,rt=poses(goal);run('actual_small_rotation',l=lt,r=rt)
# A fabricated upstream success must not substitute for independent post-format FK.
real_noop=g._already_at_full_state_target
source=copy.deepcopy(cases['explicit_float64']['ik_validation'])
g._already_at_full_state_target=lambda *a:copy.deepcopy(source)
for side in ('left','right'):
 for kind in ('position','rotation'):
  lt,rt=left.copy(),right.copy();target=lt if side=='left' else rt
  if kind=='position':target[0,3]+=.00051
  else:target[:3,:3]=Rotation.from_rotvec([0,0,np.deg2rad(.51)]).as_matrix()@target[:3,:3]
  run('false_upstream_'+side+'_'+kind,l=lt,r=rt)
# Round-to-nearest can put an exact upper-limit double outside its hard limit.
outward=next(i for i in v.arm_indices if float(np.float32(v.model.upperPositionLimit[i]))>v.model.upperPositionLimit[i])
edge=q.copy();edge[outward]=v.model.upperPositionLimit[outward]
edge_ik=copy.deepcopy(source);edge_ik['q_solution']=edge[v.arm_indices].tolist()
g._already_at_full_state_target=lambda *a:copy.deepcopy(edge_ik)
run('rounded_outside_hard_limit',l=poses(edge)[0],r=poses(edge)[1])
# Quantization also cannot push the current-to-command delta above 0.15 rad.
index=v.indices['right_wrist_pitch_joint'];step=q.copy();step[index]+=.15
assert float(np.float32(step[index]))-q[index]>.15
step_ik=copy.deepcopy(source);step_ik['q_solution']=step[v.arm_indices].tolist()
g._already_at_full_state_target=lambda *a:copy.deepcopy(step_ik)
lt,rt=poses(step);run('rounded_outside_step_bound',l=lt,r=rt)
g._already_at_full_state_target=real_noop
ticks=iter([100.,100.2,100.2]);old=module.time.monotonic;module.time.monotonic=lambda:next(ticks)
try:run('lease_expired_during_validation')
finally:module.time.monotonic=old
cases['dds_modules']=[name for name in sys.modules if name.startswith(('cyclonedds','unitree_sdk2py'))]
json.dump(dict(cases=cases,arm_indices=v.arm_indices.tolist()),open(sys.argv[2],'w'),allow_nan=False)
'''
    result = subprocess.run([str(python), '-c', code, str(repo), str(out)],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=60,
        env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                 MPLCONFIGDIR=str(out.parent/'mpl')))
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(out.read_text())


def test_exact_float32_values_receive_independent_FK_and_same_collision_path(wire_cases):
    r = wire_cases['cases']['clear']; w = r['wire_format_validation']; ik = r['ik_validation']
    assert r['allowed'] and w['valid'] and r['command_format'] == 'unitree_float32'
    assert r['q14_command'] == w['q14_after_quantization'] == ik['q_solution']
    assert r['q14_command'] != w['q14_before_quantization']
    packed = struct.pack('<14f', *r['q14_command'])
    assert list(struct.unpack('<14f', packed)) == r['q14_command']
    assert packed.hex() == w['q14_float32_le_hex']
    assert hashlib.sha256(packed).hexdigest() == w['q14_float32_le_sha256']
    assert w['maximum_quantization_error_rad'] > 0
    assert w['limits_valid'] and w['fixed_nonarm_joints_unchanged']
    assert not w['motion_authorized'] and not w['transport_serialized'] and not w['crc_validated']
    assert ik['final']['solver_success'] is None
    assert ik['prequantization_final']['solver_status'] == 'not_run_already_at_full_q43_target'
    assert r['collision_validation']['q_final_43'] == ik['q_solution_43']
    assert r['collision_validation']['q_initial_43'] == ik['q_initial_43']
    for i,(before,after) in enumerate(zip(ik['q_initial_43'],ik['q_solution_43'])):
        if i not in wire_cases['arm_indices']:assert before == after
    assert r['collision_validation']['path_validated'] and r['urdf_vs_observed_frames']['valid']
    assert r['valid_until'] == pytest.approx(10.15)


def test_default_float64_output_and_semantics_unchanged(wire_cases):
    cases=wire_cases['cases']; a=cases['default']; b=cases['explicit_float64']
    assert a['allowed'] and b['allowed']
    assert a['q14_command'] == b['q14_command'] == a['ik_validation']['q_initial']
    assert 'wire_format_validation' not in a and 'command_format' not in a
    assert a['ik_validation']['validation_method'] == 'independent_current_full_q43_FK'


@pytest.mark.parametrize('index',range(6))
def test_unsupported_format_cannot_authorize(wire_cases,index):
    r=wire_cases['cases']['invalid_format_'+str(index)]
    assert not r['allowed'] and r['q14_command'] is None and r['reason']=='unsupported_command_format'


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('kind',['position','rotation'])
def test_quantized_FK_independently_checks_both_targets(wire_cases,side,kind):
    r=wire_cases['cases']['false_upstream_'+side+'_'+kind]
    assert not r['allowed'] and r['reason']=='unitree_float32_strict_full_q43_FK_rejected'
    assert r['q14_command'] is None and not r['wire_format_validation']['valid']
    assert 'collision_validation' not in r


def test_outward_quantization_does_not_receive_a_hard_limit_tolerance(wire_cases):
    r=wire_cases['cases']['rounded_outside_hard_limit']
    assert not r['allowed'] and r['reason']=='unitree_float32_full_q43_outside_hard_joint_limits'
    assert r['q14_command'] is None and not r['wire_format_validation']['limits_valid']


def test_joint_delta_applies_to_quantized_command(wire_cases):
    r=wire_cases['cases']['rounded_outside_step_bound']
    assert not r['allowed'] and r['reason']=='command_joint_step_exceeds_bound'
    assert r['wire_format_validation']['valid'] and r['max_joint_command_change_rad'] > .15


@pytest.mark.parametrize('name',['operator_stale','state_stale','scene_stale','wrong_clock',
    'tracking_missing','missing_finger','body_mismatch','collision','lease_expired_during_validation'])
def test_wire_format_retains_existing_current_state_path_and_time_gates(wire_cases,name):
    r=wire_cases['cases'][name]
    assert not r['allowed'] and r['hold_required'] and r['q14_command'] is None
    if name in ('body_mismatch','collision'):
        assert r['wire_format_validation']['valid']
        assert not r['collision_validation']['collision_valid']
    if name=='lease_expired_during_validation':
        assert r['reason']=='command_lease_expired_during_validation'
        assert r['valid_until']==pytest.approx(10.15)


def test_small_actual_full_state_IK_step_and_transport_free_scope(wire_cases):
    r=wire_cases['cases']['actual_small_rotation']
    assert r['allowed'] and r['wire_format_validation']['valid']
    assert r['collision_validation']['path_validated'] and not r['robot_command_sent']
    for side in ('left','right'):
        assert r['ik_validation']['final'][side+'_position_error_m']<=.0005
        assert r['ik_validation']['final'][side+'_orientation_error_rad']<=.00872664626
    assert wire_cases['cases']['dds_modules']==[]
