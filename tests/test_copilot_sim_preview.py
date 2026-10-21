"""CPU worker contracts with explicit fake validators; no physics success claim."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from teleop_assist.copilot import sim_preview as module
from test_copilot_live_sim_proposals import complete_sim_fixture, digest


def request(count=512):
    f=complete_sim_fixture();batch=deepcopy(f.preview)
    one=batch['candidates'][0]
    batch['candidates']=[dict(deepcopy(one),candidate_id=f'candidate{i}',model_score=1-i/(count+1),
        world_canonical=np.eye(4).tolist()) for i in range(count)]
    return dict(policy=module.POLICY,batch=batch,robot_state=f.state(),scene=f.scene())


class Validators:
    def __init__(self):
        self.ik_sizes=[];self.collision_sizes=[];self.fault=None
        self.manifest={'scope':'explicit_test_double_not_robot_geometry'}

    def ik(self, repository):
        owner=self
        class IK:
            manifest=owner.manifest
            def validate_request(self,value):
                part=deepcopy(value['batch']);owner.ik_sizes.append(len(part['candidates']))
                for row in part['candidates']:
                    index=int(row['candidate_id'].removeprefix('candidate'))
                    row['ik_valid']=index%3!=1
                    row['kinematic_validation']=dict(ik_valid=row['ik_valid'],reason='explicit_fake_IK',
                        q_initial=value['current_arm_q'],q_solution=[.01]*14,
                        left_target_base_ee=value['left_base_ee'],right_target_base_ee=row['base_ee'],
                        final=dict(solver_success=True,limits_valid=True),iterations=['must_be_removed']*100)
                if owner.fault=='changed_id':part['candidates'][0]['candidate_id']='wrong'
                if owner.fault=='missing_candidate':part['candidates'].pop()
                return part
        return IK()

    def collision(self, repository):
        owner=self
        class Collision:
            manifest=owner.manifest
            def validate_request(self,value,*,endpoint_only=False):
                assert endpoint_only is True
                part=deepcopy(value['batch']);owner.collision_sizes.append(len(part['candidates']))
                state,scene=deepcopy(value['robot_state']),deepcopy(value['scene'])
                proof=dict(reason='evaluated',scope='sim_candidate_endpoint_only',path_check_requested=False,
                    live_execution_authorized=False,current_state_precheck={'collision_valid':True},
                    urdf_vs_observed_frames={'valid':True},robot_state=state,scene=scene,
                    snapshot_sha256=digest(dict(robot_state=state,scene=scene)))
                if owner.fault=='body_fk':proof['urdf_vs_observed_frames']['valid']=False
                if owner.fault=='current_collision':proof['current_state_precheck']['collision_valid']=False
                if owner.fault=='snapshot_hash':proof['snapshot_sha256']='wrong'
                if owner.fault=='scene_changed':proof['scene']['source']='unexpected'
                part['arm_collision_validation']=proof
                for row in part['candidates']:
                    index=int(row['candidate_id'].removeprefix('candidate')) if row['candidate_id'].startswith('candidate') else 1
                    row['collision_valid']=row.get('ik_valid') is True and index%5==0
                    row['arm_collision_validation']=dict(reason='explicit_fake_endpoint',
                        collision_valid=row['collision_valid'],witness={'a':'fake_robot','b':'fake_table'},
                        minimum_sample_distance_m=.002,full_q43_endpoint_fk={'explicit_test_double':True})
                return part
        return Collision()

    def run(self,value):
        return module.evaluate_sim_preview(value,'/explicit/fake/repository',ik_factory=self.ik,
            collision_factory=self.collision,descriptor_validator=lambda _:dict(geometry_valid=True,descriptor_sha256='a'*64))


def test_all_512_checked_in_bounded_chunks_and_clear_candidates_sorted_first():
    source=request();before=deepcopy(source);validator=Validators();result=validator.run(source)
    assert source==before and len(result['candidates'])==512
    assert validator.ik_sizes==[128]*4 and validator.collision_sizes==[1,128,128,128,128]
    evidence=result['sim_preview_validation']
    expected=[i for i in range(512) if i%3!=1 and i%5==0]
    assert evidence['evaluated_count']==512 and evidence['recorded_sim_endpoint_clear_count']==len(expected)
    assert evidence['ik_accepted_count']==sum(i%3!=1 for i in range(512))
    assert [c['candidate_id'] for c in result['candidates'][:len(expected)]]==[f'candidate{i}' for i in expected]
    assert result['capture_timestamp']==source['batch']['capture_timestamp'] and result['clock_id']=='mono'
    assert evidence['source_sha256']['robot_state']==module._hash(source['robot_state'])
    assert result['valid'] is False and result['motion_static'] is False and result['motion_authorized'] is False
    for row in result['candidates']:
        assert row['collision_valid'] is False and row['motion_authorized'] is False and row['preview_only']
        assert 'iterations' not in row['kinematic_validation'] and len(row['kinematic_validation']['q_solution'])==14
        assert row['recorded_sim_endpoint_validation']['path_validated'] is False


@pytest.mark.parametrize('fault',['policy','source','complete','static','state_time','scene_time','clock',
    'velocity','descriptor','duplicate','candidate_pose','candidate_score','q43_missing'])
def test_invalid_or_unbound_inputs_never_become_preview_hints(fault):
    source=request(3);validator=Validators()
    if fault=='policy':source['policy']='real_robot'
    elif fault=='source':source['scene']['source']='real_camera'
    elif fault=='complete':source['scene']['geometry_complete']=False
    elif fault=='static':source['scene']['static']=False
    elif fault=='state_time':source['robot_state']['acquisition_timestamp']+=.001
    elif fault=='scene_time':source['scene']['acquisition_timestamp']+=.021
    elif fault=='clock':source['scene']['clock_id']='other'
    elif fault=='velocity':source['scene']['observed_target_velocity'][0]=.1
    elif fault=='descriptor':source['batch']['descriptor_sha256']='b'*64
    elif fault=='duplicate':source['batch']['candidates'][1]['candidate_id']='candidate0'
    elif fault=='candidate_pose':source['batch']['candidates'][0]['base_ee'][0][0]=2.
    elif fault=='candidate_score':source['batch']['candidates'][0]['model_score']=float('nan')
    else:source['robot_state']['joint_positions'].pop('waist_yaw_joint')
    with pytest.raises((ValueError,TypeError,KeyError)):validator.run(source)
    assert validator.ik_sizes==[] and validator.collision_sizes==[]


@pytest.mark.parametrize('fault',['body_fk','current_collision','snapshot_hash','scene_changed','changed_id','missing_candidate'])
def test_independent_validator_faults_fail_closed_without_partial_results(fault):
    validator=Validators();validator.fault=fault
    with pytest.raises(ValueError):validator.run(request(3))


def test_incoming_permission_flags_cannot_override_actual_preview_checks():
    source=request(3);candidate=source['batch']['candidates'][1]
    candidate.update(collision_valid=True,recorded_sim_endpoint_clear=True,motion_authorized=True)
    result=Validators().run(source);row=next(c for c in result['candidates'] if c['candidate_id']=='candidate1')
    assert not row['ik_valid'] and not row['recorded_sim_endpoint_clear']
    assert not row['collision_valid'] and not row['motion_authorized']


def test_file_ipc_supports_more_than_mailbox_limit_but_enforces_64mib(tmp_path,monkeypatch):
    path=tmp_path/'large.json';value={'data':'x'*(3*1024*1024)}
    module.write_json_file(path,value);assert module.read_json_file(path)==value
    monkeypatch.setattr(module,'MAX_JSON_BYTES',100)
    with pytest.raises(ValueError):module.read_json_file(path)
    with pytest.raises(ValueError):module.write_json_file(tmp_path/'too_large.json',value)
    assert not (tmp_path/'too_large.json').exists()


def fake_worker(monkeypatch, *, corrupt=None, failure=None, seen=None):
    def run(command,**kwargs):
        if seen is not None:seen.append(command)
        if failure is not None:raise failure
        source=Path(command[command.index('--input')+1]);target=Path(command[command.index('--output')+1])
        result=Validators().run(module.read_json_file(source))
        if corrupt=='permission':result['candidates'][0]['collision_valid']=True
        elif corrupt=='timestamp':result['capture_timestamp']+=.1
        elif corrupt=='source_hash':result['sim_preview_validation']['source_sha256']['scene']='wrong'
        module.write_json_file(target,result)
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(module.subprocess,'run',run)


def test_isolated_wrapper_preserves_venv_interpreter_and_returns_bound_batch(tmp_path,monkeypatch):
    interpreter=tmp_path/'venv/bin/python';interpreter.parent.mkdir(parents=True);interpreter.symlink_to(sys.executable)
    seen=[];fake_worker(monkeypatch,seen=seen)
    result=module.isolated_sim_preview(request(3),'/fake/repository',interpreter)
    assert seen[0][0]==str(interpreter.absolute())
    assert len(result['candidates'])==3 and not result['valid']
    assert not Path(seen[0][seen[0].index('--input')+1]).exists()


@pytest.mark.parametrize('fault',['permission','timestamp','source_hash'])
def test_isolated_wrapper_rejects_corrupt_worker_result(monkeypatch,fault):
    fake_worker(monkeypatch,corrupt=fault)
    with pytest.raises(ValueError):module.isolated_sim_preview(request(3),'/fake',sys.executable)


@pytest.mark.parametrize('failure',[subprocess.TimeoutExpired('fake',.01),subprocess.CalledProcessError(3,'fake')])
def test_isolated_worker_hard_failure_never_substitutes_a_result(monkeypatch,failure):
    fake_worker(monkeypatch,failure=failure)
    with pytest.raises((subprocess.TimeoutExpired,RuntimeError)):
        module.isolated_sim_preview(request(3),'/fake',sys.executable)
