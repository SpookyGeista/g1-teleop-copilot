"""Public recheck contracts with fake endpoint evidence; no GPU or physics proof."""
from copy import deepcopy
import json
import os

import pytest

from test_copilot_sim_target_feed import FeedFixture, pose, recheck


def authorized_fixture(directory):
    fixture=FeedFixture(directory)
    fixture.select()
    prepared=fixture.prepare()
    assert prepared['batch'] is not None
    assert recheck(fixture)['accepted']
    return fixture,prepared


def assert_revoked(fixture,result):
    assert result['accepted'] is False
    assert result['motion_authorized'] is False
    assert result['deadline_renewed'] is False
    assert fixture.feed.active_candidate_id is None
    assert len(fixture.calls)==1


def test_identical_preview_bytes_still_expire_and_cannot_renew_existing_batch(tmp_path):
    fixture,prepared=authorized_fixture(tmp_path)
    batch=prepared['batch']
    original_stamp=batch.capture_timestamp
    original_model_stamp=batch.model_source_capture_timestamp
    path=tmp_path/'latest_proposals.json';payload=path.read_bytes()
    for elapsed in (.04,.04,.04):
        fixture.base.now+=elapsed
        checked=recheck(fixture)
        assert checked['accepted']
        assert checked['motion_authorized'] is False and checked['deadline_renewed'] is False
        assert batch.capture_timestamp==original_stamp
        assert batch.model_source_capture_timestamp==original_model_stamp
    fixture.base.now=original_stamp+.251
    assert_revoked(fixture,recheck(fixture))
    assert path.read_bytes()==payload


@pytest.mark.parametrize('filename',['latest_proposals.json','target_selection.json'])
@pytest.mark.parametrize('fault',['missing','corrupt','oversized','non_object'])
def test_previously_valid_mailbox_failure_never_reuses_cached_authority(tmp_path,filename,fault):
    fixture,_=authorized_fixture(tmp_path)
    path=tmp_path/filename
    if fault=='missing':path.unlink()
    elif fault=='corrupt':path.write_bytes(b'{broken')
    elif fault=='oversized':path.write_bytes(b'{'+b' '*(2*1024*1024)+b'}')
    else:path.write_bytes(b'[]')
    assert_revoked(fixture,recheck(fixture))
    fixture.write()
    assert not recheck(fixture)['accepted']  # Repair is not a fresh selection.
    assert fixture.prepare()['batch'] is None
    assert len(fixture.calls)==1


@pytest.mark.parametrize('filename',['latest_proposals.json','target_selection.json'])
def test_same_size_and_mtime_changed_bytes_revoke_bound_selection(tmp_path,filename):
    fixture,_=authorized_fixture(tmp_path)
    path=tmp_path/filename;before=path.read_bytes();metadata=path.stat()
    changed=before.replace(b'"target_id": "learned_1"',b'"target_id": "learned_2"')
    assert changed!=before and len(changed)==len(before)
    path.write_bytes(changed)
    os.utime(path,ns=(metadata.st_atime_ns,metadata.st_mtime_ns))
    assert path.stat().st_mtime_ns==metadata.st_mtime_ns
    assert_revoked(fixture,recheck(fixture))


@pytest.mark.parametrize('fault',[
    'non_mapping','duplicate_id','empty_id','long_id','empty_source','long_source',
    'string_score','boolean_score','nan_score','missing_base_pose','nonrigid_base_pose',
    'malformed_world_pose','reflected_world_pose',
])
def test_recheck_validates_nonselected_candidates_beyond_ui_display_limit(tmp_path,fault):
    fixture,_=authorized_fixture(tmp_path)
    original=fixture.base.preview['candidates'][0]
    candidates=[deepcopy(original)]+[
        dict(deepcopy(original),candidate_id=f'unselected_{index}') for index in range(40)]
    candidate=candidates[-1]
    if fault=='non_mapping':candidates[-1]=None
    elif fault=='duplicate_id':candidate['candidate_id']='grasp1'
    elif fault=='empty_id':candidate['candidate_id']=''
    elif fault=='long_id':candidate['candidate_id']='x'*257
    elif fault=='empty_source':candidate['source']=''
    elif fault=='long_source':candidate['source']='x'*257
    elif fault=='string_score':candidate['model_score']='high'
    elif fault=='boolean_score':candidate['model_score']=True
    elif fault=='nan_score':candidate['model_score']=float('nan')
    elif fault=='missing_base_pose':candidate.pop('base_ee')
    elif fault=='nonrigid_base_pose':candidate['base_ee'][0][0]=2.
    elif fault=='malformed_world_pose':candidate['world_canonical']=[[1.]]
    else:
        candidate['world_canonical']=pose()
        candidate['world_canonical'][0][0]=-1.
    fixture.base.preview['candidates']=candidates
    # NaN is intentionally malformed untrusted JSON, not emitted by atomic_json.
    (tmp_path/'latest_proposals.json').write_text(json.dumps(fixture.base.preview))
    assert_revoked(fixture,recheck(fixture))


def test_recheck_preserves_eligibility_filter_for_nonselected_candidates(tmp_path):
    fixture,_=authorized_fixture(tmp_path)
    ineligible=dict(deepcopy(fixture.base.preview['candidates'][0]),candidate_id='ineligible',
        geometry_valid=False,base_ee=None,world_canonical=None,model_score='not_a_score',source='')
    fixture.base.preview['candidates'].append(ineligible)
    fixture.write()
    assert recheck(fixture)['accepted']
    assert len(fixture.calls)==1


def test_identical_preview_bytes_do_not_hide_new_recorded_scene_motion(tmp_path):
    fixture,_=authorized_fixture(tmp_path)
    payload=(tmp_path/'latest_proposals.json').read_bytes()
    fixture.base.now+=.06;fixture.base.index+=1
    scene=fixture.base.scene()
    scene['obstacles'][1]['world_pose'][0][3]+=.01
    fixture.feed.record_frame(fixture.base.frame(),scene)
    assert_revoked(fixture,recheck(fixture))
    assert (tmp_path/'latest_proposals.json').read_bytes()==payload


@pytest.mark.parametrize('context',['clock','calibration','confirmed_selection'])
def test_identical_preview_bytes_do_not_cache_dynamic_context_acceptance(tmp_path,context):
    fixture,_=authorized_fixture(tmp_path)
    changes={'clock_id':'other'} if context=='clock' else (
        {'calibration_id':'other'} if context=='calibration' else
        {'session_status':{'snapshot':{'confirmed_candidate_id':'another_grasp'}}})
    assert_revoked(fixture,recheck(fixture,**changes))


def test_new_valid_preview_is_read_without_retiming_already_issued_batch(tmp_path):
    fixture,prepared=authorized_fixture(tmp_path)
    batch=prepared['batch'];stamp=batch.capture_timestamp;model_stamp=batch.model_source_capture_timestamp
    fixture.base.now+=.06;fixture.base.index+=1
    assert fixture.feed.record_frame(fixture.base.frame(),fixture.base.scene())['recorded']
    fixture.base.renew_preview();fixture.write()
    checked=recheck(fixture)
    assert checked['accepted']
    assert checked['preview_evidence']['preview_capture_timestamp']==fixture.base.now
    assert batch.capture_timestamp==stamp and batch.model_source_capture_timestamp==model_stamp
    assert len(fixture.calls)==1


def test_recheck_invalidates_prepared_capability_without_creating_another(tmp_path):
    fixture,_=authorized_fixture(tmp_path)
    previous=fixture.feed.gate.inspect(fixture.base.preview,selected_candidate_id='grasp1',
        confirmed_candidate_id='grasp1',now=fixture.base.now,clock_id='mono',calibration_id='cal')
    assert previous.ready and fixture.feed.gate._prepared is not None
    for _ in range(3):
        checked=recheck(fixture)
        assert checked['accepted']
        assert fixture.feed.gate._prepared is None
        assert not {'batch','candidate','prepared','token'} & checked.keys()
    rejected=fixture.feed.gate.authorize(previous,fixture.base.endpoint(),
        robot_state=fixture.base.state(),scene=fixture.base.scene(),now=fixture.base.now,clock_id='mono')
    assert not rejected.accepted and rejected.batch is None
    assert rejected.reason=='prepared_selection_missing_changed_or_revoked'
    assert len(fixture.calls)==1


def test_recheck_cannot_recreate_explicitly_reset_frozen_binding(tmp_path):
    fixture,_=authorized_fixture(tmp_path)
    fixture.feed.gate.reset()
    assert_revoked(fixture,recheck(fixture))
    assert fixture.feed.gate._frozen is None and fixture.feed.gate._prepared is None


def test_mutating_public_preview_outputs_cannot_poison_private_parsed_mailbox(tmp_path):
    fixture=FeedFixture(tmp_path)
    fixture.base.preview['candidates'][0].update(world_canonical=pose(.4,-.18,.9),
        observed_hand_collision=dict(pregrasp_collision_free=True,status='explicit_test_fixture',
            final={'forbidden_object_points':0,'scene_collision_points':0,'allowed_target_pad_contacts':1}))
    fixture.write();fixture.select();prepared=fixture.prepare()
    assert prepared['batch'] is not None
    source=(tmp_path/'latest_proposals.json').read_bytes()
    assert recheck(fixture)['accepted']
    row=prepared['candidates'][0]
    row['base_ee'][0][3]=99.
    row['world_canonical'][0][0]=2.
    row['observed_hand_collision']['pregrasp_collision_free']=False
    row['observed_hand_collision']['final']['scene_collision_points']=99
    row.update(candidate_id='changed_output',source='changed_output',model_score=-99.)
    prepared['goal'][0,3]=99.
    prepared['preview_evidence']['target_id']='changed_output'
    checked=recheck(fixture)
    assert checked['accepted']
    checked['preview_evidence'].update(target_id='another_mutation',selected_candidate_id='another_mutation')
    assert recheck(fixture)['accepted']
    again=fixture.prepare()
    assert again['batch'] is not None
    assert again['candidates'][0]['candidate_id']=='grasp1'
    assert again['candidates'][0]['base_ee']==fixture.base.preview['candidates'][0]['base_ee']
    assert again['candidates'][0]['observed_hand_collision']['final']['scene_collision_points']==0
    assert (tmp_path/'latest_proposals.json').read_bytes()==source
