"""Synthetic association contracts; the real model replay is reported separately."""
import copy
import hashlib
import json

import numpy as np
import pytest

from teleop_assist.cad_identity import current_cad_comparison_mask,reconcile_current_proposals
from teleop_assist.pose_bridge import PoseRuntime
from test_perception_runtime_multiview import NoWorker
from test_visual_tracker import frames


@pytest.fixture
def observation(tmp_path,monkeypatch):
    mesh=tmp_path/'patch.obj'
    mesh.write_text('v -0.1 -0.1 1\nv 0.095 -0.1 1\nv 0.095 0.095 1\nv -0.1 0.095 1\nf 1 2 3\nf 1 3 4\n')
    digest=hashlib.sha256(mesh.read_bytes()).hexdigest();catalog=tmp_path/'catalog.json'
    catalog.write_text(json.dumps({'units':'meters','objects':[{'id':'PullHandleGrasp','mesh_units':'meters',
        'cad_mesh':mesh.name,'cad_sha256':digest}]}))
    monkeypatch.setattr('teleop_assist.cad_surface.CATALOG',catalog)
    f,t,m=frames();t.update(frame_id=1,timestamp=1.,semantic_keyframe_timestamp=.7)
    evidence={'accepted':True,'frame_id':1,'timestamp':1.,'cad_asset_id':'PullHandleGrasp',
              'mesh_sha256':digest,'T_world_object':np.eye(4).tolist()}
    return f,t,m,evidence


def test_unique_proposal_requires_current_pixel_and_surface_evidence(observation):
    f,t,m,e=observation
    proposal={**t,'id':'new','semantic_keyframe_timestamp':.9}
    result=reconcile_current_proposals(f,[proposal],{'new':m},t['id'],e,t['center_world'])
    assert result['status']=='unique_duplicate' and result['proposal_id']=='new'
    assert result['proposal_semantic_keyframe_timestamp']==.9
    assert result['proposals'][0]['mask_iou']>.95
    # Centroids may differ because different object faces are visible. Neither
    # closeness nor a copied label is the positive identity criterion.
    proposal['center_world']=[.08,0,1]
    assert reconcile_current_proposals(f,[proposal],{'new':m},t['id'],e,t['center_world'])['status']=='unique_duplicate'
    assert t['semantic_keyframe_timestamp']==.7


def test_two_compatible_proposals_and_nearby_different_depth_are_ambiguous(observation):
    f,t,m,e=observation
    proposals=[{**t,'id':'one'},{**t,'id':'two'}]
    result=reconcile_current_proposals(f,proposals,{'one':m,'two':m.copy()},t['id'],e,t['center_world'])
    assert result['status']=='ambiguous' and len(result['matching_ids'])==2
    depth=f['depth'].copy();depth[m]=.97
    result=reconcile_current_proposals({**f,'depth':depth},proposals[:1],{'one':m},t['id'],e,t['center_world'])
    assert result['status']=='ambiguous' and not result['matching_ids']
    assert not result['proposals'][0]['surface']['accepted']


def test_same_depth_adjacent_object_cannot_be_merged_by_nearness(observation):
    f,t,m,e=observation
    shifted=np.roll(m,38,axis=1);neighbor={**t,'id':'neighbor','center_world':[.035,0,1.]}
    result=reconcile_current_proposals(f,[neighbor],{'neighbor':shifted},t['id'],e,t['center_world'])
    assert result['status']=='ambiguous' and result['conflicting_ids']==['neighbor']
    assert not result['proposals'][0]['matches_current_CAD']


def test_missing_mask_old_proposal_or_old_CAD_is_not_identity_evidence(observation):
    f,t,m,e=observation;other={**t,'id':'new'}
    assert reconcile_current_proposals(f,[other],{},t['id'],e,t['center_world'])['status']=='ambiguous'
    other['timestamp']=.9
    assert reconcile_current_proposals(f,[other],{'new':m},t['id'],e,t['center_world'])['status']=='ambiguous'
    with pytest.raises(ValueError,match='Same-frame'):
        current_cad_comparison_mask(f,{**e,'frame_id':0})
    with pytest.raises(ValueError,match='Same-frame'):
        current_cad_comparison_mask(f,{**e,'timestamp':.9})


def test_dedup_existing_canonical_is_per_frame_and_preserves_other_objects(observation):
    f,t,m,e=observation;other={**t,'id':'new'}
    distant={**t,'id':'distant','center_world':[1.,0,1.]}
    runtime=PoseRuntime(NoWorker(),interval=float('inf'))
    runtime.accepted={'id':t['id'],'frame_id':1,'timestamp':1.,'center_world':np.asarray(t['center_world']),'result':e}
    try:
        targets=[copy.deepcopy(t),other,distant]
        assert runtime._reconcile_current_identity(f,targets,{'new':m},t['id'])
        assert [x['id'] for x in targets]==[t['id'],'distant']
        assert targets[0]['semantic_keyframe_timestamp']==.7
        assert runtime.identity_reconciliation['proposal_id']=='new'
        # No permanent alias: a later proposal gets checked against later
        # actual pixels, and ambiguous alternatives leave their IDs intact.
        both=[copy.deepcopy(t),{**other},{**other,'id':'second'}]
        assert not runtime._reconcile_current_identity(f,both,{'new':m,'second':m},t['id'])
        assert [x['id'] for x in both]==['new','second']
        none=[copy.deepcopy(t),{**other}]
        assert runtime._reconcile_current_identity(f,none,{'new':m},None)
        assert len(none)==2  # preview/unconfirmed never reconciles
    finally:runtime.close()


def test_disjoint_distant_proposals_skip_all_point_triangle_work(observation,monkeypatch):
    f,t,m,e=observation
    other={**t,'id':'distant','center_world':[1.,0.,1.]}
    disjoint=np.zeros(m.shape,bool);disjoint[5:20,5:20]=True
    def unexpected(*args):raise AssertionError('Disjoint projection must not inspect triangle depth')
    monkeypatch.setattr('teleop_assist.cad_surface.point_triangle_surface_distance',unexpected)
    result=reconcile_current_proposals(f,[other],{'distant':disjoint},t['id'],e,t['center_world'])
    assert result['status']=='no_duplicate' and not result['comparison_depth_evaluated']
    assert result['proposals']==[]
    # Missing masks remain ambiguous when their measured center is close,
    # without paying the unrelated full comparison-silhouette depth cost.
    other['center_world']=t['center_world']
    result=reconcile_current_proposals(f,[other],{},t['id'],e,t['center_world'])
    assert result['status']=='ambiguous' and not result['comparison_depth_evaluated']


def test_broad_phase_never_skips_near_object_or_overlapping_distant_centroid(observation):
    f,t,m,e=observation
    other={**t,'id':'other','center_world':t['center_world']}
    disjoint=np.zeros(m.shape,bool);disjoint[5:20,5:20]=True
    result=reconcile_current_proposals(f,[other],{'other':disjoint},t['id'],e,t['center_world'])
    assert result['status']=='ambiguous' and result['comparison_depth_evaluated']
    other['center_world']=[1.,0.,1.]
    result=reconcile_current_proposals(f,[other],{'other':m},t['id'],e,t['center_world'])
    assert result['status']=='unique_duplicate' and result['comparison_depth_evaluated']


def independent_depth_evidence(e,selected_id):
    return {**e,'target_id':selected_id,'method':'FoundationPose','operation':'track',
        'recovered_from_current_depth':True,'mask_prior_filter':{'applied':True,'prior_age_s':.1},
        'pose_observability':{'observable':True,'scaled_rank':6},
        'quality':{'minimum_observed_extent_passed':True,'silhouette_iou':.8,'measured_depth_support':.95,
                   'median_depth_error_m':.002,'p90_depth_error_m':.003},
        'pose_position_sigma_m':.012,'pose_orientation_sigma_rad':.1}


def contaminated_observation(observation):
    f,t,m,e=observation
    depth=f['depth'].copy();bad=m.copy();bad[:,:66]=False;depth[bad]=.97
    return {**f,'depth':depth},t,m,independent_depth_evidence(e,t['id'])


def test_current_depth_disproves_secondary_association_without_merging_or_erasing_it(observation):
    f,t,m,e=contaminated_observation(observation);other={**t,'id':'secondary'}
    result=reconcile_current_proposals(f,[other],{'secondary':m},t['id'],e,t['center_world'])
    assert result['status']=='unassigned_secondary_proposals'
    assert result['non_vetoing_unassigned_proposal_ids']==['secondary']
    assert not result['proposals'][0]['matches_current_CAD']
    assert not result['proposals'][0]['surface']['accepted']
    assert other['id']=='secondary' and 'proposal_id' not in result


@pytest.mark.parametrize('change',[
    {'recovered_from_current_depth':False},{'operation':'register'},{'target_id':'different'},
    {'mask_prior_filter':{'applied':True,'prior_age_s':.501}},
    {'pose_observability':{'observable':False,'scaled_rank':5}}])
def test_secondary_cannot_override_ambiguity_without_independent_current_CAD_proof(observation,change):
    f,t,m,e=contaminated_observation(observation);other={**t,'id':'secondary'}
    result=reconcile_current_proposals(f,[other],{'secondary':m},t['id'],{**e,**change},t['center_world'])
    assert result['status']=='ambiguous'
    assert not result.get('non_vetoing_unassigned_proposal_ids')


def test_compatible_shape_ambiguity_and_multiple_matching_proposals_still_withhold(observation):
    f,t,m,e=observation;e=independent_depth_evidence(e,t['id']);other={**t,'id':'secondary'}
    partial=m.copy();partial[45:]=False
    result=reconcile_current_proposals(f,[other],{'secondary':partial},t['id'],e,t['center_world'])
    assert result['proposals'][0]['surface']['accepted'] and result['status']=='ambiguous'
    twins=[other,{**other,'id':'second'}]
    result=reconcile_current_proposals(f,twins,{'secondary':m,'second':m},t['id'],e,t['center_world'])
    assert result['status']=='ambiguous' and len(result['matching_ids'])==2


def test_non_vetoing_proposals_are_current_frame_only_and_remain_visible(observation):
    f,t,m,e=contaminated_observation(observation);other={**t,'id':'secondary'}
    actual_mask=current_cad_comparison_mask(f,e)
    runtime=PoseRuntime(NoWorker(),interval=float('inf'))
    runtime.accepted={'id':t['id'],'frame_id':1,'timestamp':1.,'camera_id':'head','recovery':True,
        'target':copy.deepcopy(t),'center_world':np.asarray(t['center_world']),
        'normal_world':np.asarray(t['normal_world']),'recovered_mask':actual_mask,'result':e}
    try:
        targets=[other];runtime.update(f,targets,{'secondary':m},t['id'])
        assert {x['id'] for x in targets}=={t['id'],'secondary'}
        assert other['association_to_confirmed_CAD']['status'].startswith('unassigned')
        assert not other['association_to_confirmed_CAD']['persistent_alias']
        assert next(x for x in targets if x['id']==t['id'])['pose_valid']
        # The next frame cannot inherit an ignore list or publish the old pose
        # as a new recovery. It needs its own accepted current measurement.
        next_frame={**f,'frame_id':2,'timestamp':1.1};next_targets=[{**other,'frame_id':2,'timestamp':1.1}]
        runtime.update(next_frame,next_targets,{'secondary':m},t['id'])
        assert not runtime._current_non_vetoing_proposals
        assert [x['id'] for x in next_targets]==['secondary']
    finally:runtime.close()
