"""Offline phase acceptance and adversarial mutations, without Isaac/GPU."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from tools.copilot.check_sim_run import (AcceptanceConfig,check_directory,main,validate_run,_inactive_cdr_hash,
    UI_ACTION_POLICY,UI_ACTION_TICKS)
from teleop_assist.copilot.sim_control import G1_JOINT_NAMES


def pose(angle):
    a=math.radians(angle);c,s=math.cos(a),math.sin(a)
    return [[c,-s,0.,.32],[s,c,0.,-.14],[0.,0.,1.,.25],[0.,0.,0.,1.]]


def fixture(*,measured=False):
    rows=[]
    names=sorted(G1_JOINT_NAMES)
    for i in range(145):
        stamp=100.+i*.1
        consumed=5<=i<70 or 75<=i<94 or i>=113
        clutched=4<=i<70 or 74<=i<94 or i>=112
        confirmed=9<=i<70 or 79<=i<94 or i>=117
        held=12<=i<65 or 82<=i<94 or i>=120
        offset=0.
        if 12<=i<65:offset=min(14.,3.*(i-11))
        elif 65<=i<70:offset=max(0.,14.-3.*(i-64))
        elif 82<=i<94:offset=min(14.,3.*(i-81))
        elif i>=120:offset=3.
        error=14.
        if 12<=i<65:error=max(3.,14.-2.*(i-11))
        elif 65<=i<70:error=min(14.,3.+3.*(i-64))
        elif 82<=i<94:error=max(3.,14.-2.*(i-81))
        elif i>=94:error=3.
        reason='guarded_sim_command_available' if consumed else 'clutch_required'
        if i in (4,74,112):reason='clutched_waiting_for_next_XR_sample'
        elif i==70:reason='explicit_release_reclutch_required'
        elif 94<=i<=100:reason='operator_sample_stale_or_future'
        snapshot=dict(clutched=clutched,confirmed=confirmed,confirmed_candidate_id='grasp' if confirmed else None,
            selected_candidate_id='grasp' if clutched and i not in (4,5,6,74,75,76,112,113,114) else None,
            assist_held=held,orientation_offset_rad=math.radians(offset),input_age_s=.001,
            state='assisting' if held else 'releasing' if offset>0 else 'confirmed' if confirmed else 'manual',
            translation_assistance=False,finger_assistance=False,robot_command_sent=False)
        controller=dict(mode='isolated_PhysX',reason=reason,allowed=consumed,input_hold_required=not consumed,
            snapshot=snapshot,robot_transport_started=False,real_robot_authorized=False,downstream_ik_allowed=False,
            assistance_applied=offset>0,command_consumed_once=consumed,command_pending=False,
            human_source='scripted_test_fixture_NO_HEADSET',operator_source_sequence=i,
            finger_control='measured_q43_values_unchanged',operator_acquisition_timestamp=stamp,
            operator_expires_at=stamp+.15,ui_expires_at=stamp+.25,expires_at=stamp+.15,
            handoff_inputs_checked_at=stamp+.041,guard={'valid_until':stamp+.15},
            mapped_left_base_ee=pose(0.),
            mapped_right_base_ee=pose(3. if i>=112 else 14.),
            assisted_right_base_ee=pose((3. if i>=112 else 14.)-offset))
        row=dict(tick=i,command_consumed=consumed,endpoint_accepted=True,source_sequence=i,scene_error=None,
            orientation_error_deg=error,contact_force_n=0.,controller=controller,timing=dict(
                loop_started=stamp-.03,camera_preview_s=.01,endpoint_and_snapshot_s=.02,controller_and_guard_s=.04,
                source_recheck_s=.001,handoff_snapshot_s=.001,consume_s=.006,loop_s=.1))
        if measured:row.update(actual_left_base_ee=pose(0),actual_right_base_ee=pose(error),actual_joint_names=names,actual_q43=[0.]*43)
        rows.append(row)
    report=dict(completed=True,ticks=len(rows),commands_consumed=sum(r['command_consumed'] for r in rows),
        max_contact_force_n=0.,gravity_saturation_steps=0,scripted_operator=True,headset_verified=False,
        real_robot_tested=False,dds_imported=False,physical_contact_sensor_scope='right arm/hand/camera vs table and target',
        max_rss_start_kib=1000,max_rss_end_kib=1000)
    gc=[dict(generation=0,started=100.,finished=100.0001,elapsed_s=.0001,collected=0,uncollectable=0)]
    return rows,report,gc


def failures(result):return {c['id'] for c in result['checks'] if not c['passed']}


def state_only_fixture(*,live=False):
    rows,report,gc=fixture(measured=True)
    report.update(ui_action_policy=UI_ACTION_POLICY,live_ai=live)
    for row in rows:
        controller=row['controller'];controller['ui_action_policy']=UI_ACTION_POLICY
        if live:controller['snapshot']['model_source_id']='model-1'
        if row['tick'] in UI_ACTION_TICKS:
            row['command_consumed']=False
            controller.update(allowed=False,command_pending=False,command_consumed_once=False,
                q14_command=None,input_hold_required=True,reason='ui_action_waiting_for_fresh_control_sample')
    report['commands_consumed']=sum(row['command_consumed'] for row in rows)
    return rows,report,gc,dict(ui_action_policy=UI_ACTION_POLICY)


@pytest.mark.parametrize('live',[False,True])
def test_state_only_actions_keep_all_other_phases_and_require_110_commands(live):
    rows,report,gc,manifest=state_only_fixture(live=live)
    result=validate_run(rows,report,gc,source_manifest=manifest,config=AcceptanceConfig(require_measured_poses=True))
    assert result['accepted'],failures(result)
    assert report['commands_consumed']==110
    assert result['metrics']['ui_action_policy']==UI_ACTION_POLICY
    phase_check=next(c for c in result['checks'] if c['id']=='command_availability_matches_deliberate_phases')
    assert phase_check['optional_live_reselection_edges']==([8,78,116] if live else [])


@pytest.mark.parametrize('tick',sorted(UI_ACTION_TICKS))
def test_new_policy_never_treats_one_shot_motion_as_an_optional_live_edge(tick):
    rows,report,gc,manifest=state_only_fixture(live=True)
    rows[tick]['command_consumed']=True;report['commands_consumed']+=1
    result=validate_run(rows,report,gc,source_manifest=manifest)
    assert not result['accepted']
    assert {'command_availability_matches_deliberate_phases','blocked_phases_never_command',
        'one_shot_ui_actions_change_state_without_motion_offer'}<=failures(result)


@pytest.mark.parametrize('field,value',[('reason','clutch_required'),('allowed',True),
    ('command_pending',True),('q14_command',[0.]*14)])
def test_state_transition_must_record_its_reason_and_no_offer(field,value):
    rows,report,gc,manifest=state_only_fixture()
    rows[9]['controller'][field]=value
    result=validate_run(rows,report,gc,source_manifest=manifest)
    assert 'one_shot_ui_actions_change_state_without_motion_offer' in failures(result)


@pytest.mark.parametrize('fault',['report_missing','controller_missing','manifest_missing','manifest_policy_missing',
    'unknown_report','unknown_controller','unknown_manifest','null_report','post_physics_missing_policy'])
def test_new_policy_cannot_fall_back_to_legacy_on_missing_or_unknown_declarations(fault):
    rows,report,gc,manifest=state_only_fixture()
    if fault=='report_missing':report.pop('ui_action_policy')
    elif fault=='controller_missing':rows[0]['controller'].pop('ui_action_policy')
    elif fault=='manifest_missing':manifest=None
    elif fault=='manifest_policy_missing':manifest={}
    elif fault=='unknown_report':report['ui_action_policy']='future_policy'
    elif fault=='unknown_controller':rows[0]['controller']['ui_action_policy']='future_policy'
    elif fault=='unknown_manifest':manifest['ui_action_policy']='future_policy'
    elif fault=='null_report':report['ui_action_policy']=None
    elif fault=='post_physics_missing_policy':rows[0]['post_physics_controller']={}
    result=validate_run(rows,report,gc,source_manifest=manifest)
    assert not result['accepted']
    assert 'known_ui_action_policy_bound_to_every_status_and_manifest' in failures(result)
    assert 'ui_action_policy' not in result['metrics']


def test_manifest_alone_cannot_reclassify_unversioned_history():
    rows,report,gc=fixture()
    result=validate_run(rows,report,gc,source_manifest={'ui_action_policy':UI_ACTION_POLICY})
    assert not result['accepted']
    assert 'known_ui_action_policy_bound_to_every_status_and_manifest' in failures(result)


def test_new_policy_directory_requires_and_hashes_source_manifest(tmp_path):
    rows,report,gc,manifest=state_only_fixture()
    for name,value in [('rows.json',rows),('report.json',report),('gc_events.json',gc)]:
        (tmp_path/name).write_text(json.dumps(value))
    assert not check_directory(tmp_path)['accepted']
    path=tmp_path/'runtime_source_manifest.json';path.write_text(json.dumps(manifest))
    result=check_directory(tmp_path)
    assert result['accepted'],failures(result)
    assert 'runtime_source_manifest.json' in result['artifact_sha256']
    path.write_text('null')
    with pytest.raises(ValueError,match='source manifest must be a JSON object'):check_directory(tmp_path)


def test_complete_legacy_phases_pass_with_explicit_physical_measurement_gap():
    result=validate_run(*fixture())
    assert result['accepted'],failures(result)
    assert fixture()[1]['commands_consumed']==116
    assert result['metrics']['ui_action_policy']=='legacy_unversioned'
    assert result['metrics']['phase_commands']['stale']['consumed']==0
    assert not result['coverage']['physical_pose_drift_verified']
    assert result['coverage']['command_target_translation_verified']
    assert any('No measured EE/q43' in reason for reason in result['limitations'])


def test_strict_mode_refuses_to_invent_missing_measured_pose_trace():
    result=validate_run(*fixture(),config=AcceptanceConfig(require_measured_poses=True))
    assert not result['accepted'] and 'complete_measured_pose_and_joint_trace' in failures(result)


def test_complete_measured_trace_verifies_all_three_clutches_and_fingers():
    result=validate_run(*fixture(measured=True),config=AcceptanceConfig(require_measured_poses=True))
    assert result['accepted'],failures(result)
    assert result['coverage']['physical_pose_drift_verified'] and result['coverage']['measured_finger_drift_verified']


def test_post_release_reclutch_excursion_is_not_hidden_by_other_stable_clutches():
    rows,report,gc=fixture(measured=True)
    rows[75]['actual_right_base_ee']=pose(17.)
    rows[75]['orientation_error_deg']=17.
    result=validate_run(rows,report,gc)
    assert not result['accepted']
    assert 'post_release_reclutch_preserves_recorded_goal_error' in failures(result)
    assert 'post_release_right_measured_clutch_drift' in failures(result)
    assert 'initial_right_measured_clutch_drift' not in failures(result)
    assert 'post_stale_right_measured_clutch_drift' not in failures(result)


@pytest.mark.parametrize('side',['left','right'])
def test_actual_position_error_cannot_hide_behind_unchanged_command_translation(side):
    rows,report,gc=fixture(measured=True)
    rows[30][f'actual_{side}_base_ee'][0][3]+=.0021
    result=validate_run(rows,report,gc)
    assert f'{side}_measured_manual_position_accuracy' in failures(result)
    assert 'assistance_never_changes_command_translation' not in failures(result)
    assert not result['coverage']['physical_translation_accuracy_verified']


def test_reanchoring_cannot_hide_accumulated_physical_position_shift():
    rows,report,gc=fixture(measured=True)
    # Smooth drift, then an equally displaced manual reference: individually
    # small servo residuals must not conceal accumulated stationary-input drift.
    for i,r in enumerate(rows):
        shift=.0031*min(1,max(0,(i-20)/30))
        r['actual_right_base_ee'][0][3]+=shift
        r['controller']['mapped_right_base_ee'][0][3]+=shift
        r['controller']['assisted_right_base_ee'][0][3]+=shift
    result=validate_run(rows,report,gc)
    assert 'right_stationary_input_does_not_accumulate_translation' in failures(result)
    assert 'right_measured_manual_position_accuracy' not in failures(result)


@pytest.mark.parametrize('fault,expected',[
    ('stale_command','blocked_phases_never_command'),('automatic_restart','no_automatic_restart_after_source_recovers'),
    ('unconfirmed_hold','held_assistance_uses_confirmed_candidate'),('early_assist','confirmation_precedes_held_assistance'),
    ('candidate_swap','candidate_identity_never_auto_switches'),('late_handoff','consumption_finishes_within_original_source_lease'),
    ('false_lease_extension','consumption_finishes_within_original_source_lease'),('translation','assistance_never_changes_command_translation'),
    ('overrotation','assistance_orientation_correction_is_bounded'),('instant_fade','button_release_fades_with_bounded_rate_without_rearm'),
    ('not_fading','button_release_fades_with_bounded_rate_without_rearm'),('initial_drift','initial_clutch_preserves_recorded_goal_error'),
    ('reclutch_drift','post_stale_reclutch_preserves_recorded_goal_error'),('no_benefit','first_assistance_has_measurable_orientation_benefit'),
    ('contact','no_recorded_contacts_in_instrumented_scope'),('scene_loss','known_scene_stays_valid'),
    ('dds','scripted_simulation_not_hardware_or_headset'),('hardware_claim','scripted_simulation_not_hardware_or_headset'),
    ('human_claim','consumed_commands_have_guard_and_scripted_source'),('downstream_ik','no_robot_transport_or_downstream_ik_in_rows'),
    ('count_mismatch','report_counts_match_rows'),('missing_tick','complete_scripted_scenario'),
    ('source_replay','source_sequences_increase'),('gc_corrupt','gc_event_trace_consistent')])
def test_adversarial_phase_and_provenance_mutations_fail(fault,expected):
    rows,report,gc=fixture();c=rows[20]['controller']
    if fault=='stale_command':rows[96]['command_consumed']=True
    elif fault=='automatic_restart':rows[103]['controller']['snapshot']['clutched']=True
    elif fault=='unconfirmed_hold':c['snapshot']['confirmed']=False
    elif fault=='early_assist':rows[9]['controller']['assistance_applied']=True
    elif fault=='candidate_swap':c['snapshot']['confirmed_candidate_id']='other'
    elif fault=='late_handoff':rows[20]['timing']['consume_s']=.15
    elif fault=='false_lease_extension':c['operator_expires_at']+=1
    elif fault=='translation':c['assisted_right_base_ee'][0][3]+=.001
    elif fault=='overrotation':c['snapshot']['orientation_offset_rad']=math.radians(21)
    elif fault=='instant_fade':rows[65]['controller']['snapshot']['orientation_offset_rad']=0.
    elif fault=='not_fading':
        for i in range(65,70):rows[i]['controller']['snapshot']['orientation_offset_rad']=math.radians(14)
    elif fault=='initial_drift':rows[8]['orientation_error_deg']+=1.
    elif fault=='reclutch_drift':rows[114]['orientation_error_deg']+=1.
    elif fault=='no_benefit':
        for i in range(55,65):rows[i]['orientation_error_deg']=14.
    elif fault=='contact':rows[55]['contact_force_n']=.5
    elif fault=='scene_loss':rows[33]['scene_error']='target_moved'
    elif fault=='dds':report['dds_imported']=True
    elif fault=='hardware_claim':report['real_robot_tested']=True
    elif fault=='human_claim':c['human_source']='verified_Meta_Quest'
    elif fault=='downstream_ik':c['downstream_ik_allowed']=True
    elif fault=='count_mismatch':report['commands_consumed']-=1
    elif fault=='missing_tick':rows.pop(99)
    elif fault=='source_replay':rows[20]['source_sequence']=19
    elif fault=='gc_corrupt':gc[0]['finished']=99.
    result=validate_run(rows,report,gc)
    assert not result['accepted'] and expected in failures(result)


@pytest.mark.parametrize('fault,expected',[('position','initial_right_measured_clutch_drift'),
    ('orientation','post_stale_left_measured_clutch_drift'),('finger','measured_fingers_remain_fixed'),
    ('partial','complete_measured_pose_and_joint_trace'),('nonfinite','artifact_structure')])
def test_measured_physical_drift_is_checked_independently_of_scalar_goal_error(fault,expected):
    rows,report,gc=fixture(measured=True)
    if fault=='position':rows[6]['actual_right_base_ee'][0][3]+=.002
    elif fault=='orientation':rows[114]['actual_left_base_ee']=pose(1.)
    elif fault=='finger':
        index=next(i for i,n in enumerate(rows[0]['actual_joint_names']) if '_hand_' in n)
        rows[20]['actual_q43'][index]=.02
    elif fault=='partial':rows[20].pop('actual_right_base_ee')
    else:rows[5]['actual_q43'][0]=float('nan')
    result=validate_run(rows,report,gc)
    assert not result['accepted'] and expected in failures(result)


def test_real_scenario_index_skips_warmup_without_treating_it_as_clutch_script():
    rows,report,gc=fixture()
    warmup=[]
    for i in range(3):
        r=deepcopy(rows[0]);r.update(tick=i,scenario_tick=None if i<2 else -1,source_sequence=i)
        warmup.append(r)
    for i,r in enumerate(rows):
        r.update(scenario_tick=i,tick=i+3,source_sequence=i+3)
        r['controller']['operator_source_sequence']=i+3
    rows=warmup+rows;report.update(ticks=len(rows),scripted_scenario_completed=True)
    result=validate_run(rows,report,gc)
    assert result['accepted'],failures(result)
    assert result['metrics']['warmup_rows']==3 and result['metrics']['scenario_rows']==145
    report['scripted_scenario_completed']=False
    assert not validate_run(rows,report,gc)['accepted']


def test_live_scenario_requires_explicit_frozen_model_binding():
    rows,report,gc=fixture();report['live_ai']=True
    assert not validate_run(rows,report,gc)['accepted']
    for r in rows:r['controller']['snapshot']['model_source_id']='model-1'
    assert validate_run(rows,report,gc)['accepted']
    rows[25]['controller']['snapshot']['model_source_id']='model-2'
    assert not validate_run(rows,report,gc)['accepted']


def test_float32_matrix_transport_roundoff_does_not_create_fake_rotation():
    rows,report,gc=fixture()
    for r in rows:
        for key in ('mapped_right_base_ee','assisted_right_base_ee'):
            r['controller'][key]=np.asarray(r['controller'][key],np.float32).astype(float).tolist()
    assert validate_run(rows,report,gc)['accepted']


def test_cli_outputs_reproducible_artifact_hashes_and_nonzero_exit_on_failure(tmp_path):
    rows,report,gc=fixture()
    for name,value in [('rows.json',rows),('report.json',report),('gc_events.json',gc)]:
        (tmp_path/name).write_text(json.dumps(value))
    result=check_directory(tmp_path)
    assert result['accepted'] and set(result['artifact_sha256'])=={'rows.json','report.json','gc_events.json'}
    output=tmp_path/'acceptance.json'
    assert main([str(tmp_path),'--output',str(output)])==0
    assert json.loads(output.read_text())['artifact_sha256']==result['artifact_sha256']
    assert main([str(tmp_path),'--output',str(output),'--require-measured-poses'])==1
    (tmp_path/'report.json').write_text('{broken')
    assert main([str(tmp_path),'--output',str(output)])==1


def test_nonfinite_acceptance_threshold_is_not_an_accidental_bypass():
    with pytest.raises(ValueError):replace(AcceptanceConfig(),contact_tolerance_n=float('inf'))


def barrier_fixture(*,internal=True):
    rows,report,gc=fixture();report['live_ai']=True
    for row in rows:
        c=row['controller'];stamp=c['operator_acquisition_timestamp'];frame_stamp=stamp-.03
        identity=f'camera_cal:{row["tick"]}'
        c['clock_id']='host_clock';c['snapshot']['model_source_id']='frozen-model'
        c.update(proposal_observation_id=identity,proposal_capture_timestamp=frame_stamp)
        row['timing']['loop_started']-=.006;row['timing']['frame_barrier_s']=.006
        row['frame_barrier']=dict(schema='punktnull.sim_frame_barrier.v1',ready=True,
            reason='exact_current_frame_processed',frame_id=row['tick'],capture_timestamp=frame_stamp,
            clock_id='host_clock',calibration_id='camera_cal',expected_validation_observation_id=identity,
            timeout_s=.1,started_at=stamp-.005,elapsed_s=.004,processed_age_s=.029,
            motion_authorized=False,deadline_renewed=False)
        prepared=dict(phase='prepare_current_frame' if internal else 'prepare_endpoint',
            expected_clock_id='host_clock',expected_calibration_id='camera_cal',
            model_source_id='frozen-model',selected_candidate_id='grasp',
            validation_observation_id=identity,preview_capture_timestamp=frame_stamp)
        if not internal:
            prepared.update(validation_observation_id='camera_cal:older',preview_capture_timestamp=stamp-.13)
            c.update(proposal_observation_id='camera_cal:older',proposal_capture_timestamp=stamp-.13)
        row['prepared_preview_evidence']=prepared
        row['handoff_target_recheck']=dict(accepted=True,motion_authorized=False,deadline_renewed=False,
            preview_evidence=dict(validation_observation_id=identity,preview_capture_timestamp=frame_stamp,
                preview_clock_id='host_clock',expected_clock_id='host_clock',
                preview_calibration_id='camera_cal',expected_calibration_id='camera_cal',
                checked_at=stamp+.045,preview_age_s=.075,failed_preview_checks=[],motion_static=True,
                preview_valid=False,motion_authorized=False,deadline_renewed=False,
                model_source_id='frozen-model',selected_candidate_id='grasp'))
    return rows,report,gc


@pytest.mark.parametrize('internal',[False,True])
def test_live_barrier_accepts_both_recorded_phase_placements_without_retiming_old_batch(internal):
    result=validate_run(*barrier_fixture(internal=internal))
    assert result['accepted'],[x for x in result['checks'] if not x['passed']]
    assert result['coverage']['frame_barrier_evidence_verified']
    assert any('do not establish physical real-time' in x for x in result['limitations'])


@pytest.mark.parametrize('fault',[
    'not_ready','missing','list','bool_ready','clock','calibration','observation_id','capture',
    'renewed','authority','too_late','invalid_timeout','missing_duration','handoff_missing',
    'handoff_clock','handoff_calibration','handoff_frame','handoff_capture','handoff_renewed',
    'handoff_age','handoff_changed_model','handoff_changed_candidate','new_batch_old_capture',
    'new_offer_old_capture','outside_phase'])
def test_live_barrier_ack_cannot_mask_wrong_source_permission_or_failed_handoff(fault):
    rows,report,gc=barrier_fixture();row=rows[20];b=row['frame_barrier'];h=row['handoff_target_recheck']
    if fault=='not_ready':b['ready']=False
    elif fault=='missing':row.pop('frame_barrier')
    elif fault=='list':row['frame_barrier']=[]
    elif fault=='bool_ready':b['ready']=1
    elif fault=='clock':b['clock_id']='other_clock'
    elif fault=='calibration':b['calibration_id']='other_calibration'
    elif fault=='observation_id':b['expected_validation_observation_id']='camera_cal:19'
    elif fault=='capture':b['capture_timestamp']-=.001
    elif fault=='renewed':b['deadline_renewed']=True
    elif fault=='authority':b['motion_authorized']=True
    elif fault=='too_late':b['elapsed_s']=.11
    elif fault=='invalid_timeout':b['timeout_s']=1.
    elif fault=='missing_duration':row['timing'].pop('frame_barrier_s')
    elif fault=='handoff_missing':row['handoff_target_recheck']=None
    elif fault=='handoff_clock':h['preview_evidence']['preview_clock_id']='wrong'
    elif fault=='handoff_calibration':h['preview_evidence']['preview_calibration_id']='wrong'
    elif fault=='handoff_frame':h['preview_evidence']['validation_observation_id']='camera_cal:19'
    elif fault=='handoff_capture':h['preview_evidence']['preview_capture_timestamp']-=.001
    elif fault=='handoff_renewed':h['deadline_renewed']=True
    elif fault=='handoff_age':h['preview_evidence']['preview_age_s']=.26
    elif fault=='handoff_changed_model':h['preview_evidence']['model_source_id']='new-model'
    elif fault=='handoff_changed_candidate':h['preview_evidence']['selected_candidate_id']='other-grasp'
    elif fault=='new_batch_old_capture':row['prepared_preview_evidence']['preview_capture_timestamp']-=.1
    elif fault=='new_offer_old_capture':row['controller']['proposal_capture_timestamp']-=.1
    elif fault=='outside_phase':b['started_at']-=.1
    result=validate_run(rows,report,gc)
    assert not result['accepted']
    assert 'live_frame_barrier_and_final_recheck_preserve_capture_and_authority' in failures(result)


def test_barrier_wait_must_be_counted_in_original_handoff_lease():
    rows,report,gc=barrier_fixture()
    row=rows[20];row['timing'].update(frame_barrier_s=.099,consume_s=.02)
    result=validate_run(rows,report,gc)
    assert 'consumption_finishes_within_original_source_lease' in failures(result)
    assert 'live_frame_barrier_and_final_recheck_preserve_capture_and_authority' not in failures(result)


def test_failed_barrier_without_authorization_is_a_valid_recorded_blocked_phase():
    rows,report,gc=barrier_fixture();row=rows[96]
    row['endpoint_accepted']=False
    row['frame_barrier'].update(ready=False,reason='current_frame_preview_timeout')
    row['handoff_target_recheck']=None
    result=validate_run(rows,report,gc)
    assert result['accepted'],failures(result)


def test_historical_live_without_barrier_field_does_not_acquire_new_coverage_or_failure():
    rows,report,gc=fixture();report['live_ai']=True
    for row in rows:row['controller']['snapshot']['model_source_id']='frozen-model'
    result=validate_run(rows,report,gc)
    assert result['accepted'],failures(result)
    assert not result['coverage']['frame_barrier_evidence_verified']


@pytest.mark.parametrize('fault,expected', [(None,None),('confirmation','held_assistance_uses_confirmed_candidate'),
    ('identity','held_assistance_uses_confirmed_candidate'),('lease','consumption_finishes_within_original_source_lease'),
    ('command','command_availability_matches_deliberate_phases'),('binding','live_model_binding_stays_explicit_and_frozen')])
def test_extended_run_checks_every_late_command_and_confirmation(fault,expected):
    rows,report,gc=fixture(measured=True)
    for i in range(145,1000):
        row=deepcopy(rows[144]);shift=(i-144)*.1
        row.update(tick=i,source_sequence=i)
        row['timing']['loop_started']+=shift
        controller=row['controller'];controller['operator_source_sequence']=i
        for key in ('operator_acquisition_timestamp','operator_expires_at','ui_expires_at','expires_at','handoff_inputs_checked_at'):
            controller[key]+=shift
        controller['guard']['valid_until']+=shift
        rows.append(row)
    report.update(ticks=len(rows),commands_consumed=sum(r['command_consumed'] for r in rows),live_ai=True)
    for row in rows:row['controller']['snapshot']['model_source_id']='frozen-model'
    c=rows[998]['controller']
    if fault=='confirmation':c['snapshot']['confirmed']=False
    elif fault=='identity':c['snapshot']['confirmed_candidate_id']=None
    elif fault=='lease':rows[998]['timing']['consume_s']=.2
    elif fault=='command':rows[998]['command_consumed']=False
    elif fault=='binding':c['snapshot'].pop('model_source_id')
    result=validate_run(rows,report,gc)
    if fault is None:assert result['accepted'],failures(result)
    else:assert not result['accepted'] and expected in failures(result)


def actor_fixture(*,codec=True):
    rows,report,gc=fixture(measured=True);fmt='unitree_float32' if codec else 'float64'
    manifest=dict(scope='inactive_simulator_q14_carrier_only',hardware_authorized=False,
        transport_created=False,arm_slots=list(range(15,29)),format='XCDR1_little_endian',
        cdr_header_hex='00010000',cdr_length=1004,motor_mode=0,kp=0.,kd=0.,dq=0.,tau=0.,non_arm_q=0.,
        IDL_source_hashes={
            '/sdk/_LowCmd_.py':'a23cec88e4bf782263ac2bea29d3908c9d21a44a14361ffc0bf3424cd36597ae',
            '/sdk/_MotorCmd_.py':'710a8321be4d6731093b6f8328c70ab11606372561ed01a7af9a187143f54c4d'},
        CRC=dict(sha256='b95a530423f72c5acc96811f75677699cb3a185b6327353e7c44b355a425a20e',
            original_methods_unmodified=True,CRC_constructor_called=False,backend='original_python_polynomial',
            SDK_HG_payload_length=1004,original_methods=['Crc','__Crc32','__PackHGLowCmd','__Trans','_crc_py']))
    report.update(command_format=fmt,command_codec=manifest if codec else None,DDS_transport_started=False,
        transport_modules_loaded=[],dds_imported=codec)
    q=[.125]*14;payload_hash=_inactive_cdr_hash(q)
    for row in rows:
        c=row['controller'];stamp=c['operator_acquisition_timestamp'];sequence=row['tick']+1
        actor=dict(applied=row['command_consumed'],command_format=fmt,DDS_transport_started=False,hardware_authorized=False)
        if row['command_consumed']:
            c.update(command_token=f'fixture_session:1:{sequence}',receipt_inputs_checked_at=stamp+.051)
            c['guard']['sequence']=sequence
            actor.update(receipt_valid_until=c['expires_at'],applied_valid_until=c['expires_at'],
                receipt_token=c['command_token'],guard_sequence=sequence,consumed_snapshot_sha256='a'*64,
                guard_snapshot_sha256='a'*64,checked_before_PhysX_handoff_at=stamp+.052)
            if codec:actor['codec']=dict(accepted=True,bytes=1004,sha256=payload_hash,
                q14_before=list(q),q14_after=list(q),scope='local_Unitree_IDL_CDR_roundtrip_no_DDS_transport')
        row['final_actor']=actor
    return rows,report,gc


@pytest.mark.parametrize('codec',[False,True])
def test_final_receipt_trace_is_accepted_with_honest_IDL_import_and_transport_scope(codec):
    result=validate_run(*actor_fixture(codec=codec),config=AcceptanceConfig(require_measured_poses=True))
    assert result['accepted'],failures(result)
    assert result['coverage']['final_receipt_handoff_evidence_verified']
    assert result['coverage']['no_DDS_transport_manifest_verified']
    assert result['coverage']['unitree_CDR_roundtrip_evidence_verified']==codec


def test_legacy_without_final_actor_never_acquires_receipt_or_CDR_coverage():
    r=validate_run(*fixture())
    assert r['accepted'] and not r['coverage']['final_receipt_handoff_evidence_verified']
    assert not r['coverage']['unitree_CDR_roundtrip_evidence_verified']
    assert any('Legacy run has no final_actor' in item for item in r['limitations'])


@pytest.mark.parametrize('fault',[None,'missing','expired','extended','token','sequence','skipped_generation',
    'wrong_source','early_check','lost_command','idle_command','claims_hardware','shutdown_hold_failed'])
def test_original_sender_requires_all_five_owned_physics_interlocks(fault):
    rows,report,gc=actor_fixture()
    report.update(sender='guarded_original',sender_close_error=None,sender_manifest=dict(
        sha256='793e0774bb0a7a29b0cd163336babd45b29a36c3518528c87a854969c4132f73',
        publisher_override_installed_before_constructor=True,original_class_bodies_unmodified=True,
        original_module_imported=False,DDS_transport_started=False,hardware_authorized=False,
        stock_XR_application_started=False,complete_XR_entry_integration_verified=False,
        original_self_msg_used_for_commands=False,
        runtime_scope='original_controller_shim_in_isolated_complete_PhysX_scene'))
    for row in rows:
        actor=row['final_actor'];stamp=row['controller']['operator_acquisition_timestamp']
        actor.update(physics_generation=row['tick']*5,source_instance_id='owned-physics')
        actor['physics_step_interlocks']=[dict(accepted=row['command_consumed'],
            checked_at=stamp+.053+step*.001,valid_until=actor.get('applied_valid_until'),
            receipt_token=actor.get('receipt_token'),guard_sequence=actor.get('guard_sequence'),
            source_instance_id='owned-physics',physics_generation=row['tick']*5+step,
            scope='single_owner_before_PhysX_write_data_to_sim',DDS_transport_started=False,hardware_authorized=False)
            for step in range(5)]
    actor=rows[12]['final_actor'];entry=actor['physics_step_interlocks'][2]
    if fault=='missing':actor['physics_step_interlocks'].pop()
    elif fault=='expired':entry['valid_until']=entry['checked_at']
    elif fault=='extended':entry['valid_until']+=.001
    elif fault=='token':entry['receipt_token']='other'
    elif fault=='sequence':entry['guard_sequence']+=1
    elif fault=='skipped_generation':entry['physics_generation']+=1
    elif fault=='wrong_source':entry['source_instance_id']='other'
    elif fault=='early_check':entry['checked_at']=actor['checked_before_PhysX_handoff_at']-.01
    elif fault=='lost_command':entry['accepted']=False
    elif fault=='idle_command':rows[0]['final_actor']['physics_step_interlocks'][0]['accepted']=True
    elif fault=='claims_hardware':report['sender_manifest']['hardware_authorized']=True
    elif fault=='shutdown_hold_failed':report['sender_close_error']='nonfinite_measured_state'
    result=validate_run(rows,report,gc,config=AcceptanceConfig(require_measured_poses=True))
    if fault is None:
        assert result['accepted'],failures(result)
        assert result['coverage']['guarded_original_simulator_receiver_verified']
    else:
        assert not result['accepted']
        assert 'original_controller_owned_receiver_and_every_physics_interlock' in failures(result)


@pytest.mark.parametrize('fault',[
    'missing_actor','missing_all_actors','missing_codec','codec_rejected','applied_flag','format',
    'transport','hardware','receipt_extended','final_extended','final_different_controller',
    'late_apply','before_consume','before_recheck','after_loop','missing_final_clock',
    'token','sequence','replayed_token','replayed_sequence','snapshot_sha','malformed_sha',
    'codec_valid_but_wrong_sha','codec_malformed_sha','codec_bytes','codec_scope',
    'q_changed','both_q_changed','q_not_float32','q_bool','q_short','q_negative_zero',
    'missing_receipt_deadline','missing_source_limit'])
def test_final_actor_negative_evidence_cannot_claim_a_checked_application(fault):
    rows,report,gc=actor_fixture();row=rows[20];a=row['final_actor'];c=row['controller'];codec=a['codec']
    if fault=='missing_actor':row.pop('final_actor')
    elif fault=='missing_all_actors':
        for r in rows:r.pop('final_actor')
    elif fault=='missing_codec':a.pop('codec')
    elif fault=='codec_rejected':codec['accepted']=False
    elif fault=='applied_flag':a['applied']=False
    elif fault=='format':a['command_format']='float64'
    elif fault=='transport':a['DDS_transport_started']=True
    elif fault=='hardware':a['hardware_authorized']=True
    elif fault=='receipt_extended':a['receipt_valid_until']+=1.
    elif fault=='final_extended':a['applied_valid_until']+=1.
    elif fault=='final_different_controller':a['applied_valid_until']-=.001
    elif fault=='late_apply':a['checked_before_PhysX_handoff_at']=a['applied_valid_until']
    elif fault=='before_consume':a['checked_before_PhysX_handoff_at']=c['operator_acquisition_timestamp']
    elif fault=='before_recheck':a['checked_before_PhysX_handoff_at']=c['receipt_inputs_checked_at']-.001
    elif fault=='after_loop':a['checked_before_PhysX_handoff_at']=row['timing']['loop_started']+row['timing']['loop_s']+.001
    elif fault=='missing_final_clock':a.pop('checked_before_PhysX_handoff_at')
    elif fault=='token':a['receipt_token']='unknown'
    elif fault=='sequence':a['guard_sequence']+=1
    elif fault=='replayed_token':a['receipt_token']=c['command_token']=rows[19]['final_actor']['receipt_token']
    elif fault=='replayed_sequence':a['guard_sequence']=c['guard']['sequence']=rows[19]['final_actor']['guard_sequence']
    elif fault=='snapshot_sha':a['guard_snapshot_sha256']='b'*64
    elif fault=='malformed_sha':a['guard_snapshot_sha256']=a['consumed_snapshot_sha256']='not_a_digest'
    elif fault=='codec_valid_but_wrong_sha':codec['sha256']='b'*64
    elif fault=='codec_malformed_sha':codec['sha256']='bad'
    elif fault=='codec_bytes':codec['bytes']=1008
    elif fault=='codec_scope':codec['scope']='DDS_hardware'
    elif fault=='q_changed':codec['q14_after'][0]=.25
    elif fault=='both_q_changed':codec['q14_before'][0]=codec['q14_after'][0]=.25
    elif fault=='q_not_float32':codec['q14_before'][0]=codec['q14_after'][0]=.1
    elif fault=='q_bool':codec['q14_before'][0]=codec['q14_after'][0]=True
    elif fault=='q_short':codec['q14_before'].pop();codec['q14_after'].pop()
    elif fault=='q_negative_zero':codec['q14_before'][0]=0.;codec['q14_after'][0]=-0.
    elif fault=='missing_receipt_deadline':a.pop('receipt_valid_until')
    else:c.pop('operator_expires_at')
    r=validate_run(rows,report,gc)
    assert not r['accepted']
    assert (('final_actor_tokens_and_guard_sequences_are_not_replayed' if fault.startswith('replayed_') else
        'every_applied_command_has_exact_receipt_codec_and_final_deadline_evidence') in failures(r))
    assert not r['coverage']['unitree_CDR_roundtrip_evidence_verified']


@pytest.mark.parametrize('fault',['transport','loaded_transport','missing_transport_audit','false_import',
    'no_codec_manifest','codec_scope','active_motor','active_gain','CRC_hash','IDL_hash','format','header'])
def test_new_report_requires_explicit_no_transport_and_the_exact_inactive_codec_manifest(fault):
    rows,report,gc=actor_fixture();m=report['command_codec']
    if fault=='transport':report['DDS_transport_started']=True
    elif fault=='loaded_transport':report['transport_modules_loaded']=['unitree_sdk2py.core.channel']
    elif fault=='missing_transport_audit':report.pop('transport_modules_loaded')
    elif fault=='false_import':report['dds_imported']=False
    elif fault=='no_codec_manifest':report['command_codec']=None
    elif fault=='codec_scope':m['scope']='hardware_ready'
    elif fault=='active_motor':m['motor_mode']=1
    elif fault=='active_gain':m['kp']=40.
    elif fault=='CRC_hash':m['CRC']['sha256']='a'*64
    elif fault=='IDL_hash':m['IDL_source_hashes']['/sdk/_LowCmd_.py']='a'*64
    elif fault=='format':report['command_format']='float16'
    else:m['cdr_header_hex']='00030000'
    r=validate_run(rows,report,gc)
    assert not r['accepted'] and 'final_actor_transport_and_codec_manifest' in failures(r)


def test_postencoding_receipt_can_shorten_but_never_extend_earlier_source_deadlines():
    rows,report,gc=actor_fixture();row=rows[20];c=row['controller'];a=row['final_actor']
    old=a['receipt_valid_until'];a['applied_valid_until']=c['expires_at']=old-.01
    r=validate_run(rows,report,gc)
    assert r['accepted'],failures(r)
    assert a['receipt_valid_until']==old


def test_independent_carrier_reconstruction_matches_real_recorded_Unitree_IDL_payload():
    # Golden first applied q14/hash from copilot_sim_codec_001, produced by the
    # actual pinned Unitree/Cyclone IDL encoder rather than this checker.
    q=[.6393718719482422,.2656629979610443,.06890416890382767,.011597912758588791,
       .0763719454407692,.0018246122635900974,-.07114741206169128,-.6571518778800964,
       -.3943840265274048,.3298664093017578,-.11722542345523834,-1.335355281829834,
       .6149327158927917,.7454424500465393]
    assert _inactive_cdr_hash(q)=='bcd793094da389cfde7e1f869ba14867ddd404c626bbef814d7cfb4cd72895f7'
