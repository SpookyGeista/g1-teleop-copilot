#!/usr/bin/env python3
"""Reproduce acceptance of the scripted, isolated orientation-control scenario.

This audits recorded evidence, not hardware safety or general task success.
Legacy recordings without measured EE/q43 data are explicitly incomplete for
physical drift; --require-measured-poses makes that missing coverage a failure.
No simulator, GPU, network, robot transport, or model is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

UI_ACTION_POLICY = 'state_only_with_fresh_control_v1'
UI_ACTION_TICKS = frozenset((7, 9, 77, 79, 115, 117))


@dataclass(frozen=True)
class AcceptanceConfig:
    initial_goal_error_drift_deg: float = .5
    clutch_position_drift_m: float = .001
    clutch_orientation_drift_deg: float = .5
    settled_orientation_error_deg: float = 6.
    minimum_assist_improvement_fraction: float = .30
    minimum_initial_challenge_deg: float = 3.
    contact_tolerance_n: float = 1e-6
    max_finger_drift_rad: float = .01
    # Engineering budgets for this stationary-human orientation fixture. These
    # are physical post-step errors, separate from the solver's 0.5-mm FK limit.
    maximum_manual_position_error_m: float = .002
    maximum_stationary_position_excursion_m: float = .003
    require_measured_poses: bool = False

    def __post_init__(self):
        numeric=[value for name,value in asdict(self).items() if name!='require_measured_poses']
        if (not all(_number(value) and value>=0 for value in numeric) or
                not 0<=self.minimum_assist_improvement_fraction<=1 or type(self.require_measured_poses) is not bool):
            raise ValueError('Acceptance thresholds must be finite, nonnegative and explicit')


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _pose(value):
    a=np.asarray(value,dtype=float)
    if (a.shape!=(4,4) or not np.isfinite(a).all() or
            not np.allclose(a[3],[0,0,0,1],atol=1e-8,rtol=0) or
            not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-5,rtol=0) or
            abs(np.linalg.det(a[:3,:3])-1)>1e-5):
        raise ValueError('Missing or invalid measured/mapped rigid pose')
    return a


def _angle(a,b):
    # Logged float32-derived matrices are checked by _pose first. Project their
    # transport roundoff onto SO(3); otherwise acos(R @ R.T) can report a false
    # nonzero correction even when the two stored matrices are identical.
    rotations=[]
    for pose in (a,b):
        u,_,vt=np.linalg.svd(pose[:3,:3])
        rotations.append(u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt)
    cosine=(np.trace(rotations[0]@rotations[1].T)-1)/2
    return math.degrees(math.acos(float(np.clip(cosine,-1,1))))


def _time(row):
    c=row['controller'];stamp=c.get('operator_acquisition_timestamp');age=c['snapshot'].get('input_age_s')
    if not _number(stamp) or not _number(age):raise ValueError('Controller source time missing')
    return stamp+age


def _finite_json(value):
    try:json.dumps(value,allow_nan=False);return True
    except (ValueError,TypeError,OverflowError):return False


_CODEC_SCOPE='local_Unitree_IDL_CDR_roundtrip_no_DDS_transport'
_IDL_HASHES={'_LowCmd_.py':'a23cec88e4bf782263ac2bea29d3908c9d21a44a14361ffc0bf3424cd36597ae',
             '_MotorCmd_.py':'710a8321be4d6731093b6f8328c70ab11606372561ed01a7af9a187143f54c4d'}
_CRC_HASH='b95a530423f72c5acc96811f75677699cb3a185b6327353e7c44b355a425a20e'


def _sha256_value(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _exact_q14_bytes(values):
    if not isinstance(values,list) or len(values)!=14 or not all(_number(v) for v in values):
        raise ValueError('codec_q14_requires_fourteen_finite_numbers')
    packed=struct.pack('<14f',*values)
    if tuple(values)!=struct.unpack('<14f',packed):raise ValueError('codec_q14_not_exact_float32')
    return packed


def _inactive_cdr_hash(q14):
    """Independent fixed-carrier reconstruction, without importing any SDK.

    Reviewed HG CRC input is 1004 bytes: <2B2x + 35*(B3x5fI) + 5I.
    All carrier fields are zero except arm q14 and CRC. In the reviewed XCDR1
    representation, the encapsulation replaces the first four zero bytes;
    the q slots and final CRC therefore retain the same absolute offsets.
    This equivalence is specific to the inactive carrier, not generic LowCmd.
    """
    wire=_exact_q14_bytes(q14);payload=bytearray(1004)
    for i in range(14):payload[8+(15+i)*28:12+(15+i)*28]=wire[i*4:i*4+4]
    crc=0xffffffff
    for word in struct.unpack('<250I',payload[:-4]):
        crc^=word
        for _ in range(32):crc=((crc<<1)^ (0x04c11db7 if crc&0x80000000 else 0))&0xffffffff
    struct.pack_into('<I',payload,1000,crc)
    payload[:4]=b'\x00\x01\x00\x00'
    return hashlib.sha256(payload).hexdigest()


def _codec_manifest_valid(manifest):
    if not isinstance(manifest,dict):return False
    crc=manifest.get('CRC');sources=manifest.get('IDL_source_hashes')
    return bool(manifest.get('scope')=='inactive_simulator_q14_carrier_only' and
        manifest.get('hardware_authorized') is False and manifest.get('transport_created') is False and
        manifest.get('arm_slots')==list(range(15,29)) and manifest.get('format')=='XCDR1_little_endian' and
        manifest.get('cdr_header_hex')=='00010000' and type(manifest.get('cdr_length')) is int and
        manifest['cdr_length']==1004 and all(_number(manifest.get(k)) and manifest[k]==0 for k in
            ('motor_mode','kp','kd','dq','tau','non_arm_q')) and
        isinstance(sources,dict) and len(sources)==2 and
        {Path(k).name:v for k,v in sources.items() if isinstance(k,str)}==_IDL_HASHES and
        isinstance(crc,dict) and crc.get('sha256')==_CRC_HASH and
        crc.get('original_methods_unmodified') is True and crc.get('CRC_constructor_called') is False and
        crc.get('backend')=='original_python_polynomial' and crc.get('SDK_HG_payload_length')==1004 and
        crc.get('original_methods')==['Crc','__Crc32','__PackHGLowCmd','__Trans','_crc_py'])


def _actor_evidence_issues(row,command_format):
    issues=[];actor=row.get('final_actor');c=row['controller'];applied=row['command_consumed']
    try:
        if (not isinstance(actor,dict) or type(actor.get('applied')) is not bool or
                actor['applied']!=applied):raise ValueError('missing_or_inconsistent_final_actor')
        if (actor.get('command_format')!=command_format or actor.get('DDS_transport_started') is not False or
                actor.get('hardware_authorized') is not False):raise ValueError('actor_format_or_transport_scope_invalid')
        codec=actor.get('codec')
        if command_format=='float64' and codec is not None:raise ValueError('unexpected_codec_in_float64_path')
        if command_format=='unitree_float32' and (applied or codec is not None):
            if (not isinstance(codec,dict) or codec.get('accepted') is not True or codec.get('scope')!=_CODEC_SCOPE or
                    type(codec.get('bytes')) is not int or codec['bytes']!=1004 or not _sha256_value(codec.get('sha256'))):
                raise ValueError('codec_roundtrip_evidence_missing_or_invalid')
            if _exact_q14_bytes(codec.get('q14_before'))!=_exact_q14_bytes(codec.get('q14_after')):
                raise ValueError('codec_changed_exact_float32_q14')
            if _inactive_cdr_hash(codec['q14_before'])!=codec['sha256']:
                raise ValueError('codec_payload_hash_does_not_match_recorded_q14')
        if not applied:return issues
        original,deadline,checked=(actor.get(k) for k in
            ('receipt_valid_until','applied_valid_until','checked_before_PhysX_handoff_at'))
        original_limits=[c['guard']['valid_until'],c['operator_expires_at'],c['ui_expires_at']]
        receipt_checked=c.get('receipt_inputs_checked_at');end=row['timing']['loop_started']+row['timing']['loop_s']
        if not all(_number(v) for v in [original,deadline,checked,receipt_checked,end,c['expires_at'],*original_limits]):
            raise ValueError('actor_deadline_evidence_missing')
        if (not deadline<=original<=min(original_limits) or deadline!=c['expires_at'] or
                not _handoff_time(row)-1e-6<=receipt_checked<=checked<deadline or checked>end+1e-6):
            raise ValueError('actor_applied_outside_private_original_deadline')
        if (not isinstance(actor.get('receipt_token'),str) or not actor['receipt_token'] or
                actor['receipt_token']!=c.get('command_token') or type(actor.get('guard_sequence')) is not int or
                actor['guard_sequence']<1 or actor['guard_sequence']!=c['guard'].get('sequence')):
            raise ValueError('actor_receipt_token_or_guard_sequence_mismatch')
        if (not _sha256_value(actor.get('guard_snapshot_sha256')) or
                actor.get('consumed_snapshot_sha256')!=actor['guard_snapshot_sha256']):
            raise ValueError('actor_original_and_consumed_snapshot_binding_mismatch')
    except (ValueError,KeyError,TypeError,AttributeError,IndexError,OverflowError,struct.error) as exc:
        issues.append(str(exc))
    return issues


def _handoff_time(row):
    timing=row['timing']
    required=('loop_started','camera_preview_s','endpoint_and_snapshot_s','controller_and_guard_s',
              'source_recheck_s','handoff_snapshot_s','consume_s')
    # New recordings split the wait from controller/guard work. Omitting it
    # would make an expired command appear to have been consumed earlier.
    if 'frame_barrier' in row or 'frame_barrier_s' in timing:
        required+=('frame_barrier_s',)
    if any(not _number(timing.get(key)) or timing[key]<0 for key in required):
        raise ValueError('Incomplete handoff timing')
    return sum(timing[key] for key in required)


def _original_sender_evidence_issues(row):
    """Verify the recorded owned receiver and every physics substep lease."""
    try:
        actor=row['final_actor'];ledger=actor.get('physics_step_interlocks')
        if not isinstance(ledger,list) or len(ledger)!=5:
            raise ValueError('five_original_sender_physics_interlocks_required')
        applied=row['command_consumed']
        previous_time=actor.get('checked_before_PhysX_handoff_at') if applied else row['timing']['loop_started']
        deadline=actor.get('applied_valid_until')
        generations=[]
        for entry in ledger:
            if (not isinstance(entry,dict) or entry.get('scope')!='single_owner_before_PhysX_write_data_to_sim'
                    or entry.get('DDS_transport_started') is not False or entry.get('hardware_authorized') is not False
                    or type(entry.get('accepted')) is not bool or not _number(entry.get('checked_at'))
                    or not _number(previous_time) or entry['checked_at']<previous_time
                    or type(entry.get('physics_generation')) is not int or entry['physics_generation']<0):
                raise ValueError('original_sender_physics_interlock_invalid')
            if entry['accepted'] is not applied:
                raise ValueError('installed_command_not_retained_through_all_physics_steps_or_idle_commanded')
            if applied:
                if (not _number(entry.get('valid_until')) or not _number(deadline)
                        or not entry['checked_at']<entry['valid_until']<=deadline
                        or entry.get('receipt_token')!=actor.get('receipt_token')
                        or entry.get('guard_sequence')!=actor.get('guard_sequence')
                        or entry.get('source_instance_id')!=actor.get('source_instance_id')):
                    raise ValueError('original_sender_physics_lease_extended_expired_or_rebound')
                deadline=entry['valid_until']
            generations.append(entry['physics_generation']);previous_time=entry['checked_at']
        if generations!=list(range(generations[0],generations[0]+5)):
            raise ValueError('original_sender_physics_generations_not_contiguous')
        if applied and generations[0]!=actor.get('physics_generation'):
            raise ValueError('original_sender_physics_did_not_start_at_consumed_sample')
        if previous_time>row['timing']['loop_started']+row['timing']['loop_s']+1e-6:
            raise ValueError('original_sender_physics_checks_after_recorded_loop')
        return []
    except (ValueError,KeyError,TypeError,AttributeError,IndexError) as error:
        return [str(error)]


def _barrier_evidence_issues(row):
    """Check recorded scheduling evidence, never turn the ACK into permission."""
    issues=[];barrier=row.get('frame_barrier');controller=row['controller']
    needed=row.get('endpoint_accepted') is True or row['command_consumed']
    if needed and (not isinstance(barrier,dict) or barrier.get('ready') is not True):
        issues.append('accepted_without_ready_barrier')
    if barrier is None:return issues
    try:
        if not isinstance(barrier,dict):raise ValueError('barrier_not_mapping')
        if (barrier.get('schema')!='punktnull.sim_frame_barrier.v1' or
                type(barrier.get('ready')) is not bool or barrier.get('motion_authorized') is not False or
                barrier.get('deadline_renewed') is not False):raise ValueError('barrier_scope_or_flags')
        stamp,start,elapsed,timeout=(barrier.get(key) for key in
            ('capture_timestamp','started_at','elapsed_s','timeout_s'))
        if (not all(_number(value) for value in (stamp,start,elapsed,timeout)) or
                not 0<timeout<=.1 or not 0<=elapsed or not stamp<=start):
            raise ValueError('barrier_timing_invalid')
        frame_id=barrier.get('frame_id');clock_id=barrier.get('clock_id');calibration=barrier.get('calibration_id')
        if (type(frame_id) is not int or frame_id<0 or not isinstance(clock_id,str) or not clock_id or
                not isinstance(calibration,str) or not calibration or
                barrier.get('expected_validation_observation_id')!=f'{calibration}:{frame_id}' or
                controller.get('clock_id')!=clock_id):raise ValueError('barrier_capture_identity_invalid')
        prepared=row.get('prepared_preview_evidence') or {}
        if (prepared.get('expected_clock_id')!=clock_id or
                prepared.get('expected_calibration_id')!=calibration):
            raise ValueError('barrier_different_validation_frame_convention')
        timing=row['timing'];prepare_started=timing['loop_started']+timing['camera_preview_s']
        wait=timing.get('frame_barrier_s')
        # Synchronization may run inside prepare(), overlapping perception with
        # endpoint computation, or immediately after it in older recordings.
        # The split durations add to the same pre-command interval either way.
        if (not _number(wait) or wait<0 or not _number(timing.get('endpoint_and_snapshot_s')) or
                timing['endpoint_and_snapshot_s']<0 or start<prepare_started-1e-6 or
                elapsed>wait+1e-6 or start+elapsed>
                prepare_started+timing['endpoint_and_snapshot_s']+wait+1e-6):
            raise ValueError('barrier_phase_timing_mismatch')
        if barrier['ready']:
            age=barrier.get('processed_age_s')
            if (barrier.get('reason')!='exact_current_frame_processed' or elapsed>=timeout or
                    not _number(age) or abs(age-(start+elapsed-stamp))>1e-6 or not 0<=age<=.25):
                raise ValueError('barrier_ready_age_invalid')
        if barrier['ready'] and needed and prepared.get('phase')=='prepare_current_frame':
            if (prepared.get('validation_observation_id')!=barrier['expected_validation_observation_id'] or
                    prepared.get('preview_capture_timestamp')!=stamp):
                raise ValueError('new_authorization_not_bound_to_acknowledged_capture')
        if row['command_consumed']:
            recheck=row.get('handoff_target_recheck') or {};evidence=recheck.get('preview_evidence') or {}
            if (recheck.get('accepted') is not True or recheck.get('motion_authorized') is not False or
                    recheck.get('deadline_renewed') is not False or
                    evidence.get('motion_authorized') is not False or evidence.get('deadline_renewed') is not False):
                raise ValueError('barrier_final_recheck_missing_or_authority_changed')
            if (evidence.get('validation_observation_id')!=barrier['expected_validation_observation_id'] or
                    evidence.get('preview_capture_timestamp')!=stamp or
                    evidence.get('preview_clock_id')!=clock_id or evidence.get('expected_clock_id')!=clock_id or
                    evidence.get('preview_calibration_id')!=calibration or
                    evidence.get('expected_calibration_id')!=calibration):
                raise ValueError('barrier_final_recheck_different_capture')
            checked=evidence.get('checked_at');age=evidence.get('preview_age_s')
            if (not _number(checked) or not _number(age) or not start+elapsed<=checked<=_handoff_time(row)+1e-6 or
                    abs(age-(checked-stamp))>1e-6 or not 0<=age<=.25 or
                    evidence.get('failed_preview_checks')!=[] or evidence.get('motion_static') is not True or
                    evidence.get('preview_valid') is not False):raise ValueError('barrier_final_recheck_not_current')
            if (evidence.get('model_source_id')!=prepared.get('model_source_id') or
                    evidence.get('selected_candidate_id')!=prepared.get('selected_candidate_id')):
                raise ValueError('barrier_final_recheck_changed_model_or_candidate')
            if prepared.get('phase')=='prepare_current_frame' and (
                    controller.get('proposal_observation_id')!=barrier['expected_validation_observation_id'] or
                    controller.get('proposal_capture_timestamp')!=stamp):
                raise ValueError('offered_batch_different_from_new_authorization')
    except (ValueError,KeyError,TypeError,AttributeError,OverflowError) as error:
        issues.append(str(error))
    return issues


def validate_run(rows, report, gc_events, *, config=None, source_manifest=None):
    config=config or AcceptanceConfig()
    result=dict(schema='punktnull.sim_phase_acceptance.v1',accepted=False,
        scope='recorded_scripted_isolated_control_phases_only',thresholds=asdict(config),checks=[],metrics={},
        coverage={'hardware_verified':False,'human_headset_trial':False,
            'physical_pose_drift_verified':False,'measured_finger_drift_verified':False,
            'physical_translation_accuracy_verified':False,
            'command_target_translation_verified':False,'frame_barrier_evidence_verified':False,
            'final_receipt_handoff_evidence_verified':False,'unitree_CDR_roundtrip_evidence_verified':False,
            'no_DDS_transport_manifest_verified':False},limitations=[])
    def check(name,passed,**detail):
        result['checks'].append(dict(id=name,passed=bool(passed),**detail))
        return bool(passed)
    if not check('artifact_structure',isinstance(rows,list) and bool(rows) and isinstance(report,dict)
        and isinstance(gc_events,list) and _finite_json([rows,report,gc_events])):
        return result
    valid_rows=all(isinstance(r,dict) and type(r.get('tick')) is int and isinstance(r.get('controller'),dict)
        and isinstance(r['controller'].get('snapshot'),dict) and isinstance(r.get('timing'),dict)
        and type(r.get('command_consumed')) is bool and _number(r.get('orientation_error_deg'))
        and 0<=r['orientation_error_deg']<=180 and _number(r.get('contact_force_n'))
        and r['contact_force_n']>=0 for r in rows)
    if not check('required_row_evidence',valid_rows):return result
    policy_records=[report]+[r['controller'] for r in rows]
    policy_records += [r['post_physics_controller'] for r in rows if 'post_physics_controller' in r]
    if source_manifest is not None:policy_records.append(source_manifest)
    if not check('ui_action_policy_evidence_consistent',
        all(isinstance(record,dict) for record in policy_records)):
        return result
    versioned=any('ui_action_policy' in record for record in policy_records)
    state_only=versioned and isinstance(source_manifest,dict) and all(
        record.get('ui_action_policy')==UI_ACTION_POLICY for record in policy_records)
    if versioned and not check('known_ui_action_policy_bound_to_every_status_and_manifest',state_only,
        report_policy=report.get('ui_action_policy'),
        manifest_policy=source_manifest.get('ui_action_policy') if isinstance(source_manifest,dict) else None):
        return result
    result['metrics']['ui_action_policy']=UI_ACTION_POLICY if state_only else 'legacy_unversioned'
    if not versioned:
        result['limitations'].append('Unversioned historical UI action policy: original phase expectations apply; '
            'state-only actions and fresh-control separation are not established by this recording.')
    ticks=[r['tick'] for r in rows]
    check('monotonic_unique_runtime_ticks',all(a<b for a,b in zip(ticks,ticks[1:])))
    indexed=any('scenario_tick' in r for r in rows)
    if indexed:
        if not check('scenario_index_contract',all('scenario_tick' in r and
            (r['scenario_tick'] is None or (type(r['scenario_tick']) is int and r['scenario_tick']>=-1)) for r in rows)):
            return result
        warmup=[r for r in rows if r['scenario_tick'] is None or r['scenario_tick']==-1]
        scenario=[r for r in rows if type(r['scenario_tick']) is int and r['scenario_tick']>=0]
    else:warmup=[];scenario=rows
    indices=[r['scenario_tick'] if indexed else r['tick'] for r in scenario]
    contiguous=bool(indices) and indices==list(range(len(indices))) and len(indices)>=130
    if not check('complete_scripted_scenario',contiguous,scenario_rows=len(indices),minimum_rows=130,
                 index_source='scenario_tick' if indexed else 'legacy_tick'):
        return result
    by=dict(zip(indices,scenario));end=indices[-1]
    result['metrics'].update(runtime_rows=len(rows),scenario_rows=len(scenario),warmup_rows=len(warmup))
    check('scenario_completion_report',report.get('completed') is True and
          (report.get('scripted_scenario_completed') is True if 'scripted_scenario_completed' in report else not indexed))
    check('warmup_never_commands',all(not r['command_consumed'] for r in warmup))
    consumed=sum(r['command_consumed'] for r in rows)
    check('report_counts_match_rows',report.get('ticks')==len(rows) and report.get('commands_consumed')==consumed,
          actual_commands=consumed,reported_commands=report.get('commands_consumed'))
    final_path=any('final_actor' in row for row in rows) or any(key in report for key in
        ('command_format','command_codec','DDS_transport_started','transport_modules_loaded'))
    final_manifest_ok=False
    sender=report.get('sender','direct')
    check('known_simulator_sender',sender in ('direct','guarded_original'))
    if sender=='guarded_original':
        manifest=report.get('sender_manifest') or {}
        sender_manifest_ok=(report.get('command_format')=='unitree_float32'
            and manifest.get('sha256')=='793e0774bb0a7a29b0cd163336babd45b29a36c3518528c87a854969c4132f73'
            and manifest.get('publisher_override_installed_before_constructor') is True
            and manifest.get('original_class_bodies_unmodified') is True
            and manifest.get('original_module_imported') is False
            and manifest.get('DDS_transport_started') is False
            and manifest.get('hardware_authorized') is False
            and manifest.get('stock_XR_application_started') is False
            and manifest.get('complete_XR_entry_integration_verified') is False
            and manifest.get('original_self_msg_used_for_commands') is False
            and manifest.get('runtime_scope')=='original_controller_shim_in_isolated_complete_PhysX_scene'
            and report.get('sender_close_error') is None)
        original_issues=[dict(runtime_tick=row['tick'],issues=issues) for row in rows
            if (issues:=_original_sender_evidence_issues(row))]
        sender_ok=check('original_controller_owned_receiver_and_every_physics_interlock',
            sender_manifest_ok and not original_issues,manifest_valid=sender_manifest_ok,failing_rows=original_issues)
        result['coverage']['guarded_original_simulator_receiver_verified']=sender_ok
        result['limitations'].append('Guarded original-controller shim and owned simulator receiver only; '
            'stock XR application, warehouse collision scene, headset and robot were not started.')
    if final_path:
        fmt=report.get('command_format');manifest=report.get('command_codec')
        final_manifest_ok=(fmt in ('float64','unitree_float32') and report.get('DDS_transport_started') is False and
            report.get('transport_modules_loaded')==[] and
            ((fmt=='float64' and manifest is None and report.get('dds_imported') is False) or
             (fmt=='unitree_float32' and _codec_manifest_valid(manifest) and report.get('dds_imported') is True)))
        check('final_actor_transport_and_codec_manifest',final_manifest_ok,
            command_format=fmt,SDK_or_IDL_imported=report.get('dds_imported'),
            DDS_transport_started=report.get('DDS_transport_started'),
            transport_modules_loaded=report.get('transport_modules_loaded'))
        result['coverage']['no_DDS_transport_manifest_verified']=bool(final_manifest_ok)
        actor_failures=[dict(runtime_tick=row['tick'],issues=issues) for row in rows
            if (issues:=_actor_evidence_issues(row,fmt))]
        actor_ok=check('every_applied_command_has_exact_receipt_codec_and_final_deadline_evidence',
            not actor_failures and consumed>0,failing_rows=actor_failures)
        command_actors=[row.get('final_actor') for row in rows if row['command_consumed']]
        sequences=[a.get('guard_sequence') if isinstance(a,dict) else None for a in command_actors]
        tokens=[a.get('receipt_token') if isinstance(a,dict) else None for a in command_actors]
        unique=all(type(s) is int and s>0 for s in sequences) and all(a<b for a,b in zip(sequences,sequences[1:]))
        unique=unique and all(isinstance(t,str) and t for t in tokens) and len(set(tokens))==len(tokens)
        check('final_actor_tokens_and_guard_sequences_are_not_replayed',unique)
        result['coverage']['final_receipt_handoff_evidence_verified']=bool(actor_ok and unique and final_manifest_ok)
        result['coverage']['unitree_CDR_roundtrip_evidence_verified']=bool(
            actor_ok and unique and final_manifest_ok and fmt=='unitree_float32')
        result['limitations'].append('Final actor evidence audits the recorded receipt deadline and local target handoff; '
            'snapshot digest equality is recorded binding consistency, not independent reconstruction of every pre-command snapshot.')
        if fmt=='unitree_float32':
            result['limitations'].append('The Unitree generated IDL/serialization library was imported. The recorded inactive CDR carrier '
                'is reconstructed offline; no DDS transport, complete LowCmd regulator or actuator reception is established.')
    else:
        result['limitations'].append('Legacy run has no final_actor receipt/serializer trace: final private deadline and '
            'Unitree CDR roundtrip coverage are unavailable.')
    scope_ok=(report.get('scripted_operator') is True and report.get('headset_verified') is False and
        report.get('real_robot_tested') is False and
        (final_manifest_ok if final_path else report.get('dds_imported') is False))
    check('scripted_simulation_not_hardware_or_headset',scope_ok)
    scope_bad=[r['tick'] for r in rows if r['controller'].get('mode')!='isolated_PhysX' or
        r['controller'].get('robot_transport_started') is not False or
        r['controller'].get('real_robot_authorized') is not False or
        r['controller'].get('downstream_ik_allowed') is not False or
        r['controller']['snapshot'].get('robot_command_sent',False) is not False]
    check('no_robot_transport_or_downstream_ik_in_rows',not scope_bad,failing_runtime_ticks=scope_bad)
    source_sequences=[r.get('source_sequence') for r in scenario]
    check('source_sequences_increase',all(type(x) is int and x>=0 for x in source_sequences) and
        all(a<b for a,b in zip(source_sequences,source_sequences[1:])))
    live=report.get('live_ai') is True
    if live and any('frame_barrier' in row for row in rows):
        barrier_failures=[dict(runtime_tick=row['tick'],issues=issues) for row in rows
            if (issues:=_barrier_evidence_issues(row))]
        barrier_ok=check('live_frame_barrier_and_final_recheck_preserve_capture_and_authority',
            not barrier_failures,failing_rows=barrier_failures)
        result['coverage']['frame_barrier_evidence_verified']=barrier_ok
        result['limitations'].append('Frame-barrier checks audit recorded scheduling and capture identity; '
            'they do not establish physical real-time operation or hardware safety.')
    active=set(range(5,70))|set(range(75,94))|set(range(113,end+1))
    if state_only:active-=UI_ACTION_TICKS
    optional_edges=({8,78,116} if state_only else {7,8,77,78,115,116}) if live else set()
    wrong=[i for i,r in by.items() if i not in optional_edges and r['command_consumed']!=(i in active)]
    check('command_availability_matches_deliberate_phases',not wrong,failing_scenario_ticks=wrong,
          optional_live_reselection_edges=sorted(optional_edges))
    blocked=set(range(0,5))|set(range(70,75))|set(range(94,113))
    if state_only:blocked|=UI_ACTION_TICKS
    check('blocked_phases_never_command',all(not by[i]['command_consumed'] for i in blocked))
    if state_only:
        action_bad=[i for i in sorted(UI_ACTION_TICKS) if by[i]['command_consumed'] or
            by[i]['controller'].get('allowed') is not False or
            by[i]['controller'].get('command_pending') is not False or
            by[i]['controller'].get('q14_command') is not None or
            by[i]['controller'].get('reason')!='ui_action_waiting_for_fresh_control_sample']
        check('one_shot_ui_actions_change_state_without_motion_offer',not action_bad,
            failing_scenario_ticks=action_bad,strictly_blocked_scenario_ticks=sorted(UI_ACTION_TICKS))
    phases={'first_held':(12,64),'button_fade':(65,69),'second_held':(82,93),
            'stale':(94,100),'restored_without_clutch':(101,111),'rearmed_held':(120,end)}
    result['metrics']['phase_commands']={name:dict(consumed=sum(by[i]['command_consumed'] for i in range(a,b+1)),
        total=b-a+1) for name,(a,b) in phases.items()}
    clutch_bad=[]
    for i in (4,74,112):
        c=by[i]['controller'];s=c['snapshot']
        if by[i]['command_consumed'] or s.get('clutched') is not True or s.get('confirmed') is not False:
            clutch_bad.append(i)
    check('explicit_clutch_rebases_without_immediate_command',not clutch_bad,failing_scenario_ticks=clutch_bad)
    confirm_bad=[]
    for i in (9,79,117):
        c=by[i]['controller'];s=c['snapshot']
        if s.get('confirmed') is not True or not s.get('confirmed_candidate_id') or c.get('assistance_applied') is not False:
            confirm_bad.append(i)
    check('confirmation_precedes_held_assistance',not confirm_bad,failing_scenario_ticks=confirm_bad)
    held_bad=[];binding_bad=[]
    confirmed_ids=set();bindings=set()
    for a,b in ((12,64),(82,93),(120,end)):
        for i in range(a,b+1):
            c=by[i]['controller'];s=c['snapshot']
            if (s.get('confirmed') is not True or not s.get('confirmed_candidate_id') or s.get('assist_held') is not True or
                    c.get('assistance_applied') is not True or s.get('state')!='assisting'):
                held_bad.append(i)
            if s.get('confirmed_candidate_id'):confirmed_ids.add(s['confirmed_candidate_id'])
            if s.get('model_source_id'):bindings.add(s['model_source_id'])
            elif live:binding_bad.append(i)
    check('held_assistance_uses_confirmed_candidate',not held_bad,failing_scenario_ticks=held_bad)
    check('candidate_identity_never_auto_switches',len(confirmed_ids)==1,confirmed_candidate_ids=sorted(confirmed_ids))
    if live:check('live_model_binding_stays_explicit_and_frozen',len(bindings)==1 and not binding_bad,
                  model_source_ids=sorted(bindings),failing_scenario_ticks=binding_bad)
    for label,a,b,expected in (('stale_tracking_hold',94,100,'operator_sample_stale_or_future'),
                              ('no_automatic_restart_after_source_recovers',101,111,'clutch_required')):
        failed=[i for i in range(a,b+1) if by[i]['command_consumed'] or
            by[i]['controller']['snapshot'].get('clutched') is not False or
            by[i]['controller']['snapshot'].get('confirmed') is not False or
            by[i]['controller'].get('reason')!=expected]
        check(label,not failed,failing_scenario_ticks=failed)
    release=by[70]['controller']
    check('explicit_release_revokes_clutch_and_confirmation',release.get('reason')=='explicit_release_reclutch_required'
        and release['snapshot'].get('clutched') is False and release['snapshot'].get('confirmed') is False)
    fade_bad=[];offsets=[]
    for i in range(64,70):
        s=by[i]['controller']['snapshot'];value=s.get('orientation_offset_rad')
        if not _number(value):fade_bad.append(i);continue
        offsets.append(value)
        if i==64:continue
        try:
            previous=by[i-1]['controller']['snapshot']['orientation_offset_rad'];dt=_time(by[i])-_time(by[i-1])
            if (dt<=0 or value>previous+1e-8 or previous-value>math.radians(30)*dt+1e-6 or
                s.get('assist_held') is not False or s.get('confirmed') is not True):fade_bad.append(i)
        except (KeyError,TypeError,ValueError):fade_bad.append(i)
    responsive=len(offsets)==6 and offsets[0]>1e-6 and offsets[-1]<offsets[0]-math.radians(.5)
    check('button_release_fades_with_bounded_rate_without_rearm',not fade_bad and responsive,
        failing_scenario_ticks=fade_bad,offsets_deg=[math.degrees(v) for v in offsets])
    pose_bad=[];lease_bad=[];scope_consumed_bad=[];correction_bad=[]
    for i,r in by.items():
        if not r['command_consumed']:continue
        c=r['controller'];s=c['snapshot']
        if (r.get('endpoint_accepted') is not True or c.get('allowed') is not True or
                c.get('command_consumed_once') is not True or c.get('command_pending') is not False or
                c.get('input_hold_required') is not False or s.get('clutched') is not True or
                c.get('human_source')!='scripted_test_fixture_NO_HEADSET' or
                c.get('finger_control')!='measured_q43_values_unchanged' or
                c.get('operator_source_sequence')!=r.get('source_sequence')):scope_consumed_bad.append(i)
        try:
            raw,assisted=_pose(c['mapped_right_base_ee']),_pose(c['assisted_right_base_ee'])
            if np.linalg.norm(raw[:3,3]-assisted[:3,3])>1e-8:pose_bad.append(i)
            measured_offset=math.radians(_angle(assisted,raw));reported=s['orientation_offset_rad']
            if (not _number(reported) or not 0<=reported<=math.radians(20)+1e-8 or
                    abs(measured_offset-reported)>1e-6 or s.get('translation_assistance') is not False or
                    s.get('finger_assistance') is not False):correction_bad.append(i)
        except (ValueError,KeyError,TypeError,IndexError):
            pose_bad.append(i);correction_bad.append(i)
        try:
            consumed_at=_handoff_time(r)
            times=[c['expires_at'],c['operator_expires_at'],c['ui_expires_at'],c['guard']['valid_until'],
                   c['operator_acquisition_timestamp'],c['handoff_inputs_checked_at']]
            if not all(_number(x) for x in times):raise ValueError('Missing lease provenance')
            if (not c['handoff_inputs_checked_at']<=consumed_at<min(times[:4]) or
                    abs(c['operator_expires_at']-c['operator_acquisition_timestamp']-.15)>1e-6 or
                    c['expires_at']>min(times[1:4])+1e-8 or consumed_at<c['operator_acquisition_timestamp']):
                lease_bad.append(i)
        except (ValueError,KeyError,TypeError,IndexError):
            lease_bad.append(i)
    check('consumed_commands_have_guard_and_scripted_source',not scope_consumed_bad,failing_scenario_ticks=scope_consumed_bad)
    check('consumption_finishes_within_original_source_lease',not lease_bad,failing_scenario_ticks=sorted(set(lease_bad)))
    check('assistance_never_changes_command_translation',not pose_bad,failing_scenario_ticks=sorted(set(pose_bad)))
    result['coverage']['command_target_translation_verified']=not pose_bad
    check('assistance_orientation_correction_is_bounded',not correction_bad,failing_scenario_ticks=correction_bad)
    baseline=float(np.median([by[i]['orientation_error_deg'] for i in range(4)]))
    initial_drift=max(abs(by[i]['orientation_error_deg']-baseline) for i in range(4,12))
    restored_baseline=by[111]['orientation_error_deg']
    restored_drift=max(abs(by[i]['orientation_error_deg']-restored_baseline) for i in range(112,120))
    intermediate_drift=max(abs(by[i]['orientation_error_deg']-by[73]['orientation_error_deg']) for i in range(74,82))
    settled=float(np.median([by[i]['orientation_error_deg'] for i in range(55,65)]))
    improvement=1-settled/baseline if baseline>0 else 0.
    result['metrics'].update(initial_goal_error_deg=baseline,initial_goal_error_drift_deg=initial_drift,
        post_stale_reclutch_goal_error_drift_deg=restored_drift,first_assist_settled_error_deg=settled,
        first_assist_improvement_fraction=improvement,final_error_deg=by[end]['orientation_error_deg'],
        intermediate_reclutch_goal_error_excursion_deg=intermediate_drift)
    check('initial_clutch_preserves_recorded_goal_error',initial_drift<=config.initial_goal_error_drift_deg,
          observed_deg=initial_drift,limit_deg=config.initial_goal_error_drift_deg)
    check('post_stale_reclutch_preserves_recorded_goal_error',restored_drift<=config.initial_goal_error_drift_deg,
          observed_deg=restored_drift,limit_deg=config.initial_goal_error_drift_deg)
    check('post_release_reclutch_preserves_recorded_goal_error',intermediate_drift<=config.initial_goal_error_drift_deg,
          observed_deg=intermediate_drift,limit_deg=config.initial_goal_error_drift_deg)
    check('first_assistance_has_measurable_orientation_benefit',baseline>=config.minimum_initial_challenge_deg and
          settled<=config.settled_orientation_error_deg and improvement>=config.minimum_assist_improvement_fraction,
          baseline_deg=baseline,settled_deg=settled,improvement_fraction=improvement)
    for name,a,b in (('second_assist',89,93),('rearmed_assist',max(120,end-4),end)):
        median=float(np.median([by[i]['orientation_error_deg'] for i in range(a,b+1)]))
        check(name+'_settles',median<=config.settled_orientation_error_deg,median_error_deg=median,
              limit_deg=config.settled_orientation_error_deg)
    measured_fields=('actual_left_base_ee','actual_right_base_ee','actual_joint_names','actual_q43')
    any_measured=any(any(k in r for k in measured_fields) for r in scenario)
    all_measured=all(all(k in r for k in measured_fields) for r in scenario)
    if any_measured or config.require_measured_poses:
        check('complete_measured_pose_and_joint_trace',all_measured)
    if all_measured:
        try:
            drift_checks=[]
            for label,anchor,a,b in (('initial',3,4,11),('post_release',73,74,81),('post_stale',111,112,119)):
                for side in ('left','right'):
                    key=f'actual_{side}_base_ee';reference=_pose(by[anchor][key]);values=[_pose(by[i][key]) for i in range(a,b+1)]
                    position=max(float(np.linalg.norm(v[:3,3]-reference[:3,3])) for v in values)
                    rotation=max(_angle(v,reference) for v in values)
                    drift_checks.append(check(f'{label}_{side}_measured_clutch_drift',position<=config.clutch_position_drift_m and
                        rotation<=config.clutch_orientation_drift_deg,position_m=position,orientation_deg=rotation))
            names=by[0]['actual_joint_names'];initial=np.asarray(by[0]['actual_q43'],float)
            if len(names)!=43 or len(set(names))!=43 or initial.shape!=(43,) or not np.isfinite(initial).all():
                raise ValueError('Invalid complete named q43')
            fingers=[i for i,name in enumerate(names) if '_hand_' in name]
            if len(fingers)!=14:raise ValueError('Expected fourteen measured Dex3 joints')
            maximum=0.
            for r in scenario:
                q=np.asarray(r['actual_q43'],float)
                if r['actual_joint_names']!=names or q.shape!=(43,) or not np.isfinite(q).all():raise ValueError('Inconsistent measured q43')
                maximum=max(maximum,float(np.max(abs(q[fingers]-initial[fingers]))))
            check('measured_fingers_remain_fixed',maximum<=config.max_finger_drift_rad,max_drift_rad=maximum,limit_rad=config.max_finger_drift_rad)
            result['coverage'].update(physical_pose_drift_verified=all(drift_checks),measured_finger_drift_verified=True)
            translation_checks=[]
            for side in ('left','right'):
                key=f'actual_{side}_base_ee';reference=_pose(by[0][key])
                excursions=[float(np.linalg.norm(_pose(r[key])[:3,3]-reference[:3,3])) for r in scenario]
                errors=[]
                for r in scenario:
                    if r['command_consumed']:
                        target=_pose(r['controller'][f'mapped_{side}_base_ee'])
                        errors.append(float(np.linalg.norm(_pose(r[key])[:3,3]-target[:3,3])))
                if not errors:raise ValueError('No consumed command with a measured manual target')
                maximum_error=max(errors);maximum_excursion=max(excursions)
                translation_checks.append(check(f'{side}_measured_manual_position_accuracy',
                    maximum_error<=config.maximum_manual_position_error_m,
                    maximum_error_m=maximum_error,median_error_m=float(np.median(errors)),
                    limit_m=config.maximum_manual_position_error_m))
                translation_checks.append(check(f'{side}_stationary_input_does_not_accumulate_translation',
                    maximum_excursion<=config.maximum_stationary_position_excursion_m,
                    maximum_excursion_m=maximum_excursion,limit_m=config.maximum_stationary_position_excursion_m,
                    scope='stationary_synthetic_operator_across_all_three_clutches'))
            result['coverage']['physical_translation_accuracy_verified']=all(translation_checks)
        except (ValueError,KeyError,TypeError,IndexError) as exc:check('measured_trace_valid',False,reason=str(exc))
    else:
        result['limitations'].append('No measured EE/q43 trace: scalar goal-error stability does not prove physical pose or finger stability.')
    maximum=max(r['contact_force_n'] for r in rows)
    check('no_recorded_contacts_in_instrumented_scope',maximum<=config.contact_tolerance_n,
          max_force_n=maximum,tolerance_n=config.contact_tolerance_n,sensor_scope=report.get('physical_contact_sensor_scope'))
    check('contact_and_gravity_report_consistent',_number(report.get('max_contact_force_n')) and
          abs(report['max_contact_force_n']-maximum)<=1e-9 and report.get('gravity_saturation_steps')==0)
    check('known_scene_stays_valid',all(r.get('scene_error') is None for r in rows))
    valid_gc=all(isinstance(e,dict) and type(e.get('generation')) is int and e['generation'] in (0,1,2) and
        all(_number(e.get(k)) for k in ('started','finished','elapsed_s')) and e['finished']>=e['started'] and
        abs(e['elapsed_s']-(e['finished']-e['started']))<=1e-7 for e in gc_events)
    check('gc_event_trace_consistent',valid_gc)
    if valid_gc:
        result['metrics']['gc']=dict(events=len(gc_events),by_generation=dict(Counter(str(e['generation']) for e in gc_events)),
            maximum_pause_s=max((e['elapsed_s'] for e in gc_events),default=0.),
            total_pause_s=sum(e['elapsed_s'] for e in gc_events),startup_frozen_objects=report.get('startup_heap_frozen_objects'))
    result['metrics']['rss_peak_kib']=dict(start=report.get('max_rss_start_kib'),end=report.get('max_rss_end_kib'))
    result['limitations'].extend(['Contact telemetry covers '+str(report.get('physical_contact_sensor_scope','unknown sensor scope'))+'.',
        'Physical self-collision, grasp closure/lift, human usability and hardware operation are not established by these phase logs.',
        'Short-run peak RSS and GC telemetry do not establish long-run memory stability.'])
    result['accepted']=all(item['passed'] for item in result['checks'])
    return result


def check_directory(directory, *, config=None):
    directory=Path(directory)
    payload={};hashes={}
    names=['rows.json','report.json','gc_events.json']
    if (directory/'runtime_source_manifest.json').exists():names.append('runtime_source_manifest.json')
    for name in names:
        raw=(directory/name).read_bytes()
        if len(raw)>64*1024*1024:raise ValueError('Run artifact exceeds 64 MiB')
        payload[name]=json.loads(raw,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+value)))
        hashes[name]=hashlib.sha256(raw).hexdigest()
    manifest=payload.get('runtime_source_manifest.json')
    if 'runtime_source_manifest.json' in payload and not isinstance(manifest,dict):
        raise ValueError('Runtime source manifest must be a JSON object')
    result=validate_run(payload['rows.json'],payload['report.json'],payload['gc_events.json'],config=config,
        source_manifest=manifest)
    result.update(artifact_directory=str(directory.resolve()),artifact_sha256=hashes)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--require-measured-poses',action='store_true')
    args=parser.parse_args(argv)
    try:result=check_directory(args.directory,config=AcceptanceConfig(require_measured_poses=args.require_measured_poses))
    except (OSError,ValueError,TypeError,KeyError) as exc:
        result=dict(schema='punktnull.sim_phase_acceptance.v1',accepted=False,error=str(exc))
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(encoded)
    else:sys.stdout.write(encoded)
    return 0 if result['accepted'] else 1


if __name__=='__main__':raise SystemExit(main())
