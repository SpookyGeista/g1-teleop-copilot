"""Artificial telemetry tests evaluation logic only, not physical grasp success."""
import json

import numpy as np
import pytest

from teleop_assist.core import AssistConfig, AssistSession
from teleop_assist.vision_grasp_evaluation import VisionGraspPullDriver, _KnownFixturePullEvidence


def pair_telemetry(sim,force=0.,partner='PullHandle',clock_offset=0.):
    bodies=list(_KnownFixturePullEvidence.BODY_NAMES)
    vectors={body:[0.,0.,0.] for body in bodies}
    norms={body:0. for body in bodies}
    vectors[bodies[0]]=[-force,0.,0.];norms[bodies[0]]=force
    return {'schema':'punktnull.evaluation_hand_target_contacts.v1','evaluation_only':True,'valid':True,
        'target_asset_id':'PullHandleGrasp' if partner=='PullHandle' else 'OtherObject',
        'target_rigid_body_asset_id':partner,'target_body_paths':['/World/AssistFixture/Parts/'+partner],
        'sensor_body_names':bodies,'timestamp':100.+sim+clock_offset,'sim_time':sim+clock_offset,
        'per_body_normal_forces_W':vectors,'per_body_normal_force_norms_N':norms,
        'force_units':'N','measurement':'post_step_filtered_normal_force_matrix','negative_control':False}


def run_evaluation(*,hand_distance=.032,drawer_distance=.032,release_loss=.0,force=.05,
                   partner='PullHandle',contact_schedule=lambda fraction:True,missing_pair=False,
                   clock_offset=0.,noise_force=0.,noise_seconds=.3,missing_start=False,
                   gap=False):
    driver=VisionGraspPullDriver(AssistSession(AssistConfig(require_action=True)))
    driver.target_id='camera_handle'
    driver.target_description={'id':'camera_handle','cad_asset_id':'PullHandleGrasp'}
    sim=0.

    def observe(phase,hand,drawer,pair_force,*,pull=False,omit_fixture=False):
        driver.phase=phase
        state={'valid':True,'ee_position':[.5-hand,-.2,1.],'timestamp':100.+sim,'sim_time':sim,
               'contact_force_max':pair_force,'contact_body':'right_hand_thumb_0_link',
               'evaluation_hand_target_contacts':pair_telemetry(sim,pair_force,partner if pull else 'PullHandle',clock_offset if pull else 0.)}
        if not omit_fixture:state['fixture_joint_states']={'PullHandle':{'position':-drawer}}
        if pull and missing_pair:del state['evaluation_hand_target_contacts']
        driver.observe(state,{'hold':False},sim)

    for i in range(int(round(noise_seconds/.005))+1):
        sim=i*.005
        observe('align_pregrasp',0.,0.,noise_force)
    sim+=.005
    observe('close_gripper',0.,.035,force)
    driver.pull_origin=np.array([.5,-.2,1.]);driver.pull_axis=np.array([-1.,0.,0.])
    sim+=.005
    observe('bounded_pull',0.,.035,force,pull=True,omit_fixture=missing_start)
    for i in range(1,601):
        sim+=.005+(.1 if gap and i==300 else 0.)
        fraction=i/600.
        observed_force=force if contact_schedule(fraction) else 0.
        observe('bounded_pull',hand_distance*fraction,.035+drawer_distance*fraction,observed_force,pull=True)
    sim+=.005
    observe('open_gripper',hand_distance,.035+drawer_distance-release_loss,0.)
    driver.script_completed=True;driver.done=True
    report=driver.report()
    json.dumps(report,allow_nan=False)
    return report,driver


def test_v2_accepts_sub_one_newton_coupled_pull_and_reports_release_loss_separately():
    report,driver=run_evaluation(release_loss=.02)
    assert report['schema_version']==2 and report['acceptance_revision']=='known_40mm_fixture_v2'
    assert report['passed']  # Artificial known-pair measurements, not a simulator claim.
    checks=report['checks']
    assert checks['functional_coupled_pull']['hand_incremental_progress_m']==pytest.approx(.032)
    assert checks['functional_coupled_pull']['max_relative_hand_drawer_error_m']<1e-12
    assert checks['distributed_target_hand_contacts']['maximum_pair_normal_force_N']==.05
    assert min(checks['distributed_target_hand_contacts']['contact_seconds_per_travel_third'])>.9
    assert not checks['measured_hand_contact_during_pull']['passed']
    assert not checks['measured_hand_contact_during_pull']['required_for_success']
    release=checks['release_retains_incremental']
    assert not release['passed'] and not release['required_for_success']
    assert release['status']=='partial_quality'
    assert release['retained_incremental_opening_m']==pytest.approx(.012)
    assert release['opening_lost_during_release_m']==pytest.approx(.02)
    assert len(driver._pull_evidence.contact_bins)==450
    assert checks['distributed_target_hand_contacts']['fixed_spatial_bins']==450


def test_release_incremental_quality_can_pass_without_changing_mandatory_task_scope():
    report,_=run_evaluation(release_loss=.002)
    assert report['passed']
    assert report['checks']['release_retains_incremental']['passed']
    assert report['checks']['release_retains_incremental']['status']=='demonstrated'


@pytest.mark.parametrize('kwargs,failed_check',[
    ({'drawer_distance':0.},'functional_coupled_pull'), # Closure alone opens 35 mm.
    ({'drawer_distance':0.,'force':0.},'distributed_target_hand_contacts'), # Free hand.
    ({'partner':'OtherPanel'},'distributed_target_hand_contacts'),
    ({'contact_schedule':lambda fraction:fraction<.1},'distributed_target_hand_contacts'),
    ({'drawer_distance':.045},'functional_coupled_pull'), # Both move, but slip by 13 mm.
    ({'missing_pair':True,'force':2.},'distributed_target_hand_contacts'),
    ({'clock_offset':-.005},'distributed_target_hand_contacts'),
    ({'force':1e-4},'distributed_target_hand_contacts'), # Must exceed fixed threshold.
    ({'noise_force':1e-4},'pair_contact_noise_baseline'),
    ({'noise_seconds':.1},'pair_contact_noise_baseline'),
    ({'missing_start':True},'functional_coupled_pull'),
    ({'gap':True},'distributed_target_hand_contacts'),
    ({'hand_distance':.046,'drawer_distance':.046},'functional_coupled_pull'),
    ({'hand_distance':.024,'drawer_distance':.024},'functional_coupled_pull'),
])
def test_missing_or_noncausal_evidence_fails_closed(kwargs,failed_check):
    report,_=run_evaluation(**kwargs)
    assert not report['checks'][failed_check]['passed']
    assert not report['passed']


def test_contact_only_at_pull_start_cannot_supply_middle_or_last_third():
    report,_=run_evaluation(contact_schedule=lambda fraction:fraction<.1)
    seconds=report['checks']['distributed_target_hand_contacts']['contact_seconds_per_travel_third']
    assert seconds[0]>.2
    assert seconds[1:]==[0.,0.]
    assert report['checks']['functional_coupled_pull']['passed']
    assert not report['passed']


def test_intermittent_contacts_distributed_over_actual_travel_are_measured_in_sim_seconds():
    report,_=run_evaluation(contact_schedule=lambda fraction:int(round(fraction*600))%20<10)
    seconds=report['checks']['distributed_target_hand_contacts']['contact_seconds_per_travel_third']
    assert all(.3<duration<.6 for duration in seconds)
    assert report['passed']


def test_same_position_hold_cannot_accumulate_contact_time_or_invent_progress():
    report,driver=run_evaluation(hand_distance=0.,drawer_distance=0.)
    assert np.count_nonzero(driver._pull_evidence.contact_bins)==0
    assert report['checks']['distributed_target_hand_contacts']['positive_progress_contact_seconds']==0.
    assert not report['passed']


def test_pair_schema_rejects_other_partner_bad_norm_missing_finger_and_stale_timestamp():
    base={'sim_time':1.,'timestamp':101.,'evaluation_hand_target_contacts':pair_telemetry(1.,.05)}
    assert _KnownFixturePullEvidence._pair(base,1.)[0]
    for mutate in (
        lambda value:value.update(target_body_paths=['/World/AssistFixture/Parts/OtherPanel']),
        lambda value:value['per_body_normal_force_norms_N'].update(right_hand_thumb_0_link=.5),
        lambda value:value['sensor_body_names'].pop(),
        lambda value:value.update(timestamp=100.),
    ):
        data=pair_telemetry(1.,.05);mutate(data)
        assert not _KnownFixturePullEvidence._pair({**base,'evaluation_hand_target_contacts':data},1.)[0]
