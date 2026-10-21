"""Synthetic contracts only; physical V3 success requires a separate Isaac run."""
import json

import numpy as np
import pytest

from teleop_assist.core import AssistConfig, AssistSession
from teleop_assist.manual_release_evaluation import ManualReleaseGraspDriver, _ReleaseHoldEvidence
from teleop_assist.vision_grasp_evaluation import VisionGraspPullDriver
from test_vision_grasp_evaluation import target
from test_vision_grasp_v2_evidence import pair_telemetry


def run_evidence(*,release_loss=.004,manual_travel=.04,hold_force=0.,joint_angle=0.,
                 missing_joint=False,wrong_partner=False,hold_seconds=1.25,hold_drift=0.,
                 missing_pair=False,gap=False,pull_travel=.032,drawer_pull=.032):
    session=AssistSession(AssistConfig(require_action=True,assist_enabled=False))
    driver=ManualReleaseGraspDriver(session)
    driver.target_description={'id':'camera_handle','cad_asset_id':'PullHandleGrasp'}
    sim=0.

    def observe(phase,hand,drawer,force,grip=.85):
        driver.phase=phase
        state={'valid':True,'ee_position':[.5-hand,-.2,1.],'timestamp':100.+sim,'sim_time':sim,
            'ee_quaternion':[0.,0.,0.,1.],'accepted_grip':grip,
            'joint_names':list(_ReleaseHoldEvidence.JOINT_NAMES),'joints':[joint_angle]*7,
            'fixture_joint_states':{'PullHandle':{'position':-drawer}},
            'evaluation_hand_target_contacts':pair_telemetry(sim,force,'Other' if wrong_partner and phase=='manual_release_hold' else 'PullHandle')}
        if phase=='manual_release_hold':
            if missing_joint:state['joint_names'].pop();state['joints'].pop()
            if missing_pair:del state['evaluation_hand_target_contacts']
        driver.observe(state,{'hold':False},sim)

    for i in range(61):
        sim=i*.005;observe('align_pregrasp',0.,0.,0.)
    sim+=.005;observe('close_gripper',0.,.035,.05)
    driver.pull_origin=np.array([.5,-.2,1.]);driver.pull_axis=np.array([-1.,0.,0.])
    for i in range(601):
        sim+=.005;observe('bounded_pull',pull_travel*i/600.,.035+drawer_pull*i/600.,.05)
    driver.manual_origin=np.array([.5-pull_travel,-.2,1.])
    driver.manual_protocol={'rearmed_sim_time':sim+.005,'assist_enabled':False}
    for i in range(601):
        sim+=.005
        observe('manual_release_withdraw',pull_travel+manual_travel*i/600.,
                .035+drawer_pull-release_loss*i/600.,.05*(1.-i/600.),.85*(1.-i/600.))
    for i in range(int(round(hold_seconds/.005))+1):
        sim+=.005+(.05 if gap and i==150 else 0.)
        observe('manual_release_hold',pull_travel+manual_travel,
                .035+drawer_pull-release_loss+hold_drift*i/(hold_seconds/.005),hold_force,0.)
    driver.script_completed=True;driver.done=True
    report=driver.report();json.dumps(report,allow_nan=False)
    return report,driver


def test_v3_separately_bounds_assisted_manual_and_total_and_requires_open_measured_fingers():
    report,driver=run_evidence()
    assert report['passed'],report['checks']
    assert report['schema_version']==3
    checks=report['checks']
    assert checks['bounded_actual_pull']['maximum_measured_execution_displacement_m']==pytest.approx(.032)
    assert checks['bounded_manual_release']['maximum_measured_displacement_m']==pytest.approx(.04)
    assert checks['bounded_combined_motion']['maximum_measured_displacement_m']==pytest.approx(.072)
    assert driver.max_execution_progress==pytest.approx(.072)  # Not silently excluded from the base aggregate.
    assert checks['release_retains_incremental']['required_for_success'] is True
    assert checks['release_retains_incremental']['retained_incremental_opening_m']==pytest.approx(.028)
    final=checks['final_open_finger_contact_free_hold']
    assert final['passed'] and final['continuous_final_window_s']>=1.-1e-9
    assert final['maximum_pair_normal_force_N']==0.
    assert final['maximum_absolute_measured_finger_joint_rad']==0.
    assert len(driver._release_hold.samples)<=202
    assert not checks['measured_hand_contact_during_pull']['required_for_success']


@pytest.mark.parametrize('kwargs,check',[
    ({'release_loss':.02},'release_retains_incremental'),
    ({'release_loss':.006},'release_retains_incremental'),
    ({'manual_travel':.046},'bounded_manual_release'),
    ({'hold_force':.001},'final_open_finger_contact_free_hold'),
    ({'joint_angle':.3},'final_open_finger_contact_free_hold'),
    ({'missing_joint':True},'final_open_finger_contact_free_hold'),
    ({'wrong_partner':True},'final_open_finger_contact_free_hold'),
    ({'missing_pair':True},'final_open_finger_contact_free_hold'),
    ({'hold_seconds':.9},'final_open_finger_contact_free_hold'),
    ({'hold_drift':.002},'final_open_finger_contact_free_hold'),
    ({'gap':True},'final_open_finger_contact_free_hold'),
    ({'drawer_pull':0.},'functional_coupled_pull'),
])
def test_v3_missing_release_or_pull_evidence_fails_closed(kwargs,check):
    report,_=run_evidence(**kwargs)
    assert not report['checks'][check]['passed']
    assert not report['passed']


def test_commanded_open_with_physically_closed_fingers_does_not_pass():
    report,_=run_evidence(joint_angle=.3)
    final=report['checks']['final_open_finger_contact_free_hold']
    assert final['accepted_grip_required']==0.
    assert final['all_hold_rejection_counts']['measured_fingers_not_in_known_open_posture']>0
    assert not final['passed']


def test_final_window_does_not_erase_recent_contact_but_old_contact_can_clear():
    report,driver=run_evidence(hold_force=.01)
    assert not report['passed']
    sample_time=driver._release_hold.samples[-1]['time']
    state={'joint_names':list(_ReleaseHoldEvidence.JOINT_NAMES),'joints':[0.]*7,'accepted_grip':0.,
           'fixture_joint_states':{'PullHandle':{'position':-.06}}}
    for index in range(1,211):
        sim=sample_time+index*.005
        state.update(timestamp=100.+sim,sim_time=sim,evaluation_hand_target_contacts=pair_telemetry(sim))
        driver._release_hold.observe(state,sim)
        if index==190:assert not driver._release_hold.report()['passed']
    assert driver._release_hold.report()['passed']


def run_production_driver(*,lose_target_in_manual=False,stop_in_manual=False):
    now=[10.]
    session=AssistSession(AssistConfig(require_action=True),clock=lambda:now[0])
    driver=ManualReleaseGraspDriver(session)
    state={'valid':True,'ee_position':[.404,-.2,1.],'ee_quaternion':[2**-.5,0.,0.,2**-.5],
           'accepted_grip':0.}
    trace=[];commands=[]
    command=session.command
    def recorded(value):
        if value['type']!='input':commands.append((driver.phase,dict(value)))
        return command(value)
    session.command=recorded
    for index in range(2300):
        sim=index*.02;now[0]=10.+sim
        manual=driver.phase in driver.MANUAL_PHASES
        session.update_frame([] if manual and lose_target_in_manual else [target(now[0])],index,now[0],{'width':100,'height':100})
        if stop_in_manual and driver.phase=='manual_release_withdraw':session.command({'type':'stop'})
        driver.update(state,sim)
        if driver.done:break
        reference=session.step(state,.02)
        state={**state,'ee_position':reference['ee_target_world'],
               'ee_quaternion':reference['ee_quat_xyzw'] or state['ee_quaternion'],
               'accepted_grip':reference['grip']}
        trace.append({'phase':driver.phase,'sim':sim,'assist':session.config.assist_enabled,
                      'clutch':session.input['clutch'],'rearm':session.rearm_required,
                      'grip':reference['grip'],'hold':reference['hold'],'position':state['ee_position']})
        driver.observe(state,reference,sim)
    return driver,session,trace,commands


def test_manual_release_uses_production_disable_calibrate_fresh_clutch_even_after_target_loss():
    driver,session,trace,commands=run_production_driver(lose_target_in_manual=True)
    assert driver.script_completed,driver.report()
    assert not driver.report()['passed']  # No fixture/contact/joint evidence fabricated.
    for phase in driver.MANUAL_PHASES:
        samples=[s for s in trace if s['phase']==phase]
        assert samples and all(not s['assist'] for s in samples)
    prepare=[s for s in trace if s['phase']=='manual_release_prepare']
    assert any(s['hold'] and not s['clutch'] for s in prepare)
    withdraw=[s for s in trace if s['phase']=='manual_release_withdraw']
    assert all(s['clutch'] and not s['rearm'] for s in withdraw)
    assert ('manual_release_prepare',{'type':'calibrate'}) in commands
    assert any(cmd=={'type':'assist_enable','enabled':False} for _,cmd in commands)
    final=[s for s in trace if s['phase']=='manual_release_hold']
    assert final[-1]['grip']==0.
    assert final[-1]['sim']-final[0]['sim']>=1.2
    assert driver.manual_protocol['rearmed_sim_time']>driver.manual_protocol['disarmed_sim_time']
    assert driver.assisted_max_distance<=.045 and driver.manual_max_distance<=.045 and driver.total_max_distance<=.09
    assert not session.input['clutch']
    assert session.selection.mode=='lost'
    # V2 remains the stationary-wrist opening experiment, with its own report.
    assert VisionGraspPullDriver(session).report()['schema_version']==2


def test_manual_release_respects_stop_without_automatic_resume_or_rearm():
    driver,session,trace,commands=run_production_driver(stop_in_manual=True)
    assert driver.done and not driver.script_completed
    assert session.stopped and not session.input['clutch']
    assert 'Operator stop' in driver.report()['checks']['driver']['reason']
    assert not any(cmd['type']=='resume' for _,cmd in commands)
