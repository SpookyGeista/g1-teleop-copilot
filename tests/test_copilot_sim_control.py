"""DDS-free session adversaries. Fake IK gates test contracts, not robot safety."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_assist.copilot.sim_control import G1_JOINT_NAMES, SimControlSession, SimUIIntent
from teleop_assist.copilot.kinematics import ARM_JOINT_NAMES
from teleop_assist.copilot.types import GraspCandidate, ProposalBatch


def pose(angle=0, xyz=(.35, -.2, .9)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('z', angle, degrees=True).as_matrix()
    result[:3, 3] = xyz
    return result


class Worker:
    def __init__(self, fixture):
        self.fixture, self.calls, self.delay, self.reject = fixture, [], 0., False

    def call(self, operation, **payload):
        assert operation == 'validate'
        self.calls.append(deepcopy(payload))
        f = self.fixture
        f.clock += self.delay
        return dict(allowed=not self.reject, hold_required=self.reject,
            q14_command=np.linspace(-.1, .1, 14).tolist(), clock_id='mono',
            validated_at=f.now, valid_until=f.now + .15, sequence=len(self.calls),
            reason='fake_test_only', scope='static_simulation_numerical_precheck')


class Fixture:
    def __init__(self):
        self.now = self.clock = 10.
        self.sequence = self.revision = 0
        self.left, self.right = pose(-15, (.3, .2, .9)), pose(5)
        self.raw_left, self.raw_right = pose(80, (.1, .5, .7)), pose(-70, (-.2, -.5, .7))
        self.worker = Worker(self)
        self.session = SimControlSession(self.worker, clock=lambda: self.clock)
        self.last = None

    def state(self):
        wrist_offset = np.eye(4); wrist_offset[0, 3] = -.05
        return dict(source='synthetic_test_only', complete=True, acquisition_timestamp=self.now,
            clock_id='mono', joint_positions={name: 0. for name in G1_JOINT_NAMES},
            base_in_world=np.eye(4).tolist(), observed_body_world_poses={
                'left_wrist_yaw_link': (self.left @ wrist_offset).tolist(),
                'right_wrist_yaw_link': (self.right @ wrist_offset).tolist()})

    def scene(self):
        return dict(source='synthetic_test_only', acquisition_timestamp=self.now, clock_id='mono',
            scene_id='fixture_scene', geometry_complete=True, static=True, obstacles=[
                dict(id='table', type='box', size_xyz=[.5, .8, .04], world_pose=pose(0, (.5, 0, .8)).tolist())])

    def proposal(self):
        return ProposalBatch('target', 'frozen_model_observation', self.now, 'mono', 'cal', (
            GraspCandidate('grasp', pose(20), .9, True, True, True, 'fake_test_only'),), True, True)

    def packet(self):
        envelope = dict(schema='punktnull.operator_envelope.v1', sequence=self.sequence,
            acquisition_timestamp=self.now, clock_id='mono', tracking_valid=True,
            clock_uncertainty_s=.001, latency_unknown=False, sample_kind='webxr_pose_read',
            received_at_host_monotonic=self.now)
        return dict(schema='punktnull.xr_operator.v1', publish_sequence=self.sequence + 1,
            source_sequence=self.sequence, acquisition_timestamp=self.now, clock_id='mono',
            published_at=self.now, publication_clock_id='mono', expires_at=self.now+.15,
            tracking_valid=True, envelope=envelope, left_base_ee=self.raw_left.tolist(),
            right_base_ee=self.raw_right.tolist(), command_source='synthetic_test_NO_HEADSET',
            target_frame='original_G1_29_L_ee_R_ee_nominal_base', world_transform_calibrated=False,
            source_instance_id='publisher1', source_event_clock_id='browser1')

    def tick(self, *, packet=None, ui=None, state=None, scene=None, batch=None,
             delta=.02, calibration_id='cal', clock_id='mono', **actions):
        self.now += delta; self.clock = self.now
        self.sequence += 1; self.revision += 1
        packet = self.packet() if packet is None else packet
        ui = dict(revision=self.revision, capture_timestamp=self.now, clock_id='mono',
                  client_id='test_ui', **actions) if ui is None else ui
        state, scene = self.state() if state is None else state, self.scene() if scene is None else scene
        self.last = dict(packet=deepcopy(packet), ui=deepcopy(ui), state=deepcopy(state), scene=deepcopy(scene))
        return self.session.step(packet, ui, state, scene, self.proposal() if batch is None else batch,
            measured_left_base_ee=self.left, measured_right_base_ee=self.right,
            now=self.now, clock_id=clock_id, calibration_id=calibration_id)

    def start(self):
        assert not self.tick(release=True)['allowed']
        result = self.tick(clutch=True)
        assert result['snapshot']['clutched'], result['reason']
        assert not result['allowed']
        return self.tick()

    def consume(self, offer, **kwargs):
        return self.session.consume(offer, now=kwargs.pop('now', self.now), clock_id=kwargs.pop('clock_id', 'mono'),
            measured_state=kwargs.pop('state', self.last['state']), scene=kwargs.pop('scene', self.last['scene']), **kwargs)


def test_startup_ignores_persisted_action_and_explicit_clutch_has_no_jump():
    f = Fixture()
    first = f.tick(clutch=True)
    assert not first['snapshot']['clutched'] and first['snapshot']['clutch_available']
    assert first['snapshot']['last_ui_revision'] == 1
    assert f.tick(clutch=True)['snapshot']['clutched']
    result = f.tick()
    assert result['allowed'] and result['q14_command'] is None
    np.testing.assert_allclose(f.worker.calls[-1]['left_base_ee'], f.left, atol=1e-14)
    np.testing.assert_allclose(f.worker.calls[-1]['right_base_ee'], f.right, atol=1e-14)
    assert result['human_source'] == 'synthetic_test_NO_HEADSET'
    json.dumps(result, allow_nan=False)


def test_manual_two_arm_translation_stays_in_base_axes_after_rotated_clutch():
    f = Fixture(); f.start()
    f.raw_left[:3, 3] += [.02, .01, -.01]
    f.raw_right[:3, 3] += [-.01, .025, .02]
    result = f.tick()
    assert result['allowed']
    np.testing.assert_allclose(np.asarray(f.worker.calls[-1]['left_base_ee'])[:3, 3], f.left[:3, 3]+[.02, .01, -.01])
    np.testing.assert_allclose(np.asarray(f.worker.calls[-1]['right_base_ee'])[:3, 3], f.right[:3, 3]+[-.01, .025, .02])


def test_valid_webxr_float32_rotations_do_not_accumulate_scale_at_relative_mapping():
    # Two separately valid matrices from the failed Quest session had tiny
    # Float32 scale errors; direct multiplication produced det(R)-1=1.08e-6.
    from teleop_assist.copilot.controller import rigid
    f=Fixture()
    anchor=np.array([[.5735146403312683,-.8092843890190125,-.12704399228096008],
        [.8191848397254944,.5673640966415405,.08387257158756256],
        [.0042035384103655815,-.15217441320419312,.9883449077606201]])
    current=np.array([[.5964977741241455,-.7847815155982971,-.16825182735919952],
        [.8010521531105042,.5951839089393616,.06381125748157501],
        [.05006294324994087,-.17284157872200012,.9836766719818115]])
    f.raw_left[:3,:3]=anchor
    rigid(f.raw_left)
    f.start()
    f.raw_left[:3,:3]=current
    rigid(f.raw_left)
    assert abs(np.linalg.det(current@anchor.T@f.left[:3,:3])-1)>1e-6
    result=f.tick()
    assert result['allowed'] and result['snapshot']['clutched']
    mapped=rigid(f.worker.calls[-1]['left_base_ee'])
    np.testing.assert_allclose(mapped[:3,:3].T@mapped[:3,:3],np.eye(3),atol=1e-14)
    np.testing.assert_allclose(mapped[:3,3],f.left[:3,3])
    # The ordinary input gate still rejects a real scale error.
    f.raw_left[:3,:3]*=1.001
    result=f.tick()
    assert not result['allowed'] and not result['snapshot']['clutched']


def test_clutch_joint_reference_is_measured_once_and_never_retimestamped():
    f = Fixture(); f.tick()
    measured = f.state()
    for index, name in enumerate(ARM_JOINT_NAMES):
        measured['joint_positions'][name] = .001 * index
    original_stamp = measured['acquisition_timestamp']
    f.tick(clutch=True, state=measured)
    measured['joint_positions'][ARM_JOINT_NAMES[0]] = 99.  # Caller mutation cannot alter it.
    for _ in range(15):
        assert f.tick()['allowed']
        reference = f.worker.calls[-1]['joint_reference']
        assert reference['acquisition_timestamp'] == original_stamp
        assert reference['clock_id'] == 'mono' and reference['calibration_id'] == 'cal'
        assert reference['source'] == 'synthetic_test_only'
        assert reference['joint_names'] == list(ARM_JOINT_NAMES)
        np.testing.assert_array_equal(reference['q14'], np.arange(14) * .001)
    assert reference['acquisition_timestamp'] < f.now - .15
    assert f.worker.calls[-1]['robot_state']['acquisition_timestamp'] == f.now
    # The reference cannot change either manual Cartesian target.
    f.raw_right[0, 3] += .02
    assert f.tick()['allowed']
    assert f.worker.calls[-1]['right_base_ee'][0][3] == pytest.approx(f.right[0, 3] + .02)
    assert f.worker.calls[-1]['joint_reference'] == reference


def test_hold_clears_joint_reference_and_deliberate_reclutch_captures_new_state():
    f = Fixture(); f.start(); old = f.worker.calls[-1]['joint_reference']
    f.tick(release=True)
    assert f.session._clutch_joint_reference is None
    assert not f.tick()['allowed']
    measured = f.state(); measured['joint_positions'][ARM_JOINT_NAMES[-1]] = .012
    f.tick(clutch=True, state=measured)
    assert f.tick()['allowed']
    new = f.worker.calls[-1]['joint_reference']
    assert new['acquisition_timestamp'] > old['acquisition_timestamp']
    assert new['q14'][-1] == .012


def test_exact_private_guard_command_is_consumed_once_and_offer_cannot_override_it():
    f = Fixture(); result = f.start()
    result['q14_command'] = [42.] * 14
    q = f.consume(result)
    np.testing.assert_array_equal(q, np.linspace(-.1, .1, 14))
    assert f.session.status['command_consumed_once']
    assert f.consume(result) is None
    assert not f.session.clutched


def test_assistance_requires_selection_confirmation_hold_and_preserves_translation():
    f = Fixture(); f.start()
    assert not f.tick(assist_held=True)['assistance_applied']
    f.tick(select_candidate_id='grasp')
    confirmed = f.tick(confirm=True)
    assert confirmed['snapshot']['confirmed_candidate_id'] == 'grasp'
    for _ in range(20):
        result = f.tick(assist_held=True)
        assert result['allowed'] and result['assistance_applied']
        actual = np.asarray(result['assisted_right_base_ee'])
        raw = np.asarray(result['mapped_right_base_ee'])
        np.testing.assert_array_equal(actual[:3, 3], raw[:3, 3])
        assert Rotation.from_matrix(actual[:3, :3]@raw[:3, :3].T).magnitude() <= np.deg2rad(20)
    assert result['controller_reason'] == 'assist_held'


def test_preview_confirmation_can_precede_first_clutch_but_reclutch_revokes_it():
    f = Fixture(); f.tick()
    f.tick(select_candidate_id='grasp'); assert f.tick(confirm=True)['snapshot']['confirmed']
    assert f.tick(clutch=True)['snapshot']['confirmed']
    assert f.tick(assist_held=True)['assistance_applied']
    assert not f.tick(clutch=True)['snapshot']['confirmed']
    assert not f.tick(assist_held=True)['assistance_applied']


@pytest.mark.parametrize('fault', ['tracking', 'stale', 'proposal', 'scene', 'state', 'target', 'base', 'calibration'])
def test_loss_holds_and_fresh_data_cannot_automatically_resume(fault):
    f = Fixture(); f.start(); kwargs = {}
    if fault == 'tracking':
        packet=f.packet(); packet['tracking_valid']=False; kwargs['packet']=packet
    elif fault == 'stale':
        packet=f.packet(); packet['acquisition_timestamp']-=1; packet['envelope']['acquisition_timestamp']-=1
        kwargs['packet']=packet
    elif fault == 'proposal': kwargs['batch']=replace(f.proposal(), valid=False)
    elif fault == 'scene':
        scene=f.scene(); scene['obstacles'][0]['world_pose'][0][3]+=.01; kwargs['scene']=scene
    elif fault == 'state':
        state=f.state(); state['joint_positions'].pop('right_hand_thumb_0_joint'); kwargs['state']=state
    elif fault == 'target': kwargs['batch']=replace(f.proposal(), target_id='other')
    elif fault == 'base':
        state=f.state(); state['base_in_world'][0][3]+=.01; kwargs['state']=state
    else: kwargs['calibration_id']='changed'
    result = f.tick(**kwargs)
    assert not result['allowed'] and not result['snapshot']['clutched'], result
    fresh = f.tick()
    assert not fresh['allowed'] and not fresh['snapshot']['confirmed']


def test_new_XR_sample_is_required_after_clutch():
    f = Fixture(); f.tick(); f.tick(clutch=True)
    old = f.last['packet']
    assert not f.tick(packet=old)['allowed']
    assert f.session.clutched and not f.worker.calls
    assert f.tick()['allowed']


@pytest.mark.parametrize('mutation', ['sequence_regression', 'same_source_changed_pose', 'same_publication_changed_content'])
def test_replayed_source_or_publication_cannot_generate_motion(mutation):
    f = Fixture(); f.start(); packet = deepcopy(f.last['packet'])
    if mutation == 'sequence_regression':
        packet['source_sequence']-=1; packet['envelope']['sequence']-=1; packet['publish_sequence']-=1
    elif mutation == 'same_source_changed_pose':
        packet['publish_sequence']+=1; packet['right_base_ee'][0][3]+=.01
    else: packet['command_source']='incorrect_human_label'
    result=f.tick(packet=packet)
    assert not result['allowed'] and not f.session.clutched
    assert 'sequence' in result['reason'] or 'publication' in result['reason']


@pytest.mark.parametrize('key', ['source_instance_id', 'source_event_clock_id'])
def test_source_restart_changes_epoch_holds_then_allows_explicit_new_clutch(key):
    f=Fixture(); f.start(); packet=f.packet(); packet[key]='new_epoch'
    packet['source_sequence']=0; packet['envelope']['sequence']=0; packet['publish_sequence']=1
    result=f.tick(packet=packet)
    assert 'epoch_changed' in result['reason'] and not f.session.clutched
    for edge in (None, 'clutch', None):
        f.now += .01
        packet=f.packet(); packet[key]='new_epoch'
        result=f.tick(packet=packet, **({edge: True} if edge else {}))
        if edge is None and not f.session.clutched: assert not result['allowed']
    assert result['allowed']


def test_stale_ui_exposes_clutch_availability_but_does_not_execute_old_click():
    f=Fixture(); f.start(); ui=deepcopy(f.last['ui']); ui['revision']+=1
    ui['capture_timestamp']-=1; ui['clutch']=True
    result=f.tick(ui=ui)
    assert not result['allowed'] and result['snapshot']['clutch_available']
    assert result['snapshot']['last_ui_revision']==ui['revision']
    assert f.tick()['reason']=='clutch_required'
    assert f.tick(clutch=True)['snapshot']['clutched']


@pytest.mark.parametrize('fault', ['changed_revision', 'client_changed', 'combined_clutch', 'extended_expiry', 'future', 'missing'])
def test_bad_ui_actions_are_acknowledged_but_hold(fault):
    f=Fixture(); f.start(); ui=deepcopy(f.last['ui']); ui['revision']+=1
    if fault=='changed_revision': ui['revision']-=1; ui['confirm']=True
    elif fault=='client_changed': ui['client_id']='other_owner'
    elif fault=='combined_clutch': ui.update(clutch=True,assist_held=True)
    elif fault=='extended_expiry': ui['expires_at']=ui['capture_timestamp']+1
    elif fault=='future': ui['capture_timestamp']=f.now+1
    else: ui.pop('client_id')
    result=f.tick(ui=ui)
    assert not result['allowed'] and not f.session.clutched
    if fault not in ('missing','changed_revision'): assert result['snapshot']['last_ui_revision']==ui['revision']


@pytest.mark.parametrize('delay', [.151, 1.])
def test_expiry_during_cpu_guard_never_yields_command(delay):
    f=Fixture(); f.start(); f.worker.delay=delay
    result=f.tick()
    assert not result['allowed'] and not f.session.clutched


def test_guard_rejection_never_yields_command():
    f=Fixture(); f.start(); f.worker.reject=True
    assert not f.tick()['allowed'] and not f.session.clutched


@pytest.mark.parametrize('fault', ['time', 'clock', 'finger', 'arm', 'base', 'body', 'scene', 'external_stop'])
def test_final_handoff_rechecks_lease_and_unchanged_complete_state(fault):
    f=Fixture(); offer=f.start(); kwargs={}
    if fault=='time': kwargs['now']=f.now+.16
    elif fault=='clock': kwargs['clock_id']='other_clock'
    elif fault in ('finger','arm','base','body'):
        state=deepcopy(f.last['state'])
        if fault=='base':state['base_in_world'][0][3]+=.001
        elif fault=='body':state['observed_body_world_poses']['right_wrist_yaw_link'][0][3]+=.001
        else:state['joint_positions']['right_hand_thumb_0_joint' if fault=='finger' else 'left_elbow_joint']+=.001
        kwargs['state']=state
    elif fault=='scene':
        scene=deepcopy(f.last['scene']);scene['obstacles'][0]['world_pose'][0][3]+=.001;kwargs['scene']=scene
    else:f.session.hold('external_stop')
    assert f.consume(offer,**kwargs) is None and not f.session.clutched
    assert f.session.status['reason']=='command_handoff_rejected_reclutch_required'
    assert f.session.status['handoff_rejection_reason']


def test_source_acquisition_is_forwarded_without_refreshing_to_worker_time():
    f=Fixture(); f.start(); packet=f.packet(); acquired=packet['acquisition_timestamp']
    # Reusing the identical still-valid source is allowed after the clutch;
    # republishing does not create a new acquisition.
    result=f.tick(packet=deepcopy(f.last['packet']))
    assert result['allowed']
    assert f.worker.calls[-1]['acquisition_timestamp']==acquired
    assert result['expires_at']<=acquired+.15


def test_new_offer_supersedes_unconsumed_command():
    f=Fixture(); old=f.start(); f.tick()
    assert f.consume(old) is None and not f.session.clutched


def test_invalid_typed_candidate_is_held_not_raised():
    f=Fixture();f.start()
    assert not f.tick(batch=replace(f.proposal(),candidates=(None,)))['allowed']


def test_ui_from_packet_does_not_invent_provenance():
    with pytest.raises(ValueError):SimUIIntent.from_packet({'clutch':True})
    intent=SimUIIntent.from_packet(dict(revision=1,capture_timestamp=10.,clock_id='mono',client_id='ui',irrelevant=True))
    assert intent.revision==1 and not intent.clutch


def test_long_idle_with_expired_ui_does_not_deadlock_new_explicit_clutch():
    f=Fixture();f.start();old=deepcopy(f.last['ui'])
    for _ in range(30):
        result=f.tick(ui=old)
    assert result['snapshot']['clutch_available'] and not result['allowed']
    assert f.tick(clutch=True)['snapshot']['clutched']


@pytest.mark.parametrize('dt', [0., -.01, .21])
def test_nonpositive_or_long_control_tick_holds(dt):
    f=Fixture();f.start()
    assert f.tick(delta=dt)['reason']=='session_step_interval_invalid'
    assert not f.session.clutched


def assisting_fixture():
    f=Fixture();f.start();f.tick(select_candidate_id='grasp');f.tick(confirm=True)
    for _ in range(6):
        offer=f.tick(assist_held=True)
        assert f.consume(offer) is not None
    offer=f.tick(assist_held=True)
    assert offer['snapshot']['confirmed'] and offer['assistance_applied']
    return f,offer


def race_ui(f, **actions):
    ui=deepcopy(f.last['ui'])
    ui.update(revision=f.revision+1, capture_timestamp=f.now+.005, **actions)
    return ui


def recheck(f, *, packet=None, ui=None, now=None, clock_id='mono'):
    return f.session.recheck_inputs(f.last['packet'] if packet is None else packet,
        f.last['ui'] if ui is None else ui, now=f.now+.005 if now is None else now, clock_id=clock_id)


def test_race_unchanged_source_and_ui_do_not_require_rearm_or_extend_deadline():
    f,offer=assisting_fixture();deadline=offer['expires_at']
    assert recheck(f)
    assert f.session.status['expires_at']==deadline
    assert f.session.status['snapshot']['confirmed']
    assert f.consume(offer,now=f.now+.006) is not None


def test_race_assist_button_release_cancels_pending_but_next_step_returns_gradually():
    f,offer=assisting_fixture()
    before=Rotation.from_matrix(f.session._committed_offset).magnitude()
    revision=f.session.status['snapshot']['last_ui_revision']
    ui=race_ui(f,assist_held=False)
    assert not recheck(f,ui=ui)
    assert f.consume(offer) is None  # A stale caller cannot accidentally declutch.
    assert f.session.clutched and f.session.status['snapshot']['confirmed']
    assert f.session.status['snapshot']['last_ui_revision']==revision
    returned=f.tick(ui=ui)
    assert returned['allowed'] and returned['snapshot']['confirmed']
    assert returned['snapshot']['state']=='releasing'
    assert 0 < returned['snapshot']['orientation_offset_rad'] < before
    assert before-returned['snapshot']['orientation_offset_rad']<=np.deg2rad(30)*.02+1e-10
    assert f.consume(returned) is not None


def test_repeated_unconsumed_offers_cannot_accumulate_assistance_offset():
    f=Fixture();f.start();f.tick(select_candidate_id='grasp');f.tick(confirm=True)
    for _ in range(15):
        offer=f.tick(assist_held=True,delta=.04)
        assert offer['allowed']
        assert offer['snapshot']['orientation_offset_rad']==pytest.approx(np.deg2rad(1))
        np.testing.assert_array_equal(f.session._committed_offset,np.eye(3))
    assert f.consume(offer) is not None
    assert Rotation.from_matrix(f.session._committed_offset).magnitude()==pytest.approx(np.deg2rad(1))


def test_only_successful_consumption_advances_the_next_proposed_offset():
    f=Fixture();f.start();f.tick(select_candidate_id='grasp');f.tick(confirm=True)
    offered=f.tick(assist_held=True,delta=.04)
    assert f.consume(offered) is not None
    committed=f.session._committed_offset.copy()
    for _ in range(5):
        offered=f.tick(assist_held=True,delta=.04)
        assert offered['snapshot']['orientation_offset_rad']==pytest.approx(np.deg2rad(2))
        np.testing.assert_array_equal(f.session._committed_offset,committed)
    exact_pending=f.session._pending['assist_offset'].copy()
    # Public output fields cannot change the private guarded offset commitment.
    offered['snapshot']['orientation_offset_rad']=np.deg2rad(20)
    offered['assisted_right_base_ee']=pose(100).tolist()
    assert f.consume(offered) is not None
    np.testing.assert_array_equal(f.session._committed_offset,exact_pending)
    assert f.tick(assist_held=True,delta=.04)['snapshot']['orientation_offset_rad']==pytest.approx(np.deg2rad(3))


def test_offset_commit_occurs_only_after_the_last_consumer_time_check():
    f,offer=assisting_fixture();before=f.session._committed_offset.copy();observed=[]
    def inspected_clock():
        observed.append(f.session._committed_offset.copy())
        return f.clock
    f.session.clock=inspected_clock
    expected=f.session._pending['assist_offset'].copy()
    assert f.consume(offer) is not None
    assert len(observed)==2
    for offset in observed:np.testing.assert_array_equal(offset,before)
    np.testing.assert_array_equal(f.session._committed_offset,expected)


def test_abandoned_fade_offers_do_not_silently_consume_multiple_return_steps():
    f,offer=assisting_fixture();committed=f.session._committed_offset.copy()
    before=Rotation.from_matrix(committed).magnitude()
    for _ in range(6):
        returned=f.tick(assist_held=False)
        assert returned['allowed'] and returned['snapshot']['state']=='releasing'
        assert returned['snapshot']['orientation_offset_rad']==pytest.approx(before-np.deg2rad(.6))
        np.testing.assert_array_equal(f.session._committed_offset,committed)
    assert f.consume(returned) is not None
    assert Rotation.from_matrix(f.session._committed_offset).magnitude()==pytest.approx(before-np.deg2rad(.6))


@pytest.mark.parametrize('reset',['reclutch','explicit_hold','expired_consume','guard_rejection'])
def test_reset_rebases_committed_offset_without_committing_rejected_offer(reset):
    f,offer=assisting_fixture()
    assert Rotation.from_matrix(f.session._committed_offset).magnitude()>0
    if reset=='reclutch':
        result=f.tick(clutch=True)
        assert result['snapshot']['clutched'] and not result['snapshot']['confirmed']
    elif reset=='explicit_hold':f.session.hold('test_stop')
    elif reset=='expired_consume':assert f.consume(offer,now=offer['expires_at']) is None
    else:
        f.worker.reject=True
        assert not f.tick(assist_held=True)['allowed']
    np.testing.assert_array_equal(f.session._committed_offset,np.eye(3))


def test_manual_opposition_history_survives_unconsumed_offers():
    f,offer=assisting_fixture();committed=f.session._committed_offset.copy()
    # Three genuine source rotations oppose the currently committed assistance.
    # Discarding the resulting offers must not erase the operator's intent.
    for _ in range(3):
        f.raw_right[:3,:3]=Rotation.from_euler('z',-3,degrees=True).as_matrix()@f.raw_right[:3,:3]
        result=f.tick(assist_held=True)
        assert result['allowed']
        np.testing.assert_array_equal(f.session._committed_offset,committed)
    assert not result['snapshot']['confirmed']
    assert result['controller_reason']=='manual_counterrotation'
    assert f.session.clutched


@pytest.mark.parametrize('action', ['clutch','select_candidate_id','confirm'])
def test_race_ui_edge_cancels_command_without_swallowing_the_next_step_action(action):
    f,offer=assisting_fixture();revision=f.revision
    ui=race_ui(f,assist_held=False,**{action:'grasp' if action=='select_candidate_id' else True})
    assert not recheck(f,ui=ui)
    assert f.session.clutched and f.session.status['snapshot']['last_ui_revision']==revision
    assert f.consume(offer) is None and f.session.clutched
    next_result=f.tick(ui=ui)
    assert next_result['snapshot']['last_ui_revision']==ui['revision']
    if action=='clutch':
        assert next_result['snapshot']['clutched'] and not next_result['allowed']
        assert not next_result['snapshot']['confirmed']
    elif action=='select_candidate_id':
        assert next_result['snapshot']['selected_candidate_id']=='grasp'
        assert not next_result['snapshot']['confirmed']
    else:
        assert next_result['snapshot']['confirmed'] and not next_result['allowed']
        assert next_result['reason'] == 'ui_action_waiting_for_fresh_control_sample'


@pytest.mark.parametrize('fault', ['release','owner','ui_stale','ui_regression','ui_mutated_revision',
    'tracking','source_epoch','source_regression','operator_stale','clock','expired_command','missing_ui'])
def test_race_stop_or_untrusted_inputs_revoke_confirmed_session(fault):
    f,offer=assisting_fixture();kwargs={}
    if fault in ('release','owner','ui_stale','ui_regression','ui_mutated_revision'):
        ui=race_ui(f)
        if fault=='release':ui['release']=True
        elif fault=='owner':ui['client_id']='new_owner'
        elif fault=='ui_stale':ui['capture_timestamp']-=1
        elif fault=='ui_regression':ui['revision']-=2
        else:ui['revision']-=1;ui['assist_held']=False
        kwargs['ui']=ui
    elif fault in ('tracking','source_epoch','source_regression','operator_stale'):
        packet=deepcopy(f.last['packet'])
        if fault=='tracking':packet['tracking_valid']=False
        elif fault=='source_epoch':packet['source_instance_id']='restarted_publisher'
        elif fault=='source_regression':
            packet['publish_sequence']-=1;packet['source_sequence']-=1;packet['envelope']['sequence']-=1
        else:packet['published_at']-=1
        kwargs['packet']=packet
    elif fault=='clock':kwargs['clock_id']='unknown'
    elif fault=='expired_command':kwargs['now']=offer['expires_at']
    else:kwargs['ui']={}
    assert not recheck(f,**kwargs)
    assert not f.session.clutched and not f.session.status['snapshot']['confirmed']
    assert f.consume(offer) is None


def test_race_fresh_newer_source_does_not_retarget_or_renew_the_checked_command():
    f,offer=assisting_fixture();original_request=deepcopy(f.worker.calls[-1])
    packet=deepcopy(f.last['packet']);stamp=f.now+.005
    packet.update(publish_sequence=packet['publish_sequence']+1,source_sequence=packet['source_sequence']+1,
        published_at=stamp,acquisition_timestamp=stamp,expires_at=stamp+.15)
    packet['envelope'].update(sequence=packet['source_sequence'],acquisition_timestamp=stamp,received_at_host_monotonic=stamp)
    packet['right_base_ee'][0][3]+=.01
    assert recheck(f,packet=packet)
    assert f.session.status['expires_at']==offer['expires_at']
    assert f.worker.calls[-1]==original_request
    assert f.consume(offer,now=stamp) is not None


def test_race_heartbeat_revision_is_observed_for_regression_but_not_edge_acknowledged():
    f,offer=assisting_fixture();old_ui=deepcopy(f.last['ui']);new_ui=race_ui(f)
    assert recheck(f,ui=new_ui)
    assert f.session.status['snapshot']['last_ui_revision']==old_ui['revision']
    assert not recheck(f,ui=old_ui)
    assert not f.session.clutched


def test_recheck_cannot_create_permission_when_no_guarded_offer_exists():
    f=Fixture();f.tick();f.tick(clutch=True)
    assert not recheck(f)
    assert f.session.clutched and not f.worker.calls


@pytest.mark.parametrize('method',['_state','_scene','_scene_matches'])
def test_lease_expiring_inside_final_consumer_checks_never_returns_q14(monkeypatch,method):
    f,offer=assisting_fixture();original=getattr(f.session,method)
    def delayed(*args,**kwargs):
        result=original(*args,**kwargs)
        f.clock+=.151  # Represents a scheduler/GC pause inside validation.
        return result
    monkeypatch.setattr(f.session,method,delayed)
    assert f.consume(offer) is None
    assert f.session.status['handoff_rejection_reason']=='command_expired_during_sim_handoff_checks'
    assert not f.session.clutched


@pytest.mark.parametrize('method',['_ui','_operator_sequence'])
def test_lease_expiring_inside_latest_input_recheck_revokes_permission(monkeypatch,method):
    f,offer=assisting_fixture();original=getattr(f.session,method)
    def delayed(*args,**kwargs):
        result=original(*args,**kwargs)
        f.clock+=.151
        return result
    monkeypatch.setattr(f.session,method,delayed)
    assert not recheck(f)
    assert f.session.status['reason']=='handoff_source_lease_expired_during_recheck'
    assert not f.session.clutched and f.consume(offer) is None


def test_small_recheck_elapsed_time_is_recorded_without_renewing_original_lease(monkeypatch):
    f,offer=assisting_fixture();original=f.session._operator_sequence
    def delayed(*args,**kwargs):
        result=original(*args,**kwargs);f.clock+=.01;return result
    monkeypatch.setattr(f.session,'_operator_sequence',delayed)
    assert recheck(f)
    assert f.session.status['handoff_inputs_checked_at']==pytest.approx(f.now+.015)
    assert f.session.status['expires_at']==offer['expires_at']
    assert f.consume(offer,now=f.now+.015) is not None


def test_handoff_reserve_burns_short_lease_without_replaying_or_recalibrating():
    f=Fixture();offer=f.start()
    deadline=offer['expires_at'];calls=len(f.worker.calls)
    f.now=f.clock=deadline-.01
    f.sequence+=1;f.revision+=1
    packet=f.packet()
    ui=dict(revision=f.revision,capture_timestamp=f.now,clock_id='mono',client_id='test_ui')
    assert not f.session.recheck_inputs(packet,ui,now=f.now,clock_id='mono',execution_reserve_s=.02)
    assert f.session.status['reason']=='handoff_execution_budget_pending_replan'
    assert f.session.clutched and f.session.status['allowed'] is False
    assert f.consume(offer) is None and f.session.clutched
    assert len(f.worker.calls)==calls  # No guard or renewed receipt in cancellation.
    # The next genuine source/UI sample gets a separate checked command.
    fresh=f.tick()
    assert fresh['allowed'] and fresh['command_token']!=offer['command_token']
    assert len(f.worker.calls)==calls+1


@pytest.mark.parametrize('fault', ['release','tracking','source_epoch','expired_command'])
def test_handoff_reserve_never_preserves_calibration_for_invalid_inputs(fault):
    f=Fixture();offer=f.start()
    f.now=f.clock=offer['expires_at']-.01
    f.sequence+=1;f.revision+=1
    packet=f.packet()
    ui=dict(revision=f.revision,capture_timestamp=f.now,clock_id='mono',client_id='test_ui')
    if fault=='release':ui['release']=True
    elif fault=='tracking':packet['tracking_valid']=False
    elif fault=='source_epoch':packet['source_instance_id']='replacement'
    else:f.now=f.clock=offer['expires_at']
    assert not f.session.recheck_inputs(packet,ui,now=f.now,clock_id='mono',execution_reserve_s=.02)
    assert not f.session.clutched and f.consume(offer) is None


@pytest.mark.parametrize('method',['consume','recheck'])
def test_elapsed_clock_regression_never_returns_permission(monkeypatch,method):
    f,offer=assisting_fixture();original=f.session._scene if method=='consume' else f.session._operator_sequence
    def clock_regression(*args,**kwargs):
        result=original(*args,**kwargs);f.clock-=.01;return result
    monkeypatch.setattr(f.session,'_scene' if method=='consume' else '_operator_sequence',clock_regression)
    assert (f.consume(offer) is None) if method=='consume' else not recheck(f)
    assert not f.session.clutched
