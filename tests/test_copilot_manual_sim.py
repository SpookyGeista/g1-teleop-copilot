"""Manual simulation baseline: real state transitions, no mock safety verdicts."""
from copy import deepcopy
import numpy as np
import pytest
from teleop_assist.copilot.manual_sim_control import ManualSimControl
from test_copilot_sim_control import Fixture


class OriginalWorker:
    def __init__(self):self.calls=[]
    def call(self,operation,**payload):
        self.calls.append((operation,payload))
        assert operation in ('original_manual_ik','original_manual_reset')
        return {'q14':np.arange(14).tolist()}


def make():
    f=Fixture();w=OriginalWorker();c=ManualSimControl(w,simulation_mode=True)
    def tick(**actions):
        f.now+=.02;f.sequence+=1;f.revision+=1
        ui=dict(revision=f.revision,capture_timestamp=f.now,clock_id='mono',**actions)
        return c.step(f.packet(),ui,(f.left,f.right),np.zeros(14),np.zeros(14),now=f.now,clock_id='mono')
    tick();tick(clutch=True);tick();tick(start=True)
    return f,w,c,tick


def test_manual_has_no_motion_guard_and_preserves_full_human_target():
    f,w,c,tick=make()
    assert c.active and c.anchors is not None
    f.raw_left[0,3]+=.04;f.raw_right[2,3]+=.03
    np.testing.assert_array_equal(tick(),np.arange(14))
    op,p=w.calls[-1]
    assert op=='original_manual_ik'
    np.testing.assert_allclose(np.array(p['left_base_ee'])[:3,3],f.left[:3,3]+[.04,0,0])
    np.testing.assert_allclose(np.array(p['right_base_ee'])[:3,3],f.right[:3,3]+[0,0,.03])
    assert c.status['custom_motion_guard_enabled'] is False


def test_AX_pause_resume_rebases_both_hands_without_queued_motion():
    f,w,c,tick=make();tick()
    assert tick(start=True) is None and not c.active
    n=len(w.calls)
    f.raw_left[:3,3]+=.2;f.raw_right[:3,3]-=.2
    assert tick() is None and len(w.calls)==n
    assert tick(start=True) is None and c.active
    tick()
    np.testing.assert_allclose(w.calls[-1][1]['left_base_ee'],f.left)
    np.testing.assert_allclose(w.calls[-1][1]['right_base_ee'],f.right)


def test_tracking_pause_retains_calibration_and_requires_new_AX():
    f,w,c,tick=make();tick()
    packet=f.packet();packet['tracking_valid']=False
    ui=dict(revision=f.revision,capture_timestamp=f.now,clock_id='mono')
    assert not c.input_is_current(packet,ui,now=f.now,clock_id='mono')
    assert c.active  # Hold immediately, without revoking on one bad sample.
    assert not c.input_is_current(packet,ui,now=f.now+.31,clock_id='mono')
    assert not c.active and c.anchors is not None
    assert tick() is None
    tick(start=True);assert c.active


def test_stop_arriving_during_IK_prevents_install_and_is_processed_once():
    f,w,c,tick=make();tick()
    ui=dict(revision=f.revision+1,capture_timestamp=f.now,clock_id='mono',start=True)
    assert not c.input_is_current(f.packet(),ui,now=f.now,clock_id='mono')
    assert tick(start=True) is None and not c.active
    assert tick() is None and not c.active


def test_release_and_malformed_transform_do_not_reach_IK():
    f,w,c,tick=make();tick(release=True,source_kind='webxr_controller_event')
    n=len(w.calls);assert tick() is None and len(w.calls)==n and c.anchors is None
    f,w,c,tick=make();n=len(w.calls)
    f.raw_right[0,0]=float('nan')
    assert tick() is None and len(w.calls)==n


def test_simulation_only():
    with pytest.raises(ValueError,match='isolated simulation'):
        ManualSimControl(OriginalWorker(),simulation_mode=False)


def test_short_tracking_gap_keeps_calibration_and_recovers_displacement_smoothly():
    f,w,c,tick=make();tick();n=len(w.calls)
    packet=f.packet();packet['tracking_valid']=False
    ui=dict(revision=f.revision,capture_timestamp=f.now,clock_id='mono')
    assert not c.input_is_current(packet,ui,now=f.now,clock_id='mono')
    assert c.active and len(w.calls)==n
    f.raw_left[:3,3]+=.2;f.raw_right[:3,3]-=.2
    anchors=deepcopy(c.anchors)
    assert tick() is not None and c.active
    previous=[f.left.copy(),f.right.copy()]
    for _ in range(100):
        assert tick() is not None
        for side,index in [('left',0),('right',1)]:
            target=np.array(w.calls[-1][1][side+'_base_ee'])
            assert np.linalg.norm(target[:3,3]-previous[index][:3,3])<=.004001
            previous[index]=target
    for original,kept in zip(anchors,c.anchors):
        np.testing.assert_array_equal(original,kept)
    np.testing.assert_allclose(previous[0][:3,3],f.left[:3,3]+.2)
    np.testing.assert_allclose(previous[1][:3,3],f.right[:3,3]-.2)
    assert c.recovery_targets is None


def test_repeated_short_dropouts_do_not_accumulate_controller_robot_drift():
    f,w,c,tick=make();tick();original_left=f.left.copy();original_right=f.right.copy()
    for iteration in range(3):
        packet=f.packet();packet['tracking_valid']=False
        ui=dict(revision=f.revision,capture_timestamp=f.now,clock_id='mono')
        assert not c.input_is_current(packet,ui,now=f.now,clock_id='mono')
        f.raw_left[0,3]+=.025;f.raw_right[2,3]+=.025
        for _ in range(20):tick()
        f.left=np.array(w.calls[-1][1]['left_base_ee']);f.right=np.array(w.calls[-1][1]['right_base_ee'])
        np.testing.assert_allclose(f.left[:3,3],original_left[:3,3]+[.025*(iteration+1),0,0])
        np.testing.assert_allclose(f.right[:3,3],original_right[:3,3]+[0,0,.025*(iteration+1)])
    assert c.active


def test_short_gap_recovers_orientation_without_changing_rotation_reference():
    from scipy.spatial.transform import Rotation
    f,w,c,tick=make();tick()
    packet=f.packet();packet['tracking_valid']=False
    ui=dict(revision=f.revision,capture_timestamp=f.now,clock_id='mono')
    assert not c.input_is_current(packet,ui,now=f.now,clock_id='mono')
    delta=Rotation.from_euler('z',30,degrees=True).as_matrix()
    f.raw_right[:3,:3]=delta@f.raw_right[:3,:3]
    previous=f.right[:3,:3].copy()
    for _ in range(40):
        tick();actual=np.array(w.calls[-1][1]['right_base_ee'])[:3,:3]
        assert Rotation.from_matrix(actual@previous.T).magnitude()<=.020001
        previous=actual
    np.testing.assert_allclose(previous,delta@f.right[:3,:3],atol=1e-8)
    assert c.active and c.recovery_targets is None


def test_host_lease_release_acknowledges_even_invalid_tracking_and_keeps_calibration():
    f,w,c,tick=make();f.raw_right[0,0]=float('nan')
    assert tick(release=True) is None
    assert not c.active and c.anchors is not None and c.revision==f.revision


def test_real_button_parser_and_dashboard_bridge_ABXY_then_AX_stay_active(tmp_path):
    from test_copilot_vr_buttons import Source,bridge,CALIBRATE,START
    from teleop_assist.copilot.mailbox import read_json
    source=Source(tmp_path/'operator.json');dashboard=bridge(source,tmp_path)
    worker=OriginalWorker();core=ManualSimControl(worker,simulation_mode=True)
    f=Fixture()
    def event(buttons=()):
        packet=source.sample(buttons)
        runtime=dict(core.status,timestamp=source.now,clock_id=source.clock_id,candidates=[])
        dashboard.consume_vr(packet,runtime_status=runtime,now=source.now)
        ui=read_json(dashboard.sim_control.path,{})
        return core.step(packet,ui,(f.left,f.right),np.zeros(14),np.zeros(14),
            now=source.now,clock_id=source.clock_id)
    event();assert event(CALIBRATE) is None
    assert core.anchors is not None and not core.active
    event();assert event(START) is None and core.active
    for _ in range(30):
        assert event() is not None
        assert core.active and core.anchors is not None
    assert event(START) is None and not core.active
    assert event() is None


def test_choose_and_confirm_before_ABXY_survives_calibration_and_AX(tmp_path):
    from test_copilot_vr_buttons import Source,bridge,CALIBRATE,START
    from test_copilot_manual_grasp_assist import fixture
    from teleop_assist.copilot.manual_grasp_assist import ManualGraspAssist
    from teleop_assist.copilot.mailbox import read_json
    source=Source(tmp_path/'operator.json');dashboard=bridge(source,tmp_path)
    worker=OriginalWorker();assist=ManualGraspAssist(simulation_mode=True)
    core=ManualSimControl(worker,simulation_mode=True,target_filter=assist.apply)
    pose,_,batch,frame=fixture()
    batch['clock_id']=frame['clock_id']=source.clock_id
    def event(buttons=()):
        packet=source.sample(buttons)
        batch['capture_timestamp']=frame['timestamp']=source.now
        runtime=dict(assist.annotate(core.status),timestamp=source.now,clock_id=source.clock_id)
        dashboard.consume_vr(packet,runtime_status=runtime,now=source.now)
        ui=read_json(dashboard.sim_control.path,{})
        assist.update(batch,ui,frame,now=source.now,clock_id=source.clock_id,active=core.active)
        return core.step(packet,ui,(pose,pose),np.zeros(14),np.zeros(14),
            now=source.now,clock_id=source.clock_id)
    event();event(['left.bButton']);event();event(['right.bButton']);event()
    assert assist.confirmed['candidate_id']=='learned:1' and core.anchors is None
    assert event(CALIBRATE) is None
    assert assist.confirmed['candidate_id']=='learned:1'
    assert core.anchors is not None and not core.active and not assist.applied
    event();assert event(START) is None and core.active
    assert assist.confirmed['candidate_id']=='learned:1'
    event();assert not assist.applied
    np.testing.assert_allclose(worker.calls[-1][1]['right_base_ee'],pose)
    event(['right.squeeze']);event(['right.squeeze'])
    assert assist.applied and assist.translation[0]>0
    event();assert event(CALIBRATE) is None
    assert core.anchors is None and not core.active
    assert assist.selected is None and assist.confirmed is None


def test_long_tracking_loss_can_resume_AX_after_dashboard_ownership_expires(tmp_path):
    from test_copilot_vr_buttons import Source,bridge,CALIBRATE,START
    from teleop_assist.copilot.mailbox import read_json
    source=Source(tmp_path/'operator.json');dashboard=bridge(source,tmp_path)
    worker=OriginalWorker();core=ManualSimControl(worker,simulation_mode=True);f=Fixture()
    def event(buttons=()):
        packet=source.sample(buttons)
        runtime=dict(core.status,timestamp=source.now,clock_id=source.clock_id,candidates=[])
        dashboard.consume_vr(packet,runtime_status=runtime,now=source.now)
        return core.step(packet,read_json(dashboard.sim_control.path,{}),(f.left,f.right),
            np.zeros(14),np.zeros(14),now=source.now,clock_id=source.clock_id)
    event();event(CALIBRATE);event();event(START);event()
    core._input_unavailable('test_dropout',source.now)
    source.now+=2.5;core._input_unavailable('test_dropout',source.now)
    assert not core.active and core.anchors is not None
    assert event() is None
    assert dashboard.sim_ownership.client_id is not None
    assert event(START) is None and core.active
    assert event() is not None


def test_valid_webxr_float32_roundoff_does_not_cause_tracking_pause():
    from teleop_assist.copilot.controller import rigid
    f,w,c,tick=make()
    f.raw_left[:3,:3]*=1.000001;f.raw_right[:3,:3]*=1.000001
    # This was accepted upstream but wrongly rejected by the assist-only gate.
    with pytest.raises(ValueError):rigid(f.raw_right)
    original_positions=(f.raw_left[:3,3].copy(),f.raw_right[:3,3].copy())
    for _ in range(25):
        assert tick() is not None and c.active and c.anchors is not None
        for key in ('left_base_ee','right_base_ee'):rigid(w.calls[-1][1][key])
    np.testing.assert_array_equal(f.raw_left[:3,3],original_positions[0])
    np.testing.assert_array_equal(f.raw_right[:3,3],original_positions[1])
    # A genuine scale error still fails the original input validator.
    f.raw_right[:3,:3]*=1.01
    n=len(w.calls)
    assert tick() is None and len(w.calls)==n


def test_expired_button_event_does_not_synthesize_release_while_pose_bound_is_current(tmp_path):
    from test_copilot_vr_buttons import Source,bridge,CALIBRATE,START
    from teleop_assist.copilot.mailbox import read_json
    from tools.copilot.xr_source import validate_operator_packet
    source=Source(tmp_path/'operator.json');dashboard=bridge(source,tmp_path)
    worker=OriginalWorker();core=ManualSimControl(worker,simulation_mode=True);f=Fixture()
    def event(buttons=()):
        packet=source.sample(buttons)
        runtime=dict(core.status,timestamp=source.now,clock_id=source.clock_id,candidates=[])
        dashboard.consume_vr(packet,runtime_status=runtime,now=source.now)
        core.step(packet,read_json(dashboard.sim_control.path,{}),(f.left,f.right),np.zeros(14),np.zeros(14),
            now=source.now,clock_id=source.clock_id)
        return packet
    event();event(CALIBRATE);event();event(START);packet=event()
    old=read_json(dashboard.sim_control.path);owner=dashboard.sim_ownership.client_id
    source.now+=.151
    # Vuer may invoke the callback again for the same XR event: the host
    # receipt/publication is new, while acquisition and button expiry stay old.
    packet['published_at']=source.now
    packet['envelope']['received_at_host_monotonic']=source.now
    packet['vr_buttons']['received_at_host_monotonic']=source.now
    packet['expires_at']=packet['acquisition_timestamp']+.15+packet['envelope']['clock_uncertainty_s']
    assert validate_operator_packet(packet,now=source.now,clock_id=source.clock_id) is None
    result=dashboard.consume_vr(packet,runtime_status=dict(core.status,timestamp=source.now,
        clock_id=source.clock_id,candidates=[]),now=source.now)
    assert not result['written'] and result['reason'].startswith('manual_tracking_hold:')
    assert read_json(dashboard.sim_control.path)==old
    assert dashboard.sim_ownership.client_id==owner
    assert not core.input_is_current(packet,old,now=source.now,clock_id=source.clock_id)
    assert core.active and core.anchors is not None
    event();event();assert core.active
    event(CALIBRATE);assert not core.active and core.anchors is None
