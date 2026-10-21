import ast
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools.copilot.instrument_xr import (BROWSER_HELPER, BROWSER_CHUNK, DEFAULT_CLIENT,
    PINNED_FILES, SharedXRTelemetry, digest, instrument_browser, instrument_entry, stage_checkout)
from tools.copilot.launch_xr import NODE, checked_arguments
from teleop_assist.copilot.xr_adapter import host_clock_id

REPO = Path('/home/loya2/xr_teleoperate')


def event(kind='hand', timestamp=9.98, with_metadata=True):
    pose = np.eye(4).flatten(order='F').tolist()
    value = {'camera': {'matrix': pose}} if kind == 'head' else {
        'left': pose * 25, 'right': pose * 25, 'leftState': {}, 'rightState': {}}
    if with_metadata:
        value['_copilot'] = dict(schema='punktnull.xr_frame.v1', sequence=9,
            sample_kind='camera_pose_callback' if kind == 'head' else 'webxr_pose_read',
            sample_time_s=1.2, sample_clock_id='browser:session',
            host_sample_time_s=timestamp, host_clock_id=host_clock_id(), clock_uncertainty_s=.003,
            clock_sync_age_s=.4, sensor_capture_time_s=None, latency_unknown=False,
            left_tracking_valid=True, right_tracking_valid=True)
    return value


def test_receipt_time_never_relabels_uninstrumented_events_as_samples():
    telemetry = SharedXRTelemetry()
    telemetry.record('head', event('head', with_metadata=False), now=10.)
    telemetry.record('hand', event(with_metadata=False), now=10.)
    result = telemetry.envelope()
    assert result['acquisition_timestamp'] is None and result['latency_unknown']
    assert result['received_at_host_monotonic'] == 10.
    assert result['sensor_capture_timestamp'] is None


def test_atomic_pose_envelope_keeps_oldest_source_capture_not_receipt():
    telemetry = SharedXRTelemetry()
    telemetry.record('head', event('head', timestamp=9.95), now=9.97)
    telemetry.record('hand', event(timestamp=9.98), now=10.)
    result = telemetry.envelope()
    assert result['acquisition_timestamp'] == 9.95
    assert result['tracking_valid'] and not result['latency_unknown']


def test_current_xrframe_head_and_controllers_replace_stale_camera_callback_together():
    telemetry = SharedXRTelemetry()
    telemetry.record('head', event('head', timestamp=1.), now=1.)
    sample=event(timestamp=9.98)
    pose=np.eye(4).flatten(order='F').tolist()
    sample.update(left=pose,right=pose,_copilot_viewer_matrix=pose)
    telemetry.record('controller',sample,now=10.)
    result=telemetry.envelope()
    assert result['tracking_valid'] and result['acquisition_timestamp']==9.98
    assert result['raw_event_metadata']['head']['sequence']==result['raw_event_metadata']['arms']['sequence']
    sample['_copilot_viewer_matrix']=None
    telemetry.record('controller',sample,now=10.01)
    assert not telemetry.envelope()['tracking_valid']


def test_uncertain_clock_exchange_does_not_replace_good_estimate_or_extend_expiry():
    code='''const vm=require('vm');let now=1000,slow=false;
const context={performance:{now:()=>now,timeOrigin:1234},fetch:async()=>{
 if(slow)now+=200;
 return {ok:true,json:async()=>({host_time_s:slow?99:11,clock_id:'host'})};},setInterval:()=>0};
context.globalThis=context;vm.createContext(context);vm.runInContext(SOURCE,context);
(async()=>{await context.__punktnullCopilot.sync();now=3500;slow=true;
 await context.__punktnullCopilot.sync();const retained=context.__punktnullCopilot.sample('webxr_pose_read');
 now=6100;const expired=context.__punktnullCopilot.sample('webxr_pose_read');
 console.log(JSON.stringify({retained,expired}));})();'''.replace('SOURCE',json.dumps(BROWSER_HELPER))
    result=subprocess.run([str(NODE),'-e',code],capture_output=True,text=True,check=True,timeout=10)
    values=json.loads(result.stdout)
    assert values['retained']['host_sample_time_s']==pytest.approx(13.7)
    assert values['retained']['clock_uncertainty_s']==pytest.approx(.002)
    assert values['expired']['latency_unknown'] and values['expired']['host_sample_time_s'] is None


@pytest.mark.parametrize('alteration', ['missing_head', 'old_mapping', 'wrong_clock', 'lost_hand'])
def test_missing_stale_or_invalid_sources_cannot_form_valid_fresh_envelope(alteration):
    telemetry = SharedXRTelemetry()
    if alteration != 'missing_head':
        telemetry.record('head', event('head'), now=10.)
    sample = event()
    if alteration == 'old_mapping': sample['_copilot']['clock_sync_age_s'] = 6.
    if alteration == 'wrong_clock': sample['_copilot']['host_clock_id'] = 'different_host'
    if alteration == 'lost_hand': sample['right'] = [0.] * 400
    telemetry.record('hand', sample, now=10.)
    result = telemetry.envelope()
    assert result['latency_unknown'] or not result['tracking_valid']


def test_invalid_new_event_replaces_previous_validity():
    telemetry = SharedXRTelemetry()
    telemetry.record('head', event('head'), now=10.)
    telemetry.record('hand', event(), now=10.)
    assert telemetry.envelope()['tracking_valid']
    telemetry.record('hand', {}, now=10.01)
    assert not telemetry.envelope()['tracking_valid']


def test_browser_clock_mapping_sequence_and_expiration_without_network():
    code = '''const vm=require('vm');let timer=0;let now=1000;
const context={performance:{now:()=>now,timeOrigin:1234},fetch:async()=>({ok:true,json:async()=>({host_time_s:11,clock_id:'host'})}),setInterval:()=>++timer};
context.globalThis=context;vm.createContext(context);vm.runInContext(SOURCE,context);
(async()=>{await context.__punktnullCopilot.sync();
const one=context.__punktnullCopilot.sample('webxr_pose_read');now=1020;
const two=context.__punktnullCopilot.sample('webxr_pose_read');now=7000;
const expired=context.__punktnullCopilot.sample('webxr_pose_read');
console.log(JSON.stringify({one,two,expired,timer}));})();'''.replace('SOURCE', json.dumps(BROWSER_HELPER))
    result = subprocess.run([str(NODE), '-e', code], capture_output=True, text=True, check=True, timeout=10)
    values = json.loads(result.stdout)
    assert values['one']['sample_time_s'] == 1.
    assert values['one']['host_sample_time_s'] == 11.
    assert values['two']['sequence'] == values['one']['sequence'] + 1
    assert values['two']['host_sample_time_s'] == pytest.approx(11.02)
    assert values['expired']['latency_unknown']
    assert values['expired']['host_sample_time_s'] is None
    assert values['one']['sensor_capture_time_s'] is None


def test_reviewed_source_patches_fail_on_drift_and_compile(tmp_path):
    source = (REPO / 'teleop/teleop_hand_and_arm.py').read_text()
    patched = instrument_entry(source)
    ast.parse(patched)
    assert 'current_lr_arm_q.tolist()' in patched
    assert 'joint_state_acquisition_timestamp=None' in patched
    assert 'copilot_adapter.input_hold_required' in patched
    with pytest.raises(ValueError, match='patch anchor'):
        instrument_entry(source.replace('    args = parser.parse_args()', '    args=parser.parse_args()'))
    browser = instrument_browser((DEFAULT_CLIENT / BROWSER_CHUNK).read_text())
    path = tmp_path / 'instrumented.mjs'; path.write_text(browser)
    subprocess.run([str(NODE), '--check', str(path)], check=True, capture_output=True, timeout=10)
    assert 'Te.left=void 0;Te.right=void 0' in browser
    assert 'pe.left=void 0;pe.right=void 0' in browser


def test_actual_browser_controller_indices_and_capabilities_are_instrumented():
    browser=instrument_browser((DEFAULT_CLIENT/BROWSER_CHUNK).read_text())
    state_function=browser[browser.index('function getState('):browser.index('function MotionControllers(')]
    code='''function getTargetRayTransform(){return [Array(16).fill(0),null]};
'''+state_function+'''
const buttons=Array.from({length:6},()=>({pressed:false,value:0}));
buttons[1].pressed=true;buttons[4].pressed=true;
const available=getState(null,{gamepad:{mapping:'xr-standard',buttons,axes:[]}},null);
const missing=getState(null,{gamepad:{mapping:'xr-standard',buttons:buttons.slice(0,4),axes:[]}},null);
console.log(JSON.stringify({available,missing}));
'''
    result=subprocess.run([str(NODE),'-e',code],capture_output=True,text=True,check=True,timeout=10)
    values=json.loads(result.stdout)
    assert values['available']['squeeze'] and values['available']['aButton'] and not values['available']['bButton']
    assert values['available']['_copilot_gamepad']['required_buttons_available']
    assert not values['missing']['_copilot_gamepad']['required_buttons_available']


def test_staging_does_not_change_original_and_contains_reviewable_patch(tmp_path):
    before = {name: digest(REPO / name) for name in PINNED_FILES}
    target = tmp_path / 'staged'
    manifest = stage_checkout(REPO, target)
    assert (target / 'original-entry.patch').is_file()
    assert (target / 'assets').is_symlink()
    assert manifest['original_mutated'] is False
    assert {name: digest(REPO / name) for name in PINNED_FILES} == before
    assert digest(target / 'teleop/teleop_hand_and_arm.py') != before['teleop/teleop_hand_and_arm.py']
    with pytest.raises(ValueError, match='exists'):
        stage_checkout(REPO, target)


def test_actual_original_vuer_event_wrapper_is_instrumented_without_server():
    # __new__ plus shared-memory fixtures exercise the original event handlers
    # and coordinate conversion, deliberately bypassing its server constructor.
    project = Path(__file__).resolve().parents[1]
    code = r'''
import asyncio,json,time
from types import SimpleNamespace
from multiprocessing import Array,Value
import numpy as np
from televuer.televuer import TeleVuer
from televuer.tv_wrapper import TeleVuerWrapper
from tools.copilot.instrument_xr import SharedXRTelemetry,install_instrumentation,DEFAULT_CLIENT
from teleop_assist.copilot.xr_adapter import host_clock_id
original_get=TeleVuerWrapper.get_tele_data
install_instrumentation(DEFAULT_CLIENT)
tv=TeleVuer.__new__(TeleVuer);tv._copilot_telemetry=SharedXRTelemetry()
for name,size in [('head_pose',16),('left_arm_pose',16),('right_arm_pose',16),('left_hand_position',75),('right_hand_position',75),('left_hand_orientation',225),('right_hand_orientation',225)]:
 setattr(tv,name+'_shared',Array('d',size,lock=True))
for side in ('left','right'):
 for field in ('pinch','pinchValue','squeeze','squeezeValue'):
  setattr(tv,side+'_hand_'+field+'_shared',Value('d',0.,lock=True))
pose=np.eye(4).flatten(order='F').tolist()
stamp=time.monotonic()
meta=dict(schema='punktnull.xr_frame.v1',sequence=1,sample_kind='webxr_pose_read',sample_time_s=1.,sample_clock_id='browser:test',host_sample_time_s=stamp,host_clock_id=host_clock_id(),clock_uncertainty_s=.003,clock_sync_age_s=.1,sensor_capture_time_s=None,latency_unknown=False,left_tracking_valid=True,right_tracking_valid=True)
asyncio.run(tv.on_cam_move(SimpleNamespace(value={'camera':{'matrix':pose},'_copilot':meta}),None))
hands={'left':pose*25,'right':pose*25,'leftState':{},'rightState':{},'_copilot':meta}
asyncio.run(tv.on_hand_move(SimpleNamespace(value=hands),None))
wrapper=TeleVuerWrapper.__new__(TeleVuerWrapper);wrapper.tvuer=tv;wrapper.use_hand_tracking=True;wrapper.return_hand_rot_data=False
before=original_get(wrapper);after=wrapper.get_tele_data()
unchanged=np.array_equal(before.right_wrist_pose,after.right_wrist_pose) and np.array_equal(before.left_wrist_pose,after.left_wrist_pose)
assert after.copilot_envelope['tracking_valid'] and not after.copilot_envelope['latency_unknown']
asyncio.run(tv.on_hand_move(SimpleNamespace(value={}),None))
lost=wrapper.get_tele_data()
for side in ('left','right'):
 for field in ('trigger','triggerValue','squeeze','squeezeValue','thumbstick','aButton','bButton'):
  setattr(tv,side+'_ctrl_'+field+'_shared',Value('d',0.,lock=True))
 setattr(tv,side+'_ctrl_thumbstickValue_shared',Array('d',2,lock=True))
states={side:dict(aButton=side=='left',bButton=False,squeeze=side=='right',triggerValue=.25 if side=='left' else .8,_copilot_gamepad=dict(mapping='xr-standard',button_count=6,required_buttons_available=True)) for side in ('left','right')}
controllers=dict(left=pose,right=pose,leftState=states['left'],rightState=states['right'],_copilot=dict(meta,sequence=2))
controllers['_copilot_viewer_matrix']=pose
controllers['_copilot_viewer_matrix'][12]=.15
asyncio.run(tv.on_controller_move(SimpleNamespace(value=controllers),None))
assert tv.head_pose_shared[12]==.15
asyncio.run(tv.on_cam_move(SimpleNamespace(value={'camera':{'matrix':np.eye(4).flatten(order='F').tolist()},'_copilot':meta}),None))
assert tv.head_pose_shared[12]==.15
wrapper.use_hand_tracking=False
before=original_get(wrapper);after=wrapper.get_tele_data()
assert np.array_equal(before.left_wrist_pose,after.left_wrist_pose) and np.array_equal(before.right_wrist_pose,after.right_wrist_pose)
raw=after.copilot_envelope['raw_event_metadata']['arms']
assert after.copilot_envelope['raw_event_metadata']['head']['sequence']==2
assert raw['event_kind']=='controller' and raw['sequence']==2 and raw['controller_states']['left']['aButton'] is True
assert raw['controller_states']['right']['squeeze'] is True and after.left_ctrl_aButton and after.right_ctrl_squeeze
assert raw['controller_states']['left']['triggerValue']==.25 and raw['controller_states']['right']['triggerValue']==.8
print(json.dumps({'unchanged':unchanged,'tracking_after_invalid_event':lost.copilot_envelope['tracking_valid'],'server_instantiated':False,'controller_event_verified':True}))
'''
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join([str(project), str(REPO / 'teleop/televuer/src')]))
    result = subprocess.run(['/home/loya2/miniconda3/envs/to/bin/python', '-c', code],
        capture_output=True, text=True, env=environment, check=True, timeout=20)
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output['unchanged'] and not output['tracking_after_invalid_event'] and output['controller_event_verified']


@pytest.mark.parametrize('args', [[], ['--sim', '--network-interface', 'eth0'], ['--sim', '--motion'], ['--sim', '--arm', 'H1']])
def test_launcher_rejects_hardware_external_network_and_motion_modes(tmp_path, args):
    config = tmp_path / 'config.json'; config.write_text('{"xr":{"mode":"shadow"}}')
    with pytest.raises(SystemExit):
        checked_arguments(['--assist-config', str(config), *args])


def test_regular_bounded_clock_sync_has_no_periodic_five_second_tracking_gap():
    # A slightly worse but still bounded RTT must replace a good estimate
    # BEFORE its five-second expiry, instead of leaving a gap until next sync.
    code='''const vm=require('vm');let now=1000,rtt=2;
const context={performance:{now:()=>now,timeOrigin:1234},fetch:async()=>{
 const before=now;now+=rtt;
 return {ok:true,json:async()=>({host_time_s:10+(before+rtt/2)/1000,clock_id:'host'})};},setInterval:()=>0};
context.globalThis=context;vm.createContext(context);vm.runInContext(SOURCE,context);
(async()=>{await context.__punktnullCopilot.sync();rtt=8;const gaps=[];
 for(let i=1;i<=400;i++){now=1000+i*50;
  if(i%20===0)await context.__punktnullCopilot.sync();
  const sample=context.__punktnullCopilot.sample('webxr_pose_read');
  if(sample.latency_unknown)gaps.push(now);
 }
 console.log(JSON.stringify(gaps));})();'''.replace('SOURCE',json.dumps(BROWSER_HELPER))
    result=subprocess.run([str(NODE),'-e',code],capture_output=True,text=True,check=True,timeout=10)
    assert json.loads(result.stdout)==[]
