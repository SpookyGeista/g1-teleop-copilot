"""Pinned, staged instrumentation of the local original Unitree/Vuer checkout.

Nothing in this module launches a server merely by being imported. Installation
is called only by a deliberately executed staged simulator launcher. Staging and
offline tests never instantiate TeleVuer or DDS.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from multiprocessing import Array, RLock
import os
from pathlib import Path
import shutil
import time

import numpy as np

from teleop_assist.copilot.xr_adapter import host_clock_id, rigid

PINNED_FILES = {
    'teleop/teleop_hand_and_arm.py': '7ec570d92e2583007ad2b409247c91d8fcc32de4d419a0f9a8a3c3bceffa3164',
    'teleop/televuer/src/televuer/tv_wrapper.py': '5a962fdd271745e68c940dd257fcfd99fc60c22c55d00092097d6a0b77d82e6b',
    'teleop/televuer/src/televuer/televuer.py': '48945f4e90eb158f62dcc0d6a7569e52960e8a5b462989544337af361861e15d',
    'teleop/robot_control/robot_arm_ik.py': '779bdcab2a8e276b3d55b7ea31a9c3b65afc240daa8f74d65243f1f59043fd68',
    'teleop/robot_control/robot_arm.py': '793e0774bb0a7a29b0cd163336babd45b29a36c3518528c87a854969c4132f73',
    'assets/g1/g1_body29_hand14.urdf': '8bbf006633fc50b616f665c7a970780cc296577a0adfd7d28b049e751c238735',
}
BROWSER_CHUNK = 'assets/chunks/chunk-Dd3xtWba.js'
BROWSER_SHA256 = '03fa38712fe2d00b7e83648d5e9e08d6ff1202c2129037a2610718af3c9a6a11'
DEFAULT_CLIENT = Path('/home/loya2/miniconda3/envs/to/lib/python3.10/site-packages/vuer/client_build')

# This is the time at which the browser reads poses from a WebXR frame, not a
# claim about the physical cameras/IMU inside the headset. Mapping uses a same-
# origin request/response interval; uncertainty includes half RTT plus 2 ms
# engineering allowance and expires after five seconds.
BROWSER_HELPER = r'''
;(function(){
  if(globalThis.__punktnullCopilot)return;
  const clock={mapping:null,sequence:0,nonce:String(performance.timeOrigin)};
  clock.sync=async function(){
    const before=performance.now()/1000;
    try{
      const answer=await fetch('/copilot/clock',{cache:'no-store'});
      if(!answer.ok)return;
      const value=await answer.json(),after=performance.now()/1000;
      if(typeof value.host_time_s!=='number'||typeof value.clock_id!=='string')return;
      const measured={offset:value.host_time_s-(before+after)/2,
        uncertainty_s:(after-before)/2+0.002,at:after,clock_id:value.clock_id};
      // Reject uncertain exchanges, but refresh a bounded estimate BEFORE
      // expiry. Waiting for expiry creates periodic invalid tracking between
      // polling instants whenever a newer RTT is not strictly better.
      if(measured.uncertainty_s>0.02)return;
      if(!clock.mapping||after-clock.mapping.at>=3||measured.uncertainty_s<clock.mapping.uncertainty_s)
        clock.mapping=measured;
    }catch(error){/* Missing mapping deliberately leaves assistance unavailable. */}
  };
  clock.viewer=function(frame,space){
    try{
      const pose=frame&&space&&frame.getViewerPose(space);
      return pose&&pose.transform?Array.from(pose.transform.matrix):null;
    }catch(error){return null;}
  };
  clock.sample=function(kind){
    const now=performance.now()/1000,m=clock.mapping;
    const mapped=!!m&&now-m.at<=5;
    return {schema:'punktnull.xr_frame.v1',sequence:++clock.sequence,
      sample_kind:kind,sample_time_s:now,sample_clock_id:'browser:'+clock.nonce,
      host_sample_time_s:mapped?now+m.offset:null,host_clock_id:mapped?m.clock_id:null,
      clock_uncertainty_s:mapped?m.uncertainty_s:null,
      clock_sync_age_s:mapped?now-m.at:null,sensor_capture_time_s:null,
      latency_unknown:!mapped,left_tracking_valid:false,right_tracking_valid:false};
  };
  globalThis.__punktnullCopilot=clock;
  clock.sync();setInterval(()=>clock.sync(),1000);
})();
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, old, new, description):
    if text.count(old) != 1:
        raise ValueError(f'Expected exactly one reviewed patch anchor: {description}')
    return text.replace(old, new, 1)


def instrument_browser(source):
    source = replace_once(source,
        'return{transform:Array.from(pe),trigger:',
        'return{_copilot_gamepad:{mapping:ge?ge.mapping:null,button_count:me.length,required_buttons_available:[1,4,5].every(i=>!!me[i]&&typeof me[i].pressed==="boolean")},transform:Array.from(pe),trigger:',
        'controller capability metadata')
    source = replace_once(source,
        'useFrame(function(Re,we,Pe){if(!Ae)return;const Ie=ye.current,$e=ve.current;',
        'useFrame(function(Re,we,Pe){if(!Ae)return;Te._copilot=globalThis.__punktnullCopilot.sample("webxr_pose_read");Te.left=void 0;Te.right=void 0;const Ie=ye.current,$e=ve.current;',
        'hand frame sample')
    source = replace_once(source, 'Te.left=Ue,Te.leftState=Ge}',
        'Te.left=Ue,Te.leftState=Ge,Te._copilot.left_tracking_valid=!!Ue&&Ue.length===400}', 'left hand tracking')
    source = replace_once(source, 'Te.right=Ue,Te.rightState=Ge,handState.right=Ge,handState.rightLandmarks=ke}',
        'Te.right=Ue,Te.rightState=Ge,handState.right=Ge,handState.rightLandmarks=ke,Te._copilot.right_tracking_valid=!!Ue&&Ue.length===400}', 'right hand tracking')
    source = replace_once(source,
        'useFrame(function(ge,me,Ae){if(!de)return;const{transform:ye,...ve}=getState(Ae,ce,le)||{},',
        'useFrame(function(ge,me,Ae){if(!de)return;pe._copilot=globalThis.__punktnullCopilot.sample("webxr_pose_read");pe._copilot_viewer_matrix=globalThis.__punktnullCopilot.viewer(Ae,le);pe.left=void 0;pe.right=void 0;const{transform:ye,...ve}=getState(Ae,ce,le)||{},',
        'controller frame sample')
    source = replace_once(source, 'pe.left=ye,pe.leftState=ve,controllerState.leftPose=ye,controllerState.left=ve)',
        'pe.left=ye,pe.leftState=ve,controllerState.leftPose=ye,controllerState.left=ve,pe._copilot.left_tracking_valid=!!ye&&ye.length===16)', 'left controller tracking')
    source = replace_once(source, 'pe.right=_e,pe.rightState=Te,controllerState.rightPose=_e,controllerState.right=Te)',
        'pe.right=_e,pe.rightState=Te,controllerState.rightPose=_e,controllerState.right=Te,pe._copilot.right_tracking_valid=!!_e&&_e.length===16)', 'right controller tracking')
    source = replace_once(source, 'value:{camera:{...Je,height:',
        'value:{_copilot:globalThis.__punktnullCopilot.sample("camera_pose_callback"),camera:{...Je,height:', 'head pose callback sample')
    return BROWSER_HELPER + '\n' + source


def instrument_entry(source):
    source = replace_once(source, 'from sshkeyboard import listen_keyboard, stop_listening\n',
        'from sshkeyboard import listen_keyboard, stop_listening\n'
        'from tools.copilot.instrument_xr import install_instrumentation\n'
        'from teleop_assist.copilot.xr_adapter import XRAdapter, ARM_JOINT_NAMES\n'
        'install_instrumentation(Path(os.environ["PUNKTNULL_XR_CLIENT_ROOT"]))\n', 'instrumentation imports')
    source = replace_once(source, 'import os \n', 'import os \nfrom pathlib import Path\n', 'Path import')
    source = replace_once(source, '    args = parser.parse_args()\n',
        '    parser.add_argument("--assist-config", required=True)\n'
        '    args = parser.parse_args()\n'
        '    if not args.sim or args.arm != "G1_29" or args.motion or args.network_interface != "lo":\n'
        '        raise ValueError("Copilot release 1 requires stationary G1_29 simulator mode")\n'
        '    copilot_adapter = XRAdapter.from_config(args.assist_config)\n', 'sim-only parser guard')
    old = '            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)'
    new = '''            operator_envelope = dict(getattr(tele_data, "copilot_envelope", {}))
            operator_envelope.update(backend="unitree_isaac_sim", dds_domain=1,
                current_lr_arm_q=current_lr_arm_q.tolist(), current_lr_arm_dq=current_lr_arm_dq.tolist(),
                joint_names=list(ARM_JOINT_NAMES), joint_state_acquisition_timestamp=None,
                joint_state_getter_timestamp=time.monotonic())
            copilot_left, copilot_right = copilot_adapter.process(
                tele_data.left_wrist_pose, tele_data.right_wrist_pose, operator_envelope)
            if copilot_adapter.mode == "assist" and copilot_adapter.input_hold_required:
                sol_q, sol_tauff = current_lr_arm_q.copy(), current_lr_arm_dq * 0.0
            else:
                sol_q, sol_tauff = arm_ik.solve_ik(copilot_left, copilot_right, current_lr_arm_q, current_lr_arm_dq)'''
    source = replace_once(source, old, new, 'pre-IK hook')
    ast.parse(source)
    return source


def stage_checkout(repo, destination, *, client_root=DEFAULT_CLIENT):
    repo, destination, client_root = map(lambda p: Path(p).resolve(), (repo, destination, client_root))
    if destination.exists():
        raise ValueError('Staging destination exists; choose a new reviewable directory')
    actual = {name: digest(repo / name) for name in PINNED_FILES}
    if actual != PINNED_FILES:
        raise ValueError('Original XR source differs from reviewed hashes; re-audit before staging')
    if digest(client_root / BROWSER_CHUNK) != BROWSER_SHA256:
        raise ValueError('Vuer browser source differs from reviewed hash; re-audit before staging')
    original_entry = (repo / 'teleop/teleop_hand_and_arm.py').read_text()
    original_browser = (client_root / BROWSER_CHUNK).read_text()
    new_entry, new_browser = instrument_entry(original_entry), instrument_browser(original_browser)
    destination.mkdir(parents=True)
    shutil.copytree(repo / 'teleop', destination / 'teleop',
                    ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '*model_cache.pkl', 'data'))
    (destination / 'assets').symlink_to(repo / 'assets', target_is_directory=True)
    (destination / 'teleop/teleop_hand_and_arm.py').write_text(new_entry)
    staged_client = destination / 'vuer_client_build'
    for original in client_root.rglob('*'):
        relative = original.relative_to(client_root)
        target = staged_client / relative
        if original.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif relative.as_posix() == BROWSER_CHUNK:
            target.write_text(new_browser)
        else:
            target.symlink_to(original)
    # Full browser bundle is retained for exact review; compact patch contexts
    # avoid duplicating its multi-megabyte minified line in the human diff.
    (destination / 'original-entry.patch').write_text(''.join(difflib.unified_diff(
        original_entry.splitlines(True), new_entry.splitlines(True),
        fromfile='original/teleop_hand_and_arm.py', tofile='staged/teleop_hand_and_arm.py')))
    manifest = {
        'schema': 'punktnull.xr_staging.v1', 'original_repo': str(repo),
        'original_source_sha256': actual, 'original_browser_sha256': BROWSER_SHA256,
        'staged_entry_sha256': digest(destination / 'teleop/teleop_hand_and_arm.py'),
        'staged_browser_sha256': digest(staged_client / BROWSER_CHUNK),
        'browser_source': str(client_root / BROWSER_CHUNK),
        'browser_change_scope': ['WebXR pose-read time/sequence/availability',
            'clear stale hand/controller values each frame', 'head callback time',
            'same-origin host monotonic clock mapping with measured RTT'],
        'original_mutated': False, 'dds_or_server_started_during_staging': False,
    }
    (destination / 'staging_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


class SharedXRTelemetry:
    """One shared lock covers event updates and wrapper pose+metadata reads."""
    def __init__(self):
        self.lock = RLock()
        self.buffer = Array('B', 32768, lock=False)
        self.buffer[:3] = b'{}\0'

    def read(self):
        with self.lock:
            raw = bytes(self.buffer[:]).split(b'\0', 1)[0]
            return json.loads(raw)

    def record(self, kind, event_value, *, now=None):
        now = time.monotonic() if now is None else now
        event_value = event_value if isinstance(event_value, dict) else {}
        metadata = event_value.get('_copilot', {})
        if not isinstance(metadata, dict):
            metadata = {}
        try:
            if kind == 'head':
                poses_valid = rigid(np.asarray(event_value['camera']['matrix'], float).reshape(4, 4, order='F'))
            else:
                expected_size = 400 if kind == 'hand' else 16
                poses_valid = all(
                    isinstance(event_value.get(side + 'State'), dict)
                    and np.asarray(event_value[side], float).shape == (expected_size,)
                    and np.isfinite(np.asarray(event_value[side], float)).all()
                    and rigid(np.asarray(event_value[side][:16], float).reshape(4, 4, order='F'))
                    for side in ('left', 'right'))
        except (KeyError, ValueError, TypeError):
            poses_valid = False
        sample = {name: metadata.get(name) for name in (
            'schema', 'sequence', 'sample_kind', 'sample_time_s', 'sample_clock_id',
            'host_sample_time_s', 'host_clock_id', 'clock_uncertainty_s', 'clock_sync_age_s',
            'sensor_capture_time_s', 'latency_unknown', 'left_tracking_valid', 'right_tracking_valid')}
        sample.update(received_at_host_monotonic=now, rigid_poses_valid=bool(poses_valid), event_kind=kind)
        if kind == 'controller':
            # Copy from the SAME pose event under the shared update lock. Do not
            # read separately updated TeleData boolean values after that event.
            sample['controller_states'] = {}
            for side in ('left', 'right'):
                state = event_value.get(side + 'State', {})
                state = state if isinstance(state, dict) else {}
                gamepad = state.get('_copilot_gamepad', {})
                gamepad = gamepad if isinstance(gamepad, dict) else {}
                sample['controller_states'][side] = {
                    **{name: state.get(name) if type(state.get(name)) is bool else None
                       for name in ('aButton', 'bButton', 'squeeze')},
                    'triggerValue': (state['triggerValue'] if type(state.get('triggerValue')) in (float,int)
                        and np.isfinite(state['triggerValue']) and 0 <= state['triggerValue'] <= 1 else None),
                    'gamepad': {
                        'mapping': gamepad.get('mapping') if isinstance(gamepad.get('mapping'), str) and len(gamepad['mapping'])<=64 else None,
                        'button_count': gamepad.get('button_count') if type(gamepad.get('button_count')) is int and 0<=gamepad['button_count']<=128 else None,
                        'required_buttons_available': gamepad.get('required_buttons_available') is True}}
        # Upstream without browser instrumentation remains explicitly unknown.
        sample['latency_unknown'] = sample['latency_unknown'] is not False
        with self.lock:
            if kind == 'controller' and '_copilot_viewer_matrix' in event_value:
                # Head and controller transforms were read from the same
                # actual XRFrame/reference space. A null head invalidates it;
                # never give an older camera callback a new source time.
                self.record('head', dict(camera=dict(matrix=event_value['_copilot_viewer_matrix']),
                    _copilot=metadata), now=now)
            state = self.read()
            state['head' if kind == 'head' else 'arms'] = sample
            encoded = json.dumps(state, allow_nan=False, separators=(',', ':')).encode()
            if len(encoded) + 1 > len(self.buffer):
                raise ValueError('XR telemetry record too large')
            self.buffer[:len(encoded) + 1] = encoded + b'\0'

    def envelope(self):
        records = self.read()
        head, arms = records.get('head', {}), records.get('arms', {})
        valid = bool(head.get('rigid_poses_valid') and arms.get('rigid_poses_valid')
                     and arms.get('left_tracking_valid') is True and arms.get('right_tracking_valid') is True)
        clock = host_clock_id()
        samples = [head, arms]
        mapped = all(record.get('schema') == 'punktnull.xr_frame.v1'
                     and record.get('host_clock_id') == clock
                     and record.get('latency_unknown') is False
                     and isinstance(record.get('host_sample_time_s'), (int, float))
                     and isinstance(record.get('clock_uncertainty_s'), (int, float))
                     and isinstance(record.get('clock_sync_age_s'), (int, float))
                     and 0 <= record['clock_sync_age_s'] <= 5 for record in samples)
        sample_kind = arms.get('sample_kind')
        provenance_kind = sample_kind if isinstance(sample_kind, str) else None
        return {
            'schema': 'punktnull.operator_envelope.v1',
            'sequence': arms.get('sequence', -1),
            'acquisition_timestamp': min(record['host_sample_time_s'] for record in samples) if mapped else None,
            'clock_id': clock if mapped else None,
            'clock_uncertainty_s': max(record['clock_uncertainty_s'] for record in samples) if mapped else None,
            'latency_unknown': not mapped,
            'sample_kind': arms.get('sample_kind'), 'tracking_valid': valid,
            'received_at_host_monotonic': min(record.get('received_at_host_monotonic', 0) for record in samples),
            'sensor_capture_timestamp': None, 'raw_event_metadata': records,
            'timestamp_provenance': ({
                'simulation_pose_sample': 'Synthetic browser event plus browser-to-host clock estimate; no physical headset sample',
                'webxr_pose_read': 'WebXR API pose-read plus browser-to-host clock estimate; physical sensor capture unknown',
            }.get(provenance_kind, 'Unknown pose source; physical sensor capture unknown')),
        }


def install_instrumentation(client_root):
    """Install composition hooks before the deliberately launched Vuer forks."""
    from televuer.televuer import TeleVuer
    from televuer.tv_wrapper import TeleVuerWrapper
    if getattr(TeleVuer, '_punktnull_instrumented', False):
        return
    original_init, original_run = TeleVuer.__init__, TeleVuer._vuer_run
    original_get = TeleVuerWrapper.get_tele_data

    def initialize(self, *args, **kwargs):
        self._copilot_telemetry = SharedXRTelemetry()
        original_init(self, *args, **kwargs)

    def run(self):
        self.vuer.client_root = Path(client_root)
        self.vuer.add_route('/copilot/clock', lambda: json.dumps({
            'host_time_s': time.monotonic(), 'clock_id': host_clock_id()}), content_type='application/json')
        return original_run(self)

    def get(self):
        with self.tvuer._copilot_telemetry.lock:
            data = original_get(self)
            data.copilot_envelope = self.tvuer._copilot_telemetry.envelope()
            return data

    for name, kind in (('on_cam_move', 'head'), ('on_hand_move', 'hand'), ('on_controller_move', 'controller')):
        original = getattr(TeleVuer, name)

        def wrap(function, event_kind):
            async def handler(self, event, session, fps=60):
                with self._copilot_telemetry.lock:
                    if event_kind == 'head' and getattr(self, '_copilot_coherent_head', False):
                        return
                    if event_kind == 'controller' and '_copilot_viewer_matrix' in event.value:
                        self._copilot_coherent_head = True
                        try:
                            matrix = np.asarray(event.value['_copilot_viewer_matrix'], float)
                            if matrix.shape == (16,) and rigid(matrix.reshape(4,4,order='F')):
                                with self.head_pose_shared.get_lock():
                                    self.head_pose_shared[:] = matrix.tolist()
                        except (ValueError, TypeError):
                            pass
                    # Record invalid events too; swallowed upstream exceptions
                    # must not leave an old sample looking fresh/valid.
                    self._copilot_telemetry.record(event_kind, event.value)
                    return await function(self, event, session, fps)
            return handler

        setattr(TeleVuer, name, wrap(original, kind))
    TeleVuer.__init__, TeleVuer._vuer_run = initialize, run
    TeleVuerWrapper.get_tele_data = get
    TeleVuer._punktnull_instrumented = True
