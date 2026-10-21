#!/usr/bin/env python3
"""Exercise production JS eye-ray packets against the real CPU HTTP/core service.

The ray, head pose and observed plane are synthetic. No headset, eye tracker,
Isaac simulation or robot controller is started.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop_assist.core import AssistSession
from teleop_assist.server import AssistHTTPServer

NODE_TEST = r"""
import assert from 'node:assert/strict';
const {eyeProjectionCommand}=await import(process.argv[1]);
const base=process.argv[2];
const claim=await fetch(base+'/api/command',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({type:'control_claim',client_id:'synthetic-http-contract',deadman_released:true})});
assert.equal(claim.status,200,'Projection client must explicitly take browser control');
const data=await (await fetch(base+'/api/gaze')).json();
assert.equal(data.status,'projection_required');
const context={enabled:true,immersive:true,replay:false,cameraFresh:true,
 ageSeconds:data.device_age,headToStage:[[1,0,0,.1],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
 frameId:1,clientId:'synthetic-http-contract',epoch:1,seq:1};
// An observed synthetic plane at optical z=1 m, not a production fallback.
const observedPlane=(origin,direction)=>{
 const t=(-1-origin[2])/direction[2];if(t<=0)return null;
 const p=origin.map((v,i)=>v+t*direction[i]);
 return [(p[0]*500+320)/640,(-p[1]*500+180)/360];
};
const packet=eyeProjectionCommand(data.device_sample,context,observedPlane);
assert.deepEqual(packet.sample.uv,[370/640,.5]);
assert.equal(packet.sample.projection_of_session,'synthetic-device');
assert.equal(packet.sample.projection_of_seq,7);
for(const change of [{enabled:false},{immersive:false},{replay:true},{cameraFresh:false},{ageSeconds:.251}])
 assert.equal(eyeProjectionCommand(data.device_sample,{...context,...change},observedPlane),null);
assert.equal(eyeProjectionCommand({...data.device_sample,calibrated:false},context,observedPlane),null);
assert.equal(eyeProjectionCommand({...data.device_sample,confidence:NaN},context,observedPlane),null);
assert.equal(eyeProjectionCommand(data.device_sample,context,()=>null),null,'Unobserved space must not create gaze');
const response=await fetch(base+'/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(packet)});
assert.equal(response.status,200,await response.clone().text());
assert.equal((await response.json()).accepted,true);
const wrong={...packet,sample:{...packet.sample,seq:2,projection_of_seq:999}};
const rejected=await fetch(base+'/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(wrong)});
assert.equal(rejected.status,400,'Backend must reject a fabricated device linkage');
console.log(JSON.stringify({packet,checks:['real GET device_sample contract','production JS linked projection payload accepted by real POST/core','metric translated-head plane projection','disabled/nonimmersive/replay/stale/unobserved rays produce no proposal','invalid device linkage rejected by backend']}));
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', default=shutil.which('node'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.node:
        parser.error('Supply --node with a Node.js executable')
    if args.output and args.output.exists():
        parser.error('Output already exists; choose a new report path')
    # A fixed clock makes freshness checks deterministic across process startup.
    session = AssistSession(clock=lambda: 1.)
    session.config.eye_enabled = True
    session.update_frame([{'id': 'observed_part', 'bbox_px': [350, 160, 390, 200],
                           'center_world': [.5, 0, 1.], 'normal_world': [-1, 0, 0]}],
                         1, 1., {'width': 640, 'height': 360})
    session.set_gaze({'source': 'synthetic_device', 'session_id': 'synthetic-device',
                      'seq': 7, 'head_ray': {'origin': [0, 0, 0], 'direction': [0, 0, -1]},
                      'valid': True, 'calibrated': True, 'confidence': .92})
    server = AssistHTTPServer(session, port=0)
    server.start()
    try:
        module = ROOT / 'teleop_assist/static/gaze.mjs'
        result = subprocess.run([args.node, '--input-type=module', '-e', NODE_TEST,
                                 module.as_uri(), f'http://{server.address[0]}:{server.address[1]}'],
                                check=True, capture_output=True, text=True, timeout=20)
        details = json.loads(result.stdout)
        assert session.eye_status(1.) == 'active'
        assert session.selection.id == 'observed_part'
        assert not server.errors
        hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [module, module.with_name('geometry.mjs'),
                            ROOT / 'teleop_assist/core.py', ROOT / 'teleop_assist/server.py']}
        report = {'kind': 'synthetic_ray_real_js_http_core_protocol_test', 'passed': True,
                  'immersive_vr_tested': False, 'eye_hardware_tested': False,
                  'hashes': hashes, **details}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as output:
                json.dump(report, output, indent=2)
                output.write('\n')
        print(json.dumps(report, indent=2))
    finally:
        server.close()


if __name__ == '__main__':
    main()
