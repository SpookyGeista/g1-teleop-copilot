import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ReferenceTeleopControls,referencePosture,referenceInputRotation} from '../teleop_assist/static/reference_teleop.mjs';
const pad=(...down)=>({buttons:Array.from({length:6},(_,i)=>({value:down.includes(i)?1:0}))});
const good={trackingValid:true,ownsControl:true,stopped:false,poseReady:true};
const step=(c,t,l=[],r=[],extra={})=>c.update(pad(...l),pad(...r),{...good,now:t,...extra});
test('ABXY acquires/calibrates from observer; AX then streams without grip',()=>{
 const c=new ReferenceTeleopControls();
 assert.deepEqual(step(c,0,[4,5],[4,5],{ownsControl:false}),[]);
 assert.deepEqual(step(c,.7,[4,5],[4,5],{ownsControl:false}),['calibrate']);
 assert.equal(c.active,false);c.completeCalibration(true);step(c,.8);
 step(c,1,[4],[4]);assert.deepEqual(step(c,1.5,[4],[4]),['start']);assert.equal(c.active,true);
 step(c,1.6);assert.equal(c.active,true);
 assert.deepEqual(step(c,2,[4],[4]),['pause']);assert.equal(c.active,false);
});
test('calibration chord never triggers object, assist or resume controls',()=>{
 const c=new ReferenceTeleopControls();let actions=[];
 for(const [t,l,r] of [[0,[],[4]],[.1,[4],[4]],[.2,[4,5],[4,5]],[.9,[4,5],[4,5]],[1,[5],[]],[1.1,[],[]]])actions.push(...step(c,t,l,r));
 assert.deepEqual(actions,['calibrate']);
});
test('bad posture and tracking cannot calibrate; no repeated chord while held',()=>{
 const c=new ReferenceTeleopControls();step(c,0,[4,5],[4,5]);
 assert.deepEqual(step(c,1,[4,5],[4,5],{poseReady:false}),[]);
 assert.deepEqual(step(c,2,[4,5],[4,5],{trackingValid:false}),[]);
 assert.deepEqual(step(c,3,[4,5],[4,5]),['calibrate']);
 assert.deepEqual(step(c,4,[4,5],[4,5]),[]);
});
test('ABXY stops even when AX was detected a frame earlier',()=>{
 const c=new ReferenceTeleopControls();c.completeCalibration(true);c.active=true;
 assert.deepEqual(step(c,0,[4],[4]),['pause']);
 assert.deepEqual(step(c,.1,[4,5],[4,5]),['stop']);assert.equal(c.active,false);
 step(c,.2);assert.deepEqual(step(c,.3,[4],[4],{stopped:true}),[]);
 assert.deepEqual(step(c,1,[4],[4],{stopped:true}),[]);
});
test('tracking and ownership loss require a deliberate restart',()=>{
 for(const lost of [{trackingValid:false},{ownsControl:false},{stopped:true}]){
  const c=new ReferenceTeleopControls();c.completeCalibration(true);c.active=true;
  step(c,0,[],[],lost);assert.equal(c.active,false);step(c,1);assert.equal(c.active,false);
 }
});
test('single A/B/Y act only on release',()=>{
 const c=new ReferenceTeleopControls();
 for(const [l,r,action] of [[[],[4],'confirm'],[[],[5],'release'],[[5],[],'assist']]){
  assert.deepEqual(step(c,0,l,r),[]);assert.deepEqual(step(c,1),[action]);
 }
});
test('posture check uses head heading and both measured controllers',()=>{
 const pose=p=>({valid:true,position:p,quaternion_xyzw:[0,0,0,1]});
 const h=pose([0,1.7,0]),l=pose([-.23,1.2,-.3]),r=pose([.23,1.2,-.3]);
 assert.equal(referencePosture(h,l,r).ready,true);
 assert.equal(referencePosture(h,{...l,valid:false},r).ready,false);
 assert.equal(referencePosture(h,r,l).ready,false);
 assert.equal(referencePosture(h,l,pose([.23,1.2,.3])).ready,false);
 const turn=p=>({...p,position:[-p.position[2],p.position[1],p.position[0]]});
 assert.equal(referencePosture({...turn(h),quaternion_xyzw:[0,-Math.SQRT1_2,0,Math.SQRT1_2]},turn(l),turn(r)).ready,true);
});
test('calibrated body axes are independent of the pitched robot camera',()=>{
 const mul=(r,v)=>r.map(row=>row.reduce((s,x,i)=>s+x*v[i],0));
 const approx=(a,b)=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-12));
 const basis=referenceInputRotation([0,0,0,1]);
 approx(mul(basis,[0,0,-.1]),[.1,0,0]);approx(mul(basis,[0,.1,0]),[0,0,.1]);
 approx(mul(basis,[.1,0,0]),[0,-.1,0]);
 const turned=referenceInputRotation([0,-Math.SQRT1_2,0,Math.SQRT1_2]);
 approx(mul(turned,[.1,0,0]),[.1,0,0]);
 assert.equal(referenceInputRotation([0,0,0,0]),null);
});
