import assert from 'node:assert/strict';
import {intrinsics,backproject,projectOptical,decodeDepth,depthMesh,XRButtonEdges,normalizedPointer,xrToWorldRotation,headLocalRayInStage} from '../geometry.mjs';
const k=intrinsics({intrinsics:[[400,0,320],[0,400,180],[0,0,1]]});
assert.deepEqual(k,{fx:400,fy:400,cx:320,cy:180});
assert.deepEqual(backproject(400,220,2,k),[.4,-.2,-2]);
assert.deepEqual(projectOptical([.4,.2,2],k,640,360),[.625,220/360]);
assert.equal(projectOptical([1,1,-1],k,640,360),null);
const bytes=new ArrayBuffer(16),dv=new DataView(bytes);[1,1,1,1].forEach((v,i)=>dv.setFloat32(i*4,v,true));
assert.deepEqual([...decodeDepth(bytes,2,2)],[1,1,1,1]);
assert.throws(()=>decodeDepth(bytes,3,3));
let mesh=depthMesh(new Float32Array([1,1,1,1]),2,2,{fx:1,fy:1,cx:0,cy:0},1);
assert.equal(mesh.indices.length,6);assert.equal(mesh.validCount,4);assert.deepEqual([...mesh.positions.slice(3,6)],[1,-0,-1]);
mesh=depthMesh(new Float32Array([1,2,1,2]),2,2,{fx:1,fy:1,cx:0,cy:0},1);assert.equal(mesh.indices.length,0,'Depth discontinuities must not become artificial surfaces');
mesh=depthMesh(new Float32Array([NaN,1,0,Infinity]),2,2,{fx:1,fy:1,cx:0,cy:0},1);assert.equal(mesh.validCount,1);assert.equal(mesh.indices.length,0);
const buttons=new XRButtonEdges(),pad=(values)=>({buttons:Array.from({length:6},(_,i)=>({value:values[i]||0}))});
assert.deepEqual(buttons.update(pad({5:1}),{assist:{enabled:false}}),[{type:'assist_enable',enabled:true}]);
assert.deepEqual(buttons.update(pad({5:1}),{assist:{enabled:true}}),[],'Holding a toggle must not oscillate');
buttons.update(pad({}));assert.deepEqual(buttons.update(pad({3:1}),{eye_enabled:false}),[{type:'eye_enable',enabled:true}]);
buttons.update(pad({}));assert.deepEqual(buttons.update(pad({4:1}),{stopped:true}),[{type:'resume'}]);
buttons.update(pad({}));assert.deepEqual(buttons.update(pad({0:1,1:1,4:1}),{}),[{type:'stop'}],'Stop takes priority over calibration');
buttons.reset();assert.deepEqual(buttons.update(pad({4:1}),{}),[{type:'calibrate'}]);
assert.deepEqual(normalizedPointer({clientX:400,clientY:300},{left:0,top:0,width:800,height:600},640,360),[.5,.5]);
assert.equal(normalizedPointer({clientX:400,clientY:20},{left:0,top:0,width:800,height:600},640,360),null,'Letterbox must reject pointer');
console.log('PASS: camera projection, optical/GL basis, little-endian depth, depth discontinuity, invalid depth, XR toggle edges/stop priority, letterbox mapping.');

const frontCamera=[[0,0,1,1],[-1,0,0,2],[0,-1,0,3],[0,0,0,1]];
const identityMap=xrToWorldRotation(frontCamera,[0,0,0,1]);
assert.deepEqual(identityMap,[[0,0,-1],[-1,0,0],[0,1,0]],'XR forward maps to robot forward for camera facing world X');
const rotatedMap=xrToWorldRotation(frontCamera,[0,Math.SQRT1_2,0,Math.SQRT1_2]);
function apply(m,v){return m.map(row=>row.reduce((s,a,i)=>s+a*v[i],0));}
const movedInStage=[-1,0,0]; // head anchor yaw +90: visual forward is XR-stage -X
assert(apply(rotatedMap,movedInStage).every((v,i)=>Math.abs(v-[1,0,0][i])<1e-12),'Anchor yaw must be removed before mapping robot motion');
const pitch=.3,pitchedCamera=[[0,-Math.sin(pitch),Math.cos(pitch),0],[-1,0,0,0],[0,-Math.cos(pitch),-Math.sin(pitch),0],[0,0,0,1]];
const forwardWorld=apply(xrToWorldRotation(pitchedCamera,[0,0,0,1]),[0,0,-1]);
assert(Math.abs(forwardWorld[0]-Math.cos(pitch))<1e-12);assert(Math.abs(forwardWorld[2]+Math.sin(pitch))<1e-12,'Optical pitch must map to downward world approach');
console.log('PASS: XR/world displacement mapping handles camera pitch and initial headset yaw.');

// Independent synthetic eye rays through a metric plane: head yaw and lateral
// viewpoint shifts must be accounted for before camera-pixel projection.
const headYaw90=[[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]];
let eyeRay=headLocalRayInStage({origin:[0,0,0],direction:[1,0,0]},headYaw90);
assert.deepEqual(eyeRay.direction,[0,0,-1]);
const translatedHead=[[1,0,0,.1],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
eyeRay=headLocalRayInStage({origin:[0,0,0],direction:[0,0,-1]},translatedHead);
function uvOnPlane(ray,z){const t=(-z-ray.origin[2])/ray.direction[2],hit=ray.origin.map((v,i)=>v+t*ray.direction[i]);return projectOptical([hit[0],-hit[1],-hit[2]],{fx:500,fy:500,cx:320,cy:180},640,360);}
assert.deepEqual(uvOnPlane(eyeRay,1),[370/640,.5]);
assert.deepEqual(uvOnPlane(eyeRay,2),[345/640,.5],'Eye parallax must depend on actual surface depth');
assert.equal(headLocalRayInStage({origin:[0,0,0],direction:[0,0,-2]},translatedHead),null,'Uncalibrated nonunit ray rejected');
assert.equal(headLocalRayInStage({origin:[NaN,0,0],direction:[0,0,-1]},translatedHead),null);
console.log('PASS: calibrated head-local eye ray handles headset yaw, metric parallax, and invalid input.');
