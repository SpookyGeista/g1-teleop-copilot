import assert from 'node:assert/strict';
import {compatibleDisplayCamera,chooseDisplayFrame,measurementFresh,visibleMeasuredTargets,displaySurfaceTransform} from '../display.mjs';
const camera={width:640,height:480,intrinsics:[[500,0,320],[0,500,240],[0,0,1]],T_world_camera:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],depth_convention:'optical_z_meters'};
const state={frame_id:10,timestamp:100,frame_age:.2,camera,settings:{frame_timeout:.7},targets:[{id:'surface'},{id:'cad',pose_valid:true,pose_timestamp:99.8,box_camera:{source:'registered_CAD_bounds'}}],display_frame:{frame_id:13,timestamp:99.99,camera_id:'head',camera:structuredClone(camera)}};
assert(compatibleDisplayCamera(camera,structuredClone(camera)));
assert.equal(chooseDisplayFrame(state).frame_id,13);
assert.equal(state.frame_id,10);assert.equal(state.frame_age,.2);
for(const modify of [c=>c.width++,c=>c.intrinsics[0][0]+=1e-12,c=>c.T_world_camera[0][3]+=1e-12,c=>c.depth_convention='range',c=>delete c.T_world_camera,c=>c.intrinsics[0][0]=NaN]){
 const next=structuredClone(state);modify(next.display_frame.camera);
 assert.equal(chooseDisplayFrame(next).frame_id,10);
}
for(const modify of [s=>s.display_frame.camera_id='right_wrist',s=>s.display_frame.frame_id=9,s=>s.display_frame.timestamp=99.7,s=>s.viewer_only=true]){
 const next=structuredClone(state);modify(next);assert.equal(chooseDisplayFrame(next).frame_id,10);
}
assert.deepEqual(visibleMeasuredTargets(state,.29).map(t=>t.id),['surface','cad']);
assert.deepEqual(visibleMeasuredTargets(state,.31).map(t=>t.id),['surface']);
assert.deepEqual(visibleMeasuredTargets(state,.51),[]);
assert(!measurementFresh({...state,frame_age:.71,display_frame:{...state.display_frame,timestamp:100}}));
assert.equal(chooseDisplayFrame({...state,frame_age:.71}).frame_id,13,'Fresh raw video remains available while measurement is stale');
assert.deepEqual(visibleMeasuredTargets({...state,viewer_only:true,frame_age:999},1).map(t=>t.id),['surface','cad']);
const moving=structuredClone(state);moving.display_frame.camera.T_world_camera[0][3]=.001;
assert.equal(chooseDisplayFrame(moving).frame_id,10,'2D must remain measurement-paired when the camera moves');
assert.equal(chooseDisplayFrame(moving,{reproject3D:true}).frame_id,13);
assert.equal(displaySurfaceTransform(camera,moving.display_frame.camera)[0][3],.001);
const translated=structuredClone(camera);translated.T_world_camera[1][3]=.2;translated.T_world_camera[2][3]=.3;
assert.deepEqual(displaySurfaceTransform(camera,translated).map(r=>r[3]),[0,-.2,-.3,1]);
const rotated=structuredClone(camera);rotated.T_world_camera=[[0,-1,0,.1],[1,0,0,.2],[0,0,1,.3],[0,0,0,1]];
const transform=displaySurfaceTransform(camera,rotated),point=[.2,-.4,-1,1];
const transformed=transform.map(r=>r.reduce((sum,v,i)=>sum+v*point[i],0));
// Optical [.2,.4,1] -> world[-.3,.4,1.3] -> measured GL[-.3,-.4,-1.3].
assert(transformed.every((v,i)=>Math.abs(v-[-.3,-.4,-1.3,1][i])<1e-12));
for(const modify of [c=>c.T_world_camera[0][0]=2,c=>c.T_world_camera[0][0]=-1,c=>c.T_world_camera[3][0]=1,c=>c.intrinsics[0][1]=1]){
 const invalid=structuredClone(camera);modify(invalid);assert.equal(displaySurfaceTransform(camera,invalid),null);
}
console.log('Multi-rate display freshness, exact 2D compatibility and calibrated 3D reprojection guards passed');
