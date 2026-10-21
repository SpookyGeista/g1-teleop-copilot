/** Pure geometry and input primitives; optical camera: x right, y down, z forward. */
export function intrinsics(camera) {
 const k=camera?.intrinsics;
 if (Array.isArray(k)) { const a=Array.isArray(k[0])?k.flat():k; return {fx:a[0],fy:a[4],cx:a[2],cy:a[5]}; }
 return k || {fx:1,fy:1,cx:0,cy:0};
}
export function backproject(u,v,z,k) { return [(u-k.cx)*z/k.fx,-(v-k.cy)*z/k.fy,-z]; }
export function projectOptical(p,k,width,height) { if(!p||p[2]<=0) return null;return [(p[0]*k.fx/p[2]+k.cx)/width,(p[1]*k.fy/p[2]+k.cy)/height]; }
export function opticalToGL(p) {return [p[0],-p[1],-p[2]];}
export function decodeDepth(buffer,width,height) {
 if(buffer.byteLength!==width*height*4)throw new Error('RGB-D frame has wrong depth size');
 const view=new DataView(buffer), result=new Float32Array(width*height);
 for(let i=0;i<result.length;i++)result[i]=view.getFloat32(i*4,true);
 return result;
}
export function depthMesh(depth,width,height,k,stride=3,maxJump=.04) {
 const cols=Math.ceil(width/stride),rows=Math.ceil(height/stride),positions=new Float32Array(cols*rows*3),uv=new Float32Array(cols*rows*2),valid=new Uint8Array(cols*rows),zs=new Float32Array(cols*rows),indices=[];
 for(let y=0;y<rows;y++)for(let x=0;x<cols;x++){const i=y*cols+x,u=Math.min(x*stride,width-1),v=Math.min(y*stride,height-1),z=depth[v*width+u];if(Number.isFinite(z)&&z>.04&&z<20){positions.set(backproject(u,v,z,k),i*3);valid[i]=1;zs[i]=z;}uv.set([u/(width-1),1-v/(height-1)],i*2);}
 function tri(a,b,c){if(valid[a]&&valid[b]&&valid[c]&&Math.max(zs[a],zs[b],zs[c])-Math.min(zs[a],zs[b],zs[c])<maxJump)indices.push(a,b,c);}
 for(let y=0;y<rows-1;y++)for(let x=0;x<cols-1;x++){const a=y*cols+x,b=a+1,c=a+cols,d=c+1;tri(a,c,b);tri(b,c,d);}
 return {positions,uv,indices:new Uint32Array(indices),validCount:valid.reduce((a,b)=>a+b,0)};
}
export class XRButtonEdges {
 constructor(){this.previous={calibrate:false,assist:false,eye:false,stop:false};}
 update(gamepad,state={}){const value=i=>Number(gamepad?.buttons?.[i]?.value||0),current={calibrate:value(4)>.5,assist:value(5)>.5,eye:value(3)>.5,stop:value(0)>.8&&value(1)>.8},commands=[];
 if(current.stop&&!this.previous.stop)commands.push({type:'stop'});
 else if(current.calibrate&&!this.previous.calibrate)commands.push({type:state.stopped?'resume':'calibrate'});
 if(current.assist&&!this.previous.assist)commands.push({type:'assist_enable',enabled:!state.assist?.enabled});
 if(current.eye&&!this.previous.eye)commands.push({type:'eye_enable',enabled:!state.eye_enabled});
 this.previous=current;return commands;}
 reset(){this.previous={calibrate:false,assist:false,eye:false,stop:false};}
}
export function normalizedPointer(event,rect,imageWidth,imageHeight){const scale=Math.min(rect.width/imageWidth,rect.height/imageHeight),w=imageWidth*scale,h=imageHeight*scale,x=event.clientX-rect.left-(rect.width-w)/2,y=event.clientY-rect.top-(rect.height-h)/2;if(x<0||y<0||x>w||y>h)return null;return [x/w,y/h];}
/** Map raw XR-stage displacement into robot world, compensating scene anchor.
 * R_world_camera_first · diag(1,-1,-1) · R_xr_anchor^-1.
 * Matrix returned as explicit rows; quaternion is XYZW.
 */
export function xrToWorldRotation(cameraMatrix,anchorQuaternion){
 if(!Array.isArray(cameraMatrix)||cameraMatrix.length!==4||!Array.isArray(anchorQuaternion)||anchorQuaternion.length!==4)return null;
 const [x,y,z,w]=anchorQuaternion,n=Math.hypot(x,y,z,w);if(!Number.isFinite(n)||Math.abs(n-1)>.01)return null;
 const r=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];
 const signs=[1,-1,-1];return [0,1,2].map(i=>[0,1,2].map(j=>[0,1,2].reduce((sum,k)=>sum+cameraMatrix[i][k]*signs[k]*r[j][k],0)));
}
/** Calibrated eye ray in the OpenXR head frame -> current XR stage ray. */
export function headLocalRayInStage(ray,headToStage){
 if(!ray||!Array.isArray(ray.origin)||!Array.isArray(ray.direction)||ray.origin.length!==3||ray.direction.length!==3||![...ray.origin,...ray.direction].every(Number.isFinite))return null;
 const norm=Math.hypot(...ray.direction);if(Math.abs(norm-1)>.01)return null;
 if(!Array.isArray(headToStage)||headToStage.length!==4||headToStage.some(r=>!Array.isArray(r)||r.length!==4||!r.every(Number.isFinite)))return null;
 const origin=[0,1,2].map(i=>headToStage[i][3]+[0,1,2].reduce((s,j)=>s+headToStage[i][j]*ray.origin[j],0));
 const direction=[0,1,2].map(i=>[0,1,2].reduce((s,j)=>s+headToStage[i][j]*ray.direction[j],0));
 const transformedNorm=Math.hypot(...direction);if(Math.abs(transformedNorm-1)>.01)return null;
 return {origin,direction:direction.map(v=>v/transformedNorm)};
}
