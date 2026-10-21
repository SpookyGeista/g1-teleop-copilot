// Retained Quest convention: ABXY calibrates; AX starts/pauses pose streaming.
// Single face-button actions happen on release so chords never select objects.
const pressed=(g,i)=>Number(g?.buttons?.[i]?.value||0)>.5;
export class ReferenceTeleopControls {
 constructor(){this.calibrated=false;this.active=false;this.phase='reference';this.mask=0;this.consumed=false;this.fired=false;this.since=0;this.gestureActive=false;}
 invalidate(){this.calibrated=false;this.active=false;this.phase='reference';}
 pause(){this.active=false;if(this.calibrated)this.phase='paused';}
 completeCalibration(ok){this.calibrated=ok;this.active=false;this.phase=ok?'ready':'reference';}
 update(left,right,{now,trackingValid,ownsControl,stopped,poseReady}){
  const actions=[];
  if(this.active&&(!trackingValid||!ownsControl||stopped)){this.pause();if(!ownsControl)this.invalidate();}
  const mask=(pressed(right,4)?1:0)|(pressed(right,5)?2:0)|(pressed(left,4)?4:0)|(pressed(left,5)?8:0);
  if(mask&&!this.mask){this.gestureActive=this.active;this.consumed=false;this.fired=false;}
  if(mask!==this.mask)this.since=now;
  if(mask&&(mask&(mask-1)))this.consumed=true;
  if(mask===15&&!this.fired){
   if(this.gestureActive){this.pause();this.phase='stopped';this.fired=true;actions.push('stop');}
   else if(now-this.since>=.6&&trackingValid&&poseReady){this.active=false;this.phase='calibrating';this.fired=true;actions.push('calibrate');}
  }else if(mask===5&&!this.fired){
   if(this.gestureActive){this.pause();this.fired=true;actions.push('pause');}
   else if(now-this.since>=.4&&this.calibrated&&trackingValid&&ownsControl&&!stopped){this.active=true;this.phase='active';this.fired=true;actions.push('start');}
  }
  // A partial AX gesture may precede ABXY while stopping. ABXY must still
  // issue the backend stop even if AX already paused this gesture.
  if(mask===15&&this.gestureActive&&this.phase!=='stopped'){
   this.pause();this.phase='stopped';this.fired=true;actions.push('stop');
  }
  if(!mask&&this.mask&&!this.consumed&&!this.fired){
   if(this.mask===1)actions.push('confirm');
   if(this.mask===2)actions.push('release');
   if(this.mask===8)actions.push('assist');
  }
  if(!mask){this.consumed=false;this.fired=false;}
  this.mask=mask;
  return actions;
 }
}

// Head and both controller positions can check only a broad hand posture,
// not elbow angle or whole-body alignment. Human dimensions are not fixed.
export function referencePosture(head,left,right){
 if(!head?.valid||!left?.valid||!right?.valid)return {ready:false,message:'Beide Controller vor dem Körper sichtbar halten.'};
 const q=head.quaternion_xyzw,p=head.position;
 if(!q||!p||![...q,...p,...left.position,...right.position].every(Number.isFinite))return {ready:false,message:'Tracking fehlt.'};
 const [x,y,z,w]=q;
 const rightAxis=[1-2*(y*y+z*z),0,2*(x*z-y*w)];
 const n=Math.hypot(...rightAxis);if(n<.5)return {ready:false,message:'Geradeaus schauen.'};
 const r=rightAxis.map(v=>v/n),forward=[r[2],0,-r[0]];
 for(const [hand,sign] of [[left,-1],[right,1]]){
  const d=hand.position.map((v,i)=>v-p[i]);
  const lateral=sign*d.reduce((s,v,i)=>s+v*r[i],0),front=d.reduce((s,v,i)=>s+v*forward[i],0),below=-d[1];
  if(lateral<.08||lateral>.65||front<.08||front>.65||below<.18||below>.85)
   return {ready:false,message:'Aufrecht · Unterarme nach vorn · Hände auf Bauchhöhe.'};
 }
 return {ready:true,message:'Referenzhaltung bereit · ABXY kurz halten.'};
}

// Capture the initial head frame, as the retained sender captures the inverse
// neck orientation. XR right/up/back becomes G1 forward/left/up. This basis
// stays fixed after calibration; the pitched robot camera is not a body axis.
export function referenceInputRotation(quaternion){
 if(!Array.isArray(quaternion)||quaternion.length!==4||!quaternion.every(Number.isFinite)||Math.abs(Math.hypot(...quaternion)-1)>.01)return null;
 const [x,y,z,w]=quaternion;
 const r=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
          [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
          [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];
 return [[-r[0][2],-r[1][2],-r[2][2]],[-r[0][0],-r[1][0],-r[2][0]],[r[0][1],r[1][1],r[2][1]]];
}
