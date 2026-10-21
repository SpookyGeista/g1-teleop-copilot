/** Versioned RGB-D display scheduling. Never upgrades a measurement timestamp. */
const replay=s=>s?.viewer_only===true||s?.mode==='recorded_replay';
const matrix=(m,n)=>Array.isArray(m)&&m.length===n&&m.every(r=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite));
function calibratedCamera(c){
 if(!c||!Number.isSafeInteger(c.width)||!Number.isSafeInteger(c.height)||c.width<=0||c.height<=0||c.depth_convention!=='optical_z_meters'||!matrix(c.intrinsics,3)||!matrix(c.T_world_camera,4))return false;
 const k=c.intrinsics,t=c.T_world_camera;
 if(k[0][0]<=0||k[1][1]<=0||k[0][1]!==0||k[1][0]!==0||k[2][0]!==0||k[2][1]!==0||k[2][2]!==1||t[3].some((v,i)=>v!==(i===3?1:0)))return false;
 for(let i=0;i<3;i++)for(let j=0;j<3;j++)if(Math.abs(t[0][i]*t[0][j]+t[1][i]*t[1][j]+t[2][i]*t[2][j]-(i===j?1:0))>1e-5)return false;
 const determinant=t[0][0]*(t[1][1]*t[2][2]-t[1][2]*t[2][1])-t[0][1]*(t[1][0]*t[2][2]-t[1][2]*t[2][0])+t[0][2]*(t[1][0]*t[2][1]-t[1][1]*t[2][0]);
 return Math.abs(determinant-1)<1e-5;
}
/** B * inverse(T_world_measurement) * T_world_live * B, B=optical->GL.
 * Geometric reprojection only; moving objects retain their measurement age.
 */
export function displaySurfaceTransform(measured,display){
 if(!calibratedCamera(measured)||!calibratedCamera(display))return null;
 const a=measured.T_world_camera,b=display.T_world_camera,sign=[1,-1,-1],out=Array.from({length:4},()=>[0,0,0,0]);
 for(let i=0;i<3;i++){
  for(let j=0;j<3;j++)out[i][j]=sign[i]*sign[j]*[0,1,2].reduce((s,k)=>s+a[k][i]*b[k][j],0);
  out[i][3]=sign[i]*[0,1,2].reduce((s,k)=>s+a[k][i]*(b[k][3]-a[k][3]),0);
 }out[3][3]=1;return out;
}
export function compatibleDisplayCamera(measured,display){
 if(!measured||!display||!Number.isSafeInteger(measured.width)||!Number.isSafeInteger(measured.height)||measured.width<=0||measured.height<=0)return false;
 if(measured.width!==display.width||measured.height!==display.height||measured.depth_convention!=='optical_z_meters'||display.depth_convention!==measured.depth_convention)return false;
 for(const [key,n] of [['intrinsics',3],['T_world_camera',4]]){
  if(!matrix(measured[key],n)||!matrix(display[key],n))return false;
  if(measured[key].some((row,i)=>row.some((v,j)=>v!==display[key][i][j])))return false;
 }
 return measured.intrinsics[0][0]>0&&measured.intrinsics[1][1]>0;
}
// Flat JPEGs retain exact camera compatibility. Only the calibrated RGB-D mesh
// renderer may opt into reprojection; this is never a refreshed object pose.
export function chooseDisplayFrame(state,{reproject3D=false}={}){
 const measured={frame_id:state.frame_id,timestamp:state.timestamp-state.frame_age,
  age_s:state.frame_age,camera_id:'head',camera:state.camera,source:'measurement_paired'};
 const live=state.display_frame;
 if(!replay(state)&&Number.isSafeInteger(live?.frame_id)&&live.frame_id>state.frame_id
   &&Number.isFinite(live.timestamp)&&Number.isFinite(measured.timestamp)&&live.timestamp>=measured.timestamp
   &&live.camera_id==='head'&&(compatibleDisplayCamera(state.camera,live.camera)
     ||(reproject3D&&displaySurfaceTransform(state.camera,live.camera)!==null)))return live;
 return measured;
}
export function measurementFresh(state,elapsed=0){
 if(replay(state))return true;
 const age=state?.frame_age,limit=state?.settings?.frame_timeout??.7;
 return Number.isFinite(age)&&Number.isFinite(elapsed)&&elapsed>=0&&age>=-.01&&age+elapsed<=limit;
}
export function visibleMeasuredTargets(state,elapsed=0){
 if(!measurementFresh(state,elapsed))return [];
 return (state.targets||[]).filter(t=>{
  if(t.stale)return false;
  if(replay(state)||t.box_camera?.source!=='registered_CAD_bounds')return true;
  const age=state.timestamp+elapsed-t.pose_timestamp;
  if(!Number.isFinite(age)||age<-.01||age>.5)return false;
  if(t.pose_source==='foundationpose_rgbd_tracking'){
   const registrationAge=state.timestamp+elapsed-t.pose_registration_timestamp;
   if(!Number.isFinite(registrationAge)||registrationAge<0||registrationAge>.7)return false;
  }
  return t.pose_valid===true;
 });
}
