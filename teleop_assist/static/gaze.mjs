import {headLocalRayInStage} from './geometry.mjs';

/** Build a linked camera-UV gaze proposal from a calibrated head-local ray.
 * intersectObserved must return camera UV for an actual metric surface hit,
 * or null. It must not substitute a plane at an assumed depth.
 */
export function eyeProjectionCommand(device, context, intersectObserved) {
 const {enabled,immersive,replay,cameraFresh,ageSeconds,headToStage,frameId,clientId,epoch,seq}=context;
 if(!enabled||!immersive||replay||!cameraFresh||!device?.valid||!device.calibrated)return null;
 if(!Number.isFinite(ageSeconds)||ageSeconds<0||ageSeconds>.25)return null;
 if(!Number.isFinite(device.confidence)||device.confidence<.6||device.confidence>1)return null;
 if(!Number.isSafeInteger(device.seq)||typeof device.session_id!=='string'||!device.session_id)return null;
 if(!Number.isSafeInteger(frameId)||!Number.isSafeInteger(seq)||typeof clientId!=='string'||!clientId)return null;
 const ray=headLocalRayInStage(device.head_ray,headToStage);if(!ray)return null;
 const uv=intersectObserved(ray.origin,ray.direction);
 if(!Array.isArray(uv)||uv.length!==2||!uv.every(v=>Number.isFinite(v)&&v>=0&&v<=1))return null;
 return {type:'gaze',sample:{source:'webxr_gaze_projection',session_id:clientId+':eye:'+epoch,
  client_id:clientId,seq,projection_of_seq:device.seq,projection_of_session:device.session_id,
  uv,valid:true,calibrated:true,frame_id:frameId,confidence:device.confidence}};
}
