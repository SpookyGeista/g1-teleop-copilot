import * as THREE from './vendor/three.module.min.js';
import {ImmersiveSessionController} from './xr_session.mjs';
import {ReferenceTeleopControls,referencePosture,referenceInputRotation} from './reference_teleop.mjs';
import {intrinsics,depthMesh,decodeDepth,normalizedPointer,XRButtonEdges,xrToWorldRotation} from './geometry.mjs';
import {eyeProjectionCommand} from './gaze.mjs';
import {chooseDisplayFrame,measurementFresh,visibleMeasuredTargets,compatibleDisplayCamera,displaySurfaceTransform} from './display.mjs';
const $=id=>document.getElementById(id), canvas=$('camera-canvas'),ctx=canvas.getContext('2d');
const clientId=globalThis.crypto?.randomUUID?.()||('browser-'+Date.now()+'-'+Math.random().toString(16).slice(2));
let deviceGaze=null,deviceGazeFetchedAt=0,deviceGazePollBusy=false,deviceGazeLastPoll=0,gazeProjectionBusy=false,gazeProjectionSeq=0,lastProjectedDeviceKey=null;
let eyeCalibration=null,xrEpoch=0,xrNeedsInvalidation=false,xrRequireGripRelease=false;
const eyeText={active:'Blickdaten aktiv',disabled:'Aus',no_device:'Kein Tracker verbunden',no_data:'Keine Blickdaten',stale:'Blickdaten veraltet',invalid:'Blick ungültig',calibration_required:'Kalibrierung nötig',low_confidence:'Blick unsicher',projection_required:'VR-Blickprojektion wartet',device_stale:'Tracker-Daten veraltet'};
const assistText={cad_pose_stale:'Griffpose veraltet',execution_travel_limit:'Geprüfter Hub erreicht · Rückzug möglich',target_confirmation_required:'Objekt bestätigen',robot_invalid:'Roboterzustand ungültig',action_confirmation_required:'Handlung bestätigen',calibrated_action_unavailable:'Mechanik nicht eingerichtet',action_calibration_lost:'Aktionskalibrierung fehlt',action_geometry_jump:'Zielpose hat sich sprunghaft geändert',fresh_clutch_required:'Griff lösen und erneut halten',perception_uncertain:'Zielmessung unsicher',operator_stationary:'Operator hält an',predictive_alignment:'vorausschauende Ausrichtung',predictive_aligned:'ausgerichtet',action_required:'Handlung wählen und bestätigen',action_rearm_required:'Freigabe lösen und erneut halten',action_unconfirmed:'Handlung bestätigen',uncertain_target:'Zielmessung unsicher',prediction_alignment:'vorausschauende Ausrichtung',no_input:'warte auf Eingabe',calibration_required:'Hand kalibrieren',deadman_released:'Freigabe halten',tracking_invalid:'Tracking fehlt',input_stale:'Eingabe veraltet',manual_control:'manuelle Bewegung',outside_assist_region:'näher an das Ziel führen',lateral_alignment:'seitliche Ausrichtung',aligned:'ausgerichtet',perception_stale:'Kamera veraltet',target_lost:'Ziel verloren',joint_limit:'Gelenkgrenze',collision:'Kontaktstopp',contact_retreat:'Kontakt · geführter Rückzug',stopped:'angehalten'};
const palette=['#6ce2ce','#89b9ff','#d2a6ff','#f0ab80','#f4d37a','#99d09c'];
let state=null,image=null,depth=null,displayedFrame=null,lastFrameAt=0,lastStateAt=0,seq=0,virtualHand=[.20,-.20,-.40],pointerUV=[.5,.5],held=false,view3D=false,frameRequestBusy=false,inputBusy=false,keys=new Set(),lastTick=performance.now(),session=null,referenceSpace=null,xrAnchorPending=false,lastInputAt=0,pointerDown=false,lastFrameError='';
let displayDescriptor=null,measuredDepth=null,measuredDepthFrame=null,stateSampleAt=0,lastOverlayKey='',lastAgePaint=0;
const depthCache=new Map();
const referenceControls=new ReferenceTeleopControls();
let referenceBusy=false,referenceHint='',lastReferenceSample=null,referenceEyeDown=false,referenceBasis=null;
let ownershipAck=null,ownershipAckAt=-Infinity;
const referenceMode=()=>state?.robot?.start_pose==='reference';
const pose=(position,quaternion=[0,0,0,1],valid=true)=>({position,quaternion_xyzw:quaternion,valid});
const invalidPose=()=>pose([0,0,0],[0,0,0,1],false);
function message(text,error=false){$('message').textContent=text;$('message').classList.toggle('error',error);}
function replayMode(){return !!state&&(state.viewer_only===true||state.mode==='recorded_replay');}
function ownsControl(){return !!state&&(!state.control_session||state.control_session.owner_client_id===clientId);}
function updateControlOwner(){const owned=ownsControl(),contract=state?.control_session;$('control-owner-status').textContent=replayMode()?'Aufzeichnung · keine Steuerung':owned?'Steuerung in diesem Browser':contract?.owner_client_id?'Beobachter · anderer Browser steuert':'Beobachter · Steuerung frei';$('control-claim').hidden=!contract||owned||replayMode();$('control-release').hidden=!contract||!owned||replayMode();if(!owned){held=false;keys.clear();xrRequireGripRelease=true;if(referenceMode()&&!referenceBusy)referenceControls.invalidate();}}
async function command(data){if(!state||replayMode())return null;if(!ownsControl()&&!['control_claim','control_release','stop'].includes(data.type))return null;data={...data,client_id:clientId};try{const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok)throw new Error((await r.text()).slice(0,180));const result=await r.json().catch(()=>({}));if(result.error)throw new Error(result.error);if(result.control_session){ownershipAck=result.control_session;ownershipAckAt=performance.now();state.control_session=result.control_session;updateControlOwner();renderState();}return result;}catch(e){message('Befehl nicht übernommen: '+e.message,true);return null;}}
function stateElapsed(){return Math.max(0,(performance.now()-stateSampleAt)/1000);}
function imageAge(){return state&&displayDescriptor&&Number.isFinite(displayDescriptor.timestamp)?Math.max(0,state.timestamp-displayDescriptor.timestamp)+stateElapsed():Infinity;}
function fresh(){return displayedFrame!==null&&imageAge()<1.5&&performance.now()-lastFrameAt<1500&&performance.now()-lastStateAt<1500&&!document.hidden;}
function measuredFresh(){return !!state&&measuredDepthFrame===state.frame_id&&measurementFresh(state,stateElapsed())&&fresh();}
function visibleTargets(){return state&&measuredDepthFrame===state.frame_id?visibleMeasuredTargets(state,stateElapsed()):[];}
function ageLabel(value){return Number.isFinite(value)?Math.round(1000*Math.max(0,value))+' ms':'—';}
function paintAges(){if(!state)return;const measured=Number.isFinite(state.frame_age)?state.frame_age+stateElapsed():Infinity;$('frame-age-status').textContent=`Bildalter ${ageLabel(imageAge())} · Messalter ${ageLabel(measured)} · Boxen aus Messung ${state.frame_id}${measuredFresh()?'':' · Messung veraltet'}`;}
function displayLabel(t){if(!t)return 'Noch kein Ziel';const names={handle:'Griff',knob:'Drehknopf',button:'Taster',screw:'Schraube',lever:'Hebel',switch:'Schalter',connector:'Anschluss',cable:'Kabel',drawer:'Schublade',round_component:'Rundes Bauteil',elongated_component:'Längliches Bauteil',raised_component:'Erhabenes Bauteil'},name=names[t.label]||t.label;const suffix=/component_(\d+)/.exec(t.id||'');return name+(suffix?' '+String(Number(suffix[1])).padStart(2,'0'):'');}
function currentTarget(){return state?.targets?.find(t=>t.id===state.selection?.id);}
function colorFor(target,index){if(target.id===state?.selection?.id)return state.selection.mode==='confirmed'?'#f7c16b':'#66dfcc';return palette[index%palette.length];}
function actionLabel(){const a=state?.action;return a?.available?.find(x=>x.id===a.id)?.label||a?.id||'Handlung wählen';}
function renderActions(){const a=state?.action,panel=$('action-panel');panel.hidden=!a?.required;if(!a?.required)return;
 const locked=state.selection?.mode==='confirmed',list=$('action-list');
 const signature=JSON.stringify([a.available,a.id,a.confirmed,locked,replayMode()]);
 if(list.dataset.signature!==signature){list.dataset.signature=signature;list.replaceChildren();for(const item of a.available||[]){const b=document.createElement('button');b.className='target-button';b.textContent=item.label||item.id;b.dataset.actionId=item.id;b.classList.toggle('selected',a.id===item.id);b.classList.toggle('confirmed',a.confirmed&&a.id===item.id);b.disabled=!locked||replayMode();b.onclick=()=>{held=false;keys.delete(' ');command({type:'action_select',action_id:item.id});};list.append(b);}}
 $('action-status').textContent=!locked?'Zuerst ein Ziel bestätigen.':!a.available?.length?'Für dieses Bauteil ist noch keine Mechanik eingerichtet.':a.confirmed?`${actionLabel()} · ${a.phase==='execute'?'Ausführung':'Annäherung'}${a.rearm_required?' · Freigabe lösen und erneut halten':''}`:'Mit Stick oder Auswahl eine Handlung festlegen.';
 const current=currentTarget();if(locked&&current)$('action-status').textContent+=current.pose_valid?' · Griffpose erkannt':' · sichtbare Oberfläche, Griffpose noch offen';
 $('action-confirm').disabled=!locked||!a.id||a.confirmed||replayMode();$('action-execute').disabled=!a.confirmed||a.phase==='execute'||a.available?.find(x=>x.id===a.id)?.execution_ready===false||replayMode();$('action-approach').disabled=!a.confirmed||a.phase==='approach'||replayMode();
}
async function actionPhase(phase){held=false;keys.delete(' ');xrRequireGripRelease=true;
 if(referenceMode()){referenceControls.pause();if(lastReferenceSample)await sendInput({...lastReferenceSample,clutch:false,squeeze:0},true);}
 return command({type:'action_phase',phase,engagement_confirmed:phase==='execute'});
}
function renderState(){if(!state)return;updateControlOwner();$('reference-help').hidden=!referenceMode();if(referenceMode()){$('controls-help').textContent='VR: gleiche Referenzhaltung wie G1 einnehmen. ABXY übernimmt und kalibriert; AX startet/pausiert ohne Griffknopf. Trigger schließt die Hand. ABXY während der Übertragung hält an. A bestätigt, B löst, Y schaltet Assistenz; danach AX zum Weiterbewegen. Desktop: Steuerung übernehmen, C kalibriert, Leertaste halten und WASD/QE bewegen.';}const replay=replayMode();for(const control of document.querySelectorAll('aside button, aside input, #stop, #resume'))control.disabled=replay;if(replay){message('Aufgezeichnete Sitzung · keine Steuerung');$('input-source').textContent='Aufzeichnung';}if(replay&&eyeCalibration)finishEyeCalibration();const t=currentTarget(),mode=state.selection?.mode||'preview';$('empty-state').hidden=!!image;$('eye-toggle').checked=!!state.eye_enabled;$('assist-toggle').checked=!!state.assist?.enabled;
 $('eye-status').textContent=state.eye_enabled?(state.eye_status==='active'?'Blickdaten aktiv':(eyeText[state.eye_status]||'Warte auf Blickdaten')+' · Kopf/Zeiger bleibt nutzbar'):'Aus · Kopf/Zeiger und Stick';
 $('frame-info').textContent=`Bild ${displayedFrame} · Messung ${state.frame_id} · ${state.camera?.width||'—'} × ${state.camera?.height||'—'}`;paintAges();
 $('candidate-name').textContent=t?displayLabel(t):'Noch kein Ziel';$('target-id').textContent=t?.id||'—';$('selection-mode').textContent=mode==='confirmed'?'BESTÄTIGT':mode==='manual'?'MANUELL GEHALTEN':mode==='lost'?'ZIEL VERLOREN':'VORSCHLAG';
 $('resume').hidden=!state.stopped;$('stop').hidden=!!state.stopped;
 $('assist-status').textContent=state.assist?.active?'Korrigiert':state.assist?.enabled?'Bereit · '+(assistText[state.assist.reason]||'Freigabe halten'):'Aus';
 $('source-badge').textContent=replay?'AUFZEICHNUNG · RGB-D':'ISAAC · RGB-D';
 const ee=state.robot?.ee_position;$('ee-position').textContent=Array.isArray(ee)?ee.map(v=>Number(v).toFixed(3)).join(' / ')+' m':'—';
 const list=$('target-list'),signature=JSON.stringify((state.targets||[]).map(t=>[t.id,t.label,t.confidence,state.selection?.id,state.selection?.mode]));
 if(list.dataset.signature!==signature){list.dataset.signature=signature;list.replaceChildren();(state.targets||[]).forEach((t,i)=>{const b=document.createElement('button');b.className='target-button';b.disabled=replay;b.classList.toggle('selected',t.id===state.selection?.id);b.classList.toggle('confirmed',t.id===state.selection?.id&&mode==='confirmed');b.dataset.targetId=t.id;const dot=document.createElement('span');dot.className='dot';dot.style.background=colorFor(t,i);const name=document.createElement('span');name.textContent=displayLabel(t);const conf=document.createElement('small');conf.textContent=Number.isFinite(t.confidence)?Math.round(t.confidence*100)+'%':'';b.append(dot,name,conf);b.onclick=()=>command({type:'select',id:t.id});list.append(b);});if(!state.targets?.length){const p=document.createElement('p');p.className='hint';p.textContent='Keine sichtbaren Bauteile erkannt.';list.append(p);}}
 renderActions();if(!ownsControl()){for(const c of document.querySelectorAll('aside button, aside input, #resume'))c.disabled=true;}
 const pd=state.perception||{},sensors=pd.sensors||{};
 $('sensor-status').textContent=[pd.method||pd.backend||'Wahrnehmung wartet',pd.cad_pose?`CAD-Pose: ${pd.cad_pose.status}`:'',pd.inference_ms?`Inferenz ${Math.round(pd.inference_ms)} ms`:'',...Object.entries(sensors).map(([key,value])=>`${key}: ${typeof value==='object'?(value.status||value.source||'aktiv'):value}`)].filter(Boolean).join(' · ');
 if(!measuredFresh()&&!replay)for(const c of document.querySelectorAll('#target-list button, #confirm, .navigation-pad button'))c.disabled=true;
 draw();updateBoxes();window.__assistDiagnostics={...window.__assistDiagnostics,frameId:displayedFrame,measurementFrameId:state.frame_id,pickFrameId:measuredDepthFrame,displaySource:displayDescriptor?.source,displayReprojected:displayDescriptor?.source==='raw_camera_capture'&&!compatibleDisplayCamera(state.camera,displayDescriptor.camera),surfaceToMeasurement:surface.matrix.toArray(),displayDepthCenter:depth?.[Math.floor(depth.length/2)],measuredDepthCenter:measuredDepth?.[Math.floor(measuredDepth.length/2)]};
}
function draw(){if(!image||!state?.camera)return;const w=state.camera.width,h=state.camera.height;if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}ctx.clearRect(0,0,w,h);ctx.drawImage(image,0,0,w,h);ctx.font=`${Math.max(11,w/70)}px system-ui`;
 visibleTargets().forEach((t,i)=>{if(!Array.isArray(t.bbox_px))return;const [x0,y0,x1,y1]=t.bbox_px,c=colorFor(t,i);ctx.fillStyle=c+'18';ctx.fillRect(x0,y0,x1-x0,y1-y0);ctx.lineWidth=t.id===state.selection?.id?2.5:1;ctx.strokeStyle=c;ctx.strokeRect(x0,y0,x1-x0,y1-y0);const caption=t.id===state.selection?.id?displayLabel(t):(/component_(\d+)/.exec(t.id||'')?.[1]||t.label);const tw=ctx.measureText(caption).width;ctx.fillStyle='#09131ce6';ctx.fillRect(x0,y0-19,tw+10,17);ctx.fillStyle=c;ctx.fillText(caption,x0+5,y0-6);});
 if(eyeCalibration){const uv=eyeCalibration.points[eyeCalibration.index],x=uv[0]*w,y=uv[1]*h;ctx.strokeStyle='#ffffff';ctx.fillStyle='#101b27';ctx.lineWidth=3;ctx.beginPath();ctx.arc(x,y,12,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#66dfcc';ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();}
 if(pointerUV&&!eyeCalibration){const x=pointerUV[0]*w,y=pointerUV[1]*h;ctx.strokeStyle='#ffffffcc';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x-7,y);ctx.lineTo(x+7,y);ctx.moveTo(x,y-7);ctx.lineTo(x,y+7);ctx.stroke();}
}
async function poll(){if(frameRequestBusy)return;frameRequestBusy=true;let imagePromise=null,committed=false;try{
 const sampledAt=performance.now(),r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw new Error(`Status ${r.status}`);const next=await r.json(),chosen=chooseDisplayFrame(next,{reproject3D:!!renderer&&(view3D||!!session)});
 if(!next.camera?.width||next.frame_id<0)return;
 const id=chosen.frame_id,pendingDepth=new Map();
 const loadDepth=(frameId,c)=>{const cached=depthCache.get(frameId);if(cached&&cached.width===c.width&&cached.height===c.height)return Promise.resolve(cached.depth);if(!pendingDepth.has(frameId))pendingDepth.set(frameId,(async()=>{const response=await fetch(`/api/depth.bin?camera_id=head&frame_id=${encodeURIComponent(frameId)}`,{cache:'no-store'});if(!response.ok)throw new Error('Kamera-Tiefe nicht mehr verfügbar');const value=decodeDepth(await response.arrayBuffer(),c.width,c.height);depthCache.set(frameId,{width:c.width,height:c.height,depth:value});while(depthCache.size>8)depthCache.delete(depthCache.keys().next().value);return value;})());return pendingDepth.get(frameId);};
 const changed=id!==displayedFrame;
 imagePromise=changed?(async()=>{const response=await fetch(`/api/frame.jpg?camera_id=head&frame_id=${encodeURIComponent(id)}`,{cache:'no-store'});if(!response.ok)throw new Error('Kamera-Frame nicht mehr verfügbar');return createImageBitmap(await response.blob());})():Promise.resolve(null);
 const [newImage,newDepth,newMeasuredDepth]=await Promise.all([imagePromise,changed?loadDepth(id,chosen.camera):depth,loadDepth(next.frame_id,next.camera)]);
 // A mode switch while requests were in flight must not expose a moved-camera
 // JPEG beneath measurement-frame 2D boxes, even for a single render tick.
 if(chooseDisplayFrame(next,{reproject3D:!!renderer&&(view3D||!!session)}).frame_id!==id){newImage?.close();return;}
 const measurementChanged=measuredDepthFrame!==next.frame_id;
 if(changed){if(image?.close)image.close();image=newImage;depth=newDepth;displayedFrame=id;lastFrameAt=performance.now();}
 // A camera poll begun before a claim response must not undo that newer ownership.
 if(ownershipAck&&sampledAt<ownershipAckAt)next.control_session=ownershipAck;
 measuredDepth=newMeasuredDepth;measuredDepthFrame=next.frame_id;displayDescriptor=chosen;state=next;stateSampleAt=sampledAt;lastStateAt=sampledAt;committed=true;
 if(changed||measurementChanged)updateSurface();if(measurementChanged)updateMeasuredSurface();canvas.style.visibility=view3D||session?'hidden':'visible';lastFrameError='';renderState();
 }catch(e){if(!committed&&imagePromise)imagePromise.then(value=>value?.close()).catch(()=>{});lastFrameError=e.message;message('Verbindung: '+e.message,true);}finally{frameRequestBusy=false;setTimeout(poll,33);}}

// One metric camera frame is reconstructed into XR. Both pixels and cuboids use
// the SAME optical->GL basis, then a single initial-head anchor (never per-eye).
const scene=new THREE.Scene();scene.background=new THREE.Color('#08111b');
const camera=new THREE.PerspectiveCamera(58,1,.02,30);camera.position.set(0,0,0);
const content=new THREE.Group();scene.add(content);
const sensorFrame=new THREE.Group();content.add(sensorFrame);let firstCameraWorld=null,firstCameraRows=null;
const textureCanvas=document.createElement('canvas'),textureContext=textureCanvas.getContext('2d');
const surfaceGeometry=new THREE.BufferGeometry(),surfaceTexture=new THREE.Texture();surfaceTexture.colorSpace=THREE.SRGBColorSpace;surfaceTexture.minFilter=THREE.LinearFilter;surfaceTexture.magFilter=THREE.LinearFilter;
const surface=new THREE.Mesh(surfaceGeometry,new THREE.MeshBasicMaterial({map:surfaceTexture,side:THREE.DoubleSide}));sensorFrame.add(surface);surface.frustumCulled=false;
// Picking uses the measured depth, never newer unmeasured display geometry.
const measuredSurface=new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshBasicMaterial({side:THREE.DoubleSide}));measuredSurface.visible=false;sensorFrame.add(measuredSurface);
const boxes=new THREE.Group();sensorFrame.add(boxes);const cubeGeometry=new THREE.BoxGeometry(1,1,1),cubeEdges=new THREE.EdgesGeometry(cubeGeometry),boxNodes=new Map();
const opticalBasis=new THREE.Matrix4().makeScale(1,-1,-1),basisQuaternion=new THREE.Quaternion().setFromRotationMatrix(opticalBasis),tmpM=new THREE.Matrix4(),raycaster=new THREE.Raycaster();
let renderer=null;try{renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.xr.enabled=true;renderer.xr.setReferenceSpaceType('local');$('three-container').append(renderer.domElement);}catch(e){$('view-3d').disabled=true;message('3D-Darstellung nicht verfügbar: '+e.message,true);}
const rayLine=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3(0,0,-2)]),new THREE.LineBasicMaterial({color:0x66dfcc,transparent:true,opacity:.8}));rayLine.visible=false;scene.add(rayLine);
const hudCanvas=document.createElement('canvas');hudCanvas.width=1024;hudCanvas.height=128;const hudCtx=hudCanvas.getContext('2d'),hudTexture=new THREE.CanvasTexture(hudCanvas),hud=new THREE.Mesh(new THREE.PlaneGeometry(.70,.0875),new THREE.MeshBasicMaterial({map:hudTexture,transparent:true,depthTest:false}));hud.renderOrder=1000;hud.visible=false;scene.add(hud);
function updateHUD(viewerPose){if(!session||!viewerPose)return;if(referenceMode()&&!referenceControls.active){updateReferenceHUD(viewerPose);return;}const target=currentTarget();hudCtx.clearRect(0,0,1024,128);hudCtx.fillStyle='#0b152de8';hudCtx.fillRect(0,0,1024,128);hudCtx.fillStyle='#66dfcc';hudCtx.font='bold 29px system-ui';hudCtx.fillText(target?displayLabel(target):'Ziel mit Kopf oder Controller anvisieren',24,44);hudCtx.font='17px system-ui';hudCtx.textAlign='right';hudCtx.fillText(`Bild ${ageLabel(imageAge())} · Messung ${ageLabel((state?.frame_age??Infinity)+stateElapsed())}`,1000,44);hudCtx.textAlign='left';hudCtx.font='22px system-ui';hudCtx.fillStyle='#e4edf7';hudCtx.fillText(replayMode()?'Aufgezeichnete Sitzung · keine Steuerung':!ownsControl()?'Beobachter · zur Übernahme VR beenden und Steuerung übernehmen':(state?.action?.required&&state?.selection?.mode==='confirmed'?`${actionLabel()} · ${state.action.confirmed?(state.action.phase==='execute'?'AUSFÜHRUNG':'ANNÄHERN'):'Stick wählt · A bestätigt'} · B löst`:(referenceMode()?'TELEOP AKTIV · AX Pause · ABXY Halt · Trigger schließt Hand':`${state?.selection?.mode||'preview'} · A bestätigt · B löst · Griff hält Freigabe · X kalibriert`)),24,82);hudCtx.font='17px system-ui';hudCtx.fillStyle='#91a4b8';hudCtx.fillText(`${state?.eye_enabled?'Augen EIN':'Augen AUS'} · ${state?.assist?.enabled?'Assistenz EIN':'Assistenz AUS'} · Links: Y Assistenz · Stickklick Augen · Griff+Trigger Halt`,24,111);hudTexture.needsUpdate=true;tmpM.fromArray(viewerPose.transform.matrix);hud.position.set(0,-.27,-.75).applyMatrix4(tmpM);hud.quaternion.setFromRotationMatrix(tmpM);hud.visible=true;}
function updateSurface(){if(!renderer||!state?.camera||!depth)return;const c=displayDescriptor?.camera||state.camera,m=state.camera;
 if(Array.isArray(m.T_world_camera)&&m.T_world_camera.length===4){const current=new THREE.Matrix4().set(...m.T_world_camera.flat());if(!firstCameraWorld){firstCameraWorld=current.clone();firstCameraRows=m.T_world_camera.map(row=>[...row]);}sensorFrame.matrixAutoUpdate=false;sensorFrame.matrix.copy(opticalBasis).multiply(firstCameraWorld.clone().invert()).multiply(current).multiply(opticalBasis);sensorFrame.matrixWorldNeedsUpdate=true;}
 const relative=displaySurfaceTransform(m,c);surface.matrixAutoUpdate=false;surface.matrix.identity();if(relative)surface.matrix.set(...relative.flat());surface.matrixWorldNeedsUpdate=true;
 const k=intrinsics(c),mesh=depthMesh(depth,c.width,c.height,k,Math.max(1,Math.ceil(c.width/240)),.055);surface.geometry.dispose();surface.geometry=new THREE.BufferGeometry();surface.geometry.setAttribute('position',new THREE.BufferAttribute(mesh.positions,3));surface.geometry.setAttribute('uv',new THREE.BufferAttribute(mesh.uv,2));surface.geometry.setIndex(new THREE.BufferAttribute(mesh.indices,1));textureCanvas.width=c.width;textureCanvas.height=c.height;textureContext.drawImage(image,0,0,c.width,c.height);surfaceTexture.image=textureCanvas;surfaceTexture.needsUpdate=true;surface.visible=mesh.indices.length>0;$('depth-status').textContent=`Tiefe: ${mesh.validCount.toLocaleString('de-DE')} Messpunkte · 3D-Fläche`;
 const viewK=intrinsics(m);camera.fov=THREE.MathUtils.radToDeg(2*Math.atan(m.height/(2*viewK.fy)));camera.aspect=m.width/m.height;camera.updateProjectionMatrix();window.__assistDiagnostics={...window.__assistDiagnostics,frameId:displayedFrame,depthVertices:mesh.validCount,depthTriangles:mesh.indices.length/3,geometry:'metric_rgbd_mesh',boxCount:boxNodes.size};}
function updateMeasuredSurface(){if(!state?.camera||!measuredDepth)return;const c=state.camera,mesh=depthMesh(measuredDepth,c.width,c.height,intrinsics(c),Math.max(1,Math.ceil(c.width/240)),.055);measuredSurface.geometry.dispose();measuredSurface.geometry=new THREE.BufferGeometry();measuredSurface.geometry.setAttribute('position',new THREE.BufferAttribute(mesh.positions,3));measuredSurface.geometry.setIndex(new THREE.BufferAttribute(mesh.indices,1));}
function updateBoxes(){const present=new Set();visibleTargets().forEach((t,i)=>{const b=t.box_camera;if(!b||!b.center||!b.size)return;present.add(t.id);let g=boxNodes.get(t.id);if(!g){g=new THREE.Group();const fill=new THREE.Mesh(cubeGeometry,new THREE.MeshBasicMaterial({transparent:true,opacity:.09,depthWrite:false}));fill.userData.targetId=t.id;const wire=new THREE.LineSegments(cubeEdges,new THREE.LineBasicMaterial({transparent:true,opacity:.94}));g.add(fill,wire);boxes.add(g);boxNodes.set(t.id,g);}g.position.set(b.center[0],-b.center[1],-b.center[2]);g.scale.set(...b.size.map(x=>Math.max(.002,Number(x))));g.quaternion.copy(basisQuaternion).multiply(new THREE.Quaternion(...(b.quaternion_xyzw||[0,0,0,1])));const color=colorFor(t,i);g.children.forEach(o=>o.material.color.set(color));g.children[0].material.opacity=t.id===state.selection?.id ? .18 : .065;});for(const [id,g]of boxNodes)if(!present.has(id)){g.children.forEach(o=>o.material.dispose());boxes.remove(g);boxNodes.delete(id);}if(window.__assistDiagnostics)window.__assistDiagnostics.boxCount=boxNodes.size;}
function resize(){if(!renderer||session)return;const r=$('viewport').getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();}
new ResizeObserver(resize).observe($('viewport'));
function setView(is3D){view3D=is3D;$('three-container').hidden=!is3D;canvas.style.visibility=is3D||(state&&displayDescriptor&&!compatibleDisplayCamera(state.camera,displayDescriptor.camera))?'hidden':'visible';$('view-2d').classList.toggle('selected',!is3D);$('view-3d').classList.toggle('selected',is3D);$('view-badge').textContent=is3D?'METRISCHE 3D-QUADER':'RGB-D';resize();}
$('view-2d').onclick=()=>setView(false);$('view-3d').onclick=()=>setView(true);
// Shift-drag previews parallax on desktop. Reset when switching to immersive XR.
let orbitStart=null;$('three-container').addEventListener('pointerdown',e=>{orbitStart=[e.clientX,e.clientY,camera.position.x,camera.position.y];});window.addEventListener('pointerup',()=>orbitStart=null);$('three-container').addEventListener('pointermove',e=>{if(!orbitStart||session)return;camera.position.set(orbitStart[2]-(e.clientX-orbitStart[0])*.0007,orbitStart[3]+(e.clientY-orbitStart[1])*.0007,0);});
function projectedRayUV(origin,direction,requireObservedSurface=false){if(!state?.camera||!measuredFresh())return null;raycaster.set(origin,direction);content.updateMatrixWorld(true);let hit=raycaster.intersectObjects([...boxNodes.values()].map(g=>g.children[0]),false)[0];if(requireObservedSurface){const observedHit=raycaster.intersectObject(measuredSurface,false)[0];if(observedHit&&(!hit||observedHit.distance<hit.distance-.003))hit=observedHit;}let p;if(hit)p=sensorFrame.worldToLocal(hit.point.clone());else {if(requireObservedSurface)return null;const inv=sensorFrame.matrixWorld.clone().invert(),o=origin.clone().applyMatrix4(inv),d=direction.clone().transformDirection(inv),z=-(currentTarget()?.center_camera?.[2]||1);if(d.z>=-.01)return null;const t=(z-o.z)/d.z;if(t<=0)return null;p=o.addScaledVector(d,t);}const k=intrinsics(state.camera),z=-p.z;if(z<=0)return null;const uv=[(p.x*k.fx/z+k.cx)/state.camera.width,(-p.y*k.fy/z+k.cy)/state.camera.height];return uv.every(v=>v>=0&&v<=1)?uv:null;}
function rayPointer(rayPose){if(!rayPose)return null;const mat=new THREE.Matrix4().fromArray(rayPose.transform.matrix);return projectedRayUV(new THREE.Vector3().setFromMatrixPosition(mat),new THREE.Vector3(0,0,-1).transformDirection(mat));}
async function pollEyeDevice(){const now=performance.now();if(!session||!state?.eye_enabled||replayMode()||deviceGazePollBusy||now-deviceGazeLastPoll<50)return;deviceGazePollBusy=true;deviceGazeLastPoll=now;try{const r=await fetch('/api/gaze',{cache:'no-store'});if(!r.ok)return;const data=await r.json();deviceGaze=data.device_sample?{...data.device_sample,received_age:data.device_sample.received_age??data.device_age}:null;deviceGazeFetchedAt=performance.now();}catch{deviceGaze=null;}finally{deviceGazePollBusy=false;}}
async function projectEyeDevice(head){if(!ownsControl())return;pollEyeDevice();const d=deviceGaze;if(!d||gazeProjectionBusy)return;const key=d.session_id+':'+d.seq;if(lastProjectedDeviceKey===key)return;const e=head.transform.matrix,headToStage=[0,1,2,3].map(i=>[e[i],e[i+4],e[i+8],e[i+12]]);const packet=eyeProjectionCommand(d,{enabled:state?.eye_enabled,immersive:!!session,replay:replayMode(),cameraFresh:measuredFresh(),ageSeconds:d.received_age+(performance.now()-deviceGazeFetchedAt)/1000,headToStage,frameId:state.frame_id,clientId,epoch:xrEpoch,seq:gazeProjectionSeq+1},(origin,direction)=>projectedRayUV(new THREE.Vector3(...origin),new THREE.Vector3(...direction),true));if(!packet)return;gazeProjectionBusy=true;gazeProjectionSeq++;lastProjectedDeviceKey=key;try{const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(packet)});if(!r.ok)lastProjectedDeviceKey=null;}catch{lastProjectedDeviceKey=null;}finally{gazeProjectionBusy=false;}}

function xrPose(p){if(!p)return invalidPose();const t=p.transform;return pose([t.position.x,t.position.y,t.position.z],[t.orientation.x,t.orientation.y,t.orientation.z,t.orientation.w]);}
function button(g,index){return Number(g?.buttons?.[index]?.value||0);}
const xrButtons=new XRButtonEdges();let phasePressStart=null,phasePressUsed=false;
function updateReferenceHUD(viewerPose){
 hudCtx.clearRect(0,0,1024,128);hudCtx.fillStyle='#0b152df0';hudCtx.fillRect(0,0,1024,128);
 const phase=referenceControls.phase;
 hudCtx.fillStyle='#66dfcc';hudCtx.font='bold 28px system-ui';
 hudCtx.fillText(referenceBusy?'Referenz wird kalibriert …':state?.stopped?'HALT · Haltung an G1 angleichen · ABXY zum Neustart':referenceControls.calibrated?'Bereit · AX startet / pausiert die Übertragung':'Gemeinsame Referenzpose · ABXY kalibriert',20,34);
 hudCtx.fillStyle='#e4edf7';hudCtx.font='23px system-ui';
 hudCtx.fillText(referenceControls.calibrated?'Nach AX folgt der rechte Arm deiner Bewegung.': 'Aufrecht · Oberarme am Körper · Unterarme nach vorn',20,71);
 hudCtx.font='20px system-ui';hudCtx.fillStyle='#a9bdd0';hudCtx.fillText(referenceHint||'Handflächen nach innen · ABXY etwa 1 Sekunde halten',20,108);
 hudTexture.needsUpdate=true;tmpM.fromArray(viewerPose.transform.matrix);hud.position.set(0,-.27,-.75).applyMatrix4(tmpM);hud.quaternion.setFromRotationMatrix(tmpM);hud.visible=true;
}
function inputPacket(sample){return {type:'input',client_id:clientId,seq:++seq,frame_id:state.frame_id,session_id:clientId+':'+sample.source+(sample.source==='webxr_controller'?':'+xrEpoch:''),...sample};}
async function calibrateReference(sample){
 if(referenceBusy)return;referenceBusy=true;referenceControls.pause();
 try{
  if(!ownsControl()){const claimed=await command({type:'control_claim',deadman_released:true});if(!claimed?.accepted)throw new Error('Steuerung konnte nicht übernommen werden.');}
  referenceBasis=referenceInputRotation(sample.head.quaternion_xyzw);
  if(!referenceBasis)throw new Error('Gültige Kopfreferenz erforderlich.');
  const released={...sample,xr_to_world_rotation:referenceBasis,clutch:false,squeeze:0,trigger:0,primary:false,secondary:false};
  if(!await sendInput(released,true))throw new Error('Aktuelle Controllerdaten fehlen.');
  if(state.stopped){const resumed=await command({type:'resume'});if(!resumed?.accepted)throw new Error('Halt konnte nicht zurückgesetzt werden.');}
  const result=await command({type:'calibrate',sample:inputPacket(released)});
  if(!result?.calibrated)throw new Error('Referenzkalibrierung fehlgeschlagen.');
  referenceControls.completeCalibration(true);referenceHint='Kalibriert · Tasten loslassen, dann AX zum Starten.';
  message(referenceHint);
 }catch(e){referenceControls.completeCalibration(false);referenceHint=e.message;message(e.message,true);}
 finally{referenceBusy=false;}
}
function xrInput(frame){
 $('input-source').textContent=replayMode()?'Aufzeichnung · VR':'WebXR · Headset / Controller';
 if(!session||!referenceSpace||!frame)return;
 const head=frame.getViewerPose(referenceSpace);
 if(!head){referenceControls.pause();sendInput({source:'webxr_controller',head:invalidPose(),right:invalidPose(),axes:[0,0],trigger:0,squeeze:0,primary:false,secondary:false,clutch:false});return;}
 if(xrAnchorPending){content.matrixAutoUpdate=false;content.matrix.fromArray(head.transform.matrix);content.matrixWorldNeedsUpdate=true;xrAnchorPending=false;}
 let rightSource=null,leftSource=null;
 for(const source of session.inputSources){if(source.handedness==='right')rightSource=source;else if(source.handedness==='left')leftSource=source;}
 const aim=rightSource?frame.getPose(rightSource.targetRaySpace,referenceSpace):null;
 const grip=rightSource?.gripSpace?frame.getPose(rightSource.gripSpace,referenceSpace):aim;
 const leftGrip=leftSource?.gripSpace?frame.getPose(leftSource.gripSpace,referenceSpace):null;
 const g=rightSource?.gamepad,lg=leftSource?.gamepad,ref=referenceMode();
 if(button(g,3)>.8&&button(g,1)<.2&&state?.action?.confirmed){if(phasePressStart===null)phasePressStart=performance.now();if(!phasePressUsed&&performance.now()-phasePressStart>1000){phasePressUsed=true;actionPhase(state.action.phase==='execute'?'approach':'execute');}}else{phasePressStart=null;phasePressUsed=false;}
 if(button(g,1)<.2)xrRequireGripRelease=false;
 const axes=g?.axes?.length>=4?[g.axes[2],g.axes[3]]:(g?.axes?.slice(0,2)||[0,0]);pointerUV=rayPointer(aim||head)||pointerUV;
 if(aim){rayLine.matrixAutoUpdate=false;rayLine.matrix.fromArray(aim.transform.matrix);rayLine.visible=true;}else rayLine.visible=false;
 const xrMapping=ref?(referenceBasis||referenceInputRotation(xrPose(head).quaternion_xyzw)):xrToWorldRotation(firstCameraRows,new THREE.Quaternion().setFromRotationMatrix(content.matrix).toArray());
 const sample={source:'webxr_controller',orientation_mode:'pose',xr_to_world_rotation:xrMapping,head:xrPose(head),left:xrPose(leftGrip),right:xrMapping?xrPose(grip):invalidPose(),axes,trigger:button(g,0),squeeze:xrRequireGripRelease?0:button(g,1),primary:button(g,4)>.5,secondary:button(g,5)>.5,clutch:!xrRequireGripRelease&&button(g,1)>.5,pointer_uv:pointerUV};
 if(ref){
  const posture=referencePosture(sample.head,sample.left,sample.right);
  const released=button(g,0)<.2&&button(g,1)<.2&&button(lg,0)<.2&&button(lg,1)<.2;
  const valid=sample.right.valid&&(referenceControls.calibrated||sample.left.valid)&&fresh()&&!!xrMapping;
  if(!referenceBusy){
   const actions=referenceControls.update(lg,g,{now:performance.now()/1000,trackingValid:valid,ownsControl:ownsControl(),stopped:!!state?.stopped,poseReady:released&&(posture.ready||referenceControls.calibrated)});
   if(!referenceControls.calibrated&&referenceControls.phase!=='calibrating')referenceHint=released?posture.message:'Griffe und Trigger zum Kalibrieren loslassen.';
   for(const action of actions){
    if(action==='calibrate')calibrateReference(sample);
    else if(action==='stop'){command({type:'stop'});referenceHint='Halt · ABXY in angeglichener Haltung startet neu.';}
    else if(action==='confirm'){referenceControls.pause();command({type:state?.selection?.mode==='confirmed'&&state?.action?.required&&!state?.action?.confirmed?'action_confirm':'confirm'});}
    else if(action==='release'){referenceControls.pause();command({type:'release'});}
    else if(action==='assist'){referenceControls.pause();command({type:'assist_enable',enabled:!state?.assist?.enabled});}
    else if(action==='pause')referenceHint='Pausiert · AX setzt mit frischer Handreferenz fort.';
    else if(action==='start')referenceHint='Teleoperation aktiv · AX pausiert · ABXY hält an.';
   }
  }
  const eyeDown=button(lg,3)>.5;if(eyeDown&&!referenceEyeDown)command({type:'eye_enable',enabled:!state?.eye_enabled});referenceEyeDown=eyeDown;
  sample.primary=false;sample.secondary=false;sample.clutch=referenceControls.active&&!referenceBusy&&valid;sample.squeeze=sample.clutch?1:0;
  lastReferenceSample=sample;
 }else for(const action of xrButtons.update(lg,state))command(action);
 projectEyeDevice(head);updateHUD(head);
 if(xrNeedsInvalidation){referenceControls.invalidate();sendInput({...sample,right:invalidPose(),axes:[0,0],trigger:0,squeeze:0,primary:false,secondary:false,clutch:false}).then(sent=>{if(sent)xrNeedsInvalidation=false;});return;}
 if(!referenceBusy)sendInput(sample);
 window.__assistDiagnostics={...window.__assistDiagnostics,xrActive:true,referenceMode:ref,teleopPhase:referenceControls.phase,headTracked:true,rightTracked:!!grip,leftTracked:!!leftGrip};
}
async function sendInput(sample,force=false){
 if(!state||replayMode()||!ownsControl())return false;
 const now=performance.now();if(!force&&(inputBusy||now-lastInputAt<40))return false;lastInputAt=now;
 if(!fresh())sample={...sample,right:invalidPose(),squeeze:0,clutch:false};
 if(!measuredFresh()){sample={...sample};delete sample.pointer_uv;}
 const packet=inputPacket(sample);inputBusy=true;
 try{const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(packet),keepalive:force});
  if(!r.ok){if(r.status!==409)message('Eingabe abgelehnt · '+r.status,true);return false;}
  return (await r.json()).accepted===true;
 }catch{return false;}finally{inputBusy=false;}
}
function desktopInput(dt){if(session||eyeCalibration||replayMode()||!ownsControl()||!document.hasFocus())return;const speed=.12,forward=keys.has('w')?1:keys.has('s')?-1:0,side=keys.has('d')?1:keys.has('a')?-1:0,up=keys.has('e')?1:keys.has('q')?-1:0;const gripHeld=held||keys.has(' ');if(gripHeld){virtualHand[0]+=side*speed*dt;virtualHand[1]+=up*speed*dt;virtualHand[2]-=forward*speed*dt;}
 const gamepad=[...(navigator.getGamepads?.()||[])].find(g=>g?.connected&&g.mapping==='standard'),axes=gamepad?[gamepad.axes[0]||0,gamepad.axes[1]||0]:[0,0];const deadman=gripHeld||(gamepad?button(gamepad,6)>.5:false);$('deadman').setAttribute('aria-pressed',String(deadman));$('input-source').textContent=gamepad?'Desktop-Gamepad · virtuelle Hand':'Tastatur / Maus · virtuelle Hand';
 sendInput({source:gamepad?'desktop_gamepad':'keyboard_controller',orientation_mode:'position',head:pose([0,0,0]),right:pose([...virtualHand]),axes,trigger:gamepad?button(gamepad,7):(keys.has('r')?1:0),squeeze:deadman?1:0,primary:gamepad?button(gamepad,0)>.5:false,secondary:gamepad?button(gamepad,1)>.5:false,clutch:deadman,pointer_uv:pointerUV});}
function animation(now,frame){const overlayKey=visibleTargets().map(t=>t.id).join('|');if(overlayKey!==lastOverlayKey){lastOverlayKey=overlayKey;draw();updateBoxes();}if(now-lastAgePaint>100){lastAgePaint=now;paintAges();}const dt=Math.min((now-lastTick)/1000,.05);lastTick=now;if(session)xrInput(frame);else desktopInput(dt);if(renderer&&(view3D||session))renderer.render(scene,camera);const live=fresh();$('connection').textContent=replayMode()?'Aufzeichnung':live?'Kamera verbunden':image?'Kamerabild veraltet':'Warte auf Simulation';$('connection').parentElement.classList.toggle('live',live);if(!replayMode()&&!live&&image&&!lastFrameError)$('assist-status').textContent='Gesperrt · Kamera veraltet';}
if(renderer)renderer.setAnimationLoop(animation);else{const tick=t=>{animation(t);requestAnimationFrame(tick);};requestAnimationFrame(tick);}
async function initXR(){if(!isSecureContext){$('xr-status').textContent='VR benötigt HTTPS: diese Host-Adresse mit einem im Headset vertrauenswürdigen Zertifikat öffnen.';return;}if(!navigator.xr){$('xr-status').textContent='Dieser Browser stellt kein WebXR bereit. In Quest/PICO im Headset-Browser über HTTPS öffnen.';return;}try{const supported=await navigator.xr.isSessionSupported('immersive-vr');$('xr-status').textContent=supported?'VR verfügbar · Referenzmodus: ABXY kalibriert · AX startet/pausiert · rechter Stick wählt.':'Kein immersives Headset erkannt. Desktop und 3D-Vorschau sind nutzbar.';}catch(e){$('xr-status').textContent='WebXR-Prüfung: '+e.message;}}
const xrLifecycle=new ImmersiveSessionController(()=>{
 session=null;referenceSpace=null;held=false;keys.clear();xrButtons.reset();referenceControls.invalidate();
 content.matrixAutoUpdate=true;content.position.set(0,0,0);content.quaternion.identity();content.updateMatrix();
 hud.visible=false;rayLine.visible=false;command({type:'release'});$('enter-vr').textContent='VR starten ↗';
 // Three's end listener restores its render target later in this dispatch.
 queueMicrotask(()=>{if(!session)resize();});
});
$('enter-vr').onclick=async()=>{
 if(xrLifecycle.busy||$('enter-vr').disabled)return;
 if(!renderer)return;
 if(!navigator.xr||!isSecureContext){message('VR benötigt den Headset-Browser über USB-localhost oder vertrauenswürdiges HTTPS.',true);return;}
 $('enter-vr').disabled=true;
 try{
  if(xrLifecycle.session){await xrLifecycle.end();return;}
  $('enter-vr').textContent='VR wird gestartet …';
  await xrLifecycle.start(navigator.xr,async acquired=>{
   session=acquired;
   referenceSpace=await acquired.requestReferenceSpace('local');xrEpoch++;xrRequireGripRelease=true;
   referenceSpace.addEventListener('reset',()=>{referenceControls.invalidate();xrEpoch++;deviceGaze=null;lastProjectedDeviceKey=null;xrNeedsInvalidation=true;xrRequireGripRelease=true;xrButtons.reset();firstCameraWorld=null;updateSurface();xrAnchorPending=true;message('VR-Koordinaten zurückgesetzt · Griff lösen und X neu kalibrieren.');});
   camera.position.set(0,0,0);xrAnchorPending=true;firstCameraWorld=null;updateSurface();
   await renderer.xr.setSession(acquired);renderer.xr.setReferenceSpace(referenceSpace);
  });
  message(replayMode()?'Aufgezeichnete Sitzung in VR · keine Steuerung':referenceMode()?'Referenzhaltung einnehmen · ABXY kalibriert und übernimmt · AX startet.':'VR aktiv. X kalibriert, Y schaltet Assistenz, linker Stickklick schaltet Augen. Links Griff+Trigger hält an.');
 }catch(e){
  console.error('VR startup/exit failed',e);
  const conflict=/already.*active.*immersive|active.*XRSession/i.test(e.message);
  message('VR konnte nicht starten: '+e.message+(conflict?' · Andere VR-Sitzung oder alte Test-Tabs schließen; bei Bedarf den Quest-Browser neu starten.':'')+(e.cleanupError?' · VR-Sitzung noch offen: VR beenden drücken oder Browser neu starten.':''),true);
 }finally{
  $('enter-vr').textContent=xrLifecycle.session?'VR beenden':'VR starten ↗';$('enter-vr').disabled=false;
 }
};
for(const b of document.querySelectorAll('[data-direction]'))b.onclick=()=>command({type:'direction',direction:b.dataset.direction});
$('control-claim').onclick=async()=>{held=false;keys.clear();xrRequireGripRelease=true;const r=await command({type:'control_claim',deadman_released:true});if(r)message('Steuerung übernommen · Freigabe lösen, Hand kalibrieren und erneut freigeben.');};
$('control-release').onclick=async()=>{held=false;keys.clear();xrRequireGripRelease=true;await command({type:'control_release'});};
$('stop').onclick=()=>{held=false;keys.clear();command({type:'stop'});};$('resume').onclick=()=>command({type:'resume'});
$('action-confirm').onclick=()=>{held=false;keys.delete(' ');command({type:'action_confirm'});};$('action-execute').onclick=()=>actionPhase('execute');$('action-approach').onclick=()=>actionPhase('approach');
$('confirm').onclick=()=>command({type:'confirm'});$('release').onclick=()=>{held=false;keys.delete(' ');command({type:'release'});};$('eye-toggle').onchange=e=>command({type:'eye_enable',enabled:e.target.checked});$('assist-toggle').onchange=e=>command({type:'assist_enable',enabled:e.target.checked});$('calibrate').onclick=()=>command({type:'calibrate'});
$('deadman').addEventListener('pointerdown',e=>{e.preventDefault();held=true;$('deadman').setPointerCapture(e.pointerId);});for(const event of ['pointerup','pointercancel','lostpointercapture'])$('deadman').addEventListener(event,()=>held=false);
canvas.addEventListener('pointermove',e=>{if(!state?.camera)return;const uv=normalizedPointer(e,canvas.getBoundingClientRect(),state.camera.width,state.camera.height);if(!uv)return;if((held||keys.has(' '))&&pointerUV){virtualHand[0]+=(uv[0]-pointerUV[0])*.55;virtualHand[1]-=(uv[1]-pointerUV[1])*.40;}pointerUV=uv;draw();});
canvas.addEventListener('click',()=>{if(measuredFresh()&&pointerUV)command({type:'pointer',uv:pointerUV,frame_id:state.frame_id});});
window.addEventListener('keydown',e=>{if(eyeCalibration||e.target.matches('input,textarea,select,[contenteditable=true]'))return;const key=e.key.toLowerCase();if(['arrowup','arrowdown','arrowleft','arrowright',' ','enter','escape','w','a','s','d','q','e','c','r'].includes(key))e.preventDefault();keys.add(key);if(e.repeat)return;const dir={arrowup:'up',arrowdown:'down',arrowleft:'left',arrowright:'right'}[key];if(dir)command({type:'direction',direction:dir});if(key==='enter')command({type:state?.selection?.mode==='confirmed'&&state?.action?.required&&!state.action.confirmed?'action_confirm':'confirm'});if(key==='escape'){held=false;keys.clear();command({type:'release'});}if(key==='c')command({type:'calibrate'});});window.addEventListener('keyup',e=>keys.delete(e.key.toLowerCase()));
function loseFocus(){referenceControls.pause();held=false;keys.clear();sendInput({source:session?'webxr_controller':'keyboard_controller',head:invalidPose(),right:invalidPose(),axes:[0,0],trigger:0,squeeze:0,primary:false,secondary:false,clutch:false},true);}
window.addEventListener('blur',loseFocus);document.addEventListener('visibilitychange',()=>{if(document.hidden)loseFocus();});window.addEventListener('pagehide',loseFocus);

const calibrationPoints=[[.15,.15],[.5,.15],[.85,.15],[.15,.5],[.5,.5],[.85,.5],[.15,.85],[.5,.85],[.85,.85]];
function finishEyeCalibration(){eyeCalibration=null;$('eye-calibration-panel').hidden=true;document.body.classList.remove('calibrating');draw();}
$('eye-calibrate').onclick=()=>{if(!fresh()){message('Für die Kalibrierung wird ein aktuelles Kamerabild benötigt.',true);return;}if(session){message('Kalibrierung im Desktop-Kamerabild vorbereiten; immersive Kalibrierung benötigt einen zum VR-Bild passenden Tracker-Adapter.',true);return;}loseFocus();setView(false);eyeCalibration={points:calibrationPoints,index:0,samples:[]};$('eye-calibration-panel').hidden=false;document.body.classList.add('calibrating');$('calibration-title').textContent='Augen kalibrieren · Punkt 1 / 9';$('calibration-instruction').textContent='Auf den weißen Punkt im Kamerabild schauen. Zehn frische Rohmessungen werden gemittelt. Der Tracker muss /api/gaze Rohdaten liefern.';draw();};
$('calibration-cancel').onclick=()=>{finishEyeCalibration();message('Augenkalibrierung abgebrochen. Kopf und Controller bleiben verfügbar.');};
$('calibration-capture').onclick=async()=>{const calibration=eyeCalibration;if(!calibration)return;$('calibration-capture').disabled=true;const values=[],seen=new Set(),deadline=performance.now()+3500;try{while(values.length<10&&performance.now()<deadline){if(eyeCalibration!==calibration)return;const r=await fetch('/api/gaze',{cache:'no-store'});if(!r.ok)throw new Error('Tracker-Rohdaten nicht verfügbar');const data=await r.json(),s=data.sample;if(s?.raw?.length===2&&s.raw.every(Number.isFinite)&&s.valid&&s.confidence>=.6&&!['stale','invalid','no_device','low_confidence'].includes(data.status)&&!seen.has(s.seq)){seen.add(s.seq);values.push(s.raw);}$('calibration-instruction').textContent=`Punkt weiter ansehen · ${values.length} / 10 frische Messungen`;await new Promise(resolve=>setTimeout(resolve,70));}if(values.length<10)throw new Error('Keine zehn frischen, gültigen Rohmessungen. Tracker verbinden und denselben Punkt erneut aufnehmen.');const raw=[0,1].map(i=>values.reduce((sum,v)=>sum+v[i],0)/values.length);calibration.samples[calibration.index]={raw,uv:calibration.points[calibration.index]};calibration.index++;if(calibration.index===calibration.points.length){const r=await fetch('/api/calibration/eye',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({samples:calibration.samples,client_id:clientId})}),result=await r.json();if(!r.ok)throw new Error(result.error||'Kalibrierung abgelehnt');finishEyeCalibration();message(`Kalibrierung gespeichert. Anpassungsrestfehler: ${(100*result.fit_rms_uv).toFixed(2)} % der Bildgröße. Das ist keine unabhängige Genauigkeitsmessung; andere Prüfpunkte separat testen.`);}else{$('calibration-title').textContent=`Augen kalibrieren · Punkt ${calibration.index+1} / 9`;$('calibration-instruction').textContent='Den nächsten weißen Punkt ansehen und Messung aufnehmen.';draw();}}catch(e){if(eyeCalibration===calibration){if(calibration.index>=9)calibration.index=8;$('calibration-instruction').textContent=e.message;}}finally{$('calibration-capture').disabled=false;}};

window.__assistDiagnostics={geometry:'waiting',boxCount:0};if(new URLSearchParams(location.search).get('view')==='3d'&&renderer)setView(true);initXR();poll();

// Lease liveness is separate from motion freshness; stale input still holds.
setInterval(()=>{if(ownsControl()&&!replayMode()&&!document.hidden&&(session||document.hasFocus()))command({type:'control_heartbeat'});},400);
