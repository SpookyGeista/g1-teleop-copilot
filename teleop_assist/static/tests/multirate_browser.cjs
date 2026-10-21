/** Real Chromium integration with synthetic immutable acquisitions, no Isaac. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http'),crypto=require('node:crypto');
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/home/loya2/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'..'),out=process.env.ASSIST_TEST_OUTPUT;
if(!out||fs.existsSync(out))throw new Error('ASSIST_TEST_OUTPUT must name a new directory');fs.mkdirSync(out,{recursive:true});
const camera={width:640,height:360,intrinsics:[[500,0,320],[0,500,180],[0,0,1]],T_world_camera:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],depth_convention:'optical_z_meters'};
let stamp=performance.now()/1000,measurementId=1,live=false,movingCamera=false,replay=false,failRaw=false,owner=null;
const commands=[],requests=[],errors=[];
const depth=value=>{const b=Buffer.alloc(640*360*4);for(let i=0;i<640*360;i++)b.writeFloatLE(value,4*i);return b;};
const depths={measured:depth(1.15),live:depth(2.)},jpeg=fs.readFileSync(path.join(__dirname,'fixture.jpg'));
const state=()=>{const now=performance.now()/1000,c=structuredClone(camera);if(movingCamera)c.T_world_camera[0][3]=.001;return {
 timestamp:now,frame_id:measurementId,frame_age:now-stamp,camera,settings:{frame_timeout:.7},
 display_frame:live?{frame_id:100+Math.floor(now*15),timestamp:now-.025,camera_id:'head',camera:c,source:'raw_camera_capture'}:null,
 control_session:{owner_client_id:owner},viewer_only:replay,mode:replay?'recorded_replay':'fixed_base_manipulation_fixture',
 targets:[{id:'handle',label:'Griff',bbox_px:[210,145,280,215],center_camera:[-.165,0,1.15],box_camera:{center:[-.165,0,1.15],size:[.16,.16,.04],quaternion_xyzw:[0,0,0,1]},confidence:.94}],
 selection:{id:'handle',mode:'preview'},eye_enabled:false,eye_status:'disabled',assist:{enabled:false,active:false},robot:{ee_position:[.42,-.15,1.03]}};};
const server=http.createServer(async(req,res)=>{const u=new URL(req.url,'http://localhost');
 if(u.pathname==='/api/state'){res.setHeader('Content-Type','application/json');return res.end(JSON.stringify(state()));}
 if(u.pathname==='/api/frame.jpg'||u.pathname==='/api/depth.bin'){
  const id=Number(u.searchParams.get('frame_id'));requests.push({path:u.pathname,id});
  if(failRaw&&id>=100){res.writeHead(410);return res.end('expired');}
  res.setHeader('Content-Type',u.pathname.endsWith('.jpg')?'image/jpeg':'application/octet-stream');
  return res.end(u.pathname.endsWith('.jpg')?jpeg:id>=100?depths.live:depths.measured);
 }
 if(u.pathname==='/api/command'){const chunks=[];for await(const c of req)chunks.push(c);const data=JSON.parse(Buffer.concat(chunks));commands.push(data);if(data.type==='control_claim')owner=data.client_id;res.setHeader('Content-Type','application/json');return res.end(JSON.stringify({control_session:{owner_client_id:owner}}));}
 const name=u.pathname==='/'?'index.html':u.pathname.replace(/^\/static\//,''),file=path.resolve(root,name);
 if(!file.startsWith(root+path.sep)){res.writeHead(403);return res.end();}
 try{res.setHeader('Content-Type',({'.mjs':'text/javascript','.js':'text/javascript','.css':'text/css','.html':'text/html'})[path.extname(file)]||'text/plain');return res.end(fs.readFileSync(file));}catch{res.writeHead(404);res.end();}
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||'/usr/bin/google-chrome',args:['--no-sandbox','--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1040}});page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/?view=3d`);
  await page.waitForFunction(()=>window.__assistDiagnostics?.measurementFrameId===1);stamp=performance.now()/1000;
  assert.equal(commands.filter(c=>c.type==='input').length,0);
  await page.click('#control-claim');await page.waitForFunction(()=>!document.querySelector('#confirm').disabled);
  live=true;stamp=performance.now()/1000;
  await page.waitForFunction(()=>window.__assistDiagnostics?.frameId>100&&window.__assistDiagnostics?.boxCount===1);
  const first=await page.evaluate(()=>window.__assistDiagnostics);
  assert.equal(first.measurementFrameId,1);assert.equal(first.pickFrameId,1);assert.equal(first.displaySource,'raw_camera_capture');
  assert.equal(first.displayDepthCenter,2);assert(Math.abs(first.measuredDepthCenter-1.15)<1e-6);
  await page.click('#view-2d');await page.locator('#camera-canvas').click({position:{x:180,y:150}});
  await page.waitForTimeout(50);assert(commands.some(c=>c.type==='pointer'&&c.frame_id===1));
  assert(commands.filter(c=>c.type==='input').every(c=>c.frame_id===1),'Input proposals must keep measured frame identity');
  await page.screenshot({path:path.join(out,'multirate_live.png'),fullPage:true});
  await page.waitForFunction(()=>window.__assistDiagnostics?.boxCount===0);
  await page.waitForFunction(()=>document.querySelector('#frame-age-status').textContent.includes('Messung veraltet'));
  const stale=await page.evaluate(()=>window.__assistDiagnostics);assert(stale.frameId>first.frameId);assert.equal(stale.measurementFrameId,1);
  movingCamera=true;stamp=performance.now()/1000;
  await page.waitForFunction(()=>window.__assistDiagnostics?.frameId===1&&window.__assistDiagnostics?.displaySource==='measurement_paired');
  const fallback=await page.evaluate(()=>window.__assistDiagnostics);assert(Math.abs(fallback.displayDepthCenter-1.15)<1e-6);
  await page.screenshot({path:path.join(out,'multirate_fallback.png'),fullPage:true});
  await page.click('#view-3d');stamp=performance.now()/1000;
  await page.waitForFunction(()=>window.__assistDiagnostics?.frameId>100&&window.__assistDiagnostics?.displayReprojected===true);
  const reprojected=await page.evaluate(()=>window.__assistDiagnostics);
  assert(Math.abs(reprojected.surfaceToMeasurement[12]-.001)<1e-12);
  assert.equal(reprojected.displayDepthCenter,2);assert.equal(reprojected.pickFrameId,1);
  await page.screenshot({path:path.join(out,'multirate_reprojected.png'),fullPage:true});
  await page.click('#view-2d');
  await page.waitForFunction(()=>window.__assistDiagnostics?.frameId===1);
  movingCamera=false;failRaw=true;
  await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('nicht mehr verfügbar'));
  assert.equal((await page.evaluate(()=>window.__assistDiagnostics)).frameId,1,'410 must not replace either half of displayed RGB-D');
  failRaw=false;measurementId=2;stamp=performance.now()/1000;
  await page.waitForFunction(()=>window.__assistDiagnostics?.frameId>100&&window.__assistDiagnostics?.measurementFrameId===2&&window.__assistDiagnostics?.pickFrameId===2);
  replay=true;
  await page.waitForFunction(()=>window.__assistDiagnostics?.frameId===2&&document.querySelector('#source-badge').textContent.includes('AUFZEICHNUNG'));
  await page.waitForTimeout(100);const before=commands.length;await page.keyboard.press('ArrowRight');await page.waitForTimeout(150);assert.equal(commands.length,before,'Replay must remain POST-free');
  assert.equal(errors.length,0,errors.join('\n'));
  for(const id of new Set(requests.filter(r=>r.id>=100&&r.path.endsWith('.jpg')).map(r=>r.id)))assert(requests.some(r=>r.id===id&&r.path.endsWith('.bin')),'Raw JPEG/depth must use identical IDs');
  const files=['app.mjs','display.mjs','geometry.mjs','gaze.mjs','index.html','style.css'];
  const report={kind:'synthetic_multirate_browser_fixture_not_isaac',passed:true,immersive_vr_tested:false,
   checks:['raw video advances while measurement remains fixed','independent measured raycast depth','measured pointer/input identity','measurement-age hiding despite fresh video','2D exact-camera mismatch fallback','3D calibrated moving-camera RGBD reprojection','RGBD pair retained on410','new measurement switches pick geometry atomically','replay sends no commands','no JavaScript errors'],
   first_live:first,stale_live:stale,fallback,reprojected,frontendHashes:Object.fromEntries(files.map(f=>[f,crypto.createHash('sha256').update(fs.readFileSync(path.join(root,f))).digest('hex')]))};
  fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
 }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exit(1);});
