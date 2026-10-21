/** Read-only real-archive UI probe. No control claim or POST is performed. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/home/loya2/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const [url,out]=process.argv.slice(2),evidence=JSON.parse(fs.readFileSync(path.join(out,'evidence.json')));
const requests=[],errors=[];
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||'/usr/bin/google-chrome',args:['--no-sandbox','--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1600,height:1160}});
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>requests.push({method:request.method(),url:request.url()}));
  await page.goto(url+'/?view=3d');
  await page.waitForFunction(({i,j})=>window.__assistDiagnostics?.frameId===j&&window.__assistDiagnostics?.measurementFrameId===i&&window.__assistDiagnostics?.displayReprojected&&window.__assistDiagnostics?.boxCount>0,{i:evidence.measurement_frame_id,j:evidence.raw_frame_id});
  // Truthful fixture label, separate from the unchanged product renderer.
  await page.evaluate(({i,j,recording})=>{
   const label=document.createElement('div');label.textContent=`VISUELLE ARCHIVPRÜFUNG · ${recording} · Messung ${i} / Bild ${j} · eingefrorene Offline-Zeit · keine Steuerung / neue Inferenz`;
   label.style.cssText='background:#46310c;color:#fff0c5;padding:10px 18px;font:14px system-ui;border-bottom:1px solid #92703a';document.body.prepend(label);
  },{i:evidence.measurement_frame_id,j:evidence.raw_frame_id,recording:path.basename(evidence.recording)});
  const viewport=await page.locator('#three-container').boundingBox();
  await page.mouse.move(viewport.x+viewport.width*.65,viewport.y+viewport.height*.55);
  const reprojected=await page.evaluate(()=>window.__assistDiagnostics);
  assert.equal(reprojected.frameId,evidence.raw_frame_id);
  assert.equal(reprojected.measurementFrameId,evidence.measurement_frame_id);
  assert.equal(reprojected.pickFrameId,evidence.measurement_frame_id);
  assert.equal(reprojected.displayReprojected,true);assert(reprojected.boxCount>0);
  for(let n=0;n<16;n++)assert(Math.abs(reprojected.surfaceToMeasurement[n]-evidence.expected_surface_to_measurement_column_major[n])<1e-8);
  await page.screenshot({path:path.join(out,'actual_multirate_3d.png'),fullPage:true});
  await page.click('#view-2d');
  await page.waitForFunction(i=>window.__assistDiagnostics?.frameId===i&&window.__assistDiagnostics?.displaySource==='measurement_paired',evidence.measurement_frame_id);
  const fallback=await page.evaluate(()=>window.__assistDiagnostics);
  assert.equal(fallback.pickFrameId,evidence.measurement_frame_id);assert(fallback.boxCount>0);
  await page.screenshot({path:path.join(out,'actual_measurement_2d.png'),fullPage:true});
  assert.equal(errors.length,0,errors.join('\n'));
  assert(requests.every(request=>request.method==='GET'),'Observer attempted a write');
  const requestedIds={jpeg:[],depth:[]};
  for(const request of requests){const u=new URL(request.url);if(u.pathname==='/api/frame.jpg')requestedIds.jpeg.push(Number(u.searchParams.get('frame_id')));if(u.pathname==='/api/depth.bin')requestedIds.depth.push(Number(u.searchParams.get('frame_id')));}
  assert(requestedIds.jpeg.includes(evidence.raw_frame_id));assert(requestedIds.depth.includes(evidence.raw_frame_id));assert(requestedIds.depth.includes(evidence.measurement_frame_id));
  const staticRoot=path.resolve(__dirname,'../teleop_assist/static');
  const report={kind:'actual_archived_isaac_multirate_UI_visual_probe',passed:true,scope:'rendering and read-only observer contracts; no model inference, controller, Isaac runtime, or immersive headset test',
   frozen_offline_clock:true,evidence_file:'evidence.json',reprojected,fallback,requests,errors,checks:['real archived RGB-D displayed','newer actual camera extrinsics geometrically reprojected','CAD boxes retain measured frame ID','hover leaves picking depth on measured frame','moving-camera 2D falls back to measured acquisition','zero POSTs','zero JavaScript errors'],
   source_hashes:Object.fromEntries(['app.mjs','display.mjs','geometry.mjs','index.html','style.css'].map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(path.join(staticRoot,name))).digest('hex')]))};
  fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:true,measurement:evidence.measurement_frame_id,raw:evidence.raw_frame_id,boxes:reprojected.boxCount,report:path.join(out,'report.json')}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
