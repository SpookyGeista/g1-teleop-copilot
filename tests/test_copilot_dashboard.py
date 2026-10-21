import http.client
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import pytest

from teleop_assist.copilot.dashboard import Dashboard, CommandConflict, OperatorOwnership
from teleop_assist.copilot.mailbox import ControlWriter, atomic_json, read_json
from teleop_assist.copilot.mailbox import host_clock_id
from tools.copilot.launch_xr import NODE


def request(dashboard, path, *, method='GET', payload=None, token=None, sequence=1, headers=None):
    connection = http.client.HTTPConnection(*dashboard.address, timeout=3)
    request_headers = dict(headers or {})
    body = None
    if payload is not None:
        body = json.dumps(payload)
        request_headers.setdefault('Content-Type', 'application/json')
    if token:
        request_headers['X-Copilot-Token'] = token
        request_headers['X-Copilot-Sequence'] = str(sequence)
    try:
        connection.request(method, path, body=body, headers=request_headers)
        result = connection.getresponse()
        data = result.read()
        content = json.loads(data) if result.getheader('Content-Type', '').startswith('application/json') else data
        return result.status, content, dict(result.getheaders())
    finally:
        connection.close()


@pytest.fixture
def dashboard(tmp_path):
    atomic_json(tmp_path / 'ui_state.json', {'objects':[{'target_id':'cup', 'label':'Cup', 'bbox_px':[1,2,20,30]}]})
    atomic_json(tmp_path / 'latest_proposals.json', {'candidates':[{'candidate_id':'grasp-1'}]})
    service = Dashboard(tmp_path, port=0).start()
    try:
        yield service
    finally:
        service.close()


def session(dashboard):
    status, result, _ = request(dashboard, '/api/session')
    assert status == 200
    return result['token']


def test_host_origin_csrf_and_fetch_site_gates(dashboard):
    token = session(dashboard)
    for headers in ({'Host':'evil.example'}, {'Origin':'http://evil.example'},
                    {'Sec-Fetch-Site':'cross-site'}, {'Host':f'127.0.0.1:{dashboard.address[1]+1}'}):
        status, _, _ = request(dashboard, '/api/control', method='POST',
            payload={'action':'hold'}, token=token, headers=headers)
        assert status == 403
    assert request(dashboard, '/api/control', method='POST', payload={'action':'hold'})[0] == 403
    assert request(dashboard, '/api/state')[0] == 403
    assert request(dashboard, '/api/state', token='invalid')[0] == 403
    status, _, headers = request(dashboard, '/')
    assert status == 200 and headers['X-Frame-Options'] == 'DENY'
    assert "frame-ancestors 'none'" in headers['Content-Security-Policy']


def test_two_tabs_cannot_override_owners_selection_hold_or_release(dashboard):
    first, second = session(dashboard), session(dashboard)
    assert first != second
    assert request(dashboard, '/api/control', method='POST', token=first,
        payload={'action':'select','candidate_id':'grasp-1'})[0] == 200
    for sequence, action in enumerate(('confirm','hold','release'), 1):
        assert request(dashboard, '/api/control', method='POST', token=second,
            sequence=sequence, payload={'action':action})[0] == 409
    _, state, _ = request(dashboard, '/api/state', token=second)
    assert not state['ownership']['available'] and not state['ownership']['owned_by_this_client']
    assert request(dashboard, '/api/control', method='POST', token=first, sequence=2,
        payload={'action':'hold'})[0] == 200
    assert read_json(dashboard.directory / 'copilot_control.json')['assist_held']
    assert request(dashboard, '/api/control', method='POST', token=first, sequence=4,
        payload={'action':'release'})[0] == 200
    # A previously sent hold arriving late may not re-arm after release.
    assert request(dashboard, '/api/control', method='POST', token=first, sequence=3,
        payload={'action':'hold'})[0] == 409
    assert read_json(dashboard.directory / 'copilot_control.json')['release']


def test_reselecting_same_object_has_a_new_explicit_request_revision(dashboard):
    token = session(dashboard)
    result = []
    for sequence in (1, 2):
        assert request(dashboard, '/api/target', method='POST', token=token, sequence=sequence,
            payload={'target_id':'cup'})[0] == 200
        result.append(read_json(dashboard.directory / 'target_selection.json'))
    assert result[1]['revision'] > result[0]['revision']
    assert result[1]['request_id'] != result[0]['request_id']


@pytest.mark.parametrize('payload', [[], {'action':'unknown'}, {'action':'hold','extra':True},
    {'action':'select','candidate_id':'not-offered'}, {'action':'confirm','candidate_id':'grasp-1'},
    {'action':'select','candidate_id':123}, {'action':None}])
def test_wrong_command_schema_never_changes_control(dashboard, payload):
    token = session(dashboard)
    before = read_json(dashboard.directory / 'copilot_control.json')
    assert request(dashboard, '/api/control', method='POST', token=token, payload=payload)[0] == 400
    assert read_json(dashboard.directory / 'copilot_control.json') == before


def test_ownership_timeout_releases_then_requires_new_selection(tmp_path):
    now = [10.]
    control = ControlWriter(tmp_path, clock=lambda: now[0])
    lease = OperatorOwnership(control, clock=lambda: now[0])
    lease.require('one', acquire=True)
    now[0] += 1.
    lease.heartbeat('one')
    now[0] += 1.5
    with pytest.raises(CommandConflict):
        lease.require('two', acquire=True)
    now[0] += .6
    lease.require('two', acquire=True)
    assert lease.client_id == 'two'
    assert read_json(tmp_path / 'copilot_control.json')['release']


def test_close_before_start_and_repeated_close_are_nonblocking(tmp_path):
    service = Dashboard(tmp_path, host='localhost', port=0)
    assert service.address[0] == '127.0.0.1'
    started = time.monotonic()
    service.close(); service.close()
    assert time.monotonic() - started < .5
    with pytest.raises(RuntimeError, match='closed'):
        service.start()


def test_ipv6_uses_ipv6_loopback_and_bracketed_origin(tmp_path):
    if not socket.has_ipv6:
        pytest.skip('IPv6 unavailable in this environment')
    try:
        service = Dashboard(tmp_path, host='::1', port=0).start()
    except OSError as exc:
        if exc.errno in (97, 99):
            pytest.skip('IPv6 loopback unavailable in this environment')
        raise
    try:
        assert service.url.startswith('http://[::1]:')
        assert request(service, '/api/session')[0] == 200
    finally:
        service.close()


def test_dashboard_script_syntax_and_service_schema(tmp_path):
    html = Path(__file__).resolve().parents[1] / 'teleop_assist/copilot/static/index.html'
    code = html.read_text().split('<script>', 1)[1].split('</script>', 1)[0]
    script = tmp_path / 'dashboard.js'; script.write_text(code)
    subprocess.run([str(NODE), '--check', str(script)], capture_output=True, check=True, timeout=10)
    assert 'p.objects||[]' in code and 't.target_id' in code
    assert "'X-Copilot-Token'" in code and "'X-Copilot-Sequence'" in code
    assert 'pendingHold' in code and "command(clearChoice?'release':'idle')" in code


def test_image_hash_request_rejects_a_later_image_instead_of_mislabeling_it(dashboard):
    frame=dashboard.directory/'frame.jpg';frame.write_bytes(b'original recorded JPEG bytes')
    digest=hashlib.sha256(frame.read_bytes()).hexdigest()
    assert request(dashboard,'/frame.jpg?sha256='+digest)[0]==200
    frame.write_bytes(b'a different image')
    assert request(dashboard,'/frame.jpg?sha256='+digest)[0]==409
    for query in ('','not-a-hash','a'*64+'&sha256='+'b'*64):
        assert request(dashboard,'/frame.jpg?sha256='+query)[0]==400
    assert request(dashboard,'/frame.jpg')[0]==200  # existing immutable recording route


def test_dashboard_uses_separate_visual_batch_with_legacy_recording_fallback(dashboard):
    token=session(dashboard)
    critical=read_json(dashboard.directory/'latest_proposals.json')
    visual=dict(critical,preview_frame={'key':'bound-image'},preview_current=False)
    atomic_json(dashboard.directory/'preview_proposals.json',visual)
    assert request(dashboard,'/api/state',token=token)[1]['proposals']==visual
    assert read_json(dashboard.directory/'latest_proposals.json')==critical
    (dashboard.directory/'preview_proposals.json').unlink()
    assert request(dashboard,'/api/state',token=token)[1]['proposals']==critical


def test_browser_only_draws_geometry_on_its_loaded_bound_camera_frame(dashboard):
    if not Path('/opt/google/chrome/chrome').is_file():pytest.skip('Chrome unavailable')
    from PIL import Image
    image=dashboard.directory/'frame.jpg';Image.new('RGB',(100,50),(70,80,90)).save(image)
    binding=dict(schema='punktnull.preview_frame.v1',key='a'*64,frame_id=1,
        capture_timestamp=time.monotonic(),clock_id=host_clock_id(),calibration_id='test',
        width=100,height=50,jpeg_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),projection_sha256='f'*64)
    atomic_json(dashboard.directory/'ui_state.json',dict(mode='live',camera_current=True,
        display_frame=binding,selected_target_id='cup',
        objects=[dict(target_id='cup',label='Testobjekt',bbox_px=[10,10,30,30])]))
    atomic_json(dashboard.directory/'latest_proposals.json',dict(target_id='cup',valid=False,
        preview_current=True,preview_frame=binding,candidates=[dict(candidate_id='grasp-1',
            geometry_valid=True,ik_valid=True,collision_valid=False,model_score=.9,
            preview_uv=[[20,20],[30,20],[20,10],[25,25]],
            preview_hand_polygons=[dict(link='fixture',uv=[[40,25],[60,25],[50,45]])])]))
    code=r'''
const{chromium}=require('playwright'),assert=require('assert');
(async()=>{const browser=await chromium.launch({executablePath:'/opt/google/chrome/chrome',headless:true,args:['--disable-gpu','--no-sandbox']});try{
 const page=await browser.newPage({viewport:{width:1000,height:800}}),errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(process.argv[1]);await page.waitForFunction(()=>loadedFrameKey==='a'.repeat(64));
 await page.locator('#candidates button').first().click();await page.locator('#ghost-legend').waitFor({state:'visible'});
 const ratio=await page.locator('.camera').evaluate(el=>el.getBoundingClientRect().width/el.getBoundingClientRect().height);assert(Math.abs(ratio-2)<.01);
 const hasGhost=()=>page.evaluate(()=>document.getElementById('overlay').getContext('2d').getImageData(50,30,1,1).data[3]>0);
 assert(await hasGhost());let fixture=await page.evaluate(()=>state);
 await page.route('**/api/state',route=>route.fulfill({json:fixture}));
 fixture={...fixture,perception:{...fixture.perception,display_frame:{...fixture.perception.display_frame,key:'b'.repeat(64),frame_id:2}}};
 const immediatelyCleared=await page.evaluate(s=>{render(s);return document.getElementById('overlay').getContext('2d').getImageData(50,30,1,1).data[3]===0},fixture);
 assert(immediatelyCleared);await page.waitForFunction(()=>loadedFrameKey==='b'.repeat(64));
 assert.strictEqual(await hasGhost(),false); // proposal belongs to the old observation even with identical JPEG
 fixture={...fixture,proposals:{...fixture.proposals,preview_frame:fixture.perception.display_frame}};
 await page.evaluate(s=>render(s),fixture);assert(await hasGhost());
 fixture={...fixture,perception:{...fixture.perception,camera_current:false}};
 await page.evaluate(s=>render(s),fixture);assert.strictEqual(await hasGhost(),false);
 assert.match(await page.locator('#notice').textContent(),/Letztes Kamerabild/);
 fixture={...fixture,perception:{...fixture.perception,camera_current:true,display_frame:{...fixture.perception.display_frame,key:'c'.repeat(64),jpeg_sha256:'0'.repeat(64)}}};
 const changed=page.waitForResponse(r=>r.url().includes('sha256='+'0'.repeat(64)));
 await page.evaluate(s=>render(s),fixture);assert.strictEqual((await changed).status(),409);
 await page.waitForFunction(()=>loadedFrameKey===null);assert.strictEqual(await hasGhost(),false);
 assert.deepStrictEqual(errors,[]);console.log(JSON.stringify({bound_frame_browser_passed:true}));
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exit(1)});
'''
    result=subprocess.run([str(NODE),'-e',code,dashboard.url],capture_output=True,text=True,timeout=40,
        env=dict(os.environ,NODE_PATH='/home/loya2/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'))
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['bound_frame_browser_passed']


def test_real_browser_selection_confirmation_deadman_and_explicit_release(dashboard, tmp_path):
    """Real Chrome UI test against a simulated status publisher, no robot."""
    if not Path('/opt/google/chrome/chrome').is_file():
        pytest.skip('Local Chrome executable unavailable')
    atomic_json(dashboard.directory / 'ui_state.json', {
        'objects':[{'target_id':'cup','label':'Cup','bbox_px':[50,60,180,220]}],
        'selected_target_id':'cup', 'status':'ready', 'width':640, 'height':480})
    rejected = [{'candidate_id':f'raw-{i}', 'model_score':1., 'geometry_valid':True,
        'ik_valid':False, 'collision_valid':False} for i in range(63)]
    atomic_json(dashboard.directory / 'latest_proposals.json', {
        'target_id':'cup', 'valid':True, 'candidates':rejected+[{
            'candidate_id':'grasp-1', 'base_ee':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            'model_score':.01, 'observed_hand_collision':{'pregrasp_collision_free':True},
            'geometry_valid':True,'ik_valid':True,'collision_valid':True,
            'preview_uv':[[110,140],[135,140],[110,115],[120,130]],
            'preview_hand_polygons':[{'link':'right_hand_palm_link','uv':[[95,160],[135,160],[115,195]]}]}]})
    stop = threading.Event()
    def simulated_status():
        confirmed = False
        while not stop.is_set():
            control = read_json(dashboard.directory / 'copilot_control.json', {})
            if control.get('confirm'): confirmed = True
            if control.get('release'): confirmed = False
            atomic_json(dashboard.directory / 'xr_status.json', {
                'mode':'assist', 'timestamp':time.monotonic(), 'input_hold_required':False,
                'reason':'confirmed' if confirmed else 'manual',
                'snapshot':{'confirmed_candidate_id':'grasp-1' if confirmed else None}})
            stop.wait(.02)
    publisher = threading.Thread(target=simulated_status, daemon=True)
    publisher.start()
    code = r'''
const {chromium}=require('playwright'),fs=require('fs'),assert=require('assert');
(async()=>{
 const browser=await chromium.launch({executablePath:'/opt/google/chrome/chrome',headless:true,args:['--disable-gpu','--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1200,height:850}});const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(process.argv[1],{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>document.querySelectorAll('#candidates button').length===8);
  assert.strictEqual(await page.locator('#candidates button').first().getAttribute('data-candidate-id'),'grasp-1');
  assert.match(await page.locator('#candidate-count').textContent(),/8 von 64 angezeigt · 56 weitere · 63 ohne/);
  const reviewMode=await page.evaluate(()=>{render({...state,perception:{...state.perception,mode:'recorded',status:'camera_stream_expired'},xr:{}});return [document.getElementById('mode').textContent,document.getElementById('status').textContent]});
  assert.deepStrictEqual(reviewMode,['Aufzeichnung · Vorschau','Kamerabilder fehlen oder sind zu alt. Bitte erneut auswählen.']);
  await page.getByRole('button',{name:'Cup',exact:true}).click();
  const selectionResponse=page.waitForResponse(r=>r.url().endsWith('/api/control')&&r.request().postDataJSON()?.action==='select');
  await page.getByText('Griff 1',{exact:true}).click();
  await selectionResponse;
  assert.strictEqual(JSON.parse(fs.readFileSync(process.argv[2],'utf8')).select_candidate_id,'grasp-1');
  await page.locator('#ghost-legend').waitFor({state:'visible'});
  assert.strictEqual(await page.evaluate(()=>document.getElementById('overlay').getContext('2d').getImageData(115,170,1,1).data[3]>0),true);
  await page.waitForFunction(()=>!document.getElementById('confirm').disabled);
  await page.getByRole('button',{name:'Griff bestätigen',exact:true}).click();
  await page.waitForFunction(()=>!document.getElementById('hold').disabled);
  assert.strictEqual(await page.locator('#candidates canvas').count(),8);
  const hold=page.locator('#hold');await hold.hover();await page.mouse.down();
  await page.waitForTimeout(180);
  assert.strictEqual(JSON.parse(fs.readFileSync(process.argv[2],'utf8')).assist_held,true);
  await page.mouse.up();await page.waitForTimeout(150);
  const idle=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));assert.strictEqual(idle.assist_held,false);assert.strictEqual(idle.release,false);
  await page.screenshot({path:process.argv[3],fullPage:true});
  await page.getByRole('button',{name:'Assistenz lösen',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('hold').disabled);
  assert.strictEqual(JSON.parse(fs.readFileSync(process.argv[2],'utf8')).release,true);
  assert.deepStrictEqual(errors,[]);console.log(JSON.stringify({browser_ui_passed:true,no_robot:true}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
'''
    try:
        result = subprocess.run([str(NODE), '-e', code, dashboard.url,
            str(dashboard.directory / 'copilot_control.json'), '/tmp/punktnull_copilot_dashboard_ui.png'],
            env=dict(os.environ, NODE_PATH='/home/loya2/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'),
            capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout.strip())['browser_ui_passed']
    finally:
        stop.set(); publisher.join(timeout=2)


def sim_status(**changes):
    result=dict(schema='punktnull.sim_runtime.v1',mode='isolated_PhysX',
        timestamp=time.monotonic(),clock_id=host_clock_id(),candidates=[{'candidate_id':'sim-grasp'}])
    result.update(changes);return result


def test_preview_browser_full_sim_counts_partial_observed_ranking_and_sim_link(dashboard, tmp_path):
    """Contradictory preview scores/flags never imply permission or 512 observed checks."""
    if not Path('/opt/google/chrome/chrome').is_file():
        pytest.skip('Local Chrome executable unavailable')
    sim_control_before = read_json(dashboard.directory/'sim_control.json')
    pose = [[1,0,0,.3],[0,1,0,-.2],[0,0,1,.1],[0,0,0,1]]
    candidates = []
    for index in range(8):
        candidates.append(dict(candidate_id=f'original-id-{index}', base_ee=pose,
            model_score=.9-index*.05, geometry_valid=True, ik_valid=True,
            collision_valid=False, recorded_sim_endpoint_clear=index!=0,
            observed_hand_collision=dict(hand_collision_free=index==0,
                pregrasp_collision_free=index==0, status='observed_clear' if index==0 else 'observed_approach_collision')))
    batch = dict(target_id='cup', valid=False,
        preview_ranking_policy='complete_static_sim_orientation', candidates=candidates,
        candidate_counts=dict(total=512, ik_valid=103, recorded_sim_endpoint_clear=16,
            observed_hand_checked=8, ik_and_pregrasp_clear=1))
    atomic_json(dashboard.directory/'latest_proposals.json', batch)
    atomic_json(dashboard.directory/'ui_state.json', dict(mode='live', status='preview_ready',
        selected_target_id='cup', objects=[dict(target_id='cup',label='Zylinder',bbox_px=[107,93,200,253])]))
    stopped = threading.Event()
    def simulated_runtime():
        while not stopped.is_set():
            atomic_json(dashboard.directory/'sim_runtime.json', sim_status(live_ai=True))
            stopped.wait(.02)
    worker = threading.Thread(target=simulated_runtime, daemon=True)
    worker.start()
    code = r'''
const {chromium}=require('playwright'),assert=require('assert'),fs=require('fs');
(async()=>{
 const browser=await chromium.launch({executablePath:'/opt/google/chrome/chrome',headless:true,args:['--disable-gpu','--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1200,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(process.argv[1],{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>document.querySelectorAll('#candidates button').length===8);
  const ids=await page.locator('#candidates button').evaluateAll(els=>els.map(e=>e.dataset.candidateId));
  assert.deepStrictEqual(ids,['original-id-1','original-id-2','original-id-3','original-id-4','original-id-5','original-id-6','original-id-7','original-id-0']);
  const summary=await page.locator('#candidate-count').textContent();
  assert.match(summary,/8 von 512 angezeigt/);assert.match(summary,/103 kinematisch erreichbar/);
  assert.match(summary,/16 statische Sim-Endposen frei/);
  assert.match(summary,/nur 8 von 512 geprüft, davon 1 erreichbar und frei/);
  assert.match(summary,/Keine Bewegungs- oder Anfahrfreigabe/);assert(!summary.includes('511 ohne'));
  const details=await page.locator('#candidates button').first().getAttribute('title');
  assert.match(details,/Gerade Annäherung an beobachteten Punkten: nicht frei/);
  assert.match(details,/pregrasp_collision_free=false/);assert.match(details,/observed_approach_collision/);
  assert.match(await page.locator('#candidates button').first().textContent(),/Sim-Endpose frei · nur Vorschau/);
  await page.locator('#sim-link').waitFor({state:'visible'});
  assert.strictEqual(await page.locator('#sim-link').getAttribute('href'),'/sim');
  const linkCases=await page.evaluate(()=>{
   const initial=state,run=initial.sim;
   const cases=[null,{...run,timestamp:initial.server_monotonic-1},
    {...run,timestamp:initial.server_monotonic+1},{...run,clock_id:'other'},
    {...run,mode:'hardware'},{...run,schema:'other'},{...run,timestamp:null}];
   const hidden=cases.map(sim=>{render({...initial,sim});return document.getElementById('sim-workflow').hidden});
   render(initial);return hidden;
  });
  assert(linkCases.every(Boolean));
  const malformed=await page.evaluate(()=>{
   const batch={...state.proposals,candidate_counts:{total:512,ik_valid:513,
    recorded_sim_endpoint_clear:-1,observed_hand_checked:8,ik_and_pregrasp_clear:9}};
   return candidateSummary(batch,batch.candidates,8);
  });
  assert.match(malformed,/Erreichbarkeit: Zahl unbekannt/);assert.match(malformed,/Sim-Endposen: Zahl unbekannt/);
  assert(!malformed.includes('davon 9'));assert(!malformed.includes('504 ohne'));
  const controlBefore=fs.readFileSync(process.argv[2],'utf8'),commands=[];page.on('request',r=>{if(r.method()==='POST')commands.push(r.url())});
  await page.locator('#candidates button').first().click();
  assert.strictEqual(await page.locator('#candidates button').first().getAttribute('aria-pressed'),'true');
  assert.strictEqual(fs.readFileSync(process.argv[2],'utf8'),controlBefore);assert.deepStrictEqual(commands,[]);
  await page.waitForTimeout(200);
  assert.strictEqual(await page.locator('#confirm').isDisabled(),true);
  assert.strictEqual(await page.locator('#hold').isDisabled(),true);
  assert.strictEqual(await page.evaluate(()=>state.proposals.valid),false);
  assert.strictEqual(await page.evaluate(()=>state.proposals.candidates.every(c=>c.collision_valid===false)),true);
  await page.setViewportSize({width:390,height:844});
  assert.strictEqual(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.deepStrictEqual(errors,[]);console.log(JSON.stringify({full_sim_partial_observed_ui_passed:true,no_robot:true}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
'''
    try:
        result = subprocess.run([str(NODE), '-e', code, dashboard.url, str(dashboard.directory/'copilot_control.json')],
            env=dict(os.environ, NODE_PATH='/home/loya2/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'),
            capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout.strip())['full_sim_partial_observed_ui_passed']
        assert read_json(dashboard.directory/'sim_control.json') == sim_control_before
    finally:
        stopped.set(); worker.join(timeout=2)


@pytest.mark.parametrize('changes',[{'mode':'hardware'},{'schema':'unknown'},
    {'clock_id':'wrong'},{'timestamp':None},{'timestamp':0.},{'timestamp':float('inf')},{'timestamp':10**500},
    {'scripted_operator':True},{'reason':'runtime_stopped'}])
def test_sim_route_rejects_missing_or_wrong_runtime_without_changing_mailbox(dashboard,changes):
    token=session(dashboard);before=read_json(dashboard.directory/'sim_control.json')
    runtime=sim_status(**changes)
    # Malformed JSON/nonfinite data is also an untrusted runtime state.
    (dashboard.directory/'sim_runtime.json').write_text(json.dumps(runtime))
    assert request(dashboard,'/api/sim/control',method='POST',token=token,payload={'action':'clutch'})[0]==409
    assert read_json(dashboard.directory/'sim_control.json')==before


@pytest.mark.parametrize('changes',[{'scripted_operator':True},{'scripted_test':True},
    {'reason':'runtime_stopped'},{'timestamp':0.},{'clock_id':'other'}])
def test_main_object_choice_cannot_change_scripted_stopped_or_stale_sim(dashboard,changes):
    token=session(dashboard)
    atomic_json(dashboard.directory/'sim_runtime.json',sim_status(**changes))
    paths=[dashboard.directory/name for name in ('sim_control.json','copilot_control.json','target_selection.json')]
    before=[path.read_bytes() if path.exists() else None for path in paths]
    assert request(dashboard,'/api/target',method='POST',token=token,payload={'target_id':'cup'})[0]==409
    assert [path.read_bytes() if path.exists() else None for path in paths]==before


def test_sim_owner_cannot_be_overridden_through_main_object_choice(dashboard):
    first,second=session(dashboard),session(dashboard)
    atomic_json(dashboard.directory/'sim_runtime.json',sim_status())
    assert request(dashboard,'/api/sim/control',method='POST',token=first,payload={'action':'clutch'})[0]==200
    before=read_json(dashboard.directory/'sim_control.json')
    assert request(dashboard,'/api/target',method='POST',token=second,payload={'target_id':'cup'})[0]==409
    assert read_json(dashboard.directory/'sim_control.json')==before
    assert request(dashboard,'/api/target',method='POST',token=first,sequence=2,payload={'target_id':'cup'})[0]==200
    assert read_json(dashboard.directory/'sim_control.json')==before


def test_legacy_control_mailbox_is_not_a_sim_selection_route(dashboard):
    token=session(dashboard);before=read_json(dashboard.directory/'copilot_control.json')
    atomic_json(dashboard.directory/'sim_runtime.json',sim_status())
    assert request(dashboard,'/api/control',method='POST',token=token,
        payload={'action':'select','candidate_id':'grasp-1'})[0]==409
    assert read_json(dashboard.directory/'copilot_control.json')==before


def test_browser_actual_sim_target_is_distinct_from_local_preview_and_never_sends_control(dashboard,tmp_path):
    if not Path('/opt/google/chrome/chrome').is_file():pytest.skip('Chrome unavailable')
    from PIL import Image
    image=dashboard.directory/'frame.jpg';Image.new('RGB',(100,60),(50,60,70)).save(image)
    binding=dict(key='a'*64,width=100,height=60,jpeg_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    pose=[[1,0,0,.3],[0,1,0,-.2],[0,0,1,.1],[0,0,0,1]]
    def candidate(identity,u,score):
        return dict(candidate_id=identity,base_ee=pose,source='fixture_model',model_score=score,
            preview_uv=[[u,15],[u+5,15],[u,10],[u+2,20]],
            preview_hand_polygons=[dict(link='fixture',uv=[[u-8,30],[u+8,30],[u,48]])])
    candidates=[candidate('preview-first',25,.9),candidate('actual-confirmed',65,.8)]
    fixture=dict(clock_id='clock',server_monotonic=10.1,ownership=dict(available=True,owned_by_this_client=False),
        sim_ownership=dict(available=False,owned_by_this_client=False,owner_client_id='xr:other'),
        perception=dict(mode='live',camera_current=True,display_frame=binding,selected_target_id='cup',
            objects=[dict(target_id='cup',label='Cup',bbox_px=[1,1,8,8])]),
        proposals=dict(target_id='cup',clock_id='clock',calibration_id='cal',model_source_capture_timestamp=5.,
            preview_current=True,preview_frame=binding,valid=False,candidates=candidates),
        sim=dict(schema='punktnull.sim_runtime.v1',mode='isolated_PhysX',timestamp=10.,clock_id='clock',
            candidates=candidates,reason='confirmed',assistance_applied=False,
            snapshot=dict(confirmed=True,confirmed_candidate_id='actual-confirmed',selected_candidate_id='actual-confirmed',
                target_id='cup',model_source_capture_timestamp=5.,model_source_clock_id='clock',calibration_id='cal')))
    source=tmp_path/'ui_fixture.json';source.write_text(json.dumps(fixture))
    code=r'''
const{chromium}=require('playwright'),assert=require('assert'),fs=require('fs');
(async()=>{const browser=await chromium.launch({executablePath:'/opt/google/chrome/chrome',headless:true,args:['--disable-gpu','--no-sandbox']});try{
 const page=await browser.newPage({viewport:{width:1250,height:1000}}),errors=[],posts=[];
 let fixture=JSON.parse(fs.readFileSync(process.argv[2]));const original=structuredClone(fixture);
 page.on('pageerror',e=>errors.push(String(e)));
 await page.route('**/*',route=>{const request=route.request();if(request.method()==='POST'){posts.push(request.url());return route.abort();}if(request.url().endsWith('/api/state'))return route.fulfill({json:fixture});return route.continue();});
 await page.goto(process.argv[1]);await page.waitForFunction(()=>loadedFrameKey==='a'.repeat(64));
 const marks=()=>page.locator('[data-sim-target="true"]').evaluateAll(els=>els.map(e=>e.dataset.candidateId));
 const blue=()=>page.evaluate(()=>{const pixel=document.getElementById('overlay').getContext('2d').getImageData(65,35,1,1).data;return pixel[3]>0&&pixel[2]>pixel[0]});
 assert.deepStrictEqual(await marks(),['actual-confirmed']);assert(await blue());
 assert.match(await page.locator('#sim-target-status').textContent(),/Bestätigtes Orientierungsziel · Griff 2/);
 assert(await page.locator('#xr-controls').isHidden());assert(await page.locator('#objects button').isDisabled());
 await page.locator('#candidates button').first().click();assert.deepStrictEqual(posts,[]);
 assert.deepStrictEqual(await marks(),['actual-confirmed']);assert(await blue());
 assert.match(await page.locator('#ghost-legend').textContent(),/Blau: Sim-Ziel.*Gelb: lokale Ansicht/);
 for(const change of ['pose','source','clock','stamp','target']){
  fixture=structuredClone(original);
  if(change==='pose')fixture.proposals.candidates[1].base_ee[0][3]+=.01;
  if(change==='source')fixture.proposals.candidates[1].source='changed';
  if(change==='clock')fixture.proposals.clock_id='other';
  if(change==='stamp')fixture.proposals.model_source_capture_timestamp=6.;
  if(change==='target')fixture.proposals.target_id='other';
  await page.evaluate(s=>render(s),fixture);assert.deepStrictEqual(await marks(),[]);assert(!(await blue()));
  assert.match(await page.locator('#sim-target-detail').textContent(),/nicht eindeutig/);
 }
 for(const change of ['stale','future','clock','stopped']){
  fixture=structuredClone(original);
  if(change==='stale')fixture.server_monotonic=10.6;
  if(change==='future')fixture.sim.timestamp=11.;
  if(change==='clock')fixture.sim.clock_id='other';
  if(change==='stopped')fixture.sim.reason='runtime_stopped';
  await page.evaluate(s=>render(s),fixture);assert.deepStrictEqual(await marks(),[]);assert(!(await blue()));
  assert.match(await page.locator('#sim-target-heading').textContent(),/nicht aktuell/);
 }
 fixture=structuredClone(original);fixture.sim.snapshot={};await page.evaluate(s=>render(s),fixture);
 assert.match(await page.locator('#sim-target-status').textContent(),/Kein Griff ausgewählt/);assert.deepStrictEqual(await marks(),[]);
 fixture=structuredClone(original);fixture.sim.snapshot.confirmed=false;fixture.sim.snapshot.confirmed_candidate_id=null;
 await page.evaluate(s=>render(s),fixture);assert.match(await page.locator('#sim-target-status').textContent(),/In der Simulation ausgewählt/);
 fixture=structuredClone(original);await page.reload();await page.waitForFunction(()=>loadedFrameKey==='a'.repeat(64));
 assert.deepStrictEqual(await marks(),['actual-confirmed']);assert.strictEqual(await page.evaluate(()=>choice),null);
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(100);
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 assert.deepStrictEqual(posts,[]);assert.deepStrictEqual(errors,[]);console.log('ok');
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exit(1)});
'''
    result=subprocess.run([str(NODE),'-e',code,dashboard.url,str(source)],capture_output=True,text=True,timeout=35,
        env=dict(os.environ,NODE_PATH='/home/loya2/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'))
    assert result.returncode==0,result.stdout+result.stderr


def test_sim_clutch_is_separate_owned_and_can_disarm_after_runtime_expires(dashboard):
    first,second=session(dashboard),session(dashboard)
    preview_before=read_json(dashboard.directory/'copilot_control.json')
    atomic_json(dashboard.directory/'sim_runtime.json',sim_status())
    assert request(dashboard,'/api/sim/control',method='POST',token=first,payload={'action':'clutch'})[0]==200
    packet=read_json(dashboard.directory/'sim_control.json')
    assert packet['clutch'] and not packet['assist_held']
    assert read_json(dashboard.directory/'copilot_control.json')==preview_before
    assert request(dashboard,'/api/sim/control',method='POST',token=second,payload={'action':'release'})[0]==409
    assert request(dashboard,'/api/sim/control',method='POST',token=first,sequence=2,
        payload={'action':'select','candidate_id':'grasp-1'})[0]==400  # live-preview ID is not a sim candidate
    atomic_json(dashboard.directory/'sim_runtime.json',sim_status(timestamp=0.))
    assert request(dashboard,'/api/sim/control',method='POST',token=first,sequence=3,payload={'action':'release'})[0]==200
    assert read_json(dashboard.directory/'sim_control.json')['release']


@pytest.mark.parametrize('live_ai',[False,True])
def test_sim_browser_clutch_ack_manual_heartbeat_hold_and_disarm(dashboard,live_ai):
    """UI with simulated status only; no physics or arm command is executed."""
    if not Path('/opt/google/chrome/chrome').is_file():pytest.skip('Chrome unavailable')
    stopped=threading.Event()
    def simulated_consumer():
        clutched=False;selected=confirmed=None;last=0
        while not stopped.is_set():
            packet=read_json(dashboard.directory/'sim_control.json',{})
            if packet.get('revision',0)>last:
                last=packet['revision']
                if packet.get('release'):clutched=False;selected=confirmed=None
                elif packet.get('clutch'):clutched=True;confirmed=None
                elif packet.get('select_candidate_id'):
                    selected=packet['select_candidate_id']
                    if live_ai:clutched=False;confirmed=None
                elif packet.get('confirm') and clutched:confirmed=selected
            if time.monotonic()-packet.get('capture_timestamp',0)>.25:clutched=False;confirmed=None
            atomic_json(dashboard.directory/'sim_runtime.json',sim_status(input_hold_required=False,live_ai=live_ai,
                snapshot=dict(clutched=clutched,clutch_required=not clutched,clutch_available=True,
                    selected_candidate_id=selected,confirmed_candidate_id=confirmed,
                    input_hold_required=False,last_ui_revision=last)))
            stopped.wait(.02)
    worker=threading.Thread(target=simulated_consumer,daemon=True);worker.start()
    code=r'''
const {chromium}=require('playwright'),fs=require('fs'),assert=require('assert');
(async()=>{
 const liveAI=process.argv[3]==='true';
 const browser=await chromium.launch({executablePath:'/opt/google/chrome/chrome',headless:true,args:['--disable-gpu','--no-sandbox']});
 try{
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(process.argv[1]+'/sim');await page.bringToFront();
  await page.waitForFunction(()=>!document.getElementById('clutch').disabled);
  await page.locator('#candidates button').first().click();
  await page.waitForFunction(()=>pendingEdge===null&&waitingForClutch);
  let prior=JSON.parse(fs.readFileSync(process.argv[2],'utf8')).revision;
  await page.waitForTimeout(400);
  let neutral=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
  assert(neutral.revision>prior+3&&!neutral.clutch&&!neutral.assist_held&&!neutral.confirm&&!neutral.release);
  assert.strictEqual(await page.evaluate(()=>state.sim.snapshot.clutched),false);
  assert.strictEqual(await page.locator('#hold').isDisabled(),true);
  assert.strictEqual((await page.locator('#workflow').textContent()).includes('Live-KI'),liveAI);
  assert.strictEqual(await page.locator('#orientation-only').isVisible(),liveAI);
  await page.locator('#clutch').click();
  await page.waitForFunction(()=>state.sim.snapshot.clutched&&pendingEdge===null);
  const revision=JSON.parse(fs.readFileSync(process.argv[2],'utf8')).revision;
  await page.waitForTimeout(400);
  assert.strictEqual(await page.evaluate(()=>state.sim.snapshot.clutched),true);
  let packet=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));assert(packet.revision>revision+3);assert(!packet.assist_held&&!packet.release&&!packet.clutch);
  await page.locator('#candidates button').first().click();
  if(liveAI){
   await page.waitForFunction(()=>pendingEdge===null&&!state.sim.snapshot.clutched&&waitingForClutch);
   prior=JSON.parse(fs.readFileSync(process.argv[2],'utf8')).revision;
   await page.waitForTimeout(400);
   neutral=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
   assert(neutral.revision>prior+3&&!neutral.clutch&&!neutral.assist_held&&!neutral.confirm&&!neutral.release);
   assert.strictEqual(await page.locator('#hold').isDisabled(),true);
   await page.locator('#clutch').click();
  }else{
   await page.waitForFunction(()=>pendingEdge===null&&state.sim.snapshot.clutched&&!waitingForClutch);
  }
  await page.waitForFunction(()=>!document.getElementById('confirm').disabled);
  await page.locator('#confirm').click();await page.waitForFunction(()=>!document.getElementById('hold').disabled);
  await page.locator('#hold').hover();await page.mouse.down();await page.waitForTimeout(170);
  assert.strictEqual(JSON.parse(fs.readFileSync(process.argv[2],'utf8')).assist_held,true);
  await page.mouse.up();await page.waitForTimeout(150);
  packet=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));assert(!packet.assist_held&&!packet.release);
  await page.evaluate(()=>window.dispatchEvent(new Event('blur')));
  await page.waitForFunction(()=>!state.sim.snapshot.clutched);
  assert.strictEqual(JSON.parse(fs.readFileSync(process.argv[2],'utf8')).release,true);
  assert.deepStrictEqual(errors,[]);console.log('sim_ui_passed');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
'''
    try:
        result=subprocess.run([str(NODE),'-e',code,dashboard.url,str(dashboard.directory/'sim_control.json'),str(live_ai).lower()],
            env=dict(os.environ,NODE_PATH='/home/loya2/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'),
            capture_output=True,text=True,timeout=45)
        assert result.returncode==0,result.stdout+result.stderr
        assert 'sim_ui_passed' in result.stdout
    finally:stopped.set();worker.join(timeout=2)
