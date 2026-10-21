"""CPU-only tests. Original display method is compiled without its constructor."""
import ast
import asyncio
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from teleop_assist.copilot.xr_camera import (BoundSimCamera, binding_key, install_on_staged_televuer,
                              ORIGINAL_TELEVUER_SHA256, native_background, publish_operator_frame)

CLOCK = 'host_monotonic:offline_camera_test'
ORIGINAL = Path('/home/loya2/xr_teleoperate/teleop/televuer/src/televuer/televuer.py')


class Clock:
    def __init__(self): self.value=100.1
    def __call__(self): return self.value


def write_frame(directory, *, frame_id=4, stamp=100., color=(190,30,10)):
    buffer=io.BytesIO();Image.new('RGB',(160,120),color).save(buffer,format='JPEG',quality=85)
    jpeg=buffer.getvalue();(directory/'frame.jpg').write_bytes(jpeg)
    binding=dict(schema='punktnull.preview_frame.v1',frame_id=frame_id,capture_timestamp=stamp,
        clock_id=CLOCK,calibration_id='sim_camera_v1',width=160,height=120,
        jpeg_sha256=hashlib.sha256(jpeg).hexdigest(),projection_sha256='a'*64)
    binding['key']=binding_key(binding)
    state=dict(mode='KI-Vorschau · keine Bewegungsfreigabe',frame_id=frame_id,camera_current=True,display_frame=binding)
    (directory/'ui_state.json').write_text(json.dumps(state))
    return state,np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB')).copy()


def fixture(tmp_path):
    state,pixels=write_frame(tmp_path);clock=Clock()
    return BoundSimCamera(tmp_path,clock_id=CLOCK,width=160,height=120,clock=clock),clock,state,pixels


def test_current_bound_jpeg_preserves_rgb_and_original_capture(tmp_path):
    feed,clock,state,pixels=fixture(tmp_path);sample=feed.sample()
    assert sample.evidence['image_visible'] is True
    assert sample.evidence['source_capture_timestamp']==100.
    assert sample.evidence['source_frame_id']==4
    assert sample.evidence['movement_authorized'] is False
    assert sample.evidence['headset_delivery_verified'] is False
    np.testing.assert_array_equal(sample.rgb[-120:],pixels)
    assert sample.rgb.flags.writeable is False


def test_operator_publication_keeps_head_image_separate_from_wrist_perception(tmp_path):
    perception=tmp_path/'perception'; perception.mkdir()
    write_frame(perception,color=(0,255,0))
    original={p.name:p.read_bytes() for p in perception.iterdir()}
    display=tmp_path/'operator_camera'
    frame=dict(rgb=np.full((120,160,3),(190,30,10),dtype=np.uint8),
        T_world_camera=np.eye(4),intrinsics=np.eye(3),frame_id=22,timestamp=100.,
        clock_id=CLOCK,calibration_id='unitree_usd_head_measured_link_mount_sync_v3',camera_source='head')
    binding=publish_operator_frame(display,frame)
    assert binding['capture_timestamp']==100. and binding['calibration_id']==frame['calibration_id']
    assert original=={p.name:p.read_bytes() for p in perception.iterdir()}
    clock=Clock(); feed=BoundSimCamera(display,clock_id=CLOCK,width=160,height=120,clock=clock)
    sample=feed.sample()
    assert sample.evidence['image_visible'] and not sample.evidence['movement_authorized']
    assert sample.rgb[-60,80,0]>150 and sample.rgb[-60,80,1]<60
    clock.value=100.251
    assert not feed.sample().evidence['image_visible']


def test_operator_header_reports_start_and_never_reuses_stale_active_status(tmp_path):
    display=tmp_path/'operator_camera';display.mkdir()
    write_frame(display);clock=Clock()
    feed=BoundSimCamera(display,clock_id=CLOCK,width=160,height=120,clock=clock)
    control=dict(mode='isolated_PhysX',timestamp=100.1,clock_id=CLOCK,
                 snapshot=dict(clutched=True,teleop_active=False))
    path=tmp_path/'sim_runtime.json';path.write_text(json.dumps(control))
    assert feed.sample().evidence['header_lines'][0].startswith('KALIBRIERT')
    control['snapshot']['teleop_active']=True;path.write_text(json.dumps(control))
    assert feed.sample().evidence['header_lines'][0].startswith('TELEOP AKTIV')
    control['timestamp']=99.;path.write_text(json.dumps(control))
    assert feed.sample().evidence['header_lines'][0]=='STEUERSTATUS UNBEKANNT'


def test_repeated_frame_ages_without_renewal_then_disappears(tmp_path):
    feed,clock,state,pixels=fixture(tmp_path);first=feed.sample();clock.value=100.24;second=feed.sample()
    assert second.evidence['source_age_at_preparation_s']>first.evidence['source_age_at_preparation_s']
    assert second.evidence['source_capture_timestamp']==first.evidence['source_capture_timestamp']==100.
    clock.value=100.251;stale=feed.sample()
    assert not stale.evidence['image_visible'] and stale.evidence['reason']=='camera_frame_stale_or_future'
    assert stale.evidence['source_capture_timestamp']==100.
    assert not np.array_equal(stale.rgb[-120:],pixels)


@pytest.mark.parametrize('fault,reason',[
    ('hash','camera_jpeg_hash_mismatch'),('binding','display_binding_hash_mismatch'),
    ('clock','camera_clock_mismatch'),('future','camera_frame_stale_or_future'),
    ('producer_stale','producer_does_not_claim_current_camera'),('dimensions','camera_dimensions_changed')])
def test_invalid_or_unbound_frames_are_hidden(tmp_path,fault,reason):
    feed,clock,state,pixels=fixture(tmp_path)
    if fault=='hash':(tmp_path/'frame.jpg').write_bytes(b'not the bound JPEG')
    elif fault=='binding':state['display_frame']['key']='0'*64
    elif fault=='producer_stale':state['camera_current']=False
    else:
        field,value={'clock':('clock_id','host_monotonic:other'), 'future':('capture_timestamp',101.),
                     'dimensions':('width',161)}[fault]
        state['display_frame'][field]=value;state['display_frame']['key']=binding_key(state['display_frame'])
    (tmp_path/'ui_state.json').write_text(json.dumps(state))
    sample=feed.sample()
    assert sample.evidence['image_visible'] is False and sample.evidence['reason']==reason
    assert not np.array_equal(sample.rgb[-120:],pixels)


def test_disappearing_source_never_reuses_camera_pixels(tmp_path):
    feed,clock,state,pixels=fixture(tmp_path);assert feed.sample().evidence['image_visible']
    (tmp_path/'frame.jpg').unlink();sample=feed.sample()
    assert not sample.evidence['image_visible'] and sample.evidence['reason']=='FileNotFoundError'
    assert not np.array_equal(sample.rgb[-120:],pixels)


def test_atomic_pair_race_is_rejected(tmp_path,monkeypatch):
    from teleop_assist.copilot import xr_camera as camera_candidate
    feed,clock,state,pixels=fixture(tmp_path);original=camera_candidate.read_state;reads=0
    def changing(directory):
        nonlocal reads
        value=original(directory);reads+=1
        if reads%2==0:value['display_frame']['frame_id']+=1
        return value
    monkeypatch.setattr(camera_candidate,'read_state',changing)
    sample=feed.sample()
    assert sample.evidence['reason']=='camera_binding_changed_during_read' and not sample.evidence['image_visible']


def test_single_publication_race_retries_current_pixels_without_black_frame(tmp_path,monkeypatch):
    from teleop_assist.copilot import xr_camera
    feed,clock,state,pixels=fixture(tmp_path);original=xr_camera.read_state;reads=0
    def changing(directory):
        nonlocal reads
        value=original(directory);reads+=1
        if reads==2:value['display_frame']['frame_id']+=1
        return value
    monkeypatch.setattr(xr_camera,'read_state',changing)
    sample=feed.sample()
    assert reads==4 and sample.evidence['image_visible']
    assert sample.evidence['source_capture_timestamp']==100.
    np.testing.assert_array_equal(sample.rgb[-120:],pixels)


def test_decode_time_counts_and_cannot_retimestamp_source(tmp_path):
    feed,clock,state,pixels=fixture(tmp_path)
    values=iter([100.1,100.26,100.26]);feed.clock=lambda:next(values)
    sample=feed.sample()
    assert not sample.evidence['image_visible']
    assert sample.evidence['reason']=='camera_expired_during_display_preparation'
    assert sample.evidence['source_capture_timestamp']==100.


def test_frame_regression_and_identity_reuse_are_hidden(tmp_path):
    feed,clock,state,pixels=fixture(tmp_path);assert feed.sample().evidence['image_visible']
    write_frame(tmp_path,frame_id=3,stamp=100.01)
    assert feed.sample().evidence['reason']=='camera_frame_regressed'
    write_frame(tmp_path,frame_id=4,stamp=100.01)
    assert feed.sample().evidence['reason']=='camera_frame_identity_reused'


def test_original_native_scene_shapes_are_used_without_constructor_or_server(tmp_path,monkeypatch):
    from teleop_assist.copilot import xr_camera as camera_candidate
    feed,clock,state,pixels=fixture(tmp_path)
    source=ORIGINAL.read_bytes();assert hashlib.sha256(source).hexdigest()==ORIGINAL_TELEVUER_SHA256
    tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TeleVuer')
    original=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='main_pass_through')
    def schema(name):return lambda *args,**kwargs:dict(schema=name,args=args,kwargs=kwargs)
    namespace={name:schema(name) for name in ('Hands','MotionControllers','ImageBackground')}
    namespace['asyncio']=asyncio
    exec(compile(ast.Module(body=[original],type_ignores=[]),str(ORIGINAL),'exec'),namespace)
    fake_class=type('DisplayOnlyOriginalMethod',(),{'main_pass_through':namespace['main_pass_through']})
    calls=[];session=SimpleNamespace(upsert=lambda value,**kwargs:calls.append((value,kwargs)),
        CURRENT_WS_ID=1,vuer=SimpleNamespace(ws={1:SimpleNamespace(closed=False)}))
    async def stop_after_one_cycle(_):raise asyncio.CancelledError()
    monkeypatch.setattr(camera_candidate.asyncio,'sleep',stop_after_one_cycle)
    report=install_on_staged_televuer(fake_class,original_source=ORIGINAL,directory=tmp_path,
        clock_id=CLOCK,width=160,height=120,clock=clock)
    owner=SimpleNamespace(use_hand_tracking=False)
    with pytest.raises(asyncio.CancelledError):asyncio.run(fake_class.main_pass_through(owner,session))
    assert calls[0][0]['schema']=='MotionControllers'
    assert calls[0][0]['kwargs']==dict(stream=True,key='motionControllers',left=True,right=True)
    image=calls[1][0][0]
    assert image['schema']=='ImageBackground' and image['kwargs']['key']=='background-mono'
    assert image['kwargs']['height']==1.8 and image['kwargs']['distanceToCamera']==2
    assert image['kwargs']['format']=='jpeg' and image['args'][0].shape[1:]==(160,3) and image['args'][0].shape[0]>184
    np.testing.assert_array_equal(image['args'][0][-120:],pixels)
    assert not report['headset_tested'] and not report['movement_authorized']
    assert ORIGINAL.read_bytes()==source


def test_unreviewed_original_source_cannot_install(tmp_path):
    original=tmp_path/'different.py';original.write_text('unreviewed')
    with pytest.raises(ValueError,match='unreviewed_original_televuer_source'):
        install_on_staged_televuer(object,original_source=original,directory=tmp_path,clock_id=CLOCK)


@pytest.mark.parametrize('fault', ['deep_json', 'jpeg_bomb'])
def test_bad_decoders_produce_hidden_image_instead_of_ending_stream(tmp_path,monkeypatch,fault):
    from teleop_assist.copilot import xr_camera as camera_candidate
    feed,clock,state,pixels=fixture(tmp_path)
    assert feed.sample().evidence['image_visible']
    if fault=='deep_json':
        (tmp_path/'ui_state.json').write_text('['*1100+'0'+']'*1100)
    else:
        def bomb(*args,**kwargs): raise Image.DecompressionBombError('malformed dimensions')
        monkeypatch.setattr(camera_candidate.Image,'open',bomb)
    displayed=feed.sample()
    assert not displayed.evidence['image_visible']
    assert displayed.evidence['reason'] in ('RecursionError','DecompressionBombError')
    assert not np.array_equal(displayed.rgb[-120:],pixels)


def test_header_wraps_all_warning_text_at_small_width(tmp_path):
    from PIL import ImageDraw
    feed,clock,state,pixels=fixture(tmp_path);displayed=feed.sample()
    lines=displayed.evidence['header_lines']
    measure=ImageDraw.Draw(Image.new('RGB',(1,1)))
    assert all(measure.textbbox((0,0),line,font=feed.font)[2]<=feed.width-12 for line in lines)
    text=' '.join(lines)
    assert 'NOT headset/display latency.' in text
    assert 'If this counter freezes, freshness is unknown.' in text
    assert 'Image grants NO movement permission.' in text
    assert displayed.evidence['camera_row_offset']>=10+17*len(lines)
    np.testing.assert_array_equal(displayed.rgb[-120:],pixels)


def test_expiration_during_canvas_work_is_hidden(tmp_path,monkeypatch):
    feed,clock,state,pixels=fixture(tmp_path)
    original=feed._compose
    def delayed(*args):
        result=original(*args);clock.value=100.26;return result
    monkeypatch.setattr(feed,'_compose',delayed)
    displayed=feed.sample()
    assert not displayed.evidence['image_visible']
    assert displayed.evidence['reason']=='camera_expired_during_display_preparation'
    assert displayed.evidence['source_capture_timestamp']==100.


def test_expiration_during_native_encoding_cannot_enqueue_old_camera(tmp_path):
    feed,clock,state,pixels=fixture(tmp_path);calls=[]
    def delayed_encoder(rgb,**kwargs):
        calls.append(rgb.copy());clock.value=100.26
        return dict(pixels=rgb,kwargs=kwargs)
    background,displayed=native_background(feed,delayed_encoder)
    assert len(calls)==2
    np.testing.assert_array_equal(calls[0][-120:],pixels)
    assert not np.array_equal(background['pixels'][-120:],pixels)
    assert not displayed.evidence['image_visible']
    assert displayed.evidence['reason']=='camera_expired_during_native_jpeg_encoding'
    assert displayed.evidence['source_capture_timestamp']==100.


VUER_SERVER = Path('/home/loya2/miniconda3/envs/to/lib/python3.10/site-packages/vuer/server.py')


def native_session_without_server():
    """Execute the actual session enqueue contract; never Vuer's constructor."""
    import __future__
    from collections import deque
    from functools import partial
    source=VUER_SERVER.read_bytes()
    assert hashlib.sha256(source).hexdigest() == '528f8a24c7b77dfb061c28f87d47336d045247171de1aff265dadd1647c28876'
    tree=ast.parse(source)
    at=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='At')
    session=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VuerSession')
    session.body=[n for n in session.body if isinstance(n,ast.FunctionDef) and n.name in ('__init__','__matmul__','upsert')]
    class Event:
        def __init__(self,*elements,**kwargs):self.data=dict(elements=elements,**kwargs)
        def serialize(self):return self.data
    class Frame(Event):pass
    namespace=dict(partial=partial,deque=deque,ServerEvent=Event,Frame=Frame,Upsert=Event,
        packb=lambda event,**kwargs:event)
    exec(compile(ast.Module(body=[at,session],type_ignores=[]),str(VUER_SERVER),'exec',
        flags=__future__.annotations.compiler_flag),namespace)
    class Socket:
        closed=False
        def __bool__(self):return False  # Real aiohttp response may be falsy while open.
    vuer=SimpleNamespace(ws={1:Socket()})
    return namespace['VuerSession'](vuer,1),namespace


def install_fixture_scene(tmp_path,clock):
    tree=ast.parse(ORIGINAL.read_bytes())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TeleVuer')
    method=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='main_pass_through')
    namespace={name:(lambda *args,**kwargs:dict(args=args,kwargs=kwargs))
        for name in ('Hands','MotionControllers','ImageBackground')}
    namespace['asyncio']=asyncio
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(ORIGINAL),'exec'),namespace)
    cls=type('DisplayOnlyOriginalMethod',(),{'main_pass_through':namespace['main_pass_through']})
    install_on_staged_televuer(cls,original_source=ORIGINAL,directory=tmp_path,
        clock_id=CLOCK,width=160,height=120,clock=clock)
    return cls.main_pass_through


@pytest.mark.parametrize('end', ['removed', 'closed'])
def test_actual_vuer_session_disconnect_finishes_after_last_image_without_error(tmp_path,monkeypatch,end):
    from teleop_assist.copilot import xr_camera
    feed,clock,state,pixels=fixture(tmp_path);session,_=native_session_without_server()
    scene=install_fixture_scene(tmp_path,clock);sleeps=[]
    async def disconnect(delay):
        sleeps.append(delay)
        if end=='removed':session.vuer.ws.pop(1)
        else:session.vuer.ws[1].closed=True
    monkeypatch.setattr(xr_camera.asyncio,'sleep',disconnect)
    asyncio.run(scene(SimpleNamespace(use_hand_tracking=False),session))
    assert len(session.uplink_queue)==2 and sleeps==[.05]
    assert session.uplink_queue[-1]['elements'][0]['kwargs']['height']==1.8


@pytest.mark.parametrize('end', ['removed', 'closed'])
def test_already_disconnected_session_does_not_enqueue_input_or_image(tmp_path,end):
    feed,clock,state,pixels=fixture(tmp_path);session,_=native_session_without_server()
    if end=='removed':session.vuer.ws.pop(1)
    else:session.vuer.ws[1].closed=True
    asyncio.run(install_fixture_scene(tmp_path,clock)(SimpleNamespace(use_hand_tracking=True),session))
    assert not session.uplink_queue


def test_disconnect_during_native_encoding_does_not_enqueue_image(tmp_path,monkeypatch):
    from teleop_assist.copilot import xr_camera
    feed,clock,state,pixels=fixture(tmp_path);session,_=native_session_without_server()
    original=xr_camera.native_background
    def close_during_encoding(*args):
        result=original(*args);session.vuer.ws.pop(1);return result
    monkeypatch.setattr(xr_camera,'native_background',close_during_encoding)
    asyncio.run(install_fixture_scene(tmp_path,clock)(SimpleNamespace(use_hand_tracking=False),session))
    assert len(session.uplink_queue)==1  # The input component preceded the close.


def test_unrelated_original_enqueue_assertion_is_not_swallowed(tmp_path):
    feed,clock,state,pixels=fixture(tmp_path);session,namespace=native_session_without_server()
    def bad_serializer(*args,**kwargs):raise AssertionError('TEST unrelated serialization failure')
    namespace['packb']=bad_serializer
    with pytest.raises(AssertionError,match='unrelated serialization failure'):
        asyncio.run(install_fixture_scene(tmp_path,clock)(SimpleNamespace(use_hand_tracking=False),session))
    assert not session.uplink_queue and 1 in session.vuer.ws


def test_camera_preparation_leaves_event_loop_available_for_controller_receiver(tmp_path,monkeypatch):
    import threading
    from teleop_assist.copilot import xr_camera
    _,clock,_,_=fixture(tmp_path);session,_=native_session_without_server()
    scene=install_fixture_scene(tmp_path,clock);original=xr_camera.native_background
    started=threading.Event();input_received=threading.Event()
    def slow_drawing(*args):
        started.set()
        assert input_received.wait(.5), 'Camera drawing blocked the input receiver'
        result=original(*args);session.vuer.ws[1].closed=True
        return result
    monkeypatch.setattr(xr_camera,'native_background',slow_drawing)
    async def receive_controller():
        for _ in range(200):
            if started.is_set():input_received.set();return
            await asyncio.sleep(.001)
        raise AssertionError('Camera worker did not start')
    async def run():
        await asyncio.wait_for(asyncio.gather(scene(SimpleNamespace(use_hand_tracking=False),session),
            receive_controller()),timeout=2.)
    asyncio.run(run())
    assert input_received.is_set()


def test_slow_socket_does_not_accumulate_old_camera_frames(tmp_path,monkeypatch):
    from teleop_assist.copilot import xr_camera
    _,clock,_,_=fixture(tmp_path);session,_=native_session_without_server()
    scene=install_fixture_scene(tmp_path,clock);sizes=[]
    async def stalled_socket(_):
        sizes.append(len(session.uplink_queue))
        if len(sizes)==4:session.vuer.ws[1].closed=True
    monkeypatch.setattr(xr_camera.asyncio,'sleep',stalled_socket)
    asyncio.run(scene(SimpleNamespace(use_hand_tracking=False),session))
    assert sizes==[2,2,2,2]  # Original controller component plus one pending image.


def operator_grasp_fixture(tmp_path):
    """Production-sized head camera and a separate read-only grasp publication."""
    display=tmp_path/'operator_camera'
    pose=np.eye(4).tolist()
    candidate=dict(candidate_id='selected-grasp',base_ee=pose,world_canonical=deepcopy(pose))
    projected=dict(deepcopy(candidate),preview_hand_polygons=[dict(link='palm',
        uv=[[380,260],[410,260],[410,300],[380,300]])],
        preview_uv=[[395,280],[415,280],[395,260],[405,270]])
    preview=dict(capture_timestamp=100.02,clock_id=CLOCK,calibration_id='head-v1',
        preview_current=True,motion_static=True,candidates=[projected])
    frame=dict(rgb=np.full((480,640,3),(50,60,70),dtype=np.uint8),
        T_world_camera=np.eye(4),intrinsics=np.array([[300,0,320],[0,300,240],[0,0,1]]),
        frame_id=55,timestamp=100.05,clock_id=CLOCK,calibration_id='head-v1',camera_source='head')
    binding=publish_operator_frame(display,frame,grasp_preview=preview)
    control=dict(mode='isolated_PhysX',timestamp=100.09,clock_id=CLOCK,
        candidates=[candidate],ai_assist=dict(reason='learned_candidates_ready'),
        snapshot=dict(clutched=True,teleop_active=True,confirmed=False,
            selected_candidate_id='selected-grasp',confirmed_candidate_id=None))
    (tmp_path/'sim_runtime.json').write_text(json.dumps(control))
    # These files deliberately look actionable; the display must leave them alone.
    (tmp_path/'ui_commands.json').write_bytes(b'{"revision":7,"assist_held":false}')
    (tmp_path/'control_lease.json').write_bytes(b'{"owner":"real-input-owner"}')
    pixels=np.asarray(Image.open(display/'frame.jpg').convert('RGB')).copy()
    clock=Clock()
    return SimpleNamespace(display=display,frame=frame,preview=preview,binding=binding,
        control=control,pixels=pixels,clock=clock,
        feed=BoundSimCamera(display,clock_id=CLOCK,width=640,height=480,clock=clock))


def test_operator_publisher_binds_preview_to_exact_camera_without_retiming(tmp_path):
    item=operator_grasp_fixture(tmp_path)
    published=json.loads((item.display/'grasp_preview.json').read_text())
    display=json.loads((item.display/'ui_state.json').read_text())
    assert published['preview_frame']==item.binding==display['display_frame']
    assert published['preview_frame']['jpeg_sha256']==hashlib.sha256((item.display/'frame.jpg').read_bytes()).hexdigest()
    assert published['capture_timestamp']==item.preview['capture_timestamp']==100.02
    assert published['preview_frame']['capture_timestamp']==100.05
    assert 'preview_frame' not in item.preview  # No mutation of producer evidence.
    assert published['candidates']==item.preview['candidates']
    assert display['camera_source']=='head' and display['movement_authorized'] is False


def test_operator_camera_renders_selected_then_confirmed_without_control_writes(tmp_path):
    item=operator_grasp_fixture(tmp_path)
    originals={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    selected=item.feed.sample()
    overlay=selected.evidence['grasp_overlay']
    assert selected.evidence['image_visible'] and overlay['geometry_visible']
    assert overlay['state']=='selected' and overlay['candidate_id']=='selected-grasp'
    assert 'B bestaetigen' in ' '.join(selected.evidence['header_lines'])
    selected_pixel=selected.rgb[-480:][300,395]
    assert selected_pixel[0]>230 and selected_pixel[1]>190 and selected_pixel[2]<90
    assert originals=={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    item.control['snapshot'].update(confirmed=True,confirmed_candidate_id='selected-grasp')
    (tmp_path/'sim_runtime.json').write_text(json.dumps(item.control))
    originals={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    confirmed=item.feed.sample()
    overlay=confirmed.evidence['grasp_overlay']
    assert confirmed.evidence['image_visible'] and overlay['state']=='confirmed'
    assert 'Seitengriff rechts halten' in ' '.join(confirmed.evidence['header_lines'])
    confirmed_pixel=confirmed.rgb[-480:][300,395]
    assert confirmed_pixel[1]>200 and confirmed_pixel[0]<100
    assert not overlay['movement_authorized'] and not confirmed.evidence['movement_authorized']
    assert originals=={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    np.testing.assert_array_equal(confirmed.rgb[-100:],item.pixels[-100:])


@pytest.mark.parametrize('fault,reason',[
    ('stale','preview_or_camera_not_current'),('projection','preview_projection_mismatch'),
    ('pose','selected_candidate_not_matched'),('missing','projection_binding_missing')])
def test_operator_bad_grasp_preview_hides_geometry_but_keeps_head_camera(tmp_path,fault,reason):
    item=operator_grasp_fixture(tmp_path)
    assert item.feed.sample().evidence['grasp_overlay']['geometry_visible']
    path=item.display/'grasp_preview.json'
    preview=json.loads(path.read_text())
    if fault=='stale':preview['capture_timestamp']=99.8
    elif fault=='projection':preview['preview_frame']['projection_sha256']='b'*64
    elif fault=='pose':preview['candidates'][0]['base_ee'][0][3]=.1
    if fault=='missing':path.unlink()
    else:path.write_text(json.dumps(preview))
    originals={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    displayed=item.feed.sample()
    assert displayed.evidence['image_visible'] and displayed.evidence['reason']=='bound_current_sim_camera'
    assert not displayed.evidence['grasp_overlay']['geometry_visible']
    assert displayed.evidence['grasp_overlay']['reason']==reason
    np.testing.assert_array_equal(displayed.rgb[-480:],item.pixels)
    assert originals=={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_operator_preview_cannot_restore_an_expired_camera(tmp_path):
    item=operator_grasp_fixture(tmp_path)
    assert item.feed.sample().evidence['grasp_overlay']['geometry_visible']
    item.clock.value=100.31
    displayed=item.feed.sample()
    assert not displayed.evidence['image_visible']
    assert displayed.evidence['grasp_overlay'] is None
    assert displayed.evidence['reason']=='camera_frame_stale_or_future'
