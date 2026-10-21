import asyncio
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from teleop_assist.copilot.xr_camera import BoundSimCamera, publish_operator_frame
from teleop_assist.copilot.xr_panels import CameraPanel, PANEL_HEADER, native_panel, stream_operator_panels


CLOCK = 'host_monotonic:panel_test'


def frame(identity=1):
    return dict(rgb=np.full((480,640,3),(180,60,25),dtype=np.uint8), frame_id=identity,
        timestamp=100.+identity*.001, clock_id=CLOCK, calibration_id='test_head',
        intrinsics=np.eye(3), T_world_camera=np.eye(4), camera_source='head')


def panel(tmp_path, role='head'):
    directory=tmp_path/('operator_camera' if role=='head' else 'assist_camera')
    publish_operator_frame(directory, frame())
    now=[100.1]
    feed=CameraPanel(directory,role=role,clock_id=CLOCK,clock=lambda:now[0])
    return feed, now


def test_atomic_packet_removes_legacy_pair_race_and_rejects_packet_corruption(tmp_path):
    feed,_=panel(tmp_path)
    (feed.directory/'frame.jpg').write_bytes(b'half of the next publication')
    (feed.directory/'ui_state.json').write_text('{}')
    assert feed.sample().evidence['camera_current']
    packet=feed.directory/'display_frame.bin'
    content=packet.read_bytes();packet.write_bytes(content[:-10]+b'corruption')
    sample=feed.sample()
    assert sample.evidence['cached_display'] and not sample.evidence['camera_current']
    assert sample.evidence['reason']=='camera_jpeg_hash_mismatch'


def test_concurrent_atomic_publication_never_displays_mismatched_frame(tmp_path):
    directory=tmp_path/'operator_camera'
    publish_operator_frame(directory,frame())
    feed=BoundSimCamera(directory,clock_id=CLOCK,clock=lambda:100.2)
    errors=[]
    def publish():
        for identity in range(2,70):
            publish_operator_frame(directory,frame(identity))
    writer=threading.Thread(target=publish);writer.start()
    samples=0
    while writer.is_alive() or samples<100:
        sample=feed.sample();samples+=1
        if not sample.evidence['image_visible']:errors.append(sample.evidence['reason'])
    writer.join()
    assert not errors and feed.last_binding['frame_id']==69


def test_main_camera_never_calls_grasp_renderer_and_retains_exact_pixels(tmp_path,monkeypatch):
    from teleop_assist.copilot import xr_grasp_overlay
    feed,_=panel(tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('Main camera used assistance renderer')
    monkeypatch.setattr(xr_grasp_overlay,'render_grasp_overlay',forbidden)
    (tmp_path/'sim_runtime.json').write_text(json.dumps(dict(timestamp=100.1,clock_id=CLOCK,
        mode='isolated_PhysX',snapshot=dict(teleop_active=True),ai_assist={})))
    result=feed.sample()
    expected=np.asarray(Image.open(feed.directory/'frame.jpg').convert('RGB'))
    np.testing.assert_array_equal(result.rgb[PANEL_HEADER:],expected)
    assert result.evidence['grasp_overlay'] is None


def test_short_gap_holds_pixels_but_never_refreshes_capture_or_authorizes_motion(tmp_path):
    feed,now=panel(tmp_path)
    first=feed.sample()
    now[0]=100.35
    stale=feed.sample()
    assert stale.evidence['cached_display'] and not stale.evidence['camera_current']
    assert stale.evidence['source_capture_timestamp']==first.evidence['source_capture_timestamp']
    assert stale.evidence['movement_authorized'] is False
    assert 'BILD VERZOEGERT' in stale.evidence['header_lines'][0]
    np.testing.assert_array_equal(stale.rgb[PANEL_HEADER+6:-6,6:-6],first.rgb[PANEL_HEADER+6:-6,6:-6])
    now[0]=102.
    frozen=feed.sample()
    assert 'STANDBILD' in frozen.evidence['header_lines'][0]
    assert not np.array_equal(frozen.rgb,frozen.rgb*0)
    fresh=frame(90);fresh['timestamp']=101.99
    publish_operator_frame(feed.directory,fresh)
    resumed=feed.sample()
    assert resumed.evidence['camera_current'] and resumed.evidence['source_frame_id']==90
    assert resumed.rgb.shape==frozen.rgb.shape==first.rgb.shape


def test_assist_renderer_failure_and_missing_camera_remain_local(tmp_path,monkeypatch):
    from teleop_assist.copilot import xr_grasp_overlay
    feed,_=panel(tmp_path,'assist')
    def broken(*args,**kwargs):raise RuntimeError('test renderer failure')
    monkeypatch.setattr(xr_grasp_overlay,'render_grasp_overlay',broken)
    result=feed.sample()
    assert result.evidence['image_visible'] and result.evidence['camera_current']
    assert result.evidence['grasp_overlay']['reason']=='RuntimeError'
    empty=CameraPanel(tmp_path/'missing',role='assist',clock_id=CLOCK,clock=lambda:100.1)
    assert not empty.sample().evidence['image_visible']


def test_two_native_panels_have_stable_distinct_keys_and_no_eye_split(tmp_path):
    head,_=panel(tmp_path);assist,_=panel(tmp_path,'assist')
    schema=lambda pixels,**kwargs:dict(pixels=pixels,**kwargs)
    left,_=native_panel(head,schema);right,_=native_panel(assist,schema)
    assert left['key']=='operator-head' and right['key']=='operator-assist'
    assert left['position']==[0,0,0] and right['position'][0]>0
    assert 'layers' not in left and 'layers' not in right
    assert left['position'][0]+left['height']*left['aspect']/2 < right['position'][0]-right['height']*right['aspect']/2
    assert left['pixels'].shape==right['pixels'].shape==(608,640,3)


def test_stalled_preview_does_not_block_head_stream_or_controller_event_loop(tmp_path,monkeypatch):
    from teleop_assist.copilot import xr_panels
    head,_=panel(tmp_path);panel(tmp_path,'assist')
    original=xr_panels.native_panel
    started=threading.Event();release=threading.Event();sent=[];heartbeat=[]
    def slow(feed,schema):
        if feed.role=='assist':
            started.set()
            assert release.wait(2.)
        return original(feed,schema)
    monkeypatch.setattr(xr_panels,'native_panel',slow)
    session=SimpleNamespace(upsert=lambda elements,**kwargs:sent.extend(elements))
    schema=lambda pixels,**kwargs:dict(**kwargs)
    opened=[True]
    async def receiver():
        for _ in range(25):
            heartbeat.append(1);await asyncio.sleep(.01)
        assert started.is_set()
        assert len([s for s in sent if s['key']=='operator-head'])>=3
        assert not any(s['key']=='operator-assist' for s in sent)
        opened[0]=False;release.set()
    async def run():
        try:
            await asyncio.gather(receiver(),stream_operator_panels(session,schema,
                directory=head.directory,clock_id=CLOCK,width=640,height=480,fps=20,
                clock=lambda:100.1,session_open=lambda:opened[0]))
        finally:release.set()
    asyncio.run(run())
    assert len(heartbeat)==25


def test_slow_socket_keeps_at_most_one_queued_panel_update(tmp_path):
    head,_=panel(tmp_path);panel(tmp_path,'assist');queue=[];opened=[True]
    def upsert(elements,**kwargs):queue.append(elements)
    session=SimpleNamespace(upsert=upsert,uplink_queue=queue)
    async def close():
        await asyncio.sleep(.2);opened[0]=False
    async def run():
        await asyncio.gather(close(),stream_operator_panels(session,lambda pixels,**kw:kw,
            directory=head.directory,clock_id=CLOCK,width=640,height=480,fps=20,
            clock=lambda:100.1,session_open=lambda:opened[0]))
    asyncio.run(run())
    assert len(queue)==1
