"""Actual HTTP checks of RGB/depth identity across independently cached cameras."""
import gzip
import http.client
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from teleop_assist.core import AssistSession
from teleop_assist.server import AssistHTTPServer


@pytest.fixture
def server():
    value = AssistHTTPServer(AssistSession(), port=0)
    value.start()
    try:
        yield value
    finally:
        value.close()


def get(server, route):
    connection = http.client.HTTPConnection(*server.address, timeout=2)
    try:
        connection.request('GET', route)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def publish(server, camera, identifier, color, depth_value, shape=(24, 32)):
    rgb = np.full((*shape, 3), color, dtype=np.uint8)
    depth = np.full(shape, depth_value, dtype=np.float32)
    server.publish_frame({'frame_id': identifier, 'rgb': rgb, 'depth': depth}, camera_id=camera)
    return rgb, depth


def assert_frame(server, camera, identifier, color, depth, shape=(24, 32)):
    suffix = f'?camera_id={camera}&frame_id={identifier}'
    status, headers, payload = get(server, '/api/frame.jpg'+suffix)
    assert status == 200 and headers['Content-Type'] == 'image/jpeg'
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    assert image.shape == (*shape, 3)
    np.testing.assert_allclose(image[10, 10], color, atol=3)
    status, headers, payload = get(server, '/api/depth.bin'+suffix)
    assert status == 200 and headers['Content-Encoding'] == 'gzip'
    values = np.frombuffer(gzip.decompress(payload), dtype='<f4').reshape(shape)
    np.testing.assert_allclose(values, depth)


def test_same_frame_number_does_not_mix_cameras_or_dimensions(server):
    configurations = [('head', [170, 40, 60], .4, (24, 32)),
                      ('left_wrist', [50, 170, 40], .7, (32, 48)),
                      ('right_wrist', [40, 60, 170], 1.2, (40, 64))]
    for camera, color, depth, shape in configurations:
        for identifier in (11, 12):
            publish(server, camera, identifier, color, depth+identifier*.01, shape)
    for camera, color, depth, shape in configurations:
        for identifier in (11, 12):
            assert_frame(server, camera, identifier, color, depth+identifier*.01, shape)
    assert get(server, '/api/depth.bin?camera_id=right_wrist&frame_id=13')[0] == 410
    assert get(server, '/api/frame.jpg?camera_id=missing&frame_id=12')[0] == 410
    assert get(server, '/api/frame.jpg?camera_id=right_wrist&frame_id=NaN')[0] == 400


def test_cache_expiration_and_latest_are_per_camera(server):
    publish(server, 'head', 1, [20, 40, 80], .3)
    publish(server, 'right_wrist', 1, [80, 40, 20], .5)
    for identifier in range(1, 10):
        publish(server, 'left_wrist', identifier, [30, 60, 90], identifier*.1)
    for kind in ('frame.jpg', 'depth.bin'):
        assert get(server, '/api/'+kind+'?camera_id=left_wrist&frame_id=1')[0] == 410
        assert get(server, '/api/'+kind+'?camera_id=head&frame_id=1')[0] == 200
        assert get(server, '/api/'+kind+'?camera_id=right_wrist&frame_id=1')[0] == 200
    status, _, payload = get(server, '/api/depth.bin?camera_id=left_wrist')
    assert status == 200
    np.testing.assert_allclose(np.frombuffer(gzip.decompress(payload), '<f4'), .9)
    # Omitting camera_id must continue serving the head stream.
    status, _, payload = get(server, '/api/depth.bin?frame_id=1')
    assert status == 200
    np.testing.assert_allclose(np.frombuffer(gzip.decompress(payload), '<f4'), .3)


@pytest.mark.parametrize('camera', ['head', 'left_wrist', 'right_wrist'])
def test_lazy_encoding_uses_immutable_published_snapshot(server, camera):
    rgb, depth = publish(server, camera, 4, [45, 120, 70], .42)
    rgb[:] = [230, 5, 5]
    depth[:] = 9.9
    assert_frame(server, camera, 4, [45, 120, 70], .42)
    publish(server, camera, 5, [5, 20, 230], .84)
    assert_frame(server, camera, 4, [45, 120, 70], .42)
    assert_frame(server, camera, 5, [5, 20, 230], .84)


def test_publication_never_encodes_and_tabs_share_per_format_payload(server, monkeypatch):
    counts={'jpeg':0,'depth':0}
    lock=threading.Lock()
    original_jpeg=cv2.imencode;original_depth=gzip.compress
    def jpeg(*args,**kwargs):
        with lock:counts['jpeg']+=1
        return original_jpeg(*args,**kwargs)
    def depth(*args,**kwargs):
        with lock:counts['depth']+=1
        return original_depth(*args,**kwargs)
    monkeypatch.setattr(cv2,'imencode',jpeg);monkeypatch.setattr(gzip,'compress',depth)
    publish(server,'head',10,[80,150,210],.73)
    assert counts=={'jpeg':0,'depth':0}
    routes=['/api/frame.jpg?frame_id=10','/api/depth.bin?frame_id=10']*4
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses=list(pool.map(lambda route:get(server,route),routes))
    assert all(status==200 for status,_,_ in responses)
    assert all(responses[i][2]==responses[i%2][2] for i in range(len(responses)))
    assert counts=={'jpeg':1,'depth':1}
    assert_frame(server,'head',10,[80,150,210],.73)
    assert counts=={'jpeg':1,'depth':1}
    # Identical numeric IDs in another view own distinct format caches.
    publish(server,'left_wrist',10,[10,20,30],.25)
    assert_frame(server,'left_wrist',10,[10,20,30],.25)
    assert counts=={'jpeg':2,'depth':2}


def test_slow_jpeg_request_does_not_block_publication_or_depth(server, monkeypatch):
    started=threading.Event();release=threading.Event()
    original=cv2.imencode
    def blocked(*args,**kwargs):
        started.set()
        assert release.wait(2),'Test did not release JPEG encoder'
        return original(*args,**kwargs)
    monkeypatch.setattr(cv2,'imencode',blocked)
    publish(server,'head',10,[80,150,210],.73)
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending=pool.submit(get,server,'/api/frame.jpg?frame_id=10')
        try:
            assert started.wait(1)
            # Completion is required while JPEG remains deliberately blocked.
            pool.submit(publish,server,'head',11,[10,20,30],.25).result(timeout=1)
            status,_,body=pool.submit(get,server,'/api/depth.bin?frame_id=10').result(timeout=1)
            assert status==200
            np.testing.assert_allclose(np.frombuffer(gzip.decompress(body),'<f4'),.73)
            assert not pending.done()
        finally:
            release.set()
        status,_,body=pending.result(timeout=1)
        assert status==200
        actual=cv2.imdecode(np.frombuffer(body,np.uint8),cv2.IMREAD_COLOR)[...,::-1]
        np.testing.assert_allclose(actual[10,10],[80,150,210],atol=3)


def test_evicted_lazy_head_frame_returns_gone_for_both_formats(server):
    for identifier in range(10):
        publish(server,'head',identifier,[identifier,40,80],.2+identifier*.01)
    assert len(server.frames)==8
    for kind in ('frame.jpg','depth.bin'):
        assert get(server,f'/api/{kind}?frame_id=1')[0]==410
    assert_frame(server,'head',9,[9,40,80],.29)


def live_bundle(server,identifier,color=(50,100,150),depth=.4):
    stamp=server.session.clock()
    frames={}
    for camera_id in ('head','left_wrist','right_wrist'):
        frames[camera_id]={'frame_id':identifier,'timestamp':stamp,'sim_time':identifier*.1,
            'rgb':np.full((24,32,3),color,np.uint8),'depth':np.full((24,32),depth,np.float32),
            'intrinsics':{'fx':30.,'fy':30.,'cx':16.,'cy':12.},'T_world_camera':np.eye(4),
            'depth_convention':'optical_z_meters','source':'unit_test_render'}
    return {'frames':frames}


def test_raw_display_does_not_renew_control_measurement_and_copies_metadata(server):
    bundle=live_bundle(server,11)
    before=server.session.snapshot()
    assert server.publish_live_bundle(bundle)
    head=bundle['frames']['head'];stamp=head['timestamp']
    head['T_world_camera'][0,3]=99;head['intrinsics']['fx']=999
    head['rgb'][:]=255;head['depth'][:]=999
    status,_,body=get(server,'/api/state');state=json.loads(body)
    assert status==200
    for key in ('frame_id','frame_age','camera','targets','selection','hand_calibrated'):
        assert state[key]==before[key]
    display=state['display_frame']
    assert display['frame_id']==11 and display['timestamp']==stamp and display['age_s']>=0
    assert display['camera']['intrinsics'][0][0]==30
    assert display['camera']['T_world_camera'][0][3]==0
    for camera in bundle['frames']:assert_frame(server,camera,11,[50,100,150],.4)
    # Callers cannot mutate the metadata subsequently exposed by HTTP.
    snapshot=server.display_snapshot();snapshot['camera']['intrinsics'][0][0]=-1
    assert server.display_snapshot()['camera']['intrinsics'][0][0]==30


def test_live_history_cannot_evict_measured_frame_and_promotion_reuses_snapshot(server):
    bundle=live_bundle(server,1)
    server.publish_live_bundle(bundle)
    server.publish_frame(bundle['frames']['head'])
    assert server.frames[1] is server.live_sensor_frames['head'][1]
    for identifier in range(2,12):server.publish_live_bundle(live_bundle(server,identifier,depth=identifier*.1))
    assert len(server.live_sensor_frames['head'])==8
    assert_frame(server,'head',1,[50,100,150],.4)
    for kind in ('frame.jpg','depth.bin'):
        assert get(server,f'/api/{kind}?frame_id=2')[0]==410
        assert get(server,f'/api/{kind}?camera_id=right_wrist&frame_id=1')[0]==410
    assert_frame(server,'head',11,[50,100,150],1.1)
    assert server.display_snapshot()['frame_id']==11


def test_invalid_or_late_live_bundle_cannot_partially_replace_display(server):
    server.publish_live_bundle(live_bundle(server,5))
    before=server.display_snapshot()
    assert not server.publish_live_bundle(live_bundle(server,4))
    invalid=live_bundle(server,6);invalid['frames']['right_wrist']['frame_id']=7
    with pytest.raises(ValueError,match='share one acquisition'):server.publish_live_bundle(invalid)
    invalid=live_bundle(server,6);invalid['frames']['head']['T_world_camera'][0,0]=np.nan
    with pytest.raises(ValueError,match='Finite calibrated'):server.publish_live_bundle(invalid)
    after=server.display_snapshot()
    for key in ('frame_id','timestamp','camera'):assert after[key]==before[key]
    assert all(6 not in cache for cache in server.live_sensor_frames.values())


def test_delayed_measurement_publication_never_rewinds_latest_display(server):
    old=live_bundle(server,3);server.publish_live_bundle(old)
    server.publish_live_bundle(live_bundle(server,9,depth=.9))
    server.publish_frame(old['frames']['head'])
    assert server.display_snapshot()['frame_id']==9
    status,_,body=get(server,'/api/depth.bin')
    assert status==200
    np.testing.assert_allclose(np.frombuffer(gzip.decompress(body),'<f4'),.9)
    assert_frame(server,'head',3,[50,100,150],.4)
