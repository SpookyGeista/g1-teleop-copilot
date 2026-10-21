"""Actual localhost HTTP/OSC boundary checks, no robot or GPU process."""
import copy
import gzip
import http.client
import json
import socket
import struct
import time
import unittest

import cv2
import numpy as np

from teleop_assist.core import AssistSession
from teleop_assist.control_ownership import BrowserControlLease
from teleop_assist.server import AssistHTTPServer, GazeOSCReceiver, decode_osc_message


def osc(address, tags, values):
    def text(value):
        b=value.encode()+b'\0'
        return b+b'\0'*((-len(b)) % 4)
    return text(address)+text(','+tags)+b''.join(struct.pack('>'+t,v) for t,v in zip(tags,values))


def sample(seq=1):
    return {'source':'http_test','session_id':'http','seq':seq,'right':{'valid':True,'position':[0,0,0]},
            'axes':[0,0],'trigger':0,'squeeze':0,'clutch':False}


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=AssistHTTPServer(AssistSession(),port=0)
        cls.server.start()

    @classmethod
    def tearDownClass(cls): cls.server.close()

    def setUp(self):
        self.server.session=AssistSession()
        self.server.control_lease=BrowserControlLease(self.server.session)
        with self.server.frame_lock:self.server.frames.clear()
        self.assertEqual(self.request('POST','/api/command',{'type':'control_claim','deadman_released':True})[0],200)

    def request(self,method,path,data=None,headers=None,raw=None):
        if isinstance(data,dict):data={'client_id':'http_test_owner',**data}
        body=raw if raw is not None else json.dumps(data).encode() if data is not None else None
        request_headers={'Content-Type':'application/json'} if body is not None else {}
        request_headers.update(headers or {})
        connection=http.client.HTTPConnection(*self.server.address,timeout=2)
        try:
            connection.request(method,path,body=body,headers=request_headers)
            response=connection.getresponse();result=(response.status,dict(response.getheaders()),response.read())
        finally:connection.close()
        return result

    def test_actual_state_and_static_es_modules(self):
        status,headers,body=self.request('GET','/api/state')
        self.assertEqual(status,200)
        state=json.loads(body);self.assertEqual(state['schema_version'],1)
        self.assertFalse(state['eye_enabled']);self.assertEqual(headers['Cache-Control'],'no-store')
        for path,kind in [('/','text/html'),('/static/app.mjs','text/javascript'),
                          ('/static/geometry.mjs','text/javascript'),('/static/style.css','text/css')]:
            status,headers,body=self.request('GET',path)
            self.assertEqual(status,200,path);self.assertIn(kind,headers['Content-Type']);self.assertGreater(len(body),40)
        self.assertEqual(self.request('GET','/static/../../core.py')[0],404)

    def test_head_local_ray_roundtrip_and_linked_projection_contract(self):
        ray={'source':'http_test_ray','session_id':'ray-device','seq':5,'valid':True,
             'calibrated':True,'confidence':.9,'head_ray':{'origin':[0,0,0],'direction':[0,0,-1]}}
        self.assertEqual(self.request('POST','/api/gaze',ray)[0],200)
        status,_,body=self.request('GET','/api/gaze');data=json.loads(body)
        self.assertEqual(status,200);self.assertEqual(data['device_sample']['seq'],5)
        self.assertIsNone(data['sample']);self.assertLess(data['device_age'],.25)
        projection={'source':'webxr_gaze_projection','session_id':'projection-client','seq':1,
                    'valid':True,'calibrated':True,'confidence':1.,'uv':[.4,.6],'frame_id':1,
                    'projection_of_seq':5,'projection_of_session':'ray-device'}
        self.assertEqual(self.request('POST','/api/gaze',projection)[0],200)
        self.assertFalse(self.server.session.config.eye_enabled)
        self.assertEqual(self.server.session.gaze['confidence'],.9)
        # Browser head-ray projection uses a nested sample with its client ID.
        nested=dict(projection,seq=2,client_id='http_test_owner')
        self.assertEqual(self.request('POST','/api/command',
            raw=json.dumps({'type':'gaze','sample':nested}).encode())[0],200)
        projection['seq']=2;projection['projection_of_session']='unrelated'
        projection['seq']=3
        self.assertEqual(self.request('POST','/api/gaze',projection)[0],400)

    def test_frame_ids_serve_corresponding_rgb_and_depth_and_expire(self):
        h,w=20,30
        for identifier in range(1,10):
            rgb=np.full((h,w,3),[identifier*15,40,90],np.uint8)
            depth=np.full((h,w),identifier*.1,dtype=np.float32)
            self.server.publish_frame({'frame_id':identifier,'rgb':rgb,'depth':depth})
        self.assertEqual(self.request('GET','/api/frame.jpg?frame_id=1')[0],410)
        self.assertEqual(self.request('GET','/api/depth.bin?frame_id=1')[0],410)
        for identifier in (2,5,9):
            status,headers,body=self.request('GET',f'/api/depth.bin?frame_id={identifier}')
            self.assertEqual(status,200);self.assertEqual(headers['Content-Encoding'],'gzip')
            actual=np.frombuffer(gzip.decompress(body),dtype='<f4').reshape(h,w)
            np.testing.assert_allclose(actual,identifier*.1)
            status,headers,body=self.request('GET',f'/api/frame.jpg?frame_id={identifier}')
            self.assertEqual(status,200);self.assertIn('image/jpeg',headers['Content-Type'])
            decoded=cv2.imdecode(np.frombuffer(body,np.uint8),cv2.IMREAD_COLOR)
            np.testing.assert_allclose(decoded[5,5,::-1],[identifier*15,40,90],atol=3)
        self.assertEqual(self.request('GET','/api/frame.jpg?frame_id=invalid')[0],400)

    def test_valid_and_malformed_commands_preserve_atomic_input(self):
        command={'type':'input','sample':sample()}
        status,_,body=self.request('POST','/api/command',command)
        self.assertEqual(status,200);self.assertTrue(json.loads(body)['accepted'])
        original=copy.deepcopy(self.server.session.input)
        bad=sample(2);bad['pointer_uv']=[float('nan'),0]
        self.assertEqual(self.request('POST','/api/command',{'type':'input','sample':bad})[0],400)
        self.assertEqual(self.server.session.input,original)
        self.assertEqual(self.request('POST','/api/command',{'type':'unknown'})[0],400)
        self.assertEqual(self.request('POST','/api/command',raw=b'badjson')[0],400)
        self.assertEqual(self.request('POST','/api/command',data=['not','object'])[0],400)
        self.assertEqual(self.request('POST','/api/command',data={},headers={'Content-Type':'text/plain'})[0],415)
        self.assertFalse(self.server.errors)

    def test_crossorigin_post_refused_and_sameorigin_accepted(self):
        command={'type':'eye_enable','enabled':True}
        self.assertEqual(self.request('POST','/api/command',command,headers={'Origin':'https://unrelated.invalid'})[0],403)
        self.assertFalse(self.server.session.config.eye_enabled)
        authority=f'{self.server.address[0]}:{self.server.address[1]}'
        self.assertEqual(self.request('POST','/api/command',command,headers={'Origin':'http://'+authority})[0],200)
        self.assertTrue(self.server.session.config.eye_enabled)
        self.assertEqual(self.request('POST','/api/command',command,headers={'Origin':'https://'+authority})[0],403)

    def test_http_gaze_calibration_and_toggle_errors(self):
        self.assertEqual(self.request('POST','/api/gaze',{'sample':{'seq':1,'valid':True,'calibrated':True,'confidence':1.}})[0],400)
        self.assertEqual(self.request('POST','/api/command',{'type':'eye_enable','enabled':'false'})[0],400)
        self.assertEqual(self.request('POST','/api/calibration/eye',{'samples':[]})[0],400)
        self.assertEqual(self.request('POST','/api/calibrate',{})[0],400)
        status,_,body=self.request('GET','/api/gaze');self.assertEqual(status,200)
        self.assertEqual(json.loads(body)['status'],'disabled')

    def test_second_browser_observes_without_mutation_and_cannot_take_held_control(self):
        active=sample();active.update(clutch=True,squeeze=1.)
        self.assertEqual(self.request('POST','/api/command',{'type':'input','sample':active})[0],200)
        before=copy.deepcopy(self.server.session.input)
        observer={'client_id':'observer_pc'}
        self.assertEqual(self.request('POST','/api/command',dict(observer,type='input',sample=sample(20)))[0],409)
        self.assertEqual(self.request('POST','/api/command',dict(observer,type='assist_enable',enabled=True))[0],409)
        self.assertEqual(self.request('POST','/api/command',dict(observer,type='control_claim',deadman_released=True))[0],409)
        self.assertEqual(self.server.session.input,before)
        status,_,body=self.request('GET','/api/state')
        self.assertEqual(status,200)
        self.assertEqual(json.loads(body)['control_session']['owner_client_id'],'http_test_owner')
        self.assertEqual(self.request('POST','/api/command',dict(observer,type='stop'))[0],200)
        self.assertTrue(self.server.session.stopped)
        self.assertEqual(self.request('POST','/api/command',{'type':'input','sample':sample(2)})[0],200)
        self.assertEqual(self.request('POST','/api/command',dict(observer,type='control_claim',deadman_released=True))[0],200)
        self.assertIsNone(self.server.session.input)
        self.assertIsNone(self.server.session.hand_origin)
        self.assertEqual(self.request('POST','/api/command',{'type':'input','sample':sample(3)})[0],409)


class OscTests(unittest.TestCase):
    def test_decode_message_and_reject_truncation_or_types(self):
        packet=osc('/punktnull/gaze/uv','ffifii',[.2,.7,1,.9,7,11])
        address,values=decode_osc_message(packet)
        self.assertEqual(address,'/punktnull/gaze/uv')
        np.testing.assert_allclose(values,[.2,.7,1,.9,7,11],atol=1e-6)
        for bad in (packet[:-1],packet+b'\0',b'not_osc',osc('/x','',[]).replace(b',',b'x',1)):
            with self.assertRaises((ValueError,UnicodeDecodeError)):decode_osc_message(bad)

    def test_actual_udp_receiver_valid_and_invalid_samples(self):
        session=AssistSession();receiver=GazeOSCReceiver(session,port=0);sender=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        receiver.start()
        def send(packet,condition):
            sender.sendto(packet,receiver.address)
            deadline=time.monotonic()+1.
            while not condition() and time.monotonic()<deadline:time.sleep(.005)
            self.assertTrue(condition(),'OSC receiver did not process packet before deadline')
        try:
            send(osc('/punktnull/gaze/uv','ffifii',[.2,.7,1,.9,1,11]),lambda:receiver.received==1)
            np.testing.assert_allclose(session.gaze['uv'],[.2,.7],atol=1e-6)
            self.assertTrue(session.gaze['calibrated']);self.assertTrue(session.gaze['valid'])
            initial=copy.deepcopy(session.gaze)
            for bad in (b'bad',osc('/other','ffifii',[.2,.7,1,.9,2,11]),
                        osc('/punktnull/gaze/uv','ffifii',[.2,.7,2,.9,2,11]),
                        osc('/punktnull/gaze/uv','ffffff',[.2,.7,1.,.9,2.5,11.5])):
                before=receiver.errors
                send(bad,lambda:receiver.errors>before)
                self.assertEqual(session.gaze,initial)
            send(osc('/punktnull/gaze/raw','ffifii',[.2,.7,1,.9,3,11]),lambda:receiver.received==2)
            self.assertFalse(session.gaze['calibrated'])
        finally:
            sender.close();receiver.close()


if __name__=='__main__':unittest.main()
