"""Local viewer and control service; no real robot/DDS endpoints are opened."""
from __future__ import annotations
from collections import OrderedDict
import copy
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import ssl
import struct
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np
from .control_ownership import BrowserControlLease, ControlConflict


def json_bytes(value):
    return json.dumps(value,allow_nan=False,ensure_ascii=False,
                      default=lambda v: v.tolist() if hasattr(v,"tolist") else str(v)).encode()


class PublishedRGBDFrame:
    """Immutable acquisition with independent once-per-format HTTP encoding.

    The producer only copies sensor buffers. Compression never holds the
    server's frame-cache lock or runs on the control-observation publication
    path. Separate format locks allow the browser's RGB/depth requests to run
    in parallel while concurrent tabs share their encoded payloads.
    """
    def __init__(self, rgb, depth):
        self.rgb=np.array(rgb,dtype=np.uint8,copy=True,order='C')
        self.depth=np.array(depth,dtype='<f4',copy=True,order='C')
        if self.rgb.ndim!=3 or self.rgb.shape[2]!=3 or self.depth.shape!=self.rgb.shape[:2]:
            raise ValueError('RGB and optical depth must have matching image dimensions')
        self.rgb.setflags(write=False);self.depth.setflags(write=False)
        self._payloads={}
        self._locks={kind:threading.Lock() for kind in ('jpeg','depth')}

    def encoded(self,kind):
        with self._locks[kind]:
            if kind not in self._payloads:
                if kind=='jpeg':
                    import cv2
                    ok,value=cv2.imencode('.jpg',cv2.cvtColor(self.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85])
                    if not ok:raise RuntimeError('Sensor JPEG encoding failed')
                    self._payloads[kind]=value.tobytes()
                else:
                    self._payloads[kind]=gzip.compress(self.depth.tobytes(),compresslevel=1)
            return self._payloads[kind]


class AssistHTTPServer:
    def __init__(self,session,host="127.0.0.1",port=8765,cert=None,key=None,static_dir=None):
        self.session=session
        self.frames=OrderedDict(); self.frame_lock=threading.Lock()
        self.sensor_frames={}
        # Raw display and measured histories have independent bounded lifetimes.
        # A faster video producer must not evict the last measured RGB-D frame.
        self.live_sensor_frames={}
        self.display_frame=None
        self.static_dir=Path(static_dir or Path(__file__).parent/"static").resolve()
        self.errors=[]
        self.control_lease=BrowserControlLease(session)
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                return
            def send(self,status,body,content_type="application/json",encoding=None):
                self.send_response(status)
                self.send_header("Content-Type",content_type)
                self.send_header("Content-Length",str(len(body)))
                self.send_header("Cache-Control","no-store")
                self.send_header("X-Content-Type-Options","nosniff")
                self.send_header("Permissions-Policy","xr-spatial-tracking=(self), camera=(), microphone=()")
                if encoding:self.send_header("Content-Encoding",encoding)
                self.end_headers()
                try:self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
            def error(self,status,message):
                self.send(status,json_bytes({"error":message}))
            def do_GET(self):
                parsed=urlparse(self.path); path=parsed.path
                if path=="/api/state":
                    with owner.session.lock:
                        control=owner.control_lease.snapshot()
                        state=owner.session.snapshot()
                        state['control_session']=control
                    state['display_frame']=owner.display_snapshot()
                    return self.send(200,json_bytes(state))
                if path=="/api/gaze":
                    with owner.session.lock:
                        sample=owner.session.gaze
                        device=owner.session.gaze_device
                        device_age=owner.session.clock()-owner.session.gaze_device_time if device is not None else None
                        status=owner.session.eye_status(owner.session.clock())
                    return self.send(200,json_bytes({"sample":sample,"device_sample":device,"device_age":device_age,
                                                     "status":status}))
                if path in ("/api/frame.jpg","/api/depth.bin"):
                    try:
                        requested=parse_qs(parsed.query).get("frame_id",[None])[0]
                        camera_id=parse_qs(parsed.query).get("camera_id",["head"])[0]
                        with owner.frame_lock:
                            frames=owner.frames if camera_id=="head" else owner.sensor_frames.get(camera_id,{})
                            live=owner.live_sensor_frames.get(camera_id,{})
                            key=int(requested) if requested is not None else max((*frames,*live),default=-1)
                            frame=frames.get(key) or live.get(key)
                        if frame is None:return self.error(410,"Frame no longer available; request current state")
                        if path.endswith("jpg"):
                            return self.send(200,frame.encoded('jpeg'),"image/jpeg")
                        return self.send(200,frame.encoded('depth'),"application/octet-stream","gzip")
                    except (ValueError,TypeError):return self.error(400,"Invalid frame identifier")
                    except RuntimeError as exc:
                        owner.errors.append(repr(exc));return self.error(500,"Sensor encoding failed")
                relative="index.html" if path=="/" else path.removeprefix("/static/").lstrip("/")
                file=(owner.static_dir/relative).resolve()
                if not file.is_relative_to(owner.static_dir) or not file.is_file():
                    return self.error(404,"Not found")
                types={".html":"text/html; charset=utf-8",".js":"text/javascript; charset=utf-8",".mjs":"text/javascript; charset=utf-8",
                       ".css":"text/css; charset=utf-8",".json":"application/json",".txt":"text/plain; charset=utf-8"}
                self.send(200,file.read_bytes(),types.get(file.suffix,"application/octet-stream"))
            def do_POST(self):
                path=urlparse(self.path).path
                if path not in ("/api/command","/api/calibrate","/api/gaze","/api/calibration/eye"):
                    return self.error(404,"Unknown endpoint")
                origin=self.headers.get("Origin")
                if origin and (urlparse(origin).netloc!=self.headers.get("Host") or urlparse(origin).scheme!=owner.scheme):
                    return self.error(403,"Cross-origin commands are not accepted")
                try:
                    length=int(self.headers.get("Content-Length","0"))
                    if not 0<length<=65536:return self.error(413,"Command body must be 1..65536 bytes")
                    if "application/json" not in self.headers.get("Content-Type",""):
                        return self.error(415,"Use application/json")
                    data=json.loads(self.rfile.read(length))
                    if not isinstance(data,dict):raise ValueError("Expected JSON object")
                    if path=="/api/calibrate":data["type"]="calibrate"
                    if path=="/api/gaze":data["type"]="gaze"
                    if path=="/api/calibration/eye":data["type"]="eye_calibrate"
                    result=owner.control_lease.dispatch(data)
                    self.send(200,json_bytes(result))
                except ControlConflict as exc:
                    self.error(409,str(exc)[:240])
                except (ValueError,KeyError,TypeError,OverflowError) as exc:
                    self.error(400,str(exc)[:240])
                except Exception as exc:
                    owner.errors.append(repr(exc));self.error(500,"Control service error")
        self.httpd=ThreadingHTTPServer((host,port),Handler)
        self.httpd.daemon_threads=True
        if bool(cert)!=bool(key):raise ValueError("TLS needs both certificate and key")
        if cert:
            context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(cert),str(key));self.httpd.socket=context.wrap_socket(self.httpd.socket,server_side=True)
        self.address=self.httpd.server_address
        self.scheme="https" if cert else "http"
        self.thread=threading.Thread(target=self.httpd.serve_forever,daemon=True,name="assist-http")

    def start(self):
        self.thread.start()

    def publish_frame(self,frame,camera_id="head"):
        # Retain one coherent snapshot even if the acquisition reuses buffers.
        # Copy before taking the cache lock; encoding belongs to HTTP threads.
        identifier=int(frame['frame_id'])
        with self.frame_lock:
            published=self.live_sensor_frames.get(camera_id,{}).get(identifier)
        if published is None:published=PublishedRGBDFrame(frame['rgb'],frame['depth'])
        with self.frame_lock:
            frames=self.frames if camera_id=="head" else self.sensor_frames.setdefault(camera_id,OrderedDict())
            frames[identifier]=published
            while len(frames)>8:frames.popitem(last=False)

    def publish_live_bundle(self,bundle):
        """Publish immutable synchronized sensor pixels, never a control frame.

        The additive display descriptor is head-only. Existing frame endpoints
        serve explicit IDs from either history. Measurement identity, pose age,
        selection and control timestamps remain exclusively owned by Session.
        """
        frames=bundle['frames'];head=frames['head']
        identifier=int(head['frame_id']);timestamp=float(head['timestamp'])
        if identifier<0 or not np.isfinite(timestamp) or timestamp<0 or timestamp>self.session.clock()+.01:
            raise ValueError('Invalid live acquisition identity')
        with self.frame_lock:
            if self.display_frame is not None and identifier<=self.display_frame['frame_id']:return False
        published={}
        for camera_id,frame in frames.items():
            if int(frame['frame_id'])!=identifier or float(frame['timestamp'])!=timestamp:
                raise ValueError('Live bundle cameras must share one acquisition ID and timestamp')
            published[camera_id]=PublishedRGBDFrame(frame['rgb'],frame['depth'])
        height,width=published['head'].depth.shape
        k=head['intrinsics']
        if isinstance(k,dict):k=[[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]]
        k=np.asarray(k,dtype=float);transform=np.asarray(head['T_world_camera'],dtype=float)
        if (k.shape!=(3,3) or transform.shape!=(4,4) or not np.isfinite(k).all()
                or not np.isfinite(transform).all() or k[0,0]<=0 or k[1,1]<=0):
            raise ValueError('Finite calibrated live-camera intrinsics/extrinsics required')
        camera={'width':width,'height':height,'intrinsics':k.tolist(),
                'T_world_camera':transform.tolist(),'depth_convention':str(head['depth_convention']),
                'source':str(head['source'])}
        descriptor={'frame_id':identifier,'timestamp':timestamp,'camera_id':'head',
                    'camera':camera,'source':'raw_camera_capture'}
        if 'sim_time' in head:
            descriptor['sim_time']=float(head['sim_time'])
            if not np.isfinite(descriptor['sim_time']):raise ValueError('Invalid live simulation time')
        with self.frame_lock:
            if self.display_frame is not None and identifier<=self.display_frame['frame_id']:return False
            for camera_id,value in published.items():
                live=self.live_sensor_frames.setdefault(camera_id,OrderedDict())
                measured=self.frames if camera_id=='head' else self.sensor_frames.get(camera_id,{})
                live[identifier]=measured.get(identifier) or value
                while len(live)>8:live.popitem(last=False)
            # Publish metadata only after every payload is available.
            self.display_frame=descriptor
        return True

    def display_snapshot(self):
        with self.frame_lock:value=copy.deepcopy(self.display_frame)
        if value is not None:value['age_s']=max(0.,self.session.clock()-value['timestamp'])
        return value

    def close(self):
        self.httpd.shutdown();self.httpd.server_close();self.thread.join(timeout=3)


def decode_osc_message(packet):
    def string_at(offset):
        end=packet.find(b"\0",offset)
        if end<0:raise ValueError("Unterminated OSC string")
        return packet[offset:end].decode("utf8"),(end+4)&~3
    address,offset=string_at(0); tags,offset=string_at(offset)
    if not tags.startswith(","):raise ValueError("Invalid OSC type tags")
    values=[]
    for tag in tags[1:]:
        if tag not in "if" or offset+4>len(packet):raise ValueError("Only OSC int32/float32 values are supported")
        values.append(struct.unpack_from(">"+tag,packet,offset)[0]);offset+=4
    if offset!=len(packet):raise ValueError("Unexpected OSC trailing data")
    return address,values


class GazeOSCReceiver:
    """Canonical external adapter; hardware-specific raw mappings need calibration.

    /punktnull/gaze/raw or /punktnull/gaze/uv, type tags ,ffifii:
    x, y, valid, confidence, sequence, displayed_frame_id.
    The /uv provider must already be calibrated to the actual video view.
    """
    def __init__(self,session,host="127.0.0.1",port=9002):
        self.session=session;self.errors=0;self.received=0;self.stop=threading.Event()
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.bind((host,port));self.socket.settimeout(.2)
        self.address=self.socket.getsockname()
        self.thread=threading.Thread(target=self._run,daemon=True,name="assist-eye-osc")
    def start(self):self.thread.start()
    def _run(self):
        while not self.stop.is_set():
            try:
                packet,remote=self.socket.recvfrom(4096)
                address,values=decode_osc_message(packet)
                if address not in ("/punktnull/gaze/raw","/punktnull/gaze/uv") or len(values)!=6:
                    raise ValueError("Unexpected gaze OSC message")
                address_end=packet.find(b'\0')
                tag_offset=(address_end+4)&~3
                if packet[tag_offset:tag_offset+8]!=b',ffifii\0':
                    raise ValueError("Use canonical OSC types ,ffifii")
                x,y,valid,confidence,seq,frame=values
                if valid not in (0,1):raise ValueError("OSC valid must be 0 or 1")
                sample={"source":"external_osc","session_id":"osc:"+str(remote),"seq":int(seq),
                        "valid":bool(valid),"confidence":float(confidence),"frame_id":int(frame)}
                if address.endswith("raw"):sample["raw"]=[x,y]
                else:sample.update(uv=[x,y],calibrated=True)
                if self.session.set_gaze(sample):self.received+=1
            except socket.timeout:continue
            except OSError:
                if not self.stop.is_set():self.errors+=1
            except (ValueError,TypeError,KeyError,UnicodeDecodeError,struct.error):self.errors+=1
    def close(self):
        self.stop.set();self.socket.close();self.thread.join(timeout=2)
