"""Input arbitration, explicit target selection and bounded Cartesian assistance.

No simulator identities or task ground truth enter this module. All distances are
metres, times are host monotonic seconds, and quaternions are XYZW.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import threading
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from .intent import IntentConfig,IntentEstimator
from .actions import ActionSelection, calibrated_actions
from .predictive import PredictiveConfig, PredictiveServo


def vector(value, length, name="vector"):
    a = np.asarray(value, dtype=float)
    if a.shape != (length,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return a


def unit_quaternion(value):
    q = vector(value, 4, "quaternion")
    n = float(np.linalg.norm(q))
    if n < 0.5 or n > 1.5:
        raise ValueError("Invalid quaternion norm")
    return q / n


def limited(v, maximum):
    n = float(np.linalg.norm(v))
    return v if n <= maximum else v * (maximum / n)


def boolean(value, name="flag"):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a JSON boolean")
    return value


XR_TO_ROBOT=np.array([[0.,0.,-1.],[-1.,0.,0.],[0.,1.,0.]])


def rotation_matrix(value):
    r=np.asarray(value,dtype=float)
    if r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1.,atol=1e-5):
        raise ValueError("Input basis must be a proper 3x3 rotation")
    return r


@dataclass
class AssistConfig:
    eye_enabled: bool = False
    assist_enabled: bool = False
    require_action: bool = False
    prediction_horizon: float = .12
    max_orientation_correction: float = .20
    input_timeout: float = 0.30
    frame_timeout: float = 0.70
    gaze_timeout: float = 0.25
    gaze_min_confidence: float = 0.60
    max_speed: float = 0.12
    max_angular_speed: float = 0.6
    max_correction: float = 0.055
    activation_radius: float = 0.22
    standoff: float = 0.14
    hand_scale: float = 1.0
    stick_on: float = 0.65
    stick_off: float = 0.30
    workspace_min: tuple = (0.05, -0.65, 0.65)
    workspace_max: tuple = (0.68, 0.18, 1.45)

    def __post_init__(self):
        for key in ("input_timeout", "frame_timeout", "gaze_timeout", "max_speed",
                    "max_correction", "activation_radius", "hand_scale", "max_angular_speed",
                    "prediction_horizon", "max_orientation_correction"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {key}")
        if not 0 <= self.standoff <= .30:
            raise ValueError("Invalid standoff")
        if not 0 <= self.gaze_min_confidence <= 1 or not 0 <= self.stick_off < self.stick_on <= 1:
            raise ValueError("Invalid confidence or stick thresholds")
        lo, hi = vector(self.workspace_min, 3), vector(self.workspace_max, 3)
        if np.any(lo >= hi):
            raise ValueError("Invalid workspace")


class Selection:
    def __init__(self):
        self.mode = "preview"
        self.id = None
        self.targets = {}

    def update(self, targets):
        self.targets = {t["id"]: t for t in targets}
        if self.id is not None and self.id not in self.targets:
            if self.mode in ("confirmed", "manual", "lost"):
                self.mode = "lost"
            else:
                self.id = None

    def propose(self, target_id):
        if self.mode == "preview" and target_id in self.targets:
            self.id = target_id

    def select(self, target_id):
        if target_id not in self.targets:
            raise ValueError("Target is not currently observed")
        self.id, self.mode = target_id, "manual"

    def navigate(self, direction):
        if direction not in ("up", "down", "left", "right"):
            raise ValueError("Unknown direction")
        if self.mode == "confirmed":
            return False
        if self.id not in self.targets:
            return False
        self.mode = "manual"
        source = self.targets[self.id]["bbox_px"]
        p = np.array([(source[0]+source[2])/2, (source[1]+source[3])/2])
        d = np.array({"up": (0,-1), "down": (0,1), "left": (-1,0), "right": (1,0)}[direction])
        candidates = []
        for target_id, target in self.targets.items():
            if target_id == self.id:
                continue
            b = target["bbox_px"]
            delta = np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])-p
            along = float(delta @ d)
            across = abs(float(delta[0]*d[1]-delta[1]*d[0]))
            if along > 1 and across <= 1.5 * along:
                candidates.append((float(np.linalg.norm(delta))+3*across, target_id))
        if candidates:
            self.id = min(candidates)[1]
            return True
        return False

    def confirm(self):
        if self.id not in self.targets or self.mode == "lost":
            raise ValueError("A visible target must be selected before confirmation")
        self.mode = "confirmed"

    def release(self):
        self.id, self.mode = None, "preview"

    def snapshot(self):
        return {"mode": self.mode, "id": self.id}


class GazeCalibration:
    """Affine mapping for an external tracker; independent validation stays explicit."""
    def __init__(self):
        self.matrix = None
        self.fit_rms = None

    def fit(self, samples):
        if not isinstance(samples, list) or len(samples) < 9 or len(samples) > 200:
            raise ValueError("At least nine calibration samples are required")
        raw = np.array([vector(s["raw"],2) for s in samples])
        uv = np.array([vector(s["uv"],2) for s in samples])
        if np.any(uv < 0) or np.any(uv > 1) or np.any(np.ptp(uv,axis=0) < .5):
            raise ValueError("Calibration must span the displayed image")
        design = np.column_stack([raw, np.ones(len(raw))])
        if np.linalg.matrix_rank(design) != 3 or np.linalg.cond(design) > 1e5:
            raise ValueError("Degenerate calibration samples")
        matrix, _, _, _ = np.linalg.lstsq(design, uv, rcond=None)
        rms = float(np.sqrt(np.mean(np.sum((design @ matrix - uv)**2, axis=1))))
        if rms > .08:
            raise ValueError("Calibration residual is too large")
        self.matrix, self.fit_rms = matrix, rms
        return {"matrix": matrix.tolist(), "fit_rms_uv": rms, "samples": len(samples),
                "independent_validation": "required; fitting error is not gaze accuracy"}

    def transform(self, raw):
        if self.matrix is None:
            return None
        return np.r_[vector(raw,2),1.] @ self.matrix


class AssistSession:
    def __init__(self, config=None, clock=time.monotonic, event_sink=None):
        self.config = config or AssistConfig()
        self.clock = clock
        self.event_sink = event_sink
        self.lock = threading.RLock()
        self.selection = Selection()
        self.action = ActionSelection()
        self.predictive = PredictiveServo(PredictiveConfig(
            horizon_s=self.config.prediction_horizon,
            max_correction_m=self.config.max_correction,
            activation_radius_m=self.config.activation_radius,
            max_orientation_correction_rad=self.config.max_orientation_correction))
        self.rearm_required = self.config.require_action
        self._clutch_released_for_rearm = False
        self.frame_id = -1
        self.frame_time = -math.inf
        self.camera = {}
        self.history = deque(maxlen=16)
        self.perception_diagnostics = {}
        self.input = None
        self.input_time = -math.inf
        self.input_sequences = {}
        self.input_session = None
        self.pointer = [.5,.5]
        self.pointer_frame_id = None
        self.gaze = None
        self.gaze_time = -math.inf
        self.gaze_device=None
        self.gaze_device_time=-math.inf
        self.gaze_device_history=deque(maxlen=32)
        self.gaze_sequences = {}
        self.gaze_calibration = GazeCalibration()
        self.hand_origin = None
        self.robot_origin = None
        self.hand_orientation_origin = None
        self.robot_orientation_origin = None
        self.command_quaternion = None
        self.input_basis = XR_TO_ROBOT.copy()
        self.calibrated_orientation_mode = None
        self.robot = {}
        self.command_position = None
        self._linear_progress_direction = 0
        self._angular_progress_direction = 0
        self.last_grip = 0.
        self.stick_ready = True
        self.last_primary = False
        self.last_secondary = False
        self.stopped = False
        self.stop_requested = False
        self.control = {"enabled": self.config.assist_enabled, "active": False,
                        "reason": "no_input", "correction_m": 0.}
        self.latest_reference = None
        self.intent_estimator=IntentEstimator(IntentConfig(standoff_m=self.config.standoff,target_timeout_s=self.config.frame_timeout))
        self.intent_ranked=[]
        self.intent_time=-math.inf

    def emit(self, kind, **fields):
        if self.event_sink:
            self.event_sink({"event":kind,"timestamp":self.clock(),**fields})

    def _reset_prediction(self, fresh_clutch=True):
        self.predictive.reset()
        self._linear_progress_direction = 0
        self._angular_progress_direction = 0
        self.intent_estimator.reset()
        self.intent_ranked=[]
        if fresh_clutch and self.config.require_action:
            self.rearm_required=True
            self._clutch_released_for_rearm=False
            self.command_position=None

    def _refresh_action(self):
        before=self.action.valid
        target=self.selection.targets.get(self.selection.id)
        self.action.update(target,self.selection.mode=="confirmed")
        calibration=(target or {}).get("task_calibration",{})
        if calibration.get("source")=="cad_registration":
            timestamp=float(calibration.get("pose_timestamp",-math.inf))
            max_age=float(calibration.get("pose_max_age_s",.5))
            age=self.clock()-timestamp
            if not math.isfinite(age) or not math.isfinite(max_age) or max_age<=0 or age<-.01 or age>max_age:
                self.action.valid=False
                self.action.reason="cad_pose_stale"
        if before and not self.action.valid and self.config.assist_enabled:
            self._reset_prediction()

    def _release_selection(self):
        self.selection.release()
        self.action.reset()
        self._reset_prediction()

    def _confirm_selection(self):
        if self.selection.mode!="confirmed":
            self.selection.confirm()
            self.action.reset()
            self._refresh_action()
            self._reset_prediction()
            self.emit("target_confirmed",target_id=self.selection.id)

    def _navigate_selection(self,direction):
        self._refresh_action()
        if self.config.require_action and self.selection.mode=="confirmed":
            changed=self.action.navigate(direction)
        else:
            changed=self.selection.navigate(direction)
        if changed:
            self._reset_prediction()
        return changed

    def update_frame(self, targets, frame_id, timestamp, camera, diagnostics=None):
        with self.lock:
            if not math.isfinite(timestamp) or timestamp > self.clock()+.01:
                raise ValueError("Invalid frame timestamp")
            if frame_id <= self.frame_id:
                return False
            clean=[]
            ids=set()
            for target in targets:
                t=dict(target)
                if not isinstance(t.get("id"),str) or t["id"] in ids:
                    raise ValueError("Perception targets require unique string identities")
                ids.add(t["id"])
                box=vector(t["bbox_px"],4); vector(t["center_world"],3); normal=vector(t["normal_world"],3)
                if box[2]<=box[0] or box[3]<=box[1] or np.linalg.norm(normal)<1e-6:
                    raise ValueError("Invalid target box or surface normal")
                if t.get("stale",False):
                    continue
                calibrated_actions(t)
                clean.append(t)
            old=self.selection.snapshot()
            self.selection.update(clean)
            self._refresh_action()
            self.frame_id=int(frame_id); self.frame_time=float(timestamp)
            self.camera=dict(camera)
            self.perception_diagnostics=diagnostics or {}
            self.history.append((self.frame_id,self.frame_time,clean,dict(camera)))
            if self.selection.mode=="lost" and old["mode"]!="lost":
                self.emit("target_lost",target_id=old["id"])
            self._propose(self.clock())
            return True

    def _pick(self, uv, frame_id, now):
        if frame_id is None:
            frame_id=self.frame_id
        entry=next((e for e in reversed(self.history) if e[0]==frame_id),None)
        if entry is None or now-entry[1]>self.config.frame_timeout:
            return None
        p=vector(uv,2)
        if np.any(p<0) or np.any(p>1):
            return None
        _,_,targets,camera=entry
        px=p*np.array([camera["width"],camera["height"]])
        ranked=[]
        for t in targets:
            if t["id"] not in self.selection.targets:
                continue
            x0,y0,x1,y1=t["bbox_px"]
            outside=np.array([max(x0-px[0],0,px[0]-x1),max(y0-px[1],0,px[1]-y1)])
            center=np.array([(x0+x1)/2,(y0+y1)/2])
            score=float(np.linalg.norm(outside)+.08*np.linalg.norm(px-center))
            ranked.append((score,t["id"]))
        return min(ranked)[1] if ranked else None

    def eye_status(self, now):
        if not self.config.eye_enabled:
            return "disabled"
        if self.gaze is None:
            return "projection_required" if self.gaze_device is not None else "no_device"
        if self.gaze.get('source')=='webxr_gaze_projection':
            if (self.gaze_device is None or now-self.gaze_device_time>self.config.gaze_timeout
                    or now-self.gaze.get('projection_device_received_at',-math.inf)>self.config.gaze_timeout):
                return 'stale'
            if (not self.gaze_device.get('valid',False) or not self.gaze_device.get('calibrated',False)
                    or self.gaze.get('projection_of_session')!=self.gaze_device.get('session_id')):
                return 'projection_required'
        if now-self.gaze_time>self.config.gaze_timeout:
            return "stale"
        if not self.gaze.get("valid",False):
            return "invalid"
        if not self.gaze.get("calibrated",False):
            return "calibration_required"
        if self.gaze.get("confidence",0)<self.config.gaze_min_confidence:
            return "low_confidence"
        if self._pick(self.gaze["uv"],self.gaze.get("frame_id"),now) is None:
            return "frame_unavailable"
        return "active"

    def _propose(self, now):
        if self.selection.mode!="preview":
            return
        if self.eye_status(now)=="active":
            target_id=self._pick(self.gaze["uv"],self.gaze.get("frame_id"),now)
        else:
            target_id=self._pick(self.pointer,self.pointer_frame_id,now)
        if (self.intent_ranked and now-self.intent_time<=self.config.input_timeout
                and self.intent_ranked[0]['suggestion_eligible']
                and self.intent_ranked[0]['hand_weight']>.35):
            target_id=self.intent_ranked[0]['id']
        self.selection.propose(target_id)

    def _update_intent(self, now):
        if self.hand_origin is None or not self.input or not self.input.get('right',{}).get('valid',False):
            self.intent_ranked=[];self.intent_estimator.reset();return
        if not np.allclose(self.input.get('xr_to_world_rotation',XR_TO_ROBOT),self.input_basis,atol=1e-5):
            self.intent_ranked=[];self.intent_estimator.reset();return
        raw=self.robot_origin+self.config.hand_scale*(self.input_basis@(vector(self.input['right']['position'],3)-self.hand_origin))
        pointer={'uv':self.pointer,'confidence':1.,'valid':self.pointer_frame_id in (None,self.frame_id) and self._pick(self.pointer,self.pointer_frame_id,now) is not None}
        if self.eye_status(now)=='active':
            pointer={'uv':self.gaze['uv'],'confidence':self.gaze['confidence'],'valid':self.gaze.get('frame_id') in (None,self.frame_id)}
        self.intent_ranked=self.intent_estimator.update(list(self.selection.targets.values()),raw,now,pointer,self.camera)
        self.intent_time=now

    def set_gaze(self, sample):
        with self.lock:
            now=self.clock(); data=dict(sample)
            source=str(data.get("source","external"))[:80]
            seq=int(data.get("seq",0))
            key=str(data.get("session_id",source))
            if seq<=self.gaze_sequences.get(key,-1):
                return False
            data["valid"]=boolean(data.get("valid",False),"gaze valid")
            confidence=float(data.get('confidence',0))
            if not math.isfinite(confidence) or not 0<=confidence<=1:
                raise ValueError('Invalid gaze confidence')
            if 'head_ray' in data:
                ray=data['head_ray']
                origin=vector(ray['origin'],3,'head ray origin')
                direction=vector(ray['direction'],3,'head ray direction')
                norm=float(np.linalg.norm(direction))
                if not .9<=norm<=1.1 or np.linalg.norm(origin)>.25:
                    raise ValueError('Head-local gaze needs a unit ray and a plausible eye origin in metres')
                data.update(head_ray={'origin':origin.tolist(),'direction':(direction/norm).tolist()},
                            calibrated=boolean(data.get('calibrated',False),'gaze calibrated'),
                            seq=seq,source=source,session_id=key,confidence=confidence)
                self.gaze_sequences[key]=seq
                self.gaze_device=data;self.gaze_device_time=now
                self.gaze_device_history.append((key,seq,now,data))
                self.emit('gaze_device_sample',sample=data)
                return True
            if source=='webxr_gaze_projection':
                matched=next((entry for entry in reversed(self.gaze_device_history)
                              if entry[0]==data.get('projection_of_session') and entry[1]==data.get('projection_of_seq')),None)
                if (matched is None or self.gaze_device is None or now-matched[2]>self.config.gaze_timeout
                        or not self.gaze_device.get('valid',False) or not self.gaze_device.get('calibrated',False)
                        or not matched[3].get('valid',False) or not matched[3].get('calibrated',False)
                        or data.get('projection_of_session')!=self.gaze_device.get('session_id')):
                    raise ValueError('Projection requires the current fresh calibrated device ray')
                confidence=min(confidence,matched[3]['confidence'])
                data['projection_device_received_at']=matched[2]
            if data["valid"] and "raw" not in data and "uv" not in data:
                raise ValueError("Valid gaze requires raw or uv coordinates")
            if "raw" in data:
                uv=self.gaze_calibration.transform(data["raw"])
                data["calibrated"]=uv is not None
                data["uv"]=uv.tolist() if uv is not None else [.5,.5]
            data["uv"]=vector(data.get("uv",[.5,.5]),2).tolist()
            data["calibrated"]=boolean(data.get("calibrated",False),"gaze calibrated")
            data.update(source=source,seq=seq,confidence=confidence)
            self.gaze_sequences[key]=seq
            self.gaze=data; self.gaze_time=now
            self.emit('gaze_sample',sample=data)
            self._propose(now)
            return True

    def clear_operator_reference(self, reason):
        """Invalidate a departed browser's input before an explicit handover.

        Called under the same lock as HTTP ownership changes. A new operator
        must send released input and rearm; no prior clutch/pose is inherited.
        """
        with self.lock:
            self.input = None
            self.input_time = -math.inf
            self.input_session = None
            self.hand_origin = None
            self.robot_origin = None
            self.command_position = None
            self.last_primary = self.last_secondary = self.stick_ready = False
            self._reset_prediction()
            self.emit('operator_reference_cleared', reason=reason)

    def receive_input(self, sample):
        with self.lock:
            data=dict(sample); now=self.clock()
            source=str(data.get("source","unknown"))[:80]
            session_id=str(data.get("session_id",source))[:160]
            sequence_key=str(data.get('client_id',session_id))[:160]
            seq=int(data.get("seq",0))
            if seq<=self.input_sequences.get(sequence_key,-1):
                return False
            right=dict(data.get("right",{}))
            right["valid"]=boolean(right.get("valid",False),"right valid")
            if right["valid"]:
                right["position"]=vector(right["position"],3).tolist()
                right["quaternion_xyzw"]=unit_quaternion(right.get("quaternion_xyzw",[0,0,0,1])).tolist()
            axes=vector(data.get("axes",[0,0]),2)
            if np.any(np.abs(axes)>1.01):
                raise ValueError("Invalid stick axes")
            for key in ("trigger","squeeze"):
                value=float(data.get(key,0))
                if not math.isfinite(value) or not 0<=value<=1:
                    raise ValueError("Invalid "+key)
                data[key]=value
            for key in ("primary","secondary","clutch"):
                data[key]=boolean(data.get(key,False),key)
            if "pointer_uv" in data:
                data["pointer_uv"]=vector(data["pointer_uv"],2).tolist()
            mode=data.get("orientation_mode","position")
            if mode not in ("position","pose"):
                raise ValueError("orientation_mode must be position or pose")
            data["orientation_mode"]=mode
            if "xr_to_world_rotation" in data:
                data["xr_to_world_rotation"]=rotation_matrix(data["xr_to_world_rotation"]).tolist()
            if self.input_session!=session_id:
                self.hand_origin=None; self.robot_origin=None; self.command_position=None
                self.input_session=session_id
                self.last_primary=False; self.last_secondary=False; self.stick_ready=False
                self.emit("input_session_changed",source=source,session_id=session_id)
                self._reset_prediction()
            self.input_sequences[sequence_key]=seq
            data.update(right=right,source=source,session_id=session_id,seq=seq,axes=axes.tolist())
            self.input=data; self.input_time=now
            if "pointer_uv" in data:
                self.pointer=vector(data["pointer_uv"],2).tolist()
                self.pointer_frame_id=data.get("frame_id",self.frame_id)
            if float(np.max(np.abs(axes)))<self.config.stick_off:
                self.stick_ready=True
            elif self.stick_ready and float(np.max(np.abs(axes)))>=self.config.stick_on:
                direction=("right" if axes[0]>0 else "left") if abs(axes[0])>abs(axes[1]) else ("down" if axes[1]>0 else "up")
                self._navigate_selection(direction); self.stick_ready=False
                self.emit("joystick_selection",direction=direction,selection=self.selection.snapshot())
            primary=bool(data.get("primary",False)); secondary=bool(data.get("secondary",False))
            if secondary and not self.last_secondary:
                self._release_selection()
            elif primary and not self.last_primary:
                try:
                    if self.config.require_action and self.selection.mode=="confirmed":
                        if not self.action.confirmed:
                            self.action.confirm(); self._reset_prediction()
                            self.emit("action_confirmed",target_id=self.selection.id,action_id=self.action.id)
                    else:
                        self._confirm_selection()
                except ValueError:
                    pass
            self.last_primary,self.last_secondary=primary,secondary
            if self.config.require_action:
                clutch=data["clutch"] or data["squeeze"]>.5
                if not clutch:
                    self.predictive.reset()
                    self._clutch_released_for_rearm=True
                    self.rearm_required=True
                elif self.rearm_required and self._clutch_released_for_rearm and not self.stopped:
                    ready=not self.config.assist_enabled or (self.selection.mode=="confirmed" and self.action.confirmed and self.action.valid)
                    if ready and self.robot.get("valid",False) and right["valid"]:
                        self.calibrate_hand()
                        self.rearm_required=False
                        self._clutch_released_for_rearm=False
                        self.predictive.prime(self.robot_origin,now,self.robot_orientation_origin)
                        self.action.latch_execution_origin(self.robot_origin)
            self._update_intent(now)
            self._propose(now)
            return True

    def calibrate_hand(self):
        with self.lock:
            if self.input is None or self.clock()-self.input_time>self.config.input_timeout:
                raise ValueError("Fresh controller input required for calibration")
            if not self.input.get("right",{}).get("valid",False):
                raise ValueError("Valid controller pose required")
            if not self.robot.get("valid",False):
                raise ValueError("Valid measured robot state required")
            self.hand_origin=vector(self.input["right"]["position"],3)
            self.robot_origin=vector(self.robot["ee_position"],3)
            self.input_basis=rotation_matrix(self.input.get("xr_to_world_rotation",XR_TO_ROBOT))
            self.hand_orientation_origin=unit_quaternion(self.input["right"].get("quaternion_xyzw",[0,0,0,1]))
            self.robot_orientation_origin=unit_quaternion(self.robot.get("ee_quaternion",[0,0,0,1]))
            self.command_quaternion=self.robot_orientation_origin.copy()
            self.calibrated_orientation_mode=self.input.get("orientation_mode","position")
            self.command_position=self.robot_origin.copy()
            self.intent_estimator.reset();self.intent_ranked=[]
            self.predictive.reset()
            self.emit("hand_calibrated",source=self.input["source"],ee_position=self.robot_origin.tolist())
            return {"calibrated":True,"robot_origin":self.robot_origin.tolist()}

    def command(self, data):
        with self.lock:
            kind=data.get("type")
            self.emit("command",data=data)
            if kind=="input":
                return {"accepted":self.receive_input(data.get("sample",data))}
            if kind=="gaze":
                return {"accepted":self.set_gaze(data.get("sample",data))}
            if kind=="direction":
                changed=self._navigate_selection(data.get("direction")); self.emit("manual_selection",selection=self.selection.snapshot())
                return {"changed":changed}
            if kind=="select":
                if self.config.require_action and self.selection.mode in ("confirmed","lost") and data["id"]!=self.selection.id:
                    raise ValueError("Release the locked object before choosing another")
                if self.config.require_action and self.selection.mode=="confirmed" and data["id"]==self.selection.id:
                    return {"accepted":True,"selection":self.selection.snapshot(),
                            "action":self.action.snapshot(self.config.require_action,self.rearm_required)}
                self.selection.select(data["id"])
                self.action.reset(); self._reset_prediction()
            elif kind=="confirm":
                self._confirm_selection()
            elif kind=="release":
                self._release_selection(); self._propose(self.clock())
            elif kind=="action_select":
                self._refresh_action()
                self.action.select(data.get("action_id")); self._reset_prediction()
            elif kind=="action_confirm":
                self._refresh_action()
                self.action.confirm(); self._reset_prediction()
                self.emit("action_confirmed",target_id=self.selection.id,action_id=self.action.id)
            elif kind=="action_phase":
                if self.input and (self.input.get("clutch",False) or self.input.get("squeeze",0)>.5):
                    raise ValueError("Release the deadman before changing action phase")
                self._refresh_action()
                origin=self.robot.get("ee_position") if self.robot.get("valid",False) else None
                self.action.set_phase(data.get("phase"),data.get("engagement_confirmed",False),origin)
                self._reset_prediction()
                self.emit("action_phase",action_id=self.action.id,phase=self.action.phase,engagement_status=self.action.engagement_status)
            elif kind=="pointer":
                self.pointer=vector(data["uv"],2).tolist(); self.pointer_frame_id=data.get("frame_id",self.frame_id)
                self._propose(self.clock())
            elif kind=="eye_enable":
                self.config.eye_enabled=boolean(data["enabled"],"enabled")
                self._propose(self.clock()); self.emit("eye_enabled",enabled=self.config.eye_enabled)
            elif kind=="assist_enable":
                enabled=boolean(data["enabled"],"enabled")
                if enabled!=self.config.assist_enabled:self._reset_prediction()
                self.config.assist_enabled=enabled
                self.emit("assist_enabled",enabled=self.config.assist_enabled)
            elif kind=="calibrate":
                if "sample" in data:
                    self.receive_input(data["sample"])
                result=self.calibrate_hand()
                self._reset_prediction()
                return result
            elif kind=="eye_calibrate":
                result=self.gaze_calibration.fit(data["samples"])
                self.emit("eye_calibration_fit",**result); return result
            elif kind=="stop":
                self.stopped=True; self._reset_prediction(); self.emit("operator_stop")
            elif kind=="resume":
                if self.input and (self.input.get("clutch",False) or self.input.get("squeeze",0)>.5):
                    raise ValueError("Release the deadman before resuming")
                self.stopped=False; self.hand_origin=None; self.command_position=None
                self._reset_prediction()
            elif kind=="shutdown":
                self.stopped=True; self.stop_requested=True; self._reset_prediction()
            else:
                raise ValueError("Unknown command type")
            self._refresh_action()
            return {"accepted":True,"selection":self.selection.snapshot(),
                    "action":self.action.snapshot(self.config.require_action,self.rearm_required)}

    def step(self, robot_state, dt, now=None):
        with self.lock:
            now=self.clock() if now is None else now
            if not math.isfinite(dt) or dt<0 or not math.isfinite(now) or now<0:
                raise ValueError("Control time and dt must be finite and nonnegative")
            self.robot=dict(robot_state)
            self._refresh_action()
            ee=vector(robot_state.get("ee_position",[.25,-.25,1.]),3)
            if self.command_position is None:
                self.command_position=ee.copy()
            reason=None
            if self.stopped: reason="stopped"
            elif not robot_state.get("valid",False): reason="robot_invalid"
            elif float(robot_state.get("joint_limit_violation",0))>.035: reason="joint_limit"
            elif self.input is None: reason="no_input"
            elif now-self.input_time>self.config.input_timeout: reason="input_stale"
            elif not self.input.get("right",{}).get("valid",False): reason="tracking_invalid"
            elif self.config.require_action and self.config.assist_enabled and self.selection.mode!="confirmed": reason="target_confirmation_required" if self.selection.mode!="lost" else "target_lost"
            elif self.config.require_action and self.config.assist_enabled and (not self.action.confirmed or not self.action.valid): reason=self.action.reason
            elif self.config.require_action and self.rearm_required: reason="fresh_clutch_required"
            elif self.hand_origin is None: reason="calibration_required"
            elif self.input.get("orientation_mode","position")!=self.calibrated_orientation_mode: reason="calibration_required"
            elif not np.allclose(self.input.get('xr_to_world_rotation',XR_TO_ROBOT),self.input_basis,atol=1e-5): reason="calibration_required"
            elif not (self.input.get("clutch",False) or self.input.get("squeeze",0)>.5): reason="deadman_released"
            elif self.config.assist_enabled and self.selection.mode=="lost": reason="target_lost"
            if reason:
                if self.config.require_action and reason not in ("fresh_clutch_required","deadman_released") and not self.rearm_required:
                    self._reset_prediction()
                self.command_position=ee.copy()
                self.command_quaternion=unit_quaternion(robot_state.get("ee_quaternion",[0,0,0,1]))
                reference={"ee_target_world":ee.tolist(),"ee_quat_xyzw":None,"grip":self.last_grip,"hold":True}
                self.control={"enabled":self.config.assist_enabled,"active":False,"reason":reason,"correction_m":0.}
                self.latest_reference=reference
                return reference
            # OpenXR x-right/y-up/z-back -> robot X-forward/Y-left/Z-up.
            raw=self.robot_origin+self.config.hand_scale*(self.input_basis @ (vector(self.input["right"]["position"],3)-self.hand_origin))
            raw=np.clip(raw,self.config.workspace_min,self.config.workspace_max)
            correction=np.zeros(3); active=False
            orientation_correction=np.zeros(3)
            raw_quaternion=None
            if self.calibrated_orientation_mode=="pose":
                hand_delta=Rotation.from_quat(self.input["right"]["quaternion_xyzw"])*Rotation.from_quat(self.hand_orientation_origin).inv()
                basis=Rotation.from_matrix(self.input_basis)
                raw_quaternion=(basis*hand_delta*basis.inv()*Rotation.from_quat(self.robot_orientation_origin)).as_quat()
            reason="manual_control"
            contact=bool(robot_state.get('contact_active',robot_state.get('collision_stop',False)))
            contact_desired=None
            if contact:
                normal=vector(robot_state.get('contact_force_world',[0,0,0]),3)
                magnitude=float(np.linalg.norm(normal))
                if magnitude>1e-6:normal/=magnitude
                away=float((raw-ee)@normal) if magnitude>1e-6 else 0.
                self.last_grip=min(self.last_grip,float(self.input.get('trigger',0)))
                if away<=.001:
                    self.predictive.reset()
                    self.command_position=ee.copy()
                    reference={'ee_target_world':ee.tolist(),'ee_quat_xyzw':None,'grip':self.last_grip,'hold':True}
                    self.control={'enabled':self.config.assist_enabled,'active':False,'reason':'collision','correction_m':0.}
                    self.latest_reference=reference
                    return reference
                # Only explicit withdrawal along the measured outward force is
                # accepted. Drop pending inward/tangential arm references.
                self.command_position=ee+normal*max(0.,float((self.command_position-ee)@normal))
                contact_desired=ee+normal*min(away,.05)
                reason='contact_retreat'
            target=self.selection.targets.get(self.selection.id)
            if not contact and self.config.assist_enabled and self.selection.mode=="confirmed" and target is not None:
                if now-self.frame_time>self.config.frame_timeout:
                    reason="perception_stale"
                    self.command_position=ee.copy()
                    self.control={"enabled":True,"active":False,"reason":reason,"correction_m":0.}
                    reference={"ee_target_world":ee.tolist(),"ee_quat_xyzw":None,"grip":self.last_grip,"hold":True}
                    self.latest_reference=reference
                    if self.config.require_action and not self.rearm_required:self._reset_prediction()
                    return reference
                if self.config.require_action:
                    prediction=self.predictive.evaluate(raw,self.input_time,now,self.action.current(),self.action.phase,target,raw_quaternion)
                    correction=prediction["correction"]
                    orientation_correction=prediction["orientation_correction"]
                    active=prediction["active"];reason=prediction["reason"]
                    if prediction["hold"]:
                        self.command_position=ee.copy()
                        self.command_quaternion=unit_quaternion(robot_state.get("ee_quaternion",[0,0,0,1]))
                        self._linear_progress_direction=0
                        self._angular_progress_direction=0
                        self.last_grip=float(self.input.get("trigger",0))
                        reference={"ee_target_world":ee.tolist(),"ee_quat_xyzw":None,"grip":self.last_grip,"hold":True,
                                   "raw_operator_target":raw.tolist(),"corrected_target":ee.tolist(),"correction":[0.,0.,0.]}
                        self.control={"enabled":True,"active":False,"reason":reason,"correction_m":0.,
                                      "prediction":dict(self.predictive.diagnostics)}
                        self.latest_reference=reference
                        if reason!="operator_stationary" and not self.rearm_required:self._reset_prediction()
                        return reference
                else:
                    normal=vector(target["normal_world"],3)
                    normal=normal/max(float(np.linalg.norm(normal)),1e-9)
                    goal=vector(target["center_world"],3)+self.config.standoff*normal
                    error=goal-raw
                    if np.linalg.norm(error)<=self.config.activation_radius:
                        # Legacy baseline: preserve operator approach/retraction.
                        correction=limited(error-normal*float(error@normal),self.config.max_correction)
                        active=float(np.linalg.norm(correction))>1e-5
                        reason="lateral_alignment" if active else "aligned"
                    else:
                        reason="outside_assist_region"
            desired=np.clip(contact_desired if contact else raw+correction,self.config.workspace_min,self.config.workspace_max)
            bounded_dt=max(0.,min(float(dt),.05))
            execution=None
            reset_pending=False
            if self.config.require_action and self.config.assist_enabled and not contact:
                diagnostics=self.predictive.diagnostics
                if "progress_axis_world" in diagnostics:
                    progress_axis=np.asarray(diagnostics["progress_axis_world"])
                    velocity=diagnostics["operator_progress_velocity_m_s"]
                    pending=float((self.command_position-ee)@progress_axis)
                    threshold=self.predictive.config.progress_zero_velocity_m_s
                    direction=1 if velocity>threshold else -1 if velocity<-threshold else 0
                    if self._linear_progress_direction and direction!=self._linear_progress_direction:
                        # Only an operator stop/reversal flushes the whole
                        # backend queue. A measured arm oscillating across its
                        # reference is not a new operator direction: resetting
                        # all axes there repeatedly starves transverse motion.
                        self.command_position=ee.copy()
                        reset_pending=True
                    elif not direction or direction*pending<0:
                        # Remove forbidden axial backlog while retaining the
                        # orthogonal tracking error needed by the physical
                        # servo. This is a reference projection, not a stop.
                        self.command_position-=progress_axis*pending
                    self._linear_progress_direction=direction
                    if velocity*float((desired-ee)@progress_axis)<0:
                        # The absolute hand goal can remain beyond the measured
                        # arm after reversal. Integrate the newly requested
                        # direction immediately instead of waiting for the hand
                        # to traverse the accumulated servo tracking error.
                        progress=float((self.command_position-ee)@progress_axis)
                        progress+=float(np.clip(velocity,-self.config.max_speed,self.config.max_speed))*bounded_dt
                        desired=desired+progress_axis*(progress-float((desired-ee)@progress_axis))
            if self.config.require_action and self.config.assist_enabled and not contact and self.action.phase=="execute":
                capability=self.action.current()
                if capability and "execution_travel_limit_m" in capability:
                    self.action.latch_execution_origin(ee)
                    origin=np.asarray(self.action.execution_origin_world)
                    axis=np.asarray(self.action.execution_axis_world)
                    limit=self.action.execution_travel_limit_m
                    progress=float((ee-origin)@axis)
                    outside_envelope=abs(progress)>limit+1e-8
                    if outside_envelope and not self.control.get("execution",{}).get("outside_envelope",False):
                        # External disturbance / plant overshoot must not turn
                        # the envelope projection into an autonomous snap back.
                        self.command_position=ee.copy();reset_pending=True
                    velocity=float(np.asarray(self.predictive.diagnostics.get("operator_velocity_m_s",[0,0,0]))@axis)
                    pending=float((self.command_position-ee)@axis)
                    if (abs(velocity)<self.predictive.config.progress_zero_velocity_m_s and abs(pending)>1e-6) or velocity*pending<-1e-9:
                        # The direction transition above already flushes a
                        # real stop/reversal. Do not erase transverse tracking
                        # because measured dynamics crossed this axial goal.
                        self.command_position-=axis*pending
                    command_progress=float((self.command_position-origin)@axis)
                    requested_step=float(np.clip(velocity,-self.config.max_speed,self.config.max_speed))*bounded_dt
                    wanted_progress=command_progress+requested_step
                    # Outside the nominal envelope, freeze further outward
                    # progress but let explicit reverse input recover gradually.
                    recovery_lower=min(-limit,progress)
                    recovery_upper=max(limit,progress)
                    allowed_progress=float(np.clip(wanted_progress,recovery_lower,recovery_upper))
                    allowed_step=allowed_progress-command_progress
                    # Integrate fresh operator progress against a fixed measured
                    # engagement origin. A moving object pose cannot renew the
                    # 40 mm envelope, and a reversed hand has no queued catch-up.
                    desired=desired+axis*(allowed_progress-float((desired-origin)@axis))
                    execution={"origin_world":origin.tolist(),"axis_world":axis.tolist(),"limit_m":limit,
                               "progress_m":progress,"operator_step_m":requested_step,"allowed_step_m":allowed_step,
                               "command_progress_m":command_progress,"requested_progress_m":allowed_progress,
                               "outside_envelope":outside_envelope,
                               "recovery_lower_m":recovery_lower,"recovery_upper_m":recovery_upper,
                               "at_limit":abs(allowed_step-requested_step)>1e-10}
                    if execution["at_limit"]:reason="execution_travel_limit"
            next_position=self.command_position+limited(desired-self.command_position,self.config.max_speed*bounded_dt)
            # A controller cannot accumulate a distant goal against a blocked arm.
            next_position=ee+limited(next_position-ee,.075)
            if self.config.require_action and self.config.assist_enabled and not contact:
                # Reference speed is bounded against the previous command,
                # while the measured arm may lag by at most 75 mm. Bounding
                # every reference against measured EE by velocity*dt would
                # starve a resolved-rate physical plant of tracking error.
                diagnostics=self.predictive.diagnostics
                if "progress_axis_world" in diagnostics:
                    progress_axis=np.asarray(diagnostics["progress_axis_world"])
                    progress_velocity=diagnostics["operator_progress_velocity_m_s"]
                    delta=next_position-ee
                    requested_progress=float(delta@progress_axis)
                    progress=(max(0.,requested_progress) if progress_velocity>self.predictive.config.progress_zero_velocity_m_s else
                              min(0.,requested_progress) if progress_velocity<-self.predictive.config.progress_zero_velocity_m_s else 0.)
                    next_position=next_position+progress_axis*(progress-requested_progress)
            if execution is not None:
                axis=np.asarray(execution["axis_world"])
                origin=np.asarray(execution["origin_world"])
                progress=float((next_position-origin)@axis)
                bounded=float(np.clip(progress,execution["recovery_lower_m"],execution["recovery_upper_m"]))
                next_position=next_position+axis*(bounded-progress)
            self.command_position=next_position
            if not contact:self.last_grip=float(self.input.get("trigger",0))
            quaternion=None
            if self.calibrated_orientation_mode=="pose" and not contact:
                desired_rotation=Rotation.from_rotvec(orientation_correction)*Rotation.from_quat(raw_quaternion)
                previous=Rotation.from_quat(self.command_quaternion)
                measured=Rotation.from_quat(unit_quaternion(robot_state.get("ee_quaternion",[0,0,0,1])))
                diagnostics=self.predictive.diagnostics
                angular_progress=(self.config.require_action and self.config.assist_enabled and
                                  "angular_progress_axis_world" in diagnostics)
                if angular_progress:
                    axis=np.asarray(diagnostics["angular_progress_axis_world"])
                    velocity=diagnostics["operator_angular_progress_rad_s"]
                    pending=float((previous*measured.inv()).as_rotvec()@axis)
                    threshold=self.predictive.config.progress_zero_angular_velocity_rad_s
                    direction=1 if velocity>threshold else -1 if velocity<-threshold else 0
                    if self._angular_progress_direction and direction!=self._angular_progress_direction:
                        previous=measured
                        reset_pending=True
                    elif not direction or direction*pending<0:
                        error=(previous*measured.inv()).as_rotvec()
                        previous=Rotation.from_rotvec(error-axis*pending)*measured
                    self._angular_progress_direction=direction
                    error=(desired_rotation*measured.inv()).as_rotvec()
                    if velocity*float(error@axis)<0:
                        # As for translation, a reversed operator may still be
                        # ahead of the delayed robot. Integrate reverse motion
                        # from the flushed command instead of waiting to unwind
                        # the preceding orientation tracking error by hand.
                        progress=float((previous*measured.inv()).as_rotvec()@axis)
                        progress+=float(np.clip(velocity,-self.config.max_angular_speed,self.config.max_angular_speed))*bounded_dt
                        desired_rotation=Rotation.from_rotvec(error+axis*(progress-float(error@axis)))*measured
                delta=(desired_rotation*previous.inv()).as_rotvec()
                next_rotation=Rotation.from_rotvec(limited(delta,self.config.max_angular_speed*bounded_dt))*previous
                # Keep useful orientation tracking error for the physical servo
                # while bounding measured-to-command lead to 0.25 radians.
                next_rotation=Rotation.from_rotvec(limited((next_rotation*measured.inv()).as_rotvec(),.25))*measured
                if angular_progress:
                    delta=(next_rotation*measured.inv()).as_rotvec()
                    requested=float(delta@axis)
                    progress=(max(0.,requested) if velocity>threshold else min(0.,requested) if velocity<-threshold else 0.)
                    next_rotation=Rotation.from_rotvec(delta+axis*(progress-requested))*measured
                self.command_quaternion=next_rotation.as_quat()
                quaternion=self.command_quaternion.tolist()
            reference={"ee_target_world":next_position.tolist(),"ee_quat_xyzw":quaternion,
                       "grip":self.last_grip,"hold":False,"raw_operator_target":raw.tolist(),
                       "corrected_target":desired.tolist(),"correction":correction.tolist(),
                       "reset_pending":reset_pending,
                       "orientation_correction_rotvec":orientation_correction.tolist()}
            self.control={"enabled":self.config.assist_enabled,"active":active,"reason":reason,
                          "correction_m":float(np.linalg.norm(correction)),"max_speed_m_s":self.config.max_speed,
                          "orientation_correction_rad":float(np.linalg.norm(orientation_correction)),
                          "prediction":dict(self.predictive.diagnostics)}
            if execution is not None:self.control["execution"]=execution
            self.latest_reference=reference
            return reference

    def snapshot(self):
        with self.lock:
            now=self.clock(); self._propose(now)
            self._refresh_action()
            return {"schema_version":1,"mode":"fixed_base_manipulation_fixture",
                    "timestamp":now,"frame_id":self.frame_id,"frame_age":now-self.frame_time if self.frame_id>=0 else None,
                    "camera":dict(self.camera),"targets":list(self.selection.targets.values()),
                    "selection":self.selection.snapshot(),"eye_enabled":self.config.eye_enabled,
                    "action":self.action.snapshot(self.config.require_action,self.rearm_required),
                    "eye_status":self.eye_status(now),"input_source":self.input.get("source") if self.input else "none",
                    "input_age":now-self.input_time if self.input else None,"hand_calibrated":self.hand_origin is not None,
                    "robot":dict(self.robot),"assist":dict(self.control),"stopped":self.stopped,
                    "intent":{"ranked":self.intent_ranked[:3],"diagnostics":dict(self.intent_estimator.diagnostics)},
                    "perception":dict(self.perception_diagnostics),"settings":asdict(self.config)}
