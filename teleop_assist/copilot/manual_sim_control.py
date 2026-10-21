"""Manual Unitree IK path for a local Isaac articulation; no robot transport.

ABXY/AX and relative two-arm calibration are UI adaptations. IK is the original
continuous Unitree solver/filter. No custom collision/path/receipt guard runs.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from tools.copilot.xr_source import validate_operator_packet
from .controller import rigid
from .kinematics import arm_vector


class ManualSimControl:
    def __init__(self, worker, *, simulation_mode, target_filter=None):
        if simulation_mode is not True:
            raise ValueError('Manual baseline is available only in isolated simulation')
        self.worker = worker
        self.target_filter = target_filter
        self.anchors = None
        self.active = False
        self.revision = None
        self.sequence = None
        self.epoch = None
        self.source_stamp = self.source_sequence = None
        self.missing_since = None
        self.recovery_targets = None
        self.recovery_stamp = None
        self.recovery_speed_m_s = .20
        self.recovery_speed_rad_s = 1.0
        self.status = self._status('clutch_required')

    def _status(self, reason, *, allowed=False):
        self.status = dict(schema='punktnull.sim_runtime.v1', mode='isolated_PhysX',
            control_mode='unitree_manual', reason=reason, allowed=allowed,
            custom_motion_guard_enabled=False, contact_stop_enabled=False, assistance_applied=False,
            input_hold_required=not allowed,
            operator_source_sequence=self.source_sequence,operator_acquisition_timestamp=self.source_stamp,
            robot_transport_started=False, real_robot_authorized=False,
            snapshot=dict(clutched=self.anchors is not None,
                clutch_required=self.anchors is None, clutch_available=True,
                teleop_active=self.active, teleop_start_required=self.anchors is not None and not self.active,
                last_ui_revision=self.revision, confirmed=False, input_hold_required=not allowed))
        return deepcopy(self.status)

    def pause(self, reason):
        self.active = False
        self.recovery_targets = self.recovery_stamp = None
        return self._status(reason)

    def _input_unavailable(self, reason, now):
        if self.missing_since is None:self.missing_since=now
        if self.active and now-self.missing_since < .3:
            # No command is produced while tracking is absent. A short gap
            # keeps the original controller-to-robot anchors; recovery catches
            # up smoothly instead of discarding the missed displacement.
            return self._status('tracking_gap_measured_hold:'+reason)
        return self.pause('tracking_paused_press_AX:'+reason)

    def _inputs(self, packet, ui, now, clock_id):
        issue = validate_operator_packet(packet, now=now, clock_id=clock_id)
        if issue: raise ValueError(issue)
        if not isinstance(ui, dict) or type(ui.get('revision')) is not int:
            raise ValueError('waiting_for_controller_buttons')
        stamp = ui.get('capture_timestamp')
        if (ui.get('clock_id') != clock_id or type(stamp) not in (float,int)
                or not np.isfinite(stamp) or not -.01 <= now-stamp <= .25
                or (ui.get('expires_at') is not None and now >= ui['expires_at'])):
            raise ValueError('controller_buttons_stale')
        # validate_operator_packet already checks the original XR source's
        # rigid-transform tolerance (1e-5). Float32 WebXR rotations can pass
        # that gate while failing the assist controller's 1e-6 gate. Project
        # only these already-valid input rotations onto SO(3), without changing
        # translation or accepting poses rejected by the source validator.
        poses=[]
        for key in ('left_base_ee','right_base_ee'):
            pose=np.array(packet[key],dtype=float,copy=True)
            pose[:3,:3]=Rotation.from_matrix(pose[:3,:3]).as_matrix()
            poses.append(rigid(pose))
        return tuple(poses)

    def step(self, packet, ui, measured, q, dq, *, now, clock_id):
        # Stop/lease-loss remains actionable even when no valid pose exists.
        # A host lease timeout pauses; a deliberate ABXY release decouples.
        if isinstance(ui,dict) and ui.get('release') is True:
            if type(ui.get('revision')) is int:
                self.revision=max(ui['revision'],self.revision or 0)
            if ui.get('source_kind') == 'webxr_controller_event':self.anchors=None
            self.pause('released' if self.anchors is None else 'control_lease_paused_press_AX')
            return None
        try:
            raw = self._inputs(packet,ui,now,clock_id)
        except (ValueError,TypeError,KeyError) as error:
            self._input_unavailable(str(error),now)
            return None
        try:
            recovering=self.missing_since is not None
            self.missing_since=None
            self.source_stamp=packet['acquisition_timestamp'];self.source_sequence=packet['source_sequence']
            measured = tuple(rigid(p) for p in measured)
            q, dq = arm_vector(q), arm_vector(dq)
            epoch = (packet.get('source_instance_id'),packet.get('source_event_clock_id'))
            if self.epoch is not None and epoch != self.epoch:
                self.anchors = None; self.active = False; self.revision = None
            self.epoch = epoch
            edge = self.revision is not None and ui['revision'] > self.revision
            if self.revision is not None and ui['revision'] < self.revision:
                raise ValueError('controller_revision_regressed')
            self.revision = ui['revision']
            if edge and ui.get('clutch'):
                self.anchors = tuple((a.copy(),b.copy()) for a,b in zip(raw,measured))
                self.worker.call('original_manual_reset',current_q=q.tolist())
                self.sequence = packet['source_sequence']
                self.pause('teleop_start_required')
                return None
            if edge and ui.get('start') and self.anchors is not None:
                self.active = not self.active
                if self.active:
                    self.anchors = tuple((a.copy(),b.copy()) for a,b in zip(raw,measured))
                    self.worker.call('original_manual_reset',current_q=q.tolist())
                self.sequence = packet['source_sequence']
                self._status('teleop_started_waiting_for_next_XR_sample' if self.active else 'teleop_paused')
                return None
            if not self.active:
                self._status('teleop_start_required' if self.anchors is not None else 'clutch_required')
                return None
            if recovering:
                self.recovery_targets = tuple(p.copy() for p in measured)
                self.recovery_stamp = now
            if ui.get('clutch') or ui.get('start') or packet['source_sequence'] == self.sequence:
                self._status('waiting_for_new_XR_sample')
                return None
            if self.sequence is not None and packet['source_sequence'] < self.sequence:
                raise ValueError('controller_sequence_regressed')
            self.sequence = packet['source_sequence']
            targets = []
            for pose,(origin,robot) in zip(raw,self.anchors):
                target = robot.copy()
                target[:3,3] += pose[:3,3]-origin[:3,3]
                target[:3,:3] = (Rotation.from_matrix(pose[:3,:3]) *
                    Rotation.from_matrix(origin[:3,:3]).inv() * Rotation.from_matrix(robot[:3,:3])).as_matrix()
                targets.append(target)
            if self.target_filter is not None:
                targets=self.target_filter(targets,now=now)
                targets=[rigid(target) for target in targets]
            if self.recovery_targets is not None:
                dt=min(.05,max(0.,now-self.recovery_stamp));self.recovery_stamp=now
                recovered=[];caught_up=True
                for target,previous in zip(targets,self.recovery_targets):
                    delta=target[:3,3]-previous[:3,3];distance=np.linalg.norm(delta)
                    turn=Rotation.from_matrix(target[:3,:3]@previous[:3,:3].T).as_rotvec()
                    angle=np.linalg.norm(turn)
                    bounded=previous.copy()
                    bounded[:3,3]+=delta*min(1.,self.recovery_speed_m_s*dt/max(distance,1e-12))
                    bounded[:3,:3]=Rotation.from_rotvec(turn*min(1.,
                        self.recovery_speed_rad_s*dt/max(angle,1e-12))).as_matrix()@previous[:3,:3]
                    recovered.append(bounded)
                    caught_up &= distance<=self.recovery_speed_m_s*dt+1e-9 and angle<=self.recovery_speed_rad_s*dt+1e-9
                self.recovery_targets=None if caught_up else tuple(recovered)
                targets=recovered
            result = self.worker.call('original_manual_ik',left_base_ee=targets[0].tolist(),
                right_base_ee=targets[1].tolist(),current_q=q.tolist(),current_dq=dq.tolist())
            target_q = arm_vector(result['q14'])
            self._status('unitree_manual_ik',allowed=True)
            self.status.update(mapped_left_base_ee=targets[0].tolist(),mapped_right_base_ee=targets[1].tolist(),
                tracking_recovery_active=self.recovery_targets is not None,
                operator_source_sequence=self.sequence,operator_acquisition_timestamp=packet['acquisition_timestamp'])
            return target_q
        except (ValueError,TypeError,KeyError,RuntimeError,TimeoutError,OSError) as error:
            self.pause(str(error))
            return None

    def input_is_current(self, packet, ui, *, now, clock_id):
        """Connection/stop watchdog; does not rerun IK or clear calibration."""
        try:
            self._inputs(packet,ui,now,clock_id)
            if (packet.get('source_instance_id'),packet.get('source_event_clock_id')) != self.epoch:
                raise ValueError('controller_reconnected')
            if (ui.get('release') or (ui['revision'] > self.revision and
                    (ui.get('start') or ui.get('clutch')))):
                # The next step handles this edge exactly once.
                self._status('controller_action_pending')
                return False
            return self.active
        except (ValueError,TypeError,KeyError) as error:
            self._input_unavailable(str(error),now)
            return False
