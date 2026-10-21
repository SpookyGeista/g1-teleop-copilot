"""Pinned original G1 controller with a receipt-only, non-network simulator sink.

The original constructor and controller class are loaded into an explicitly
owned namespace, not imported. Their channel factories are permanently local
and blocked for writes. Both original thread targets are overridden BEFORE the
constructor executes. This is the executable shim used by a future staged XR
entry, not evidence that that entry or a physical headset has been exercised.

Only a local consumed SimControlSession receipt can reach the owned PhysX sink.
The inactive LowCmd CDR is a q14 carrier, never an actuable LowCmd regulator:
PD, FF, modes and non-arm fields are not copied from the original self.msg.
The sink preserves measured fingers/non-arms and does not invoke downstream IK.
The caller must own physics stepping, honor expiry before subsequent steps,
and never attach another actor to the same private session. No DDS is created.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from enum import IntEnum
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import weakref

import numpy as np

from .kinematics import ARM_JOINT_NAMES
from .sim_control import SimCommandReceipt, SimControlSession, SimReceiptReplan
from .unitree_codec import UnitreeSimArmCodec


ORIGINAL_ARM_SHA256 = '793e0774bb0a7a29b0cd163336babd45b29a36c3518528c87a854969c4132f73'
INACTIVE_POLICY = 'inactive_simulator_q14_carrier_only'
NONARM_POLICY = 'preserve_measured_joint_positions'
# Prospective admission policy for the five-substep isolated simulator batch.
# This is an early-refusal margin, NOT a scheduling/WCET guarantee. Every
# post-install physics interlock continues to enforce the original deadline.
PREINSTALL_EXECUTION_RESERVE_S = .060
_SESSION_CLAIMS = weakref.WeakKeyDictionary()
_CLAIM_LOCK = threading.Lock()
BODY_JOINT_NAMES = tuple(f'{side}_{name}_joint' for side in ('left','right')
    for name in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')) + (
    'waist_yaw_joint','waist_roll_joint','waist_pitch_joint') + tuple(ARM_JOINT_NAMES)


def _now(clock):
    value = clock()
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('finite_named_clock_required')
    return value


def _numpy(tensor):
    # Isaac tensor interface; no torch/Isaac import or device construction here.
    return np.asarray(tensor.detach().cpu().numpy())


class OriginalSimReceiptSink:
    """Owned synchronous CDR receiver for one paused original Isaac articulation.

    Snapshot callbacks prove the complete actual q43/base/body/scene, and the
    session's private inflight registry proves the receipt. Neither a JSON flag
    nor a supplied success callback can authorize motion. There is one sink per
    session; attempted receipts are burned even if encoding or application fails.
    Validation and target installation are synchronous on the source owner thread.
    A release after the last input read is observed on the next owner iteration;
    there is no atomic cross-process mailbox/physics transaction.
    """
    def __init__(self, session, snapshots, *, simulation_mode, motion_mode=False,
                 nonarm_policy=NONARM_POLICY, codec=None, clock=time.monotonic):
        if simulation_mode is not True or motion_mode is not False:
            raise ValueError('explicit_stationary_simulation_required')
        if nonarm_policy != NONARM_POLICY:
            raise ValueError('unknown_nonarm_policy')
        from .original_sim_snapshot import OriginalSimSnapshotAdapter
        if type(session) is not SimControlSession or type(snapshots) is not OriginalSimSnapshotAdapter:
            raise ValueError('owned_session_and_authoritative_sim_snapshot_required')
        snapshots.assert_owner()
        if not callable(clock):
            raise ValueError('acquisition_clock_required')
        if codec is not None and type(codec) is not UnitreeSimArmCodec:
            raise ValueError('reviewed_inactive_Unitree_codec_required')
        self.session, self.snapshots, self.clock = session, snapshots, clock
        self.robot = snapshots.robot
        if not callable(getattr(self.robot, 'set_joint_position_target', None)):
            raise ValueError('original_simulator_articulation_target_sink_required')
        self.joint_names = tuple(self.robot.joint_names)
        if len(self.joint_names)!=43 or len(set(self.joint_names))!=43:
            raise ValueError('complete_unique_original_q43_order_required')
        self.arm_ids = tuple(self.joint_names.index(name) for name in ARM_JOINT_NAMES)
        self.owner = threading.get_ident()
        self.codec = codec
        self._last_sequence = 0
        self._last_deadline = None
        self._last_receipt = self._read_inputs = self._recheck_target = None
        self.last_physics_interlock = None
        self._closed = False
        with _CLAIM_LOCK:
            if session in _SESSION_CLAIMS:
                raise ValueError('private_session_already_has_original_sim_sink')
            _SESSION_CLAIMS[session] = True

    def _owner(self):
        self.snapshots.assert_owner()
        if threading.get_ident()!=self.owner:
            raise ValueError('single_physics_owner_thread_required')
        if self.robot is not self.snapshots.robot or tuple(self.robot.joint_names)!=self.joint_names:
            raise ValueError('original_simulator_robot_or_joint_order_changed')

    def _codec(self):
        if self.codec is None:
            self.codec = UnitreeSimArmCodec(simulation_mode=True)
        if type(self.codec) is not UnitreeSimArmCodec:
            raise ValueError('reviewed_inactive_Unitree_codec_required')
        manifest = self.codec.manifest
        if (manifest.get('scope')!=INACTIVE_POLICY or manifest.get('hardware_authorized') is not False
                or manifest.get('transport_created') is not False or manifest.get('motor_mode')!=0):
            raise ValueError('inactive_carrier_policy_required')
        return self.codec

    def _hold(self):
        """Local measured-state hold, never a zero/home or LowCmd write."""
        self._owner()
        current = self.robot.data.joint_pos.clone()
        values = _numpy(current)
        if values.shape!=(1,43) or not np.isfinite(values).all():
            raise ValueError('measured_hold_state_unavailable')
        self.robot.set_joint_position_target(current)

    def cancel(self, reason='original_sender_cancelled'):
        self._owner()
        self._last_deadline = None
        self._last_receipt = self._read_inputs = self._recheck_target = None
        self.session.hold(reason,now=_now(self.clock),clock_id=self.snapshots.clock_id)
        self._hold()

    def idle_hold(self):
        """No offered command: hold measured joints, preserve logical UI state.

        The owner uses this after step() offered no receipt and thus already
        revoked any old inflight command. Selection/clutch waiting must survive.
        """
        self._owner()
        self._last_deadline = None
        self._last_receipt = self._read_inputs = self._recheck_target = None
        self._hold()

    def close(self):
        self._owner()
        if not self._closed:
            self._closed = True
            self.cancel('original_sender_closed')

    def expire_before_physics_step(self):
        """Caller-owned simulator watchdog; no background thread or DDS timer.

        The staged owner must invoke this before advancing physics. This only
        preserves the last command until its original private deadline; it
        never renews that deadline or authorizes a new command.
        """
        self._owner()
        receipt = self._last_receipt
        ledger = dict(accepted=False,checked_at=None,valid_until=self._last_deadline,
            receipt_token=receipt.token if receipt is not None else None,
            guard_sequence=receipt.guard_sequence if receipt is not None else None,
            reason='original_sender_physics_interlock_not_completed',
            scope='single_owner_before_PhysX_write_data_to_sim',DDS_transport_started=False,
            hardware_authorized=False)
        self.last_physics_interlock = ledger
        if self._last_deadline is None:
            self.idle_hold()
            ledger.update(checked_at=_now(self.clock),reason='idle_measured_hold_without_session_revocation')
            return False
        try:
            checked_at = _now(self.clock)
            ledger['checked_at'] = checked_at
            if self._closed or checked_at>=self._last_deadline:
                raise ValueError('original_sender_lease_not_current')
            operator,ui = self._read_inputs()
            target = self._recheck_target()
            if not isinstance(target,dict) or target.get('accepted') is not True:
                raise ValueError('original_sender_target_revoked_before_physics')
            latest = self.session.recheck_receipt_inputs(self._last_receipt,operator,ui,
                now=_now(self.clock),clock_id=self.snapshots.clock_id)
            if type(latest) is SimReceiptReplan:
                if not self.session.consume_receipt_replan(latest,receipt,
                        now=_now(self.clock),clock_id=self.snapshots.clock_id):
                    raise ValueError('original_sender_replan_private_binding_rejected')
                self.idle_hold()
                ledger.update(checked_at=_now(self.clock),receipt_replan=True,
                    reason='valid_UI_replan_measured_hold',measured_sim_hold_installed=True)
                return False
            if latest is None:
                raise ValueError('original_sender_receipt_revoked_before_physics')
            if type(latest) is not SimCommandReceipt:
                raise ValueError('original_sender_receipt_recheck_type_invalid')
            self._last_receipt,self._last_deadline = latest,latest.valid_until
            ledger['valid_until'] = latest.valid_until
            checked_at = _now(self.clock)
            ledger['checked_at'] = checked_at
            if checked_at>=latest.valid_until:
                raise ValueError('original_sender_lease_expired_before_physics')
            ledger.update(accepted=True,reason='private_receipt_current_before_PhysX_write_data_to_sim')
            return True
        except Exception as error:
            ledger['reason'] = str(error) or type(error).__name__
            self.cancel('original_sender_interlock_not_current_before_physics')
            return False

    def write_once(self, receipt, *, read_inputs, recheck_target):
        self._owner()
        result = dict(accepted=False,applied=False,write_attempted=False,delivery_unknown=False,
            reason='receipt_required',backend='same_process_isolated_PhysX',
            DDS_transport_started=False,hardware_authorized=False,
            command_policy=INACTIVE_POLICY,nonarm_policy=NONARM_POLICY,
            command_format='unitree_float32')
        attempted = False
        try:
            if self._closed:
                raise ValueError('original_sender_closed')
            if (type(receipt) is not SimCommandReceipt or receipt.scope!='isolated_PhysX_consumed_arm_command'
                    or receipt.command_format!='unitree_float32'
                    or type(receipt.guard_sequence) is not int or receipt.guard_sequence<=self._last_sequence):
                raise ValueError('fresh_local_float32_receipt_required')
            self._last_sequence = receipt.guard_sequence
            if not callable(read_inputs) or not callable(recheck_target):
                raise ValueError('fresh_input_and_target_rechecks_required')
            codec = self._codec()
            payload = codec.encode_arm_targets(receipt.q14)
            decoded = tuple(codec.decode_arm_targets(payload))
            if decoded != receipt.q14:
                raise ValueError('codec_changed_private_q14')
            result.update(cdr_sha256=hashlib.sha256(payload).hexdigest(),cdr_bytes=len(payload),
                q14=list(decoded),receipt_token=receipt.token,guard_sequence=receipt.guard_sequence,
                receipt_valid_until=receipt.valid_until,guard_snapshot_sha256=receipt.guard_snapshot_sha256,
                codec=dict(accepted=True,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
                    q14_before=list(receipt.q14),q14_after=list(decoded),
                    scope='local_Unitree_IDL_CDR_roundtrip_no_DDS_transport'))
            # ALL encoding/CRC precedes renewed reads, preserving acquisition
            # times. No original message, torque, clip, filter or IK is used.
            operator, ui = read_inputs()
            target = recheck_target()
            if not isinstance(target,dict) or target.get('accepted') is not True:
                raise ValueError('target_revoked_before_original_sim_write')
            sample = self.snapshots.read_same()
            if sample.snapshot_sha256!=receipt.consumed_snapshot_sha256:
                raise ValueError('authoritative_snapshot_differs_from_consumed_receipt')
            if sample.clock_id!=receipt.clock_id:
                raise ValueError('authoritative_snapshot_clock_mismatch')
            state = sample.robot_state
            # No third-party writer callback: this sink owns the exact complete
            # target it installs on the sampled articulation.
            target_tensor = self.robot.data.joint_pos.clone()
            before = _numpy(target_tensor).copy()
            expected = np.asarray([state['joint_positions'][name] for name in self.joint_names],dtype=float)
            if (before.shape!=(1,43) or before.dtype!=np.float32
                    or not np.array_equal(before[0],expected)):
                raise ValueError('measured_q43_changed_before_target_build')
            target_tensor[0,list(self.arm_ids)] = target_tensor.new_tensor(decoded)
            after = _numpy(target_tensor)
            expected_target = before.copy()
            expected_target[0,list(self.arm_ids)] = np.asarray(decoded,dtype=np.float32)
            if not np.array_equal(after.view(np.uint32),expected_target.view(np.uint32)):
                raise ValueError('simulator_target_changed_arms_or_nonarms')
            # Tensor allocation/copies may block. Re-prove sampled geometry,
            # then consult the PRIVATE receipt registry after all callbacks.
            final_sample = self.snapshots.read_same()
            if final_sample!=sample:
                raise ValueError('authoritative_sample_changed_during_target_build')
            result.update(consumed_snapshot_sha256=sample.snapshot_sha256,
                physics_generation=sample.physics_generation,source_instance_id=sample.source_instance_id,
                sample_sequence=sample.sample_sequence,target_recheck=deepcopy(target))
            latest = self.session.recheck_receipt_inputs(receipt,operator,ui,
                now=_now(self.clock),clock_id=sample.clock_id,require_uninstalled=True,
                execution_reserve_s=PREINSTALL_EXECUTION_RESERVE_S)
            if 'execution_budget' in self.session.status:
                # Reporting only: neither these fields nor a public reason
                # can authorize the private decision or a simulator write.
                result['execution_budget'] = dict(self.session.status['execution_budget'])
            if type(latest) is SimReceiptReplan:
                if not self.session.consume_receipt_replan(latest,receipt,
                        now=_now(self.clock),clock_id=sample.clock_id):
                    raise ValueError('original_sender_replan_private_binding_rejected')
                self.idle_hold()
                budget_replan=self.session.status.get('receipt_replan_kind')=='execution_budget'
                result.update(reason=('execution_budget_replan_measured_hold' if budget_replan
                    else 'valid_UI_replan_measured_hold'),receipt_replan=True,
                    receipt_replan_kind=self.session.status.get('receipt_replan_kind'),
                    measured_sim_hold_installed=True,write_attempted=False)
                return result
            if latest is None:
                result['interlock_reason'] = self.session.status.get('reason')
                raise ValueError('private_receipt_interlock_rejected')
            if type(latest) is not SimCommandReceipt:
                raise ValueError('original_sender_receipt_recheck_type_invalid')
            result['applied_valid_until'] = latest.valid_until
            checked_at = _now(self.clock)
            if checked_at>=latest.valid_until or self._closed:
                raise ValueError('private_receipt_expired_at_original_sim_write')
            attempted = True
            self.robot.set_joint_position_target(target_tensor)
            if not self.session.mark_receipt_installed(latest,
                    now=_now(self.clock),clock_id=sample.clock_id):
                raise ValueError('original_sender_receipt_install_ack_rejected')
            self._last_deadline = latest.valid_until
            self._last_receipt,self._read_inputs,self._recheck_target = latest,read_inputs,recheck_target
            result.update(accepted=True,applied=True,reason='exact_receipt_q14_installed_on_sampled_simulator',
                checked_before_write_at=checked_at,checked_before_PhysX_handoff_at=checked_at)
        except Exception as error:
            result['reason'] = str(error) or type(error).__name__
            result['delivery_unknown'] = attempted
            try:
                self.cancel('guarded_original_sender_rejected')
                result['measured_sim_hold_installed'] = True
            except Exception as hold_error:
                result['measured_sim_hold_installed'] = False
                result['hold_error'] = str(hold_error) or type(hold_error).__name__
        result['write_attempted'] = attempted
        return result


class _BlockedLegacyPublisher:
    def __init__(self,*_args):
        self.write_attempts = 0
    def Init(self):
        pass
    def Write(self,_message):
        self.write_attempts += 1
        raise RuntimeError('original_unchecked_LowCmd_publication_forbidden')


class _InlineThread:
    """Original constructor compatibility handle, never an OS thread."""
    def __init__(self, *, target):
        self.target, self.daemon, self.started = target, False, False
    def start(self):
        if self.started:raise RuntimeError('thread_handle_already_started')
        self.started = True
        self.target()
    def join(self,*_args,**_kwargs):
        pass
    def is_alive(self):
        return False


class _NoLegacyCRC:
    def Crc(self,_message):
        raise RuntimeError('original_unchecked_message_CRC_forbidden')


def _constructor_message():
    # Compatibility-only object for original initialization; NEVER a payload.
    return SimpleNamespace(mode_pr=0,mode_machine=0,reserve=[0]*4,crc=0,
        motor_cmd=[SimpleNamespace(mode=0,q=0.,dq=0.,tau=0.,kp=0.,kd=0.,reserve=0) for _ in range(35)])


def guarded_original_controller_class(source, sink):
    """Return the exact pinned original class with local guarded overrides.

    Selecting this class before construction replaces the original import in a
    staged XR entry. No original file/global is edited and no original import
    executes. Dependency injection is local to this new class's namespace.
    The source hash intentionally rejects upstream changes pending review.
    """
    if type(sink) is not OriginalSimReceiptSink:
        raise ValueError('owned_original_sim_receipt_sink_required')
    source = Path(source)
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest()!=ORIGINAL_ARM_SHA256:
        raise ValueError('unreviewed_original_G1_29_controller_source')
    tree = ast.parse(content,filename=str(source))
    names = ('MotorState','G1_29_LowState','DataBuffer','G1_29_ArmController',
             'G1_29_JointArmIndex','G1_29_JointIndex')
    classes = {node.name:node for node in tree.body if isinstance(node,ast.ClassDef)}
    selected = [deepcopy(classes[name]) for name in names]
    module = ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
    if any(isinstance(node,(ast.Import,ast.ImportFrom)) for node in ast.walk(module)):
        raise ValueError('imports_forbidden_in_pinned_controller_AST')

    def legacy_q():
        values = sink.snapshots.read_same().robot_state['joint_positions']
        return np.asarray([values[name] for name in BODY_JOINT_NAMES]+[0.]*6,dtype=float)

    class LocalSubscriber:
        def __init__(self,*_args):pass
        def Init(self):pass
        def Read(self):
            return SimpleNamespace(mode_machine=0,motor_state=[SimpleNamespace(q=float(q),dq=0.) for q in legacy_q()])

    def forbidden_sleep(_seconds):
        raise RuntimeError('unexpected_original_controller_wait_or_loop')

    namespace = dict(__name__='punktnull_pinned_original_G1_29',np=np,IntEnum=IntEnum,
        threading=SimpleNamespace(Thread=_InlineThread,Lock=threading.Lock),
        time=SimpleNamespace(time=sink.clock,sleep=forbidden_sleep),
        logger_mp=SimpleNamespace(info=lambda *_a:None,debug=lambda *_a:None,warning=lambda *_a:None),
        ChannelPublisher=_BlockedLegacyPublisher,ChannelSubscriber=LocalSubscriber,
        hg_LowCmd=object,hg_LowState=object,CRC=_NoLegacyCRC,
        unitree_hg_msg_dds__LowCmd_=_constructor_message,
        G1_29_Num_Motors=35,kTopicLowCommand_Debug='blocked/lowcmd',
        kTopicLowCommand_Motion='blocked/arm_sdk',kTopicLowState='local/measured_state')
    exec(compile(module,str(source),'exec'),namespace)
    original = namespace['G1_29_ArmController']
    if tuple(int(member) for member in namespace['G1_29_JointArmIndex'])!=tuple(range(15,29)):
        raise ValueError('unreviewed_original_arm_order')

    class GuardedG1_29SimController(original):
        def __init__(self,motion_mode=False,simulation_mode=False):
            if motion_mode is not False or simulation_mode is not True:
                raise ValueError('explicit_stationary_simulation_required_before_original_constructor')
            self._guarded_sink = sink
            self._guarded_sink._owner()
            super().__init__(motion_mode=False,simulation_mode=True)

        def _subscribe_motor_state(self):
            # One bounded local read makes the original constructor ready.
            # Later display getters read one authoritative sample each.
            self.lowstate_buffer.SetData(self.lowstate_subscriber.Read())

        def _ctrl_motor_state(self):
            return  # Bound before the original constructor selects its target.

        def get_mode_machine(self):
            return 0  # Compatibility-only stationary inactive-carrier policy.

        def get_current_motor_q(self):
            return legacy_q()

        def get_current_dual_arm_q(self):
            return legacy_q()[15:29].copy()

        def get_current_dual_arm_dq(self):
            self._guarded_sink._owner()
            values = _numpy(self._guarded_sink.robot.data.joint_vel)
            if values.shape!=(1,43) or not np.isfinite(values).all():
                raise ValueError('measured_joint_velocity_unavailable')
            return values[0,list(self._guarded_sink.arm_ids)].copy()

        def ctrl_dual_arm(self,*_args,**_kwargs):
            self.cancel('unchecked_original_setter_rejected')
            raise ValueError('raw_q14_and_feedforward_forbidden_use_private_receipt')

        def clip_arm_q_target(self,*_args,**_kwargs):
            self.cancel('downstream_clipping_rejected')
            raise ValueError('validated_q14_must_not_be_clipped')

        def ctrl_dual_arm_go_home(self):
            self.cancel('original_go_home_replaced_by_measured_sim_hold')

        def publish_receipt(self,receipt,*,read_inputs,recheck_target):
            return self._guarded_sink.write_once(receipt,read_inputs=read_inputs,recheck_target=recheck_target)

        def cancel(self,reason='original_sender_cancelled'):
            self._guarded_sink.cancel(reason)

        def idle_hold(self):
            self._guarded_sink.idle_hold()

        def close(self):
            self._guarded_sink.close()

    GuardedG1_29SimController.original_manifest = dict(source=str(source),sha256=ORIGINAL_ARM_SHA256,
        extracted_classes=list(names),original_class_bodies_unmodified=True,
        guarded_methods=['__init__','_subscribe_motor_state','_ctrl_motor_state','get_mode_machine',
            'get_current_motor_q','get_current_dual_arm_q','get_current_dual_arm_dq',
            'ctrl_dual_arm','clip_arm_q_target','ctrl_dual_arm_go_home'],
        constructor_dependencies='owned_local_non_network_namespace',
        publisher_override_installed_before_constructor=True,OS_threads_started=False,
        original_module_imported=False,DDS_transport_started=False,hardware_authorized=False,
        original_self_msg_used_for_commands=False,complete_XR_entry_integration_verified=False)
    return GuardedG1_29SimController
