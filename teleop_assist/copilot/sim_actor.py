"""Final receipt-bound arm target application to the isolated PhysX bench.

Optional Unitree IDL serialization is a local round trip, not a DDS transport.
Only arm q14 is applied; the bench owns measured fingers/non-arms and its
existing PD/gravity effort budget. No LowCmd gain or torque field controls it.
"""
from __future__ import annotations
import hashlib
import time

from .sim_control import SimCommandReceipt, SimReceiptReplan
from .xr_send import snapshot_digest


class SimReceiptActor:
    def __init__(self, session, bench, *, codec=None, clock=time.monotonic):
        self.session, self.bench, self.codec, self.clock = session, bench, codec, clock
        self._last_guard_sequence = 0

    def apply(self, receipt, *, read_inputs, recheck_target=None):
        result = dict(applied=False, reason='no_consumed_receipt',
            command_format='unitree_float32' if self.codec else 'float64',
            DDS_transport_started=False, hardware_authorized=False)
        if receipt is None:
            self.bench.command(None)
            return result
        try:
            if type(receipt) is not SimCommandReceipt or receipt.guard_sequence <= self._last_guard_sequence:
                raise ValueError('receipt_missing_replayed_or_regressed')
            self._last_guard_sequence = receipt.guard_sequence  # Burn even on codec failure.
            if receipt.command_format != result['command_format']:
                raise ValueError('receipt_command_format_does_not_match_actor')
            q14 = receipt.q14
            if self.codec is not None:
                payload = self.codec.encode_arm_targets(q14)
                decoded = tuple(self.codec.decode_arm_targets(payload))
                if decoded != q14:
                    raise ValueError('Unitree_codec_changed_guarded_q14')
                result['codec'] = dict(accepted=True, bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    q14_before=list(q14), q14_after=list(decoded),
                    scope='local_Unitree_IDL_CDR_roundtrip_no_DDS_transport')
                q14 = decoded
            # The codec has no I/O, but even local serialization can stall.
            # Re-read operator/UI/target AFTER it, preserving original leases.
            operator, ui = read_inputs()
            if recheck_target is not None:
                target = recheck_target()
                if not isinstance(target, dict) or target.get('accepted') is not True:
                    raise ValueError('target_revoked_during_final_handoff')
                result['target_recheck'] = target
            _, state, scene, _ = self.bench.read_command_sample()
            if snapshot_digest(state, scene) != receipt.consumed_snapshot_sha256:
                raise ValueError('receipt_snapshot_changed_before_actor')
            latest = self.session.recheck_receipt_inputs(receipt, operator, ui,
                now=self.clock(), clock_id=self.bench.clock_id, require_uninstalled=True)
            if type(latest) is SimReceiptReplan:
                if not self.session.consume_receipt_replan(latest, receipt,
                        now=self.clock(), clock_id=self.bench.clock_id):
                    raise ValueError('receipt_replan_private_binding_rejected')
                self.bench.command(None)
                result.update(reason='valid_UI_replan_measured_hold', receipt_replan=True,
                    measured_sim_hold_installed=True, receipt_token=receipt.token,
                    guard_sequence=receipt.guard_sequence)
                return result
            if latest is None:
                result['interlock_reason'] = self.session.status.get('reason')
                raise ValueError('receipt_operator_or_UI_interlock_rejected')
            if type(latest) is not SimCommandReceipt:
                raise ValueError('receipt_recheck_return_type_invalid')
            result.update(receipt_valid_until=receipt.valid_until, applied_valid_until=latest.valid_until,
                receipt_token=receipt.token, guard_sequence=receipt.guard_sequence,
                consumed_snapshot_sha256=receipt.consumed_snapshot_sha256,
                guard_snapshot_sha256=receipt.guard_snapshot_sha256)
            # The bench performs its own last clock check after allocation,
            # immediately before setting targets; no public offer expiry used.
            if not self.bench.command(q14, expires_at=latest.valid_until, clock_id=latest.clock_id):
                raise ValueError('receipt_expired_at_PhysX_actor')
            if not self.session.mark_receipt_installed(latest,
                    now=self.clock(), clock_id=self.bench.clock_id):
                raise ValueError('receipt_install_ack_rejected')
            result.update(applied=True, reason='exact_receipt_q14_applied',
                checked_before_PhysX_handoff_at=self.bench.last_command_checked_at)
        except Exception as error:
            self.session.hold('final_receipt_handoff_rejected', now=self.clock(), clock_id=self.bench.clock_id)
            self.bench.command(None)
            result['reason'] = str(error) or type(error).__name__
        return result
