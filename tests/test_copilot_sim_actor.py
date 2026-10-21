"""Actual session receipts through final actor; fake physics and byte codec."""
from copy import deepcopy
import struct

import pytest

from teleop_assist.copilot.sim_actor import SimReceiptActor
from test_copilot_sim_receipt import fixture, receipt
from test_copilot_sim_control import race_ui


class Bench:
    clock_id = 'mono'
    def __init__(self, f): self.f, self.commands, self.refuse = f, [], False
    def read_command_sample(self): return None, self.f.last['state'], self.f.last['scene'], None
    def command(self, q, *, expires_at=None, clock_id=None):
        self.last_command_checked_at = self.f.clock
        if q is None:
            self.commands.append(None)
            return False
        if self.refuse or clock_id != self.clock_id or self.f.clock >= expires_at: return False
        self.commands.append((tuple(q), expires_at))
        return True


class Codec:
    def __init__(self, callback=None): self.callback = callback
    def encode_arm_targets(self,q):
        if self.callback: self.callback()
        return struct.pack('<14f',*q)
    def decode_arm_targets(self,data): return list(struct.unpack('<14f',data))


def ready(callback=None):
    f, offer = fixture(assisting=True)
    consumed = receipt(f, offer)
    bench = Bench(f)
    actor = SimReceiptActor(f.session, bench, codec=Codec(callback), clock=lambda:f.clock)
    def apply(value=consumed, **changes):
        return actor.apply(value, read_inputs=lambda:(f.last['packet'],f.last['ui']), **changes)
    return f, consumed, bench, actor, apply


def test_exact_consumed_command_is_applied_once_and_fingers_do_not_enter_the_arm_carrier():
    f, r, bench, actor, apply = ready()
    result = apply()
    assert result['applied'] and result['codec']['q14_before']==result['codec']['q14_after']==list(r.q14)
    assert bench.commands == [(r.q14, r.valid_until)]
    assert not apply()['applied'] and bench.commands[-1] is None


@pytest.mark.parametrize('fault',['release','new_owner','new_source','hold_button','new_selection','stale_source',
    'expired','finger','obstacle','cancel','physics_advance','different_q','crc_error'])
def test_faults_during_serialization_cannot_reach_the_actor(fault):
    f,r,bench,actor,apply=ready()
    def mutate():
        f.clock=f.now+.006
        if fault=='release':f.last['ui']=race_ui(f,release=True)
        if fault=='new_owner':f.last['ui']=race_ui(f,client_id='other')
        if fault=='new_source':f.last['packet']['source_instance_id']='other'
        if fault=='hold_button':f.last['ui']=race_ui(f,assist_held=False)
        if fault=='new_selection':f.last['ui']=race_ui(f,select_candidate_id='other')
        if fault=='stale_source':f.last['packet']['expires_at']=f.now-.01
        if fault=='expired':f.clock=r.valid_until
        if fault=='finger':f.last['state']['joint_positions']['right_hand_index_0_joint']+=.001
        if fault=='obstacle':f.last['scene']['obstacles'][0]['size_xyz'][0]+=.001
        if fault=='cancel':f.session.hold('explicit_cancel',now=f.clock,clock_id='mono')
        if fault=='physics_advance':
            def advanced():raise ValueError('physics_advanced')
            bench.read_command_sample=advanced
        if fault=='crc_error':raise ValueError('original_HG_CRC_mismatch')
    actor.codec=Codec(mutate)
    if fault=='different_q':actor.codec.decode_arm_targets=lambda data:[0.]*14
    result=apply()
    assert not result['applied'],result
    assert bench.commands==[None]
    if fault in ('hold_button','new_selection'):
        assert f.session.clutched and result['receipt_replan'] is True
    else:
        assert not f.session.clutched


def test_new_shorter_UI_lease_after_encoding_is_used_by_actual_target_application():
    f,r,bench,actor,apply=ready()
    short=f.now+.025
    def heartbeat():
        f.clock=f.now+.006
        f.last.update(ui=race_ui(f,expires_at=short,assist_held=True))
    actor.codec=Codec(heartbeat)
    result=apply()
    assert result['applied'] and result['applied_valid_until']==short<r.valid_until
    assert bench.commands==[(r.q14,short)]


def test_target_revocation_after_encoding_causes_hold():
    f,r,bench,actor,apply=ready()
    result=apply(recheck_target=lambda:dict(accepted=False))
    assert not result['applied'] and bench.commands==[None] and not f.session.clutched


def test_final_bench_rejection_erases_the_previously_committed_offset():
    f,r,bench,actor,apply=ready();bench.refuse=True
    assert not apply()['applied'] and bench.commands==[None]
    assert not f.session.clutched
    assert f.session._committed_offset.tolist()==[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]


def test_no_receipt_holds_physics_without_erasing_idle_selection_or_clutch():
    f,r,bench,actor,apply=ready()
    assert not apply(None)['applied'] and bench.commands==[None]
    assert f.session.clutched


def test_new_session_step_invalidates_even_an_unapplied_old_receipt():
    f,r,bench,actor,apply=ready()
    f.tick(assist_held=True)
    assert not apply()['applied'] and bench.commands==[None]


def test_default_direct_actor_also_uses_private_receipt_and_post_consume_input_check():
    f,offer=fixture();f.worker.command_format='float64'
    offer=f.tick();r=receipt(f,offer);bench=Bench(f)
    actor=SimReceiptActor(f.session,bench,clock=lambda:f.clock)
    result=actor.apply(r,read_inputs=lambda:(f.last['packet'],f.last['ui']))
    assert result['applied'] and result['command_format']=='float64' and 'codec' not in result
    assert bench.commands==[(r.q14,r.valid_until)]
