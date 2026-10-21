"""Offline source-behavior evidence, deliberately without original imports."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.copilot.audit_original_xr_send import (ARM_SHA256, ARM_SOURCE, ENUMS, METHODS,
    MOTOR_SOURCE, RecordingPublisher, FakeTime, audit, arm_values, controller_fixture,
    extracted_module, run_loop, sdk_numeric_contract, sha256)

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = Path('/home/loya2/xr_teleoperate')
SDK = Path('/home/loya2/unitree_sdk2_python')


@pytest.fixture
def source():
    path = REPOSITORY/ARM_SOURCE
    if not path.is_file():
        pytest.skip('Pinned original XR checkout is unavailable')
    return path


def test_compiled_ast_contains_only_reviewed_methods_and_literal_enums(source):
    module, locations = extracted_module(source)
    assert [node.name for node in module.body] == [*ENUMS, 'ExtractedArmMethods']
    assert {node.name for node in module.body[-1].body} == set(METHODS)
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(module))
    assert '__init__' not in locations
    original = ast.parse(source.read_text())
    original_class = next(node for node in original.body if isinstance(node, ast.ClassDef) and node.name=='G1_29_ArmController')
    for method in module.body[-1].body:
        untouched = next(node for node in original_class.body if isinstance(node, ast.FunctionDef) and node.name==method.name)
        assert ast.dump(method, include_attributes=True) == ast.dump(untouched, include_attributes=True)


def test_source_drift_rejected_before_any_extracted_code_executes(source, tmp_path):
    changed = tmp_path/'robot_arm.py'
    changed.write_bytes(source.read_bytes()+b'\nraise RuntimeError("must not execute")\n')
    with pytest.raises(ValueError, match='Unreviewed source hash'):
        controller_fixture(changed, simulation_mode=True)
    assert sha256(source) == ARM_SHA256


def test_recording_publisher_copies_message_at_actual_write_boundary(source):
    controller, clock, _, _ = controller_fixture(source, simulation_mode=True)
    sink = RecordingPublisher(clock)
    sink.Write(controller.msg)
    original = sink.records[0]['motor_cmd'][15]['q']
    controller.msg.motor_cmd[15].q = 99.
    sink.Write(controller.msg)
    assert sink.records[0]['motor_cmd'][15]['q'] == original
    assert sink.records[1]['motor_cmd'][15]['q'] == 99.


@pytest.fixture
def report(source):
    if not (SDK/MOTOR_SOURCE).is_file():
        pytest.skip('Original SDK IDL source is unavailable')
    return audit(REPOSITORY, SDK)


def test_original_submit_aliases_both_arrays_and_publishes_later_mutation(report):
    value = report['aliasing']
    assert value['q_reference_preserved'] and value['tau_reference_preserved']
    assert value['mutation_after_submit_was_published']
    assert value['submitted_q14'] != value['first_published_q14']
    assert value['submitted_tau14'] != value['first_published_tau14']
    assert [index for index,(a,b) in enumerate(zip(value['submitted_q14'],value['first_published_q14'])) if a!=b] == [2]
    assert [index for index,(a,b) in enumerate(zip(value['submitted_tau14'],value['first_published_tau14'])) if a!=b] == [3]


def test_original_loop_repeats_same_unrenewed_target_beyond_one_second(report):
    value = report['repeated_old_target']
    assert value['writes'] == 302
    assert value['last_write_age_since_only_submit_s'] == pytest.approx(301/250)
    assert value['target_updates_after_submit'] == 0
    assert value['all_writes_repeat_same_mutated_target']
    assert value['non_arm_slots_unchanged']
    assert [item['slot'] for item in report['arm_slot_order']] == list(range(15,29))


def test_hardware_branch_changes_q_while_simulation_branch_preserves_it(report):
    value = report['clipping']
    assert value['hardware_branch_changed_q14']
    assert value['hardware_loop_equals_original_clip']
    assert value['simulation_branch_preserves_q14']
    assert value['hardware_branch_preserves_tau'] and value['hardware_branch_zero_dq']
    assert value['max_hardware_step_rad'] == pytest.approx(.08)
    current, target = np.array(value['measured_q14']), np.array(value['input_q14'])
    assert np.argmax(np.abs(target-current)) == 5
    expected = current + (target-current) * (.08 / np.max(np.abs(target-current)))
    assert np.allclose(value['hardware_branch_published_q14'], expected, atol=1e-15, rtol=0)


@pytest.mark.parametrize('method,count,offset', [
    ('get_current_motor_q',35,0.), ('get_current_dual_arm_q',14,0.),
    ('get_current_dual_arm_dq',14,10.)])
def test_getter_can_mix_distinct_complete_source_snapshots(report, method, count, offset):
    value = report['getter_snapshot_mixing'][method]
    assert value['GetData_calls'] == count
    slots = list(range(35)) if count == 35 else list(range(15,29))
    assert value['source_slot_indices'] == slots
    assert value['returned'] == [slot/1000.+offset+(1. if index>=7 else 0.) for index,slot in enumerate(slots)]
    assert value['not_equal_to_either_complete_source_snapshot']
    assert value['single_GetData_control_calls'] == 1
    assert value['single_GetData_control_equals_snapshot_A']


def test_original_loop_populates_actual_q_dq_tau_slots_and_calls_crc_before_write(source):
    controller, clock, _, _ = controller_fixture(source, simulation_mode=True)
    controller.ctrl_dual_arm(np.arange(14)/10., np.arange(14)/20.)
    item = run_loop(controller, clock)[0]
    assert arm_values(item) == (np.arange(14)/10.).tolist()
    assert arm_values(item, 'dq') == [0]*14
    assert arm_values(item, 'tau') == (np.arange(14)/20.).tolist()
    assert item['crc'] == 123456 and controller.crc.calls == 1


def test_static_sdk_float32_contract_is_not_misrepresented_as_wire_test(report):
    contract = report['motor_command_numeric_contract']
    for field in ('q','dq','tau','kp','kd'):
        assert contract['fields'][field] == 'types.float32'
    assert contract['quantization_example']['python_float'] != contract['quantization_example']['numpy_float32_as_python_float']
    assert report['findings_reproduced'] and report['sources_unchanged']
    assert not report['original_crc_or_wire_serialization_verified']
    assert not report['network_or_dds_started'] and not report['hardware_or_webxr_verified']


@pytest.mark.parametrize('ticks', [0,1001,-1,True,1.5])
def test_loop_fixture_requires_bounded_iterations(ticks):
    with pytest.raises(ValueError, match='bounded tick count'):
        FakeTime(ticks)


def test_cli_executes_with_original_imports_network_and_process_spawns_forbidden(source, tmp_path):
    if not (SDK/MOTOR_SOURCE).is_file():
        pytest.skip('Original SDK IDL source is unavailable')
    code = '''import importlib.abc,json,runpy,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'unitree_sdk2py','cyclonedds','logging_mp','robot_control','torch','omni','pxr'}:
   raise RuntimeError('Forbidden original/heavy import: '+fullname)
sys.meta_path.insert(0,Deny())
def reject(event,args):
 if event in {'socket.__new__','subprocess.Popen','os.system','os.fork'}:
  raise RuntimeError('Forbidden effect: '+event)
sys.addaudithook(reject)
sys.argv=['audit_original_xr_send.py','--repository',sys.argv[1],'--sdk-repository',sys.argv[2],'--output',sys.argv[3]]
runpy.run_module('tools.copilot.audit_original_xr_send',run_name='__main__')
'''
    output = tmp_path/'audit.json'
    result = subprocess.run([sys.executable,'-c',code,str(REPOSITORY),str(SDK),str(output)],
                            cwd=ROOT, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['findings_reproduced']
    assert json.loads(output.read_text())['sources_unchanged']
