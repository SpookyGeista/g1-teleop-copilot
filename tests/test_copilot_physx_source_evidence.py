"""Exercise the actual configuration loader without importing Isaac or Torch."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import sys
import time
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1]/'teleop_assist/copilot/physx_bench.py'


@pytest.fixture
def loaded_source():
    tree = ast.parse(SOURCE.read_bytes(), filename=str(SOURCE))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in ('load_file', 'PhysXOrientationBench')]
    namespace = dict(hashlib=hashlib, importlib=importlib, Path=Path, time=time,
                     __package__='teleop_assist.copilot')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


def test_records_exact_executed_source_and_module_file(loaded_source, tmp_path):
    path = tmp_path/'configuration.py'
    content = b'answer = 6 * 7\nloaded_file = __file__\n'
    path.write_bytes(content)
    manifest = dict(files={})
    module = loaded_source['load_file']('test_configuration', path, source_manifest=manifest)
    entry = manifest['files'][str(path.resolve())]
    assert module.answer == 42 and module.loaded_file == str(path.resolve())
    assert entry['module_name'] == 'test_configuration'
    assert entry['sha256'] == hashlib.sha256(content).hexdigest()
    assert entry['source_bytes'] == len(content)
    assert entry['read_at_host_monotonic'] <= entry['loaded_at_host_monotonic']
    assert entry['load_method'] == 'compile_and_exec_captured_source_bytes'
    json.dumps(manifest, allow_nan=False)


def test_timestamp_valid_stale_bytecode_cannot_disagree_with_recorded_source(loaded_source, tmp_path):
    path = tmp_path/'cached_configuration.py'
    path.write_bytes(b"value = 'old'\n")
    original_stat = path.stat()
    py_compile.compile(str(path), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    content = b"value = 'new'\n"
    assert len(content) == original_stat.st_size
    path.write_bytes(content)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    # Establish the actual counterexample: ordinary timestamp-based loading
    # accepts the old cached code even though these source bytes changed.
    spec = importlib.util.spec_from_file_location('stale_control', path)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    assert control.value == 'old'
    manifest = dict(files={})
    module = loaded_source['load_file']('fresh_configuration', path, source_manifest=manifest)
    assert module.value == 'new'
    assert manifest['files'][str(path)]['sha256'] == hashlib.sha256(content).hexdigest()


def test_source_changed_during_execution_retains_executed_bytes_hash(loaded_source, tmp_path):
    path = tmp_path/'self_changing_configuration.py'
    content = (b'from pathlib import Path\n'
               b'Path(__file__).write_text("value = 99\\n")\nvalue = 7\n')
    path.write_bytes(content)
    manifest = dict(files={})
    module = loaded_source['load_file']('changing_configuration', path, source_manifest=manifest)
    assert module.value == 7 and path.read_bytes() != content
    assert manifest['files'][str(path)]['sha256'] == hashlib.sha256(content).hexdigest()


def test_failed_configuration_is_not_recorded_as_successfully_loaded(loaded_source, tmp_path):
    path = tmp_path/'broken_configuration.py'
    path.write_text('raise RuntimeError("configuration failed")\n')
    manifest = dict(files={})
    with pytest.raises(RuntimeError, match='configuration failed'):
        loaded_source['load_file']('broken_configuration', path, source_manifest=manifest)
    assert manifest['files'] == {}


def test_startup_captures_both_configurations_before_simulation_creation(loaded_source, tmp_path, monkeypatch):
    files = ('robots/unitree.py', 'tasks/common_config/camera_configs.py')
    for relative in files:
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('configuration_loaded = True\n')
    monkeypatch.setitem(sys.modules, 'carb', SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: {})))
    monkeypatch.setitem(sys.modules, 'teleop_assist.copilot.render_sync',
                        SimpleNamespace(verify_render_sync=lambda settings: {'verified': True}))

    class StopBeforeSimulator(RuntimeError):
        pass

    def stop_before_simulator(*args, **kwargs):
        raise StopBeforeSimulator

    loaded_source['sim_utils'] = SimpleNamespace(SimulationCfg=lambda **kwargs: kwargs,
                                                 SimulationContext=stop_before_simulator)
    bench = loaded_source['PhysXOrientationBench'].__new__(loaded_source['PhysXOrientationBench'])
    with pytest.raises(StopBeforeSimulator):
        bench.__init__(tmp_path, tmp_path/'unused_xr', {}, camera_source='right_wrist')
    manifest = bench.external_source_manifest
    assert set(manifest['files']) == {str((tmp_path/relative).resolve()) for relative in files}
    for path, entry in manifest['files'].items():
        assert entry['sha256'] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert manifest['captured_at_host_monotonic'] <= entry['read_at_host_monotonic']
    assert 'historical run verification' in manifest['scope']
    assert 'copilot_sim_original_003' in bench.open_geometry_limitations[0]
    assert 'remain unresolved' in bench.open_geometry_limitations[0]
    json.dumps(manifest, allow_nan=False)
