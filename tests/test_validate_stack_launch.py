"""Launcher resolution and failure reporting; no subprocesses or browser run."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


spec=importlib.util.spec_from_file_location('validate_stack',Path(__file__).resolve().parents[1]/'tools/validate_stack.py')
tool=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_explicit_node_override_is_not_silently_replaced(monkeypatch):
    monkeypatch.setattr(tool.shutil,'which',lambda name:'/a/different/node')
    assert tool.resolve_node('/requested/missing-node')==('/requested/missing-node','explicit')


def test_node_path_precedes_installed_runtime_and_fallback_requires_executable(tmp_path,monkeypatch):
    root=tmp_path/'runtime'
    binary=root/'bin/node'
    binary.parent.mkdir(parents=True)
    binary.write_text('test fixture only')
    binary.chmod(0o755)
    monkeypatch.setattr(tool,'CODEX_NODE_ROOT',root)
    monkeypatch.setattr(tool.shutil,'which',lambda name:'/path/node')
    assert tool.resolve_node()==('/path/node','PATH')
    monkeypatch.setattr(tool.shutil,'which',lambda name:None)
    assert tool.resolve_node()==(str(binary),'installed_codex_runtime')
    binary.chmod(0o644)
    assert tool.resolve_node()==('node','unavailable')


def test_playwright_override_and_environment_win_over_installed_fallback(tmp_path,monkeypatch):
    root=tmp_path/'runtime'
    package=root/'node_modules/playwright/package.json'
    package.parent.mkdir(parents=True)
    package.write_text('{}')
    monkeypatch.setattr(tool,'CODEX_NODE_ROOT',root)
    monkeypatch.setattr(tool,'ROOT',tmp_path/'project')
    monkeypatch.delenv('NODE_PATH',raising=False)
    monkeypatch.setenv('PLAYWRIGHT_PATH','/environment/playwright')
    assert tool.resolve_playwright('/explicit/playwright')==('/explicit/playwright','explicit')
    assert tool.resolve_playwright()==('/environment/playwright','PLAYWRIGHT_PATH')
    monkeypatch.delenv('PLAYWRIGHT_PATH')
    assert tool.resolve_playwright()==(str(package.parent),'installed_codex_runtime')


def test_missing_node_writes_final_failure_report_after_prior_check(tmp_path,monkeypatch):
    output=tmp_path/'validation'
    calls=[]

    def run(command,**kwargs):
        calls.append(command)
        if len(calls)==1:return SimpleNamespace(returncode=0)
        raise FileNotFoundError(2,'No such file or directory',command[0])

    monkeypatch.setattr(tool.subprocess,'run',run)
    code=tool.main(['--browser','--node','/requested/missing-node','--output',str(output)])
    report=json.loads((output/'report.json').read_text())
    assert code==2 and report['passed'] is False
    assert report['browser_dependencies']['node_source']=='explicit'
    assert report['checks'][0]['passed']
    failed=report['checks'][1]
    assert failed['name']=='browser_fixture' and failed['command'][0]=='/requested/missing-node'
    assert failed['exit_code'] is None and not failed['passed']
    assert failed['launch_error']['type']=='FileNotFoundError'
    assert failed['launch_error']['errno']==2
    assert '/requested/missing-node' in (output/'browser_fixture.log').read_text()
    assert report['headset_tested'] is False


def test_python_permission_error_also_writes_failed_final_report(tmp_path,monkeypatch):
    output=tmp_path/'validation'

    def run(command,**kwargs):
        raise PermissionError(13,'Permission denied',command[0])

    monkeypatch.setattr(tool.subprocess,'run',run)
    code=tool.main(['--output',str(output)])
    report=json.loads((output/'report.json').read_text())
    assert code==2 and not report['passed']
    assert len(report['checks'])==1
    assert report['checks'][0]['launch_error']['type']=='PermissionError'
