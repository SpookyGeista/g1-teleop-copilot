"""Local source packaging must preserve geometry and truthful scope."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import hashlib

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('copilot_package',ROOT/'tools/copilot/package_release.py')
PACKAGE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(PACKAGE)


@pytest.mark.parametrize('value',[{},[],{'hardware_tested':True},{'hardware_tested':'false'},
    {'hardware_tested':False,'hardware_ready':True},
    {'hardware_tested':False,'checks':[{'motion_ready':True}]},
    {'hardware_tested':False,'hardware_authorized':'yes'}])
def test_acceptance_cannot_upgrade_to_hardware_or_motion_readiness(value):
    with pytest.raises(ValueError):PACKAGE.validate_acceptance(value)


def test_synthetic_acceptance_remains_explicitly_synthetic():
    result=PACKAGE.validate_acceptance({'hardware_tested':False,'synthetic':True,'status':'PACKAGING_TEST_ONLY'})
    assert result['synthetic'] and result['status']=='PACKAGING_TEST_ONLY'
    assert result['hardware_authorized'] is False and result['hardware_ready'] is False
    assert result['packaging_scope']=='simulation_source_and_evidence_only'


@pytest.mark.parametrize('name',['teleop_assist/.venv-models/lib/x.py','teleop_assist/.private/config.json',
    'teleop_assist/models/config.json','tools/copilot/certificates/server.json',
    'tools/copilot/credentials.json','tools/copilot/secrets.txt','runs/demo/access_token.json'])
def test_environment_model_certificate_and_credential_paths_excluded(name):
    assert PACKAGE.excluded(Path(name))


def test_current_source_bundle_contains_all_descriptor_assets_and_licenses():
    entries=PACKAGE.sources();PACKAGE.verify_descriptor(entries);names={n for n,_ in entries}
    for name in ('vis_mesh.obj','coll_mesh.obj','UNITREE_ASSETS_LICENSE.txt'):
        assert 'teleop_assist/copilot/assets/dex3_right/'+name in names
    assert 'research/masters_20261003/ai_assistance/dex3_descriptor/presets.json' in names
    assert 'research/masters_20261003/ai_assistance/dex3_descriptor/UNITREE_ASSETS_LICENSE.txt' in names
    assert 'teleop_assist/static/vendor/THREE-LICENSE.txt' in names
    assert 'tools/setup_models.py' in names  # setup --install calls this helper
    assert all(not PACKAGE.excluded(Path(n)) for n in names)
    assert all(Path(n).suffix.lower() not in {'.pem','.key','.crt','.pth','.safetensors','.whl'} for n in names)


def test_missing_or_changed_required_descriptor_asset_refuses_bundle():
    entries=PACKAGE.sources();name='teleop_assist/copilot/assets/dex3_right/coll_mesh.obj'
    with pytest.raises(ValueError,match='missing or changed'):
        PACKAGE.verify_descriptor([(n,d) for n,d in entries if n!=name])
    with pytest.raises(ValueError,match='missing or changed'):
        PACKAGE.verify_descriptor([(n,b'wrong' if n==name else d) for n,d in entries])


def test_xr_check_subprocess_imports_bundled_sibling_after_relocation(tmp_path):
    # Run the real standalone entry point from only the selected release files.
    # --help loads its top-level shadow_bridge dependency without importing the
    # original robot launcher, constructing IK or starting DDS/server processes.
    relocated=tmp_path/'extracted'
    for name,data in PACKAGE.sources():
        path=relocated/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    probe=relocated/'research/masters_20261003/ai_assistance/xr_bridge/probe_original_ik.py'
    environment=dict(os.environ,PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,str(probe),'--help'],cwd=tmp_path,env=environment,
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert '--repo' in result.stdout and '--output' in result.stdout


@pytest.mark.parametrize('missing',['probe_original_ik.py','shadow_bridge.py'])
def test_missing_xr_check_runtime_helper_refuses_release(tmp_path,missing):
    for name,data in PACKAGE.sources():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    helper=tmp_path/'research/masters_20261003/ai_assistance/xr_bridge'/missing
    helper.unlink()
    with pytest.raises(ValueError,match='Missing release sources: .*'+missing.replace('.','\\.')):
        PACKAGE.sources(tmp_path)


def configure_packaging_call(monkeypatch,tmp_path,evidence):
    entries=PACKAGE.sources()
    acceptance=tmp_path/'acceptance_input.json'
    acceptance.write_text(json.dumps({'hardware_tested':False,'status':'PACKAGING_TEST_ONLY'}))
    output=tmp_path/'bundle.zip'
    monkeypatch.setattr(PACKAGE,'ROOT',tmp_path)
    monkeypatch.setattr(PACKAGE,'sources',lambda:entries)
    monkeypatch.setattr(sys,'argv',['package_release.py','--output',str(output),
        '--acceptance',str(acceptance),'--evidence',str(evidence)])
    return output


def test_explicit_jpg_review_evidence_is_archived_with_exact_manifest(monkeypatch,tmp_path):
    evidence=tmp_path/'runs/review/frame.jpg';evidence.parent.mkdir(parents=True)
    content=b'\xff\xd8\xff\xe0explicit-test-image\xff\xd9';evidence.write_bytes(content)
    output=configure_packaging_call(monkeypatch,tmp_path,evidence)
    PACKAGE.main()
    with zipfile.ZipFile(output) as archive:
        assert archive.read('runs/review/frame.jpg')==content
        manifest=json.loads(archive.read('SHA256_MANIFEST.json'))
        assert manifest['runs/review/frame.jpg']=={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}


@pytest.mark.parametrize('fault',['symlink','oversized','excluded'])
def test_jpg_evidence_retains_existing_package_restrictions(monkeypatch,tmp_path,fault):
    evidence=tmp_path/('access_token.jpg' if fault=='excluded' else 'frame.jpg')
    if fault=='symlink':
        source=tmp_path/'source.jpg';source.write_bytes(b'test');evidence.symlink_to(source)
    elif fault=='oversized':
        with evidence.open('wb') as stream:stream.truncate(PACKAGE.MAX_EVIDENCE_BYTES+1)
    else:evidence.write_bytes(b'test')
    output=configure_packaging_call(monkeypatch,tmp_path,evidence)
    with pytest.raises(SystemExit):PACKAGE.main()
    assert not output.exists()
