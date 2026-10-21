"""Actual loopback HTTP tests for isolated Vuer public-file routing.

Only temporary harmless marker files are used; no TLS key or project data is
requested. No headset, GPU, DDS or simulator is started.
"""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from aiohttp import ClientSession, ClientTimeout, web
import pytest
from yarl import URL

from tools.copilot import xr_source
from tools.copilot.instrument_xr import BROWSER_CHUNK


def digest(value):
    return hashlib.sha256(value).hexdigest()


def bundle(tmp_path):
    original=tmp_path/'known_client';original.mkdir()
    stage=tmp_path/'stage';staged=stage/'vuer_client_build';staged.mkdir(parents=True)
    values={'index.html':b'<html>PUBLIC_INDEX</html>',
            'editor/index.html':b'<html>PUBLIC_EDITOR</html>',
            BROWSER_CHUNK:b'ORIGINAL_BROWSER_CHUNK',
            'assets/public.js':b'PUBLIC_JS',
            'not_a_public_route.pem':b'HARMLESS_NONPUBLIC_MARKER'}
    for name,value in values.items():
        source=original/name;source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(value)
        target=staged/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name==BROWSER_CHUNK:target.write_bytes(b'INSTRUMENTED_BROWSER_CHUNK')
        else:target.symlink_to(source)
    (stage/'staging_manifest.json').write_text(json.dumps(dict(schema='punktnull.xr_staging.v1',
        original_browser_sha256=digest(values[BROWSER_CHUNK]),
        staged_browser_sha256=digest(b'INSTRUMENTED_BROWSER_CHUNK'))))
    (stage/'private-marker.txt').write_text('HARMLESS_STAGE_PRIVATE_MARKER')
    (staged/'assets/unrecognized_link.txt').symlink_to(stage/'private-marker.txt')
    return original,stage


def test_public_copy_dereferences_only_known_assets_and_preserves_instrumented_bytes(tmp_path):
    original,stage=bundle(tmp_path)
    client,static=xr_source.prepare_public_web(stage,client_root=original)
    assert list(static.iterdir())==[]
    assert (client/BROWSER_CHUNK).read_bytes()==b'INSTRUMENTED_BROWSER_CHUNK'
    assert not (client/'not_a_public_route.pem').exists()
    assert not (client/'assets/unrecognized_link.txt').exists()
    manifest=json.loads((stage/'public_web_manifest.json').read_text())
    assert set(manifest['files'])=={'index.html','editor/index.html',BROWSER_CHUNK,'assets/public.js'}
    for name,info in manifest['files'].items():
        path=client/name
        assert not path.is_symlink() and digest(path.read_bytes())==info['sha256']
        assert path.stat().st_size==info['bytes']
    # Replacing a staging symlink later cannot redirect the served copy.
    source=stage/'vuer_client_build/assets/public.js';source.unlink();source.symlink_to(stage/'private-marker.txt')
    assert (client/'assets/public.js').read_bytes()==b'PUBLIC_JS'


@pytest.mark.parametrize('fault',['changed_link','changed_chunk','chunk_symlink','external_source_link','changed_source','existing_public_root'])
def test_unreviewed_copy_inputs_are_rejected(tmp_path,fault):
    original,stage=bundle(tmp_path)
    asset=stage/'vuer_client_build/assets/public.js';chunk=stage/'vuer_client_build'/BROWSER_CHUNK
    if fault=='changed_link':asset.unlink();asset.symlink_to(stage/'private-marker.txt')
    elif fault=='changed_chunk':chunk.write_bytes(b'UNREVIEWED_CHUNK')
    elif fault=='chunk_symlink':chunk.unlink();chunk.symlink_to(original/BROWSER_CHUNK)
    elif fault=='external_source_link':
        (original/'assets/public.js').unlink();(original/'assets/public.js').symlink_to(stage/'private-marker.txt')
    elif fault=='changed_source':(original/BROWSER_CHUNK).write_bytes(b'OTHER_ORIGINAL')
    else:(stage/'public_web').mkdir()
    with pytest.raises((ValueError,FileExistsError)):
        xr_source.prepare_public_web(stage,client_root=original)


def test_real_http_serves_public_assets_but_refuses_private_traversal_absolute_and_symlink_escape(tmp_path):
    original,stage=bundle(tmp_path)
    client,static=xr_source.prepare_public_web(stage,client_root=original)
    project_marker=tmp_path/'project-private-marker.txt';project_marker.write_text('HARMLESS_PROJECT_PRIVATE')
    (client/'assets/escape.txt').symlink_to(project_marker)
    (static/'escape.txt').symlink_to(project_marker)
    async def run():
        app=web.Application()
        vuer=SimpleNamespace(_add_route=lambda path,handler,method:app.router.add_route(method,path,handler))
        xr_source.install_public_file_routes(vuer,client,static)
        vuer._add_static('/assets',client/'assets')
        vuer._add_static('/static',static)
        vuer._static_file('/editor',client,'editor/index.html')
        server=web.AppRunner(app);await server.setup()
        try:
            site=web.TCPSite(server,'127.0.0.1',0);await site.start()
            port=site._server.sockets[0].getsockname()[1]
            async with ClientSession(timeout=ClientTimeout(total=3),trust_env=False) as http:
                for path,expected in [('/assets/public.js',b'PUBLIC_JS'),('/assets/'+BROWSER_CHUNK.removeprefix('assets/'),b'INSTRUMENTED_BROWSER_CHUNK'),('/editor',b'<html>PUBLIC_EDITOR</html>')]:
                    async with http.get(URL(f'http://127.0.0.1:{port}'+path,encoded=True)) as response:
                        assert response.status==200 and await response.read()==expected
                from urllib.parse import quote
                bad=['/static/project-private-marker.txt','/static/stage/private-marker.txt',
                    '/static/..%2Fproject-private-marker.txt',
                    '/assets/..%2F..%2F..%2Fprivate-marker.txt',
                    '/assets/%2e%2e%2f%2e%2e%2fprivate-marker.txt',
                    '/assets/'+quote(str(project_marker),safe=''),
                    '/static/'+quote(str(project_marker),safe=''),
                    '/assets/escape.txt','/static/escape.txt','/assets/..%5Cprivate-marker.txt']
                for path in bad:
                    async with http.get(URL(f'http://127.0.0.1:{port}'+path,encoded=True)) as response:
                        assert response.status==404,(path,response.status)
        finally:await server.cleanup()
    asyncio.run(run())


def test_file_routes_cannot_be_registered_against_project_or_staging_root(tmp_path):
    original,stage=bundle(tmp_path)
    client,static=xr_source.prepare_public_web(stage,client_root=original)
    vuer=SimpleNamespace(_add_route=lambda *a,**kw:None)
    xr_source.install_public_file_routes(vuer,client,static)
    with pytest.raises(ValueError):vuer._add_static('/unsafe',stage)
    with pytest.raises(ValueError):vuer._static_file('/unsafe',tmp_path,'project-private-marker.txt')


def test_prepare_wrapper_installs_routes_before_wrapper_construction_without_changing_input_flags(tmp_path,monkeypatch):
    original,stage=bundle(tmp_path)
    module=SimpleNamespace(__file__=str(stage/'teleop/televuer/src/televuer/televuer.py'))
    observed={}
    def real_vuer(*a,**kw):
        observed['vuer_kwargs']=kw
        return SimpleNamespace(_add_route=lambda *a,**kw:None)
    module.Vuer=real_vuer
    def wrapper(**kw):
        observed['wrapper_kwargs']=kw
        return module.Vuer(host='original-bind',cert=kw['cert_file'],key=kw['key_file'])
    monkeypatch.setattr(xr_source.importlib,'import_module',lambda name:module if name=='televuer.televuer' else SimpleNamespace(TeleVuerWrapper=wrapper))
    monkeypatch.setattr(xr_source,'install_instrumentation',lambda root:observed.update(instrumented_root=root))
    prepare=xr_source.prepare_public_web
    monkeypatch.setattr(xr_source,'prepare_public_web',lambda root:prepare(root,client_root=original))
    # prepare_wrapper's sys.path insertion is isolated by the test fixture.
    monkeypatch.setattr(xr_source.sys,'path',list(xr_source.sys.path))
    vuer=xr_source.prepare_wrapper(stage,host='127.0.0.1',port=8012,use_hand_tracking=False,
        cert_file=tmp_path/'unused-cert',key_file=tmp_path/'unused-key')
    assert vuer.client_root==observed['instrumented_root']==stage/'public_web/client'
    assert vuer.static_root==str(stage/'public_web/static')
    assert observed['vuer_kwargs']['host']=='127.0.0.1' and observed['vuer_kwargs']['port']==8012
    assert observed['wrapper_kwargs']['display_mode']=='pass-through'
    assert observed['wrapper_kwargs']['use_hand_tracking'] is False
    assert observed['wrapper_kwargs']['zmq'] is False and observed['wrapper_kwargs']['webrtc'] is False
