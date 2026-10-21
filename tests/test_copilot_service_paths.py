"""The real service CLI must hand configured asset paths to both workers."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('surface_primary',[False,True])
def test_service_workers_use_configured_external_assets(monkeypatch,tmp_path,surface_primary):
    recorded={}
    class Model:
        manifest={'status':'ready','backend':'test_perception_worker'}
        def __init__(self,**kwargs):recorded['perception']=kwargs
        def close(self):pass
    class Grasp:
        manifest={'backend':'test_grasp_worker','motion_output':False}
        def __init__(self,**kwargs):recorded['grasp']=kwargs
        def close(self):pass
    class Perception:
        def __init__(self,model,*,selected_surface_primary=False):
            recorded['selected_surface_primary']=selected_surface_primary
        def close(self):pass
    class Engine:
        def __init__(self,*args,**kwargs):pass
    class Session:
        def __init__(self,*args):pass
        def poll(self):pass
        def close(self):pass
    monkeypatch.setitem(sys.modules,'teleop_assist.model_bridge',SimpleNamespace(ModelProcess=Model))
    monkeypatch.setitem(sys.modules,'teleop_assist.perception_runtime',SimpleNamespace(AsyncPerception=Perception))
    monkeypatch.setitem(sys.modules,'teleop_assist.copilot.process',SimpleNamespace(GraspProcess=Grasp))
    monkeypatch.setitem(sys.modules,'teleop_assist.copilot.service',SimpleNamespace(ProposalEngine=Engine,ProposalSession=Session))
    root=Path(__file__).resolve().parents[1]
    specification=importlib.util.spec_from_file_location('copilot_service_paths',root/'tools/copilot/serve.py')
    service=importlib.util.module_from_spec(specification);specification.loader.exec_module(service)
    monkeypatch.setattr(service.signal,'signal',lambda *args:None)
    repository=tmp_path/'outside GraspGenX';weights=tmp_path/'outside detector weights'
    runtime=tmp_path/'runtime';runtime.mkdir()
    monkeypatch.setattr(sys,'argv',['serve.py','--no-dashboard','--live-frames',str(tmp_path/'frames'),
        '--runtime',str(runtime),'--duration','.01','--model-python','external_perception_python',
        '--grasp-python','external_grasp_python','--grasp-repo',str(repository),'--model-root',str(weights)]
        +(['--complete-static-sim-preview'] if surface_primary else []))
    service.main()
    assert recorded['perception']['python']=='external_perception_python'
    assert recorded['perception']['model_root']==str(weights)
    assert recorded['grasp']['python']=='external_grasp_python'
    assert recorded['grasp']['repository']==str(repository)
    manifest=json.loads((runtime/'proposal_source_manifest.json').read_text())
    assert manifest['configured_model_paths']==dict(grasp_repository=str(repository),model_root=str(weights))
    assert manifest['learned_model_manifest']==dict(grasp=Grasp.manifest,perception=Model.manifest)
    assert recorded['selected_surface_primary'] is surface_primary
    assert manifest['selected_surface_primary'] is surface_primary
    assert manifest['complete_static_sim_preview'] is surface_primary
