#!/usr/bin/env python3
"""Reproduce the optional isolated FoundationPose environment and native builds.

--install explicitly downloads dependencies/weights and compiles native code.
The default only reports readiness. No parent Isaac packages are modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from teleop_assist.foundation_pose import FOUNDATIONPOSE_REVISION, FOUNDATIONPOSE_WEIGHTS, readiness
from teleop_assist.model_registry import sha256

PYTORCH3D_REV='978cd99221b9e0a6a568f1d427854d73363265cf'
NVDIFFRAST_REV='253ac4fcea7de5f396371124af597e6cc957bfae'


def run(args,**kwargs):
    print('+ '+' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),check=True,cwd=ROOT,**kwargs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install',action='store_true')
    p.add_argument('--base-python',type=Path,default=Path('/home/loya2/miniconda3/envs/unitree_sim_env/bin/python'))
    p.add_argument('--conda',type=Path,default=Path('/home/loya2/miniconda3/bin/conda'))
    p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args()
    python=ROOT/'.venv-pose/bin/python';source=ROOT/'third_party/FoundationPose';cuda=ROOT/'.cuda-toolkit'
    if args.install:
        if not python.exists():
            run([args.base_python,'-m','venv','--system-site-packages',ROOT/'.venv-pose'])
        if not source.exists():
            run(['git','clone','https://github.com/NVlabs/FoundationPose.git',source])
            run(['git','-C',source,'checkout',FOUNDATIONPOSE_REVISION])
        revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if revision!=FOUNDATIONPOSE_REVISION:
            raise RuntimeError('Existing FoundationPose checkout differs from pinned revision; no automatic reset performed')
        run([python,'-m','pip','install','-r',ROOT/'requirements-pose.txt'])
        if not (cuda/'bin/nvcc').is_file():
            run([args.conda,'create','--prefix',cuda,'-c','nvidia/label/cuda-12.8.1','-c','conda-forge',
                 'cuda-nvcc=12.8.93','cuda-cudart-dev=12.8.90','cuda-cccl=12.8.90','--yes'])
        nvidia=Path(subprocess.check_output([str(python),'-c',
             'import torch;from pathlib import Path;print(Path(torch.__file__).resolve().parent.parent/"nvidia")'],text=True).strip())
        includes=[str(cuda/'targets/x86_64-linux/include')]+[
             str(nvidia/x/'include') for x in ('cublas','cusparse','cusolver','curand','cufft')]
        env=dict(os.environ,CUDA_HOME=str(cuda),MAX_JOBS=str(args.jobs),TORCH_CUDA_ARCH_LIST='8.9',
                 CPATH=':'.join(includes),LIBRARY_PATH=str(cuda/'targets/x86_64-linux/lib'),
                 PATH=str(python.parent)+':'+os.environ.get('PATH',''))
        run([python,'-m','pip','install','--no-build-isolation',
             f'git+https://github.com/facebookresearch/pytorch3d.git@{PYTORCH3D_REV}',
             f'git+https://github.com/NVlabs/nvdiffrast.git@{NVDIFFRAST_REV}'],env=env)
        pybind=subprocess.check_output([str(python),'-m','pybind11','--cmakedir'],text=True).strip()
        run(['cmake','-S',source/'mycpp','-B',source/'mycpp/build','-DCMAKE_BUILD_TYPE=Release',
             f'-Dpybind11_DIR={pybind}',f'-DPYTHON_EXECUTABLE={python}'])
        run(['cmake','--build',source/'mycpp/build','-j',args.jobs])
        if not all((source/'weights'/name/'model_best.pth').is_file() for name in FOUNDATIONPOSE_WEIGHTS):
            run([python,'-c','import gdown;gdown.download_folder(url="https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i",output="third_party/FoundationPose/weights",quiet=False)'])
        files={str(path.relative_to(source)):{'sha256':sha256(path),'bytes':path.stat().st_size}
               for path in sorted((source/'weights').glob('*/*')) if path.is_file()}
        manifest={'source':'https://github.com/NVlabs/FoundationPose','revision':revision,
                  'license':'NVIDIA Source Code License; see upstream LICENSE',
                  'weights_source':'https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i',
                  'weights':files,'native_sources':{'pytorch3d':PYTORCH3D_REV,'nvdiffrast':NVDIFFRAST_REV},
                  'toolchain':'CUDA12.8.93, local .cuda-toolkit; sm89 RTX4090 build',
                  'inference_verified':False}
        (ROOT/'models/foundationpose_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if python.exists() and Path(sys.executable).absolute()!=python.absolute():
        run([python,__file__])
    else:
        print(json.dumps(readiness(),indent=2))


if __name__=='__main__':
    main()
