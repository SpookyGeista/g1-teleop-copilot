#!/usr/bin/env python3
"""Launch the isolated Isaac assisted-teleoperation application."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import signal

ROOT=Path(__file__).resolve().parent
DEFAULT_PYTHON=Path('/home/loya2/miniconda3/envs/unitree_sim_env/bin/python')

def main():
    interpreter=Path(os.environ.get('PUNKTNULL_ISAAC_PYTHON',str(DEFAULT_PYTHON)))
    if not interpreter.is_file():
        raise SystemExit('Set PUNKTNULL_ISAAC_PYTHON to an Isaac Sim 5.1 / Isaac Lab Python interpreter.')
    env=os.environ.copy()
    env['PYTHONUNBUFFERED']='1'
    # Small calibration/pose matrices do not benefit from 24 BLAS workers.
    # Profiling showed ~4x CPU time without a wall-time improvement while
    # Isaac, image tracking and HTTP compete for the same CPU. Set this before
    # NumPy loads, and retain an explicit deployment override.
    env.setdefault('OPENBLAS_NUM_THREADS','1')
    # Kit's skip_cleanup can exit the child with status 0; preserve the
    # application's explicit outcome across that shutdown implementation.
    with tempfile.TemporaryDirectory(prefix='punktnull-assist-exit-') as directory:
        status_file=Path(directory)/'status'
        env['PUNKTNULL_ASSIST_EXIT_FILE']=str(status_file)
        process=subprocess.Popen([str(interpreter),str(ROOT/'tools/assist_sim.py'),*sys.argv[1:]],cwd=ROOT,env=env)
        previous={}
        def forward(signum,frame):
            if process.poll() is None:process.send_signal(signum)
        try:
            for signum in (signal.SIGINT,signal.SIGTERM):
                previous[signum]=signal.signal(signum,forward)
            code=process.wait()
        finally:
            for signum,handler in previous.items():signal.signal(signum,handler)
        return int(status_file.read_text()) if status_file.exists() else code

if __name__=='__main__':
    raise SystemExit(main())
