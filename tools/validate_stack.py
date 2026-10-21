#!/usr/bin/env python3
"""Repeatable validation entry point, preserving separate evidence for each layer."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CODEX_NODE_ROOT=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node'


def resolve_node(explicit=None):
    """Honor an override even if invalid; never silently substitute another Node."""
    if explicit is not None:return str(Path(explicit).expanduser()) if '/' in explicit else explicit,'explicit'
    discovered=shutil.which('node')
    if discovered:return discovered,'PATH'
    bundled=CODEX_NODE_ROOT/'bin/node'
    if bundled.is_file() and os.access(bundled,os.X_OK):return str(bundled),'installed_codex_runtime'
    # Keep the attempted program in the failure report. subprocess.run will
    # provide the concrete launch error; resolution never installs anything.
    return 'node','unavailable'


def resolve_playwright(explicit=None):
    if explicit is not None:return explicit,'explicit'
    if os.environ.get('PLAYWRIGHT_PATH'):return os.environ['PLAYWRIGHT_PATH'],'PLAYWRIGHT_PATH'
    script_directory=ROOT/'teleop_assist/static/tests'
    candidates=[directory/'node_modules/playwright' for directory in (script_directory,*script_directory.parents)]
    candidates.extend(Path(directory)/'playwright' for directory in os.environ.get('NODE_PATH','').split(os.pathsep) if directory)
    for candidate in candidates:
        if (candidate/'package.json').is_file():return str(candidate),'node_module_path'
    bundled=CODEX_NODE_ROOT/'node_modules/playwright'
    if (bundled/'package.json').is_file():return str(bundled),'installed_codex_runtime'
    return None,'node_module_resolution'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--isaac',action='store_true',help='Run real learned-perception/action acceptance in Isaac')
    p.add_argument('--vision-grasp',action='store_true',help='Also attempt the image-guided physical handle grasp/pull in Isaac')
    p.add_argument('--browser',action='store_true')
    p.add_argument('--node',help='Explicit Node executable; otherwise PATH then installed Codex runtime')
    p.add_argument('--playwright-path')
    a=p.parse_args(argv)
    output=(a.output or ROOT/'runs'/datetime.now().strftime('stack_validation_%Y%m%d_%H%M%S')).resolve()
    output.mkdir(parents=True,exist_ok=False)
    python=os.environ.get('PUNKTNULL_ISAAC_PYTHON','/home/loya2/miniconda3/envs/unitree_sim_env/bin/python')
    report={'created_at':datetime.now(timezone.utc).isoformat(),'checks':[],
            'headset_tested':False,'real_robot_tested':False}
    commands=[('python_tests',[python,'-B','-m','pytest','tests','-q'])]
    if a.browser:
        node,node_source=resolve_node(a.node)
        playwright,playwright_source=resolve_playwright(a.playwright_path)
        report['browser_dependencies']={'node':node,'node_source':node_source,
            'playwright_path':playwright,'playwright_source':playwright_source}
        command=[node,'tools/test_assist_frontend.cjs','--fixture','--output',str(output/'browser')]
        if playwright:command+=['--playwright-path',playwright]
        commands.append(('browser_fixture',command))
    if a.isaac:
        commands.append(('isaac_action_acceptance',[sys.executable,'run_assist.py','--headless',
            '--stack-acceptance','--port','0','--output',str(output/'isaac')]))
    if a.vision_grasp:
        commands.append(('isaac_vision_grasp_acceptance',[sys.executable,'run_assist.py','--headless',
            '--vision-grasp-acceptance','--port','0','--output',str(output/'vision_grasp')]))
    for name,command in commands:
        print('VALIDATING '+name,flush=True)
        launch_error=None
        with (output/(name+'.log')).open('w') as log:
            try:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
                returncode=result.returncode
            except OSError as error:
                returncode=None
                launch_error={'type':type(error).__name__,'message':str(error),'errno':error.errno}
                log.write('Could not launch validation command: '+str(error)+'\n')
        entry={'name':name,'exit_code':returncode,'command':command,
               'log':name+'.log','passed':returncode==0 and launch_error is None}
        if launch_error is not None:entry['launch_error']=launch_error
        report['checks'].append(entry)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if not entry['passed']:break
    report['passed']=len(report['checks'])==len(commands) and all(x['passed'] for x in report['checks'])
    report['scope']='Software/fixture checks plus only explicitly requested simulation layers; no headset validation'
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('VALIDATION_REPORT '+str(output/'report.json'))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
