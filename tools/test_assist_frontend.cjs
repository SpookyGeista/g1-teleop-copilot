#!/usr/bin/env node
/** Portable launcher for the actual-server or explicitly synthetic UI checks. */
const fs=require('node:fs'),path=require('node:path'),{spawnSync}=require('node:child_process');
const root=path.resolve(__dirname,'..'),args=process.argv.slice(2),options={};
const help=`Usage: node tools/test_assist_frontend.cjs [--url http://127.0.0.1:8765] [--output NEW_DIRECTORY]
       node tools/test_assist_frontend.cjs --fixture [--output NEW_DIRECTORY]
       node tools/test_assist_frontend.cjs --geometry
Options: --playwright-path MODULE_PATH --chromium EXECUTABLE
Actual-server mode changes selection, toggles eyes/assist OFF after testing, and
calibrates the virtual hand. It does not request arm motion. Run only while no
other control client or automated controller owns the session. Fixture mode
starts only a local HTTP fixture and labels its evidence as synthetic.
Playwright must be available via NODE_PATH, local node_modules, or --playwright-path.
Chrome may be supplied using --chromium; otherwise installed Chrome/Playwright is used.
`;
for(let i=0;i<args.length;i++){const a=args[i];if(['--help','-h'].includes(a)){console.log(help);process.exit(0);}if(['--fixture','--geometry'].includes(a)){options[a.slice(2)]=true;continue;}if(['--url','--output','--playwright-path','--chromium'].includes(a)&&args[i+1])options[a.slice(2)]=args[++i];else{console.error(help);process.exit(2);}}
const dir=path.join(root,'teleop_assist/static/tests');if(options.geometry){const r=spawnSync(process.execPath,[path.join(dir,'geometry.test.mjs')],{stdio:'inherit'});process.exit(r.status??1);}
const url=options.url||process.env.ASSIST_URL||'http://127.0.0.1:8765';if(!['http:','https:'].includes(new URL(url).protocol))throw Error('HTTP(S) viewer URL required');
const stamp=new Date().toISOString().replace(/[:.]/g,'-'),out=path.resolve(options.output||path.join(root,'runs',`assist_ui_${options.fixture?'fixture':'live'}_${stamp}`));
fs.mkdirSync(path.dirname(out),{recursive:true});fs.mkdirSync(out,{recursive:false}); // never overwrite earlier evidence
if(options.fixture)fs.copyFileSync(path.join(dir,'fixture.jpg'),path.join(out,'fixture.jpg'));
const env={...process.env,ASSIST_URL:url,ASSIST_TEST_OUTPUT:out};if(options['playwright-path'])env.PLAYWRIGHT_PATH=options['playwright-path'];if(options.chromium)env.CHROMIUM_PATH=options.chromium;
const script=options.fixture?'browser.test.cjs':'live_browser.cjs';const result=spawnSync(process.execPath,[path.join(dir,script)],{env,stdio:'inherit'});console.log('UI evidence:',out);process.exit(result.status??1);
