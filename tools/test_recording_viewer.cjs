#!/usr/bin/env node
/** Verify actual archived robot imagery/CAD boxes in the read-only production viewer. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
const options = {};
for (let i = 0; i < args.length; i += 2) {
  if (!['--recording', '--output', '--start'].includes(args[i]) || !args[i + 1]) {
    throw Error('Usage: node tools/test_recording_viewer.cjs --recording RUN --output NEW_DIR [--start 24]');
  }
  options[args[i].slice(2)] = args[i + 1];
}
if (!options.recording || !options.output) throw Error('--recording and --output are required');
const out = path.resolve(options.output);
fs.mkdirSync(out, { recursive: false });
const python = process.env.PUNKTNULL_ISAAC_PYTHON || '/home/loya2/miniconda3/envs/unitree_sim_env/bin/python';
const replay = spawn(python, ['tools/replay_assist.py', '--recording', path.resolve(options.recording),
  '--port', '0', '--start', options.start || '24', '--rate', '.2', '--loop'], { cwd: root });
const log = fs.createWriteStream(path.join(out, 'replay.log'));
replay.stdout.pipe(log, { end: false });
replay.stderr.pipe(log, { end: false });
let browser;
(async () => {
  const url = await new Promise((resolve, reject) => {
    let text = '';
    const timeout = setTimeout(() => reject(Error('Replay startup timed out')), 30000);
    replay.once('error', error => { clearTimeout(timeout); reject(error); });
    replay.once('exit', code => { clearTimeout(timeout); reject(Error(`Replay exited ${code}`)); });
    replay.stdout.on('data', chunk => {
      text += chunk;
      const match = text.match(/ASSIST_REPLAY_READY (http:\/\/127\.0\.0\.1:\d+)/);
      if (match) { clearTimeout(timeout); resolve(match[1]); }
    });
  });
  let playwright;
  try { playwright = require(process.env.PLAYWRIGHT_PATH || 'playwright'); }
  catch (error) {
    if (process.env.PLAYWRIGHT_PATH) throw error;
    playwright = require('/home/loya2/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
  }
  browser = await playwright.chromium.launch({ headless: true,
    executablePath: process.env.CHROMIUM_PATH || '/usr/bin/google-chrome',
    args: ['--no-sandbox', '--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  const errors = [], writes = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (request.method() !== 'GET') writes.push(request.method() + ' ' + request.url()); });
  await page.goto(url);
  await page.waitForFunction(() => window.__assistDiagnostics?.depthVertices > 1000);
  const state = await page.evaluate(async () => (await fetch('/api/state')).json());
  assert.equal(state.viewer_only, true);
  assert(state.targets.some(target => target.semantic_model && target.cad_overlay_status?.source === 'registered_CAD_bounds'));
  assert(await page.locator('#confirm').isDisabled());
  assert(await page.locator('#calibrate').isDisabled());
  await page.screenshot({ path: path.join(out, 'actual_isaac_camera.png'), fullPage: true });
  await page.click('#view-3d');
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(out, 'actual_isaac_holo_boxes.png'), fullPage: true });
  const diagnostics = await page.evaluate(() => window.__assistDiagnostics);
  assert(diagnostics.boxCount >= 4);
  assert.equal(errors.length, 0, errors.join('\n'));
  assert.equal(writes.length, 0, writes.join('\n'));
  const hashes = Object.fromEntries(['app.mjs', 'geometry.mjs', 'gaze.mjs', 'index.html', 'style.css'].map(name =>
    [name, crypto.createHash('sha256').update(fs.readFileSync(path.join(root, 'teleop_assist/static', name))).digest('hex')]));
  const report = { passed: true, scope: 'Actual recorded Isaac RGBD and learned/CAD targets in read-only desktop viewer',
    recording: path.resolve(options.recording), diagnostics, source_hashes: hashes, headset_verified: false,
    checks: ['actual learned CAD handle displayed', 'registered CAD bounds', 'metric RGBD and at least four cuboids',
      'read-only controls', 'zero write requests', 'zero page errors'] };
  fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report));
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(async () => {
  if (browser) await browser.close();
  replay.kill('SIGTERM');
  await new Promise(resolve => {
    if (replay.exitCode !== null) return resolve();
    const timeout = setTimeout(() => { replay.kill('SIGKILL'); resolve(); }, 5000);
    replay.once('exit', () => { clearTimeout(timeout); resolve(); });
  });
  log.end();
});
