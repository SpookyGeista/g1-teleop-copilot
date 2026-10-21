import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ImmersiveSessionController} from '../teleop_assist/static/xr_session.mjs';

class Session extends EventTarget {
  ends = 0;
  async end() { this.ends++; this.dispatchEvent(new Event('end')); }
}

test('setup failure closes the acquired session and permits a retry', async () => {
  for (const stage of ['reference space', 'renderer setup']) {
    const session = new Session(); let finished = 0;
    const owner = new ImmersiveSessionController(() => finished++);
    await assert.rejects(owner.start({requestSession:async () => session}, async () => {
      throw new Error(stage);
    }), new RegExp(stage));
    assert.equal(session.ends, 1); assert.equal(finished, 1);
    assert.equal(owner.session, null); assert.equal(owner.busy, false);
    const next = new Session();
    assert.equal(await owner.start({requestSession:async () => next}, async () => {}), next);
    await owner.end();
  }
});

test('overlapping clicks cannot request a second session', async () => {
  const session = new Session(); let resolve, requests = 0;
  const owner = new ImmersiveSessionController();
  const xr = {requestSession:() => { requests++; return new Promise(r => resolve = r); }};
  const pending = owner.start(xr, async () => {});
  assert.equal(await owner.start(xr, async () => {}), null);
  resolve(session); await pending;
  assert.equal(await owner.start(xr, async () => {}), null);
  assert.equal(requests, 1); await owner.end();
});

test('request rejection does not end a session owned by another tab', async () => {
  const owner = new ImmersiveSessionController();
  await assert.rejects(owner.start({requestSession:async () => {
    throw new Error('already active');
  }}, async () => assert.fail()), /already active/);
  assert.equal(owner.session, null); assert.equal(owner.busy, false);
});

test('failed cleanup retains ownership for an explicit end retry', async () => {
  const session = new Session(), normalEnd = session.end.bind(session);
  session.end = async () => { throw new Error('end failed'); };
  const owner = new ImmersiveSessionController();
  await assert.rejects(owner.start({requestSession:async () => session}, async () => {
    throw new Error('renderer failed');
  }), error => error.message === 'renderer failed' && error.cleanupError.message === 'end failed');
  assert.equal(owner.session, session);
  session.end = normalEnd; await owner.end(); assert.equal(owner.session, null);
});

test('external end during initialization cannot become a successful start', async () => {
  const session = new Session(); let finished = 0;
  const owner = new ImmersiveSessionController(() => finished++);
  await assert.rejects(owner.start({requestSession:async () => session}, async () => {
    await session.end();
  }), /beim Start beendet/);
  assert.equal(owner.session, null); assert.equal(finished, 1);
});
