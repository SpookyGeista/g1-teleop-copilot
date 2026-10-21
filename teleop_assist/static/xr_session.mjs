// Own every acquired session, including one whose renderer setup fails.
export class ImmersiveSessionController {
  constructor(onEnded = () => {}) {
    this.session = null;
    this.busy = false;
    this.onEnded = onEnded;
  }

  finished(session) {
    if (this.session !== session) return;
    this.session = null;
    this.onEnded();
  }

  async end() {
    const session = this.session;
    if (!session) return;
    await session.end();
    this.finished(session);
  }

  async start(xr, initialize) {
    if (this.busy || this.session) return null;
    this.busy = true;
    let acquired = null;
    try {
      acquired = await xr.requestSession('immersive-vr', {optionalFeatures:['local-floor']});
      this.session = acquired;
      acquired.addEventListener('end', () => this.finished(acquired), {once:true});
      await initialize(acquired);
      if (this.session !== acquired) throw new Error('VR-Sitzung wurde beim Start beendet.');
      return acquired;
    } catch (error) {
      if (acquired && this.session === acquired) {
        try { await this.end(); }
        catch (cleanupError) {
          // Keep the reference so the UI can retry ending it, never create
          // another immersive session while this one still belongs to us.
          error.cleanupError = cleanupError;
        }
      }
      throw error;
    } finally {
      this.busy = false;
    }
  }
}
