"""Startup settings for same-state Isaac 5.1 camera observations.

NVIDIA documents up to three in-flight frames in the default rendering app.
Pairing that delayed image with current robot FK produces false object motion.
These are the rendering settings from isaacsim.exp.base.zero_delay.kit, applied
before AppLauncher starts Kit. No robot, transport or Isaac imports here.
"""
ZERO_DELAY_SETTINGS = {
    '/app/hydraEngine/waitIdle': True,
    '/app/updateOrder/checkForHydraRenderComplete': 1000,
}


def zero_delay_kit_args(existing=''):
    if not isinstance(existing,str):
        raise ValueError('Kit arguments must be text')
    # Refuse conflicting user overrides rather than depending on duplicate
    # command-line argument precedence inside Kit.
    if any(path in existing for path in ZERO_DELAY_SETTINGS):
        raise ValueError('Camera synchronization settings are owned by the isolated bench')
    return (existing+' --/app/hydraEngine/waitIdle=1'
        ' --/app/updateOrder/checkForHydraRenderComplete=1000').strip()


def verify_render_sync(settings):
    actual={key:settings.get(key) for key in ZERO_DELAY_SETTINGS}
    if actual!=ZERO_DELAY_SETTINGS:
        raise RuntimeError('Same-state camera capture requires zero-delay Kit startup settings')
    return actual
