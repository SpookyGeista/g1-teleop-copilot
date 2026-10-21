"""Rendering configuration contract; image alignment needs the actual GPU probe."""
import pytest
from teleop_assist.copilot.render_sync import ZERO_DELAY_SETTINGS,verify_render_sync,zero_delay_kit_args


def test_preserves_unrelated_kit_options_and_requires_both_documented_settings():
    args=zero_delay_kit_args('--/renderer/multiGpu/enabled=0')
    assert args.startswith('--/renderer/multiGpu/enabled=0 ')
    assert '--/app/hydraEngine/waitIdle=1' in args
    assert '--/app/updateOrder/checkForHydraRenderComplete=1000' in args
    assert verify_render_sync(dict(ZERO_DELAY_SETTINGS))==ZERO_DELAY_SETTINGS


@pytest.mark.parametrize('actual',[{}, {'/app/hydraEngine/waitIdle':True},
    dict(ZERO_DELAY_SETTINGS,**{'/app/hydraEngine/waitIdle':False}),
    dict(ZERO_DELAY_SETTINGS,**{'/app/updateOrder/checkForHydraRenderComplete':0})])
def test_missing_or_disabled_sync_is_rejected(actual):
    with pytest.raises(RuntimeError):verify_render_sync(actual)


def test_conflicting_cli_override_cannot_silently_create_misbound_camera_frames():
    with pytest.raises(ValueError):zero_delay_kit_args('--/app/hydraEngine/waitIdle=0')
