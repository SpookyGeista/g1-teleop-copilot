from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from teleop_assist.copilot.sim_snapshot import SimSnapshotTransaction


def sample():
    state = dict(clock_id='sim', acquisition_timestamp=10., q=[0.]*43, base=[0., 0., 1.])
    scene = dict(clock_id='sim', acquisition_timestamp=10., obstacles=[{'position':[1.,0.,0.]}])
    return state, scene


def test_new_reads_prove_continuity_without_renewing_original_capture():
    state, scene = sample()
    captured = SimSnapshotTransaction.capture(state, scene, physics_step=42, source_instance_id='run')
    for timestamp in (10.01, 10.05, 100.):
        state['acquisition_timestamp'] = scene['acquisition_timestamp'] = timestamp
        read_state, read_scene = captured.read(state, scene, physics_step=42, source_instance_id='run')
        assert read_state['acquisition_timestamp'] == read_scene['acquisition_timestamp'] == 10.
        read_state['q'][0] = 99.  # returned objects never alias the private anchor
    assert captured.read(state,scene,physics_step=42,source_instance_id='run')[0]['q'][0] == 0.
    # Expiry belongs to the consumer. The old timestamp is still 10, not 100.


@pytest.mark.parametrize('change', ['q', 'base', 'obstacle', 'clock', 'step', 'source', 'nonfinite', 'earlier_read'])
def test_any_actual_change_invalidates_the_command_transaction(change):
    state, scene = sample()
    captured = SimSnapshotTransaction.capture(state, scene, physics_step=42, source_instance_id='run')
    step, source = 42, 'run'
    if change == 'q': state['q'][42] = .001
    if change == 'base': state['base'][2] = 1.00001
    if change == 'obstacle': scene['obstacles'][0]['position'][0] = 2.
    if change == 'clock': state['clock_id'] = 'new'
    if change == 'step': step += 1
    if change == 'source': source = 'new'
    if change == 'nonfinite': scene['obstacles'][0]['position'][0] = float('nan')
    if change == 'earlier_read': state['acquisition_timestamp'] = 9.99
    with pytest.raises(ValueError):
        captured.read(state, scene, physics_step=step, source_instance_id=source)


def test_capture_copies_inputs_and_is_frozen():
    state, scene = sample(); saved = deepcopy((state, scene))
    captured = SimSnapshotTransaction.capture(state, scene, physics_step=42, source_instance_id='run')
    state['q'][0] = 1.
    with pytest.raises(FrozenInstanceError): captured.physics_step = 43
    assert captured.read(*saved, physics_step=42,source_instance_id='run') == saved


@pytest.mark.parametrize('change', ['mismatched_time','unknown_clock','boolean_time','nonfinite_time','boolean_step'])
def test_capture_requires_explicit_coherent_acquisition(change):
    state, scene = sample(); step = 42
    if change == 'mismatched_time': scene['acquisition_timestamp'] += .1
    if change == 'unknown_clock': state['clock_id'] = None
    if change == 'boolean_time': state['acquisition_timestamp'] = True
    if change == 'nonfinite_time': scene['acquisition_timestamp'] = float('nan')
    if change == 'boolean_step': step = True
    with pytest.raises(ValueError):
        SimSnapshotTransaction.capture(state, scene, physics_step=step, source_instance_id='run')
