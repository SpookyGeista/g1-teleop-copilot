from pathlib import Path
import pytest
from teleop_assist.retained_reference import SOURCE, retained_zero_reference, zero_reference_root_height
from teleop_assist.isaac_backend import RETAINED, HAND_URDF


def test_local_retained_reference_has_exact_named_body_joints():
    if not (RETAINED/SOURCE).exists():
        pytest.skip('Retained local teleop archive is not installed')
    reference = retained_zero_reference(RETAINED, HAND_URDF)
    assert len(reference['joint_positions']) == 29
    assert len(reference['finger_positions']) == 14
    assert all(value == 0 for value in reference['joint_positions'].values())
    assert reference['source_line'] > 0
    assert len(reference['source_sha256']) == 64


def test_unknown_upstream_reference_does_not_silently_fall_back(tmp_path):
    source = tmp_path/SOURCE; source.parent.mkdir(parents=True)
    source.write_text('def calibrate_now(self):\n    self._override_robot_q = np.ones(29)\n')
    with pytest.raises(ValueError, match='no longer'):
        retained_zero_reference(tmp_path, tmp_path/'not_read.urdf')


def test_zero_pose_fits_raised_plate_without_changing_joint_reference():
    if not HAND_URDF.exists():
        pytest.skip('Retained G1 URDF is not installed')
    floor = zero_reference_root_height(HAND_URDF, 0.)
    plate = zero_reference_root_height(HAND_URDF, .06)
    assert plate-floor == pytest.approx(.06)
    assert .85 < plate < .86  # Old .80m mount put the zero-pose heels in the plate.
