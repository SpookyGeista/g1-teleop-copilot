"""Connect the guarded original controller to the isolated, complete PhysX scene.

This adapter does not start the stock XR entry or its warehouse environment.
It shares the existing input/session/guard pipeline and applies receipts via
the pinned original-controller shim and its owned simulator receiver.
"""
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import numpy as np

from .original_sim_snapshot import OriginalSimSnapshotAdapter
from .original_xr_sender import OriginalSimReceiptSink, guarded_original_controller_class


ROBOT_ASSET_SHA256 = 'd23b9bbc8b59cf20fa1983f55874f855a76e038b9e22eae39542e62be5b4ba3d'


class OriginalSimRuntime:
    def __init__(self, session, bench, xr_repository, *, codec):
        self.bench = bench
        def current_scene(_scene, context):
            # Receiver-owned read of the actually spawned table, ground and
            # cylinder. Its private source generation is unchanged while paused.
            geometry = bench.scene_snapshot(context.acquisition_timestamp)
            geometry.update(asdict(context))
            return geometry
        self.snapshots = OriginalSimSnapshotAdapter({'robot': bench.robot},
            read_generation=lambda: bench.physics_step_index, clock_id=bench.clock_id,
            source_instance_id=bench.source_instance_id, calibration_id=bench.calibration_id,
            robot_asset_sha256=ROBOT_ASSET_SHA256, collision_scene_reader=current_scene,
            simulation_mode=True)
        self.sink = OriginalSimReceiptSink(session, self.snapshots, simulation_mode=True, codec=codec)
        controller_type = guarded_original_controller_class(
            Path(xr_repository)/'teleop/robot_control/robot_arm.py', self.sink)
        # Hash verification/AST compilation can be slow during Isaac startup.
        # Acquire the constructor's actual state only AFTER that preparation.
        self.snapshots.capture()
        self.controller = controller_type(simulation_mode=True)
        self.manifest = dict(controller_type.original_manifest,
            runtime_scope='original_controller_shim_in_isolated_complete_PhysX_scene',
            stock_XR_application_started=False, robot_asset_sha256=ROBOT_ASSET_SHA256)
        self.physics_interlocks = []

    @staticmethod
    def _sample(snapshot):
        state = snapshot.robot_state
        base_inverse = np.linalg.inv(state['base_in_world'])
        tcp = np.eye(4); tcp[0,3] = .05
        poses = {side: base_inverse @ np.asarray(
            state['observed_body_world_poses'][side+'_wrist_yaw_link']) @ tcp
            for side in ('left','right')}
        return None, state, snapshot.scene, poses

    def capture_command_sample(self):
        return self._sample(self.snapshots.capture())

    def read_command_sample(self):
        return self._sample(self.snapshots.read_same())

    def apply(self, receipt, *, read_inputs, recheck_target=None):
        self.physics_interlocks = []
        if receipt is None:
            self.sink.idle_hold()
            return dict(applied=False,reason='no_consumed_receipt',command_format='unitree_float32',
                DDS_transport_started=False,hardware_authorized=False)
        # Frozen-fixture mode has no asynchronous target service. The complete
        # actual scene is still independently checked by the sidecar and guard.
        def fixture_target():
            return dict(accepted=True,reason='fixed_fixture_independently_checked_scene',
                motion_authorized=False,deadline_renewed=False)
        return self.controller.publish_receipt(receipt,read_inputs=read_inputs,
            recheck_target=recheck_target or fixture_target)

    def before_physics_step(self):
        self.sink.expire_before_physics_step()
        self.physics_interlocks.append(dict(deepcopy(self.sink.last_physics_interlock),
            physics_generation=self.bench.physics_step_index,source_instance_id=self.bench.source_instance_id))

    def close(self):
        self.controller.close()
