"""Independent interaction and fail-closed contract checks for assist core."""
import copy
import unittest
import numpy as np

from teleop_assist.core import AssistConfig, AssistSession, GazeCalibration, Selection


class Clock:
    def __init__(self): self.value = 1.
    def __call__(self): return self.value
    def advance(self, dt): self.value += dt


def target(name, box, point=(.55, -.2, 1.1)):
    return {'id':name, 'bbox_px':list(box), 'center_world':list(point),
            'normal_world':[-1., 0., 0.], 'confidence':.9, 'stale':False,
            'label':'raised_component', 'frame_id':1, 'timestamp':1.}


def targets():
    return [target('left', [10, 40, 30, 60]), target('center', [40, 40, 60, 60]),
            target('right', [70, 40, 90, 60]), target('up', [40, 10, 60, 30]),
            target('down', [40, 70, 60, 90])]


def packet(seq=1, **fields):
    data = {'source':'test_operator', 'session_id':'test', 'seq':seq,
            'right':{'valid':True, 'position':[0., 0., 0.], 'quaternion_xyzw':[0., 0., 0., 1.]},
            'axes':[0., 0.], 'trigger':0., 'squeeze':0., 'clutch':True,
            'primary':False, 'secondary':False}
    data.update(fields)
    return data


class SelectionTests(unittest.TestCase):
    def test_manual_direction_latches_against_eye_proposals(self):
        s = Selection(); s.update(targets()); s.propose('center')
        self.assertTrue(s.navigate('right'))
        self.assertEqual(s.snapshot(), {'mode':'manual', 'id':'right'})
        s.propose('left'); self.assertEqual(s.id, 'right')
        s.confirm(); self.assertFalse(s.navigate('left')); self.assertEqual(s.id, 'right')
        s.release(); s.propose('left'); self.assertEqual(s.id, 'left')

    def test_direction_uses_image_axes_and_requires_visible_target(self):
        for direction in ('up', 'down', 'left', 'right'):
            s = Selection(); s.update(targets()); s.propose('center')
            self.assertTrue(s.navigate(direction)); self.assertEqual(s.id, direction)
        with self.assertRaises(ValueError): s.select('missing')
        with self.assertRaises(ValueError): s.navigate('forward')

    def test_target_loss_never_automatically_reconfirms(self):
        s = Selection(); s.update(targets()); s.propose('center'); s.confirm()
        s.update([]); self.assertEqual(s.snapshot(), {'mode':'lost', 'id':'center'})
        s.propose('left'); self.assertEqual(s.id, 'center')
        s.update(targets()); self.assertEqual(s.mode, 'lost')
        with self.assertRaises(ValueError): s.confirm()
        s.select('center'); s.confirm(); self.assertEqual(s.mode, 'confirmed')


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock(); self.s = AssistSession(clock=self.clock)
        self.robot = {'valid':True, 'ee_position':[.47, -.25, 1.08]}
        self.s.update_frame(targets(), 1, self.clock(), {'width':100, 'height':100})

    def arm(self):
        self.s.step(self.robot, .02)
        self.s.receive_input(packet())
        self.s.calibrate_hand()
        self.s.command({'type':'assist_enable','enabled':True})
        self.s.command({'type':'select', 'id':'center'})
        self.s.command({'type':'confirm'})

    def test_eye_off_and_device_missing_use_pointer(self):
        self.s.command({'type':'pointer', 'uv':[.2, .5], 'frame_id':1})
        self.s.set_gaze({'seq':1, 'source':'external', 'uv':[.8, .5],
                         'valid':True, 'calibrated':True, 'confidence':.9, 'frame_id':1})
        self.assertEqual(self.s.selection.id, 'left')
        self.assertEqual(self.s.eye_status(self.clock()), 'disabled')
        self.s.command({'type':'eye_enable', 'enabled':True})
        self.assertEqual(self.s.selection.id, 'right')
        self.assertEqual(self.s.eye_status(self.clock()), 'active')
        self.clock.advance(.3)
        self.s.snapshot()
        self.assertEqual(self.s.selection.id, 'left')
        self.assertEqual(self.s.eye_status(self.clock()), 'stale')

    def test_manual_gaze_latch_is_preserved(self):
        self.s.command({'type':'eye_enable', 'enabled':True})
        self.s.command({'type':'direction', 'direction':'right'})
        self.s.set_gaze({'seq':1, 'uv':[.2,.5], 'valid':True, 'calibrated':True,
                         'confidence':1., 'frame_id':1})
        self.assertEqual(self.s.selection.id, 'right')
        self.assertEqual(self.s.selection.mode, 'manual')

    def test_uncalibrated_lowconfidence_and_wrongframe_gaze_fall_back(self):
        self.s.command({'type':'eye_enable', 'enabled':True})
        self.s.command({'type':'pointer', 'uv':[.2,.5], 'frame_id':1})
        for seq, extra, status in [(1, {'raw':[.8,.5]}, 'calibration_required'),
                                   (2, {'confidence':.1}, 'low_confidence'),
                                   (3, {'frame_id':999}, 'frame_unavailable')]:
            gaze={'seq':seq, 'uv':[.8,.5], 'valid':True, 'calibrated':True,
                  'confidence':1., 'frame_id':1, **extra}
            self.s.set_gaze(gaze)
            self.assertEqual(self.s.eye_status(self.clock()), status)
            self.assertEqual(self.s.selection.id, 'left')

    def test_calibrated_gaze_affine_fit_and_degenerate_rejection(self):
        samples=[{'raw':[2*x-.4, 1.5*y+.1], 'uv':[x,y]}
                 for x in (.1,.5,.9) for y in (.1,.5,.9)]
        fit=self.s.gaze_calibration.fit(samples)
        self.assertLess(fit['fit_rms_uv'], 1e-10)
        np.testing.assert_allclose(self.s.gaze_calibration.transform([1.2,.85]), [.8,.5])
        with self.assertRaises(ValueError): GazeCalibration().fit(samples[:4])
        bad=[{'raw':[0,0], 'uv':s['uv']} for s in samples]
        with self.assertRaises(ValueError): GazeCalibration().fit(bad)

    def test_deadman_and_stale_input_hold_measured_pose(self):
        self.arm()
        moving=self.s.step(self.robot,.02); self.assertFalse(moving['hold'])
        self.s.receive_input(packet(2, clutch=False))
        held=self.s.step(self.robot,.02)
        self.assertTrue(held['hold']); self.assertEqual(self.s.control['reason'], 'deadman_released')
        np.testing.assert_allclose(held['ee_target_world'], self.robot['ee_position'])
        self.s.receive_input(packet(3)); self.clock.advance(.31)
        self.assertTrue(self.s.step(self.robot,.02)['hold'])
        self.assertEqual(self.s.control['reason'], 'input_stale')

    def test_old_sequence_cannot_refresh_live_control(self):
        self.arm(); before=self.s.input_time
        self.clock.advance(.2)
        self.assertFalse(self.s.receive_input(packet(1)))
        self.assertEqual(self.s.input_time,before)
        self.clock.advance(.11)
        self.assertTrue(self.s.step(self.robot,.02)['hold'])

    def test_joystick_edges_do_not_repeat_or_drive_arm(self):
        self.s.receive_input(packet(1, clutch=False))
        self.s.receive_input(packet(2, clutch=False, axes=[1,0]))
        self.assertEqual(self.s.selection.id,'right')
        self.s.receive_input(packet(3, clutch=False, axes=[1,0]))
        self.assertEqual(self.s.selection.id,'right')
        self.s.receive_input(packet(4, clutch=False, axes=[0,0]))
        self.s.receive_input(packet(5, clutch=False, axes=[-1,0]))
        self.assertEqual(self.s.selection.id,'center')
        self.assertTrue(self.s.step(self.robot,.02)['hold'])

    def test_button_confirmation_and_release_edges(self):
        self.s.receive_input(packet(1, clutch=False))
        self.s.receive_input(packet(2, clutch=False, primary=True))
        self.assertEqual(self.s.selection.mode,'confirmed')
        self.s.receive_input(packet(3, clutch=False, primary=True, axes=[1,0]))
        self.assertEqual(self.s.selection.id,'center')
        self.s.receive_input(packet(4, clutch=False, secondary=True))
        self.assertEqual(self.s.selection.mode,'preview')

    def test_correction_lateral_bounded_and_rate_limited(self):
        self.arm(); reference=self.s.step(self.robot,.02)
        correction=np.array(reference['correction'])
        self.assertGreater(np.linalg.norm(correction), .01)
        self.assertTrue(self.s.control['active'])
        self.assertLessEqual(np.linalg.norm(correction),self.s.config.max_correction+1e-12)
        self.assertAlmostEqual(float(correction@np.array([-1,0,0])),0.,places=12)
        self.assertLessEqual(np.linalg.norm(np.array(reference['ee_target_world'])-self.robot['ee_position']),
                             self.s.config.max_speed*.02+1e-12)
        self.s.command({'type':'assist_enable','enabled':False})
        manual=self.s.step(self.robot,.02)
        np.testing.assert_allclose(manual['correction'],[0,0,0])

    def test_target_loss_and_perception_timeout_hold(self):
        self.arm(); self.s.update_frame([],2,self.clock(),{'width':100,'height':100})
        self.assertTrue(self.s.step(self.robot,.02)['hold'])
        self.assertEqual(self.s.control['reason'],'target_lost')
        self.s.update_frame(targets(),3,self.clock(),{'width':100,'height':100})
        self.s.command({'type':'select','id':'center'});self.s.command({'type':'confirm'})
        self.clock.advance(.8);self.s.receive_input(packet(2))
        self.assertTrue(self.s.step(self.robot,.02)['hold'])
        self.assertEqual(self.s.control['reason'],'perception_stale')

    def test_stop_collision_and_invalid_tracking_hold(self):
        self.arm(); self.s.command({'type':'stop'})
        self.assertTrue(self.s.step(self.robot,.02)['hold'])
        with self.assertRaises(ValueError):self.s.command({'type':'resume'})
        self.s.receive_input(packet(2,clutch=False));self.s.command({'type':'resume'})
        self.assertIsNone(self.s.hand_origin)
        self.s.calibrate_hand();self.s.receive_input(packet(3))
        self.assertTrue(self.s.step({**self.robot,'collision_stop':True},.02)['hold'])
        self.s.receive_input(packet(4,right={'valid':False}))
        self.assertTrue(self.s.step(self.robot,.02)['hold'])

    def test_changed_coordinate_basis_requires_fresh_calibration(self):
        self.arm()
        changed=packet(2,xr_to_world_rotation=np.eye(3).tolist())
        self.s.receive_input(changed)
        self.assertTrue(self.s.step(self.robot,.02)['hold'])
        self.assertEqual(self.s.control['reason'],'calibration_required')
        self.assertEqual(self.s.intent_ranked,[])
        self.s.calibrate_hand()
        self.assertFalse(self.s.step(self.robot,.02)['hold'])

    def test_confident_intent_cannot_override_manual_or_confirmed_selection(self):
        self.s.intent_ranked=[{'id':'right','suggestion_eligible':True,'hand_weight':.9}]
        self.s.intent_time=self.clock()
        self.s.snapshot()
        self.assertEqual(self.s.selection.id,'right')
        self.s.command({'type':'select','id':'left'})
        self.s.snapshot();self.assertEqual(self.s.selection.id,'left')
        self.s.command({'type':'confirm'})
        self.s.snapshot();self.assertEqual(self.s.selection.id,'left')
        self.assertEqual(self.s.selection.mode,'confirmed')

    def test_old_frame_pointer_not_applied_to_new_boxes_in_intent(self):
        self.arm()
        self.s.command({'type':'release'})
        self.s.command({'type':'pointer','uv':[.2,.5],'frame_id':1})
        self.clock.advance(.05)
        self.s.update_frame(targets(),2,self.clock(),{'width':100,'height':100})
        self.s.receive_input(packet(2))
        self.assertTrue(all(row['pointer_score']==0 for row in self.s.intent_ranked))

    def test_malformed_pointer_packet_is_atomic(self):
        self.arm(); before=copy.deepcopy(self.s.input); before_seq=dict(self.s.input_sequences)
        self.clock.advance(.2)
        with self.assertRaises(ValueError):
            self.s.receive_input(packet(2,pointer_uv=[float('nan'),0]))
        self.assertEqual(self.s.input,before)
        self.assertEqual(self.s.input_sequences,before_seq)
        self.assertEqual(self.s.input_time,1.)

    def test_malformed_boolean_cannot_engage_deadman(self):
        with self.assertRaises(ValueError):self.s.receive_input(packet(clutch='false'))
        self.assertIsNone(self.s.input)

    def test_invalid_gaze_does_not_consume_sequence(self):
        sample={'seq':1,'uv':[.5,.5],'valid':True,'calibrated':True,'confidence':float('nan')}
        with self.assertRaises(ValueError):self.s.set_gaze(sample)
        self.assertTrue(self.s.set_gaze({**sample,'confidence':1.}))

    def test_invalid_target_normal_and_timestamp_rejected(self):
        for timestamp in (float('nan'),float('inf'),self.clock()+60):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ValueError):
                    self.s.update_frame(targets(),2,timestamp,{'width':100,'height':100})
        invalid=target('bad',[10,10,20,20]);invalid['normal_world']=[0,0,0]
        with self.assertRaises(ValueError):self.s.update_frame([invalid],3,self.clock(),{'width':100,'height':100})

    def test_nan_time_step_cannot_produce_nan_command(self):
        self.arm()
        with self.assertRaises(ValueError):self.s.step(self.robot,float('nan'))
        with self.assertRaises(ValueError):self.s.step(self.robot,.02,now=float('inf'))
        with self.assertRaises(ValueError):self.s.step(self.robot,.02,now=-1.)

    def test_valid_gaze_requires_an_actual_measured_coordinate(self):
        self.s.command({'type':'eye_enable','enabled':True})
        with self.assertRaises(ValueError):
            self.s.set_gaze({'seq':1,'valid':True,'calibrated':True,'confidence':1.})
        self.assertIsNone(self.s.gaze)


if __name__=='__main__':unittest.main()
