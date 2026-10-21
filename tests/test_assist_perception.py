"""Behavioral checks for sensor-only panel perception; no Isaac imports."""
import json
import unittest
import numpy as np

from teleop_assist.perception import CabinetPerception, PerceptionConfig, backproject


K = {'fx':300., 'fy':300., 'cx':160., 'cy':120.}
T = np.eye(4)


def frame(parts=((100, 85, 135, 125, .75), (183, 80, 215, 120, .76)), brightness=150):
    rgb = np.full((240, 320, 3), brightness, np.uint8)
    depth = np.full((240, 320), .80, np.float32)
    for i, (x0, y0, x1, y1, z) in enumerate(parts):
        depth[y0:y1, x0:x1] = z
        rgb[y0:y1, x0:x1] = np.clip(np.array([40, 80, 130]) + i*20, 0, 255)
    return rgb, depth


class PerceptionTests(unittest.TestCase):
    def test_backprojection_metric_depth_and_rotated_camera(self):
        points = backproject([[160, 120], [190, 90]], [1., 2.], K)
        np.testing.assert_allclose(points, [[0, 0, 1], [.2, -.2, 2]])
        rgb, depth = frame(parts=((140, 100, 181, 141, .75),))
        transform = np.eye(4)
        transform[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        transform[:3, 3] = [.06, 0, 1.28]
        output = CabinetPerception().process(rgb, depth, K, transform, 1, .1)
        self.assertEqual(len(output['targets']), 1)
        target = output['targets'][0]
        np.testing.assert_allclose(target['center_camera'], [0, 0, .75], atol=1e-6)
        np.testing.assert_allclose(target['center_world'], [.81, 0, 1.28], atol=1e-6)
        np.testing.assert_allclose(target['normal_world'], [-1, 0, 0], atol=1e-6)
        self.assertEqual(target['bbox_px'], [140, 100, 181, 141])
        self.assertEqual(target['box_camera']['quaternion_xyzw'], [0, 0, 0, 1])
        self.assertGreater(target['box_camera']['size'][0], .09)
        json.dumps(output, allow_nan=False)

    def test_blank_panel_and_color_only_shapes_create_no_targets(self):
        rgb, depth = frame(parts=())
        rgb[60:110, 110:160] = [0, 255, 0]
        result = CabinetPerception().process(rgb, depth, K, T, 1, .1)
        self.assertEqual(result['targets'], [])
        self.assertEqual(result['diagnostics']['reason'], 'no_separated_raised_components')
        self.assertFalse(result['diagnostics']['stale'])

    def test_invalid_depth_does_not_create_or_retain_targets(self):
        detector = CabinetPerception()
        rgb, depth = frame()
        first = detector.process(rgb, depth, K, T, 1, .1)
        self.assertEqual(len(first['targets']), 2)
        bad = np.full_like(depth, np.nan)
        bad[:60] = 0
        bad[60:120] = np.inf
        lost = detector.process(rgb, bad, K, T, 2, .2)
        self.assertEqual(lost['targets'], [])
        self.assertTrue(lost['diagnostics']['stale'])
        self.assertEqual(len(lost['diagnostics']['stale_track_ids']), 2)
        recovered = detector.process(rgb, depth, K, T, 3, .3)
        self.assertEqual([t['id'] for t in first['targets']], [t['id'] for t in recovered['targets']])

    def test_ids_survive_camera_motion_and_lighting_change(self):
        detector = CabinetPerception()
        rgb, depth = frame()
        first = detector.process(rgb, depth, K, T, 1, .1)
        # Move camera 5mm right: surface shifts two pixels left in image and the
        # independently provided camera pose moves right. Part positions agree.
        shifted = ((98, 85, 133, 125, .75), (181, 80, 213, 120, .76))
        rgb2, depth2 = frame(shifted, brightness=40)
        transform = T.copy()
        transform[0, 3] = .005
        second = detector.process(rgb2, depth2, K, transform, 2, .2)
        self.assertEqual(len(second['targets']), 2)
        by_x = lambda targets: sorted(targets, key=lambda t:t['center_world'][0])
        a, b = by_x(first['targets']), by_x(second['targets'])
        self.assertEqual([t['id'] for t in a], [t['id'] for t in b])
        for x, y in zip(a, b):
            np.testing.assert_allclose(x['center_world'], y['center_world'], atol=.001)

    def test_missing_objects_and_reacquisition_expire_without_ghosts(self):
        detector = CabinetPerception()
        rgb, depth = frame()
        first = detector.process(rgb, depth, K, T, 1, .1)
        old_ids = {t['id'] for t in first['targets']}
        blank_rgb, blank_depth = frame(parts=())
        absent = detector.process(blank_rgb, blank_depth, K, T, 2, .2)
        self.assertEqual(absent['targets'], [])
        self.assertEqual(set(absent['diagnostics']['stale_track_ids']), old_ids)
        late = detector.process(rgb, depth, K, T, 3, 1.1)
        self.assertFalse(old_ids.intersection(t['id'] for t in late['targets']))
        self.assertEqual(set(late['diagnostics']['expired_track_ids']), old_ids)

    def test_repeated_or_reversed_frames_are_stale(self):
        detector = CabinetPerception()
        rgb, depth = frame()
        detector.process(rgb, depth, K, T, 1, .1)
        for number, timestamp in [(1, .2), (2, .1), (0, .05)]:
            output = detector.process(rgb, depth, K, T, number, timestamp)
            self.assertEqual(output['targets'], [])
            self.assertTrue(output['diagnostics']['stale'])
        next_frame = detector.process(rgb, depth, K, T, 2, .2)
        self.assertEqual(len(next_frame['targets']), 2)

    def test_merged_blob_cannot_steal_either_previous_identity(self):
        detector = CabinetPerception()
        parts = ((120, 100, 139, 130, .75), (145, 100, 164, 130, .75))
        rgb, depth = frame(parts)
        initial = detector.process(rgb, depth, K, T, 1, .1)
        self.assertEqual(len(initial['targets']), 2)
        merged_rgb, merged_depth = frame(((120, 100, 164, 130, .75),))
        merged = detector.process(merged_rgb, merged_depth, K, T, 2, .2)
        self.assertEqual(merged['targets'], [])
        self.assertEqual(len(merged['diagnostics']['ambiguous_components']), 1)
        separate = detector.process(rgb, depth, K, T, 3, .3)
        self.assertEqual([t['id'] for t in initial['targets']], [t['id'] for t in separate['targets']])

    def test_holes_are_not_filled_with_invented_depth(self):
        rgb, depth = frame(parts=((120, 100, 170, 145, .75),))
        depth[113:130, 135:150] = np.nan
        result = CabinetPerception().process(rgb, depth, K, T, 1, .1)
        self.assertEqual(len(result['targets']), 1)
        target = result['targets'][0]
        self.assertLess(target['pixel_area'], 50*45)
        u, v = target['surface_pixel_px']
        self.assertTrue(np.isfinite(depth[v, u]))
        np.testing.assert_allclose(target['center_camera'], backproject([[u, v]], [depth[v, u]], K)[0])
        self.assertAlmostEqual(target['center_camera'][2], .75, places=6)
        json.dumps(result, allow_nan=False)

    def test_noisy_tilted_panel_is_estimated_from_measurements(self):
        # Camera ray intersection with a tilted plane: z=(d)/(1-a*x/z).
        y, x = np.indices((240, 320))
        slope = .18
        rays_x = (x-K['cx'])/K['fx']
        depth = (.8/(1-slope*rays_x)).astype(np.float32)
        depth[95:140, 140:185] = (.75/(1-slope*rays_x[95:140, 140:185]))
        rng = np.random.default_rng(11)
        depth += rng.normal(0, .0004, depth.shape).astype(np.float32)
        rgb = np.full((240, 320, 3), 90, np.uint8)
        output = CabinetPerception().process(rgb, depth, K, T, 1, .1)
        self.assertEqual(len(output['targets']), 1)
        expected = np.array([slope, 0, -1.])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(output['targets'][0]['normal_world'], expected, atol=.02)
        self.assertLess(output['diagnostics']['plane_residual_m'], .001)

    def test_near_occluder_is_not_clipped_into_fake_panel_parts(self):
        rgb, depth = frame(parts=((90, 100, 130, 145, .75), (180, 100, 220, 145, .75)))
        # A nearby arm fully hides one part. Reject that foreground surface,
        # rather than clipping the arm into a false permitted-depth fragment.
        depth[100:220, 180:260] = .52
        rgb[100:220, 180:260] = 240
        result = CabinetPerception().process(rgb, depth, K, T, 1, .1)
        self.assertEqual(len(result['targets']), 1)
        self.assertLess(result['targets'][0]['bbox_px'][2], 150)
        self.assertGreater(result['diagnostics']['rejected_components'].get('foreground_outside_panel_part_depth', 0), 0)

    def test_depth_edge_separates_partial_occluder_without_inventing_hand_target(self):
        rgb, depth = frame(parts=((90, 100, 130, 145, .75), (180, 100, 220, 145, .75)))
        depth[125:220, 195:260] = .52
        rgb[125:220, 195:260] = 240
        result = CabinetPerception().process(rgb, depth, K, T, 1, .1)
        self.assertEqual(len(result['targets']), 2)
        self.assertTrue(all(abs(t['center_camera'][2]-.75)<1e-6 for t in result['targets']))
        for t in result['targets']:
            u, v = t['surface_pixel_px']
            self.assertFalse(195<=u<260 and 125<=v<220)
            self.assertTrue(np.isfinite(depth[v,u]))
        self.assertGreater(result['diagnostics']['depth_discontinuity_pixels_removed'],0)
        self.assertGreater(result['diagnostics']['rejected_components'].get('foreground_outside_panel_part_depth',0),0)

    def test_malformed_inputs_fail_closed(self):
        rgb, depth = frame()
        detector = CabinetPerception()
        for bad_k in [{**K, 'fx':0}, {**K, 'cy':float('nan')}]:
            with self.assertRaises(ValueError):
                detector.process(rgb, depth, bad_k, T, 1, .1)
        bad_transform = T.copy(); bad_transform[0, 0] = 2
        with self.assertRaises(ValueError):
            detector.process(rgb, depth, K, bad_transform, 1, .1)
        with self.assertRaises(ValueError):
            detector.process(rgb, depth[:10], K, T, 1, .1)
        with self.assertRaises(ValueError):
            detector.process(rgb.astype(float), depth, K, T, 1, .1)
        with self.assertRaises(ValueError):
            detector.process(rgb, depth, K, T, True, .1)
        with self.assertRaises(ValueError):
            backproject([[0, 0]], [float('nan')], K)


if __name__ == '__main__':
    unittest.main()
