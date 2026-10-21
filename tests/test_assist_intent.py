"""Geometric intent suggestions must remain bounded and explicitly unconfirmed."""
import json
import unittest
import numpy as np

from teleop_assist.intent import IntentEstimator, IntentConfig

CAMERA={'width':100,'height':100}


def candidates(stamp=1.):
    return [{'id':'left','center_world':[.55,-.30,1.1],'normal_world':[-1,0,0],
             'bbox_px':[10,40,30,60],'confidence':.95,'timestamp':stamp},
            {'id':'right','center_world':[.55,-.15,1.1],'normal_world':[-1,0,0],
             'bbox_px':[70,40,90,60],'confidence':.95,'timestamp':stamp}]


class IntentTests(unittest.TestCase):
    def test_stationary_pointing_dominates_and_never_confirms(self):
        estimator=IntentEstimator()
        ranked=estimator.update(candidates(),[.35,-.25,1.1],1.,[.2,.5],CAMERA)
        self.assertEqual(ranked[0]['id'],'left')
        self.assertTrue(ranked[0]['suggestion_eligible'])
        self.assertEqual(ranked[0]['hand_weight'],0)
        self.assertTrue(all(row['automatic_confirmation'] is False for row in ranked))
        self.assertIn('not_calibrated',ranked[0]['probability_kind'])

    def test_deliberate_approach_can_correct_wrong_coarse_gaze(self):
        estimator=IntentEstimator()
        estimator.update(candidates(),[.47,-.28,1.1],1.,[.2,.5],CAMERA)
        ranked=estimator.update(candidates(1.1),[.47,-.255,1.1],1.1,[.2,.5],CAMERA)
        self.assertEqual(ranked[0]['id'],'right')
        self.assertGreater(ranked[0]['hand_approach_score'],ranked[1]['hand_approach_score'])
        self.assertTrue(ranked[0]['suggestion_eligible'])
        self.assertGreater(estimator.diagnostics['speed_m_s'],.12)

    def test_direction_distinguishes_adjacent_targets(self):
        estimator=IntentEstimator()
        nearby=candidates()
        nearby[0]['center_world']=[.55,-.25,1.1]
        nearby[1]['center_world']=[.55,-.15,1.1]
        estimator.update(nearby,[.27,-.15,1.1],1.,None,CAMERA)
        for row in nearby:row['timestamp']=1.1
        ranked=estimator.update(nearby,[.30,-.15,1.1],1.1,None,CAMERA)
        self.assertEqual(ranked[0]['id'],'right')
        self.assertTrue(ranked[0]['suggestion_eligible'])
        self.assertLess(ranked[0]['approach_lateral_error_m'],.001)
        self.assertGreater(ranked[1]['approach_lateral_error_m'],.09)

    def test_ambiguous_or_weak_evidence_never_forces_selection(self):
        estimator=IntentEstimator()
        ranked=estimator.update(candidates(),[.35,-.225,1.1],1.,None,CAMERA)
        self.assertTrue(all(not r['suggestion_eligible'] for r in ranked))
        self.assertAlmostEqual(ranked[0]['probability'],.5)
        estimator.reset()
        ranked=estimator.update(candidates(),[.35,-.225,1.1],1.,{'uv':[.2,.5],'confidence':.05},CAMERA)
        self.assertFalse(ranked[0]['suggestion_eligible'])

    def test_stale_missing_invalid_targets_never_hallucinate_ids(self):
        estimator=IntentEstimator()
        stale=candidates(0.)
        self.assertEqual(estimator.update(stale,[.3,-.2,1.1],1.,[.2,.5],CAMERA),[])
        valid=candidates(1.1);valid[0]['stale']=True
        result=estimator.update(valid,[.3,-.2,1.1],1.1,[.8,.5],CAMERA)
        self.assertEqual({r['id'] for r in result},{'right'})
        self.assertEqual(estimator.update([], [.3,-.2,1.1],1.2,None,CAMERA),[])
        self.assertEqual(estimator.update(None,[.3,-.2,1.1],1.3,None,CAMERA),[])

    def test_sensor_clock_jump_and_operator_jump_do_not_generate_motion_intent(self):
        estimator=IntentEstimator()
        estimator.update(candidates(),[.3,-.2,1.1],1.,None,CAMERA)
        self.assertEqual(estimator.update(candidates(),[.4,-.2,1.1],1.,None,CAMERA),[])
        jumped=estimator.update(candidates(1.01),[.8,-.2,1.1],1.01,None,CAMERA)
        self.assertEqual(jumped,[]);self.assertEqual(estimator.diagnostics['reason'],'operator_jump_rejected')
        ranked=estimator.update(candidates(2.),[.4,-.2,1.1],2.,None,CAMERA)
        self.assertFalse(estimator.diagnostics['motion_ready'])
        self.assertTrue(all(not r['suggestion_eligible'] for r in ranked))

    def test_malformed_hand_and_pointer_are_bounded(self):
        estimator=IntentEstimator()
        self.assertEqual(estimator.update(candidates(),[float('nan'),0,0],1.,[.2,.5],CAMERA),[])
        self.assertEqual(estimator.update(candidates(),[0,0,0],float('inf'),None,CAMERA),[])
        self.assertEqual(estimator.update(candidates(),[0,0,0],True,None,CAMERA),[])
        result=estimator.update(candidates(),[.3,-.2,1.1],1.,[float('nan'),.5],CAMERA)
        self.assertFalse(estimator.diagnostics['pointer_available'])
        self.assertTrue(all(not r['suggestion_eligible'] for r in result))
        json.dumps(result,allow_nan=False)

    def test_probability_bounds_and_input_not_modified(self):
        estimator=IntentEstimator();parts=candidates()
        parts[0]['normal_world']=np.array([-2.,0,0])
        original=parts[0]['normal_world'].copy()
        ranked=estimator.update(parts,[.3,-.2,1.1],1.,[.2,.5],CAMERA)
        np.testing.assert_array_equal(original,parts[0]['normal_world'])
        self.assertAlmostEqual(sum(r['probability'] for r in ranked),1.)
        self.assertTrue(all(0<=r['score']<=1 and 0<=r['probability']<=1 for r in ranked))
        with self.assertRaises(ValueError):IntentEstimator(IntentConfig(max_motion_speed_m_s=.001))


if __name__=='__main__':unittest.main()
