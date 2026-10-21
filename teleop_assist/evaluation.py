"""Automated acceptance input through the same operator/session command API.

This is a synthetic operator, not a live headset test. Targets originate only
from the current RGB-D detector. The optional ``filter_targets`` hook injects
an explicitly recorded detector dropout; it never supplies hidden geometry.
"""
from __future__ import annotations

import math
import numpy as np

from .core import Selection


class AcceptanceDriver:
    """Call update before session.step, observe after the actual backend step.

    Feed every detector result through filter_targets before update_frame.
    Simulator timestamps drive motion phases; the session's real clock drives
    transport expiry. No sleep or fabricated packet timestamp is needed.
    """
    TRIAL_DURATION = 10.0
    RETURN_DURATION = 9.0
    MAX_SIM_SECONDS = 170.0
    MAX_WALL_SECONDS = 240.0
    XR_TO_ROBOT = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])

    def __init__(self, session):
        self.session = session
        self.phase = 'await_perception'
        self.phase_start = None
        self.wall_start = session.clock()
        self.phase_wall = self.wall_start
        self.start_sim = None
        self.done = False
        self.seq = 0
        self.gaze_seq = 0
        self.start_pose = None
        self.goal = None
        self.target_id = None
        self.target_center = None
        self.target_normal = None
        self.trial = None
        self.samples = []
        self.results = {}
        self.events = []
        self.max_joint_violation = 0.
        self.invalid_states = 0
        self.contact_stops = 0
        self.fault_observed = False
        self.fault_reference_hold = True
        self.selection_checks = {}
        self._after_return = None
        self._fault_initial_sent = False
        self._last_reference = None

    def _event(self, name, sim_time, **fields):
        self.events.append({'event': name, 'sim_time': float(sim_time),
                            'wall_time': self.session.clock(), **fields})

    def _set_phase(self, phase, sim_time):
        self.phase = phase
        self.phase_start = float(sim_time)
        self.phase_wall = self.session.clock()
        self.samples = []
        self._fault_initial_sent = False
        self._event('acceptance_phase', sim_time, phase=phase)

    def _input(self, desired=None, clutch=True, axes=(0., 0.), primary=False, pointer=None):
        self.seq += 1
        if desired is None or self.session.robot_origin is None:
            raw_hand = np.zeros(3)
        else:
            raw_hand = self.XR_TO_ROBOT.T @ (np.asarray(desired)-np.asarray(self.session.robot_origin))
            raw_hand /= self.session.config.hand_scale
            raw_hand += np.asarray(self.session.hand_origin)
        sample = {'source': 'synthetic_acceptance_operator', 'session_id': 'isaac_acceptance_v1',
                  'seq': self.seq, 'right': {'valid': True, 'position': raw_hand.tolist(),
                                           'quaternion_xyzw': [0., 0., 0., 1.]},
                  'axes': list(axes), 'primary': bool(primary), 'secondary': False,
                  'trigger': 0., 'squeeze': 1. if clutch else 0., 'clutch': bool(clutch),
                  'orientation_mode': 'position', 'frame_id': self.session.frame_id}
        if pointer is not None:
            sample['pointer_uv'] = list(pointer)
        self.session.command({'type': 'input', 'sample': sample})

    def _calibrate(self, state):
        # Register the same current measured backend state that is passed to
        # session.step immediately afterward; no state is fabricated.
        self.session.robot = dict(state)
        self._input(clutch=False)
        self.session.command({'type': 'calibrate'})

    def _uv(self, target):
        box = target['bbox_px']
        return [(box[0]+box[2])*.5/self.session.camera['width'],
                (box[1]+box[3])*.5/self.session.camera['height']]

    def _gaze(self, target):
        self.gaze_seq += 1
        self.session.command({'type': 'gaze', 'sample': {
            'source': 'synthetic_acceptance_gaze', 'session_id': 'synthetic_gaze_v1',
            'seq': self.gaze_seq, 'uv': self._uv(target), 'valid': True,
            'calibrated': True, 'confidence': .99, 'frame_id': self.session.frame_id}})

    def _find_goal(self):
        targets = list(self.session.selection.targets.values())
        if not targets:
            return None
        # A requested lower-right work area, not a simulator object identity.
        # Selection is performed over measured visible component positions.
        expected = np.array([.55, -.26, .985])
        candidates = [t for t in targets if np.isfinite(t['center_world']).all()
                      and .3 < t['center_world'][0] < .75
                      and -.5 < t['center_world'][1] < .1
                      and .82 < t['center_world'][2] < 1.25]
        if not candidates:
            return None
        target = min(candidates, key=lambda t: np.linalg.norm(np.asarray(t['center_world'])-expected))
        return target if np.linalg.norm(np.asarray(target['center_world'])-expected) < .15 else None

    def _begin_trial(self, name, state, sim_time):
        target = self.session.selection.targets.get(self.target_id)
        if target is None:
            target = self._find_goal()
            if target is None:
                self.results[name] = {'passed': False, 'reason': 'perceived_target_unavailable'}
                self._finish(False, 'Target unavailable at scenario start')
                return
            self.target_id = target['id']
        self.session.command({'type': 'release'})
        self.session.command({'type': 'assist_enable', 'enabled': name != 'baseline_raw'})
        self.session.command({'type': 'eye_enable', 'enabled': name in ('eye_on_no_device', 'eye_joystick')})
        self._calibrate(state)
        self.trial = name
        self._set_phase(name, sim_time)
        if name == 'eye_joystick':
            neighbor = None
            for wrong in self.session.selection.targets.values():
                if wrong['id'] == self.target_id:
                    continue
                for direction in ('up', 'down', 'left', 'right'):
                    selection = Selection()
                    selection.update(list(self.session.selection.targets.values()))
                    selection.propose(wrong['id'])
                    if selection.navigate(direction) and selection.id == self.target_id:
                        neighbor = wrong, direction
                        break
                if neighbor:
                    break
            if neighbor is None:
                self.selection_checks['joystick_neighbor_found'] = False
                self.session.command({'type': 'select', 'id': self.target_id})
            else:
                wrong, direction = neighbor
                self.selection_checks['joystick_neighbor_found'] = True
                self._gaze(wrong)
                self.selection_checks['gaze_proposed_wrong'] = self.session.selection.id == wrong['id']
                self._input(self.start_pose, clutch=False, axes=(0., 0.))
                axes = {'right': (1., 0.), 'left': (-1., 0.), 'up': (0., -1.), 'down': (0., 1.)}[direction]
                self._input(self.start_pose, clutch=False, axes=axes)
                self.selection_checks['joystick_corrected_target'] = self.session.selection.id == self.target_id
                self.selection_checks['manual_latched'] = self.session.selection.mode == 'manual'
                self._gaze(wrong)
                self.selection_checks['gaze_did_not_override_manual'] = self.session.selection.id == self.target_id
                self._event('synthetic_eye_joystick_sequence', sim_time, wrong_id=wrong['id'],
                            intended_id=self.target_id, direction=direction, checks=dict(self.selection_checks))
            self._input(self.start_pose, clutch=False, axes=(0., 0.), primary=True)
            self.selection_checks['confirmed_by_controller_button'] = self.session.selection.mode == 'confirmed'
        else:
            self._input(self.start_pose, clutch=False, pointer=self._uv(target))
            if name != 'baseline_raw':
                self.session.command({'type': 'select', 'id': self.target_id})
                self.session.command({'type': 'confirm'})
        if name == 'eye_on_no_device':
            self.selection_checks['eye_on_no_device_fallback'] = self.session.eye_status(self.session.clock()) == 'no_device'

    def _begin_return(self, next_name, state, sim_time):
        self.session.command({'type': 'release'})
        self.session.command({'type': 'assist_enable', 'enabled': False})
        self._calibrate(state)
        self._after_return = next_name
        self._set_phase('return_start', sim_time)

    def _complete_trial(self, sim_time):
        samples = [x for x in self.samples if x['sim_time'] >= float(sim_time)-1.]
        if not samples:
            samples = self.samples
        errors = [x['lateral_error_m'] for x in samples]
        result = {'mean_lateral_error_m': float(np.mean(errors)) if errors else None,
                  'final_lateral_error_m': errors[-1] if errors else None,
                  'mean_distance_to_approach_m': float(np.mean([x['approach_error_m'] for x in samples])) if samples else None,
                  'measured_samples': len(self.samples),
                  'settled_samples': len(samples),
                  'hold_reasons': sorted(set(x['reason'] for x in self.samples if x['hold'])),
                  'valid_states': all(x['valid'] for x in self.samples),
                  'input': 'synthetic XR controller packets; shared production session API',
                  'intentional_raw_lateral_offset_m': .030,
                  'goal_from': 'visible measured RGB-D target and estimated surface normal'}
        if self.trial == 'baseline_raw':
            result['passed'] = bool(errors and .018 <= result['mean_lateral_error_m'] <= .047 and result['valid_states'])
        else:
            result['passed'] = bool(errors and result['mean_lateral_error_m'] < .012 and result['valid_states'])
        self.results[self.trial] = result

    def _finish(self, passed=None, error=None):
        self.done = True
        if error:
            self.results['driver'] = {'passed': False, 'reason': error}
        self._input(desired=self.session.robot.get('ee_position'), clutch=False)
        self.session.command({'type': 'assist_enable', 'enabled': False})

    def filter_targets(self, targets):
        if self.phase == 'target_lost':
            return [t for t in targets if t['id'] != self.target_id]
        return targets

    def update(self, robot_state, sim_time):
        if self.done:
            return
        now = self.session.clock()
        if self.start_sim is None:
            self.start_sim = float(sim_time)
            self.phase_start = float(sim_time)
        if float(sim_time)-self.start_sim > self.MAX_SIM_SECONDS or now-self.wall_start > self.MAX_WALL_SECONDS:
            self._finish(False, 'Bounded acceptance driver deadline exceeded')
            return
        elapsed = float(sim_time)-self.phase_start
        if self.phase == 'await_perception':
            target = self._find_goal()
            if target is None:
                if elapsed > 15.:
                    self._finish(False, 'No reachable RGB-D component detected')
                return
            self.start_pose = np.asarray(robot_state['ee_position'], float)
            self.target_id = target['id']
            self.target_center = np.asarray(target['center_world'], float)
            self.target_normal = np.asarray(target['normal_world'], float)
            self.target_normal /= np.linalg.norm(self.target_normal)
            self.goal = self.target_center+self.session.config.standoff*self.target_normal
            self._event('perceived_goal_selected', sim_time, target_id=self.target_id,
                        center=self.target_center.tolist(), approach=self.goal.tolist(), start=self.start_pose.tolist())
            self._begin_trial('baseline_raw', robot_state, sim_time)
            return
        if self.phase in ('baseline_raw', 'eye_off_assisted', 'eye_on_no_device', 'eye_joystick'):
            offset = np.array([0., .030, 0.])
            offset -= self.target_normal*float(offset@self.target_normal)
            self._input(self.goal+offset, clutch=True)
            if elapsed >= self.TRIAL_DURATION:
                name = self.phase
                self._complete_trial(sim_time)
                next_name = {'baseline_raw': 'eye_off_assisted', 'eye_off_assisted': 'eye_on_no_device',
                             'eye_on_no_device': 'eye_joystick', 'eye_joystick': 'input_stale'}[name]
                self._begin_return(next_name, robot_state, sim_time)
            return
        if self.phase == 'return_start':
            self._input(self.start_pose, clutch=True)
            if elapsed >= self.RETURN_DURATION:
                err = float(np.linalg.norm(np.asarray(robot_state['ee_position'])-self.start_pose))
                self.results['return_before_'+self._after_return] = {'passed': err < .015,
                                                                    'measured_start_error_m': err}
                if self._after_return == 'input_stale':
                    self._calibrate(robot_state)
                    self._set_phase('input_stale', sim_time)
                    self.fault_observed = False
                    self.fault_reference_hold = True
                else:
                    self._begin_trial(self._after_return, robot_state, sim_time)
            return
        if self.phase == 'input_stale':
            if not self._fault_initial_sent:
                self._input(robot_state['ee_position'], clutch=True)
                self._fault_initial_sent = True
                self.phase_wall = now
            if now-self.phase_wall >= self.session.config.input_timeout+.15:
                self.results['stale_input_hold'] = {'passed': self.fault_observed and self.fault_reference_hold,
                    'observed_reason': 'input_stale' if self.fault_observed else None,
                    'wall_s_without_packets': now-self.phase_wall}
                self._calibrate(robot_state)
                self._set_phase('deadman_released', sim_time)
                self.fault_observed = False
                self.fault_reference_hold = True
            return
        if self.phase == 'deadman_released':
            self._input(self.goal, clutch=False)
            if elapsed >= .5:
                self.results['deadman_release_hold'] = {'passed': self.fault_observed and self.fault_reference_hold}
                self._calibrate(robot_state)
                target = self.session.selection.targets.get(self.target_id)
                if target is None:
                    target = self._find_goal()
                    if target is not None:
                        self.target_id = target['id']
                if target is not None:
                    self.session.command({'type': 'select', 'id': self.target_id})
                    self.session.command({'type': 'confirm'})
                self.session.command({'type': 'assist_enable', 'enabled': True})
                self._set_phase('target_lost', sim_time)
                self._event('detector_dropout_injected', sim_time, withheld_target_id=self.target_id,
                            mechanism='filter current detector outputs; no synthetic geometry')
                self.fault_observed = False
                self.fault_reference_hold = True
            return
        if self.phase == 'target_lost':
            self._input(robot_state['ee_position'], clutch=True)
            if elapsed >= 1.5:
                self.results['target_lost_hold'] = {'passed': self.fault_observed and self.fault_reference_hold,
                    'fault_injection': 'selected RGB-D detection withheld at runtime measurement boundary'}
                self.session.command({'type': 'stop'})
                self._set_phase('operator_stop', sim_time)
                self.fault_observed = False
                self.fault_reference_hold = True
            return
        if self.phase == 'operator_stop':
            self._input(self.goal, clutch=True)
            if elapsed >= .5:
                self.results['operator_stop_hold'] = {'passed': self.fault_observed and self.fault_reference_hold}
                self._finish()

    def observe(self, robot_state, reference, sim_time):
        if self.done:
            return
        self.max_joint_violation = max(self.max_joint_violation, float(robot_state.get('joint_limit_violation', 0.)))
        self.invalid_states += int(not robot_state.get('valid', False))
        self.contact_stops += int(robot_state.get('collision_stop', False))
        self._last_reference = dict(reference)
        if self.phase in ('baseline_raw', 'eye_off_assisted', 'eye_on_no_device', 'eye_joystick') and self.goal is not None:
            delta = np.asarray(robot_state['ee_position'])-self.goal
            lateral = delta-self.target_normal*float(delta@self.target_normal)
            self.samples.append({'sim_time': float(sim_time), 'lateral_error_m': float(np.linalg.norm(lateral)),
                                 'approach_error_m': float(np.linalg.norm(delta)),
                                 'valid': bool(robot_state.get('valid', False)),
                                 'hold': bool(reference.get('hold', False)),
                                 'reason': self.session.control.get('reason', '')})
        expected = {'input_stale': 'input_stale', 'deadman_released': 'deadman_released',
                    'target_lost': 'target_lost', 'operator_stop': 'stopped'}.get(self.phase)
        if expected and self.session.control.get('reason') == expected:
            self.fault_observed = True
            # Reference is produced from the preceding measured state; compare
            # against that state, not a newly integrated position one dt later.
            before = np.asarray(self.session.robot['ee_position'])
            self.fault_reference_hold &= bool(reference.get('hold', False) and
                np.linalg.norm(np.asarray(reference['ee_target_world'])-before) < 1e-8)

    def report(self):
        baseline = self.results.get('baseline_raw', {}).get('mean_lateral_error_m')
        assisted = self.results.get('eye_off_assisted', {}).get('mean_lateral_error_m')
        improvement = None if baseline is None or assisted is None or baseline <= 0 else 1.-assisted/baseline
        checks = dict(self.results)
        checks['gaze_joystick_selection'] = {'passed': bool(self.selection_checks) and all(self.selection_checks.values()),
                                             'checks': dict(self.selection_checks)}
        checks['measured_state_limits'] = {'passed': self.invalid_states == 0 and self.max_joint_violation <= .035,
                                            'invalid_states': self.invalid_states,
                                            'max_joint_limit_violation_rad': self.max_joint_violation,
                                            'contact_stop_steps': self.contact_stops}
        checks['assist_improvement'] = {'passed': improvement is not None and improvement > .60,
                                        'fractional_lateral_error_reduction': improvement}
        return {'schema_version': 1, 'complete': self.done,
                'passed': bool(self.done and checks and all(v.get('passed', False) for v in checks.values())),
                'phase': self.phase, 'checks': checks, 'events': list(self.events),
                'goal_approach_world': None if self.goal is None else self.goal.tolist(),
                'input_provenance': 'automated synthetic controller and gaze; no live headset or eye hardware',
                'limits': ['Fixed-pelvis G1 manipulation fixture.',
                           'Position alignment only; no task semantics or screw turning evaluated.',
                           'Detector loss is an explicit measurement-dropout injection.',
                           'No claims about real eye-tracker accuracy or human task improvement.']}
