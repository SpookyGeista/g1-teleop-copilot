"""Real pretrained Grounding DINO + SAM 2.1, calibrated RGB-D and tracked IDs.

The models recognize and segment visible objects. Metric geometry is measured
from depth, never from simulator object poses. Object semantics do not imply an
action, hidden geometry, a joint axis or a verified grasp. No fallback detector
is used if model loading or inference fails.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import numbers
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .model_registry import DEFAULT_MODEL_ROOT, MODELS, model_paths
from .perception import _calibration, _iou, backproject


@dataclass(frozen=True)
class LearnedPerceptionConfig:
    model_root: str = str(DEFAULT_MODEL_ROOT)
    device: str = 'cuda'
    detector_model: str = 'grounding_dino'
    prompts: tuple[str, ...] = ('handle', 'knob', 'button', 'lever', 'screw',
                               'switch', 'connector', 'cable', 'drawer')
    box_threshold: float = .25
    text_threshold: float = .20
    min_mask_iou: float = .50
    max_targets: int = 20
    detector_short_edge: int = 640
    detector_long_edge: int = 1024
    min_depth_m: float = .08
    max_depth_m: float = 3.0
    min_mask_pixels: int = 24
    max_mask_fraction: float = .40
    min_valid_depth_fraction: float = .70
    max_object_extent_m: float = .60
    association_distance_m: float = .045
    track_retention_s: float = 2.0
    ambiguity_margin_m: float = .009
    max_surface_residual_m: float = .008
    label_thresholds: dict[str, float] = field(default_factory=dict)
    label_min_extent_m: dict[str, float] = field(default_factory=dict)


def canonical_label(label):
    text = str(label).lower().strip(' .')
    # These are lexical aliases, not an inferred task/action.
    for canonical, words in (
        ('handle', ('handle', 'door pull')), ('knob', ('knob', 'dial', 'wheel')),
        ('button', ('button',)), ('lever', ('lever',)), ('screw', ('screw', 'bolt')),
        ('switch', ('switch',)), ('connector', ('connector', 'socket', 'plug')),
        ('cable', ('cable', 'wire')), ('drawer', ('drawer',))):
        if any(word in text for word in words):
            return canonical
    return text.replace(' ', '_')


def mask_to_target(mask, depth, intrinsics, T_world_camera, *, label, detection_score,
                   mask_score, config=None):
    """Measured target geometry, or (None, rejection reason).

    Exported for wrist refinement and numerical tests; it does not run a model.
    The box encloses the visible depth surface, not a completed object volume.
    """
    c = config or LearnedPerceptionConfig()
    k, t = _calibration(intrinsics, T_world_camera)
    z = np.asarray(depth, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if m.shape != z.shape or m.ndim != 2:
        raise ValueError('Mask and depth must be matching HxW arrays')
    count = int(m.sum())
    if count < c.min_mask_pixels or count > m.size * c.max_mask_fraction:
        return None, 'mask_pixel_extent'
    valid = m & np.isfinite(z) & (z > c.min_depth_m) & (z < c.max_depth_m)
    fraction = float(valid.sum()/count)
    if fraction < c.min_valid_depth_fraction:
        return None, 'insufficient_mask_depth'
    yy, xx = np.nonzero(valid)
    points = backproject(np.column_stack((xx, yy)), z[yy, xx], k)
    # Depth outliers stay absent: trim measured samples, never fill holes.
    lo, hi = np.percentile(points, [1, 99], axis=0)
    extent = hi-lo
    if np.max(extent) > c.max_object_extent_m:
        return None, 'object_metric_extent'
    if np.max(extent) < c.label_min_extent_m.get(canonical_label(label), 0.):
        return None, 'configured_class_minimum_metric_extent'
    eroded = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ey, ex = np.nonzero(eroded)
    if len(ex) < 12:
        return None, 'insufficient_interior_depth'
    interior = backproject(np.column_stack((ex, ey)), z[ey, ex], k)
    center = np.median(interior, axis=0)
    index = int(np.argmin(np.sum((interior-center)**2, axis=1)))
    center = interior[index]  # an actual visible pixel, including nonconvex masks
    distances = np.sum((interior-center)**2, axis=1)
    local = interior[np.argsort(distances)[:min(120, len(interior))]]
    local_center = np.mean(local, axis=0)
    _, singular, vectors = np.linalg.svd(local-local_center, full_matrices=False)
    if len(singular) < 3 or singular[1] < .002:
        return None, 'degenerate_surface_normal'
    normal = vectors[-1]
    if normal @ center > 0:
        normal = -normal
    rms = float(np.sqrt(np.mean(((local-local_center) @ normal)**2)))
    if rms > c.max_surface_residual_m:
        return None, 'unreliable_local_surface'
    _, _, whole_vectors = np.linalg.svd(interior-np.mean(interior, axis=0), full_matrices=False)
    major = whole_vectors[0]
    major = major-normal*float(normal @ major)
    if np.linalg.norm(major) < 1e-6:
        major = vectors[0]
    major /= np.linalg.norm(major)
    world_center = t[:3, :3] @ center + t[:3, 3]
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour).reshape(-1, 2)
    confidence = float(np.clip(math.sqrt(max(0., detection_score)*max(0., mask_score)), 0, 1))
    # Geometry quality is separate from open-vocabulary detector scores. These
    # conservative heuristics are not calibrated statistical probabilities.
    pixel_footprint = float(center[2]/min(k['fx'], k['fy']))
    position_sigma = max(.004, 2*rms+2*pixel_footprint)
    pose_confidence = float(np.clip(fraction * min(1., len(local)/60)
                                    * (1-min(.5, rms/.016)), 0, .95))
    yall, xall = np.nonzero(m)
    return {
        'label': canonical_label(label), 'semantic_label': canonical_label(label),
        'semantic_phrase': str(label), 'confidence': confidence,
        'detection_confidence': float(detection_score), 'segmentation_confidence': float(mask_score),
        'bbox_px': [int(xall.min()), int(yall.min()), int(xall.max()+1), int(yall.max()+1)],
        'polygon_px': hull.astype(int).tolist(), 'pixel_area': count,
        'center_camera': center.tolist(), 'center_world': world_center.tolist(),
        'normal_world': (t[:3, :3] @ normal).tolist(),
        'major_axis_world': (t[:3, :3] @ major).tolist(),
        'box_camera': {'center': ((lo+hi)/2).tolist(), 'size': np.maximum(extent, .001).tolist(),
                       'quaternion_xyzw': [0., 0., 0., 1.]},
        'surface_pixel_px': [int(ex[index]), int(ey[index])],
        'surface_residual_m': rms, 'normal_source': 'sam2_mask_local_measured_depth',
        'depth_valid_fraction': fraction,
        'pose_confidence': pose_confidence, 'position_sigma_m': position_sigma,
        'geometry_quality_source': 'depth_support_residual_and_pixel_size_heuristic',
        'geometry_scope': 'visible_depth_surface', 'semantic_model': True,
        'perception_method': 'grounding_dino_sam2_rgbd',
        'observed': True, 'stale': False, 'action_inferred': False,
    }, None


class LearnedCabinetPerception:
    """Local-only model loading with JSON-compatible target results.

    ``last_masks`` maps current target IDs to HxW bool arrays. These arrays are
    intentionally outside the JSON response. ``segment_boxes`` reuses SAM for
    separately calibrated wrist views. Inference is single-worker serialized.
    """
    def __init__(self, config=None):
        self.config = (LearnedPerceptionConfig(**config) if isinstance(config, dict)
                       else config or LearnedPerceptionConfig())
        c = self.config
        if not (0 <= c.box_threshold <= 1 and 0 <= c.text_threshold <= 1
                and 0 <= c.min_mask_iou <= 1 and 0 < c.min_depth_m < c.max_depth_m
                and 0 < c.min_valid_depth_fraction <= 1 and 0 < c.max_mask_fraction <= 1
                and c.max_targets > 0 and c.association_distance_m > 0 and c.track_retention_s > 0
                and c.min_mask_pixels >= 12 and c.prompts):
            raise ValueError('Invalid learned perception configuration')
        if c.detector_model not in ('grounding_dino', 'grounding_dino_base'):
            raise ValueError('Unknown registered detector model')
        paths = model_paths(c.model_root, (c.detector_model, 'sam2'))
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        self.torch = torch
        if c.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; explicit device=cpu is supported')
        self.device = torch.device(c.device)
        # CPU oversubscription also slows small pre/post-processing GPU workloads.
        torch.set_num_threads(min(8, torch.get_num_threads()))
        self.detector_processor = AutoProcessor.from_pretrained(str(paths[c.detector_model]), local_files_only=True)
        self.detector_processor.image_processor.size = {
            'shortest_edge': c.detector_short_edge, 'longest_edge': c.detector_long_edge}
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(paths[c.detector_model]), local_files_only=True).to(self.device).eval()
        self.segmenter_processor = Sam2Processor.from_pretrained(str(paths['sam2']), local_files_only=True)
        self.segmenter = Sam2Model.from_pretrained(str(paths['sam2']), local_files_only=True).to(self.device).eval()
        self.prompt = '. '.join(str(x).strip(' .') for x in c.prompts) + '.'
        self.reset()

    def reset(self):
        self._tracks = {}
        self._next_id = 1
        self._last_frame = self._last_time = None
        self.last_masks = {}
        self.last_detections = []

    def synchronize_confirmed_observation(self, target):
        """Keep an existing selected identity alive after verified wrist recovery.

        This accepts only the bounded multiview module's fresh measurement;
        unknown IDs and label changes are rejected. It creates no head mask.
        """
        previous=self._tracks.get(target.get('id'))
        if (previous is None or target.get('perception_method') not in ('confirmed_identity_wrist_sam2_rgbd','confirmed_identity_head_sam2_rgbd')
                or target.get('stale') or not target.get('observed')
                or target.get('label')!=previous['label']):
            return False
        timestamp=float(target['timestamp'])
        center=np.asarray(target['center_world'],dtype=float)
        if (center.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(timestamp)
                or timestamp<previous['timestamp']
                or np.linalg.norm(center-previous['center_world'])>.05):
            return False
        for identity,other in self._tracks.items():
            if identity!=target['id'] and np.linalg.norm(center-other['center_world'])<.020:
                return False
        self._tracks[target['id']]={**target,'timestamp':timestamp,'confirmed_external_measurement':True}
        return True

    def segment_boxes(self, rgb, boxes_xyxy):
        """Run real SAM 2.1 on box prompts; return one bool mask and score per box."""
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('RGB must be HxWx3 uint8')
        boxes = np.asarray(boxes_xyxy, dtype=float)
        if boxes.size == 0:
            return []
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
            raise ValueError('Boxes must be finite Nx4 xyxy')
        if np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError('Boxes require positive area')
        inputs = self.segmenter_processor(images=image, input_boxes=[boxes.tolist()], return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            output = self.segmenter(**inputs, multimask_output=False)
        masks = self.segmenter_processor.post_process_masks(
            output.pred_masks.cpu(), inputs['original_sizes'].cpu())[0]
        scores = output.iou_scores.detach().cpu().numpy().reshape(len(boxes), -1)
        masks = masks.detach().cpu().numpy().reshape(len(boxes), -1, *image.shape[:2])
        return [{'mask': mask[0].astype(bool), 'confidence': float(np.clip(score[0], 0, 1))}
                for mask, score in zip(masks, scores)]

    def segment_points(self, rgb, box_xyxy, positive_points, negative_points=()):
        """Real SAM multimask hypotheses from measured positive/negative pixels."""
        image=np.asarray(rgb)
        points=np.asarray([*positive_points,*negative_points],dtype=float)
        labels=[1]*len(positive_points)+[0]*len(negative_points)
        box=np.asarray(box_xyxy,dtype=float)
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('RGB must be HxWx3 uint8')
        if (points.ndim!=2 or points.shape[1]!=2 or len(positive_points)<1
                or not np.isfinite(points).all() or box.shape!=(4,)):
            raise ValueError('Finite point prompts and one box required')
        inputs=self.segmenter_processor(images=image,input_boxes=[[box.tolist()]],
            input_points=[[points.tolist()]],input_labels=[[labels]],return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            output=self.segmenter(**inputs,multimask_output=True)
        masks=self.segmenter_processor.post_process_masks(output.pred_masks.cpu(),inputs['original_sizes'].cpu())[0]
        masks=masks.detach().cpu().numpy().reshape(-1,*image.shape[:2])
        scores=output.iou_scores.detach().cpu().numpy().ravel()
        return [{'mask':mask.astype(bool),'confidence':float(np.clip(score,0,1))}
                for mask,score in zip(masks,scores)]

    def _detect(self, rgb):
        inputs = self.detector_processor(images=rgb, text=self.prompt, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            output = self.detector(**inputs)
        result = self.detector_processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold, target_sizes=[rgb.shape[:2]])[0]
        labels = result.get('text_labels', result.get('labels'))
        raw = [{'bbox_px': box.detach().cpu().tolist(), 'confidence': float(score), 'label': str(label)}
               for box, score, label in zip(result['boxes'], result['scores'], labels)]
        selected = []
        allowed_labels = {canonical_label(prompt) for prompt in self.config.prompts}
        for det in sorted(raw, key=lambda x: x['confidence'], reverse=True):
            if not canonical_label(det['label']) or canonical_label(det['label']) not in allowed_labels:
                continue
            if det['confidence'] < self.config.label_thresholds.get(canonical_label(det['label']), 0.):
                continue
            # Cross-label NMS prevents a button/switch alias creating two IDs.
            if any(_iou(det['bbox_px'], other['bbox_px']) > .65 for other in selected):
                continue
            selected.append(det)
            if len(selected) >= self.config.max_targets:
                break
        return selected

    def _associate(self, detections, masks, timestamp):
        c = self.config
        expired = [identity for identity, tr in self._tracks.items()
                   if timestamp-tr['timestamp'] > c.track_retention_s]
        for identity in expired:
            del self._tracks[identity]
        ids = list(self._tracks)
        cost = np.full((len(ids), len(detections)), 1e6)
        compatible_labels = ({'button', 'switch', 'knob'}, {'handle', 'lever'})
        for row, identity in enumerate(ids):
            old = self._tracks[identity]
            for col, det in enumerate(detections):
                same = old['label'] == det['label'] or any(
                    {old['label'], det['label']} <= group for group in compatible_labels)
                distance = float(np.linalg.norm(np.asarray(old['center_world'])-det['center_world']))
                if same and distance < c.association_distance_m:
                    cost[row, col] = distance
        ambiguous = set()
        # With near-equally plausible identities, withhold the measurement. A
        # lock must fail rather than silently attach to a neighboring component.
        for col in range(len(detections)):
            values = sorted(v for v in cost[:, col] if v < 1e5)
            if len(values) > 1 and values[1]-values[0] < c.ambiguity_margin_m:
                ambiguous.add(col)
        for row in range(len(ids)):
            candidates = [col for col in range(len(detections)) if cost[row, col] < 1e5]
            if len(candidates) > 1:
                ordered = sorted(candidates, key=lambda col: cost[row, col])
                if cost[row, ordered[1]]-cost[row, ordered[0]] < c.ambiguity_margin_m:
                    ambiguous.update(ordered[:2])
        for col in ambiguous:
            cost[:, col] = 1e6
        assignments = {}
        if ids and detections:
            rows, cols = linear_sum_assignment(cost)
            assignments = {int(col): ids[int(row)] for row, col in zip(rows, cols) if cost[row, col] < 1e5}
        observed, self.last_masks = [], {}
        for col, (det, mask) in enumerate(zip(detections, masks)):
            if col in ambiguous:
                continue
            identity = assignments.get(col)
            if identity is None:
                identity = f'learned_{self._next_id:03d}'
                self._next_id += 1
            det['id'] = identity
            self._tracks[identity] = {**det, 'timestamp': timestamp}
            observed.append(det)
            self.last_masks[identity] = mask
        return observed, {'expired_track_ids': expired,
                          'stale_track_ids': [x for x in self._tracks if x not in self.last_masks],
                          'ambiguous_detection_count': len(ambiguous), 'tracked_count': len(self._tracks)}

    def process(self, rgb, depth, intrinsics, T_world_camera, frame_id, timestamp):
        image, z = np.asarray(rgb), np.asarray(depth)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 8:
            raise ValueError('RGB must be HxWx3 uint8')
        if z.shape != image.shape[:2] or z.dtype.kind not in 'fiu':
            raise ValueError('Metric optical depth must match RGB')
        _calibration(intrinsics, T_world_camera)
        if isinstance(frame_id, bool) or not isinstance(frame_id, numbers.Integral) or frame_id < 0:
            raise ValueError('Nonnegative integer frame_id required')
        if isinstance(timestamp, bool) or not isinstance(timestamp, numbers.Real) or not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Finite nonnegative acquisition timestamp required')
        frame_id, timestamp = int(frame_id), float(timestamp)
        diagnostics = {'method': 'grounding_dino_sam2_rgbd', 'semantic_model': True,
                       'source': 'pretrained_RGB_models_and_measured_optical_depth',
                       'frame_id': frame_id, 'timestamp': timestamp, 'stale': False,
                       'models': {name: {'repo_id': info['repo_id'], 'revision': info['revision']}
                                  for name, info in MODELS.items() if name in (self.config.detector_model, 'sam2')}, 'config': asdict(self.config),
                       'scene_truth_used': False, 'action_inference': False}
        if self._last_frame is not None and (frame_id <= self._last_frame or timestamp <= self._last_time):
            self.last_masks = {}
            diagnostics.update(reason='nonadvancing_sensor_frame', stale=True, stale_track_ids=list(self._tracks))
            return {'targets': [], 'diagnostics': diagnostics}
        self._last_frame, self._last_time = frame_id, timestamp
        start = time.perf_counter()
        detections = self._detect(image)
        self.last_detections = detections
        detector_s = time.perf_counter()-start
        segment_start = time.perf_counter()
        segments = self.segment_boxes(image, [d['bbox_px'] for d in detections])
        segmenter_s = time.perf_counter()-segment_start
        candidates, candidate_masks, rejected = [], [], {}
        for detection, segment in zip(detections, segments):
            reason = None
            target = None
            if segment['confidence'] < self.config.min_mask_iou:
                reason = 'low_segmentation_confidence'
            else:
                target, reason = mask_to_target(segment['mask'], z, intrinsics, T_world_camera,
                    label=detection['label'], detection_score=detection['confidence'],
                    mask_score=segment['confidence'], config=self.config)
            if target is None:
                rejected[reason] = rejected.get(reason, 0)+1
                continue
            duplicate = False
            for previous_mask in candidate_masks:
                union = int(np.count_nonzero(previous_mask | segment['mask']))
                if union and np.count_nonzero(previous_mask & segment['mask'])/union > .65:
                    duplicate = True
                    break
            if duplicate:
                rejected['duplicate_instance_mask'] = rejected.get('duplicate_instance_mask', 0)+1
                continue
            target.update(frame_id=frame_id, timestamp=timestamp)
            candidates.append(target)
            candidate_masks.append(segment['mask'])
        targets, tracking = self._associate(candidates, candidate_masks, timestamp)
        diagnostics.update(tracking)
        diagnostics.update(reason='observed_learned_components' if targets else 'no_usable_learned_components',
                           raw_detection_count=len(detections), detected_count=len(targets), rejected_components=rejected,
                           detector_seconds=detector_s, segmenter_seconds=segmenter_s,
                           total_seconds=time.perf_counter()-start)
        return {'targets': targets, 'diagnostics': diagnostics}
