#!/usr/bin/env python3
"""Evaluate existing wrist images' geometric coverage from saved measured poses.

Known test CAD/joint truth is used solely for this camera-mount audit. This
does not measure learned-mask quality, actual unoccluded area or pose accuracy.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop_assist.fixture_assets import drawer_handle_components, _component_mesh
from teleop_assist.sensor_geometry import camera_from_link_pose


def proposed_mount():
    mount = np.eye(4)
    mount[:3, 3] = [-.025, 0., .065]
    # A fixed calibration aimed at a nominal grasp region in the hand frame.
    # It never observes a scene target and cannot change with target selection.
    forward = np.array([.14, .05, 0.])-mount[:3, 3]
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    mount[:3, :3] = np.column_stack((right, down, forward))
    return mount


def coverage(vertices_world, center_world, T_world_camera, intrinsics):
    points = (vertices_world-T_world_camera[:3, 3]) @ T_world_camera[:3, :3]
    centre = (center_world-T_world_camera[:3, 3]) @ T_world_camera[:3, :3]
    f = np.array([intrinsics['fx'], intrinsics['fy']])
    c = np.array([intrinsics['cx'], intrinsics['cy']])
    pixels = points[:, :2]/points[:, 2:] * f + c
    center_pixel = centre[:2]/centre[2]*f+c
    lo, hi = pixels.min(0), pixels.max(0)
    w, h = intrinsics['width'], intrinsics['height']
    clipped_lo, clipped_hi = np.maximum(lo, [0, 0]), np.minimum(hi, [w, h])
    clipped_area = float(np.prod(np.maximum(0, clipped_hi-clipped_lo)))
    return {'center_pixel': center_pixel.tolist(), 'bounds_pixel': [*lo.tolist(), *hi.tolist()],
            'depth_range_m': [float(points[:, 2].min()), float(points[:, 2].max())],
            'projected_bbox_fraction_inside_image': clipped_area/float(np.prod(hi-lo)),
            'center_inside_image': bool(0 <= center_pixel[0] < w and 0 <= center_pixel[1] < h and centre[2] > 0),
            'passes_current_wrist_proposal_geometry_gate': bool(
                points[:, 2].min() >= .08 and points[:, 2].max() <= 1.5
                and lo[0]-4 >= 1 and lo[1]-4 >= 1 and hi[0]+4 < w-1 and hi[1]+4 < h-1
                and min(hi-lo) >= 7)}


def depth_visibility(origin_world, T_world_camera, intrinsics, measured_depth):
    """Rasterize known CAD only for evaluation against measured optical depth."""
    height, width = measured_depth.shape
    expected = np.full((height, width), np.inf, dtype=float)
    k = intrinsics
    for component in drawer_handle_components()[1:]:
        vertices, faces = _component_mesh(component)
        camera = (vertices+origin_world-T_world_camera[:3, 3]) @ T_world_camera[:3, :3]
        pixels = camera[:, :2]/camera[:, 2:] * [k['fx'], k['fy']] + [k['cx'], k['cy']]
        for face in faces:
            index = np.asarray(face)
            xy, z = pixels[index], camera[index, 2]
            if min(z) <= .02:
                continue
            lo = np.maximum(np.floor(xy.min(0)).astype(int), [0, 0])
            hi = np.minimum(np.ceil(xy.max(0)).astype(int), [width-1, height-1])
            if np.any(hi < lo):
                continue
            x, y = np.meshgrid(np.arange(lo[0], hi[0]+1)+.5, np.arange(lo[1], hi[1]+1)+.5)
            a, b, c = xy
            denominator = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(denominator) < 1e-10:
                continue
            w0 = ((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/denominator
            w1 = ((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/denominator
            w2 = 1-w0-w1
            inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
            inverse_depth = w0/z[0]+w1/z[1]+w2/z[2]
            valid = inside & (inverse_depth > 0)
            depth = np.full_like(x, np.inf)
            depth[valid] = 1/inverse_depth[valid]
            section = expected[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
            np.minimum(section, depth, out=section)
    mask = np.isfinite(expected)
    observed = np.isfinite(measured_depth) & (measured_depth > 0)
    difference = np.full_like(expected, np.inf)
    np.subtract(measured_depth, expected, out=difference, where=mask & observed)
    match = mask & observed & (np.abs(difference) < .004)
    occluded = mask & observed & (measured_depth < expected-.004)
    return {'projected_handle_pixels': int(mask.sum()), 'depth_consistent_handle_pixels': int(match.sum()),
            'foreground_occluded_handle_pixels': int(occluded.sum()),
            'depth_consistent_fraction': float(match.sum()/max(1, mask.sum())),
            'foreground_occluded_fraction': float(occluded.sum()/max(1, mask.sum())),
            'tolerance_m': .004,
            'scope': 'Known CAD rasterization vs captured depth; not learned-mask accuracy'}


def run(probe, output):
    probe = Path(probe)
    result = json.loads((probe/'result.json').read_text())
    manifest = json.loads((probe/'backend/backend_manifest.json').read_text())
    rows = [json.loads(line) for line in (probe/'trace.jsonl').read_text().splitlines()]
    vertices = np.vstack([_component_mesh(c)[0] for c in drawer_handle_components()[1:]])
    calibration = manifest['camera_streams']['right_wrist']
    actual_mount = np.asarray(calibration['T_link_camera'])
    evaluated = {}
    used_exact_frames = []
    for phase in result['phases']:
        row = [row for row in rows if row['phase'] == phase][-1]
        robot = row['robot']
        rotation = Rotation.from_quat(robot['ee_quaternion']).as_matrix()
        wrist_position = np.asarray(robot['ee_position'])-rotation @ np.asarray(manifest['ee_offset_wrist_m'])
        center = np.asarray(result['initial_handle_bar_world']) + [robot['fixture_joint_states']['PullHandle']['position'], 0, 0]
        origin = center+[.065, 0, 0]
        exact_path = probe/(phase+'_right_wrist.npz')
        if exact_path.exists():
            with np.load(exact_path, allow_pickle=False) as frame:
                exact_camera = frame['T_world_camera'].copy()
                measured_depth = frame['depth'].copy()
            final_joint = result['phases'][phase]['final_drawer_position_m']
            center = np.asarray(result['initial_handle_bar_world'])+[final_joint, 0, 0]
            origin = center+[.065, 0, 0]
            # Recover the same measured wrist pose from the exact calibrated
            # frame for a fair alternative-calibration comparison.
            exact_wrist = exact_camera @ np.linalg.inv(actual_mount)
            actual_camera = exact_camera
            proposed_camera = exact_wrist @ proposed_mount()
            used_exact_frames.append(phase)
        else:
            actual_camera = camera_from_link_pose(wrist_position, robot['ee_quaternion'], actual_mount)
            proposed_camera = camera_from_link_pose(wrist_position, robot['ee_quaternion'], proposed_mount())
        evaluated[phase] = {
            'recorded_mount': coverage(vertices+origin, center, actual_camera, calibration['intrinsics']),
            'proposed_fixed_mount': coverage(vertices+origin, center, proposed_camera, calibration['intrinsics'])}
        if exact_path.exists():
            evaluated[phase]['measured_depth_visibility'] = depth_visibility(
                origin, actual_camera, calibration['intrinsics'], measured_depth)
    report = {'kind': 'offline_oracle_wrist_camera_fov_audit', 'probe': str(probe.resolve()),
              'actual_T_link_camera': actual_mount.tolist(), 'proposed_T_link_camera': proposed_mount().tolist(),
              'phases': evaluated, 'phases_with_exact_synchronized_camera_pose': used_exact_frames,
              'limitations': ['Phases listed under exact_synchronized_camera_pose use stored RGBD+final joint state. Older runs use final logged pose up to45ms before PNG.',
                              'Known CAD+joint state used for camera evaluation only, not perception.',
                              'Projected coverage does not establish unoccluded visible area or SAM quality.',
                              'Proposed mount needs actual rendered/segmented verification.']}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.probe, arguments.output), indent=2))
