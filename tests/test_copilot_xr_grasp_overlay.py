"""VR grasp review stays bound to the actual selected control candidate."""
from copy import deepcopy

import numpy as np
import pytest

from teleop_assist.copilot.xr_grasp_overlay import (BINDING_SCHEMA, render_grasp_overlay,
                                                 build_operator_grasp_preview)


@pytest.fixture
def scene():
    camera = np.full((480, 640, 3), [56, 67, 78], dtype=np.uint8)
    binding = dict(schema=BINDING_SCHEMA, capture_timestamp=9.95, clock_id='host_monotonic:test',
                   projection_sha256='a'*64, calibration_id='head-v1', width=640, height=480)
    candidates = []
    for index in range(2):
        pose = np.eye(4); pose[0, 3] = .3 + .03 * index
        candidates.append(dict(candidate_id=f'grasp-{index}', base_ee=pose.tolist(),
                               world_canonical=pose.tolist()))
    runtime = dict(timestamp=9.99, clock_id='host_monotonic:test', mode='isolated_PhysX',
                   candidates=deepcopy(candidates), snapshot=dict(confirmed=False, selected_candidate_id=None,
                                                                confirmed_candidate_id=None))
    previews = deepcopy(candidates)
    for index, candidate in enumerate(previews):
        x = 380 + 60 * index
        candidate.update(preview_hand_polygons=[dict(link='palm', uv=[[x, 260], [x+30, 260],
                          [x+30, 300], [x, 300]]), dict(link='finger', uv=[[x, 260], [x+8, 260],
                          [x+8, 240], [x, 240]])],
                         preview_uv=[[x+15, 280], [x+35, 280], [x+15, 260], [x+25, 270]])
    preview = dict(capture_timestamp=9.95, clock_id='host_monotonic:test', calibration_id='head-v1',
                   preview_current=True, motion_static=True, preview_frame=deepcopy(binding), candidates=previews)
    return camera, binding, runtime, preview


def render(scene):
    return render_grasp_overlay(*scene, now=10., clock_id='host_monotonic:test')


def select(scene, index=0, confirmed=False):
    scene[2]['snapshot'].update(selected_candidate_id=f'grasp-{index}', confirmed=confirmed,
                                confirmed_candidate_id=f'grasp-{index}' if confirmed else None)


def test_ready_shows_y_hint_without_implying_selection(scene):
    image, evidence, lines = render(scene)
    assert lines == ['Y: Griff anzeigen']
    assert evidence['state'] == 'ready' and evidence['candidate_id'] is None
    assert not evidence['geometry_visible'] and not evidence['confirmed']
    np.testing.assert_array_equal(image[200:], scene[0][200:])
    assert np.any(image[:90] != scene[0][:90])


def test_selected_actual_mesh_is_yellow_and_confirmed_mesh_green(scene):
    select(scene)
    selected, evidence, lines = render(scene)
    assert evidence['geometry_visible'] and evidence['state'] == 'selected'
    assert evidence['candidate_index'] == 1 and evidence['candidate_count'] == 2
    assert lines == ['Griff 1/2 · B bestaetigen · Y weiter']
    pixel = selected[300, 395]
    assert pixel[0] > 230 and pixel[1] > 190 and pixel[2] < 90
    select(scene, confirmed=True)
    confirmed, evidence, lines = render(scene)
    assert evidence['state'] == 'confirmed' and evidence['confirmed']
    assert 'Seitengriff rechts halten' in lines[0]
    pixel = confirmed[300, 395]
    assert pixel[1] > 200 and pixel[0] < 100
    assert evidence['movement_authorized'] is False and evidence['controls_written'] is False


def test_y_change_moves_only_one_ghost_and_keeps_runtime_order(scene):
    select(scene)
    first, _, _ = render(scene)
    select(scene, 1)
    second, evidence, _ = render(scene)
    assert evidence['candidate_id'] == 'grasp-1' and evidence['candidate_index'] == 2
    np.testing.assert_array_equal(second[240:301, 380:411], scene[0][240:301, 380:411])
    assert np.any(first[240:301, 380:411] != scene[0][240:301, 380:411])
    assert np.any(second[240:301, 440:471] != scene[0][240:301, 440:471])


@pytest.mark.parametrize('change', [
    lambda b, r, p: p.update(capture_timestamp=9.7),
    lambda b, r, p: p.update(capture_timestamp=10.02),
    lambda b, r, p: p.update(clock_id='other'),
    lambda b, r, p: p.update(preview_current=False),
    lambda b, r, p: p.update(motion_static=False),
    lambda b, r, p: p['preview_frame'].update(projection_sha256='b'*64),
    lambda b, r, p: p['preview_frame'].update(calibration_id='wrist-v1'),
    lambda b, r, p: p['preview_frame'].update(width=320),
    lambda b, r, p: p['preview_frame'].update(capture_timestamp=9.7),
    lambda b, r, p: p['preview_frame'].update(clock_id='other'),
    lambda b, r, p: b.update(capture_timestamp=9.7),
    lambda b, r, p: r.update(timestamp=9.49),
    lambda b, r, p: r.update(clock_id='other'),
    lambda b, r, p: r.update(mode='recorded'),
    lambda b, r, p: r.update(candidates=[]),
    lambda b, r, p: p['candidates'][0]['base_ee'][0].__setitem__(3, .31),
    lambda b, r, p: p['candidates'][0]['world_canonical'][0].__setitem__(3, .31),
    lambda b, r, p: p['candidates'][0].update(candidate_id='unknown'),
    lambda b, r, p: p['candidates'].append(deepcopy(p['candidates'][0])),
])
def test_stale_mismatched_or_unoffered_preview_preserves_camera(scene, change):
    select(scene)
    change(*scene[1:])
    image, evidence, lines = render(scene)
    assert not evidence['geometry_visible'] and evidence['state'] == 'waiting'
    assert lines
    np.testing.assert_array_equal(image, scene[0])


@pytest.mark.parametrize('bad_geometry', [
    [], [{'uv': [[0, 1], [2, 3]]}], [{'uv': [[0, 1], [2, 3], [float('nan'), 2]]}],
    [{'uv': [[0, 1], [2, 3], [1e50, 2]]}], [{'uv': [[10000, 0], [11000, 0], [11000, 100]]}],
    [{'uv': [[0, 1]]*1025}], [{'uv': [[True, 1], [2, 3], [1, 2]]}],
])
def test_unbounded_or_missing_geometry_does_not_get_invented(scene, bad_geometry):
    select(scene)
    scene[3]['candidates'][0]['preview_hand_polygons'] = bad_geometry
    image, evidence, _ = render(scene)
    assert evidence['reason'] == 'projected_hand_geometry_unavailable'
    np.testing.assert_array_equal(image, scene[0])


def test_confirmation_requires_explicit_boolean_and_matching_id(scene):
    select(scene)
    scene[2]['snapshot']['confirmed_candidate_id'] = 'grasp-1'
    _, evidence, _ = render(scene)
    assert evidence['candidate_id'] == 'grasp-0' and not evidence['confirmed']
    scene[2]['snapshot']['confirmed'] = True
    _, evidence, _ = render(scene)
    assert evidence['candidate_id'] == 'grasp-1' and evidence['confirmed']
    scene[2]['snapshot']['confirmed_candidate_id'] = 'missing'
    image, evidence, _ = render(scene)
    assert not evidence['geometry_visible']
    np.testing.assert_array_equal(image, scene[0])


def test_render_has_no_input_mutations_and_preserves_outside_overlay(scene):
    select(scene)
    before = deepcopy(scene)
    image, evidence, _ = render(scene)
    np.testing.assert_array_equal(scene[0], before[0])
    assert scene[1:] == before[1:]
    x, y, width, height = evidence['inset_xywh']
    changed = np.any(image != scene[0], axis=2)
    changed[y:y+height, x:x+width] = False
    changed[238:303, 378:418] = False
    assert not changed.any()


def test_new_camera_frame_id_can_share_only_same_projection_and_recent_time(scene):
    select(scene)
    scene[1].update(frame_id=100, capture_timestamp=10.)
    scene[3]['preview_frame'].update(frame_id=99, capture_timestamp=9.95)
    _, evidence, _ = render(scene)
    assert evidence['geometry_visible']
    scene[1]['projection_sha256'] = 'c'*64
    _, evidence, _ = render(scene)
    assert not evidence['geometry_visible']


def test_actual_dex3_palm_with_144_hull_vertices_is_visible(scene):
    # Frozen actual OPEN Dex3 palm projection from the recorded head-camera
    # review 2026-10-07; embedded so the regression needs no live run or meshes.
    palm = [
        [468.8811, 211.0909], [475.3816, 210.9453], [476.4384, 210.9737], [477.5134, 211.1046], [478.5754, 211.3345], [479.5929, 211.6573],
        [480.5355, 212.0640], [486.3098, 214.7469], [486.3691, 214.7859], [486.6851, 215.0027], [487.0546, 215.2927], [487.3403, 215.5414],
        [487.6122, 215.8026], [487.8711, 216.0739], [488.1172, 216.3540], [488.3513, 216.6421], [488.7487, 217.1767], [489.1187, 217.7346],
        [489.1563, 217.7944], [489.5044, 218.3675], [489.8046, 218.8966], [490.0179, 219.3003], [490.5703, 220.4298], [491.2081, 221.9395],
        [492.1567, 224.3687], [492.2257, 224.5468], [492.9018, 226.3468], [493.3922, 227.7119], [493.5343, 228.1097], [494.0360, 229.5586],
        [494.9546, 232.4603], [495.7596, 235.4135], [495.8680, 235.8408], [496.5495, 238.8348], [497.1552, 242.2821], [497.3917, 243.9926],
        [497.4450, 244.4280], [497.6187, 246.1970], [497.7387, 247.9759], [498.3270, 257.4879], [498.3426, 257.9323], [498.3232, 258.4600],
        [498.2950, 258.7711], [498.1977, 259.4409], [498.1461, 259.7084], [497.9753, 260.4240], [497.8591, 260.8268], [497.4919, 261.8793],
        [497.2858, 262.3825], [496.9622, 263.0952], [495.8966, 265.2162], [495.5530, 265.8431], [494.9428, 266.8520], [494.2798, 267.8276],
        [493.5642, 268.7671], [493.1843, 269.2244], [492.5629, 269.9177], [491.9108, 270.5791], [491.6421, 270.8335], [491.2260, 271.2079],
        [490.7519, 271.6066], [490.5054, 271.8028], [489.7321, 272.3712], [489.1141, 272.7760], [488.6621, 273.0437], [488.4755, 273.1472],
        [487.8096, 273.4835], [487.5070, 273.6197], [486.8182, 273.8933], [486.1062, 274.1273], [485.2975, 274.3395], [484.4530, 274.5094],
        [483.7329, 274.6186], [483.5647, 274.6399], [482.9660, 274.7040], [482.6230, 274.7331], [482.4032, 274.7490], [481.4465, 274.7955],
        [481.2477, 274.8009], [480.0044, 274.8056], [478.6343, 274.7652], [478.3604, 274.7534], [477.2331, 274.6906], [476.1630, 274.6053],
        [475.7215, 274.5613], [475.1520, 274.4955], [474.2026, 274.3581], [473.3177, 274.1894], [473.0734, 274.1338], [472.3877, 273.9510],
        [471.7613, 273.7381], [469.4606, 272.9427], [468.8869, 272.6929], [468.7622, 272.6293], [468.3019, 272.3453], [467.9502, 272.0408],
        [467.7588, 271.8676], [467.1858, 271.2849], [466.3496, 270.3494], [466.1153, 270.0790], [465.8339, 269.7416], [465.4732, 269.3007],
        [465.0181, 268.6766], [464.9051, 268.5039], [464.6475, 268.0968], [464.4958, 267.8551], [461.0819, 262.0612], [460.6745, 261.3458],
        [457.4144, 255.6189], [457.3263, 255.4508], [457.3230, 255.4434], [457.2321, 255.2398], [456.5115, 253.4840], [455.8157, 251.7006],
        [455.1692, 249.9390], [455.1679, 249.9355], [455.1013, 249.7333], [455.0397, 249.4922], [454.9890, 249.2321], [454.9516, 248.9632],
        [454.9251, 248.6727], [454.9200, 248.6075], [454.9265, 248.4403], [454.9439, 248.2076], [454.9447, 248.2011], [454.9831, 248.0095],
        [455.3892, 246.4403], [455.4647, 246.1675], [455.7962, 245.1276], [455.9471, 244.6901], [457.9073, 239.0982], [457.9704, 238.9188],
        [461.8847, 227.8187], [461.9534, 227.6245], [464.2401, 221.1763], [464.4099, 220.7056], [467.5424, 212.0622], [467.5586, 212.0364],
        [467.5825, 212.0131], [468.6735, 211.1509], [468.7044, 211.1310], [468.7415, 211.1146], [468.7841, 211.1023], [468.8310, 211.0943],
    ]
    assert len(palm) == 144
    select(scene)
    scene[3]['candidates'][0].update(preview_hand_polygons=[dict(link='right_hand_palm_link', uv=palm)],
                                    preview_uv=None)
    image, evidence, _ = render(scene)
    assert evidence['geometry_visible'] and evidence['candidate_id'] == 'grasp-0'
    assert np.any(image[211:276, 454:500] != scene[0][211:276, 454:500])


@pytest.fixture
def projection_source():
    poses=[]
    for i in range(2):
        pose=np.eye(4);pose[2,3]=1.;pose[0,3]=i*.02
        poses.append(dict(candidate_id=f'grasp-{i}',base_ee=pose.tolist(),world_canonical=pose.tolist()))
    frame=dict(clock_id='host_monotonic:projection',calibration_id='operator-head-v1',timestamp=20.,
               T_world_camera=np.eye(4),intrinsics=np.array([[100,0,320],[0,100,240],[0,0,1]]))
    batch=dict(clock_id=frame['clock_id'],capture_timestamp=19.94,motion_static=True,
               processed_frame=dict(rejected=False),candidates=deepcopy(poses))
    class Preview:
        def __init__(self):self.calls=[]
        def project(self,pose,current_frame):
            self.calls.append((deepcopy(pose),current_frame))
            return [dict(link='palm',uv=[[300,220],[330,220],[330,250]])]
    return poses,deepcopy(poses[0]),batch,frame,Preview()


def test_new_source_observation_keeps_own_timestamp_and_projects_only_selected(projection_source):
    offered,selected,batch,frame,hand=projection_source
    before=deepcopy((offered,selected,batch))
    result=build_operator_grasp_preview(*projection_source)
    assert result['capture_timestamp']==19.94 and result['capture_timestamp']!=frame['timestamp']
    assert result['preview_current'] and result['motion_static'] is True
    assert result['calibration_id']=='operator-head-v1' and len(result['candidates'])==2
    assert len(hand.calls)==1 and hand.calls[0][1] is frame
    assert result['candidates'][0]['preview_uv'][0]==[320.,240.]
    assert result['candidates'][0]['preview_hand_polygons'][0]['link']=='palm'
    assert 'preview_uv' not in result['candidates'][1]
    assert (offered,selected,batch)==before
    # Independently newer validation advances only to the actual source time.
    batch['capture_timestamp']=19.97
    assert build_operator_grasp_preview(*projection_source)['capture_timestamp']==19.97


@pytest.mark.parametrize('pose_name',['base_ee','world_canonical'])
def test_source_pose_change_excludes_only_changed_offered_candidate(projection_source,pose_name):
    offered,selected,batch,frame,hand=projection_source
    batch['candidates'][0][pose_name][0][3]+=.01
    result=build_operator_grasp_preview(*projection_source)
    assert [c['candidate_id'] for c in result['candidates']]==['grasp-1']
    assert not hand.calls


@pytest.mark.parametrize('fault',['clock','rejected','missing_validation','null_validation'])
def test_invalid_source_does_not_produce_grasp_preview(projection_source,fault):
    offered,selected,batch,frame,hand=projection_source
    if fault=='clock':batch['clock_id']='host_monotonic:other'
    elif fault=='rejected':batch['processed_frame']['rejected']=True
    elif fault=='missing_validation':batch.pop('processed_frame')
    else:batch['processed_frame']=None
    result=build_operator_grasp_preview(*projection_source)
    assert not result['preview_current'] and not result['candidates'] and not hand.calls
    assert result['capture_timestamp']==19.94


def test_no_selection_or_changed_frozen_selection_cannot_project_a_grasp(projection_source):
    offered,selected,batch,frame,hand=projection_source
    result=build_operator_grasp_preview(offered,None,batch,frame,hand)
    assert len(result['candidates'])==2 and not hand.calls
    selected['base_ee'][0][3]+=.01
    result=build_operator_grasp_preview(*projection_source)
    assert len(result['candidates'])==2 and not hand.calls
    assert all('preview_hand_polygons' not in c for c in result['candidates'])


def test_projection_does_not_make_nonstationary_or_old_source_current(projection_source):
    offered,selected,batch,frame,hand=projection_source
    batch.update(capture_timestamp=5.,motion_static=False)
    result=build_operator_grasp_preview(*projection_source)
    assert result['capture_timestamp']==5. and result['motion_static'] is False
    # The renderer, not this pure projection helper, checks source age.
    assert result['preview_current'] is True
