"""Recorded acquisition, publication, and control events use an honest timeline."""
import pytest

from tools.replay_assist import prepare_replay_timeline, replay_control_sample, replay_timing


def test_publication_default_selects_control_that_existed_when_observation_arrived():
    frame={'frame_id':7,'timestamp':1.,'publish_timestamp':1.2}
    controls=[{'timestamp':1.,'selection':{'id':'old'}},
              {'timestamp':1.1,'selection':{'id':'new'}},
              {'timestamp':1.3,'selection':{'id':'future'}}]
    stamps=[sample['timestamp'] for sample in controls]
    recorded,timing=prepare_replay_timeline([frame])[0]
    assert replay_control_sample(controls,stamps,timing)['selection']['id']=='new'
    assert recorded is frame and recorded['timestamp']==1.
    assert timing['acquisition_timestamp']==1.
    assert timing['publication_timestamp']==1.2
    assert timing['capture_to_publish_s']==pytest.approx(.2)
    assert timing['timeline_timestamp']==1.2
    assert timing['timeline_source']=='publication'
    assert timing['scope']=='perception_frames_only'
    research=replay_timing(frame,'acquisition')
    assert replay_control_sample(controls,stamps,research)['selection']['id']=='old'
    assert research['timeline_source']=='acquisition'
    assert research['publication_timestamp']==1.2


def test_start_and_playback_offsets_share_publication_time_and_allow_frame_gaps():
    frames=[{'frame_id':1,'timestamp':1.,'publish_timestamp':1.2},
            {'frame_id':4,'timestamp':1.1,'publish_timestamp':1.7},
            {'frame_id':9,'timestamp':1.6,'publish_timestamp':1.8}]
    full=prepare_replay_timeline(frames)
    assert [timing['timeline_timestamp']-full[0][1]['timeline_timestamp'] for _,timing in full]==pytest.approx([0.,.5,.6])
    assert [frame['frame_id'] for frame,_ in prepare_replay_timeline(frames,start=.4)]==[4,9]
    assert [frame['frame_id'] for frame,_ in prepare_replay_timeline(frames,'acquisition',.4)]==[9]


def test_historical_acquisition_fallback_is_explicit_and_controls_use_it():
    frame={'frame_id':20,'timestamp':1.}
    timing=replay_timing(frame)
    assert timing['timeline_source']=='acquisition_fallback'
    assert timing['publication_timestamp'] is None
    assert timing['capture_to_publish_s'] is None
    controls=[{'timestamp':.9,'selection':{'id':'old'}},
              {'timestamp':1.1,'selection':{'id':'future'}}]
    assert replay_control_sample(controls,[.9,1.1],timing)['selection']['id']=='old'
    assert replay_control_sample(controls,[1.1,1.2],timing) is None


def test_mixed_historical_timeline_cannot_silently_reorder_frames():
    with pytest.raises(ValueError,match='backwards'):
        prepare_replay_timeline([{'timestamp':1.,'publish_timestamp':1.3},{'timestamp':1.2}])


@pytest.mark.parametrize('frame',[{'timestamp':float('nan')},
    {'timestamp':1.,'publish_timestamp':float('nan')},
    {'timestamp':1.,'publish_timestamp':.9}])
def test_invalid_evidence_timestamps_rejected(frame):
    with pytest.raises(ValueError,match='timestamp'):replay_timing(frame)
