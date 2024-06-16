import pytest
import torch
import pickle
from qdrant_client.conversions import common_types as types
from typing import List

from avm.matcher import AVMatcherConfig, AVMatcher, MatchedSegmentsPair, Segment, get_matched_segments, _merge_intersectioned_segments
from avm.search.index import EmbeddingIndexFolder
from avm.fingerprint.audio import AudioFingerPrinterConfig, AudioFingerPrinter
from avm.models.audio import get_default_audio_model

from avm.models.image import get_default_image_model_for_x_vector
from avm.fingerprint.video import VideoFingerPrinter, VideoFingerPrinterConfig


def _prepare_matcher():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    matcher_config = AVMatcherConfig(
        query_interval_step=1.0,
    )

    # Visual modality
    video_index = EmbeddingIndexFolder(
        index_embeddings_dir='data/rutube/embeddings/video/index/',
        collection_name="video_index",
        index_embeddings_files=[ 'ded3d179001b3f679a0101be95405d2c.pt' ],
    )
    visual_model = get_default_image_model_for_x_vector(
        from_pretrained="data/models/image/efficient-net-b0/ruby-moon-17",
    )
    visual_model.to(device)
    visual_model.eval()

    vfp_config = VideoFingerPrinterConfig()
    video_fingerprinter = VideoFingerPrinter(vfp_config, visual_model)

    # Audio modality
    audio_index = EmbeddingIndexFolder(
        index_embeddings_dir='data/rutube/embeddings/electric-yogurt-97/audio_index_embeddings/',
        index_embeddings_files=[ 'ded3d179001b3f679a0101be95405d2c.pt' ],
    )
    
    audio_validation_figerprinter_config = AudioFingerPrinterConfig(
        interval_step=matcher_config.query_interval_step,
        batch_size=10,
    )

    audio_model, feature_extractor = get_default_audio_model()
    audio_model.eval()
    audio_model.to(device)

    audio_validation_fingerprinter = AudioFingerPrinter(
        audio_validation_figerprinter_config,
        model=audio_model,
        feature_extractor=feature_extractor,
    )

    avmatcher = AVMatcher(
        matcher_config,
        audio_index=audio_index,
        audio_fingerprinter=audio_validation_fingerprinter,
        video_index=video_index,
        video_fingerprinter=video_fingerprinter,
    )

    return avmatcher


def test_matcher_end_to_end():

    avmatcher = _prepare_matcher()

    test_video_path = 'data/rutube/videos/pytest_videos/ydcrodwtz3mstjq1vhbdflx6kyhj3y0p.mp4'
    result_matches = avmatcher.find_matches(test_video_path, cleanup=False)

    assert len(result_matches) == 1
    assert result_matches[0].current_segment.duration() > 140
    assert result_matches[0].licensed_segment.file_id == 'ded3d179001b3f679a0101be95405d2c'

    return


def test_get_matched_segments_empty():

    intervals_config = AVMatcherConfig(
        threshold=0.95,
        query_interval_step=1.0,
        index_interval_step=1.0,
        interval_duration_in_seconds=5,
    )
    query_hits_intervals = [
        [ types.ScoredPoint(id=0, version=0, score=0.8, payload={"file_id":"", "interval_num":0}) ],
        [ types.ScoredPoint(id=1, version=0, score=0.0, payload={"file_id":"", "interval_num":1}) ],
        [ types.ScoredPoint(id=2, version=0, score=0.0, payload={"file_id":"", "interval_num":2}) ],
    ]
    matched_segments = get_matched_segments(
        intervals_config, "", query_hits_intervals,
        debug=True,
    )
    
    expected_matched_segments = []
    
    assert matched_segments == expected_matched_segments


def test_get_matched_segments_full():

    intervals_config = AVMatcherConfig(
        threshold=0.95,
        query_interval_step=1.0,
        index_interval_step=1.0,
        interval_duration_in_seconds=5,
        segment_min_duration=0.0,
    )
    target_file_id = "test_target"
    query_hits_intervals = [
        [ types.ScoredPoint(id=0, version=0, score=0.99, payload={"file_id":target_file_id, "interval_num":0}) ],
        [ types.ScoredPoint(id=1, version=0, score=0.99, payload={"file_id":target_file_id, "interval_num":1}) ],
        [ types.ScoredPoint(id=2, version=0, score=0.99, payload={"file_id":target_file_id, "interval_num":2}) ],
    ]

    query_file_id = "test_query"
    matched_segments = get_matched_segments(
        intervals_config, query_file_id, query_hits_intervals,
        debug=True,
    )
    
    end_second = intervals_config.interval_duration_in_seconds + intervals_config.query_interval_step * 2
    expected_matched_segments = [
        MatchedSegmentsPair(
            current_segment=Segment(file_id=query_file_id, start_second=0, end_second=end_second),
            licensed_segment=Segment(file_id=target_file_id, start_second=0, end_second=end_second),
        )
    ]
    
    assert matched_segments == expected_matched_segments


def test_get_matched_segments_full_with_impostor():

    intervals_config = AVMatcherConfig(
        threshold=0.95,
        query_interval_step=1.0,
        index_interval_step=1.0,
        interval_duration_in_seconds=5,
        segment_min_duration=0.0,
    )
    target_file_id = "test_target"
    query_hits_intervals = [
        [ types.ScoredPoint(id=0, version=0, score=0.99, payload={"file_id":target_file_id, "interval_num":0}) ],
        [ types.ScoredPoint(id=1, version=0, score=0.99, payload={"file_id":target_file_id, "interval_num":1}) ],
        [ types.ScoredPoint(id=100, version=0, score=0.99, payload={"file_id":"bad_file_id", "interval_num":10}) ],
        [ types.ScoredPoint(id=2, version=0, score=0.99, payload={"file_id":target_file_id, "interval_num":3}) ],
    ]

    query_file_id = "test_query"
    matched_segments = get_matched_segments(
        intervals_config, query_file_id, query_hits_intervals,
        debug=True
    )
    
    end_second = intervals_config.interval_duration_in_seconds + intervals_config.query_interval_step * 3
    expected_matched_segments = [
        MatchedSegmentsPair(
            current_segment=Segment(file_id=query_file_id, start_second=0, end_second=end_second),
            licensed_segment=Segment(file_id=target_file_id, start_second=0, end_second=end_second),
        )
    ]
    
    assert matched_segments == expected_matched_segments


def test_get_matches_035vv3223ajwox00vxp5saz7uuj20j8h():
    with open('./data/rutube/embeddings/electric-yogurt-97/search_val_embeddings/035vv3223ajwox00vxp5saz7uuj20j8h.pickle', 'rb') as f:
        query_hits_intervals = pickle.load(f)

    query_file_id = "035vv3223ajwox00vxp5saz7uuj20j8h"

    intervals_config = AVMatcherConfig(
        threshold=0.9,
        query_interval_step=1.0,
        index_interval_step=1.0,
        interval_duration_in_seconds=5,
        segment_min_duration=20.0,
    )
    matched_segments = get_matched_segments(intervals_config, query_file_id, query_hits_intervals)

    assert len(matched_segments) == 1, 'matched segments count is ok'
    assert matched_segments[0].licensed_segment.file_id == 'cc0904d3de995d4851de65b93860d8d5', 'file id match is valid'


def test_get_matches_0pxzpgx5qpvd3o5pyve6soihftjfacfy():
    with open('./data/rutube/embeddings/electric-yogurt-97/search_val_embeddings/0pxzpgx5qpvd3o5pyve6soihftjfacfy.pickle', 'rb') as f:
        query_hits_intervals = pickle.load(f)

    query_file_id = "0pxzpgx5qpvd3o5pyve6soihftjfacfy"

    intervals_config = AVMatcherConfig(
        threshold=0.9,
        query_interval_step=1.0,
        index_interval_step=1.0,
        interval_duration_in_seconds=5.0,
        merge_segments_with_diff_seconds=10.0,
        segment_min_duration=20.0,
    )
    matched_segments = get_matched_segments(
        intervals_config, query_file_id, query_hits_intervals,
        debug=True,
        expected_match_debug='f61df753e116c075c02a5cbff04d9d75'
    )
    print("matched_segments", matched_segments)

    assert len(matched_segments) == 1, 'matched segments count is ok'
    assert matched_segments[0].current_segment.duration() > 180

    # raise Exception


def test_get_matched_segments_full_from_0pxzpgx5qpvd3o5pyve6soihftjfacfy_1():
    segments_to_merge = [
        MatchedSegmentsPair(
            current_segment=Segment(file_id='0pxzpgx5qpvd3o5pyve6soihftjfacfy', start_second=65.0, end_second=70.0),
            licensed_segment=Segment(file_id='f61df753e116c075c02a5cbff04d9d75', start_second=934.0, end_second=939.0)
        ),
        MatchedSegmentsPair(
            current_segment=Segment(file_id='0pxzpgx5qpvd3o5pyve6soihftjfacfy', start_second=66.0, end_second=71.0),
            licensed_segment=Segment(file_id='f61df753e116c075c02a5cbff04d9d75', start_second=936.0, end_second=941.0)
        ),
        MatchedSegmentsPair(
            current_segment=Segment(file_id='0pxzpgx5qpvd3o5pyve6soihftjfacfy', start_second=67.0, end_second=72.0),
            licensed_segment=Segment(file_id='f61df753e116c075c02a5cbff04d9d75', start_second=936.0, end_second=941.0)
        ),
        MatchedSegmentsPair(
            current_segment=Segment(file_id='0pxzpgx5qpvd3o5pyve6soihftjfacfy', start_second=68.0, end_second=73.0),
            licensed_segment=Segment(file_id='f61df753e116c075c02a5cbff04d9d75', start_second=937.0, end_second=942.0)
        )
    ]
    intervals_config = AVMatcherConfig(
        threshold=0.9,
        query_interval_step=1.0,
        index_interval_step=1.0,
        interval_duration_in_seconds=5.0,
        merge_segments_with_diff_seconds=10.0,
        segment_min_duration=20.0,
    )

    merged_segment = _merge_intersectioned_segments(
        intervals_config, segments_to_merge,
        debug=True
    )

    assert len(merged_segment) == 1


def test_get_matched_segments_full_from_0pxzpgx5qpvd3o5pyve6soihftjfacfy_2():
    segments_to_merge = [
        MatchedSegmentsPair(
            current_segment=Segment(file_id='0pxzpgx5qpvd3o5pyve6soihftjfacfy', start_second=65.0, end_second=74.0),
            licensed_segment=Segment(file_id='f61df753e116c075c02a5cbff04d9d75', start_second=934.0, end_second=943.0),
        ), 
        MatchedSegmentsPair(
            current_segment=Segment(file_id='0pxzpgx5qpvd3o5pyve6soihftjfacfy', start_second=70.0, end_second=79.0),
            licensed_segment=Segment(file_id='f61df753e116c075c02a5cbff04d9d75', start_second=939.0, end_second=948.0)
        ),
        MatchedSegmentsPair(
            current_segment=Segment(file_id='0pxzpgx5qpvd3o5pyve6soihftjfacfy', start_second=75.0, end_second=84.0),
            licensed_segment=Segment(file_id='f61df753e116c075c02a5cbff04d9d75', start_second=944.0, end_second=953.0)
        ),
        MatchedSegmentsPair(
            current_segment=Segment(file_id='0pxzpgx5qpvd3o5pyve6soihftjfacfy', start_second=80.0, end_second=89.0),
            licensed_segment=Segment(file_id='f61df753e116c075c02a5cbff04d9d75', start_second=949.0, end_second=958.0)
        ),
        MatchedSegmentsPair(
            current_segment=Segment(file_id='0pxzpgx5qpvd3o5pyve6soihftjfacfy', start_second=85.0, end_second=94.0),
            licensed_segment=Segment(file_id='f61df753e116c075c02a5cbff04d9d75', start_second=954.0, end_second=964.0)
        ),
    ]
    intervals_config = AVMatcherConfig(
        threshold=0.9,
        query_interval_step=1.0,
        index_interval_step=1.0,
        interval_duration_in_seconds=5.0,
        merge_segments_with_diff_seconds=10.0,
        segment_min_duration=20.0,
    )

    merged_segment = _merge_intersectioned_segments(
        intervals_config, segments_to_merge,
        debug=True
    )

    assert len(merged_segment) == 1


# todo test for 50vccj3b4afnbpobofwy6j4ool43snud


def test_trim_intervals_with_visual_modality():

    avmatcher = _prepare_matcher()

    video_embeddings = torch.load("data/rutube/embeddings/video/index/ded3d179001b3f679a0101be95405d2c.pt")

    # video_embeddings = video_embeddings[10:20]
    # print("video_embeddings.shape", video_embeddings.shape)

    matched_intervals: List[MatchedSegmentsPair] = [
        MatchedSegmentsPair(
            current_segment=Segment(file_id='ded3d179001b3f679a0101be95405d2c', start_second=10.0, end_second=20.0),
            licensed_segment=Segment(file_id='ded3d179001b3f679a0101be95405d2c', start_second=11.0, end_second=21.0)
        ),
    ]

    expected_matched_intervals: List[MatchedSegmentsPair] = [
        MatchedSegmentsPair(
            current_segment=Segment(file_id='ded3d179001b3f679a0101be95405d2c', start_second=10.0, end_second=20.0),
            licensed_segment=Segment(file_id='ded3d179001b3f679a0101be95405d2c', start_second=10.0, end_second=20.0)
        ),
    ]

    trimmed_intervals = avmatcher.trim_intervals_with_visual_modality(
        matched_intervals,
        video_embeddings
    )

    assert trimmed_intervals[0].current_segment == expected_matched_intervals[0].current_segment
    assert abs(trimmed_intervals[0].licensed_segment.start_second - expected_matched_intervals[0].licensed_segment.start_second) < 1e-4
    assert abs(trimmed_intervals[0].licensed_segment.end_second - expected_matched_intervals[0].licensed_segment.end_second) < 1e-4

    return