import pytest

from qdrant_client.conversions import common_types as types
from scripts.evaluate.borrow_intervals import get_matched_segments, IntervalsConfig
from avm.fingerprint.audio import Segment


def test_get_matched_segments_empty():

    intervals_config = IntervalsConfig(
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
    matched_intervals = get_matched_segments(intervals_config, "", query_hits_intervals)
    
    exoected_matched_intervals = []
    
    assert matched_intervals == exoected_matched_intervals


def test_get_matched_segments_full():

    intervals_config = IntervalsConfig(
        threshold=0.95,
        query_interval_step=1.0,
        index_interval_step=1.0,
        interval_duration_in_seconds=5,
    )
    target_file_id = "test_target"
    query_hits_intervals = [
        [ types.ScoredPoint(id=0, version=0, score=0.99, payload={"file_id":target_file_id, "interval_num":0}) ],
        [ types.ScoredPoint(id=1, version=0, score=0.99, payload={"file_id":target_file_id, "interval_num":1}) ],
        [ types.ScoredPoint(id=2, version=0, score=0.99, payload={"file_id":target_file_id, "interval_num":2}) ],
    ]

    query_file_id = "test_query"
    matched_intervals = get_matched_segments(intervals_config, query_file_id, query_hits_intervals)
    
    end_second = intervals_config.interval_duration_in_seconds + intervals_config.query_interval_step * 2
    exoected_matched_intervals = [[
        Segment(file_id=query_file_id, start_second=0, end_second=end_second),
        Segment(file_id=target_file_id, start_second=0, end_second=end_second),
    ]]
    
    assert matched_intervals == exoected_matched_intervals

