from typing import List
from dataclasses import dataclass, field
import torch
from copy import deepcopy
from avm.search.audio import AudioIndex
# from qdrant_client.conversions import common_types as types
from avm.fingerprint.audio import Segment
# [
#   (Segment(cur_file), Segment(target_file) )
# ]

@dataclass
class IntervalsConfig:
    interval_duration_in_seconds:float = field(default=5.0)
    query_interval_step:float = field(default=1.0)
    index_interval_step:float = field(default=1.0)
    sampling_rate:int = field(default=16000)

    merge_segments_with_diff_seconds:int = field(default=3)
    segment_min_duration:int = field(default=7)
    threshold: float = field(default=0.9)


def parse_segment(segment):
    start, end = map(int, segment.split("-"))
    return start, end


def iou(segment_q: Segment, segment_t: Segment):
    start_q, stop_q = segment_q.start_second, segment_q.end_second
    start_t, stop_t = segment_t.start_second, segment_t.end_second
    
    intersection_start = max(start_q, start_t)
    intersection_end = min(stop_q, stop_t)

    intersection_length = max(0, intersection_end - intersection_start)
    union_length = (stop_q - start_q) + (stop_t - start_t) - intersection_length

    iou = intersection_length / union_length if union_length > 0 else 0
    return iou


def f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    print(f'Precision = {precision}')
    print(f'Recall = {recall}')
    
    return 2 * (precision * recall) / (precision + recall + 1e-6)

def final_metric(tp, fp, fn, final_iou):
    f = f1(tp, fp, fn)
    
    print(f'IOU = {final_iou}')
    
    return 2 * (final_iou * f) / (final_iou + f + 1e-6)

def evaluate_iou(query_matched_segments: List[List[Segment]], target_segment: List[Segment]):
    
    max_iou = 0.0
    for segment_q in query_matched_segments:
        current_iou = iou(segment_q=segment_q[0], segment_t=target_segment[0]) * iou(segment_q=segment_q[1], segment_t=target_segment[1])
        max_iou = max(max_iou, current_iou)

    return max_iou

def get_matched_segments(config: IntervalsConfig, query_file_id, query_hits_intervals: List[List[Segment]]):
    # todo в теории нужный интервал может быть не самым ближайшим соседом
    first_only_hits = [ h[0] for h in query_hits_intervals ]

    result_segments: List[List[Segment]] = []

    for i, query_hit in enumerate(first_only_hits):
        if query_hit.score < config.threshold:
            continue

        query_start_segment = config.query_interval_step * i
        query_segment = Segment(
            file_id=query_file_id,
            start_second=query_start_segment,
            end_second=(query_start_segment + config.interval_duration_in_seconds),
        )

        hit_file_id = query_hit.payload['file_id']
        hit_interval = query_hit.payload['interval_num']
        hit_start_segment = hit_interval * config.index_interval_step

        hit_end_segment = hit_start_segment + config.interval_duration_in_seconds
        
        hit_segment = Segment(
            file_id=hit_file_id,
            start_second=hit_start_segment,
            end_second=hit_end_segment,
        )

        result_segments.append([query_segment, hit_segment])

    if len(result_segments) == 0:
        return []

    # если сегменты одного видео близко друг к другу, то можно их смерджить
    current_segment: List[Segment] = deepcopy(result_segments[0])
    merged_segments = [ current_segment ]
    for next_segment in result_segments[1:]:
        next_segment: List[Segment]
        if next_segment[0].start_second - current_segment[0].end_second < config.merge_segments_with_diff_seconds:
            if next_segment[1].file_id == current_segment[1].file_id:
                if next_segment[1].start_second - current_segment[0].end_second < config.merge_segments_with_diff_seconds:
                    current_segment[0].end_second = next_segment[0].end_second
                    current_segment[1].end_second = next_segment[1].end_second
                    # print("merged", merged_segments)
                    continue

        merged_segments.append(next_segment)
        # print("appended", merged_segments)
        current_segment = next_segment

    # если нашли одинокий интервал, то его можно отфильтровать, тк минимум 10 секунд должно матчиться
    filtered_segments = []
    for segment in merged_segments:
        if segment[0].end_second - segment[0].start_second < config.segment_min_duration:
            continue
        filtered_segments.append(segment)

    return filtered_segments

if __name__ == '__main__':

    from tqdm import tqdm
    import os
    import pickle
    import pandas as pd

    audio_index = AudioIndex(
        index_embeddings_dir='data/rutube/embeddings/electric-yogurt-97/audio_index_embeddings/',
        # index_embeddings_files=[ 'ded3d179001b3f679a0101be95405d2c.pt' ],
    )

    query_embeddings_dir = 'data/rutube/embeddings/electric-yogurt-97/audio_val_embeddings/'
    query_embeddings_files = sorted(os.listdir(query_embeddings_dir))
    
    matched_intervals_for_queries = []

    for query_embeddings_file in tqdm(query_embeddings_files[:2]):
        query_embeddings_file: str
        file_id = query_embeddings_file.removesuffix(".pt")
        query_embeddings = torch.load(os.path.join(query_embeddings_dir, query_embeddings_file))

        query_hits_intervals = audio_index.search_sequential(query_embeddings.numpy(), limit_per_vector=1)

        intervals_config = IntervalsConfig(
            threshold=0.95,
            index_interval_step=1.0,
            query_interval_step=0.2,
            interval_duration_in_seconds=5,
        )
        matched_intervals = get_matched_segments(intervals_config, file_id, query_hits_intervals)
        matched_intervals_for_queries.append(matched_intervals)
        print('len(matched_intervals)', len(matched_intervals))
    
    with open("matched_intervals_for_queries.pickle", 'wb') as f:
        pickle.dump(matched_intervals, f)
    
    validate_file_items = []
    for mi in matched_intervals:
        piracy_interval: Segment = mi[0]
        license_interval: Segment = mi[1]

        validate_file_item = {
            "ID-piracy": piracy_interval.file_id,
            "SEG-piracy": piracy_interval.format_duration(),
            "ID-license": license_interval.file_id,
            "SEG-license": license_interval.format_duration(),
        }
        validate_file_items.append(validate_file_item)

    df = pd.DataFrame(validate_file_items)
    df.to_csv("matched_intervals_for_queries.csv")

    raise Exception

