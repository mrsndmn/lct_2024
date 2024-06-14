import torch
from dataclasses import dataclass, field

from avm.search.audio import AudioIndex

@dataclass
class AVMatcherConfig:
    pass

class AVMatcher():

    def __init__(self, config, audio_index: AudioIndex, video_index):

        self.config = config
        self.audio_index: AudioIndex = audio_index
        self.video_index = video_index

    def find_matches(self, video_file_id):
        
        audio_file: str = self.extract_audio_from_video_file(video_file_id)

        audio_matched_intervals = self.find_audio_only_matches(audio_file)

    def find_audio_only_matches(self, audio_file):
        
        query_audio_embeddings: torch.Tensor = generate_audio_embeddings(audio_file)
        
        query_hits = self.audio_index.search_sequential(query_audio_embeddings)

        matched_segments = self.get_matched_audio_segments(query_hits)

        return matched_segments

    def extract_audio_from_video_file(self, video_file) -> str:
        """
        Возвращает абсолютный путь к нормализованному .wav аудио файлу
        """

        return extract_audio_from_video_file(video_file)

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

