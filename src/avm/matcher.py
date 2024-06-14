import os
import torch
from dataclasses import dataclass, field
from copy import deepcopy

from typing import List

from avm.search.audio import AudioIndex
from avm.fingerprint.audio import AudioFingerPrinter

@dataclass
class Segment:
    # Сегмент для одного аудио/видео -- непрерывный участок
    file_id: str
    start_second: float
    end_second: float

    def format_duration(self):
        return f"{int(self.start_second)}-{int(self.end_second)}"


@dataclass
class MatchedSegmentsPair:
    """
    Пара похожих, сматченных сегментов
    """
    current_segment: Segment  # сегмент текущего сматченнго видео/аудио
    licensed_segment: Segment # сегмент лицензионного сматченного видео


@dataclass
class AVMatcherConfig:
    extracted_audios_dir: str = field(default="/tmp/avmatcher/extracted_audios")

    sampling_rate:int = field(default=16000)

    interval_duration_in_seconds:float = field(default=5.0)
    query_interval_step:float = field(default=0.2)
    index_interval_step:float = field(default=1.0)

    # Мерджим сегменты, если у них разница во времени меньше X секунд
    merge_segments_with_diff_seconds:int = field(default=3)
    # Удаляем сегменты, которые длятся менее X секунд
    segment_min_duration:int = field(default=10)
    # Удаляем сматченные эмбэддинги, которые отличаются больше, чем на
    # заданнный трешолд
    threshold: float = field(default=0.95)


class AVMatcher():

    def __init__(self,
                 config: AVMatcherConfig,
                 audio_index: AudioIndex,
                 audio_fingerprinter: AudioFingerPrinter, 
                ):

        self.config = config
        self.audio_index = audio_index
        self.audio_fingerprinter = audio_fingerprinter

        # self.video_index = video_index

        os.makedirs(self.config.extracted_audios_dir, exist_ok=True)

    def find_matches(self, video_full_path: str):

        assert video_full_path.endswith(".mp4")
        
        file_id = os.path.basename(video_full_path).removesuffix(".mp4")
        
        audio_file: str = self.extract_audio_from_video_file(video_full_path, file_id=file_id)

        audio_matched_intervals = self.find_audio_only_matches(audio_file)

        return audio_matched_intervals

    def find_audio_only_matches(self, audio_file, file_id):
        
        query_audio_embeddings: torch.Tensor = self.audio_fingerprinter.fingerprint_from_file(audio_file, file_id)
        
        query_hits = self.audio_index.search_sequential(query_audio_embeddings)

        matched_segments = get_matched_segments(self.config, query_hits)

        return matched_segments

    def extract_audio_from_video_file(self, video_full_path, file_id) -> str:
        """
        Возвращает абсолютный путь к нормализованному .wav аудио файлу
        """
        audio_file_name = file_id + ".wav"
        audio_full_path = os.path.join(self.config.extracted_audios_dir, audio_file_name)

        exit_code = os.system(f"ffmpeg -i {video_full_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_full_path}")
        if exit_code != 0:
            raise Exception(f"can't extract audio: {exit_code}")

        return audio_full_path


def _merge_intersectioned_segments(config, input_segments: List[MatchedSegmentsPair]) -> List[MatchedSegmentsPair]:
    active_segments: MatchedSegmentsPair = deepcopy(input_segments[0])
    merged_segments = [ active_segments ]
    for next_segments in input_segments[1:]:

        if next_segments.current_segment.start_second - active_segments.current_segment.end_second < config.merge_segments_with_diff_seconds:
            if next_segments.licensed_segment.file_id == active_segments.licensed_segment.file_id:
                if next_segments.licensed_segment.start_second - active_segments.current_segment.end_second < config.merge_segments_with_diff_seconds:
                    # Объединяем пересекающиеся сегменты
                    active_segments.current_segment.end_second = next_segments.current_segment.end_second
                    active_segments.licensed_segment.end_second = next_segments.licensed_segment.end_second
                    continue

        merged_segments.append(next_segments)
        active_segments = next_segments

    return merged_segments

def get_matched_segments(config, query_file_id, query_hits_intervals) -> List[MatchedSegmentsPair]:
    # todo в теории нужный интервал может быть не самым ближайшим соседом
    first_only_hits = [ h[0] for h in query_hits_intervals ]

    high_score_matched_segments: List[MatchedSegmentsPair] = []

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

        matched_segments = MatchedSegmentsPair(query_segment, hit_segment)
        high_score_matched_segments.append(matched_segments)

    if len(high_score_matched_segments) == 0:
        return []

    # если сегменты одного видео близко друг к другу, то можно их смерджить
    merged_matched_segments = _merge_intersectioned_segments(config, high_score_matched_segments)

    # если нашли одинокий интервал, то его можно отфильтровать, тк минимум 10 секунд должно матчиться
    last_position_by_file_id = dict()
    filtered_segments: List[MatchedSegmentsPair] = []
    for segments_pair in merged_matched_segments:
        matched_file_id = segments_pair.licensed_segment.file_id
        last_end_second = last_position_by_file_id.get(matched_file_id, None)
        if last_end_second is None:
            last_position_by_file_id[matched_file_id] = segments_pair.current_segment.end_second
        else:
            if segments_pair.licensed_segment.end_second - last_end_second < config.interval_duration_in_seconds:
                while len(filtered_segments) > 0 and filtered_segments[-1].licensed_segment.file_id != matched_file_id:
                    filtered_segments.pop(-1)

        filtered_segments.append(segments_pair)

    if len(filtered_segments) == 0:
        return []

    # после удаления одиночных сегментов могли образоваться
    # новые возможности для мерджа - см тест test_get_matched_segments_full_with_impostor
    merged_again_segments = _merge_intersectioned_segments(config, filtered_segments)

    filtered_segments = []
    for segments_pair in merged_again_segments:
        if segments_pair.current_segment.end_second - segments_pair.current_segment.start_second < config.segment_min_duration:
            continue
        filtered_segments.append(segments_pair)

    return filtered_segments


def dummy_get_matched_intervals():

    legal_interval1 = Segment(
        file_id="legal1",
        start_second=1,
        end_second=10,
    )
    legal_interval2 = Segment(
        file_id="legal2",
        start_second=22000,
        end_second=33000,
    )

    pirate_interval1 = Segment(
        file_id="ugc",
        start_sample=12300,
        end_sample=45600,
    )
    pirate_interval2 = Segment(
        file_id="ugc",
        start_sample=22000,
        end_sample=33000,
    )

    return [
        MatchedSegmentsPair(
            current_segment=pirate_interval1,
            licensed_segment=legal_interval1,
        ),
        MatchedSegmentsPair(
            current_segment=pirate_interval2,
            licensed_segment=legal_interval2,
        ),
    ]
