import os
import torch
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np

from typing import List

from avm.search.index import EmbeddingIndexFolder
from avm.fingerprint.audio import AudioFingerPrinter

from scripts.normalization.normalize_video_ffmpeg import normalize_video_for_matching

from avm.fingerprint.video import VideoFingerPrinter

from qdrant_client import QdrantClient, models as qdrant_models
from qdrant_client.conversions import common_types as qdrant_types


@dataclass
class Segment:
    # Сегмент для одного аудио/видео -- непрерывный участок
    file_id: str
    start_second: float
    end_second: float

    def format_string(self):
        return f"{int(self.start_second)}-{int(self.end_second)}"
    
    def duration(self):
        return self.end_second - self.start_second

    def is_valid(self):
        assert self.start_second < self.end_second


@dataclass
class MatchedSegmentsPair:
    """
    Пара похожих, сматченных сегментов
    """
    current_segment: Segment  # сегмент текущего сматченнго видео/аудио
    licensed_segment: Segment # сегмент лицензионного сматченного видео


@dataclass
class AVMatcherConfig:
    query_interval_step:float # обязательно надо явно проставлять, тк может разливаться
    index_interval_step:float = field(default=1.0)
    interval_duration_in_seconds:float = field(default=5.0)

    extracted_audios_dir: str = field(default="/tmp/avmatcher/extracted_audios")
    normalized_videos_dir: str = field(default="/tmp/avmatcher/normalized_videos")

    sampling_rate:int = field(default=16000)

    # Мерджим сегменты, если у них разница во времени меньше X секунд
    merge_segments_with_diff_seconds:float = field(default=10.)
    # Удаляем сегменты, которые длятся менее X секунд
    segment_min_duration:int = field(default=20)
    # Удаляем сматченные эмбэддинги, которые отличаются больше, чем на
    # заданнный трешолд
    threshold: float = field(default=0.93)

    enable_video_matching: bool = field(default=False)


class AVMatcher():
    """Audio and Video Matcher"""

    def __init__(self,
                 config: AVMatcherConfig,
                 audio_index: EmbeddingIndexFolder,
                 audio_fingerprinter: AudioFingerPrinter, 
                 video_index: EmbeddingIndexFolder,
                 video_fingerprinter: VideoFingerPrinter,
                ):

        self.config = config

        self.audio_index = audio_index
        self.audio_fingerprinter = audio_fingerprinter

        self.video_index = video_index
        self.video_fingerprinter = video_fingerprinter

        # self.video_index = video_index

        os.makedirs(self.config.extracted_audios_dir, exist_ok=True)

    def find_matches(self, video_full_path: str, cleanup=True, file_id=None) -> List[MatchedSegmentsPair]:

        assert video_full_path.endswith(".mp4")

        if file_id is None:
            file_id = os.path.basename(video_full_path).removesuffix(".mp4")

        # Audio Matching
        print("extracting audio from video")
        audio_file: str = self.extract_audio_from_video_file(video_full_path, file_id=file_id)
        print("audio file extracted", audio_file)

        audio_matched_intervals = self.find_audio_only_matches(audio_file, file_id=file_id)

        result_mathing_intervals = audio_matched_intervals

        # Video Matching for trimming audio intervals
        if self.config.enable_video_matching:
            normalized_video_file = self.normalize_video(video_full_path, file_id=file_id)

            video_fingerprints = self.video_fingerprinter.fingerprint_from_file(normalized_video_file)
            video_fingerprints = video_fingerprints.cpu().numpy()

            result_mathing_intervals = self.trim_intervals_with_visual_modality(
                audio_matched_intervals, video_fingerprints
            )

            if cleanup:
                os.remove(normalized_video_file)

        if cleanup:
            os.remove(audio_file)

        return result_mathing_intervals
    
    # Метод уточняет границы предсказанного интервала
    # с помощью модальности видео. Может смещать и ли расширять границы
    # интервалов.
    # 
    # 1) Сначала выравниваем время интервала для исходного и лицензионного файла
    # 2) Вычисляем исходя из длительности интервала новое окончание
    # 3) Проверяем, можем ли мы увеличить длительность интервала?
    def trim_intervals_with_visual_modality(
            self,
            audio_matched_intervals: List[MatchedSegmentsPair],
            video_fingerprint: np.ndarray,
            trim_seconds=2
            ) -> List[MatchedSegmentsPair]:

        # todo вообщег говоря, для тимминга и валидации,
        # кажется, уже не нужен векторный поиск

        # trimmed_matched_intervals = audio_matched_intervals
        trimmed_matched_intervals: List[MatchedSegmentsPair] = []
        for interval in audio_matched_intervals:
            interval = deepcopy(interval)

            # trim start interval
            license_file_id = interval.licensed_segment.file_id
            license_start_second = int(interval.licensed_segment.start_second)
            license_end_second = interval.licensed_segment.start_second

            start_second = int(interval.current_segment.start_second)
            end_second = int(interval.current_segment.end_second)

            can_trim_start_second = license_start_second > trim_seconds and start_second > trim_seconds
            if can_trim_start_second:
            
                start_frame_embedding = torch.from_numpy(video_fingerprint[start_second:start_second+1])

                must_search_condition = [
                    qdrant_models.FieldCondition(
                        key="file_id",
                        match=qdrant_models.MatchValue(value=license_file_id),
                    ),
                ]
                should_search_condition = [
                    qdrant_models.FieldCondition(
                        key="interval_num",
                        match=qdrant_models.MatchValue(value=license_start_second-2),
                    ),
                    qdrant_models.FieldCondition(
                        key="interval_num",
                        match=qdrant_models.MatchValue(value=license_start_second-1),
                    ),
                    qdrant_models.FieldCondition(
                        key="interval_num",
                        match=qdrant_models.MatchValue(value=license_start_second),
                    ),
                    qdrant_models.FieldCondition(
                        key="interval_num",
                        match=qdrant_models.MatchValue(value=license_start_second+1),
                    ),
                    qdrant_models.FieldCondition(
                        key="interval_num",
                        match=qdrant_models.MatchValue(value=license_start_second+2),
                    ),
                ]

                legal_video_embeddings_hits, _ = self.video_index.scroll(
                    scroll_filter=qdrant_models.Filter(
                        must=must_search_condition,
                        should=should_search_condition
                    )
                )
                
                legal_video_embeddings = [ torch.tensor(h.vector).unsqueeze(0) for h in legal_video_embeddings_hits ]
                legal_video_embeddings = torch.cat(legal_video_embeddings, dim=0) # [ 5, 256 ]

                matches = start_frame_embedding @ legal_video_embeddings.T # [ 1, 5 ]
                most_similar_idx = torch.argsort(matches, descending=True)[0][0]
                most_similar_score = matches[0, most_similar_idx].item()
                if most_similar_score > 0.95:
                    most_similar_idx_offest = most_similar_score - 2
                    print(f"found match idx from videos: most_similar_idx={most_similar_idx} most_similar_score={most_similar_score}")
                    interval.licensed_segment.start_second = interval.licensed_segment.start_second + most_similar_idx_offest
                    interval.licensed_segment.end_second = interval.licensed_segment.end_second + most_similar_idx_offest
            else:
                # start second trimming is not possible
                pass

            # always append interval!
            trimmed_matched_intervals.append(interval)

        return trimmed_matched_intervals
    
    def normalize_video(self, video_full_path, file_id):
        normalized_video_file_name = file_id + ".mp4"
        normalized_video_full_path = os.path.join(self.config.normalized_videos_dir, normalized_video_file_name)

        if os.path.exists(normalized_video_full_path):
            print("file was already preprocessed", normalized_video_full_path)
            return normalized_video_full_path

        return normalize_video_for_matching(video_full_path, normalized_video_full_path)

    def find_audio_only_matches(self, audio_file, file_id) -> List[MatchedSegmentsPair]:

        print("fingerprinting audio")
        query_audio_embeddings: torch.Tensor = self.audio_fingerprinter.fingerprint_from_file(audio_file)
        print("query_audio_embeddings", query_audio_embeddings.shape)
        
        print("audio embeddings search")
        query_hits = self.audio_index.search_sequential(query_audio_embeddings.numpy())

        print("generating matched segments")
        matched_segments = get_matched_segments(self.config, file_id, query_hits)

        return matched_segments

    def extract_audio_from_video_file(self, video_full_path, file_id) -> str:
        """
        Возвращает абсолютный путь к нормализованному .wav аудио файлу
        """
        audio_file_name = file_id + ".wav"
        audio_full_path = os.path.join(self.config.extracted_audios_dir, audio_file_name)

        if os.path.exists(audio_full_path):
            print("file was already preprocessed", audio_full_path)
            return audio_full_path

        exit_code = os.system(f"ffmpeg -y -i {video_full_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_full_path}")
        if exit_code != 0:
            raise Exception(f"can't extract audio: {exit_code}")

        return audio_full_path
    
    def add_to_index(self, video_full_path, cleanup=True, file_id=None):
        
        if file_id is None:
            file_id = os.path.basename(video_full_path).removesuffix(".mp4")

        # todo reuse normalized audio and video if possible
        print("extracting audio from video")
        audio_file: str = self.extract_audio_from_video_file(video_full_path, file_id=file_id)
        print("audio file extracted", audio_file)

        print("generating audio fingerprint")
        audio_fingerprints: torch.Tensor = self.audio_fingerprinter.fingerprint_from_file(audio_file)
        audio_fingerprints = audio_fingerprints.cpu().numpy()
        print("loading audio fingerprint to qdrant")
        self.audio_index.load_embeddings(audio_fingerprints, file_id, upload_points=True)

        if self.config.enable_video_matching:
            normalized_video_file = self.normalize_video(video_full_path, file_id=file_id)

            print("generating video fingerprint")
            video_fingerprints = self.video_fingerprinter.fingerprint_from_file(normalized_video_file)
            video_fingerprints = video_fingerprints.cpu().numpy()

            print("loading video fingerprint to qdrant")
            self.video_index.load_embeddings(video_fingerprints, file_id, upload_points=True)
            
            if cleanup:
                os.remove(normalized_video_file)

        if cleanup:
            os.remove(audio_file)

        return


def _merge_intersectioned_segments(config, input_segments: List[MatchedSegmentsPair], debug=False) -> List[MatchedSegmentsPair]:
    active_segments: MatchedSegmentsPair = deepcopy(input_segments[0])
    merged_segments = [ active_segments ]
    for next_segments in input_segments[1:]:
        # print("cond1 current_segment diff is ok", next_segments.current_segment.start_second - active_segments.current_segment.end_second)
        # print("cond2 file_id match", next_segments.licensed_segment.file_id == active_segments.licensed_segment.file_id)
        # print("cond3 licensed_segment diff is ok", next_segments.licensed_segment.start_second - active_segments.licensed_segment.end_second)
        if next_segments.current_segment.start_second - active_segments.current_segment.end_second < config.merge_segments_with_diff_seconds:
            if next_segments.licensed_segment.file_id == active_segments.licensed_segment.file_id:
                if next_segments.licensed_segment.start_second - active_segments.licensed_segment.end_second < config.merge_segments_with_diff_seconds:
                    # Объединяем пересекающиеся сегменты
                    # active_segments_clone = deepcopy(active_segments)

                    active_segments.current_segment.end_second = next_segments.current_segment.end_second
                    active_segments.licensed_segment.end_second = active_segments.licensed_segment.start_second + active_segments.current_segment.duration()
                    # active_segments.licensed_segment.end_second = max(next_segments.licensed_segment.end_second, active_segments.licensed_segment.end_second)
                    active_segments.current_segment.is_valid()
                    active_segments.licensed_segment.is_valid()

                    # active_segments.current_segment = active_segments_clone.current_segment
                    # active_segments.licensed_segment = active_segments_clone.licensed_segment
                    if debug:
                        print("updated active_segments", active_segments)
                    continue

        # print("appended active_segments", active_segments)
        active_segments = next_segments
        merged_segments.append(active_segments)

    return merged_segments

def get_matched_segments(config, query_file_id, query_hits_intervals, debug=False, expected_match_debug=None) -> List[MatchedSegmentsPair]:
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

    if debug:
        print("high_score_matched_segments", len(high_score_matched_segments))

    if len(high_score_matched_segments) == 0:
        return []

    # если сегменты одного видео близко друг к другу, то можно их смерджить
    merged_matched_segments = _merge_intersectioned_segments(
        config, high_score_matched_segments,
        debug=debug
    )
    if debug:
        print("merged_matched_segments", len(merged_matched_segments))
        if expected_match_debug is not None:
            expected_match_debug_mips = []
            for mip in merged_matched_segments:
                if mip.licensed_segment.file_id == expected_match_debug:
                    expected_match_debug_mips.append(mip)
                else:
                    expected_match_debug_mips.append(None)

            # breakpoint()
            print("expected_match_debug_mips", expected_match_debug_mips)

    # если нашли одинокий интервал, то его можно отфильтровать, тк минимум 10 секунд должно матчиться
    last_position_by_file_id = dict()
    filtered_segments: List[MatchedSegmentsPair] = []
    for segments_pair in merged_matched_segments:
        matched_file_id = segments_pair.licensed_segment.file_id
        last_end_second = last_position_by_file_id.get(matched_file_id, None)
        if last_end_second is None:
            last_position_by_file_id[matched_file_id] = segments_pair.current_segment.end_second
        else:
            if segments_pair.current_segment.end_second - last_end_second < config.interval_duration_in_seconds:
                while len(filtered_segments) > 0 and filtered_segments[-1].licensed_segment.file_id != matched_file_id:
                    filtered_segments.pop(-1)

        filtered_segments.append(segments_pair)

    if debug:
        print("filtered_segments", len(filtered_segments))

    if len(filtered_segments) == 0:
        return []

    # после удаления одиночных сегментов могли образоваться
    # новые возможности для мерджа - см тест test_get_matched_segments_full_with_impostor
    merged_again_segments = _merge_intersectioned_segments(config, filtered_segments)
    if debug:
        print("merged_again_segments", len(merged_again_segments))

    filtered_segments = []
    for segments_pair in merged_again_segments:
        if segments_pair.current_segment.end_second - segments_pair.current_segment.start_second < config.segment_min_duration:
            continue
        filtered_segments.append(segments_pair)

    if len(filtered_segments) == 0:
        return []

    merged_again_again_segments = _merge_intersectioned_segments(config, filtered_segments)
    if debug:
        print("merged_again_again_segments", len(merged_again_again_segments))


    # todo validate segments
    for segments_pair in merged_again_again_segments:
        segments_pair.current_segment.is_valid()
        segments_pair.licensed_segment.is_valid()
    
    if debug:
        print("filtered_segments", len(merged_again_again_segments))

    return merged_again_again_segments


