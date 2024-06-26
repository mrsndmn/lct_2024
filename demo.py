import streamlit as st
import os
from src.avm.matcher import MatchedSegmentsPair, Segment
from typing import List
from dataclasses import dataclass
import time
import shutil

from avm.matcher import AVMatcherConfig, AVMatcher, MatchedSegmentsPair, Segment, get_matched_segments, _merge_intersectioned_segments
from avm.search.index import EmbeddingIndexFolder
from avm.fingerprint.audio import AudioFingerPrinterConfig, AudioFingerPrinter
from avm.models.audio import get_default_audio_model
from avm.models.image import get_default_image_model_for_x_vector
from avm.models.image import get_default_image_model_for_x_vector
from avm.fingerprint.video import VideoFingerPrinter, VideoFingerPrinterConfig

from tempfile import NamedTemporaryFile

import torch

from qdrant_client import QdrantClient

uploaded_file = st.file_uploader("Select video")
# uploaded_file = './data/rutube/videos/pytest_videos/ydcrodwtz3mstjq1vhbdflx6kyhj3y0p.mp4'

MOCKED = False
base_videos_path = 'data/rutube/videos/compressed_normalized_index'


@dataclass
class MatchesByFile:
    file_id: str
    matches: List[MatchedSegmentsPair]  

@st.cache_data
def processing_mock(legal_file_id, pirate_file_id) -> List[MatchedSegmentsPair]:
    legal_interval1 = Segment(
        file_id=legal_file_id,
        start_second=1,
        end_second=10,
    )
    legal_interval2 = Segment(
        file_id=legal_file_id,
        start_second=100,
        end_second=150,
    )

    pirate_interval1 = Segment(
        file_id=pirate_file_id,
        start_second=1,
        end_second=10,
    )
    pirate_interval2 = Segment(
        file_id=pirate_file_id,
        start_second=100,
        end_second=150,
    )

    time.sleep(3)

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

@st.cache_resource
def get_matcher():
    matcher_config = AVMatcherConfig(
        query_interval_step=3.0,
        enable_video_matching=True,
    )

    qdrant_client = QdrantClient(host="qdrant", port=6333)
    # qdrant_client = QdrantClient(host="localhost", port=6333)
    print("qdrant_client", qdrant_client)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Visual modality
    video_index = EmbeddingIndexFolder(
        index_embeddings_dir='data/rutube/embeddings/video/index/',
        collection_name="video_index",
        # index_embeddings_files=[ 'ded3d179001b3f679a0101be95405d2c.pt' ],
        qdrant_client=qdrant_client,
    )
    visual_model = get_default_image_model_for_x_vector(
        from_pretrained="data/models/image/efficient-net-b0/ruby-moon-17",
    )
    visual_model.to(device)
    visual_model.eval()

    vfp_config = VideoFingerPrinterConfig()
    video_fingerprinter = VideoFingerPrinter(vfp_config, visual_model)

    audio_index = EmbeddingIndexFolder(
        index_embeddings_dir='data/rutube/embeddings/electric-yogurt-97/audio_index_embeddings/',
        collection_name="audio_index",
        # index_embeddings_files=[ 'ded3d179001b3f679a0101be95405d2c.pt' ],
        qdrant_client=qdrant_client,
    )
    
    audio_validation_figerprinter_config = AudioFingerPrinterConfig(
        interval_step=matcher_config.query_interval_step,
        batch_size=10,
    )

    model, feature_extractor = get_default_audio_model()
    model.eval()
    model.to(device)

    audio_validation_fingerprinter = AudioFingerPrinter(
        audio_validation_figerprinter_config,
        model=model,
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

@st.cache_resource
def find_matches(file_path):
    avmatcher = get_matcher()
    matches = avmatcher.find_matches(file_path, cleanup=False)
    return matches


def render_file_match(mbf: MatchesByFile):
    container = st.container(border=True)
    container.write(f"Matches for: {mbf.file_id}")

    matched_video_file_path = os.path.join(base_videos_path, mbf.file_id + ".mp4")

    with open(matched_video_file_path, 'rb') as f:
        match_bytes = f.read()

    selected = container.selectbox(
        label="Matches:",
        options=mbf.matches,
        format_func=lambda o: f"source({o.current_segment.format_string()}) match({o.licensed_segment.format_string()})"
    )

    col1, col2 = container.columns(2)
    col1.write("source")
    col1.video(uploaded_file, start_time=selected.current_segment.start_second, end_time=selected.current_segment.end_second)
    
    col2.write("match")
    col2.video(match_bytes, start_time=selected.licensed_segment.start_second, end_time=selected.licensed_segment.end_second)


if uploaded_file is not None:
    file_path = f"/tmp/{uploaded_file.file_id}.mp4"

    if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

    matches = find_matches(file_path)

    matches_by_file = {}

    print("matches", matches)

    if len(matches) == 0:
        st.write(f"Совпадений не найдено")

    for match in matches:
        if match.licensed_segment.file_id in matches_by_file:
            matches_by_file[match.licensed_segment.file_id].matches.append(match)
        else:
            matches_by_file[match.licensed_segment.file_id] = MatchesByFile(
                file_id= match.licensed_segment.file_id,
                matches=[match]
            )

    for file_id in matches_by_file:
        render_file_match(matches_by_file[file_id])

    if st.button("Загрузить видео в индекс"):
        file_id = uploaded_file.name.removesuffix(".mp4")
        st.write(f"Ожидайте, это может занять какое-то время. file_id={file_id}")
        get_matcher().add_to_index(file_path, file_id=file_id, cleanup=False)
        st.write(f"Готово! Видео загружено в индекс! file_id={file_id}")

        index_videos_file_path = os.path.join(base_videos_path, uploaded_file.name)
        print("copy file to indexed videos dir", index_videos_file_path)
        shutil.copyfile(file_path, index_videos_file_path)
