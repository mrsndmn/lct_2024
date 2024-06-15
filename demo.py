import streamlit as st
import os
from src.avm.matcher import MatchedSegmentsPair, Segment
from typing import List
from dataclasses import dataclass
import time

uploaded_file = st.file_uploader("Select video")

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


def render_file_match(mbf: MatchesByFile):
    container = st.container(border=True)
    container.write(f"Matches for: {mbf.file_id}")
    
    
    with open(mbf.file_id, 'rb') as f:
        match_bytes = f.read()

    selected = container.selectbox(
    label="Matches:",
    options=mbf.matches,
    format_func=lambda o: f"source({o.current_segment.start_second}-{o.current_segment.end_second}) match({o.licensed_segment.start_second}-{o.licensed_segment.end_second})"
    )

    col1, col2 = container.columns(2)
    col1.write("source")
    col1.video(uploaded_file, start_time=selected.current_segment.start_second, end_time=selected.current_segment.end_second)
    
    col2.write("match")
    col2.video(match_bytes, start_time=selected.licensed_segment.start_second, end_time=selected.licensed_segment.end_second)




if uploaded_file is not None:
    matches = processing_mock("demo/si2m5i2ne4b8oih1mjcyo2lg62ujh3si.mp4", "demo/ffsfsdfsdf.mp4")

    matches_by_file = {}

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
