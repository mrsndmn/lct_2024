from typing import List
from dataclasses import dataclass, field

@dataclass
class Segment:
    file_id: str
    start_second: float
    end_second: float

    def format_duration(self):
        return f"{int(self.start_second)}-{int(self.end_second)}"

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

    hits1 = [SearchHit(score=0.99, segment=legal_interval1), SearchHit(score=0.59, segment=legal_interval2)]
    hits2 = [SearchHit(score=0.9, segment=legal_interval2), SearchHit(score=0.59, segment=legal_interval1)]

    return [
        MatchedInterval(segment=pirate_interval1, hits=hits1),
        MatchedInterval(segment=pirate_interval2, hits=hits2),
    ]
