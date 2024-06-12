from typing import List
from dataclasses import dataclass, field

@dataclass
class SearchHit:
    score: float
    file_id: str

class MatchedIntervals:
    start_sample: int
    end_sample: int

    hits: List[SearchHit]


def dummy_get_matched_intervals():
    return 
