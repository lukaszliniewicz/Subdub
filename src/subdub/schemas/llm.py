from typing import List, Literal

from pydantic import BaseModel


class SubtitleOperation(BaseModel):
    action: Literal["edit", "delete", "merge", "split"]
    ids: List[int]
    texts: List[str]


class CorrectionResponse(BaseModel):
    operations: List[SubtitleOperation]


class ResegmentSubtitle(BaseModel):
    number: int
    text: str
    start: float
    end: float


class SubtitleList(BaseModel):
    subtitles: List[ResegmentSubtitle]
