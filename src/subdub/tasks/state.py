from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PipelineState:
    srt_content: str = ""
    srt_content_for_llm: str = ""
    srt_path: str = ""
    audio_path: Optional[str] = None

    final_srt: str = ""
    translated_srt_path: str = ""
    speech_blocks: Optional[List[Dict]] = None

    correction_cost: float = 0.0
    translation_cost: float = 0.0
    evaluation_cost: float = 0.0
    total_cost: float = 0.0

    evaluation_suffix: str = ""
