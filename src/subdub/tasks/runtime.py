from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from ..types import CliArgsProtocol, ParserProtocol
from .state import PipelineState


@dataclass
class RuntimeContext:
    args: CliArgsProtocol
    parser: ParserProtocol
    session_folder: str
    video_path: str
    video_name: str
    provider_params: Optional[Dict[str, Any]]
    hf_token: Optional[str]

    system_prompt: str
    translation_prompt_template: str
    evaluation_prompt_template: str
    glossary_prompt_instructions: str
    correction_prompt_template: str
    resegment_correction_prompt_template: str
    resegment_translation_prompt_template: str
    resegment_evaluation_prompt_template: str

    get_xtts_language_code_fn: Callable[[str], str]
    state: PipelineState = field(default_factory=PipelineState)
