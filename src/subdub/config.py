from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class AppConfig:
    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    deepl_api_key: Optional[str] = None
    hf_token: Optional[str] = None
    
    # Models
    model: str = 'anthropic/claude-3-5-sonnet-20241022'
    whisper_model: str = 'large-v3'
    align_model: Optional[str] = None
    
    # LLM / Provider Params
    provider_params: Dict = field(default_factory=dict)
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    
    # Thresholds & Limits
    max_retries: int = 3
    char_limit: int = 4000
    speech_block_char_limit: int = 160
    speech_block_min_chars: int = 10
    speech_block_merge_threshold: int = 250
    max_line_length: int = 42
    chunk_size: int = 15
    
    # Prompts
    whisper_prompt: str = "Hello, welcome to this presentation. This is a professional recording with clear speech, proper punctuation, and standard grammar."
    translation_prompt: Optional[str] = None
    evaluation_prompt: Optional[str] = None
    glossary_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    correction_prompt: Optional[str] = None
    
    # Flags
    use_deepl: bool = False
    diarize: bool = False
    boundary_correction: bool = True
    manual_correction: bool = False
    save_txt: bool = False
    evaluate: bool = False
    translation_memory: bool = False
    equalize: bool = False
    resegment: bool = False
    correct: bool = False
    context: bool = False
    no_remove_subtitles: bool = False
    
    # Audio/Video Sync
    delay_start: int = 2000
    speed_up: int = 115
