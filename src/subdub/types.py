from typing import Literal, NoReturn, Optional, Protocol

TaskName = Literal[
    "tts",
    "full",
    "transcribe",
    "translate",
    "speech_blocks",
    "sync",
    "equalize",
    "correct",
    "zoom-transcript",
    "translation",
]


class ParserProtocol(Protocol):
    def error(self, message: str) -> NoReturn:
        ...


class CliArgsProtocol(Protocol):
    input: Optional[str]
    log: bool
    source_language: str
    target_language: Optional[str]
    llm_char: int

    ant_api: Optional[str]
    openai_api: Optional[str]
    gemini_api: Optional[str]
    api_deepl: Optional[str]
    model: str
    use_deepl: bool

    whisper_model: str
    align_model: Optional[str]
    whisper_prompt: Optional[str]
    chunk_size: int
    diarize: bool
    hf_token: Optional[str]
    save_txt: bool

    session: Optional[str]
    video: Optional[str]
    task: TaskName

    merge_threshold: int
    max_line_length: int
    characters: int
    delay_start: int
    speed_up: int

    evaluate: bool
    translation_memory: bool
    equalize: bool
    resegment: bool
    correct: bool
    context: bool
    no_remove_subtitles: bool
    manual_correction: bool
    boundary_correction: bool

    tts_voice: Optional[str]

    t_prompt: Optional[str]
    eval_prompt: Optional[str]
    gloss_prompt: Optional[str]
    sys_prompt: Optional[str]
    correct_prompt: Optional[str]
    translate_prompt: Optional[str]

    api_base: Optional[str]
    provider: Optional[str]
    sort: Optional[str]
    allow_fallbacks: bool
    ignore: Optional[str]
    data_collection: Optional[str]
    require_parameters: bool

    max_tokens: Optional[int]
    reasoning_effort: Optional[str]
