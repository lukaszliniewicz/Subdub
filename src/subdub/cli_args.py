import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video subtitle translation and dubbing tool")
    parser.add_argument("-i", "--input", help="Input video path or URL")
    parser.add_argument("-log", action="store_true", help="Enable logging to a file in the session folder.")
    parser.add_argument("-sl", "--source_language", default="English", help="Source language (default: English)")
    parser.add_argument("-tl", "--target_language", help="Target language")
    parser.add_argument("-llm-char", type=int, default=4000, help="Character limit for translation (default: 4000)")
    parser.add_argument("-ant_api", help="Anthropic API key")
    parser.add_argument("-evaluate", action="store_true", help="Perform evaluation of translations")
    parser.add_argument("-translation_memory", action="store_true", help="Enable translation memory/glossary feature")
    parser.add_argument("-tts_voice", help="Path to TTS voice WAV file")
    parser.add_argument("-whisper_model", default="large-v3", help="Whisper model to use for transcription.")
    parser.add_argument("-align_model", help="Custom alignment model for WhisperX.")
    parser.add_argument("-diarize", action="store_true", help="Enable speaker diarization using WhisperX")
    parser.add_argument("--hf_token", help="Hugging Face token for diarization")
    parser.add_argument("-openai_api", help="OpenAI API key")
    parser.add_argument("-model", default="anthropic/claude-3-5-sonnet-20241022", help="LiteLLM model string")
    parser.add_argument("-api_base", help="Base URL for local/custom API endpoints")
    parser.add_argument("--use-deepl", action="store_true", help="Use DeepL instead of an LLM for translation")
    parser.add_argument("-gemini_api", help="Google Gemini API key")
    parser.add_argument("-session", help="Session name or path.")
    parser.add_argument("-merge_threshold", type=int, default=250, help="Maximum time difference (in ms) between subtitles to be merged")
    parser.add_argument(
        "-task",
        choices=["tts", "full", "transcribe", "translate", "speech_blocks", "sync", "equalize", "correct", "zoom-transcript"],
        default="full",
        help="Task to perform",
    )
    parser.add_argument("-t_prompt", help="Custom translation prompt")
    parser.add_argument("-eval_prompt", help="Custom evaluation prompt")
    parser.add_argument("-gloss_prompt", help="Custom glossary prompt")
    parser.add_argument("-sys_prompt", help="Custom system prompt")
    parser.add_argument("-equalize", action="store_true", help="Apply SRT equalizer to the final subtitle file")
    parser.add_argument("-max_line_length", type=int, default=42, help="Maximum line length for SRT equalization")
    parser.add_argument("-max_tokens", type=int, help="Max output tokens.")
    parser.add_argument("-reasoning_effort", choices=["minimal", "low", "medium", "high"], default=None, help="Reasoning effort level")
    parser.add_argument("-api_deepl", help="DeepL API key")
    parser.add_argument("-characters", type=int, default=60, help="Maximum line length for SRT equalization (default: 60)")
    parser.add_argument("-v", "--video", help="Input video file for syncing (optional)")
    parser.add_argument("-resegment", action="store_true", help="Enable word-level re-segmentation and correction.")
    parser.add_argument("-correct", action="store_true", help="Enable subtitle correction before translation")
    parser.add_argument("-correct_prompt", help="Additional context/instructions for subtitle correction")
    parser.add_argument("-context", action="store_true", help="Add previous output as context")
    parser.add_argument("-translate_prompt", help="Additional context/instructions for translation")
    parser.add_argument("--no-remove-subtitles", action="store_true", help="Instruct the LLM not to remove any subtitles.")
    parser.add_argument("--delay_start", type=int, default=2000, help="Delay audio start by this many milliseconds")
    parser.add_argument("--speed_up", type=int, default=115, help="Maximum speed-up percentage")
    parser.add_argument("-whisper_prompt", help="Custom initial prompt to guide Whisper transcription.")
    parser.add_argument("-chunk_size", type=int, default=15, help="Chunk size for WhisperX transcription")
    parser.add_argument("-provider", help="OpenRouter provider to prioritize")
    parser.add_argument("-sort", choices=["price", "throughput", "latency"], help="OpenRouter provider sorting strategy")
    parser.add_argument("-fallbacks", dest="allow_fallbacks", action="store_true", default=True, help="Allow fallbacks to other providers")
    parser.add_argument("-no-fallbacks", dest="allow_fallbacks", action="store_false", help="Disable fallbacks to other providers")
    parser.add_argument("-ignore", help="OpenRouter providers to ignore")
    parser.add_argument("-data-collection", choices=["allow", "deny"], default="allow", help="OpenRouter data collection policy")
    parser.add_argument("-require-parameters", action="store_true", help="Require providers to support all parameters")
    parser.add_argument(
        "--no-boundary-correction",
        dest="boundary_correction",
        action="store_false",
        default=True,
        help="Disable automatic boundary correction",
    )
    parser.add_argument("-manual_correction", action="store_true", help="Open GUI for manual boundary correction")
    parser.add_argument("--save_txt", action="store_true", help="Save a .txt version of the transcript")
    return parser
