import os
import shutil
from typing import Any, Dict, Optional, Tuple

from .config import AppConfig
from .context import SessionContext
from .core.network import download_from_url
from .errors import ConfigurationError
from .subtitles.srt_utils import equalize_srt
from .types import CliArgsProtocol, ParserProtocol
from .workflows.dubbing import sync_audio_video


def build_app_config(args: CliArgsProtocol) -> AppConfig:
    return AppConfig(
        model=args.model,
        whisper_model=args.whisper_model,
        align_model=args.align_model,
        char_limit=args.llm_char,
        max_line_length=args.max_line_length,
        chunk_size=args.chunk_size,
        whisper_prompt=args.whisper_prompt,
        translation_prompt=args.t_prompt,
        evaluation_prompt=args.eval_prompt,
        glossary_prompt=args.gloss_prompt,
        system_prompt=args.sys_prompt,
        correction_prompt=args.correct_prompt,
        use_deepl=args.use_deepl,
        diarize=args.diarize,
        boundary_correction=args.boundary_correction,
        manual_correction=args.manual_correction,
        save_txt=args.save_txt,
        evaluate=args.evaluate,
        translation_memory=args.translation_memory,
        equalize=args.equalize,
        resegment=args.resegment,
        correct=args.correct,
        context=args.context,
        no_remove_subtitles=args.no_remove_subtitles,
        delay_start=args.delay_start,
        speed_up=args.speed_up,
    )


def resolve_hf_token(args: CliArgsProtocol) -> Optional[str]:
    hf_token = None
    if args.diarize:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ConfigurationError("HF token is required for diarization.")
    return hf_token


def build_provider_params(args: CliArgsProtocol) -> Optional[Dict[str, Any]]:
    provider_params = {}
    if args.api_base:
        provider_params["api_base"] = args.api_base
        is_local = "localhost" in args.api_base or "127.0.0.1" in args.api_base or "0.0.0.0" in args.api_base
        if is_local and not os.environ.get("OPENAI_API_KEY"):
            provider_params["api_key"] = "lm-studio"
            os.environ["OPENAI_API_KEY"] = "lm-studio"

    if args.model.startswith("openrouter/"):
        if args.provider:
            provider_params["order"] = [p.strip() for p in args.provider.split(",")]
        if args.sort:
            provider_params["sort"] = args.sort
        if hasattr(args, "allow_fallbacks"):
            provider_params["allow_fallbacks"] = args.allow_fallbacks
        if args.ignore:
            provider_params["ignore"] = [p.strip() for p in args.ignore.split(",")]
        if args.data_collection:
            provider_params["data_collection"] = args.data_collection
        if args.require_parameters:
            provider_params["require_parameters"] = True

    if args.max_tokens:
        provider_params["max_tokens"] = args.max_tokens
    if args.reasoning_effort:
        provider_params["reasoning_effort"] = args.reasoning_effort

    return provider_params or None


def apply_default_align_model(args: CliArgsProtocol) -> None:
    if args.align_model:
        return

    language_align_models = {
        "pl": "jonatasgrosman/wav2vec2-xls-r-1b-polish",
        "nl": "GroNLP/wav2vec2-dutch-large-ft-cgn",
        "de": "aware-ai/wav2vec2-xls-r-1b-german",
        "en": "jonatasgrosman/wav2vec2-xls-r-1b-english",
        "fr": "jonatasgrosman/wav2vec2-xls-r-1b-french",
        "it": "jonatasgrosman/wav2vec2-xls-r-1b-italian",
        "ru": "jonatasgrosman/wav2vec2-xls-r-1b-russian",
        "es": "jonatasgrosman/wav2vec2-xls-r-1b-spanish",
        "ja": "vumichien/wav2vec2-xls-r-1b-japanese",
        "hu": "sarpba/wav2vec2-large-xlsr-53-hungarian",
        "sq": "Alimzhan/wav2vec2-large-xls-r-300m-albanian-colab",
    }
    lang_to_code_map = {
        "english": "en",
        "polish": "pl",
        "polski": "pl",
        "dutch": "nl",
        "nederlands": "nl",
        "german": "de",
        "deutsch": "de",
        "french": "fr",
        "français": "fr",
        "italian": "it",
        "italiano": "it",
        "russian": "ru",
        "русский": "ru",
        "rus": "ru",
        "spanish": "es",
        "español": "es",
        "japanese": "ja",
        "hungarian": "hu",
        "albanian": "sq",
        "en": "en",
        "pl": "pl",
        "nl": "nl",
        "de": "de",
        "fr": "fr",
        "it": "it",
        "ru": "ru",
        "es": "es",
        "ja": "ja",
        "hu": "hu",
        "sq": "sq",
    }
    lookup_code = lang_to_code_map.get(args.source_language.lower())
    if lookup_code:
        args.align_model = language_align_models.get(lookup_code)


def handle_preflight_tasks(args: CliArgsProtocol, parser: ParserProtocol) -> bool:
    if args.task == "equalize":
        if not args.input:
            parser.error("the following arguments are required for 'equalize' task: -i/--input")
        input_srt_path = os.path.abspath(os.path.expanduser(args.input))
        output_srt_path = os.path.splitext(input_srt_path)[0] + "_equalized.srt"
        equalize_srt(input_srt_path, output_srt_path, args.characters)
        return True

    if args.task == "sync":
        if not args.session:
            parser.error("Session folder (--session) must be specified for the 'sync' task")
        sync_audio_video(args.session, args.video, args.delay_start, args.speed_up)
        return True

    return False


def prepare_context_from_input(args: CliArgsProtocol) -> Tuple[SessionContext, str, str, str]:
    if args.input.startswith(("http://", "https://", "www.")):
        temp_session_folder = SessionContext.create("temp_download").session_folder
        video_path, video_name = download_from_url(args.input, temp_session_folder)

        context = SessionContext.create(video_name, args.session)
        session_folder = context.session_folder
        if session_folder != temp_session_folder:
            destination = os.path.join(session_folder, os.path.basename(video_path))
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.move(video_path, destination)
            if os.path.exists(temp_session_folder) and os.path.abspath(temp_session_folder) != os.path.abspath(session_folder):
                try:
                    shutil.rmtree(temp_session_folder)
                except OSError:
                    pass
        video_path = os.path.join(session_folder, os.path.basename(video_path))
        context.video_path = video_path
        return context, session_folder, video_path, video_name

    video_path = os.path.abspath(os.path.expanduser(args.input))
    video_name_raw = os.path.splitext(os.path.basename(video_path))[0]
    video_name = "".join(e for e in video_name_raw if e.isalnum() or e in ["-", "_"])
    if not video_name:
        video_name = "default_video_name"

    context = SessionContext.create(video_name, args.session)
    context.video_path = video_path
    session_folder = context.session_folder
    return context, session_folder, video_path, video_name
