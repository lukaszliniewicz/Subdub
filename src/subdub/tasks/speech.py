import logging
import os
import shutil

from ..errors import ConfigurationError
from ..media.audio import align_audio_blocks
from ..media.ffmpeg import mix_audio_tracks
from ..media.tts import generate_tts_audio
from ..subtitles.chunking import create_speech_blocks
from ..subtitles.srt_utils import equalize_srt
from ..workflows.dubbing import create_alignment_blocks
from ..workflows.pipeline import cleanup_temp_files, open_manual_correction_gui
from .runtime import RuntimeContext

logger = logging.getLogger(__name__)


def run_speech_blocks_stage(runtime: RuntimeContext) -> bool:
    args = runtime.args
    state = runtime.state
    session_folder = runtime.session_folder
    video_name = runtime.video_name

    srt_for_speech_blocks = state.final_srt if state.final_srt else state.srt_content_for_llm
    lang_for_speech_blocks = args.target_language if args.task == "full" and args.target_language else args.source_language

    if args.task in ["full", "speech_blocks"]:
        state.speech_blocks = create_speech_blocks(
            srt_for_speech_blocks,
            session_folder,
            video_name,
            lang_for_speech_blocks,
            merge_threshold=args.merge_threshold,
        )

        if args.task == "speech_blocks":
            if args.equalize:
                base_srt_path_for_speech_blocks = state.translated_srt_path if state.final_srt else state.srt_path
                if base_srt_path_for_speech_blocks and os.path.exists(base_srt_path_for_speech_blocks):
                    output_srt_eq_path = os.path.splitext(base_srt_path_for_speech_blocks)[0] + "_final.srt"
                    equalize_srt(base_srt_path_for_speech_blocks, output_srt_eq_path, args.max_line_length)
            return True

    return False


def run_full_stage(runtime: RuntimeContext):
    args = runtime.args
    state = runtime.state
    session_folder = runtime.session_folder
    video_name = runtime.video_name
    video_path = runtime.video_path
    parser = runtime.parser
    get_xtts_language_code_fn = runtime.get_xtts_language_code_fn

    if args.task != "full":
        return

    if not args.tts_voice:
        tts_voices_folder = "tts-voices"
        if os.path.exists(tts_voices_folder):
            wav_files = [f for f in os.listdir(tts_voices_folder) if f.endswith(".wav")]
            if wav_files:
                args.tts_voice = os.path.join(tts_voices_folder, wav_files[0])
            else:
                parser.error("No WAV files found in the 'tts-voices' folder and no TTS voice specified via --tts_voice.")
        else:
            parser.error("No TTS voice specified via --tts_voice and 'tts-voices' folder not found.")

    try:
        tts_language = get_xtts_language_code_fn(args.target_language)
    except ConfigurationError as e:
        logger.error(str(e))
        return

    audio_files = generate_tts_audio(state.speech_blocks, args.tts_voice, tts_language, session_folder, video_name)

    if not audio_files:
        logger.error("No TTS audio files were generated. Cannot proceed with alignment and mixing.")
        return

    use_eval_suffix_for_alignment = args.evaluate and state.evaluation_suffix == "_eval"

    alignment_blocks = create_alignment_blocks(session_folder, video_name, args.target_language, use_eval_suffix_for_alignment)
    aligned_audio_path = align_audio_blocks(alignment_blocks, session_folder, args.delay_start, args.speed_up)

    if args.manual_correction and state.audio_path:
        if state.translated_srt_path and os.path.exists(state.translated_srt_path):
            state.translated_srt_path = open_manual_correction_gui(state.translated_srt_path, state.audio_path, session_folder)
            with open(state.translated_srt_path, "r", encoding="utf-8") as f:
                state.final_srt = f.read()

            state.speech_blocks = create_speech_blocks(
                state.final_srt,
                session_folder,
                video_name,
                args.target_language,
                merge_threshold=args.merge_threshold,
            )

    if video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        mix_audio_tracks(
            video_path,
            aligned_audio_path,
            session_folder,
            video_name,
            args.target_language,
            use_eval_suffix_for_alignment,
        )
    else:
        final_dub_path = os.path.join(
            session_folder,
            f"{video_name}_{args.target_language}{state.evaluation_suffix}_dubbed_audio.wav",
        )
        shutil.copy(aligned_audio_path, final_dub_path)

    if args.equalize:
        if state.translated_srt_path and os.path.exists(state.translated_srt_path):
            output_srt_eq_path = os.path.splitext(state.translated_srt_path)[0] + "_final.srt"
            equalize_srt(state.translated_srt_path, output_srt_eq_path, args.max_line_length)

    cleanup_temp_files(session_folder, args.task)
