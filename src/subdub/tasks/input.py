import os
import shutil

from ..ai.transcribe import transcribe_audio
from ..media.ffmpeg import extract_audio
from ..subtitles.srt_utils import (
    merge_subtitles_with_speaker_awareness,
    remove_speaker_labels,
    renumber_subtitles,
)
from ..workflows.pipeline import (
    apply_boundary_correction,
    open_manual_correction_gui,
)
from .runtime import RuntimeContext


SOURCE_PREP_TASKS = ["full", "translate", "translation", "correct", "transcribe", "speech_blocks"]


def prepare_source_subtitles(runtime: RuntimeContext):
    args = runtime.args
    state = runtime.state
    session_folder = runtime.session_folder
    video_path = runtime.video_path
    video_name = runtime.video_name
    hf_token = runtime.hf_token

    if args.task not in SOURCE_PREP_TASKS:
        return

    media_input_exts = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".wav",
        ".m4a",
        ".aac",
        ".flac",
        ".mp3",
        ".ogg",
        ".opus",
    )
    source_subtitle_exts = (".srt", ".json")

    if video_path.lower().endswith(media_input_exts):
        state.audio_path = extract_audio(video_path, session_folder, video_name)
        whisper_prompt = (
            args.whisper_prompt
            or "Hello, welcome to this presentation. This is a professional recording with clear speech, proper punctuation, and standard grammar."
        )

        transcription_output = transcribe_audio(
            state.audio_path,
            args.source_language,
            session_folder,
            video_name,
            args.whisper_model,
            args.align_model,
            whisper_prompt,
            args.diarize,
            hf_token,
            args.chunk_size,
            args.boundary_correction,
            args.save_txt,
        )

        if args.boundary_correction and transcription_output.endswith(".json") and not args.resegment:
            state.srt_path = apply_boundary_correction(
                transcription_output,
                state.audio_path,
                session_folder,
                video_name,
                args.manual_correction,
            )
        else:
            state.srt_path = transcription_output

    elif video_path.lower().endswith(source_subtitle_exts):
        input_ext = os.path.splitext(video_path)[1]
        state.srt_path = os.path.join(session_folder, f"{video_name}_input{input_ext}")
        shutil.copy(video_path, state.srt_path)

        if input_ext == ".srt" and args.manual_correction:
            audio_files = [f for f in os.listdir(session_folder) if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))]
            if audio_files:
                state.audio_path = os.path.join(session_folder, audio_files[0])
                state.srt_path = open_manual_correction_gui(state.srt_path, state.audio_path, session_folder)

    if state.srt_path.lower().endswith(".srt"):
        with open(state.srt_path, "r", encoding="utf-8") as f:
            state.srt_content = f.read()

        state.srt_content = renumber_subtitles(state.srt_content)
        merged_srt_content, has_diarization = merge_subtitles_with_speaker_awareness(state.srt_content, args.merge_threshold)
        if merged_srt_content != state.srt_content:
            merged_srt_path = os.path.join(session_folder, f"{video_name}_merged.srt")
            with open(merged_srt_path, "w", encoding="utf-8") as f:
                f.write(merged_srt_content)
            state.srt_content = merged_srt_content
            state.srt_path = merged_srt_path

        if has_diarization:
            state.srt_content_for_llm = remove_speaker_labels(state.srt_content)
        else:
            state.srt_content_for_llm = state.srt_content
    else:
        state.srt_content_for_llm = ""
