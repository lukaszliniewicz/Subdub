import os

from ..subtitles.srt_utils import equalize_srt
from ..workflows.pipeline import open_manual_correction_gui
from .runtime import RuntimeContext


def run_transcribe_task(runtime: RuntimeContext) -> bool:
    args = runtime.args
    state = runtime.state
    session_folder = runtime.session_folder
    video_name = runtime.video_name

    if args.task != "transcribe":
        return False

    if args.manual_correction and state.audio_path:
        state.srt_path = open_manual_correction_gui(state.srt_path, state.audio_path, session_folder)

    if args.equalize:
        output_srt_eq_path = os.path.join(
            session_folder,
            f"{video_name}_{args.source_language}{'_corrected' if args.correct else ''}_final.srt",
        )
        equalize_srt(state.srt_path, output_srt_eq_path, args.max_line_length)

    return True
