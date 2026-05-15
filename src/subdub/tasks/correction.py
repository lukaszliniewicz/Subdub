import os

from ..ai.translate import (
    correct_subtitles,
    resegment_and_correct_with_llm,
)
from ..subtitles.chunking import create_translation_blocks, create_word_blocks
from ..subtitles.srt_utils import convert_llm_resegment_to_srt, equalize_srt
from ..workflows.pipeline import (
    apply_word_level_boundary_correction,
    open_manual_correction_gui,
    preprocess_words_from_json,
)
from .runtime import RuntimeContext


def run_correction_stage(runtime: RuntimeContext) -> bool:
    args = runtime.args
    state = runtime.state
    session_folder = runtime.session_folder
    video_name = runtime.video_name
    provider_params = runtime.provider_params
    system_prompt_to_use = runtime.system_prompt
    correction_prompt_template_to_use = runtime.correction_prompt_template
    resegment_correction_prompt_template = runtime.resegment_correction_prompt_template

    if not (args.task == "correct" or args.correct):
        return False

    if args.resegment:
        if args.boundary_correction and state.audio_path:
            words = apply_word_level_boundary_correction(state.srt_path, state.audio_path)
        else:
            words = preprocess_words_from_json(state.srt_path)
        word_blocks = create_word_blocks(words, args.llm_char)

        llm_subtitles, returned_correction_cost = resegment_and_correct_with_llm(
            word_blocks,
            args.model,
            system_prompt_to_use,
            resegment_correction_prompt_template,
            args.max_line_length,
            provider_params=provider_params,
        )
        state.correction_cost = returned_correction_cost
        state.total_cost += state.correction_cost

        corrected_srt = convert_llm_resegment_to_srt(llm_subtitles)
        corrected_srt_filename_base = f"{video_name}_{args.source_language}_resegmented_corrected"

    else:
        correction_translation_blocks = create_translation_blocks(state.srt_content_for_llm, args.llm_char, args.source_language)
        corrected_subs, returned_correction_cost = correct_subtitles(
            correction_translation_blocks,
            args.source_language,
            args.correct_prompt or "",
            args.model,
            correction_prompt_template_to_use,
            system_prompt_to_use,
            use_context=args.context,
            provider_params=provider_params,
            no_remove_subtitles=args.no_remove_subtitles,
            max_line_length=args.max_line_length,
        )
        state.correction_cost = returned_correction_cost
        state.total_cost += state.correction_cost
        corrected_srt = convert_llm_resegment_to_srt(corrected_subs)
        corrected_srt_filename_base = f"{video_name}_{args.source_language}_corrected"

    corrected_srt_path = os.path.join(session_folder, f"{corrected_srt_filename_base}.srt")
    with open(corrected_srt_path, "w", encoding="utf-8") as f:
        f.write(corrected_srt)

    state.srt_content_for_llm = corrected_srt
    state.srt_path = corrected_srt_path

    if args.manual_correction and args.task == "correct" and state.audio_path:
        corrected_srt_path = open_manual_correction_gui(corrected_srt_path, state.audio_path, session_folder)
        state.srt_path = corrected_srt_path

    if args.task == "correct":
        if args.equalize:
            output_srt_eq_path = os.path.join(session_folder, f"{corrected_srt_filename_base}_final.srt")
            equalize_srt(corrected_srt_path, output_srt_eq_path, args.max_line_length)
        return True

    return False
