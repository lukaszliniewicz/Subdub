import json
import os

from ..ai.memory import manage_glossary, save_glossary
from ..ai.translate import (
    evaluate_resegmented_translation,
    evaluate_translation,
    resegment_and_translate_with_llm,
    translate_blocks,
    translate_blocks_deepl,
)
from ..subtitles.chunking import create_translation_blocks, create_word_blocks
from ..subtitles.srt_utils import (
    convert_llm_resegment_to_srt,
    equalize_srt,
    parse_deepl_response,
    parse_translated_response,
)
from ..workflows.pipeline import (
    apply_word_level_boundary_correction,
    open_manual_correction_gui,
    preprocess_words_from_json,
)
from .runtime import RuntimeContext


TRANSLATION_TASKS = ["full", "translate", "translation"]


def run_translation_stage(runtime: RuntimeContext):
    args = runtime.args
    state = runtime.state
    session_folder = runtime.session_folder
    video_name = runtime.video_name
    provider_params = runtime.provider_params
    system_prompt_to_use = runtime.system_prompt
    translation_prompt_template_to_use = runtime.translation_prompt_template
    evaluation_prompt_template_to_use = runtime.evaluation_prompt_template
    glossary_prompt_instructions_to_use = runtime.glossary_prompt_instructions
    resegment_translation_prompt_template = runtime.resegment_translation_prompt_template
    resegment_evaluation_prompt_template = runtime.resegment_evaluation_prompt_template

    if args.task not in TRANSLATION_TASKS:
        return

    state.evaluation_suffix = "_eval" if args.evaluate else ""

    if args.resegment:
        if args.boundary_correction and state.audio_path:
            words = apply_word_level_boundary_correction(state.srt_path, state.audio_path)
        else:
            words = preprocess_words_from_json(state.srt_path)
        word_blocks = create_word_blocks(words, args.llm_char)

        grouped_llm_subtitles, returned_translation_cost = resegment_and_translate_with_llm(
            word_blocks,
            args.source_language,
            args.target_language,
            args.model,
            system_prompt_to_use,
            resegment_translation_prompt_template,
            args.max_line_length,
            provider_params=provider_params,
        )
        state.translation_cost += returned_translation_cost
        state.total_cost += state.translation_cost

        final_grouped_subtitles = grouped_llm_subtitles

        if args.evaluate:
            final_grouped_subtitles, returned_evaluation_cost = evaluate_resegmented_translation(
                word_blocks,
                grouped_llm_subtitles,
                args.source_language,
                args.target_language,
                args.model,
                system_prompt_to_use,
                resegment_evaluation_prompt_template,
                args.max_line_length,
                provider_params=provider_params,
            )
            state.evaluation_cost = returned_evaluation_cost
            state.total_cost += state.evaluation_cost

        llm_subtitles = [sub for group in final_grouped_subtitles for sub in group if sub.get("text", "").strip() != "[REMOVE]"]

        state.final_srt = convert_llm_resegment_to_srt(llm_subtitles)
        state.translated_srt_path = os.path.join(
            session_folder,
            f"{video_name}_{args.target_language}{state.evaluation_suffix}_resegmented.srt",
        )
    else:
        translation_blocks = create_translation_blocks(state.srt_content_for_llm, args.llm_char, args.source_language)
        glossary = manage_glossary(session_folder) if args.translation_memory else {}

        if args.use_deepl:
            deepl_key = os.environ.get("DEEPL_API_KEY")
            translated_blocks_responses = translate_blocks_deepl(
                translation_blocks,
                args.source_language,
                args.target_language,
                deepl_key,
            )
            final_blocks_responses = translated_blocks_responses
        else:
            translated_blocks_responses, updated_glossary, returned_translation_cost = translate_blocks(
                translation_blocks,
                args.source_language,
                args.target_language,
                args.model,
                glossary,
                args.translation_memory,
                args.evaluate,
                translation_prompt_template_to_use,
                args.translate_prompt,
                glossary_prompt_instructions_to_use,
                system_prompt_to_use,
                use_context=args.context,
                no_remove_subtitles=args.no_remove_subtitles,
                provider_params=provider_params,
            )
            state.translation_cost = returned_translation_cost
            state.total_cost += state.translation_cost

            if args.translation_memory and not args.evaluate:
                save_glossary(session_folder, updated_glossary)

            if args.evaluate:
                if not updated_glossary:
                    updated_glossary = glossary
                final_blocks_responses, final_glossary, returned_evaluation_cost = evaluate_translation(
                    translation_blocks,
                    translated_blocks_responses,
                    args.source_language,
                    args.target_language,
                    updated_glossary,
                    args.translation_memory,
                    args.model,
                    evaluation_prompt_template_to_use,
                    system_prompt_to_use,
                    provider_params=provider_params,
                    no_remove_subtitles=args.no_remove_subtitles,
                )
                state.evaluation_cost = returned_evaluation_cost
                state.total_cost += state.evaluation_cost
                state.evaluation_suffix = "_eval"

                if args.translation_memory:
                    save_glossary(session_folder, final_glossary)
            else:
                final_blocks_responses = translated_blocks_responses

        final_json_path = os.path.join(
            session_folder,
            f"{video_name}_{args.target_language}{state.evaluation_suffix}_final_blocks.json",
        )
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(final_blocks_responses, f, ensure_ascii=False, indent=2)

        if args.use_deepl:
            state.final_srt = parse_deepl_response(final_blocks_responses, state.srt_content_for_llm)
        else:
            state.final_srt = parse_translated_response(final_blocks_responses, state.srt_content_for_llm)

        state.translated_srt_path = os.path.join(
            session_folder,
            f"{video_name}_{args.target_language}{state.evaluation_suffix}.srt",
        )

    with open(state.translated_srt_path, "w", encoding="utf-8") as f:
        f.write(state.final_srt)

    if args.manual_correction and args.task in ["translate", "translation"] and state.audio_path:
        state.translated_srt_path = open_manual_correction_gui(state.translated_srt_path, state.audio_path, session_folder)
        with open(state.translated_srt_path, "r", encoding="utf-8") as f:
            state.final_srt = f.read()


def maybe_finalize_translation_task(runtime: RuntimeContext) -> bool:
    args = runtime.args
    state = runtime.state
    session_folder = runtime.session_folder
    video_name = runtime.video_name

    if args.task not in ["translate", "translation"]:
        return False

    if args.equalize:
        output_srt_eq_path = os.path.join(
            session_folder,
            f"{video_name}_{args.target_language}{state.evaluation_suffix}_final.srt",
        )
        equalize_srt(state.translated_srt_path, output_srt_eq_path, args.max_line_length)
    return True
