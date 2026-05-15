import logging
import os

from .ai.client import configure_litellm_callbacks
from .app_helpers import (
    apply_default_align_model,
    build_app_config,
    build_provider_params,
    handle_preflight_tasks,
    prepare_context_from_input,
    resolve_hf_token,
)
from .cli_helpers import get_xtts_language_code, set_api_key, validate_api_keys
from .errors import SubdubError
from .prompts import (
    CORRECTION_PROMPT_TEMPLATE,
    CORRECTION_SYSTEM_PROMPT,
    CUSTOM_SYSTEM_PROMPT,
    EVALUATION_PROMPT_TEMPLATE,
    GLOSSARY_INSTRUCTIONS_TRANSLATION,
    RESEGMENT_CORRECTION_PROMPT_TEMPLATE,
    RESEGMENT_EVALUATION_PROMPT_TEMPLATE,
    RESEGMENT_TRANSLATION_PROMPT_TEMPLATE,
    TRANSLATION_PROMPT_TEMPLATE,
)
from .tasks.correction import run_correction_stage
from .tasks.input import prepare_source_subtitles
from .tasks.runtime import RuntimeContext
from .tasks.speech import run_full_stage, run_speech_blocks_stage
from .tasks.transcribe import run_transcribe_task
from .tasks.translation import maybe_finalize_translation_task, run_translation_stage
from .types import CliArgsProtocol, ParserProtocol
from .workflows.pipeline import run_zoom_transcript_pipeline
from .core.logger import setup_logging

logger = logging.getLogger(__name__)


def run_app(args: CliArgsProtocol, parser: ParserProtocol):
    set_api_key(args.ant_api, "ANTHROPIC_API_KEY")
    set_api_key(args.openai_api, "OPENAI_API_KEY")
    set_api_key(args.gemini_api, "GEMINI_API_KEY")
    set_api_key(args.api_deepl, "DEEPL_API_KEY")

    openrouter_api = os.environ.get("OPENROUTER_API")
    if openrouter_api and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = openrouter_api

    validate_api_keys(args, args.model)
    configure_litellm_callbacks()

    if args.task != "sync" and not args.input:
        parser.error("the following arguments are required: -i/--input (unless task is 'sync')")

    config = build_app_config(args)
    hf_token = resolve_hf_token(args)

    provider_params = build_provider_params(args)
    config.provider_params = provider_params or {}

    apply_default_align_model(args)

    if handle_preflight_tasks(args, parser):
        return

    context, session_folder, video_path, video_name = prepare_context_from_input(args)
    setup_logging(session_folder, args.log)

    translation_prompt_template_to_use = args.t_prompt if args.t_prompt else TRANSLATION_PROMPT_TEMPLATE
    evaluation_prompt_template_to_use = args.eval_prompt if args.eval_prompt else EVALUATION_PROMPT_TEMPLATE
    glossary_prompt_instructions_to_use = args.gloss_prompt if args.gloss_prompt else GLOSSARY_INSTRUCTIONS_TRANSLATION

    if args.sys_prompt:
        system_prompt_to_use = args.sys_prompt
    elif args.task == "correct" or args.correct:
        system_prompt_to_use = CORRECTION_SYSTEM_PROMPT
    else:
        system_prompt_to_use = CUSTOM_SYSTEM_PROMPT

    correction_prompt_template_to_use = args.correct_prompt if args.correct_prompt else CORRECTION_PROMPT_TEMPLATE

    runtime = RuntimeContext(
        args=args,
        parser=parser,
        session_folder=session_folder,
        video_path=video_path,
        video_name=video_name,
        provider_params=provider_params,
        hf_token=hf_token,
        system_prompt=system_prompt_to_use,
        translation_prompt_template=translation_prompt_template_to_use,
        evaluation_prompt_template=evaluation_prompt_template_to_use,
        glossary_prompt_instructions=glossary_prompt_instructions_to_use,
        correction_prompt_template=correction_prompt_template_to_use,
        resegment_correction_prompt_template=RESEGMENT_CORRECTION_PROMPT_TEMPLATE,
        resegment_translation_prompt_template=RESEGMENT_TRANSLATION_PROMPT_TEMPLATE,
        resegment_evaluation_prompt_template=RESEGMENT_EVALUATION_PROMPT_TEMPLATE,
        get_xtts_language_code_fn=get_xtts_language_code,
    )

    try:
        if args.task == "zoom-transcript":
            try:
                run_zoom_transcript_pipeline(config, context)
            except ValueError as e:
                parser.error(str(e))
            return

        prepare_source_subtitles(runtime)

        if run_correction_stage(runtime):
            return

        if run_transcribe_task(runtime):
            return

        run_translation_stage(runtime)

        if maybe_finalize_translation_task(runtime):
            return

        if run_speech_blocks_stage(runtime):
            return

        run_full_stage(runtime)

    except SubdubError as e:
        logger.error(str(e), exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    finally:
        if runtime.state.total_cost > 0.0:
            logger.info("\n" + "=" * 80)
            logger.info(f"TOTAL ESTIMATED API COST FOR SESSION: ${runtime.state.total_cost:.6f}")
            logger.info("=" * 80 + "\n")
