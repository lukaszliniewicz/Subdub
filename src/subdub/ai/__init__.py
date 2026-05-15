"""AI module for transcription, translation, correction, and evaluation."""

from .client import MAX_RETRIES, calculate_cost, configure_litellm_callbacks, llm_api_request
from .correction import (
    correct_subtitles,
    correct_transcript_chunks,
    resegment_and_correct_with_llm,
    resegment_and_translate_with_llm,
)
from .evaluation import evaluate_resegmented_translation, evaluate_translation
from .translation import get_deepl_language_code, translate_blocks, translate_blocks_deepl

__all__ = [
    "MAX_RETRIES",
    "calculate_cost",
    "configure_litellm_callbacks",
    "llm_api_request",
    "get_deepl_language_code",
    "translate_blocks_deepl",
    "translate_blocks",
    "evaluate_translation",
    "evaluate_resegmented_translation",
    "correct_transcript_chunks",
    "correct_subtitles",
    "resegment_and_correct_with_llm",
    "resegment_and_translate_with_llm",
]
