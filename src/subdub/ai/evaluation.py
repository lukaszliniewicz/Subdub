import json
import logging
from typing import Dict, List, Tuple, Union

from .client import MAX_RETRIES, calculate_cost, llm_api_request
from ..prompts import GLOSSARY_INSTRUCTIONS_EVALUATION, TRANSLATION_PROMPT_TEMPLATE
from ..schemas.llm import ResegmentSubtitle

logger = logging.getLogger(__name__)


def evaluate_translation(
    translation_blocks: List[List[Dict]],
    full_responses: List[Dict[str, Union[str, List[str]]]],
    source_lang: str,
    target_lang: str,
    original_glossary: Dict[str, str],
    use_translation_memory: bool,
    model: str,
    evaluation_prompt: str,
    system_prompt: str,
    provider_params: Dict = None,
    use_thinking: bool = False,
    thinking_tokens: int = 8000,
    no_remove_subtitles: bool = False,
) -> Tuple[List[Dict[str, Union[str, List[str]]]], Dict[str, str], float]:
    total_cost = 0.0
    evaluated_responses = []
    new_glossary = original_glossary.copy()

    for i, (block, full_response) in enumerate(zip(translation_blocks, full_responses)):
        glossary_instructions = (
            GLOSSARY_INSTRUCTIONS_EVALUATION.format(glossary=json.dumps(original_glossary, ensure_ascii=False, indent=2))
            if use_translation_memory
            else ""
        )

        translation_guidelines_formatted = TRANSLATION_PROMPT_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            subtitle_count="[variable number of]",
            glossary_instructions="",
        )

        base_prompt = evaluation_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            subtitle_count=len(block),
            glossary_instructions=glossary_instructions,
            translation_guidelines=translation_guidelines_formatted,
        )

        if no_remove_subtitles:
            removal_instruction = """4. For subtitles marked as \"[REMOVE]\":
   - If you agree it should be removed, and it was already \"[REMOVE]\", you don't need to include it in your output.
   - If you agree it should be removed, but it wasn't, return it with its \"number\" and the text \"[REMOVE]\".
   - If you think it should be kept (it was previously \"[REMOVE]\"), provide your translation in the \"text\" field for that \"number\"."""
            no_removal_instruction = '4. You MUST NOT remove any subtitles. If a subtitle in the original translation was marked as "[REMOVE]", you must provide a correct translation for it.'
            base_prompt = base_prompt.replace(removal_instruction, no_removal_instruction)

        original_subtitles = json.dumps([{"number": idx + 1, "text": sub["text"]} for idx, sub in enumerate(block)], ensure_ascii=False)
        original_translation = json.dumps([{"number": idx + 1, "text": text} for idx, text in enumerate(full_response["translation"])], ensure_ascii=False)

        prompt = f"""{base_prompt}

        Original {source_lang} subtitles:
        {original_subtitles}

        Original translation response:
        {original_translation}

        Your improved translation:
        """

        for attempt in range(MAX_RETRIES):
            try:
                messages = [{"role": "user", "content": prompt}]
                content, response = llm_api_request(
                    model,
                    messages,
                    system_prompt=system_prompt,
                    provider_params=provider_params,
                    use_thinking=use_thinking,
                    thinking_tokens=thinking_tokens,
                )

                cost = calculate_cost(response, model=model)
                total_cost += cost

                logger.info(f"Complete model output for evaluation block {i + 1}:\n{content}")

                try:
                    evaluated_content, _, glossary_content = content.partition("[GLOSSARY]")
                    changes_json = json.loads(evaluated_content)

                    evaluated_subtitles = list(full_response["translation"])

                    for change in changes_json:
                        if not isinstance(change, dict) or "number" not in change or "text" not in change:
                            raise ValueError("Invalid format for a changed subtitle in evaluation response.")

                        change_index = change["number"] - 1

                        if 0 <= change_index < len(evaluated_subtitles):
                            evaluated_subtitles[change_index] = change["text"]
                        else:
                            logger.warning(
                                f"Evaluation model returned out-of-bounds subtitle number {change['number']} "
                                f"for a block of size {len(evaluated_subtitles)}. Ignoring this change."
                            )

                    evaluated_responses.append(
                        {
                            "translation": evaluated_subtitles,
                            "new_glossary": glossary_content.strip() if use_translation_memory else "",
                            "original_indices": full_response["original_indices"],
                        }
                    )

                    if use_translation_memory:
                        new_entries = dict(entry.split("=") for entry in glossary_content.strip().split("\n") if "=" in entry)
                        new_glossary.update(new_entries)

                    logger.info(f"Successfully evaluated block {i + 1}/{len(translation_blocks)}")
                    break
                except json.JSONDecodeError:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Failed to parse JSON response for block {i + 1} after {MAX_RETRIES} attempts")
                except ValueError as e:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Error in evaluation for block {i + 1}: {str(e)}")
            except Exception as e:
                logger.error(f"Error in evaluation attempt {attempt + 1} for block {i + 1}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    raise ValueError(f"Failed to evaluate block {i + 1} after {MAX_RETRIES} attempts. Last error: {str(e)}")

    return evaluated_responses, new_glossary, total_cost


def evaluate_resegmented_translation(
    word_blocks: List[Dict],
    grouped_initial_subtitles: List[List[Dict]],
    source_lang: str,
    target_lang: str,
    model: str,
    system_prompt: str,
    evaluation_prompt_template: str,
    max_line_length: int,
    provider_params: Dict = None,
    use_thinking: bool = False,
    thinking_tokens: int = 8000,
) -> Tuple[List[List[Dict]], float]:
    total_cost = 0.0
    evaluated_subtitle_groups = []

    for i, (block_data, initial_subs_for_block) in enumerate(zip(word_blocks, grouped_initial_subtitles)):
        logger.info(f"Evaluating resegmented block {i + 1}/{len(word_blocks)}.")

        words_to_process = block_data["words_to_process"]
        word_block_json = json.dumps(words_to_process, ensure_ascii=False, indent=2)
        initial_subtitles_json = json.dumps(initial_subs_for_block, ensure_ascii=False, indent=2)

        prompt = evaluation_prompt_template.format(
            source_lang=source_lang,
            target_lang=target_lang,
            max_line_length=max_line_length,
            word_block_json=word_block_json,
            initial_subtitles_json=initial_subtitles_json,
        )

        for attempt in range(MAX_RETRIES):
            try:
                messages = [{"role": "user", "content": prompt}]
                content, response = llm_api_request(
                    model=model,
                    messages=messages,
                    system_prompt=system_prompt,
                    provider_params=provider_params,
                    use_thinking=use_thinking,
                    thinking_tokens=thinking_tokens,
                    output_schema=list[ResegmentSubtitle],
                )

                cost = calculate_cost(response, model=model)
                total_cost += cost

                try:
                    corrected_block_subtitles = json.loads(content)
                    if len(corrected_block_subtitles) != len(initial_subs_for_block):
                        logger.warning(f"Evaluation for block {i + 1} returned a different number of subtitles. Using original.")
                        evaluated_subtitle_groups.append(initial_subs_for_block)
                    else:
                        evaluated_subtitle_groups.append(corrected_block_subtitles)
                    break
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON response for evaluation block {i + 1}, attempt {attempt + 1}. Retrying...")
                    if attempt == MAX_RETRIES - 1:
                        evaluated_subtitle_groups.append(initial_subs_for_block)
                        raise ValueError("Failed to get valid JSON from LLM after multiple retries.")
            except Exception as e:
                logger.error(f"LLM API call failed for evaluation block {i + 1}, attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    evaluated_subtitle_groups.append(initial_subs_for_block)
                    raise

    return evaluated_subtitle_groups, total_cost
