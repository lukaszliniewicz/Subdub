import json
import logging
from typing import Dict, List, Tuple

from .client import MAX_RETRIES, calculate_cost, llm_api_request
from ..prompts import CONTEXT_PROMPT_TEMPLATE, CORRECTION_PROMPT_TEMPLATE, ZOOM_CORRECTION_PROMPT_TEMPLATE
from ..schemas.llm import CorrectionResponse, ResegmentSubtitle

logger = logging.getLogger(__name__)


def correct_transcript_chunks(
    chunks: List[str],
    model: str,
    system_prompt: str,
    provider_params: Dict = None,
    use_thinking: bool = False,
    thinking_tokens: int = 8000,
) -> Tuple[str, float]:
    total_cost = 0.0
    corrected_chunks = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Correcting transcript chunk {i + 1}/{len(chunks)}")

        prompt = ZOOM_CORRECTION_PROMPT_TEMPLATE.format(transcript_chunk=chunk)

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
                )

                cost = calculate_cost(response, model=model)
                total_cost += cost

                corrected_chunks.append(content)
                logger.info(f"Successfully corrected chunk {i + 1}")
                break
            except Exception as e:
                logger.error(f"Error in correction attempt {attempt + 1} for chunk {i + 1}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    raise ValueError(f"Failed to correct chunk {i + 1} after {MAX_RETRIES} attempts: {str(e)}")

    return "\n".join(corrected_chunks), total_cost


def correct_subtitles(
    translation_blocks: List[List[Dict]],
    source_lang: str,
    correction_instructions: str,
    model: str,
    correction_prompt: str = CORRECTION_PROMPT_TEMPLATE,
    system_prompt: str = "",
    use_context: bool = False,
    provider_params: Dict = None,
    use_thinking: bool = False,
    thinking_tokens: int = 8000,
    no_remove_subtitles: bool = False,
    max_line_length: int = 42,
) -> Tuple[List[Dict], float]:
    total_cost = 0.0
    all_corrected_subtitles = []
    previous_response = None

    prompt_template = correction_prompt

    if no_remove_subtitles:
        removal_instruction = '- "delete": Remove a subtitle (e.g., if it\'s only filler words). Provide the ID in `ids` and an empty array for `texts`.'
        no_removal_instruction = '- "delete": DO NOT USE THIS ACTION. You MUST NOT remove any subtitles.'
        prompt_template = prompt_template.replace(removal_instruction, no_removal_instruction)

    for i, block in enumerate(translation_blocks):
        base_prompt = prompt_template.format(
            correction_instructions=correction_instructions if correction_instructions else "No additional instructions provided.",
            subtitle_count=len(block),
            max_line_length=max_line_length,
        )

        subtitles = json.dumps([{"id": idx + 1, "char_count": len(sub["text"]), "text": sub["text"]} for idx, sub in enumerate(block)], ensure_ascii=False)

        if use_context and previous_response:
            context_prompt = CONTEXT_PROMPT_TEMPLATE.format(context_previous_response=previous_response)
            final_prompt = f"{base_prompt}\n{context_prompt}\n\nThe subtitles:\n{subtitles}"
        else:
            final_prompt = f"{base_prompt}\n\nThe subtitles:\n{subtitles}"

        logger.info(f"Correcting block {i + 1}/{len(translation_blocks)}")

        for attempt in range(MAX_RETRIES):
            try:
                messages = [{"role": "user", "content": final_prompt}]
                content, response = llm_api_request(
                    model=model,
                    messages=messages,
                    system_prompt=system_prompt,
                    provider_params=provider_params,
                    use_thinking=use_thinking,
                    thinking_tokens=thinking_tokens,
                    output_schema=CorrectionResponse,
                )

                cost = calculate_cost(response, model=model)
                total_cost += cost
                previous_response = content

                try:
                    correction_data = json.loads(content)
                    operations = correction_data.get("operations", [])

                    block_dict = {idx + 1: sub.copy() for idx, sub in enumerate(block)}
                    processed_ids = set()
                    new_subs_from_ops = []

                    for op in operations:
                        action = op.get("action")
                        orig_nums = op.get("ids", [])
                        texts = op.get("texts", [])

                        if not orig_nums:
                            continue

                        if any(num in processed_ids for num in orig_nums):
                            logger.warning(f"ID collision detected for {orig_nums} in action '{action}'. Skipping operation.")
                            continue

                        for num in orig_nums:
                            processed_ids.add(num)

                        valid_subs = [block_dict[n] for n in orig_nums if n in block_dict]
                        if not valid_subs:
                            continue

                        if action == "delete":
                            if no_remove_subtitles:
                                logger.warning(
                                    f"LLM attempted to delete ID {orig_nums[0]} despite --no-remove-subtitles. Reverting to original."
                                )
                                new_subs_from_ops.append((orig_nums[0], [valid_subs[0]]))
                            else:
                                new_subs_from_ops.append((orig_nums[0], []))
                        else:
                            new_start = min(s["start"] for s in valid_subs)
                            new_end = max(s["end"] for s in valid_subs)

                            if not texts:
                                merged_text = " ".join(s["text"] for s in valid_subs)
                                new_subs_from_ops.append((orig_nums[0], [{"start": new_start, "end": new_end, "text": merged_text}]))
                            else:
                                total_chars = sum(len(t) for t in texts)
                                duration = new_end - new_start

                                split_subs = []
                                current_start = new_start
                                for text in texts:
                                    ratio = len(text) / total_chars if total_chars > 0 else 1.0 / len(texts)
                                    part_duration = duration * ratio
                                    split_subs.append(
                                        {
                                            "start": current_start,
                                            "end": current_start + part_duration,
                                            "text": text,
                                        }
                                    )
                                    current_start += part_duration
                                new_subs_from_ops.append((orig_nums[0], split_subs))

                    final_block_subs = []
                    for idx in range(1, len(block) + 1):
                        if idx not in processed_ids:
                            final_block_subs.append(block_dict[idx])
                        else:
                            for op_primary_id, subs in new_subs_from_ops:
                                if idx == op_primary_id:
                                    final_block_subs.extend(subs)
                                    break

                    all_corrected_subtitles.extend(final_block_subs)
                    logger.info(f"Successfully corrected block {i + 1}")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in block {i + 1}, attempt {attempt + 1}: {str(e)}")
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Failed to parse JSON response for block {i + 1} after {MAX_RETRIES} attempts")
            except Exception as e:
                logger.error(f"Error in correction attempt {attempt + 1} for block {i + 1}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    raise ValueError(f"Failed to correct block {i + 1} after {MAX_RETRIES} attempts: {str(e)}")

    return all_corrected_subtitles, total_cost


def resegment_and_correct_with_llm(
    word_blocks: List[List[Dict]],
    model: str,
    system_prompt: str,
    correction_prompt_template: str,
    max_line_length: int,
    provider_params: Dict = None,
    use_thinking: bool = False,
    thinking_tokens: int = 8000,
) -> Tuple[List[Dict], float]:
    total_cost = 0.0
    all_llm_subtitles = []

    for i, block_data in enumerate(word_blocks):
        logger.info(f"Processing word block {i + 1}/{len(word_blocks)} for re-segmentation.")

        words_to_process = block_data["words_to_process"]
        context_words = block_data["context_words"]

        word_block_json = json.dumps(words_to_process, ensure_ascii=False, indent=2)
        context_words_json = json.dumps(context_words, ensure_ascii=False, indent=2) if context_words else "[]"

        prompt = correction_prompt_template.format(
            max_line_length=max_line_length,
            word_block_json=word_block_json,
            context_words_json=context_words_json,
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

                logger.info(f"Complete model output for resegment block {i + 1}:\n{content}")

                try:
                    subtitles_for_block = json.loads(content)

                    if words_to_process:
                        min_time = words_to_process[0]["start"]
                        max_time = words_to_process[-1]["end"]
                        time_buffer = 5.0

                        for sub in subtitles_for_block:
                            start_t = sub.get("start")
                            end_t = sub.get("end")
                            if start_t is None or end_t is None:
                                raise ValueError("LLM returned null timestamps.")

                            if not (
                                min_time - time_buffer <= start_t <= max_time + time_buffer
                                and min_time - time_buffer <= end_t <= max_time + time_buffer
                            ):
                                raise ValueError(
                                    "Timestamp outlier detected. "
                                    f"Subtitle timestamp ({start_t}-{end_t}) is outside the expected "
                                    f"block range ({min_time}-{max_time})."
                                )

                    all_llm_subtitles.extend(subtitles_for_block)
                    logger.info(f"Successfully re-segmented block {i + 1}, got {len(subtitles_for_block)} subtitles.")
                    break
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Validation failed for block {i + 1}, attempt {attempt + 1}: {e}. Retrying...")
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Failed to get a valid response from LLM for block {i + 1} after multiple retries.")
                except Exception as e:
                    logger.error(f"Error processing LLM response for block {i + 1}: {e}")
                    if attempt == MAX_RETRIES - 1:
                        raise
            except Exception as e:
                logger.error(f"LLM API call failed for block {i + 1}, attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise

    return all_llm_subtitles, total_cost


def resegment_and_translate_with_llm(
    word_blocks: List[List[Dict]],
    source_lang: str,
    target_lang: str,
    model: str,
    system_prompt: str,
    translation_prompt_template: str,
    max_line_length: int,
    provider_params: Dict = None,
    use_thinking: bool = False,
    thinking_tokens: int = 8000,
) -> Tuple[List[List[Dict]], float]:
    total_cost = 0.0
    all_llm_subtitles_grouped = []

    for i, block_data in enumerate(word_blocks):
        logger.info(f"Processing word block {i + 1}/{len(word_blocks)} for re-segmentation and translation.")

        words_to_process = block_data["words_to_process"]
        context_words = block_data["context_words"]

        word_block_json = json.dumps(words_to_process, ensure_ascii=False, indent=2)
        context_words_json = json.dumps(context_words, ensure_ascii=False, indent=2) if context_words else "[]"

        prompt = translation_prompt_template.format(
            source_lang=source_lang,
            target_lang=target_lang,
            max_line_length=max_line_length,
            word_block_json=word_block_json,
            context_words_json=context_words_json,
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

                logger.info(f"Complete model output for resegment/translate block {i + 1}:\n{content}")

                try:
                    subtitles_for_block = json.loads(content)

                    if words_to_process:
                        min_time = words_to_process[0]["start"]
                        max_time = words_to_process[-1]["end"]
                        time_buffer = 5.0

                        for sub in subtitles_for_block:
                            if not (
                                min_time - time_buffer <= sub.get("start", -1) <= max_time + time_buffer
                                and min_time - time_buffer <= sub.get("end", -1) <= max_time + time_buffer
                            ):
                                raise ValueError(
                                    "Timestamp outlier detected. "
                                    f"Subtitle timestamp ({sub.get('start')}-{sub.get('end')}) is outside the expected "
                                    f"block range ({min_time}-{max_time})."
                                )

                    all_llm_subtitles_grouped.append(subtitles_for_block)
                    logger.info(
                        f"Successfully re-segmented and translated block {i + 1}, got {len(subtitles_for_block)} subtitles."
                    )
                    break
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Validation failed for block {i + 1}, attempt {attempt + 1}: {e}. Retrying...")
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Failed to get a valid response from LLM for block {i + 1} after multiple retries.")
                except Exception as e:
                    logger.error(f"Error processing LLM response for block {i + 1}: {e}")
                    if attempt == MAX_RETRIES - 1:
                        raise
            except Exception as e:
                logger.error(f"LLM API call failed for block {i + 1}, attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise

    return all_llm_subtitles_grouped, total_cost
