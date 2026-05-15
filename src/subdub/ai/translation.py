import json
import logging
from typing import Dict, List, Tuple, Union

import deepl

from .client import MAX_RETRIES, calculate_cost, llm_api_request
from ..prompts import CONTEXT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def get_deepl_language_code(language: str) -> str:
    deepl_language_map = {
        "English": "EN-US",
        "en": "EN-US",
        "German": "DE",
        "French": "FR",
        "Spanish": "ES",
        "Italian": "IT",
        "Dutch": "NL",
        "Polish": "PL",
        "Russian": "RU",
        "Portuguese": "PT-PT",
        "Chinese": "ZH",
        "Japanese": "JA",
        "Bulgarian": "BG",
        "Czech": "CS",
        "Danish": "DA",
        "Greek": "EL",
        "Estonian": "ET",
        "Finnish": "FI",
        "Hungarian": "HU",
        "Lithuanian": "LT",
        "Latvian": "LV",
        "Romanian": "RO",
        "Slovak": "SK",
        "Slovenian": "SL",
        "Swedish": "SV",
    }
    return deepl_language_map.get(language, language)


def translate_blocks_deepl(
    translation_blocks: List[List[Dict]],
    source_lang: str,
    target_lang: str,
    auth_key: str,
) -> List[Dict[str, Union[str, List[str]]]]:
    translator = deepl.Translator(auth_key)
    full_text = "\n\n".join(["\n\n".join([sub["text"] for sub in block]) for block in translation_blocks])

    if len(full_text.encode("utf-8")) > 120 * 1024:
        split_texts = []
        current_text = ""
        for block in translation_blocks:
            block_text = "\n\n".join([sub["text"] for sub in block])
            if len((current_text + "\n\n" + block_text).encode("utf-8")) > 120 * 1024:
                split_texts.append(current_text)
                current_text = block_text
            else:
                current_text += "\n\n" + block_text if current_text else block_text
        if current_text:
            split_texts.append(current_text)
    else:
        split_texts = [full_text]

    deepl_target_lang = get_deepl_language_code(target_lang)

    translated_texts = []
    for text in split_texts:
        result = translator.translate_text(text, target_lang=deepl_target_lang)
        translated_texts.append(result.text)

    full_translated_text = "\n\n".join(translated_texts)
    translated_blocks = full_translated_text.split("\n\n")

    translated_responses = []
    subtitle_index = 0
    for block in translation_blocks:
        block_translations = []
        for _ in range(len(block)):
            if subtitle_index < len(translated_blocks):
                block_translations.append(translated_blocks[subtitle_index])
                subtitle_index += 1
            else:
                break

        translated_responses.append(
            {
                "translation": block_translations,
                "new_glossary": "",
                "original_indices": [sub["index"] for sub in block],
            }
        )

    return translated_responses


def translate_blocks(
    translation_blocks: List[List[Dict]],
    source_lang: str,
    target_lang: str,
    model: str,
    glossary: Dict[str, str],
    use_translation_memory: bool,
    evaluation_enabled: bool,
    translation_prompt: str,
    translation_instructions: str,
    glossary_prompt: str,
    system_prompt: str,
    use_context: bool = False,
    no_remove_subtitles: bool = False,
    provider_params: Dict = None,
    use_thinking: bool = False,
    thinking_tokens: int = 8000,
) -> Tuple[List[Dict[str, Union[str, List[str]]]], Dict[str, str], float]:
    total_cost = 0.0
    translated_responses = []
    new_glossary = {}
    previous_response = None
    all_removed_subtitles_info = []
    total_subtitles_in_blocks = 0

    prompt_template = translation_prompt

    if no_remove_subtitles:
        removal_instruction = '5. If a subtitle should be removed (e.g., it contains only filler words or you are confident it is a hallucination of the STT model), replace its text with "[REMOVE]".'
        no_removal_instruction = "5. You MUST NOT remove any subtitles. Translate every subtitle, even if it contains filler words."
        prompt_template = prompt_template.replace(removal_instruction, no_removal_instruction)

    for i, block in enumerate(translation_blocks):
        total_subtitles_in_blocks += len(block)
        base_prompt = prompt_template.format(
            source_lang=source_lang,
            target_lang=target_lang,
            subtitle_count=len(block),
        )

        if translation_instructions:
            base_prompt += f"\n\nAdditional context and instructions:\n{translation_instructions}"

        if use_translation_memory and glossary_prompt:
            glossary_instructions = glossary_prompt.format(glossary=json.dumps(glossary, ensure_ascii=False, indent=2))
            base_prompt += f"\n\n{glossary_instructions}"

        subtitles = json.dumps([{"number": idx + 1, "text": sub["text"]} for idx, sub in enumerate(block)], ensure_ascii=False)

        final_prompt = base_prompt

        if use_context and previous_response:
            context_prompt = CONTEXT_PROMPT_TEMPLATE.format(context_previous_response=previous_response)
            final_prompt += f"\n{context_prompt}"

        if use_context and i < len(translation_blocks) - 1:
            next_block = translation_blocks[i + 1]
            next_subtitles = next_block[:2]
            if next_subtitles:
                next_subtitles_text = "\n".join([f'- "{sub["text"]}"' for sub in next_subtitles])
                future_context = (
                    f"\n\nFor additional context, here are the next {len(next_subtitles)} subtitle(s) from the following block. "
                    "DO NOT TRANSLATE THEM. They are only provided to help with continuity of the last subtitle of the current block (e.g. for punctuation).\n"
                    f"{next_subtitles_text}"
                )
                final_prompt += future_context

        final_prompt += f"\n\nThe subtitles:\n{subtitles}"

        for attempt in range(MAX_RETRIES):
            try:
                messages = [{"role": "user", "content": final_prompt}]
                content, response = llm_api_request(
                    model,
                    messages,
                    system_prompt,
                    provider_params,
                    use_thinking=use_thinking,
                    thinking_tokens=thinking_tokens,
                )

                cost = calculate_cost(response, model=model)
                total_cost += cost
                previous_response = content

                logger.info(f"Complete model output for block {i + 1}:\n{content}")

                translation_json, _, new_glossary_entries = content.partition("[GLOSSARY]")

                try:
                    translated_json = json.loads(translation_json)
                    if len(translated_json) != len(block):
                        raise ValueError("Mismatch in subtitle count")

                    for sub in translated_json:
                        if not isinstance(sub, dict) or "number" not in sub or "text" not in sub:
                            raise ValueError("Invalid subtitle format in response: missing 'number' or 'text' keys.")

                    translated_subtitles = [sub["text"] for sub in translated_json]

                    for original_sub, translated_text in zip(block, translated_subtitles):
                        if translated_text.strip() == "[REMOVE]":
                            all_removed_subtitles_info.append(f"  - Original index {original_sub['index']}: '{original_sub['text']}'")

                    translated_responses.append(
                        {
                            "translation": translated_subtitles,
                            "new_glossary": new_glossary_entries.strip() if use_translation_memory else "",
                            "original_indices": [sub["index"] for sub in block],
                        }
                    )

                    if use_translation_memory and not evaluation_enabled:
                        new_entries = dict(entry.split("=") for entry in new_glossary_entries.strip().split("\n") if "=" in entry)
                        new_glossary.update(new_entries)

                    break
                except json.JSONDecodeError:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Failed to parse JSON response for block {i + 1} after {MAX_RETRIES} attempts")
                except ValueError as e:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Error in translation for block {i + 1}: {str(e)}")
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise

    removed_count = len(all_removed_subtitles_info)
    final_subtitle_count = total_subtitles_in_blocks - removed_count
    logger.info(f"Translation finished. Total subtitles processed: {total_subtitles_in_blocks}")
    logger.info(f"Total subtitles marked for removal by model: {removed_count}")
    if removed_count > 0:
        logger.info("Removed subtitles (original text):")
        for info in all_removed_subtitles_info:
            logger.info(info)
    logger.info(f"Total subtitles after translation: {final_subtitle_count}")

    if not evaluation_enabled:
        glossary.update(new_glossary)

    return translated_responses, glossary, total_cost
