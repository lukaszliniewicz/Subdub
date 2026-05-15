CORRECTION_PROMPT_TEMPLATE = """
Your task is to review an array of {subtitle_count} subtitles and output a list of operations to fix them.
You are an expert subtitle editor.

Instructions:
1. ONLY output operations for subtitles that require changes. If a subtitle is perfectly fine, do not include it in your response.
2. Fix punctuation and capitalization such that they are coherent and logical.
3. Correct spelling and obvious transcription errors.
4. Remove filler words (e.g., "um", "uh") and obvious repetitions.
5. You can perform the following actions:
   - "edit": Fix text in a single subtitle. Provide its ID in `ids` and the new text in `texts`.
   - "delete": Remove a subtitle (e.g., if it's only filler words). Provide the ID in `ids` and an empty array for `texts`.
   - "merge": Combine multiple sequential subtitles. Provide their IDs in `ids`. If you want a single combined subtitle, provide EXACTLY ONE string in `texts`. If you want to redistribute the combined text into new subtitle chunks (e.g., merge and then split differently), provide multiple strings in `texts`.
   - "split": Split a single subtitle into multiple parts (e.g., if it contains two distinct sentences or is too long). Provide the ID in `ids` and the new split strings in `texts`.
6. Ensure that any text you provide in `texts` (for edit, merge, or split) is fully corrected for spelling, punctuation, and grammar.
7. Subtitle formatting: Ensure subtitles are not too long (max {max_line_length} characters per line, max 2 lines per subtitle). Use a newline character `\n` to explicitly split lines within a single subtitle string.
8. If previous conversational context is provided, DO NOT include it in your output. ONLY operate on the JSON array of subtitles provided below (IDs 1 to {subtitle_count}).

Additional context and instructions specific to your particular batch, if any:
{correction_instructions}
"""


CORRECTION_PROMPT_TEMPLATE3 = """
Your Instructions:
1. Draft your response first inside <draft></draft> tags, then check it and output the actual response within <final></final> tags.
2. You will receive an array of subtitles. Your task is to correct them
3. Fix punctuation and capitalization such that they are coherent and logical, also between subsequent subtitles
4. Correct spelling and obvious transcription errors
5. Preserve all meaning and content (you should remove filler words, though)
6. If a subtitle should be removed (and that would be ONLY if it contains only OBVIOUS filler words or gibberish), replace its text with "[REMOVE]" in your output to maintain the same number of items in the array
7. Return the corrected subtitles in the EXACT SAME array structure with the SAME number of items, including subtitles that you didn't change and "[REMOVE]" statements
8. Do not assume that something needs correcting just because you were asked to consider correcting it, make sure that it really does need correcting
9. DO NOT split or merge subtitles

Additional context and instructions specific to your particular batch, if any:
{correction_instructions}

Remember, validate your output carefully before returning it. Your most important instruction: the number of corrected items in the output array MUST match the number of items received in the input array, it is IMPERATIVE.
"""


CORRECTION_EVALUATION_PROMPT_TEMPLATE = """Your task: Review and improve the correction of subtitles in {source_lang} performed by another model.

These are your instructions. Follow them closely:

1. You will receive two JSON arrays: original subtitles and their initial correction.
2. Review the corrections for accuracy, clarity, and proper language usage.
3. Improve the corrections where necessary, following the original correction guidelines.
4. THE ABSOLUTE IMPERATIVE YOU MUST ADHERE TO: Maintain the JSON array structure of the input you received and output ONLY the reviewed correction. THE NUMBER OF ITEMS IN THE ARRAY AND THE FORMATTING OF THE ARRAY MUST BE THE SAME AS IN THE ORIGINAL SUBTITLES.
5. Before outputting your answer, validate its formatting and consider all the data you were given very carefully.

Original correction guidelines:
{correction_instructions}

Below you will find:
1. The original subtitles in {source_lang} (JSON array)
2. The initial correction in {source_lang} (JSON array)
"""
