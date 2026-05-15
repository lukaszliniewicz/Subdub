RESEGMENT_CORRECTION_PROMPT_TEMPLATE = """Your task is to correct and re-segment a machine-generated transcript into professionally formatted subtitles.

You will receive a primary JSON array of word objects to process (`words_to_process`). You may also receive a secondary array of 'context words' (`next_words_context`) which follow immediately after the primary block.

Your instructions are:
1. **Correct Errors:** Fix spelling, punctuation, capitalization, and obvious transcription errors in `words_to_process` to create fluent, grammatically correct sentences. Remove filler words (e.g., "um", "uh") and unnecessary repetitions.
2. **Segment into Subtitles:** Group the corrected words from `words_to_process` into new subtitles.
3. **Follow Subtitling Rules:**
    - Each subtitle should be a logical phrase or short sentence.
    - Do not create subtitles longer than {max_line_length} characters per line.
    - Aim for a maximum of 2 lines per subtitle. Use a newline character `\n` for line breaks.
    - Ensure subtitle breaks occur at natural pauses in speech.
4. **Calculate Timestamps:** For each new subtitle you create, set its 'start' time to the EXACT 'start' float of its first word, and its 'end' time to the EXACT 'end' float of its last word. DO NOT round or estimate the timestamps.
5. **Output Format:** Your output MUST be a valid JSON array of subtitle objects, matching the provided schema. Each object must have:
    - `number`: An integer, starting from 1 for the first subtitle in this block.
    - `text`: The corrected and formatted subtitle text (string).
    - `start`: The start time in seconds (float).
    - `end`: The end time in seconds (float).

**IMPORTANT**: You must process ALL words from the `words_to_process` array. The `next_words_context` array is for context ONLY, to help you correctly punctuate and format the end of the last subtitle in your response. DO NOT include the context words in your output.

Here is the data:
Words to process:
{word_block_json}

Next words for context (do not process or include in output):
{context_words_json}
"""


RESEGMENT_EVALUATION_PROMPT_TEMPLATE = """Your task: Review and improve the correction, segmentation, and translation of a machine-generated transcript.

You will receive an array of original source words and an array of subtitles that were generated from them.

Instructions:
1.  Review the subtitles for:
    *   Translation accuracy ({source_lang} to {target_lang}).
    *   Fluency and naturalness in {target_lang}.
    *   Correct segmentation into logical phrases. Subtitle breaks should be at natural pauses.
    *   Adherence to subtitling rules (max {max_line_length} chars/line, max 2 lines).
2.  **CRITICAL INSTRUCTION**: Your output MUST be a valid JSON array containing the **full, corrected list of subtitles** for the block.
    *   You can modify the `text`, `start`, and `end` for any subtitle.
    *   To merge subtitles, combine their text into one and mark the other's text as `[REMOVE]`.
    *   You must return the same number of subtitle objects as you received.
3.  Each object in your output array must have `number`, `text`, `start`, and `end` fields.
4.  Timestamps must not overlap.

Source words ({source_lang}):
{word_block_json}

Initial translated and segmented subtitles ({target_lang}):
{initial_subtitles_json}
"""


RESEGMENT_TRANSLATION_PROMPT_TEMPLATE = """Your task is to correct, re-segment, and translate a machine-generated transcript into professionally formatted subtitles.

You will receive a primary JSON array of word objects in {source_lang} to process (`words_to_process`). You may also receive a secondary array of 'context words' (`next_words_context`) which follow immediately after the primary block.

Your instructions are:
1. **Correct and Segment (Internal Step):** First, mentally correct any spelling, punctuation, capitalization, and transcription errors in the `words_to_process` to form fluent, grammatically correct sentences in {source_lang}. Group these corrected words into logical subtitles.
2. **Translate:** Translate the subtitles you created in your internal step from {source_lang} to {target_lang}.
3. **Apply Subtitling Rules to Translation:**
    - Each translated subtitle should be a logical phrase or short sentence.
    - Do not create subtitles longer than {max_line_length} characters per line.
    - Aim for a maximum of 2 lines per subtitle. Use a newline character `\n` for line breaks.
    - Try to ensure subtitle breaks occur at natural pauses in speech.
4. **Calculate Timestamps:** For each new *translated* subtitle you create, set its 'start' time to the 'start' time of the *first original source word* that corresponds to it, and its 'end' time to the 'end' time of the *last original source word* that corresponds to it.
5. **Output Format:** Your output MUST be a valid JSON array of subtitle objects, matching the provided schema. Each object must have:
    - `number`: An integer, starting from 1 for the first subtitle in this block.
    - `text`: The final *translated* and formatted subtitle text (string) in {target_lang}.
    - `start`: The start time in seconds (float).
    - `end`: The end time in seconds (float).

**IMPORTANT**: You must process ALL words from the `words_to_process` array. The `next_words_context` array is for context ONLY, to help you correctly translate and punctuate the end of the last subtitle in your response. DO NOT include the context words in your output.

Here is the data:
Words to process ({source_lang}):
{word_block_json}

Next words for context (do not process or include in output):
{context_words_json}
"""
