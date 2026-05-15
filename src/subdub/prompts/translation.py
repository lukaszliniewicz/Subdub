TRANSLATION_PROMPT_TEMPLATE = """Your task: translate machine-generated subtitles from {source_lang} to {target_lang}. You will receive {subtitle_count} subtitles.

Instructions:
1. You will receive an array of numbered subtitles in JSON format. Each subtitle has a "number" and "text" field.
2. Translate the "text" of each subtitle.
3. You MUST preserve the "number" field exactly as it is for each subtitle.
4. You MUST return EXACTLY {subtitle_count} subtitles in the EXACT SAME numbered format.
5. If a subtitle should be removed (e.g., it contains only filler words or you are confident it is a hallucination of the STT model), replace its text with "[REMOVE]".
6. Spell out numbers, especially Roman numerals, dates, amounts etc.
7. It is ok for a subtitle to not end in punctuation if the following subtitle continues the sentence/thought. You don't have to add "..." - in fact, don't do it. But make sure that everything is logical: if a subtitle ends with ",", for example, don't start the next one with a capital letter, unless it's a name, for example.
8. Choose concise translations suitable for dubbing while maintaining accuracy, grammatical correctness in the target language and the tone of the source.
9. Use correct punctuation that enhances a natural flow of speech for optimal speech generation.
10. Do not add ANY comments, confirmations, explanations, or questions. This is PARTICULARLY IMPORTANT: output only the translation formatted like the original JSON array.
11. Before outputting your answer, validate its formatting and consider the source text very carefully. Your most important instruction: return EXACTLY {subtitle_count} subtitles with the same structure as the input.
"""


CONTEXT_PROMPT_TEMPLATE = """
For additional context, this is the final version of the previous subtitle block processed by you before:
{context_previous_response}
"""


GLOSSARY_INSTRUCTIONS_TRANSLATION = """
Use the following glossary. Apply it flexibly, considering different forms of speech parts, like declination and conjugation. The purpuse of it is to make the translation coherent:
{glossary}

After your translation, if you identify important terms for consistent translation, add them below the [GLOSSARY] tag as 'word or phrase in source language = translated word or phrase in target language', e.g. "chowac uraze = to bear a grudge". Include only NEW entries, not ones already in the glossary.
"""


GLOSSARY_INSTRUCTIONS_EVALUATION = """
Use the following glossary. Apply it flexibly, considering different forms of speech parts, like declination and conjugation. The purpuse of it is to make the translation coherent:
{glossary}

After your evaluation, output the entire updated glossary below the [GLOSSARY] tag, including all original entries and any new or modified entries. Format each entry as in the original instructions.
"""


EVALUATION_PROMPT_TEMPLATE = """Your task: Review and improve the translation of machine-generated subtitles from {source_lang} to {target_lang} performed by another model. You must review {subtitle_count} subtitles.

These are your instructions. Follow them closely. Make sure you follow all of them, especially the ones that are emphasised:

1. You will receive two JSON arrays of numbered subtitles: original subtitles and the original translation of those subtitles. Each subtitle has a "number" (from 1 to {subtitle_count}) and "text" field.
2. Review the translation for accuracy, fluency, and suitability for dubbing. Improve the translations where necessary. It's perfectly fine to make no changes.
3. CRITICAL INSTRUCTION: Your output must be a JSON array containing ONLY the subtitles you have modified.
   - For each modified subtitle, include its original "number" and the new "text".
   - If a subtitle's translation is correct, DO NOT include it in your output.
   - If you make no changes to any subtitle, return an empty array: [].
4. For subtitles marked as "[REMOVE]":
   - If you agree it should be removed, and it was already "[REMOVE]", you don't need to include it in your output.
   - If you agree it should be removed, but it wasn't, return it with its "number" and the text "[REMOVE]".
   - If you think it should be kept (it was previously "[REMOVE]"), provide your translation in the "text" field for that "number".
5. Before outputting your answer, validate its formatting. The output must be a valid JSON array of objects, where each object has a "number" and a "text" key.

Example:
Suppose you are reviewing 3 subtitles. The original translation for subtitle number 2 is "Jestem glodnyy." and you want to correct it to "Jestem glodny.". You find subtitles 1 and 3 to be correct.
Your output should be:
[{{"number": 2, "text": "Jestem glodny."}}]

{glossary_instructions}

Original translation guidelines:
{translation_guidelines}

Below you will find:
1. The original subtitles in {source_lang} (JSON array of {subtitle_count} subtitles)
2. The initial translation in {target_lang} (JSON array of {subtitle_count} subtitles)
"""
