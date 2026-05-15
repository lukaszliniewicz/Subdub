ZOOM_CORRECTION_PROMPT_TEMPLATE = """Your task is to correct a machine-generated transcript of a meeting.

Instructions:
1. You will receive a chunk of the transcript. It contains speaker labels (e.g., "John Doe:").
2. **Preserve the speaker labels and the overall format exactly as they are.** Do not add, remove, or modify speaker labels.
3. Correct spelling, punctuation, capitalization, and obvious transcription errors to create fluent, grammatically correct sentences.
4. Remove filler words (e.g., "um", "uh") and unnecessary repetitions.
5. Ensure the text flows logically. You can merge or split paragraphs under the same speaker if it improves readability, but do not move text between different speakers.
6. Return ONLY the corrected transcript text. Do not add any comments, explanations, or introductory phrases. Your output should start directly with the first speaker's name.

Here is the transcript chunk to correct:
{transcript_chunk}
"""
