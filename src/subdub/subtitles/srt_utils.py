import logging
import srt
import re
from datetime import timedelta
from typing import List, Dict, Union
from srt_equalizer import srt_equalizer

logger = logging.getLogger(__name__)

import unicodedata

def normalize_filename(filename):
    return unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')

def renumber_subtitles(srt_content: str) -> str:
    """
    Renumbers subtitles in an SRT file to ensure they are consecutive.
    """
    try:
        subtitles = list(srt.parse(srt_content))
        renumbered_subtitles = []
        for i, subtitle in enumerate(subtitles, 1):
            renumbered_subtitles.append(srt.Subtitle(
                index=i,
                start=subtitle.start,
                end=subtitle.end,
                content=subtitle.content
            ))
        return srt.compose(renumbered_subtitles)
    except Exception as e:
        logger.error(f"Error while renumbering subtitles: {str(e)}")
        return srt_content

def equalize_srt(input_srt: str, output_srt: str, max_line_length: int) -> None:
    srt_equalizer.equalize_srt_file(input_srt, output_srt, max_line_length, method='punctuation')
    logger.info(f"SRT equalization completed: {output_srt}")

def merge_subtitles_with_speaker_awareness(srt_content: str, merge_threshold: int) -> tuple[str, bool]:
    """
    Merge subtitles that are within the merge threshold and have the same speaker.
    Returns the merged SRT content and whether diarization was detected.
    """
    try:
        subtitles = list(srt.parse(srt_content))
        if not subtitles:
            return srt_content, False
        
        has_diarization = any('[SPEAKER_' in subtitle.content for subtitle in subtitles)
        merged_subtitles = []
        
        for subtitle in subtitles:
            current_speaker = None
            current_content = subtitle.content
            
            if has_diarization:
                speaker_match = re.match(r'\[SPEAKER_(\d+)\]:\s*(.*)', current_content)
                if speaker_match:
                    current_speaker = speaker_match.group(1)
                    current_content = speaker_match.group(2)
            
            if (merged_subtitles and 
                (subtitle.start - merged_subtitles[-1].end).total_seconds() * 1000 <= merge_threshold):
                
                previous_content = merged_subtitles[-1].content
                previous_speaker = None
                
                if has_diarization:
                    prev_speaker_match = re.match(r'\[SPEAKER_(\d+)\]:\s*(.*)', previous_content)
                    if prev_speaker_match:
                        previous_speaker = prev_speaker_match.group(1)
                        previous_content = prev_speaker_match.group(2)
                
                if current_speaker == previous_speaker:
                    gap_ms = (subtitle.start - merged_subtitles[-1].end).total_seconds() * 1000
                    is_cjk = re.search("[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", current_content)
                    char_limit = 5 if is_cjk else 30

                    trimmed_prev_content = previous_content.strip()
                    last_char = ''
                    for char in reversed(trimmed_prev_content):
                        if char not in ')"\'':
                            last_char = char
                            break
                    
                    sentence_enders = ['.', '!', '?', '。', '！', '？', '…']
                    ends_with_sentence_ender = last_char in sentence_enders

                    can_merge = (
                        gap_ms >= 21 and
                        len(current_content.strip()) <= char_limit and
                        not ends_with_sentence_ender
                    )

                    if can_merge:
                        logger.info(f"Merging subtitle {subtitle.index} ('{current_content.strip()}') into subtitle {merged_subtitles[-1].index} ('{previous_content.strip()}').")
                        merged_content = previous_content + " " + current_content
                        if has_diarization and current_speaker:
                            merged_content = f"[SPEAKER_{current_speaker}]: {merged_content}"
                        
                        merged_subtitles[-1] = srt.Subtitle(
                            index=merged_subtitles[-1].index,
                            start=merged_subtitles[-1].start,
                            end=subtitle.end,
                            content=merged_content.strip()
                        )
                    else:
                        merged_subtitles.append(subtitle)
                else:
                    merged_subtitles.append(subtitle)
            else:
                merged_subtitles.append(subtitle)
        
        for i, subtitle in enumerate(merged_subtitles, 1):
            subtitle.index = i
        
        merged_srt_content = srt.compose(merged_subtitles)
        logger.info(f"Subtitle merging completed. Original: {len(subtitles)} subtitles, Merged: {len(merged_subtitles)} subtitles. Diarization detected: {has_diarization}")
        
        return merged_srt_content, has_diarization
        
    except Exception as e:
        logger.error(f"Error in subtitle merging: {str(e)}")
        return srt_content, False

def remove_speaker_labels(srt_content: str) -> str:
    """
    Remove speaker labels from SRT content for LLM processing.
    """
    try:
        subtitles = list(srt.parse(srt_content))
        for subtitle in subtitles:
            speaker_match = re.match(r'\[SPEAKER_\d+\]:\s*(.*)', subtitle.content)
            if speaker_match:
                subtitle.content = speaker_match.group(1)
        return srt.compose(subtitles)
    except Exception as e:
        logger.error(f"Error removing speaker labels: {str(e)}")
        return srt_content

def convert_llm_resegment_to_srt(llm_subtitles: List[Dict]) -> str:
    """
    Converts the list of subtitle dictionaries from the LLM into an SRT formatted string.
    """
    srt_subtitles = []
    for i, sub_data in enumerate(llm_subtitles, 1):
        try:
            start_seconds = sub_data['start']
            end_seconds = sub_data['end']

            if start_seconds >= end_seconds:
                logger.warning(f"Subtitle at index {i} has start time >= end time ({start_seconds} >= {end_seconds}). Correcting end time.")
                end_seconds = start_seconds + 0.1

            start_time = timedelta(seconds=start_seconds)
            end_time = timedelta(seconds=end_seconds)
            content = sub_data['text']
            
            srt_sub = srt.Subtitle(
                index=i,
                start=start_time,
                end=end_time,
                content=content
            )
            srt_subtitles.append(srt_sub)
        except (KeyError, TypeError) as e:
            logger.warning(f"Skipping malformed subtitle object from LLM: {sub_data}. Error: {e}")
            continue

    return srt.compose(srt_subtitles)

def parse_deepl_response(translated_blocks: List[Dict[str, Union[str, List[str]]]], original_srt: str) -> str:
    original_subtitles = list(srt.parse(original_srt))
    translated_subtitles = []
    
    for block in translated_blocks:
        for translated_text, orig_index in zip(block['translation'], block['original_indices']):
            orig = original_subtitles[orig_index - 1]
            translated_subtitles.append(srt.Subtitle(
                index=orig_index,
                start=orig.start,
                end=orig.end,
                content=translated_text.strip()
            ))
    
    translated_subtitles.sort(key=lambda x: x.index)
    return srt.compose(translated_subtitles)

def parse_translated_response(translated_blocks: List[Dict[str, Union[str, List[str]]]], original_srt: str) -> str:
    original_subtitles = list(srt.parse(original_srt))
    translated_subtitles = []
    
    for block in translated_blocks:
        for subtitle_text, orig_index in zip(block['translation'], block['original_indices']):
            if subtitle_text.strip() != '[REMOVE]':
                orig = original_subtitles[orig_index - 1]
                translated_subtitles.append(srt.Subtitle(
                    index=len(translated_subtitles) + 1,
                    start=orig.start,
                    end=orig.end,
                    content=subtitle_text.strip()
                ))

    return srt.compose(translated_subtitles)
