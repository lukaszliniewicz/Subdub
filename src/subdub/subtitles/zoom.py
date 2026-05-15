import re
from typing import List, Dict

def parse_zoom_vtt(file_handle) -> List[Dict[str, str]]:
    """
    Parses a Zoom VTT file to extract speaker and dialogue, ignoring timestamps.
    """
    speaker_line_re = re.compile(r'^\s*([^:]+):\s*(.*)')
    timestamp_re = re.compile(r'^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}')
    sequence_num_re = re.compile(r'^\d+$')

    utterances = []
    for line in file_handle:
        line = line.strip()
        if not line or timestamp_re.match(line) or sequence_num_re.match(line) or "WEBVTT" in line:
            continue

        match = speaker_line_re.match(line)
        if match:
            speaker, text = match.groups()
            utterances.append({'speaker': speaker.strip(), 'text': text.strip()})
    return utterances

def group_zoom_utterances(utterances: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Groups consecutive utterances from the same speaker.
    """
    if not utterances:
        return []

    grouped = []
    current_speaker = utterances[0]['speaker']
    current_text_parts = [utterances[0]['text']]

    for utterance in utterances[1:]:
        speaker = utterance['speaker']
        text = utterance['text']

        if speaker == current_speaker:
            current_text_parts.append(text)
        else:
            grouped.append({
                'speaker': current_speaker,
                'text': ' '.join(current_text_parts)
            })
            current_speaker = speaker
            current_text_parts = [text]

    grouped.append({
        'speaker': current_speaker,
        'text': ' '.join(current_text_parts)
    })
    return grouped

def create_transcript_chunks_from_grouped(grouped_utterances: List[Dict[str, str]], char_limit: int) -> List[str]:
    """
    Creates text chunks from grouped utterances for LLM processing.
    """
    chunks = []
    current_chunk_parts = []
    current_char_count = 0

    for utterance in grouped_utterances:
        speaker_line = f"{utterance['speaker']}:"
        text_line = utterance['text']
        block_len = len(speaker_line) + len(text_line) + 3

        if current_chunk_parts and (current_char_count + block_len > char_limit):
            chunks.append('\n'.join(current_chunk_parts))
            current_chunk_parts = []
            current_char_count = 0
        
        current_chunk_parts.append(speaker_line)
        current_chunk_parts.append(text_line)
        current_chunk_parts.append('')
        current_char_count += block_len

    if current_chunk_parts:
        chunks.append('\n'.join(current_chunk_parts))
    
    return chunks
