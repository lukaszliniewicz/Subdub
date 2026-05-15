import os
import json
import logging
import re
import srt
from typing import List, Dict

from ..media.audio import align_audio_blocks
from ..media.ffmpeg import mix_audio_tracks
from ..subtitles.srt_utils import normalize_filename

logger = logging.getLogger(__name__)

def create_alignment_blocks(session_folder: str, video_name: str, target_language: str, evaluated: bool = False, speech_blocks_path: str = None) -> List[Dict]:
    if speech_blocks_path is None:
        speech_blocks_path = os.path.join(session_folder, f"{video_name}_speech_blocks.json")
    
    logger.info(f"Attempting to open speech blocks file: {speech_blocks_path}")
    
    try:
        with open(speech_blocks_path, 'r', encoding='utf-8') as f:
            speech_blocks = json.load(f)
    except FileNotFoundError:
        logger.error(f"Speech blocks JSON file not found: {speech_blocks_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {speech_blocks_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error opening {speech_blocks_path}: {str(e)}")
        raise

    srt_files = [f for f in os.listdir(session_folder) if f.endswith('.srt')]
    if not srt_files:
        raise FileNotFoundError("No SRT files found in the session folder")
    newest_srt = max(srt_files, key=lambda f: os.path.getmtime(os.path.join(session_folder, f)))
    srt_path = os.path.join(session_folder, newest_srt)
    logger.info(f"Using the newest SRT file: {srt_path}")

    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            subtitles = list(srt.parse(f.read()))
    except FileNotFoundError:
        logger.error(f"SRT file not found: {srt_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading SRT file {srt_path}: {str(e)}")
        raise

    alignment_blocks = []
    current_block = None

    sentence_wavs_folder = os.path.join(session_folder, 'Sentence_wavs')
    all_wav_files = os.listdir(sentence_wavs_folder)
    logger.info(f"All WAV files in directory: {all_wav_files}")

    for block in speech_blocks:
        block_subtitles = [sub for sub in subtitles if sub.index in block["subtitles"]]
        
        if block_subtitles:
            start_sub = block_subtitles[0]
            end_sub = block_subtitles[-1]
            
            block_number = int(block['number'])
            logger.info(f"Processing block number: {block_number}")
            
            wav_files = [f for f in all_wav_files 
                         if re.search(rf"_{block_number}\.wav$", f) or 
                            re.search(rf"_{block_number:04d}\.wav$", f) or
                            re.search(rf"_{block_number:d}\.wav$", f)]
            
            logger.info(f"Matched WAV files for block {block_number}: {wav_files}")
            
            if not wav_files:
                logger.warning(f"No matching WAV files found for block number {block_number}")

            new_block = {
                "number": block["number"],
                "text": block["text"],
                "start": start_sub.start,
                "end": end_sub.end,
                "audio_files": wav_files,
                "subtitles": [sub.index for sub in block_subtitles]
            }

            if current_block and current_block["subtitles"][-1] == new_block["subtitles"][0]:
                current_block["number"] += f"-{new_block['number']}"
                current_block["text"] += " " + new_block["text"]
                current_block["end"] = new_block["end"]
                current_block["audio_files"].extend(new_block["audio_files"])
                current_block["subtitles"] = list(set(current_block["subtitles"] + new_block["subtitles"]))
            else:
                if current_block:
                    alignment_blocks.append(current_block)
                    logger.info(f"Added alignment block: {json.dumps(current_block, default=str, indent=2)}")
                current_block = new_block

    if current_block:
        alignment_blocks.append(current_block)
        logger.info(f"Added final alignment block: {json.dumps(current_block, default=str, indent=2)}")

    logger.info(f"Total number of alignment blocks: {len(alignment_blocks)}")
    logger.info("First few alignment blocks:")
    for block in alignment_blocks[:3]:
        logger.info(json.dumps(block, default=str, indent=2))

    return alignment_blocks

def sync_audio_video(session_folder: str, input_video: str = None, delay_start: int = 1500, speed_up: int = 100) -> None:
    xtts_languages = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "pl": "Polish", "tr": "Turkish",
        "ru": "Russian", "nl": "Dutch", "cs": "Czech", "ar": "Arabic",
        "zh-cn": "Chinese", "ja": "Japanese", "hu": "Hungarian", "ko": "Korean"
    }

    logger.info(f"Contents of session folder: {os.listdir(session_folder)}")

    if input_video and os.path.exists(input_video):
        video_path = input_video
    else:
        video_files = [f for f in os.listdir(session_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')) and not f.startswith('final_output')]
        if not video_files:
            raise ValueError("No video file found in the session folder")
        video_path = os.path.join(session_folder, max(video_files, key=lambda f: os.path.getmtime(os.path.join(session_folder, f))))
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    speech_block_files = []
    for filename in os.listdir(session_folder):
        normalized_filename = normalize_filename(filename)
        if 'speech_blocks' in normalized_filename and (normalized_filename.endswith('.json') or '.' not in normalized_filename):
            speech_block_files.append(filename)

    logger.info(f"Found speech block files: {speech_block_files}")

    if not speech_block_files:
        raise ValueError("No speech blocks JSON file found in the session folder")
    
    if len(speech_block_files) == 1:
        json_path = os.path.join(session_folder, speech_block_files[0])
    else:
        language_specific_files = [f for f in speech_block_files if any(f"_{lang}_speech_blocks" in normalize_filename(f) for lang in xtts_languages.keys())]
        logger.info(f"Language-specific files: {language_specific_files}")
        if language_specific_files:
            json_path = os.path.join(session_folder, language_specific_files[0])
        else:
            json_path = os.path.join(session_folder, max(speech_block_files, key=lambda f: os.path.getmtime(os.path.join(session_folder, f))))
    
    logger.info(f"Using speech blocks file: {json_path}")
    logger.info(f"Full path of speech blocks file: {os.path.abspath(json_path)}")
    
    speech_blocks_language = next((lang for lang, name in xtts_languages.items() if f"_{lang}_speech_blocks" in normalize_filename(os.path.basename(json_path))), "en")
    
    alignment_blocks = create_alignment_blocks(session_folder, video_name, xtts_languages[speech_blocks_language], False, json_path)
    logger.info(f"Created {len(alignment_blocks)} alignment blocks")

    aligned_audio_path = align_audio_blocks(alignment_blocks, session_folder, delay_start, speed_up)
    logger.info(f"Audio alignment completed: {aligned_audio_path}")

    final_output_path = mix_audio_tracks(video_path, aligned_audio_path, session_folder, video_name, xtts_languages[speech_blocks_language], False)
    logger.info(f"Final output saved: {final_output_path}")
