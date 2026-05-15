import os
import json
import logging
import shutil
import sys
from typing import List, Dict

from ..config import AppConfig
from ..context import SessionContext
from ..corrector.config import CorrectorConfig
from ..subtitles.srt_utils import equalize_srt
from ..ai.transcribe import transcribe_audio
from ..ai.translate import correct_transcript_chunks
from ..corrector.engine import process_whisperx_json, segments_to_srt, load_srt_file, load_audio, process_word_boundaries_at_segment_ends
from ..subtitles.zoom import parse_zoom_vtt, group_zoom_utterances, create_transcript_chunks_from_grouped
from ..prompts import CUSTOM_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def run_zoom_transcript_pipeline(config: AppConfig, context: SessionContext) -> float:
    """Runs the Zoom VTT transcript correction pipeline."""
    if not context.video_path.lower().endswith('.vtt'):
        raise ValueError("--task zoom-transcript requires a .vtt file input.")
    
    logger.info("Starting Zoom transcript processing...")
    with open(context.video_path, 'r', encoding='utf-8') as f:
        raw_utterances = parse_zoom_vtt(f)
    
    grouped_utterances = group_zoom_utterances(raw_utterances)
    logger.info(f"Processed VTT into {len(grouped_utterances)} speaker blocks.")

    transcript_chunks = create_transcript_chunks_from_grouped(grouped_utterances, config.char_limit)
    logger.info(f"Created {len(transcript_chunks)} chunks for LLM correction.")
    
    system_prompt = config.system_prompt if config.system_prompt else CUSTOM_SYSTEM_PROMPT
    
    corrected_transcript, correction_cost = correct_transcript_chunks(
        chunks=transcript_chunks,
        model=config.model,
        system_prompt=system_prompt,
        provider_params=config.provider_params
    )
    
    output_txt_path = os.path.join(context.session_folder, f"{context.video_name}_corrected_transcript.txt")
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(corrected_transcript)
        
    logger.info(f"Corrected transcript saved to: {output_txt_path}")
    if correction_cost > 0:
        logger.info(f"--- Total Estimated API Cost: ${correction_cost:.4f} ---")
    
    logger.info("Zoom transcript task completed.")
    return correction_cost

def preprocess_words_from_json(json_path: str) -> List[Dict]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read or parse JSON file at {json_path}: {e}")
        return []

    all_words = []
    for segment in data.get("segments", []):
        words = segment.get("words", [])
        if not words:
            continue
        all_words.extend(words)

    for i, word in enumerate(all_words):
        if 'start' not in word or 'end' not in word:
            logger.warning(f"Word missing timestamp: '{word.get('word', '')}'. Estimating.")
            prev_word_end = all_words[i-1]['end'] if i > 0 and 'end' in all_words[i-1] else None
            next_word_start = all_words[i+1]['start'] if i + 1 < len(all_words) and 'start' in all_words[i+1] else None
            
            if prev_word_end is not None:
                word['start'] = prev_word_end + 0.01
            elif next_word_start is not None:
                word['start'] = max(0, next_word_start - 0.2)
            else:
                word['start'] = 0

            word['end'] = word['start'] + 0.2

    for i in range(len(all_words) - 1):
        current_word = all_words[i]
        next_word = all_words[i+1]
        if 'end' in current_word and 'start' in next_word and current_word['end'] > next_word['start']:
            overlap = current_word['end'] - next_word['start']
            logger.debug(f"Correcting word timestamp overlap of {overlap:.2f}s between '{current_word['word']}' and '{next_word['word']}'")
            current_word['end'] = next_word['start']
    
    return [{'word': w.get('word'), 'start': w.get('start'), 'end': w.get('end')} for w in all_words]

def apply_word_level_boundary_correction(json_path: str, audio_path: str) -> List[Dict]:
    try:
        config = CorrectorConfig()
        audio = load_audio(audio_path, config.sample_rate)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        segments = data.get('segments', [])
        if not segments:
            logger.warning("No segments found in JSON to perform word-level correction.")
            return []

        corrected_words, corrections = process_word_boundaries_at_segment_ends(segments, audio, config)
        
        logger.info(f"Applied {len(corrections)} word-level boundary corrections at segment ends.")
        
        return [{'word': w.get('word'), 'start': w.get('start'), 'end': w.get('end')} for w in corrected_words]

    except Exception as e:
        logger.error(f"Word-level boundary correction failed: {e}. Falling back to uncorrected words.", exc_info=True)
        return preprocess_words_from_json(json_path)

def apply_boundary_correction(json_path: str, audio_path: str, session_folder: str, video_name: str, 
                            manual_correction: bool = False) -> str:
    try:
        config = CorrectorConfig()
        segments, corrections, audio, processed_json_path = process_whisperx_json(json_path, audio_path, config)
        
        logger.info(f"Automatic boundary correction: {len(corrections)} corrections applied")
        
        if manual_correction:
            try:
                from PyQt6.QtWidgets import QApplication
                from ..corrector.gui.app import BoundaryVisualizerWindow
                
                app = QApplication.instance()
                if app is None:
                    app = QApplication(sys.argv)
                
                window = BoundaryVisualizerWindow(
                    audio=audio, 
                    sr=config.sample_rate, 
                    segments=segments, 
                    corrections=corrections, 
                    config=config, 
                    json_path=processed_json_path,
                    save_folder=session_folder
                )
                
                window.show()
                window.raise_()
                window.activateWindow()
                
                logger.info("Manual correction GUI opened. Please make corrections and save as SRT to continue.")
                
                app.exec()
                
                srt_files = [f for f in os.listdir(session_folder) if f.endswith('.srt') and not f.endswith('_input.srt')]
                if srt_files:
                    latest_srt = max(srt_files, key=lambda f: os.path.getmtime(os.path.join(session_folder, f)))
                    corrected_srt_path = os.path.join(session_folder, latest_srt)
                    logger.info(f"Using manually corrected SRT: {corrected_srt_path}")
                    return corrected_srt_path
            except ImportError:
                logger.warning("PyQt6 not installed. Skipping manual correction.")
        
        corrected_srt_path = os.path.join(session_folder, f"{video_name}_corrected.srt")
        srt_content = segments_to_srt(segments)
        with open(corrected_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        logger.info(f"Automatic boundary correction completed: {corrected_srt_path}")
        
        return corrected_srt_path
        
    except Exception as e:
        logger.error(f"Boundary correction failed: {str(e)}")
        raise

def open_manual_correction_gui(srt_path: str, audio_path: str, session_folder: str) -> str:
    if not audio_path or not os.path.exists(audio_path):
        logger.warning("Manual correction requested but no valid audio file found. Skipping manual correction.")
        return srt_path
    
    try:
        from PyQt6.QtWidgets import QApplication
        from ..corrector.gui.app import BoundaryVisualizerWindow
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        segments = load_srt_file(srt_path)
        
        config = CorrectorConfig()
        audio = load_audio(audio_path, config.sample_rate)
        
        window = BoundaryVisualizerWindow(
            audio=audio, 
            sr=config.sample_rate, 
            segments=segments, 
            corrections=[], 
            config=config, 
            json_path=None,
            save_folder=session_folder
        )
        
        window.show()
        window.raise_()
        window.activateWindow()
        
        logger.info("Manual correction GUI opened for existing SRT. Please make corrections and save to continue.")
        
        app.exec()
        
        srt_files = [f for f in os.listdir(session_folder) if f.endswith('.srt') and not f.endswith('_input.srt')]
        if srt_files:
            latest_srt = max(srt_files, key=lambda f: os.path.getmtime(os.path.join(session_folder, f)))
            corrected_srt_path = os.path.join(session_folder, latest_srt)
            logger.info(f"Using manually corrected SRT: {corrected_srt_path}")
            return corrected_srt_path
        else:
            logger.warning("No SRT file found after manual correction, using original")
            return srt_path
            
    except ImportError:
        logger.warning("PyQt6 not installed. Skipping manual correction.")
        return srt_path
    except Exception as e:
        logger.error(f"Manual correction failed: {str(e)}")
        return srt_path

def cleanup_temp_files(session_folder: str, task: str) -> None:
    temp_files = ['extracted_audio.wav', 'transcription.srt', 'aligned_audio.wav']
    for file in temp_files:
        file_path = os.path.join(session_folder, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {str(e)}")
    
    if task != 'tts':
        speech_block_files = [f for f in os.listdir(session_folder) if f.startswith('speech_block_') and f.endswith('.wav')]
        for file in speech_block_files:
            file_path = os.path.join(session_folder, file)
            try:
                os.remove(file_path)
                logger.info(f"Removed speech block file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove speech block file {file_path}: {str(e)}")
