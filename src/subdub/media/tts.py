import os
import logging
import requests
import io
from typing import List, Dict
from pydub import AudioSegment

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

def generate_tts_audio(speech_blocks: List[Dict], tts_voice: str, language: str, session_folder: str, video_name: str) -> List[str]:
    audio_files = []
    sentence_wavs_folder = os.path.join(session_folder, "Sentence_wavs")
    os.makedirs(sentence_wavs_folder, exist_ok=True)
    
    for block in speech_blocks:
        output_file = os.path.join(sentence_wavs_folder, f"{video_name}_{block['number']}.wav")
        
        if os.path.isfile(tts_voice):
            speaker_arg = tts_voice
        else:
            speaker_arg = os.path.basename(tts_voice)

        data = {
            "text": block["text"],
            "speaker_wav": speaker_arg,
            "language": language
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Attempting to generate TTS for block {block['number']}")
                logger.info(f"Request data: {data}")
                
                response = requests.post("http://localhost:8020/tts_to_audio/", json=data)
                logger.info(f"Response status code: {response.status_code}")
                
                if response.status_code == 200:
                    audio_data = io.BytesIO(response.content)
                    audio = AudioSegment.from_file(audio_data, format="wav")
                    audio.export(output_file, format="wav")
                    audio_files.append(output_file)
                    logger.info(f"Successfully generated TTS for block {block['number']}")
                    break
                else:
                    logger.error(f"TTS API returned status code {response.status_code} for block {block['number']}")
                    logger.error(f"Response content: {response.text}")
            except Exception as e:
                logger.error(f"Error in TTS generation attempt {attempt + 1} for block {block['number']}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to generate TTS for block {block['number']} after {MAX_RETRIES} attempts")

    return audio_files
