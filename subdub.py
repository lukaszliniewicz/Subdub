import argparse
import os
import json
import subprocess
import datetime
import logging
from typing import List, Dict, Tuple, Union
import requests
import io
from pydub import AudioSegment
import srt
from anthropic import Anthropic
from typing import Dict
from datetime import timedelta
from pydub.silence import detect_silence
import re
import math
from openai import OpenAI
import shutil
import yt_dlp
from srt_equalizer import srt_equalizer
import unicodedata
import deepl
from google import genai
from google.generativeai import types

#Ideas
#Add numbering to subtitles
#Write app to compare / edit subtitles and their translation, including splitting, removing with renumbering, consider including in Pandrator flow?
#Improve glossary workflow, enable passing a csv file
#Improve handling of Japanese subtitles (character limit etc.)
#Improve prompts, make TTS optimisation optional
#Add option to only consider select conjunctions as valid split points or a minimum gap between words.

# Constants
MAX_RETRIES = 3
CHAR_LIMIT_DEFAULT = 2000
SPEECH_BLOCK_CHAR_LIMIT = 160
SPEECH_BLOCK_MIN_CHARS = 10
SPEECH_BLOCK_MERGE_THRESHOLD = 1  # ms
DEFAULT_WHISPER_PROMPT = "Hello, welcome to this presentation. This is a professional recording with clear speech, proper punctuation, and standard grammar."


# Configuration
TRANSLATION_PROMPT_TEMPLATE = """Your task: translate machine-generated subtitles from {source_lang} to {target_lang}. 

Instructions:
1. You will receive an array of subtitles in JSON format.
2. Translate each subtitle, maintaining the EXACT SAME array structure.
3. If a subtitle should be removed (e.g., it contains only filler words or you are confident it is a hallucination of the STT model), replace its text with "[REMOVE]".
4. Spell out numbers, especially Roman numerals, dates, amounts etc.
5. Write names, brands, acronyms, abbreviations, and foreign words phonetically in the target language.
6. Choose concise translations suitable for dubbing while maintaining accuracy, grammatical corectness in the target language and the tone of the source.
7. Use correct punctuation that enhances a natural fow of speech for optimal speech generation.
8. Do not add ANY comments, confirmations, explanations, or questions. This is PARTICULARLY IMPORTANT: output only the translation formatted like the original JSON array. Do not change the format. Do not add unneccesary comments or remarks.
10. Before outputting your answer, validate its formatting and consider the source text very carefully. 
"""

TRANSLATION_PROMPT_TEMPLATE_COT = """Your task: translate machine-generated subtitles from {source_lang} to {target_lang}. Draft your response first inside <draft></draft> tags, then analyse it inside <analysis></analysis> tags, paying attention especially to whether the number of output subtitles matches and/or [REMOVE] tags matches the number of input subtitles, and output the validated response within <final></final> tags. Make sure to close all tags.

Instructions:
1. You will receive an array of subtitles in JSON format.
2. Translate each subtitle, maintaining the EXACT SAME array structure.
3. If a subtitle should be removed (e.g., it contains only filler words or you are confident it is a hallucination of the STT model), replace its text with "[REMOVE]".
4. Spell out numbers, especially Roman numerals, dates, amounts etc.
5. Write names, brands, acronyms, abbreviations, and foreign words phonetically in the target language.
6. Choose concise translations suitable for dubbing while maintaining accuracy, grammatical corectness in the target language and the tone of the source.
7. Use correct punctuation that enhances a natural fow of speech for optimal speech generation.
8. Do not add ANY comments, confirmations, explanations, or questions. This is PARTICULARLY IMPORTANT: output only the translation formatted like the original JSON array. Do not change the format. Do not add unneccesary comments or remarks.
10. Before outputting your answer, validate its formatting and consider the source text very carefully. 
"""

CONTEXT_PROMPT_TEMPLATE = """
For additional context, this is the final vrsion of the previous susbtitle block processed by you before:
{context_previous_response}
"""

GLOSSARY_INSTRUCTIONS_TRANSLATION = """
Use the following glossary. Apply it flexibly, considering different forms of speech parts, like declination and conjugation. The purpuse of it is to make the translation coherent:
{glossary}

After your translation, if you identify important terms for consistent translation, add them below the [GLOSSARY] tag as 'word or phrase in source language = translated word or phrase in target language', e.g. "chować urazę = to bear a grudge". Include only NEW entries, not ones already in the glossary.
"""

GLOSSARY_INSTRUCTIONS_EVALUATION = """
Use the following glossary. Apply it flexibly, considering different forms of speech parts, like declination and conjugation. The purpuse of it is to make the translation coherent:
{glossary}

After your evaluation, output the entire updated glossary below the [GLOSSARY] tag, including all original entries and any new or modified entries. Format each entry as in the original instructions.
"""
EVALUATION_PROMPT_TEMPLATE = """Your task: Review and improve the translation of machine-generated subtitles from {source_lang} to {target_lang} performed by another model. 

These are your instructions. Follow them closely. Make sure you follow all of them, especially the ones that are emphasised:

1. You will receive two JSON arrays: original subtitles and the original translation of those subtitles.
2. Review the translation for accuracy, fluency, and suitability for dubbing.
3. Improve the translations where necessary. Be guided by the original translation instructions. Do not find faults where there are none. It's perfectly fine to output the original translated subtaites with no changes.
4. THE ABSOLUTE IMPERATIVE YOU MUST ADHERE TO: Maintain the JSON array structure of the input you received and output ONLY the reviewed translation. THE NUMBER OF ITEMS IN THE ARRAY AND THE FORMATTING OF THE ARRAY MUST BE THE SAME AS IN THE ORIGINAL SUBTITLES. If the original subtitle array contains 7 items, you must output 7 subtitles; if it contains 8, you must output 8. This is crucial to the success of your task. You don't outpu the original subtitle, JUST THE REVIEWD TRANSLATION.
5. For subtitles marked as "[REMOVE]":
   - If you agree it should be removed, keep "[REMOVE]" in your output for that subtitle.
   - If you think it should be kept, provide your translation instead of "[REMOVE]".
6. Before outputting your answer, validate its formatting and consider all the data you were given very carefully. 

Reminder: adhere to the formatting of your output as JSON and output the exact same number of subtitles. Example: Original subtitles: ["I am hungry.", "So am I."]; Original translation: ["Jestem głodny.", "Ja teeeż."]; Your output: ["Jestem głodny.", "Ja też."]

{glossary_instructions}

Original translation guidelines:
{translation_guidelines}

Below you will find:
1. The original subtitles in {source_lang} (JSON array)
2. The initial translation in {target_lang} (JSON array)
"""

CORRECTION_PROMPT_TEMPLATE = """
Your Instructions:
1. You will receive an array of {subtitle_count} numbered subtitles. Each subtitle has a "number" and "text" field.
2. Fix punctuation and capitalization such that they are coherent and logical, also between subsequent subtitles.
3. Correct spelling and obvious transcription errors.
4. Preserve all meaning and content, also stylistic phrases, even if not key to the meaning (you should remove filler words - like "um" - and obvious repetitions - like "it is, it is..." though).
5. You MUST preserve the "number" field exactly as it is for each subtitle.
6. You MUST return EXACTLY {subtitle_count} subtitles in the EXACT SAME numbered format.
7. Only modify the "text" field of each subtitle.
8. Do not assume that something needs correcting just because you were asked to consider correcting it.
9. DO NOT split or merge subtitles.

Example input:
[
  {{"number": 1, "text": "Hello world."}},
  {{"number": 2, "text": "how are you today"}}
]

Example output:
[
  {{"number": 1, "text": "Hello world."}},
  {{"number": 2, "text": "How are you today?"}}
]

Additional context and instructions specific to your particular batch, if any:
{correction_instructions}

Remember, validate your output carefully before returning it. Your most important instruction: return EXACTLY {subtitle_count} subtitles with the same structure as the input, with each subtitle keeping its original "number" field. If the number of the last output subtitle is not equal to {subtitle_count}, something is wrong with your output.
"""

CORRECTION_PROMPT_TEMPLATE_COT = """
You will receive an array of {subtitle_count} numbered subtitles (sub_numb). Your task is to correct them. You MUST output EXACTLY {subtitle_count} subtitles.

1. Fix punctuation and capitalization such that they are coherent and logical, also between subsequent subtitles.
2. Correct spelling and obvious transcription errors.
3. Preserve all meaning and content, also stylistic phrases, even if not key to the meaning (you should remove filler words - like "um" - and obvious repetitions - like "it is, it is..." though).
4. Each subtitle has a "number" and "text" field. You MUST preserve the "number" field exactly as it is.
5. You MUST return subtitles in the EXACT SAME numbered format with the SAME number of items ({subtitle_count} items).
6. Do not assume that something needs correcting just because you were asked to consider correcting it.
7. DO NOT split or merge subtitles.

Draft your response first inside <draft></draft> tags.
Check your draft inside <analysis></analysis> tags, especially whether the number of output subtitles matches the number of input subtitles ({subtitle_count}) and identify problems or possible improvement. If the id of the last output subtitle is not equal to {subtitle_count}, try again.
Implement other improvements you identified and output the final response within <final></final> tags.

Example input:
[
  {{"number": 1, "text": "Hello world."}},
  {{"number": 2, "text": "how are you today"}}
]

Example output:
[
  {{"number": 1, "text": "Hello world."}},
  {{"number": 2, "text": "How are you today?"}}
]

Additional context and instructions specific to your particular batch, if any:
{correction_instructions}

Remember, validate your output carefully before returning it. Your most important instruction: return EXACTLY {subtitle_count} subtitles with the same structure as the input, with each subtitle keeping its original "number" field.
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

CUSTOM_SYSTEM_PROMPT = "You are an experienced translator and text editor proficient in multiple languages. You pay great attention to detail and analyse the instructions you are given very carefully. You return ONLY the translation or corrected text requested of you in the requested format (a JSON array formatted like the input you receive), without any comments, acknowledgments, remarks, questions etc."

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def renumber_subtitles(srt_content: str) -> str:
    """
    Renumbers subtitles in an SRT file to ensure they are consecutive.
    
    Args:
        srt_content: The content of the SRT file as a string
        
    Returns:
        A string containing the SRT content with consecutive subtitle numbers
    """
    try:
        # Parse the SRT content
        subtitles = list(srt.parse(srt_content))
        
        # Create a new list of subtitles with consecutive indices
        renumbered_subtitles = []
        for i, subtitle in enumerate(subtitles, 1):
            renumbered_subtitles.append(srt.Subtitle(
                index=i,
                start=subtitle.start,
                end=subtitle.end,
                content=subtitle.content
            ))
        
        # Recompose the SRT content
        return srt.compose(renumbered_subtitles)
    except Exception as e:
        logging.error(f"Error while renumbering subtitles: {str(e)}")
        # Return the original content if renumbering fails
        return srt_content

def download_from_url(url: str, session_folder: str) -> Tuple[str, str]:
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Prefer MP4, but fall back to best available format
        'outtmpl': {
            'default': os.path.join(session_folder, '%(title)s.%(ext)s')
        },
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_title = info['title']
        sanitized_title = ''.join(e for e in video_title if e.isalnum() or e in ['-', '_', ' '])
        ydl_opts['outtmpl']['default'] = os.path.join(session_folder, f'{sanitized_title}.%(ext)s')
        
        ydl.download([url])
        
    downloaded_files = os.listdir(session_folder)
    video_file = next((f for f in downloaded_files if f.startswith(sanitized_title)), None)
    
    if not video_file:
        raise FileNotFoundError("Downloaded video file not found in the session folder.")
    
    return os.path.join(session_folder, video_file), sanitized_title

def equalize_srt(input_srt: str, output_srt: str, max_line_length: int) -> None:
    srt_equalizer.equalize_srt_file(input_srt, output_srt, max_line_length, method='punctuation')
    logging.info(f"SRT equalization completed: {output_srt}")

def llm_api_request(client, llm_api: str, model: str, messages: List[Dict[str, str]], 
                   system_prompt: str = "", provider_params: Dict = None,
                   use_thinking: bool = False, thinking_tokens: int = 8000,
                   gemini_api_key: str = None) -> str:
    """
    Send a request to a language model API and process the response.
    """
    try:
        # Print a divider for better readability
        print("\n" + "="*80)
        print("SENDING REQUEST TO LLM API")
        print("=" * 80)
        
        # Only print the last message to avoid redundancy
        last_message = messages[-1] if messages else {"role": "none", "content": "No message"}
        print(f"\nLAST MESSAGE:\n{json.dumps(last_message, ensure_ascii=False, indent=2)}")
        
        if system_prompt and llm_api != 'openrouter':
            print(f"\nSYSTEM PROMPT:\n{system_prompt}")
        
        # Initialize content to empty string
        content = ""
        
        if llm_api == "anthropic":
            user_messages = [msg for msg in messages if msg['role'] == 'user']
            
            # Build request parameters
            request_params = {
                "model": model,
                "max_tokens": 8000,
                "temperature": 1,
                "system": system_prompt,
                "messages": user_messages
            }
            
            # Add thinking parameter if requested and using Sonnet
            if use_thinking and "sonnet" in model:
                try:
                    print(f"\nENABLING EXTENDED THINKING\nBudget: {thinking_tokens} tokens")
                    request_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_tokens
                    }
                except Exception as e:
                    print(f"Warning: Could not enable thinking: {str(e)}")
            
            # Make the API request with a timeout of 15 minutes
            print("\nSending request to Anthropic API...")
            response = client.with_options(timeout=900.0).messages.create(**request_params)
            
            # Process and log the response
            print("\n" + "-"*80)
            print("RECEIVED ANTHROPIC API RESPONSE")
            print("-" * 80)
            
            # Extract all thinking blocks and log them
            thinking_content = []
            text_content = []
            
            if hasattr(response, 'content') and isinstance(response.content, list):
                for block in response.content:
                    if hasattr(block, 'type'):
                        if block.type == "thinking":
                            thinking_content.append(block.thinking)
                        elif block.type == "text":
                            text_content.append(block.text)
                        elif block.type == "redacted_thinking":
                            thinking_content.append("[REDACTED THINKING BLOCK]")
            
            # Log thinking content if available
            if thinking_content:
                print("\nTHINKING PROCESS:")
                print("-" * 80)
                for i, thinking in enumerate(thinking_content):
                    print(f"Thinking Block {i+1}:")
                    print(thinking)
                    print("-" * 40)
            
            # Extract and combine all text blocks for the final content
            if text_content:
                content = "\n".join(text_content)
                print("\nFINAL RESPONSE:")
                print("-" * 80)
                print(content)
            else:
                # Fallback in case we don't have text blocks
                content = ""
                print("\nWARNING: No text content found in response")
            
        elif llm_api == "openai":
            openai_messages = messages
            if system_prompt:
                openai_messages = [{"role": "system", "content": system_prompt}] + openai_messages
            
            print("\nSending request to OpenAI API...")
            response = client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=8000
            )
            
            content = response.choices[0].message.content
            print("\nOPENAI API RESPONSE:")
            print("-" * 80)
            print(content)
            
        elif llm_api == "openrouter":
            import requests
            
            openrouter_api = os.environ.get('OPENROUTER_API')
            headers = {
                "Authorization": f"Bearer {openrouter_api}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Subtitle Translation App",
                "Content-Type": "application/json"
            }
            
            # Build request body
            request_body = {
                "model": model,
                "messages": messages,
                "max_tokens": 32000
            }
            
            # Add provider parameters if specified
            if provider_params:
                request_body["provider"] = provider_params
            
            print("\nSending request to OpenRouter API...")
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=request_body
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
            else:
                error_msg = f"OpenRouter API returned status code {response.status_code}: {response.text}"
                logging.error(error_msg)
                raise Exception(error_msg)
            
            print("\nOPENROUTER API RESPONSE:")
            print("-" * 80)
            print(content)
            
        elif llm_api == "gemini":
            try:
                from google import genai
                
                # Create a basic client with API key
                gemini_client = genai.Client(api_key=gemini_api_key)
                
                print("\nSending request to Gemini API...")
                
                # Get content from the last message
                user_content = messages[-1]["content"]
                
                # Handle system prompt separately if provided
                if system_prompt:
                    # Simply prepend system instruction to the user content
                    formatted_prompt = f"System instruction: {system_prompt}\n\nUser request: {user_content}"
                    response = gemini_client.models.generate_content(
                        model=model,
                        contents=formatted_prompt
                    )
                else:
                    # Basic request without system prompt
                    response = gemini_client.models.generate_content(
                        model=model,
                        contents=user_content
                    )
                
                # Extract text from the response
                content = response.text
                
                print("\nGEMINI API RESPONSE:")
                print("-" * 80)
                print(content)
            except Exception as e:
                error_msg = f"Gemini API error: {str(e)}"
                logging.error(error_msg)
                raise Exception(error_msg)
            
        elif llm_api == "local":
            url = "http://127.0.0.1:5000/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            data = {
                "mode": "instruct",
                "max_new_tokens": 1500,
                "temperature": 0.4,
                "top_p": 0.9,
                "min_p": 0,
                "top_k": 20,
                "repetition_penalty": 1.15,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "typical_p": 1,
                "tfs": 1,
                "mirostat_mode": 0,
                "mirostat_tau": 5,
                "mirostat_eta": 0.1,
                "seed": -1,
                "truncate": 2500,
                "messages": messages
            }
            
            print("\nSending request to Local API...")
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            
            print("\nLOCAL API RESPONSE:")
            print("-" * 80)
            print(content)
            
        else:
            raise ValueError(f"Unsupported LLM API: {llm_api}")
        
        # Process the content for special formats
        print("\nPROCESSING RESPONSE...")
        
        # First check for <final> tags (for Chain of Thought workflow)
        if '<final>' in content and '</final>' in content:
            final_start = content.find('<final>') + len('<final>')
            final_end = content.find('</final>')
            if final_start != -1 and final_end != -1:
                content = content[final_start:final_end].strip()
                print("\nExtracted <final> content:")
                print(content)
        
        # Then check for JSON array format
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
            print("\nExtracted JSON array:")
            print(content)
        
        print("\n" + "="*80)  # Closing separator
        return content
        
    except Exception as e:
        logging.error(f"Error in LLM API request: {str(e)}")
        raise

def safe_decode(byte_string):
    try:
        return byte_string.decode('utf-8')
    except UnicodeDecodeError:
        return byte_string.decode('utf-8', errors='ignore')

def setup_logging(session_folder: str) -> None:
    log_file = os.path.join(session_folder, 'subtitle_app.log')
    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def get_or_create_session_folder(video_name: str, session: str = None) -> str:
    if session:
        if os.path.isabs(session):
            session_folder = session
        else:
            session_folder = os.path.abspath(session)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = f"subtitle_session_{video_name}_{timestamp}"
        session_folder = os.path.abspath(session_folder)
    
    os.makedirs(session_folder, exist_ok=True)
    logging.info(f"Using session folder: {session_folder}")
    
    return session_folder

def extract_audio(video_path: str, session_folder: str, video_name: str) -> str:
    audio_path = os.path.join(session_folder, f"{video_name}.wav")
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # 16-bit PCM
        '-ar', '16000',  # 16kHz (Whisper's training rate)
        '-ac', '1',  # Mono
        '-af', 'aresample=resampler=soxr:precision=28:cheby=1,loudnorm',  # High-quality resampling + normalization
        '-y',  # Overwrite output files
        audio_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
        if result.stderr:
            logging.warning(f"FFmpeg warning: {safe_decode(result.stderr)}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed. Error output:\n{safe_decode(e.stderr)}")
        raise
    return audio_path

def transcribe_audio(audio_path: str, language: str, session_folder: str, video_name: str, whisper_model: str, align_model: str = None, initial_prompt: str = None) -> str:
    output_srt = os.path.join(session_folder, f"{video_name}.srt")
    base_whisperx_args = [
        audio_path,
        '--model', whisper_model,
        '--language', language,
        '--output_format', 'srt',
        '--output_dir', session_folder,
        '--print_progress', 'True' 
    ]

    # Add initial_prompt if provided
    if initial_prompt:
        base_whisperx_args.extend(['--initial_prompt', initial_prompt])
        logging.info(f"WhisperX will use initial prompt: {initial_prompt}")

    # Add align_model argument if provided
    if align_model:
        base_whisperx_args.extend(['--align_model', align_model])
        logging.info(f"WhisperX will use alignment model: {align_model}")
    else:
        logging.info("WhisperX will use its default alignment model for the specified language (if any).")



    # First attempt with direct whisperx command
    try:
        whisperx_command = ['whisperx'] + base_whisperx_args
        logging.info(f"Attempting direct whisperx command: {' '.join(whisperx_command)}")
        result = subprocess.run(whisperx_command, check=True, capture_output=True)
        if result.stderr:
            logging.warning(f"WhisperX warning: {safe_decode(result.stderr)}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.warning(f"Direct whisperx command failed, trying conda run method. Error: {str(e)}")
        try:
            conda_whisperx_command = [
                "../conda/Scripts/conda.exe", "run", "-p", "../conda/envs/whisperx_installer", "--no-capture-output",
                "python", "-m", "whisperx"
            ] + base_whisperx_args # base_whisperx_args now includes align_model if set
            logging.info(f"Attempting conda run whisperx command: {' '.join(conda_whisperx_command)}")
            result = subprocess.run(conda_whisperx_command, check=True, capture_output=True)
            if result.stderr:
                logging.warning(f"WhisperX warning: {safe_decode(result.stderr)}")
        except subprocess.CalledProcessError as e_conda:
            logging.error(f"WhisperX command failed using both methods.")
            if isinstance(e, subprocess.CalledProcessError):
                 logging.error(f"Direct WhisperX Error output:\n{safe_decode(e.stderr)}")
            logging.error(f"Conda WhisperX Error output:\n{safe_decode(e_conda.stderr)}")
            raise e_conda
    
    whisperx_output_filename_base = os.path.splitext(os.path.basename(audio_path))[0]
    whisperx_generated_srt_path = os.path.join(session_folder, f"{whisperx_output_filename_base}.srt")
    
    if os.path.exists(whisperx_generated_srt_path):
        os.rename(whisperx_generated_srt_path, output_srt)
    else:
        # Fallback to find any .srt file generated for this audio in the session_folder
        potential_files = [f for f in os.listdir(session_folder) if f.startswith(whisperx_output_filename_base) and f.endswith(".srt")]
        if potential_files:
            actual_whisperx_output = os.path.join(session_folder, potential_files[0])
            logging.warning(f"Expected WhisperX SRT file not found at {whisperx_generated_srt_path}. Found and using: {actual_whisperx_output}")
            os.rename(actual_whisperx_output, output_srt)
        else:
            raise FileNotFoundError(f"WhisperX did not produce the expected SRT file. Looked for {whisperx_generated_srt_path} and similar pattern.")
            
    return output_srt

def create_translation_blocks(srt_content: str, char_limit: int, source_language: str) -> List[List[Dict]]:
    # Adjust character limit for Chinese and Japanese
    if source_language.lower() in ['chinese', 'japanese', 'ja', 'zh', 'zh-cn', 'zh-tw']:
        char_limit = math.floor(char_limit / 3)

    # Define sentence-ending patterns for different languages
    sentence_endings = {
        'default': ('.', '!', '?'),
        'japanese': ('。', '！', '？', 'か', 'ね', 'よ', 'わ'),
        'chinese': ('。', '！', '？', '…')
    }

    if source_language.lower() in ['japanese', 'ja']:
        endings = sentence_endings['japanese']
    elif source_language.lower() in ['chinese', 'zh', 'zh-cn', 'zh-tw']:
        endings = sentence_endings['chinese']
    else:
        endings = sentence_endings['default']

    subtitles = list(srt.parse(srt_content))
    blocks = []
    current_block = []
    current_char_count = 0

    def is_sentence_ending(text: str) -> bool:
        return any(text.strip().endswith(ending) for ending in endings)

    for i, subtitle in enumerate(subtitles):
        if current_char_count + len(subtitle.content) > char_limit:
            # We've exceeded the limit, so we need to find a good breaking point
            if is_sentence_ending(current_block[-1]['text']):
                # The last subtitle that fits ends with sentence-ending punctuation or word
                blocks.append(current_block)
            else:
                # Look backwards for the last subtitle with sentence-ending punctuation or word
                for j in range(len(current_block) - 1, -1, -1):
                    if is_sentence_ending(current_block[j]['text']):
                        blocks.append(current_block[:j+1])
                        current_block = current_block[j+1:]
                        break
                else:
                    # If no sentence-ending found, use all accumulated subtitles
                    blocks.append(current_block)

            # Start a new block
            current_block = []
            current_char_count = 0

        # Add the current subtitle to the block
        current_block.append({"index": subtitle.index, "text": subtitle.content})
        current_char_count += len(subtitle.content)

    # Add any remaining subtitles as the last block
    if current_block:
        blocks.append(current_block)

    return blocks

def translate_blocks(translation_blocks: List[List[Dict]], source_lang: str, target_lang: str,
                    anthropic_api_key: str, openai_api_key: str, llm_api: str, glossary: Dict[str, str],
                    use_translation_memory: bool, evaluation_enabled: bool, model: str,
                    translation_prompt: str, translation_instructions: str, glossary_prompt: str, 
                    system_prompt: str, use_cot: bool = False, use_context: bool = False,
                    provider_params: Dict = None, use_thinking: bool = False, 
                    thinking_tokens: int = 8000, gemini_api_key: str = None) -> Tuple[List[Dict[str, Union[str, List[str]]]], Dict[str, str]]:
    
    translated_responses = []
    new_glossary = {}
    previous_response = None

    client = None
    if llm_api == "anthropic":
        client = Anthropic(api_key=anthropic_api_key)
    elif llm_api == "openai":
        client = OpenAI(api_key=openai_api_key)
    elif llm_api == "openrouter":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get('OPENROUTER_API')
        )
    elif llm_api == "gemini":
        client = genai.Client(api_key=gemini_api_key)
    
    # Build the base prompt
    base_prompt = (TRANSLATION_PROMPT_TEMPLATE_COT if use_cot else TRANSLATION_PROMPT_TEMPLATE).format(
        source_lang=source_lang,
        target_lang=target_lang,
    )

    # Add custom translation instructions if provided
    if translation_instructions:
        base_prompt += f"\n\nAdditional context and instructions:\n{translation_instructions}"

    # Add glossary instructions only if translation memory is enabled
    if use_translation_memory and glossary_prompt:
        glossary_instructions = glossary_prompt.format(glossary=json.dumps(glossary, ensure_ascii=False, indent=2))
        base_prompt += f"\n\n{glossary_instructions}"

    for i, block in enumerate(translation_blocks):
        subtitles = json.dumps([sub['text'] for sub in block])
        
        if use_context and previous_response:
            context_prompt = CONTEXT_PROMPT_TEMPLATE.format(
                context_previous_response=previous_response
            )
            final_prompt = f"{base_prompt}\n{context_prompt}\n\nThe subtitles:\n{subtitles}"
        else:
            final_prompt = f"{base_prompt}\n\nThe subtitles:\n{subtitles}"

        for attempt in range(MAX_RETRIES):
            try:
                messages = [{"role": "user", "content": final_prompt}]
                content = llm_api_request(client, llm_api, model, messages, system_prompt, provider_params, 
                         use_thinking=use_thinking, thinking_tokens=thinking_tokens)
                previous_response = content
                
                logging.info(f"Complete model output for block {i+1}:\n{content}")
                
                translation_json, _, new_glossary_entries = content.partition("[GLOSSARY]")
                
                try:
                    translated_subtitles = json.loads(translation_json)
                    if len(translated_subtitles) != len(block):
                        raise ValueError("Mismatch in subtitle count")
                    
                    translated_responses.append({
                        "translation": translated_subtitles,
                        "new_glossary": new_glossary_entries.strip() if use_translation_memory else "",
                        "original_indices": [sub['index'] for sub in block]
                    })
                    
                    if use_translation_memory and not evaluation_enabled:
                        new_entries = dict(entry.split('=') for entry in new_glossary_entries.strip().split('\n') if '=' in entry)
                        new_glossary.update(new_entries)
                    
                    break
                except json.JSONDecodeError:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Failed to parse JSON response for block {i+1} after {MAX_RETRIES} attempts")
                except ValueError as e:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Error in translation for block {i+1}: {str(e)}")
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise

    if not evaluation_enabled:
        glossary.update(new_glossary)

    return translated_responses, glossary

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
        "Swedish": "SV"
    }
    return deepl_language_map.get(language, language)

def translate_blocks_deepl(translation_blocks: List[List[Dict]], source_lang: str, target_lang: str, auth_key: str) -> List[Dict[str, Union[str, List[str]]]]:
    translator = deepl.Translator(auth_key)
    
    # Combine all subtitles into a single text, separated by double newlines
    full_text = "\n\n".join(["\n\n".join([sub['text'] for sub in block]) for block in translation_blocks])
    
    # Check if the text exceeds 120kb
    if len(full_text.encode('utf-8')) > 120 * 1024:
        # Split the text if it's too large
        split_texts = []
        current_text = ""
        for block in translation_blocks:
            block_text = "\n\n".join([sub['text'] for sub in block])
            if len((current_text + "\n\n" + block_text).encode('utf-8')) > 120 * 1024:
                split_texts.append(current_text)
                current_text = block_text
            else:
                current_text += "\n\n" + block_text if current_text else block_text
        if current_text:
            split_texts.append(current_text)
    else:
        split_texts = [full_text]
    
    # Convert target language to DeepL-specific code
    deepl_target_lang = get_deepl_language_code(target_lang)
    
    # Translate all texts
    translated_texts = []
    for text in split_texts:
        result = translator.translate_text(text, 
                                           target_lang=deepl_target_lang, 
                                           #split_sentences='nonewlines', 
                                           )
        translated_texts.append(result.text)
    
    # Combine translated texts
    full_translated_text = "\n\n".join(translated_texts)
    
    # Split the translated text back into blocks and subtitles
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
        
        translated_responses.append({
            "translation": block_translations,
            "new_glossary": "",
            "original_indices": [sub['index'] for sub in block]
        })
    
    return translated_responses

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
    
    # Sort subtitles by their original index to maintain order
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

def evaluate_translation(
    translation_blocks: List[List[Dict]],
    full_responses: List[Dict[str, Union[str, List[str]]]],
    source_lang: str,
    target_lang: str,
    anthropic_api_key: str,
    openai_api_key: str,
    llm_api: str,
    original_glossary: Dict[str, str],
    use_translation_memory: bool,
    model: str,
    evaluation_prompt: str,
    system_prompt: str,
    provider_params: Dict = None,
    use_thinking: bool = False,
    thinking_tokens: int = 8000,
    gemini_api_key: str = None
) -> Tuple[List[Dict[str, Union[str, List[str]]]], Dict[str, str]]:
    """
    Evaluates translations using the specified LLM API.
    
    Args:
        translation_blocks: List of original subtitle blocks
        full_responses: List of translated responses
        source_lang: Source language
        target_lang: Target language
        anthropic_api_key: Anthropic API key
        openai_api_key: OpenAI API key
        llm_api: LLM API to use
        original_glossary: Original glossary dictionary
        use_translation_memory: Whether to use translation memory
        model: Model name to use
        evaluation_prompt: Evaluation prompt template
        system_prompt: System prompt
        provider_params: Optional dictionary of OpenRouter provider parameters
        
    Returns:
        Tuple of evaluated responses and updated glossary
    """
    evaluated_responses = []
    new_glossary = original_glossary.copy()

    glossary_instructions = GLOSSARY_INSTRUCTIONS_EVALUATION.format(glossary=json.dumps(original_glossary, ensure_ascii=False, indent=2)) if use_translation_memory else ""
    base_prompt = evaluation_prompt.format(
        source_lang=source_lang,
        target_lang=target_lang,
        glossary_instructions=glossary_instructions,
        translation_guidelines=TRANSLATION_PROMPT_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            glossary_instructions=""
        )
    )

    client = None
    if llm_api == "anthropic":
        client = Anthropic(api_key=anthropic_api_key)
    elif llm_api == "openai":
        client = OpenAI(api_key=openai_api_key)
    elif llm_api == "openrouter":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get('OPENROUTER_API')
        )
    elif llm_api == "gemini":
        client = genai.Client(api_key=gemini_api_key)

    for i, (block, full_response) in enumerate(zip(translation_blocks, full_responses)):
        original_subtitles = json.dumps([sub['text'] for sub in block])
        original_translation = json.dumps(full_response['translation'])

        prompt = f"""{base_prompt}

        Original {source_lang} subtitles:
        {original_subtitles}

        Original translation response:
        {original_translation}

        Your improved translation:
        """
        
        # Log the entire evaluation prompt
        logging.info(f"Evaluation prompt for block {i+1}:\n{prompt}")
        
        for attempt in range(MAX_RETRIES):
            try:
                messages = [
                    {"role": "user", "content": prompt}
                ]
                content = llm_api_request(client, llm_api, model, messages, system_prompt=system_prompt, 
                         provider_params=provider_params, use_thinking=use_thinking, 
                         thinking_tokens=thinking_tokens)
                
                # Log the complete model output
                logging.info(f"Complete model output for evaluation block {i+1}:\n{content}")
                
                try:
                    evaluated_content, _, glossary_content = content.partition("[GLOSSARY]")
                    evaluated_subtitles = json.loads(evaluated_content)
                    if len(evaluated_subtitles) != len(full_response['translation']):
                        raise ValueError("Mismatch in subtitle count")
                    
                    evaluated_responses.append({
                        "translation": evaluated_subtitles,
                        "new_glossary": glossary_content.strip() if use_translation_memory else "",
                        "original_indices": full_response['original_indices']
                    })
                    
                    if use_translation_memory:
                        new_entries = dict(entry.split('=') for entry in glossary_content.strip().split('\n') if '=' in entry)
                        new_glossary.update(new_entries)
                    
                    logging.info(f"Successfully evaluated block {i+1}/{len(translation_blocks)}")
                    break
                except json.JSONDecodeError:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Failed to parse JSON response for block {i+1} after {MAX_RETRIES} attempts")
                except ValueError as e:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Error in evaluation for block {i+1}: {str(e)}")
            except Exception as e:
                logging.error(f"Error in evaluation attempt {attempt + 1} for block {i+1}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    raise ValueError(f"Failed to evaluate block {i+1} after {MAX_RETRIES} attempts. Last error: {str(e)}")

    return evaluated_responses, new_glossary

def manage_glossary(session_folder: str) -> Dict[str, str]:
    glossary_path = os.path.join(session_folder, "translation_glossary.json")
    if os.path.exists(glossary_path):
        with open(glossary_path, 'r', encoding='utf-8') as f:
            glossary = json.load(f)
    else:
        glossary = {}
    return glossary

def save_glossary(session_folder: str, glossary: Dict[str, str]) -> None:
    glossary_path = os.path.join(session_folder, "translation_glossary.json")
    with open(glossary_path, 'w', encoding='utf-8') as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

XTTS_TO_SENTENCE_SPLITTER_LANG = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt", "pl": "pl",
    "tr": "tr", "ru": "ru", "nl": "nl", "cs": "cs", "hu": "hu", "ca": "ca",
    "da": "da", "fi": "fi", "el": "el", "is": "is", "lv": "lv", "lt": "lt",
    "no": "no", "ro": "ro", "sk": "sk", "sl": "sl", "sv": "sv",
    # Languages like zh-cn, ja, ko, ar are not in sentence_splitter's list, so they will correctly result in None
}


def _check_split_validity(text_to_split: str, split_index: int, max_len: int, min_len_part: int) -> bool:
    if split_index <= 0 or split_index >= len(text_to_split):
        return False
    
    part1 = text_to_split[:split_index].strip()
    part2 = text_to_split[split_index:].strip()

    if not part1 or not part2:
        return False
    
    return min_len_part <= len(part1) <= max_len and len(part2) >= min_len_part

def _split_further(text: str, xtts_lang_code: str, max_chars: int, min_chars: int, conjunctions_map: Dict) -> List[str]:
    text = text.strip()
    if not text: return []
    if len(text) <= max_chars:
        return [text]

    results: List[str] = []
    midpoint = len(text) // 2

    punct_sets = ['.!?', ',;:']
    best_punct_split_point = -1

    for p_set_str in punct_sets:
        min_dist_to_mid_punct = float('inf')
        current_best_for_set = -1
        for i in range(len(text) - 1, min_chars - 1, -1):
            char = text[i]
            if char in p_set_str:
                if _check_split_validity(text, i + 1, max_chars, min_chars):
                    dist = abs((i + 1) - midpoint)
                    if dist < min_dist_to_mid_punct:
                        min_dist_to_mid_punct = dist
                        current_best_for_set = i + 1
                    elif dist == min_dist_to_mid_punct and (i + 1) > current_best_for_set:
                        current_best_for_set = i + 1
        if current_best_for_set != -1:
            best_punct_split_point = current_best_for_set
            break
    
    if best_punct_split_point != -1:
        part1 = text[:best_punct_split_point].strip()
        part2 = text[best_punct_split_point:].strip()
        if part1: results.append(part1)
        if part2: results.extend(_split_further(part2, xtts_lang_code, max_chars, min_chars, conjunctions_map))
        return [r for r in results if r]

    lang_conjunctions = conjunctions_map.get(xtts_lang_code, [])
    best_conj_split_point = -1
    min_dist_to_mid_conj = float('inf')

    if lang_conjunctions:
        for conj in lang_conjunctions:
            for m in re.finditer(r'\b' + re.escape(conj) + r'\b', text, re.IGNORECASE):
                split_at_index = m.start()
                if split_at_index == 0: continue
                if _check_split_validity(text, split_at_index, max_chars, min_chars):
                    dist = abs(split_at_index - midpoint)
                    if dist < min_dist_to_mid_conj:
                        min_dist_to_mid_conj = dist
                        best_conj_split_point = split_at_index
                    elif dist == min_dist_to_mid_conj and split_at_index > best_conj_split_point:
                        best_conj_split_point = split_at_index
    
    if best_conj_split_point != -1:
        part1 = text[:best_conj_split_point].strip()
        part2 = text[best_conj_split_point:].strip()
        if part1: results.append(part1)
        if part2: results.extend(_split_further(part2, xtts_lang_code, max_chars, min_chars, conjunctions_map))
        return [r for r in results if r]

    best_word_split_point = -1
    min_dist_to_mid_word = float('inf')

    for i in range(len(text) - 1, 0, -1):
        if text[i].isspace():
            if _check_split_validity(text, i, max_chars, min_chars):
                dist = abs(i - midpoint)
                if dist < min_dist_to_mid_word:
                    min_dist_to_mid_word = dist
                    best_word_split_point = i
                elif dist == min_dist_to_mid_word and i > best_word_split_point:
                     best_word_split_point = i
    
    if best_word_split_point != -1:
        part1 = text[:best_word_split_point].strip()
        part2 = text[best_word_split_point+1:].strip()
        if part1: results.append(part1)
        if part2: results.extend(_split_further(part2, xtts_lang_code, max_chars, min_chars, conjunctions_map))
        return [r for r in results if r]

    cut_at = max_chars
    part1_final = text[:cut_at].strip()
    part2_final = text[cut_at:].strip()
    
    temp_cut_text = text[:max_chars]
    last_space = temp_cut_text.rfind(' ')

    if last_space != -1:
        temp_part1 = temp_cut_text[:last_space].strip()
        temp_part2 = text[last_space+1:].strip()
        if len(temp_part1) >= min_chars and (len(temp_part2) >= min_chars or not temp_part2):
            part1_final = temp_part1
            part2_final = temp_part2
            
    if part1_final: results.append(part1_final)
    if part2_final: results.extend(_split_further(part2_final, xtts_lang_code, max_chars, min_chars, conjunctions_map))
    return [r for r in results if r]


def _fallback_original_find_split_point(text: str, max_length: int, min_length: int, language_code: str, conjunctions_map: Dict) -> int:
    mid = len(text) // 2
    best_split_len = -1

    punct_sets_with_priority = [('.!?', 1), (',;:', 2)]

    for p_chars, priority in punct_sets_with_priority:
        current_best_len_for_set = -1
        min_dist_for_set = float('inf')
        for i in range(min_length -1, min(len(text) - min_length, max_length)):
            if text[i] in p_chars:
                part1_len = i + 1
                part2_text = text[i+1:].strip()
                if part1_len >= min_length and part1_len <= max_length and len(part2_text) >= min_length:
                    dist = abs(part1_len - mid)
                    if dist < min_dist_for_set:
                        min_dist_for_set = dist
                        current_best_len_for_set = part1_len
                    elif dist == min_dist_for_set and part1_len > current_best_len_for_set:
                        current_best_len_for_set = part1_len
        if current_best_len_for_set != -1:
            return current_best_len_for_set # Return as soon as best for current priority found

    lang_conjunctions = conjunctions_map.get(language_code, [])
    min_dist_p3 = float('inf')
    best_conj_split_len = -1
    if lang_conjunctions:
        for conj in lang_conjunctions:
            for m in re.finditer(r'\b' + re.escape(conj) + r'\b', text, re.IGNORECASE):
                part1_len = m.start()
                part2_text = text[m.start():].strip()
                if part1_len >= min_length and part1_len <= max_length and len(part2_text) >= min_length:
                    dist = abs(part1_len - mid)
                    if dist < min_dist_p3:
                        min_dist_p3 = dist
                        best_conj_split_len = part1_len
                    elif dist == min_dist_p3 and part1_len > best_conj_split_len:
                        best_conj_split_len = part1_len
    if best_conj_split_len != -1:
        return best_conj_split_len

    min_dist_p4 = float('inf')
    best_word_boundary_len = -1
    for i in range(min(len(text) - 1, max_length), min_length -1 , -1): # Iterate from right to find split point
         if text[i].isspace():
            part1_len = len(text[:i].strip())
            part2_text = text[i+1:].strip()
            if part1_len >= min_length and part1_len <= max_length and len(part2_text) >= min_length:
                dist = abs(part1_len - mid)
                if dist < min_dist_p4:
                    min_dist_p4 = dist
                    best_word_boundary_len = part1_len
                elif dist == min_dist_p4 and part1_len > best_word_boundary_len: # Prefer longer first part if equidistant
                    best_word_boundary_len = part1_len

    if best_word_boundary_len != -1:
        return best_word_boundary_len
    
    if len(text) > max_length:
        cut_text = text[:max_length]
        last_space = -1
        for i in range(max_length -1 , min_length -2, -1): # Iterate down to min_length-1 for index
             if i < 0: break
             if cut_text[i].isspace():
                 if len(text[i+1:].strip()) >= min_length:
                    last_space = i
                    break
        if last_space != -1:
            return last_space # This is index, so len is last_space for text[:last_space]
        return max_length

    return len(text)


def create_speech_blocks(
    srt_content: str,
    session_folder: str,
    video_name: str,
    target_language: str, # This is XTTS language name like "English" or "en"
    min_chars: int = SPEECH_BLOCK_MIN_CHARS,
    max_chars: int = SPEECH_BLOCK_CHAR_LIMIT,
    merge_threshold: int = SPEECH_BLOCK_MERGE_THRESHOLD
) -> List[Dict]:
    
    CONJUNCTIONS = { # Defined inside as in original code
        "en": ["and", "but", "or", "because", "although", "so", "while", "if", "then", "that", "as", "for", "since", "until", "whether"],
        "es": ["y", "pero", "o", "porque", "aunque", "así", "mientras", "si", "entonces", "que", "como", "pues", "desde", "hasta", "si"],
        "fr": ["et", "mais", "ou", "parce que", "bien que", "donc", "pendant que", "si", "alors", "que", "comme", "car", "depuis", "jusqu'à", "si"],
        "de": ["und", "aber", "oder", "weil", "obwohl", "also", "während", "wenn", "dann", "dass", "als", "denn", "seit", "bis", "ob"],
        "it": ["e", "ma", "o", "perché", "sebbene", "quindi", "mentre", "se", "allora", "che", "come", "poiché", "da quando", "fino a", "se"],
        "pt": ["e", "mas", "ou", "porque", "embora", "então", "enquanto", "se", "logo", "que", "como", "pois", "desde", "até", "se"],
        "pl": ["i", "ale", "lub", "ponieważ", "chociaż", "więc", "podczas gdy", "jeśli", "wtedy", "że", "jak", "gdyż", "od", "aż", "czy"],
        "tr": ["ve", "ama", "veya", "çünkü", "rağmen", "bu yüzden", "iken", "eğer", "o zaman", "ki", "gibi", "zira", "-den beri", "-e kadar", "acaba"],
        "ru": ["и", "но", "или", "потому что", "хотя", "так что", "пока", "если", "тогда", "что", "как", "ибо", "с", "до", "ли"],
        "nl": ["en", "maar", "of", "omdat", "hoewel", "dus", "terwijl", "als", "dan", "dat", "zoals", "want", "sinds", "tot", "of"],
        "cs": ["a", "ale", "nebo", "protože", "ačkoli", "takže", "zatímco", "jestli", "pak", "že", "jako", "neboť", "od", "až", "zda"],
        "ar": ["و", "لكن", "أو", "لأن", "رغم أن", "لذلك", "بينما", "إذا", "ثم", "أن", "كما", "ف", "منذ", "حتى", "هل"],
        "zh-cn": ["和", "但是", "或者", "因为", "虽然", "所以", "当", "如果", "那么", "那", "作为", "由于", "自从", "直到", "是否"], # Simplified
        "ja": ["そして", "しかし", "または", "なぜなら", "にもかかわらず", "だから", "間", "もし", "その時", "と", "ように", "から", "以来", "まで", "かどうか"], # Simplified
        "hu": ["és", "de", "vagy", "mert", "bár", "tehát", "míg", "ha", "akkor", "hogy", "mint", "hiszen", "óta", "ameddig", "vajon"],
        "ko": ["그리고", "하지만", "또는", "왜냐하면", "비록", "그래서", "동안", "만약", "그때", "것", "처럼", "때문에", "이후", "까지", "인지"] # Simplified
    }

    def get_xtts_language_code_local(lang_name: str) -> str:
        # Simplified local version for this function's context
        xtts_language_map = {
            "English": "en", "Spanish": "es", "French": "fr", "German": "de",
            "Italian": "it", "Portuguese": "pt", "Polish": "pl", "Turkish": "tr",
            "Russian": "ru", "Dutch": "nl", "Czech": "cs", "Arabic": "ar",
            "Chinese": "zh-cn", "Japanese": "ja", "Hungarian": "hu", "Korean": "ko",
            # Adding 2-letter codes directly
            "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt", "pl": "pl",
            "tr": "tr", "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar", "zh-cn": "zh-cn",
            "ja": "ja", "hu": "hu", "ko": "ko", "ca": "ca", "da": "da", "fi": "fi", "el": "el",
            "is": "is", "lv": "lv", "lt": "lt", "no": "no", "ro": "ro", "sk": "sk", "sl": "sl", "sv": "sv"
        }
        return xtts_language_map.get(lang_name, lang_name if len(lang_name) == 2 else "en") # Fallback to 'en' or raw 2-letter code

    xtts_language_code = get_xtts_language_code_local(target_language)
    
    splitter_lang_code = XTTS_TO_SENTENCE_SPLITTER_LANG.get(xtts_language_code)
    sentence_splitter_instance = None
    if splitter_lang_code:
        try:
            sentence_splitter_instance = SentenceSplitter(language=splitter_lang_code)
        except Exception as e:
            logger.warning(f"Could not initialize SentenceSplitter for {splitter_lang_code} ({xtts_language_code}): {e}. Will use fallback logic.")
            splitter_lang_code = None # Force fallback

    subtitles = list(srt.parse(srt_content))
    
    merged_subtitles_intermediate = []
    if not subtitles: return []

    for subtitle in subtitles:
        if not merged_subtitles_intermediate or \
           (subtitle.start - merged_subtitles_intermediate[-1].end).total_seconds() * 1000 > merge_threshold:
            merged_subtitles_intermediate.append(subtitle)
        else:
            merged_subtitles_intermediate[-1] = srt.Subtitle(
                index=merged_subtitles_intermediate[-1].index, # Keeps index of the first sub in merge
                start=merged_subtitles_intermediate[-1].start,
                end=subtitle.end,
                content=(merged_subtitles_intermediate[-1].content + " " + subtitle.content).strip()
            )
    
    all_speech_block_parts: List[Dict] = []

    for merged_sub in merged_subtitles_intermediate:
        subtitle_text = merged_sub.content.strip()
        subtitle_indices = [merged_sub.index] # Assuming index of merged sub is what's tracked

        if not subtitle_text:
            continue

        current_sub_texts: List[str] = []

        if len(subtitle_text) >= min_chars and len(subtitle_text) <= max_chars:
            current_sub_texts.append(subtitle_text)
        elif len(subtitle_text) < min_chars: # Too short from the start
            current_sub_texts.append(subtitle_text)
        else: # Needs splitting
            if sentence_splitter_instance and splitter_lang_code:
                raw_sentences = sentence_splitter_instance.split(text=subtitle_text)
                for sentence in raw_sentences:
                    sentence = sentence.strip()
                    if not sentence: continue
                    if len(sentence) >= min_chars and len(sentence) <= max_chars:
                        current_sub_texts.append(sentence)
                    elif len(sentence) < min_chars: # Sentence too short
                         current_sub_texts.append(sentence)
                    else: # Sentence too long, needs further splitting
                        split_parts = _split_further(sentence, xtts_language_code, max_chars, min_chars, CONJUNCTIONS)
                        current_sub_texts.extend(p for p in split_parts if p)
            else: # Fallback logic
                remaining_text = subtitle_text
                while remaining_text:
                    if len(remaining_text) <= max_chars:
                        if remaining_text: current_sub_texts.append(remaining_text)
                        break
                    
                    split_len = _fallback_original_find_split_point(remaining_text, max_chars, min_chars, xtts_language_code, CONJUNCTIONS)
                    
                    part_to_add = remaining_text[:split_len].strip()
                    if part_to_add: current_sub_texts.append(part_to_add)
                    remaining_text = remaining_text[split_len:].strip()
        
        # Add processed texts for this merged_sub to all_speech_block_parts
        for text_part in current_sub_texts:
            if text_part: # Ensure not empty
                 all_speech_block_parts.append({
                    "text": text_part,
                    "subtitles": subtitle_indices 
                })
    
    # Final assembly and merging of small parts
    final_speech_blocks: List[Dict] = []
    if not all_speech_block_parts:
        if os.path.exists(session_folder) and video_name: # Only try to save if path components exist
            json_output_path = os.path.join(session_folder, f"{video_name}_speech_blocks.json")
            with open(json_output_path, 'w', encoding='utf-8') as json_file:
                json.dump([], json_file, ensure_ascii=False, indent=2)
        return []

    for part_data in all_speech_block_parts:
        text = part_data["text"]
        indices = part_data["subtitles"]

        if not final_speech_blocks:
            if text: # Don't add if text is empty
                final_speech_blocks.append({
                    "number": str(len(final_speech_blocks) + 1).zfill(4),
                    "text": text,
                    "subtitles": indices
                })
        else:
            if len(text) < min_chars and text and \
               len(final_speech_blocks[-1]["text"]) + len(text) + 1 <= max_chars:
                final_speech_blocks[-1]["text"] += " " + text
                final_speech_blocks[-1]["subtitles"] = sorted(list(set(final_speech_blocks[-1]["subtitles"] + indices)))
            elif text: # Add as new block if it's not empty
                final_speech_blocks.append({
                    "number": str(len(final_speech_blocks) + 1).zfill(4),
                    "text": text,
                    "subtitles": indices
                })
    
    # Renumber blocks after all merging is done
    for i, block in enumerate(final_speech_blocks):
        block["number"] = str(i + 1).zfill(4)

    # Save the speech blocks as JSON
    if os.path.exists(session_folder) and video_name: # Check to prevent error if called without valid session/name
        json_output_path = os.path.join(session_folder, f"{video_name}_speech_blocks.json")
        try:
            with open(json_output_path, 'w', encoding='utf-8') as json_file:
                json.dump(final_speech_blocks, json_file, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save speech blocks JSON: {e}")
    else:
        logger.warning("Session folder or video name not available, skipping saving of speech_blocks.json")


    return final_speech_blocks
def get_xtts_language_code(target_language: str) -> str:
    xtts_language_map: Dict[str, str] = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Polish": "pl",
        "Turkish": "tr",
        "Russian": "ru",
        "Dutch": "nl",
        "Czech": "cs",
        "Arabic": "ar",
        "Chinese": "zh-cn",
        "Japanese": "ja",
        "Hungarian": "hu",
        "Korean": "ko"
    }

    xtts_language_code = xtts_language_map.get(target_language)
    if not xtts_language_code:
        raise ValueError(f"The target language '{target_language}' is not supported by XTTS. "
                         f"Supported languages are: {', '.join(xtts_language_map.keys())}")
    
    return xtts_language_code

def generate_tts_audio(speech_blocks: List[Dict], tts_voice: str, language: str, session_folder: str, video_name: str) -> List[str]:
    audio_files = []
    sentence_wavs_folder = os.path.join(session_folder, "Sentence_wavs")
    os.makedirs(sentence_wavs_folder, exist_ok=True)
    
    for block in speech_blocks:
        output_file = os.path.join(sentence_wavs_folder, f"{video_name}_{block['number']}.wav")  # Changed here
        
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
                logging.info(f"Attempting to generate TTS for block {block['number']}")
                logging.info(f"Request data: {data}")
                
                response = requests.post("http://localhost:8020/tts_to_audio/", json=data)
                logging.info(f"Response status code: {response.status_code}")
                
                if response.status_code == 200:
                    audio_data = io.BytesIO(response.content)
                    audio = AudioSegment.from_file(audio_data, format="wav")
                    audio.export(output_file, format="wav")
                    audio_files.append(output_file)
                    logging.info(f"Successfully generated TTS for block {block['number']}")
                    break
                else:
                    logging.error(f"TTS API returned status code {response.status_code} for block {block['number']}")
                    logging.error(f"Response content: {response.text}")
            except Exception as e:
                logging.error(f"Error in TTS generation attempt {attempt + 1} for block {block['number']}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    logging.error(f"Failed to generate TTS for block {block['number']} after {MAX_RETRIES} attempts")

    return audio_files

def create_alignment_blocks(session_folder: str, video_name: str, target_language: str, evaluated: bool = False, speech_blocks_path: str = None) -> List[Dict]:
    if speech_blocks_path is None:
        speech_blocks_path = os.path.join(session_folder, f"{video_name}_speech_blocks.json")
    
    logging.info(f"Attempting to open speech blocks file: {speech_blocks_path}")
    
    try:
        with open(speech_blocks_path, 'r', encoding='utf-8') as f:
            speech_blocks = json.load(f)
    except FileNotFoundError:
        logging.error(f"Speech blocks JSON file not found: {speech_blocks_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {speech_blocks_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error opening {speech_blocks_path}: {str(e)}")
        raise

    # Find the newest SRT file
    srt_files = [f for f in os.listdir(session_folder) if f.endswith('.srt')]
    if not srt_files:
        raise FileNotFoundError("No SRT files found in the session folder")
    newest_srt = max(srt_files, key=lambda f: os.path.getmtime(os.path.join(session_folder, f)))
    srt_path = os.path.join(session_folder, newest_srt)
    logging.info(f"Using the newest SRT file: {srt_path}")

    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            subtitles = list(srt.parse(f.read()))
    except FileNotFoundError:
        logging.error(f"SRT file not found: {srt_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading SRT file {srt_path}: {str(e)}")
        raise

    alignment_blocks = []
    current_block = None

    sentence_wavs_folder = os.path.join(session_folder, 'Sentence_wavs')
    all_wav_files = os.listdir(sentence_wavs_folder)
    logging.info(f"All WAV files in directory: {all_wav_files}")

    for block in speech_blocks:
        block_subtitles = [sub for sub in subtitles if sub.index in block["subtitles"]]
        
        if block_subtitles:
            start_sub = block_subtitles[0]
            end_sub = block_subtitles[-1]
            
            block_number = int(block['number'])  # Ensure block number is an integer
            logging.info(f"Processing block number: {block_number}")
            
            wav_files = [f for f in all_wav_files 
                         if re.search(rf"_{block_number}\.wav$", f) or 
                            re.search(rf"_{block_number:04d}\.wav$", f) or
                            re.search(rf"_{block_number:d}\.wav$", f)]
            
            logging.info(f"Matched WAV files for block {block_number}: {wav_files}")
            
            if not wav_files:
                logging.warning(f"No matching WAV files found for block number {block_number}")

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
                    logging.info(f"Added alignment block: {json.dumps(current_block, default=str, indent=2)}")
                current_block = new_block

    if current_block:
        alignment_blocks.append(current_block)
        logging.info(f"Added final alignment block: {json.dumps(current_block, default=str, indent=2)}")

    logging.info(f"Total number of alignment blocks: {len(alignment_blocks)}")
    logging.info("First few alignment blocks:")
    for block in alignment_blocks[:3]:  # Log the first 3 blocks
        logging.info(json.dumps(block, default=str, indent=2))

    return alignment_blocks

def align_audio_blocks(alignment_blocks: List[Dict], session_folder: str, delay_start: int = 1500, speed_up: int = 100) -> str:
    final_audio = AudioSegment.silent(duration=0)
    current_time = 0
    total_shift = 0

    def timedelta_to_ms(td):
        return td.total_seconds() * 1000

    def speed_up_audio_ffmpeg(audio_segment, factor, session_folder):
        """Speed up audio without changing pitch using FFmpeg's atempo filter"""
        import subprocess
        
        temp_input = os.path.join(session_folder, "temp_speedup_input.wav")
        temp_output = os.path.join(session_folder, "temp_speedup_output.wav")
        
        try:
            audio_segment.export(temp_input, format="wav")
            
            atempo_filters = []
            remaining_factor = factor
            
            while remaining_factor > 2.0:
                atempo_filters.append("atempo=2.0")
                remaining_factor /= 2.0
            
            while remaining_factor < 0.5:
                atempo_filters.append("atempo=0.5")
                remaining_factor /= 0.5
            
            if abs(remaining_factor - 1.0) > 0.01:
                atempo_filters.append(f"atempo={remaining_factor:.3f}")
            
            if not atempo_filters:
                return audio_segment
                
            filter_string = ",".join(atempo_filters)
            
            cmd = ['ffmpeg', '-i', temp_input, '-af', filter_string, '-y', temp_output]
            subprocess.run(cmd, check=True, capture_output=True)
            return AudioSegment.from_wav(temp_output)
            
        finally:
            for temp_file in [temp_input, temp_output]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    sentence_wavs_folder = os.path.join(session_folder, "Sentence_wavs")

    for i, block in enumerate(alignment_blocks):
        block_start_ms = timedelta_to_ms(block["start"])
        
        if i < len(alignment_blocks) - 1:
            next_block_start_ms = timedelta_to_ms(alignment_blocks[i + 1]["start"])
            block_duration = next_block_start_ms - block_start_ms
        else:
            block_duration = timedelta_to_ms(block["end"] - block["start"])

        adjusted_block_start = block_start_ms + total_shift
        
        if adjusted_block_start > current_time:
            silence_duration = adjusted_block_start - current_time
            final_audio += AudioSegment.silent(duration=silence_duration)
            current_time = adjusted_block_start

        # Load and combine audio files
        block_audio = AudioSegment.silent(duration=0)
        for audio_file in block["audio_files"]:
            wav_path = os.path.join(sentence_wavs_folder, audio_file)
            if os.path.exists(wav_path):
                wav_audio = AudioSegment.from_wav(wav_path)
                if len(block_audio) > 0:
                    block_audio += AudioSegment.silent(duration=100)
                block_audio += wav_audio
            else:
                logging.error(f"Audio file not found: {wav_path}")

        original_audio_duration = len(block_audio)
        processed_audio = block_audio
        audio_delay = 0

        # DELAY LOGIC: Only apply if no accumulated shift AND subtitle is longer
        if original_audio_duration < block_duration and total_shift <= 0:
            available_time = block_duration - original_audio_duration
            max_delay = min(delay_start, int(available_time * 0.7))
            audio_delay = max_delay
            logging.info(f"Block {i+1}: Applying delay of {audio_delay}ms "
                       f"(audio: {original_audio_duration}ms, subtitle: {block_duration}ms)")

        # SPEED-UP LOGIC: Different constraints based on whether we have accumulated shift
        should_speed_up = False
        actual_speedup_factor = 1.0
        speed_up_reason = ""

        if total_shift > 0 and speed_up > 100:
            # CASE 1: We have accumulated shift - prioritize eliminating it
            should_speed_up = True
            speed_up_reason = f"eliminating accumulated shift of {total_shift}ms"
            
            # Calculate speedup needed to eliminate the shift over this audio duration
            speedup_needed_for_shift = (original_audio_duration + total_shift) / original_audio_duration
            max_allowed_speedup = speed_up / 100.0
            
            # Compare against SHIFT duration, not subtitle duration
            actual_speedup_factor = min(speedup_needed_for_shift, max_allowed_speedup)
            
            logging.info(f"Block {i+1}: Shift elimination - need {speedup_needed_for_shift:.2f}x, "
                       f"max allowed {max_allowed_speedup:.2f}x, using {actual_speedup_factor:.2f}x")
            
        elif total_shift <= 0 and original_audio_duration > block_duration and speed_up > 100:
            # CASE 2: No accumulated shift, but current audio longer than subtitle
            should_speed_up = True
            speed_up_reason = "fitting current audio to subtitle"
            
            # Calculate speedup needed to fit current subtitle
            speedup_needed_for_subtitle = original_audio_duration / block_duration
            max_allowed_speedup = speed_up / 100.0
            
            # Compare against SUBTITLE duration (standard case)
            actual_speedup_factor = min(speedup_needed_for_subtitle, max_allowed_speedup)
            
            logging.info(f"Block {i+1}: Subtitle fitting - need {speedup_needed_for_subtitle:.2f}x, "
                       f"max allowed {max_allowed_speedup:.2f}x, using {actual_speedup_factor:.2f}x")

        # Apply speed-up if needed
        if should_speed_up and actual_speedup_factor > 1.01:
            processed_audio = speed_up_audio_ffmpeg(block_audio, actual_speedup_factor, session_folder)
            new_duration = len(processed_audio)
            logging.info(f"Block {i+1}: Applied {actual_speedup_factor:.2f}x speedup for {speed_up_reason} "
                       f"({original_audio_duration}ms -> {new_duration}ms)")
        elif total_shift > 0 and speed_up <= 100:
            logging.info(f"Block {i+1}: Would speed up for shift elimination, but speed_up={speed_up} (disabled)")
        elif should_speed_up:
            logging.info(f"Block {i+1}: Speedup factor {actual_speedup_factor:.2f}x too small, skipping")

        # Add delay and processed audio
        if audio_delay > 0:
            final_audio += AudioSegment.silent(duration=audio_delay)
            current_time += audio_delay

        final_audio += processed_audio
        current_time += len(processed_audio)

        # Update total_shift based on actual timing
        actual_audio_duration = len(processed_audio) + audio_delay
        
        if actual_audio_duration > block_duration:
            additional_shift = actual_audio_duration - block_duration
            total_shift += additional_shift
            logging.info(f"Block {i+1}: Added {additional_shift}ms to shift, total_shift now: {total_shift}ms")
        else:
            silence_needed = block_duration - actual_audio_duration
            if silence_needed >= total_shift:
                silence_to_add = silence_needed - total_shift
                final_audio += AudioSegment.silent(duration=silence_to_add)
                current_time += silence_to_add
                logging.info(f"Block {i+1}: Eliminated all shift ({total_shift}ms) plus {silence_to_add}ms silence")
                total_shift = 0
            else:
                total_shift -= silence_needed
                logging.info(f"Block {i+1}: Reduced shift by {silence_needed}ms, total_shift now: {total_shift}ms")

    output_path = os.path.join(session_folder, "aligned_audio.wav")
    final_audio.export(output_path, format="wav")
    logging.info(f"Alignment completed. Final total_shift: {total_shift}ms")
    return output_path


def mix_audio_tracks(video_path: str, synced_audio_path: str, session_folder: str, video_name: str, target_language: str, evaluated: bool = False) -> str:
    evaluation_suffix = "_eval" if evaluated else ""
    amplified_dubbed_audio_path = os.path.join(session_folder, f"amplified_dubbed_audio{evaluation_suffix}.wav")
    mixed_audio_path = os.path.join(session_folder, f"mixed_audio{evaluation_suffix}.wav")
    output_path = os.path.join(session_folder, f"final_output{evaluation_suffix}.mp4")

    # Extract original audio
    original_audio_path = os.path.join(session_folder, "original_audio.wav")
    extract_audio_command = [
        'ffmpeg',
        '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '44100', '-ac', '2',
        original_audio_path
    ]
    subprocess.run(extract_audio_command, check=True)

    # Analyze dubbed audio to find max volume
    analyze_command = [
        'ffmpeg',
        '-i', synced_audio_path,
        '-af', 'volumedetect',
        '-vn', '-sn', '-dn',
        '-f', 'null',
        '/dev/null'
    ]
    result = subprocess.run(analyze_command, capture_output=True, text=True, check=True)
    
    # Extract max_volume from the output
    max_volume_line = [line for line in result.stderr.split('\n') if 'max_volume' in line][0]
    max_volume = float(max_volume_line.split(':')[1].strip().split()[0])
    
    # Calculate required amplification
    amplification = -max_volume

    # Amplify dubbed audio
    amplify_command = [
        'ffmpeg',
        '-i', synced_audio_path,
        '-af', f'volume={amplification}dB',
        amplified_dubbed_audio_path
    ]
    subprocess.run(amplify_command, check=True)

    # Mix audio tracks
    ffmpeg_command = [
        'ffmpeg',
        '-i', original_audio_path,
        '-i', amplified_dubbed_audio_path,
        '-filter_complex',
        "[1]silencedetect=n=-30dB:d=2[silence];"
        "[silence]aformat=sample_fmts=u8:sample_rates=44100:channel_layouts=mono,"
        "aresample=async=1000,pan=1c|c0=c0,"
        "aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=mono[silence_mono];"
        "[0][silence_mono]sidechaincompress=threshold=0.01:ratio=20:attack=100:release=500:makeup=1[gated];"
        "[1]volume=2[subtitles];"
        "[gated][subtitles]amix=inputs=2[mixed]",
        '-map', '[mixed]',
        mixed_audio_path
    ]
    subprocess.run(ffmpeg_command, check=True)

    # Replace audio in video
    replace_audio_command = [
        'ffmpeg',
        '-i', video_path,
        '-i', mixed_audio_path,
        '-c:v', 'copy', 
        '-c:a', 'aac',   
        '-map', '0:v',   
        '-map', '1:a',   
        output_path
    ]
    subprocess.run(replace_audio_command, check=True)

    return output_path

def cleanup_temp_files(session_folder: str, task: str) -> None:
    temp_files = ['extracted_audio.wav', 'transcription.srt', 'aligned_audio.wav']
    for file in temp_files:
        file_path = os.path.join(session_folder, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {file_path}: {str(e)}")
    
    if task != 'tts':
        speech_block_files = [f for f in os.listdir(session_folder) if f.startswith('speech_block_') and f.endswith('.wav')]
        for file in speech_block_files:
            file_path = os.path.join(session_folder, file)
            try:
                os.remove(file_path)
                logging.info(f"Removed speech block file: {file_path}")
            except Exception as e:
                logging.warning(f"Failed to remove speech block file {file_path}: {str(e)}")

def create_speech_blocks_json(srt_file: str, session_folder: str, merge_threshold: int = SPEECH_BLOCK_MERGE_THRESHOLD) -> None:
    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    video_name = os.path.splitext(os.path.basename(srt_file))[0]
    speech_blocks = create_speech_blocks(srt_content, session_folder, video_name, "en", merge_threshold=merge_threshold)
    
    json_output_path = os.path.join(session_folder, f"{video_name}_speech_blocks.json")
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(speech_blocks, json_file, ensure_ascii=False, indent=2)

def normalize_filename(filename):
    return unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')

def correct_subtitles(
    translation_blocks: List[List[Dict]],
    source_lang: str,
    correction_instructions: str,
    anthropic_api_key: str = None,
    openai_api_key: str = None,
    llm_api: str = "anthropic",
    model: str = None,
    correction_prompt: str = CORRECTION_PROMPT_TEMPLATE,
    system_prompt: str = CUSTOM_SYSTEM_PROMPT,
    use_cot: bool = False,
    use_context: bool = False,
    provider_params: Dict = None,
    use_thinking: bool = False,
    thinking_tokens: int = 8000,
    gemini_api_key: str = None
) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Corrects subtitles using the specified LLM API.
    """
    corrected_responses = []
    previous_response = None
    
    # Select the appropriate template but don't format it yet
    prompt_template = CORRECTION_PROMPT_TEMPLATE_COT if use_cot else CORRECTION_PROMPT_TEMPLATE

    # Initialize appropriate client based on LLM API
    client = None
    if llm_api == "anthropic":
        client = Anthropic(api_key=anthropic_api_key)
    elif llm_api == "openai":
        client = OpenAI(api_key=openai_api_key)
    elif llm_api == "openrouter":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get('OPENROUTER_API')
        )
    elif llm_api == "gemini":
        # For Gemini, we'll pass the API key to llm_api_request
        pass
    elif llm_api == "local":
        # Local API doesn't need a client initialization
        pass
    elif llm_api == "deepl":
        # DeepL is not applicable for correction
        raise ValueError("DeepL API is not supported for subtitle correction")

    for i, block in enumerate(translation_blocks):
        # Now format the prompt for this specific block
        base_prompt = prompt_template.format(
            correction_instructions=correction_instructions if correction_instructions else "No additional instructions provided.",
            subtitle_count=len(block)  # Now block is defined
        )
        
        subtitles = json.dumps([{"number": idx+1, "text": sub['text']} for idx, sub in enumerate(block)], ensure_ascii=False)
        
        if use_context and previous_response:
            context_prompt = CONTEXT_PROMPT_TEMPLATE.format(
                context_previous_response=previous_response
            )
            final_prompt = f"{base_prompt}\n{context_prompt}\n\nThe subtitles:\n{subtitles}"
        else:
            final_prompt = f"{base_prompt}\n\nThe subtitles:\n{subtitles}"

        logging.info(f"Correcting block {i+1}/{len(translation_blocks)}")
        
        for attempt in range(MAX_RETRIES):
            try:
                messages = [{"role": "user", "content": final_prompt}]
                content = llm_api_request(
                    client=client, 
                    llm_api=llm_api, 
                    model=model, 
                    messages=messages, 
                    system_prompt=system_prompt, 
                    provider_params=provider_params, 
                    use_thinking=use_thinking, 
                    thinking_tokens=thinking_tokens,
                    gemini_api_key=gemini_api_key
                )
                previous_response = content
                
                try:
                    # Clean up the response - find JSON array if embedded in other text
                    start_idx = content.find('[')
                    end_idx = content.rfind(']')
                    if start_idx != -1 and end_idx != -1:
                        content = content[start_idx:end_idx+1]
                    
                    corrected_subtitles = json.loads(content)
                    if len(corrected_subtitles) != len(block):
                        logging.warning(f"Block {i+1}: Mismatch in subtitle count. Expected {len(block)}, got {len(corrected_subtitles)}")
                        if attempt < MAX_RETRIES - 1:
                            logging.info(f"Retrying block {i+1} (attempt {attempt+2}/{MAX_RETRIES})")
                            continue
                        raise ValueError(f"Mismatch in subtitle count for block {i+1}")
                    
                    # Check structure of each subtitle
                    try:
                        for j, sub in enumerate(corrected_subtitles):
                            if not isinstance(sub, dict) or "number" not in sub or "text" not in sub:
                                logging.warning(f"Block {i+1}: Invalid subtitle format at position {j}")
                                if attempt < MAX_RETRIES - 1:
                                    logging.info(f"Retrying block {i+1} (attempt {attempt+2}/{MAX_RETRIES})")
                                    raise ValueError("Invalid subtitle format")
                                else:
                                    raise ValueError(f"Invalid subtitle format in block {i+1}")
                    except ValueError:
                        continue
                    
                    corrected_responses.append({
                        "translation": [sub["text"] for sub in corrected_subtitles],  # Extract just the text
                        "original_indices": [sub['index'] for sub in block]
                    })
                    logging.info(f"Successfully corrected block {i+1}")
                    break
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error in block {i+1}, attempt {attempt+1}: {str(e)}")
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Failed to parse JSON response for block {i+1} after {MAX_RETRIES} attempts")
            except Exception as e:
                logging.error(f"Error in correction attempt {attempt+1} for block {i+1}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    raise ValueError(f"Failed to correct block {i+1} after {MAX_RETRIES} attempts: {str(e)}")

    return corrected_responses

def sync_audio_video(session_folder: str, input_video: str = None, delay_start: int = 1500, speed_up: int = 100) -> None:
    # Define XTTS language codes
    xtts_languages = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "pl": "Polish", "tr": "Turkish",
        "ru": "Russian", "nl": "Dutch", "cs": "Czech", "ar": "Arabic",
        "zh-cn": "Chinese", "ja": "Japanese", "hu": "Hungarian", "ko": "Korean"
    }

    logging.info(f"Contents of session folder: {os.listdir(session_folder)}")

    # First try to use the specified input video if provided
    if input_video and os.path.exists(input_video):
        video_path = input_video
    else:
        # Fall back to finding the newest video file
        video_files = [f for f in os.listdir(session_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')) and not f.startswith('final_output')]
        if not video_files:
            raise ValueError("No video file found in the session folder")
        video_path = os.path.join(session_folder, max(video_files, key=lambda f: os.path.getmtime(os.path.join(session_folder, f))))
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Find the speech blocks JSON file
    speech_block_files = []
    for filename in os.listdir(session_folder):
        normalized_filename = normalize_filename(filename)
        if 'speech_blocks' in normalized_filename and (normalized_filename.endswith('.json') or '.' not in normalized_filename):
            speech_block_files.append(filename)

    logging.info(f"Found speech block files: {speech_block_files}")

    if not speech_block_files:
        raise ValueError("No speech blocks JSON file found in the session folder")
    
    if len(speech_block_files) == 1:
        json_path = os.path.join(session_folder, speech_block_files[0])
    else:
        # If there are multiple files, prioritize the one with a language code
        language_specific_files = [f for f in speech_block_files if any(f"_{lang}_speech_blocks" in normalize_filename(f) for lang in xtts_languages.keys())]
        logging.info(f"Language-specific files: {language_specific_files}")
        if language_specific_files:
            json_path = os.path.join(session_folder, language_specific_files[0])
        else:
            # If no language-specific file found, use the most recent one
            json_path = os.path.join(session_folder, max(speech_block_files, key=lambda f: os.path.getmtime(os.path.join(session_folder, f))))
    
    logging.info(f"Using speech blocks file: {json_path}")
    logging.info(f"Full path of speech blocks file: {os.path.abspath(json_path)}")
    
    # Determine the language from the speech blocks file name
    speech_blocks_language = next((lang for lang, name in xtts_languages.items() if f"_{lang}_speech_blocks" in normalize_filename(os.path.basename(json_path))), "en")
    
    # Create alignment blocks
    alignment_blocks = create_alignment_blocks(session_folder, video_name, xtts_languages[speech_blocks_language], False, json_path)
    logging.info(f"Created {len(alignment_blocks)} alignment blocks")

    # Align audio blocks with delay and speed-up parameters
    aligned_audio_path = align_audio_blocks(alignment_blocks, session_folder, delay_start, speed_up)
    logging.info(f"Audio alignment completed: {aligned_audio_path}")

    # Mix audio tracks
    final_output_path = mix_audio_tracks(video_path, aligned_audio_path, session_folder, video_name, xtts_languages[speech_blocks_language], False)
    logging.info(f"Final output saved: {final_output_path}")

def perform_equalization(input_srt_path, output_srt_path, max_line_length):
    """Performs SRT equalization on the given input file."""
    equalize_srt(input_srt_path, output_srt_path, max_line_length)

def main():
    parser = argparse.ArgumentParser(description="Video subtitle translation and dubbing tool")
    parser.add_argument('-i', '--input', help="Input video path or URL")
    parser.add_argument('-sl', '--source_language', default='English', help="Source language (default: English)")
    parser.add_argument('-tl', '--target_language', help="Target language")
    parser.add_argument('-llm-char', type=int, default=CHAR_LIMIT_DEFAULT, help=f"Character limit for translation (default: {CHAR_LIMIT_DEFAULT})")
    parser.add_argument('-ant_api', help="Anthropic API key")
    parser.add_argument('-evaluate', action='store_true', help="Perform evaluation of translations")
    parser.add_argument('-translation_memory', action='store_true', help="Enable translation memory/glossary feature")
    parser.add_argument('-tts_voice', help="Path to TTS voice WAV file")
    parser.add_argument('-whisper_model', choices=['base', 'small', 'small.en', 'medium', 'medium.en', 'large-v2', 'large-v3'], default='large-v2', help="Whisper model to use for transcription (default: large-v2)")
    parser.add_argument('-align_model', help="Custom alignment model for WhisperX (Hugging Face ID or local path). If not set, defaults based on source language.") # New argument
    parser.add_argument('-openai_api', help="OpenAI API key")
    parser.add_argument('-llmapi', choices=['anthropic', 'openai', 'local', 'deepl', 'openrouter', 'gemini'],
                        default='anthropic', help="LLM API to use (default: anthropic)")
    parser.add_argument('-llm-model', choices=['haiku', 'sonnet', 'gpt-4.1', 'gpt-4.1-mini',
                    'deepseek-r1', 'qwq-32b', 'deepseek-v3', 'gemini-pro', 'gemini-flash'],
                    help="LLM model to use")
    parser.add_argument('-gemini_api', help="Google Gemini API key")
    parser.add_argument('-session', help="Session name or path. If not provided, a new session folder will be created.")
    parser.add_argument('-merge_threshold', type=int, default=SPEECH_BLOCK_MERGE_THRESHOLD, help=f"Maximum time difference (in ms) between subtitles to be merged (default: {SPEECH_BLOCK_MERGE_THRESHOLD})")
    parser.add_argument('-task', choices=['tts', 'full', 'transcribe', 'translate', 'speech_blocks', 'sync', 'equalize', 'correct'], default='full', help="Task to perform (default: full)")
    parser.add_argument('-t_prompt', help="Custom translation prompt")
    parser.add_argument('-eval_prompt', help="Custom evaluation prompt")
    parser.add_argument('-gloss_prompt', help="Custom glossary prompt")
    parser.add_argument('-sys_prompt', help="Custom system prompt")
    parser.add_argument('-equalize', action='store_true', help="Apply SRT equalizer to the final subtitle file")
    parser.add_argument('-max_line_length', type=int, default=42, help="Maximum line length for SRT equalization (default: 42)") # Used with -equalize flag
    parser.add_argument('-api_deepl', help="DeepL API key")
    parser.add_argument('-characters', type=int, default=60, help="Maximum line length for SRT equalization (default: 60)") # Used with task=='equalize'
    parser.add_argument('-v', '--video', help="Input video file for syncing (optional)")
    parser.add_argument('-correct', action='store_true', help="Enable subtitle correction before translation")
    parser.add_argument('-correct_prompt', help="Additional context/instructions for subtitle correction (optional)")
    parser.add_argument('-cot', action='store_true', help="Enable chain-of-thought prompting")
    parser.add_argument('-context', action='store_true', help="Add previous output as context")
    parser.add_argument('-translate_prompt', help="Additional context/instructions for translation (optional)")
    parser.add_argument('-thinking', action='store_true', help="Enable Claude's extended thinking (only for Sonnet model)")
    parser.add_argument('-thinking_tokens', type=int, default=4000, help="Budget tokens for Claude's extended thinking (default: 4000)")
    parser.add_argument('--delay_start', type=int, default=1500, 
                        help="Delay audio start by this many milliseconds when subtitle is longer than audio (default: 1500)")
    parser.add_argument('--speed_up', type=int, default=115, 
                        help="Maximum speed-up percentage when audio is longer than subtitle (default: 100, meaning no speed-up)")

    # OpenRouter provider routing arguments
    parser.add_argument('-provider', help="OpenRouter provider to prioritize (comma-separated, e.g., 'Anthropic,OpenAI')")
    parser.add_argument('-sort', choices=['price', 'throughput', 'latency'], help="OpenRouter provider sorting strategy")
    parser.add_argument('-fallbacks', dest='allow_fallbacks', action='store_true', default=True, help="Allow fallbacks to other providers (default: True)")
    parser.add_argument('-no-fallbacks', dest='allow_fallbacks', action='store_false', help="Disable fallbacks to other providers")
    parser.add_argument('-ignore', help="OpenRouter providers to ignore (comma-separated list)")
    parser.add_argument('-data-collection', choices=['allow', 'deny'], default='allow', help="OpenRouter data collection policy")
    parser.add_argument('-require-parameters', action='store_true', help="Require providers to support all parameters")
    parser.add_argument('-whisper_prompt', help="Custom initial prompt to guide Whisper transcription. If not specified, uses a default prompt for better punctuation.")


    args = parser.parse_args()

    if args.thinking and (args.llmapi != 'anthropic' or args.llm_model != 'sonnet'):
        logging.warning("Extended thinking is only available with Claude Sonnet model. Ignoring -thinking parameter.")
        args.thinking = False

    if args.task != 'sync' and not args.input: # Also 'equalize' task needs input. This check will be done later for 'equalize'.
        parser.error("the following arguments are required: -i/--input (unless task is 'sync')")

    # Determine LLM API if model implies it
    if args.llm_model in ['gpt-4.1', 'gpt-4.1-mini']:
        args.llmapi = 'openai'
    elif args.llm_model in ['gemini-pro', 'gemini-flash']:
        args.llmapi = 'gemini'

    # Model name mappings
    openai_model_mapping = {
        'gpt-4.1': 'gpt-4.1',
        'gpt-4.1-mini': 'gpt-4.1-mini'
    }

    gemini_model_mapping = {
        'gemini-flash': 'gemini-2.5-flash-preview-05-20',
        'gemini-pro': 'gemini-2.5-pro-preview-05-06'
    }

    model = None # Initialize model
    if args.llmapi == 'anthropic':
        if not args.llm_model or args.llm_model not in ['haiku', 'sonnet']:
            args.llm_model = 'sonnet' # Default for anthropic
        # Using the exact model names from the original script for Anthropic
        model = "claude-3-haiku-20240307" if args.llm_model == 'haiku' else "claude-3-5-sonnet-20240620"
    elif args.llmapi == 'openai':
        if not args.llm_model or args.llm_model not in ['gpt-4.1', 'gpt-4.1-mini']:
            args.llm_model = 'gpt-4.1-mini' # Default for openai
        model = openai_model_mapping[args.llm_model]
    elif args.llmapi == 'openrouter':
        if not args.llm_model or args.llm_model not in ['deepseek-r1', 'qwq-32b', 'deepseek-v3']:
            args.llm_model = 'deepseek-r1' # Default for openrouter
        # Corrected logic for OpenRouter model name construction
        if 'deepseek' in args.llm_model: # Handles 'deepseek-r1', 'deepseek-v3'
            model = f"deepseek/{args.llm_model}" # e.g. deepseek/deepseek-r1, deepseek/deepseek-v3
        elif args.llm_model == 'qwq-32b':
            model = f"qwen/{args.llm_model}" # e.g. qwen/qwq-32b (or qwen/qwen-32b-chat if that's the ID)
        else: # Should ideally not be reached if default is set and choices are exhaustive
            model = args.llm_model # Fallback, user provides full ID

        # Handle model name shortcuts for OpenRouter (appending :nitro or :floor)
        if model: # Ensure model is assigned before trying to append
            if args.sort == 'throughput' and ':nitro' not in model:
                model += ':nitro'
            elif args.sort == 'price' and ':floor' not in model:
                model += ':floor'
    elif args.llmapi == 'gemini':
        if not args.llm_model or args.llm_model not in ['gemini-pro', 'gemini-flash']:
            args.llm_model = 'gemini-flash' # Default for gemini
        model = gemini_model_mapping[args.llm_model] # Uses the user-constrained mapping
    elif args.llmapi == 'local':
        model = None # No specific model name for local, it's configured in the local server
    elif args.llmapi == 'deepl':
        model = None
        args.api_deepl = args.api_deepl or os.environ.get('DEEPL_API_KEY') or input("Please enter your DeepL API key: ")
    else:
        raise ValueError(f"Unsupported LLM API: {args.llmapi}")

    # Build OpenRouter provider parameters
    provider_params = None
    if args.llmapi == 'openrouter':
        provider_params = {}
        if args.provider:
            provider_params['order'] = [p.strip() for p in args.provider.split(',')]
        if args.sort: # This is a separate param for OpenRouter's routing logic
            provider_params['sort'] = args.sort
        if hasattr(args, 'allow_fallbacks'): # Check existence due to dest
            provider_params['allow_fallbacks'] = args.allow_fallbacks
        if args.ignore:
            provider_params['ignore'] = [p.strip() for p in args.ignore.split(',')]
        if args.data_collection:
            provider_params['data_collection'] = args.data_collection
        if args.require_parameters:
            provider_params['require_parameters'] = True
        if not provider_params: # Only include if there's something in it
            provider_params = None

    # Set default alignment models based on language if not specified by user
    if not args.align_model:
        language_align_models = {
            "pl": "jonatasgrosman/wav2vec2-xls-r-1b-polish",
            "nl": "GroNLP/wav2vec2-dutch-large-ft-cgn",
            "de": "aware-ai/wav2vec2-xls-r-1b-german",
            "en": "jonatasgrosman/wav2vec2-xls-r-1b-english",
            "fr": "jonatasgrosman/wav2vec2-xls-r-1b-french",
            "it": "jonatasgrosman/wav2vec2-xls-r-1b-italian",
            "ru": "jonatasgrosman/wav2vec2-xls-r-1b-russian",
            "es": "jonatasgrosman/wav2vec2-xls-r-1b-spanish"
        }
        source_lang_lower = args.source_language.lower()
        lang_to_code_map = {
            "english": "en",
            "polish": "pl", "polski": "pl",
            "dutch": "nl", "nederlands": "nl",
            "german": "de", "deutsch": "de",
            "french": "fr", "français": "fr",
            "italian": "it", "italiano": "it",
            "russian": "ru", "русский": "ru", "rus": "ru",
            "spanish": "es", "español": "es",
            # Direct short codes
            "en": "en", "pl": "pl", "nl": "nl", "de": "de",
            "fr": "fr", "it": "it", "ru": "ru", "es": "es"
        }
        lookup_code = lang_to_code_map.get(source_lang_lower)
        if lookup_code:
            args.align_model = language_align_models.get(lookup_code)
            if args.align_model:
                logging.info(f"Using default alignment model for {args.source_language} (lookup code: {lookup_code}): {args.align_model}")
            else:
                 logging.info(f"Language code {lookup_code} for {args.source_language} found, but no specific default alignment model listed in internal map. WhisperX will use its own default.")
        else:
            logging.info(f"No mapping found for source language '{args.source_language}' to a short code for default alignment model lookup. WhisperX will use its own default alignment model.")
    else:
        logging.info(f"User-specified alignment model: {args.align_model}")


    if args.task == 'equalize':
        if not args.input:
            parser.error("the following arguments are required for 'equalize' task: -i/--input")
        input_srt_path = os.path.abspath(os.path.expanduser(args.input))
        output_srt_path = os.path.splitext(input_srt_path)[0] + "_equalized.srt"
        logging.info(f"Performing SRT equalization on: {input_srt_path}")
        perform_equalization(input_srt_path, output_srt_path, args.characters) # Uses args.characters (default 60)
        logging.info(f"Equalized SRT file saved as: {output_srt_path}")
        return

    if args.task == 'sync':
        if not args.session:
            parser.error("Session folder (--session) must be specified for the 'sync' task")
        sync_audio_video(args.session, args.video, args.delay_start, args.speed_up)
        logging.info("Synchronization completed. Ending process.")
        return

    # General setup for tasks requiring input file
    video_path = ""
    video_name = ""
    if args.input.startswith(('http://', 'https://', 'www.')):
        logging.info(f"Detected URL input: {args.input}")
        temp_session_folder = get_or_create_session_folder("temp_download") # Ensure unique name
        video_path, video_name = download_from_url(args.input, temp_session_folder)
        logging.info(f"Video downloaded: {video_path}")

        session_folder = get_or_create_session_folder(video_name, args.session)
        if session_folder != temp_session_folder:
            # Ensure target directory exists for shutil.move
            os.makedirs(os.path.dirname(os.path.join(session_folder, os.path.basename(video_path))), exist_ok=True)
            shutil.move(video_path, os.path.join(session_folder, os.path.basename(video_path)))
            # Only remove temp_session_folder if it's not the same as session_folder and it exists
            if os.path.exists(temp_session_folder) and os.path.abspath(temp_session_folder) != os.path.abspath(session_folder):
                 try:
                    shutil.rmtree(temp_session_folder)
                 except OSError as e:
                    logging.warning(f"Could not remove temporary download folder {temp_session_folder}: {e}")

        video_path = os.path.join(session_folder, os.path.basename(video_path)) # Update video_path to new location
    else:
        video_path = os.path.abspath(os.path.expanduser(args.input))
        video_name_raw = os.path.splitext(os.path.basename(video_path))[0]
        video_name = ''.join(e for e in video_name_raw if e.isalnum() or e in ['-', '_'])
        if not video_name: # Handle cases where sanitization results in empty name
            video_name = "default_video_name"
        session_folder = get_or_create_session_folder(video_name, args.session)

    setup_logging(session_folder)

    logging.info("Starting subtitle translation and dubbing process")
    logging.info(f"Input file: {video_path}")
    logging.info(f"Session folder: {session_folder}")
    logging.info(f"Using LLM API: {args.llmapi}")
    if model: logging.info(f"Using LLM model: {model}")
    if provider_params: logging.info(f"OpenRouter provider params: {provider_params}")

    translation_prompt_template_to_use = args.t_prompt if args.t_prompt else (TRANSLATION_PROMPT_TEMPLATE_COT if args.cot else TRANSLATION_PROMPT_TEMPLATE)
    evaluation_prompt_template_to_use = args.eval_prompt if args.eval_prompt else EVALUATION_PROMPT_TEMPLATE
    glossary_prompt_instructions_to_use = args.gloss_prompt if args.gloss_prompt else GLOSSARY_INSTRUCTIONS_TRANSLATION
    system_prompt_to_use = args.sys_prompt if args.sys_prompt else CUSTOM_SYSTEM_PROMPT
    correction_prompt_template_to_use = args.correct_prompt if args.correct_prompt else (CORRECTION_PROMPT_TEMPLATE_COT if args.cot else CORRECTION_PROMPT_TEMPLATE)


    srt_content = ""
    srt_path = "" # Initialize srt_path

    try:
        if args.task in ['full', 'translate', 'translation', 'correct', 'transcribe', 'speech_blocks']: # Tasks that need SRT
            logging.info(f"Current task: {args.task}")
            if args.llmapi == 'anthropic':
                args.ant_api = args.ant_api or os.environ.get('ANTHROPIC_API_KEY') or input("Please enter your Anthropic API key: ")
            elif args.llmapi == 'openai':
                args.openai_api = args.openai_api or os.environ.get('OPENAI_API_KEY') or input("Please enter your OpenAI API key: ")
            elif args.llmapi == 'gemini':
                args.gemini_api = args.gemini_api or os.environ.get('GEMINI_API_KEY') or input("Please enter your Google Gemini API key: ")
            elif args.llmapi == 'openrouter':
                if not os.environ.get('OPENROUTER_API'):
                    raise ValueError("OPENROUTER_API environment variable must be set")
            elif args.llmapi == 'local':
                try:
                    response = requests.get("http://127.0.0.1:5000/v1/models") # Test connection
                    response.raise_for_status()
                    logging.info("Successfully connected to the local Text Generation WebUI API")
                except requests.RequestException as e:
                    logging.error(f"Failed to connect to the local Text Generation WebUI API: {str(e)}")
                    logging.error("Please ensure that the Text Generation WebUI is running and accessible at http://127.0.0.1:5000")
                    return # Exit if local API not available

            if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.wav', '.m4a', '.aac', '.flac', '.mp3', '.ogg', '.opus')):
                if not os.path.exists(video_path):
                     parser.error(f"Video/Audio file not found: {video_path}")
                audio_path = extract_audio(video_path, session_folder, video_name)
                logging.info(f"Audio extracted: {audio_path}")
                whisper_prompt = args.whisper_prompt or DEFAULT_WHISPER_PROMPT
                srt_path = transcribe_audio(audio_path, args.source_language, session_folder, video_name, args.whisper_model, args.align_model, whisper_prompt)
                logging.info(f"Transcription completed: {srt_path}")
            elif video_path.lower().endswith('.srt'):
                if not os.path.exists(video_path):
                     parser.error(f"SRT file not found: {video_path}")
                srt_path = os.path.join(session_folder, f"{video_name}_input.srt") # Copy to session folder
                shutil.copy(video_path, srt_path)
                logging.info(f"Using input subtitles: {srt_path}")
            else:
                parser.error(f"Unsupported input file type: {video_path}. Must be video, audio, or SRT.")

            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()

            original_srt_content = srt_content
            srt_content = renumber_subtitles(srt_content)
            if srt_content != original_srt_content:
                renumbered_srt_path = os.path.join(session_folder, f"{video_name}_renumbered.srt")
                with open(renumbered_srt_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
                logging.info(f"Non-consecutive subtitle numbers detected. Renumbered subtitles saved: {renumbered_srt_path}")
                # srt_path = renumbered_srt_path # Update srt_path to the renumbered one if further processing uses it by path

        # Handle the correct task (standalone) or correction flag
        if args.task == 'correct' or args.correct:
            logging.info("Starting subtitle correction process...")
            # Use original srt_content for creating initial blocks
            correction_translation_blocks = create_translation_blocks(srt_content, args.llm_char, args.source_language)
            corrected_blocks = correct_subtitles(
                correction_translation_blocks,
                args.source_language,
                args.correct_prompt or "",
                args.ant_api,
                args.openai_api,
                args.llmapi,
                model,
                correction_prompt_template_to_use,
                system_prompt_to_use,
                use_cot=args.cot,
                use_context=args.context,
                provider_params=provider_params,
                use_thinking=args.thinking,
                thinking_tokens=args.thinking_tokens,
                gemini_api_key=args.gemini_api
            )
            logging.info("Correction completed.")

            corrected_srt = parse_translated_response(corrected_blocks, srt_content) # srt_content here is the renumbered original
            corrected_srt_filename_base = f"{video_name}_{args.source_language}_corrected"
            corrected_srt_path = os.path.join(session_folder, f"{corrected_srt_filename_base}.srt")
            with open(corrected_srt_path, 'w', encoding='utf-8') as f:
                f.write(corrected_srt)
            logging.info(f"Corrected subtitles saved: {corrected_srt_path}")

            srt_content = corrected_srt # Update srt_content for subsequent operations
            srt_path = corrected_srt_path # Update srt_path for potential equalization

            if args.task == 'correct': # If standalone correction task
                if args.equalize: # If -equalize flag is also on for 'correct' task
                    output_srt_eq_path = os.path.join(session_folder, f"{corrected_srt_filename_base}_final.srt")
                    equalize_srt(corrected_srt_path, output_srt_eq_path, args.max_line_length) # Uses max_line_length (default 42)
                    logging.info(f"Equalized corrected subtitles saved: {output_srt_eq_path}")
                logging.info("Correction task completed. Ending process.")
                return

        if args.task == 'transcribe':
            logging.info("Transcription/correction (if enabled) completed. Ending process.")
            if args.equalize: # If -equalize flag for 'transcribe' task
                output_srt_eq_path = os.path.join(session_folder, f"{video_name}_{args.source_language}{'_corrected' if args.correct else ''}_final.srt")
                equalize_srt(srt_path, output_srt_eq_path, args.max_line_length) # Uses max_line_length (default 42)
                logging.info(f"Equalized subtitles saved: {output_srt_eq_path}")
            return

        final_srt = "" # Initialize to be used for speech blocks
        translated_srt_path = "" # Initialize

        if args.task in ['full', 'translate', 'translation']:
            if not args.target_language:
                parser.error("Target language (--target_language) is required for translation tasks.")

            translation_blocks = create_translation_blocks(srt_content, args.llm_char, args.source_language)
            logging.info(f"Created {len(translation_blocks)} translation blocks from content of length {len(srt_content)}")

            glossary = manage_glossary(session_folder) if args.translation_memory else {}
            evaluation_suffix = "" # For filename

            if args.llmapi == 'deepl':
                if not args.api_deepl: parser.error("DeepL API key (--api_deepl) is required.")
                translated_blocks_responses = translate_blocks_deepl(translation_blocks, args.source_language, args.target_language, args.api_deepl)
                logging.info("DeepL translation completed")
                final_blocks_responses = translated_blocks_responses
            else: # Anthropic, OpenAI, OpenRouter, Gemini, Local LLMs
                translated_blocks_responses, updated_glossary = translate_blocks(
                    translation_blocks,
                    args.source_language,
                    args.target_language,
                    args.ant_api,
                    args.openai_api,
                    args.llmapi,
                    glossary,
                    args.translation_memory,
                    args.evaluate, # Pass evaluate flag to decide if glossary is updated here or later
                    model,
                    translation_prompt_template_to_use,
                    args.translate_prompt, # Specific additional instructions for translation
                    glossary_prompt_instructions_to_use,
                    system_prompt_to_use,
                    use_cot=args.cot,
                    use_context=args.context,
                    provider_params=provider_params,
                    use_thinking=args.thinking,
                    thinking_tokens=args.thinking_tokens,
                    gemini_api_key=args.gemini_api
                )
                logging.info("LLM Translation completed")

                if args.translation_memory and not args.evaluate:
                    save_glossary(session_folder, updated_glossary)
                    logging.info("Updated glossary saved (pre-evaluation or no evaluation)")

                if args.evaluate:
                    logging.info("Starting evaluation of translations")
                    if not updated_glossary: updated_glossary = glossary # ensure glossary is passed
                    final_blocks_responses, final_glossary = evaluate_translation(
                        translation_blocks,
                        translated_blocks_responses,
                        args.source_language,
                        args.target_language,
                        args.ant_api,
                        args.openai_api,
                        args.llmapi,
                        updated_glossary, # Pass potentially updated glossary from translation step
                        args.translation_memory,
                        model, # Use same model for eval or allow different? Current setup uses same.
                        evaluation_prompt_template_to_use,
                        system_prompt_to_use,
                        provider_params=provider_params,
                        use_thinking=args.thinking,
                        thinking_tokens=args.thinking_tokens,
                        gemini_api_key=args.gemini_api
                    )
                    logging.info("Evaluation completed")
                    evaluation_suffix = "_eval"

                    if args.translation_memory:
                        save_glossary(session_folder, final_glossary)
                        logging.info("Final glossary saved after evaluation")
                else:
                    final_blocks_responses = translated_blocks_responses
                    # No evaluation_suffix if not evaluated

            final_json_path = os.path.join(session_folder, f"{video_name}_{args.target_language}{evaluation_suffix}_final_blocks.json")
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_blocks_responses, f, ensure_ascii=False, indent=2)
            logging.info(f"Final translation blocks (JSON) saved: {final_json_path}")

            if args.llmapi == 'deepl':
                final_srt = parse_deepl_response(final_blocks_responses, srt_content) # srt_content is the original (potentially corrected) source SRT
            else:
                final_srt = parse_translated_response(final_blocks_responses, srt_content) # srt_content is the original (potentially corrected) source SRT

            translated_srt_path = os.path.join(session_folder, f"{video_name}_{args.target_language}{evaluation_suffix}.srt")
            with open(translated_srt_path, 'w', encoding='utf-8') as f:
                f.write(final_srt)
            logging.info(f"Translated subtitles saved: {translated_srt_path}")

            if args.task in ['translate', 'translation']:
                logging.info("Translation (and evaluation if requested) completed. Ending process.")
                if args.equalize:
                    output_srt_eq_path = os.path.join(session_folder, f"{video_name}_{args.target_language}{evaluation_suffix}_final.srt")
                    equalize_srt(translated_srt_path, output_srt_eq_path, args.max_line_length) # Uses max_line_length (default 42)
                    logging.info(f"Equalized translated subtitles saved: {output_srt_eq_path}")
                return

        # Prepare srt_for_speech_blocks:
        # If translation happened, use final_srt. Otherwise (e.g. task 'speech_blocks' on original SRT), use srt_content.
        srt_for_speech_blocks = final_srt if final_srt else srt_content
        lang_for_speech_blocks = args.target_language if args.task == 'full' and args.target_language else args.source_language

        if args.task in ['full', 'speech_blocks']:
            if not srt_for_speech_blocks:
                 parser.error("SRT content is empty, cannot create speech blocks. Ensure transcription or translation ran successfully.")
            speech_blocks = create_speech_blocks(
                srt_for_speech_blocks,
                session_folder, video_name,
                lang_for_speech_blocks,
                merge_threshold=args.merge_threshold
            )
            logging.info(f"Created {len(speech_blocks)} speech blocks for language {lang_for_speech_blocks}")

            if args.task == 'speech_blocks':
                logging.info("Speech blocks creation completed. Ending process.")
                # Optionally equalize the SRT that speech blocks were based on
                if args.equalize:
                    # Determine which SRT was used for speech blocks
                    base_srt_path_for_speech_blocks = translated_srt_path if final_srt else srt_path
                    if base_srt_path_for_speech_blocks and os.path.exists(base_srt_path_for_speech_blocks):
                        output_srt_eq_path = os.path.splitext(base_srt_path_for_speech_blocks)[0] + "_final.srt"
                        equalize_srt(base_srt_path_for_speech_blocks, output_srt_eq_path, args.max_line_length)
                        logging.info(f"Equalized base SRT for speech blocks saved: {output_srt_eq_path}")
                    else:
                        logging.warning("Could not find base SRT path to equalize for speech_blocks task.")
                return

        if args.task == 'full':
            if not args.tts_voice:
                tts_voices_folder = "tts-voices"
                if os.path.exists(tts_voices_folder):
                    wav_files = [f for f in os.listdir(tts_voices_folder) if f.endswith('.wav')]
                    if wav_files:
                        args.tts_voice = os.path.join(tts_voices_folder, wav_files[0]) # Pick first one
                        logging.info(f"Using first available TTS voice from 'tts-voices' folder: {args.tts_voice}")
                    else:
                        parser.error("No WAV files found in the 'tts-voices' folder and no TTS voice specified via --tts_voice.")
                else:
                    parser.error("No TTS voice specified via --tts_voice and 'tts-voices' folder not found.")

            try:
                tts_language = get_xtts_language_code(args.target_language)
            except ValueError as e:
                logging.error(str(e))
                return # Exit if language not supported for TTS

            audio_files = generate_tts_audio(speech_blocks, args.tts_voice, tts_language, session_folder, video_name)
            logging.info(f"Generated {len(audio_files)} TTS audio files")

            if not audio_files:
                logging.error("No TTS audio files were generated. Cannot proceed with alignment and mixing.")
                return

            # Determine if evaluation suffix should be used for alignment call
            use_eval_suffix_for_alignment = args.evaluate and evaluation_suffix == "_eval"

            alignment_blocks = create_alignment_blocks(session_folder, video_name, args.target_language, use_eval_suffix_for_alignment)
            logging.info(f"Created {len(alignment_blocks)} alignment blocks")

            aligned_audio_path = align_audio_blocks(alignment_blocks, session_folder, args.delay_start, args.speed_up)
            logging.info(f"Audio alignment completed: {aligned_audio_path}")

            if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                final_output_path = mix_audio_tracks(video_path, aligned_audio_path, session_folder, video_name, args.target_language, use_eval_suffix_for_alignment)
                logging.info(f"Final output saved: {final_output_path}")
            else: # If input was audio or SRT, just save the aligned audio
                logging.info(f"Aligned audio (dub) saved: {aligned_audio_path}")
                # Optionally, copy the aligned audio to a more prominent name if it's the final product for an audio-only workflow
                final_dub_path = os.path.join(session_folder, f"{video_name}_{args.target_language}{evaluation_suffix}_dubbed_audio.wav")
                shutil.copy(aligned_audio_path, final_dub_path)
                logging.info(f"Final dubbed audio also saved as: {final_dub_path}")


            if args.equalize: # Equalize the final translated SRT
                if translated_srt_path and os.path.exists(translated_srt_path):
                    output_srt_eq_path = os.path.splitext(translated_srt_path)[0] + "_final.srt"
                    equalize_srt(translated_srt_path, output_srt_eq_path, args.max_line_length) # Uses max_line_length (default 42)
                    logging.info(f"Equalized final translated subtitles saved: {output_srt_eq_path}")
                else:
                    logging.warning("Translated SRT path not found for final equalization.")

            cleanup_temp_files(session_folder, args.task)
            logging.info("Temporary files cleaned up")

        logging.info("Process completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        # raise # Optionally re-raise for clearer exit in some environments

if __name__ == "__main__":
    main()
