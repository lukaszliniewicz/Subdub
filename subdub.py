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

#Ideas
#Add previous corrected / translated segment to context 
#Experiment with the validation in chain of thought
#Add numbering to subtitles
#Try CSV instead of JSON
#Implement openrouter
#Write app to compare / edit subtitles and their translation, including splitting, removing with renumbering, consider including in Pandrator flow?
#Improve glossary workflow, enable passing a csv file
#Improve handling of Japanese subtitles (character limit etc.)

# Constants
MAX_RETRIES = 3
CHAR_LIMIT_DEFAULT = 2000
SPEECH_BLOCK_CHAR_LIMIT = 160
SPEECH_BLOCK_MIN_CHARS = 20
SPEECH_BLOCK_MERGE_THRESHOLD = 1  # ms

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
1. You will receive an array of subtitles. Your task is to correct them
2. Fix punctuation and capitalization such that they are coherent and logical, also between subsequent subtitles 
3. Correct spelling and obvious transcription errors
4. Preserve all meaning and content (you should remove filler words, though)
5. Return the corrected subtitles in the EXACT SAME array structure with the SAME number of items, including subtitles that you didn't change
6. Do not assume that something needs correcting just because you were asked to consider correcting it, make sure that it really does need correcting
7. DO NOT split or merge subtitles

Additional context and instructions specific to your particular batch, if any:
{correction_instructions}

Remember, validate your output carefully before returning it. Your most important instruction: the number of corrected items in the output array MUST match the number of items received in the input array, it is IMPERATIVE.
"""

CORRECTION_PROMPT_TEMPLATE_COT = """
You will receive an array of subtitles. Your task is to correct them.

1. Draft your response first inside <draft></draft> tags, then check it inside <analysis></analysis> tags, especially whether the number of output subtitles matches the number of input subtitles, and output the validated response within <final></final> tags. Make sure to close the tags.
2. Fix punctuation and capitalization such that they are coherent and logical, also between subsequent subtitles 
3. Correct spelling and obvious transcription errors
4. Preserve all meaning and content, also stylistic phrases, even if not key to the meaning (you should remove filler words - like "um" - and obvious repetitions - like "it is, it is..." though)
5. Return the corrected subtitles in the EXACT SAME array structure with the SAME number of items, including subtitles that you didn't change 
6. Do not assume that something needs correcting just because you were asked to consider correcting it, make sure that it really does need correcting
7. DO NOT split or merge subtitles

Additional context and instructions specific to your particular batch, if any:
{correction_instructions}

Remember, validate your output carefully before returning it. Your most important instruction: the number of corrected items in the output array MUST match the number of items received in the input array, it is IMPERATIVE.
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
                   use_thinking: bool = False, thinking_tokens: int = 8000) -> str:
    """
    Send a request to a language model API and process the response.
    
    Args:
        client: The API client instance
        llm_api: Which API to use ('anthropic', 'openai', etc.)
        model: The specific model to use
        messages: List of message dictionaries to send
        system_prompt: Optional system prompt for the model
        provider_params: Optional parameters for OpenRouter API
        use_thinking: Whether to enable Claude's extended thinking
        thinking_tokens: Budget for Claude's thinking process
        
    Returns:
        Processed response text
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
            
            # First, extract all thinking blocks and log them
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
                temperature=0.7,
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
                "temperature": 0.7,
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
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '44100',
        '-ac', '2',
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

def transcribe_audio(audio_path: str, language: str, session_folder: str, video_name: str, whisper_model: str) -> str:
    output_srt = os.path.join(session_folder, f"{video_name}.srt")
    base_whisperx_args = [
        audio_path,
        '--model', whisper_model,
        '--language', language,
        '--output_format', 'srt',
        '--output_dir', session_folder,
        '--print_progress', 'True'
    ]

    # First attempt with direct whisperx command
    try:
        whisperx_command = ['whisperx'] + base_whisperx_args
        logging.info("Attempting direct whisperx command...")
        result = subprocess.run(whisperx_command, check=True, capture_output=True)
        if result.stderr:
            logging.warning(f"WhisperX warning: {safe_decode(result.stderr)}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.warning(f"Direct whisperx command failed, trying conda run method. Error: {str(e)}")
        try:
            conda_whisperx_command = [
                "../conda/Scripts/conda.exe", "run", "-p", "../conda/envs/whisperx_installer", "--no-capture-output",
                "python", "-m", "whisperx"
            ] + base_whisperx_args
            logging.info("Attempting conda run whisperx command...")
            result = subprocess.run(conda_whisperx_command, check=True, capture_output=True)
            if result.stderr:
                logging.warning(f"WhisperX warning: {safe_decode(result.stderr)}")
        except subprocess.CalledProcessError as e:
            logging.error(f"WhisperX command failed using both methods. Error output:\n{safe_decode(e.stderr)}")
            raise
    
    whisperx_output = os.path.join(session_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}.srt")
    os.rename(whisperx_output, output_srt)
    
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
                    thinking_tokens: int = 8000) -> Tuple[List[Dict[str, Union[str, List[str]]]], Dict[str, str]]:
    
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
    thinking_tokens: int = 8000
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

def create_speech_blocks(
    srt_content: str,
    session_folder: str,
    video_name: str,
    target_language: str,
    min_chars: int = SPEECH_BLOCK_MIN_CHARS,
    max_chars: int = SPEECH_BLOCK_CHAR_LIMIT,
    merge_threshold: int = SPEECH_BLOCK_MERGE_THRESHOLD
) -> List[Dict]:
    CONJUNCTIONS = {
        "en": ["and", "but", "or", "because", "although"],
        "es": ["y", "pero", "o", "porque", "aunque"],
        "fr": ["et", "mais", "ou", "parce que", "bien que"],
        "de": ["und", "aber", "oder", "weil", "obwohl"],
        "it": ["e", "ma", "o", "perché", "sebbene"],
        "pt": ["e", "mas", "ou", "porque", "embora"],
        "pl": ["i", "ale", "lub", "ponieważ", "chociaż"],
        "tr": ["ve", "ama", "veya", "çünkü", "rağmen"],
        "ru": ["и", "но", "или", "потому что", "хотя"],
        "nl": ["en", "maar", "of", "omdat", "hoewel"],
        "cs": ["a", "ale", "nebo", "protože", "ačkoli"],
        "ar": ["و", "لكن", "أو", "لأن", "رغم أن"],
        "zh-cn": ["和", "但是", "或者", "因为", "虽然"],
        "ja": ["そして", "しかし", "または", "なぜなら", "にもかかわらず"],
        "hu": ["és", "de", "vagy", "mert", "bár"],
        "ko": ["그리고", "하지만", "또는", "왜냐하면", "비록"]
    }

    def get_xtts_language_code(target_language: str) -> str:
        xtts_language_map = {
            "English": "en", "Spanish": "es", "French": "fr", "German": "de",
            "Italian": "it", "Portuguese": "pt", "Polish": "pl", "Turkish": "tr",
            "Russian": "ru", "Dutch": "nl", "Czech": "cs", "Arabic": "ar",
            "Chinese": "zh-cn", "Japanese": "ja", "Hungarian": "hu", "Korean": "ko"
        }
        return xtts_language_map.get(target_language, "en")

    def merge_close_subtitles(subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
        merged = []
        for subtitle in subtitles:
            if not merged or (subtitle.start - merged[-1].end).total_seconds() * 1000 > merge_threshold:
                merged.append(subtitle)
            else:
                merged[-1] = srt.Subtitle(
                    index=merged[-1].index,
                    start=merged[-1].start,
                    end=subtitle.end,
                    content=merged[-1].content + " " + subtitle.content
                )
        return merged

    def find_split_point(text: str, max_length: int, min_length: int, language: str) -> int:
        sentence_ends = [i for i, char in enumerate(text[:max_length]) if char in '.!?']
        clause_ends = [i for i, char in enumerate(text[:max_length]) if char in ',;:']
        conjunctions = CONJUNCTIONS.get(language, [])
        
        for end in sentence_ends:
            if min_length <= end < max_length:
                return end + 1

        for end in clause_ends:
            if min_length <= end < max_length:
                return end + 1

        for conj in conjunctions:
            pattern = r'\s' + re.escape(conj) + r'\s'
            matches = list(re.finditer(pattern, text[:max_length]))
            for match in reversed(matches):
                if min_length <= match.start() < max_length:
                    return match.start()

        spaces = [i for i, char in enumerate(text[:max_length]) if char == ' ']
        if spaces:
            return max(space for space in spaces if space < max_length)

        return max_length

    language_code = get_xtts_language_code(target_language)
    is_cjk = language_code in ['zh-cn', 'ja', 'ko']
    
    subtitles = list(srt.parse(srt_content))
    merged_subtitles = merge_close_subtitles(subtitles)
    
    speech_blocks = []
    
    for subtitle in merged_subtitles:
        if (is_cjk or len(subtitle.content) <= max_chars) and subtitle.content[-1] in '.!?':
            speech_blocks.append({
                "number": str(len(speech_blocks) + 1).zfill(4),
                "text": subtitle.content,
                "subtitles": [subtitle.index]
            })
            continue

        remaining_text = subtitle.content
        subtitle_blocks = []

        while remaining_text:
            if len(remaining_text) <= max_chars:
                subtitle_blocks.append(remaining_text)
                break

            split_point = find_split_point(remaining_text, max_chars, min_chars, language_code)
            
            if split_point == max_chars:
                words = remaining_text[:max_chars].split()
                if len(words) > 1:
                    split_point = len(' '.join(words[:-1]))
                else:
                    split_point = max_chars

            subtitle_blocks.append(remaining_text[:split_point].strip())
            remaining_text = remaining_text[split_point:].strip()

        for i, block in enumerate(subtitle_blocks):
            if i == len(subtitle_blocks) - 1 and len(block) < min_chars and speech_blocks:
                speech_blocks[-1]["text"] += " " + block
                speech_blocks[-1]["subtitles"].append(subtitle.index)
            else:
                speech_blocks.append({
                    "number": str(len(speech_blocks) + 1).zfill(4),
                    "text": block,
                    "subtitles": [subtitle.index]
                })

    # Save the speech blocks as JSON
    json_output_path = os.path.join(session_folder, f"{video_name}_speech_blocks.json")
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(speech_blocks, json_file, ensure_ascii=False, indent=2)

    return speech_blocks
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

def align_audio_blocks(alignment_blocks: List[Dict], session_folder: str) -> str:
    final_audio = AudioSegment.silent(duration=0)
    current_time = 0  # Time in milliseconds
    total_shift = 0

    def timedelta_to_ms(td):
        return td.total_seconds() * 1000

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

        final_audio += block_audio
        current_time += len(block_audio)

        if len(block_audio) > block_duration:
            total_shift += len(block_audio) - block_duration
        else:
            silence_needed = block_duration - len(block_audio)
            if silence_needed > total_shift:
                silence_to_add = silence_needed - total_shift
                final_audio += AudioSegment.silent(duration=silence_to_add)
                current_time += silence_to_add
                total_shift = 0
            else:
                total_shift -= silence_needed

    output_path = os.path.join(session_folder, "aligned_audio.wav")
    final_audio.export(output_path, format="wav")
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
    anthropic_api_key: str,
    openai_api_key: str,
    llm_api: str,
    model: str,
    correction_prompt: str,
    system_prompt: str,
    use_cot: bool = False,
    use_context: bool = False,
    provider_params: Dict = None,
    use_thinking: bool = False,
    thinking_tokens: int = 8000
) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Corrects subtitles using the specified LLM API.
    
    Args:
        translation_blocks: List of blocks of subtitles to correct
        source_lang: Source language
        correction_instructions: Additional correction instructions
        anthropic_api_key: Anthropic API key
        openai_api_key: OpenAI API key
        llm_api: LLM API to use
        model: Model name to use
        correction_prompt: Correction prompt template
        system_prompt: System prompt
        use_cot: Whether to use chain of thought prompting
        use_context: Whether to provide context from previous responses
        provider_params: Optional dictionary of OpenRouter provider parameters
        
    Returns:
        List of corrected responses
    """
    corrected_responses = []
    previous_response = None
    
    base_prompt = (CORRECTION_PROMPT_TEMPLATE_COT if use_cot else CORRECTION_PROMPT_TEMPLATE).format(
        source_lang=source_lang,
        correction_instructions=correction_instructions if correction_instructions else "No additional instructions provided."
    )

    client = Anthropic(api_key=anthropic_api_key) if llm_api == "anthropic" else OpenAI(api_key=openai_api_key) if llm_api == "openai" else None

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
                content = llm_api_request(client, llm_api, model, messages, system_prompt, 
                         provider_params=provider_params, use_thinking=use_thinking, 
                         thinking_tokens=thinking_tokens)
                previous_response = content
                
                try:
                    corrected_subtitles = json.loads(content)
                    if len(corrected_subtitles) != len(block):
                        raise ValueError("Mismatch in subtitle count")
                    
                    corrected_responses.append({
                        "translation": corrected_subtitles,
                        "original_indices": [sub['index'] for sub in block]
                    })
                    break
                except json.JSONDecodeError:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(f"Failed to parse JSON response for block {i+1}")
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise

    return corrected_responses

def sync_audio_video(session_folder: str, input_video: str = None) -> None:
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

    # Align audio blocks
    aligned_audio_path = align_audio_blocks(alignment_blocks, session_folder)
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
    parser.add_argument('-openai_api', help="OpenAI API key")
    parser.add_argument('-llmapi', choices=['anthropic', 'openai', 'local', 'deepl', 'openrouter'], default='anthropic', help="LLM API to use (default: anthropic)")
    parser.add_argument('-llm-model', choices=['haiku', 'sonnet', 'gpt-4o', 'gpt-4o-mini', 'deepseek-r1', 'qwq-32b', 'deepseek-v3'], help="LLM model to use")
    parser.add_argument('-session', help="Session name or path. If not provided, a new session folder will be created.")
    parser.add_argument('-merge_threshold', type=int, default=SPEECH_BLOCK_MERGE_THRESHOLD, help=f"Maximum time difference (in ms) between subtitles to be merged (default: {SPEECH_BLOCK_MERGE_THRESHOLD})")
    parser.add_argument('-task', choices=['tts', 'full', 'transcribe', 'translate', 'speech_blocks', 'sync', 'equalize', 'correct'], default='full', help="Task to perform (default: full)")    
    parser.add_argument('-t_prompt', help="Custom translation prompt")
    parser.add_argument('-eval_prompt', help="Custom evaluation prompt")
    parser.add_argument('-gloss_prompt', help="Custom glossary prompt")
    parser.add_argument('-sys_prompt', help="Custom system prompt")
    parser.add_argument('-equalize', action='store_true', help="Apply SRT equalizer to the final subtitle file")
    parser.add_argument('-max_line_length', type=int, default=42, help="Maximum line length for SRT equalization (default: 60)")
    parser.add_argument('-api_deepl', help="DeepL API key")
    parser.add_argument('-characters', type=int, default=60, help="Maximum line length for SRT equalization (default: 60)")
    parser.add_argument('-v', '--video', help="Input video file for syncing (optional)")
    parser.add_argument('-correct', action='store_true', help="Enable subtitle correction before translation")
    parser.add_argument('-correct_prompt', help="Additional context/instructions for subtitle correction (optional)")
    parser.add_argument('-cot', action='store_true', help="Enable chain-of-thought prompting")
    parser.add_argument('-context', action='store_true', help="Add previous output as context")
    parser.add_argument('-translate_prompt', help="Additional context/instructions for translation (optional)")
    parser.add_argument('-thinking', action='store_true', help="Enable Claude's extended thinking (only for Sonnet model)")
    parser.add_argument('-thinking_tokens', type=int, default=4000, help="Budget tokens for Claude's extended thinking (default: 4000)")
    
    # OpenRouter provider routing arguments
    parser.add_argument('-provider', help="OpenRouter provider to prioritize (comma-separated, e.g., 'Anthropic,OpenAI')")
    parser.add_argument('-sort', choices=['price', 'throughput', 'latency'], help="OpenRouter provider sorting strategy")
    parser.add_argument('-fallbacks', dest='allow_fallbacks', action='store_true', default=True, help="Allow fallbacks to other providers (default: True)")
    parser.add_argument('-no-fallbacks', dest='allow_fallbacks', action='store_false', help="Disable fallbacks to other providers")
    parser.add_argument('-ignore', help="OpenRouter providers to ignore (comma-separated list)")
    parser.add_argument('-data-collection', choices=['allow', 'deny'], default='allow', help="OpenRouter data collection policy")
    parser.add_argument('-require-parameters', action='store_true', help="Require providers to support all parameters")
    
    args = parser.parse_args()

    if args.thinking and (args.llmapi != 'anthropic' or args.llm_model != 'sonnet'):
        logging.warning("Extended thinking is only available with Claude Sonnet model. Ignoring -thinking parameter.")
        args.thinking = False
    
    if args.task != 'sync' and not args.input:
        parser.error("the following arguments are required: -i/--input")

    if args.llm_model in ['gpt-4o', 'gpt-4o-mini']:
        args.llmapi = 'openai'

    openai_model_mapping = {
        'gpt-4o': 'chatgpt-4o-latest',
        'gpt-4o-mini': 'gpt-4o-mini'
    }

    if args.llmapi == 'anthropic':
        if not args.llm_model or args.llm_model not in ['haiku', 'sonnet']:
            args.llm_model = 'sonnet'
        model = "claude-3-5-haiku-latest" if args.llm_model == 'haiku' else "claude-3-7-sonnet-latest"
    elif args.llmapi == 'openai':
        if not args.llm_model or args.llm_model not in ['gpt-4o', 'gpt-4o-mini']:
            args.llm_model = 'gpt-4o-mini'
        model = openai_model_mapping[args.llm_model]
    elif args.llmapi == 'openrouter':
        if not args.llm_model or args.llm_model not in ['deepseek-r1', 'qwq-32b']:
            args.llm_model = 'deepseek-r1'
        model = f"deepseek/{args.llm_model}" if args.llm_model == 'deepseek-r1' else f"qwen/{args.llm_model}"
        
        # Handle model name shortcuts for OpenRouter
        if args.sort == 'throughput' and ':nitro' not in model:
            model += ':nitro'
        elif args.sort == 'price' and ':floor' not in model:
            model += ':floor'
    elif args.llmapi == 'local':
        model = None
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
        if args.sort and ':nitro' not in model and ':floor' not in model:
            provider_params['sort'] = args.sort
        if hasattr(args, 'allow_fallbacks'):
            provider_params['allow_fallbacks'] = args.allow_fallbacks
        if args.ignore:
            provider_params['ignore'] = [p.strip() for p in args.ignore.split(',')]
        if args.data_collection:
            provider_params['data_collection'] = args.data_collection
        if args.require_parameters:
            provider_params['require_parameters'] = True
        # Only include non-empty provider_params
        if not provider_params:
            provider_params = None

    if args.task == 'equalize':
        if not args.input:
            parser.error("the following arguments are required for 'equalize' task: -i/--input")

        input_srt_path = os.path.abspath(os.path.expanduser(args.input))
        output_srt_path = os.path.splitext(input_srt_path)[0] + "_equalized.srt"

        logging.info(f"Performing SRT equalization on: {input_srt_path}")
        perform_equalization(input_srt_path, output_srt_path, args.characters)
        logging.info(f"Equalized SRT file saved as: {output_srt_path}")
        return

    if args.task == 'sync':
        if not args.session:
            raise ValueError("Session folder must be specified for the 'sync' task")
        sync_audio_video(args.session, args.video)
        logging.info("Synchronization completed. Ending process.")
        return 

    if args.input.startswith(('http://', 'https://', 'www.')):
        logging.info(f"Detected URL input: {args.input}")
        temp_session_folder = get_or_create_session_folder("temp")
        video_path, video_name = download_from_url(args.input, temp_session_folder)
        logging.info(f"Video downloaded: {video_path}")
        
        session_folder = get_or_create_session_folder(video_name, args.session)
        if session_folder != temp_session_folder:
            shutil.move(video_path, session_folder)
            shutil.rmtree(temp_session_folder)
        video_path = os.path.join(session_folder, os.path.basename(video_path))
    else:
        video_path = os.path.abspath(os.path.expanduser(args.input))
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_name = ''.join(e for e in video_name if e.isalnum() or e in ['-', '_'])
        session_folder = get_or_create_session_folder(video_name, args.session)

    setup_logging(session_folder)

    logging.info("Starting subtitle translation and dubbing process")
    logging.info(f"Input file: {video_path}")
    logging.info(f"Session folder: {session_folder}")
    logging.info(f"Using LLM API: {args.llmapi}")
    logging.info(f"Using LLM model: {model}")
    if provider_params:
        logging.info(f"OpenRouter provider params: {provider_params}")

    translation_prompt = args.t_prompt if args.t_prompt else TRANSLATION_PROMPT_TEMPLATE
    evaluation_prompt = args.eval_prompt if args.eval_prompt else EVALUATION_PROMPT_TEMPLATE
    glossary_prompt = args.gloss_prompt if args.gloss_prompt else GLOSSARY_INSTRUCTIONS_TRANSLATION
    system_prompt = args.sys_prompt if args.sys_prompt else CUSTOM_SYSTEM_PROMPT

    try:
        if args.task in ['full', 'translate', 'translation', 'correct']:
            logging.info(f"Current task: {args.task}")
            logging.info(f"Selected LLM API: {args.llmapi}")

            if args.llmapi == 'anthropic':
                args.ant_api = args.ant_api or os.environ.get('ANTHROPIC_API_KEY') or input("Please enter your Anthropic API key: ")
            elif args.llmapi == 'openai':
                args.openai_api = args.openai_api or os.environ.get('OPENAI_API_KEY') or input("Please enter your OpenAI API key: ")
            elif args.llmapi == 'openrouter':
                if not os.environ.get('OPENROUTER_API'):
                    raise ValueError("OPENROUTER_API environment variable must be set")
            elif args.llmapi == 'local':
                try:
                    response = requests.get("http://127.0.0.1:5000/v1/models")
                    response.raise_for_status()
                    logging.info("Successfully connected to the local Text Generation WebUI API")
                except requests.RequestException as e:
                    logging.error(f"Failed to connect to the local Text Generation WebUI API: {str(e)}")
                    logging.error("Please ensure that the Text Generation WebUI is running and accessible at http://127.0.0.1:5000")
                    return

        if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.wav')):
            audio_path = extract_audio(video_path, session_folder, video_name)
            logging.info(f"Audio extracted: {audio_path}")
            srt_path = transcribe_audio(audio_path, args.source_language, session_folder, video_name, args.whisper_model)
            logging.info(f"Transcription completed: {srt_path}")
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
        elif video_path.lower().endswith('.srt'):
            with open(video_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            logging.info(f"Using input subtitles: {video_path}")
        else:
            raise ValueError("Unsupported input file format. Please provide a video file or an SRT file.")

        # Handle the correct task
        if args.task == 'correct':
            logging.info("Starting standalone subtitle correction")
            translation_blocks = create_translation_blocks(srt_content, args.llm_char, args.source_language)
            corrected_blocks = correct_subtitles(
                translation_blocks,
                args.source_language,
                args.correct_prompt,
                args.ant_api,
                args.openai_api,
                args.llmapi,
                model,
                CORRECTION_PROMPT_TEMPLATE,
                system_prompt,
                use_cot=args.cot,
                use_context=args.context,
                provider_params=provider_params,
                use_thinking=args.thinking,
                thinking_tokens=args.thinking_tokens
            )
            logging.info("Correction completed")

            # Save corrected subtitles
            corrected_srt = parse_translated_response(corrected_blocks, srt_content)
            corrected_srt_path = os.path.join(session_folder, f"{video_name}_{args.source_language}_corrected.srt")
            with open(corrected_srt_path, 'w', encoding='utf-8') as f:
                f.write(corrected_srt)
            logging.info(f"Corrected subtitles saved: {corrected_srt_path}")

            if args.equalize:
                output_srt = os.path.join(session_folder, f"{video_name}_{args.source_language}_corrected_final.srt")
                equalize_srt(corrected_srt_path, output_srt, args.max_line_length)
                logging.info(f"Equalized corrected subtitles saved: {output_srt}")

            logging.info("Correction task completed. Ending process.")
            return

        # Handle correction flag for other tasks
        if args.correct:
            logging.info("Starting subtitle correction")
            translation_blocks = create_translation_blocks(srt_content, args.llm_char, args.source_language)
            corrected_blocks = correct_subtitles(
                translation_blocks,
                args.source_language,
                args.correct_prompt,
                args.ant_api,
                args.openai_api,
                args.llmapi,
                model,
                CORRECTION_PROMPT_TEMPLATE,
                system_prompt,
                use_cot=args.cot,
                use_context=args.context,
                provider_params=provider_params
            )
            logging.info("Correction completed")

            # Save corrected subtitles
            corrected_srt = parse_translated_response(corrected_blocks, srt_content)
            corrected_srt_path = os.path.join(session_folder, f"{video_name}_{args.source_language}_corrected.srt")
            with open(corrected_srt_path, 'w', encoding='utf-8') as f:
                f.write(corrected_srt)
            logging.info(f"Corrected subtitles saved: {corrected_srt_path}")
            
            # Update srt_content to use corrected version for subsequent operations
            srt_content = corrected_srt
            # Update srt_path for potential equalization
            srt_path = corrected_srt_path

        if args.task == 'transcribe':
            logging.info("Transcription/correction completed. Ending process.")
            if args.equalize:
                output_srt = os.path.join(session_folder, f"{video_name}_{args.source_language}_final.srt")
                equalize_srt(srt_path, output_srt, args.max_line_length)
            return
    
        if args.task in ['full', 'translate', 'translation']:
            translation_blocks = create_translation_blocks(srt_content, args.llm_char, args.source_language)
            logging.info(f"Created {len(translation_blocks)} translation blocks")

            glossary = manage_glossary(session_folder) if args.translation_memory else {}

            if args.llmapi == 'deepl':
                translated_blocks = translate_blocks_deepl(translation_blocks, args.source_language, args.target_language, args.api_deepl)
                logging.info("DeepL translation completed")
                final_blocks = translated_blocks
                evaluation_suffix = ""
            else:
                translated_blocks, updated_glossary = translate_blocks(
                    translation_blocks, 
                    args.source_language, 
                    args.target_language, 
                    args.ant_api,
                    args.openai_api,
                    args.llmapi,
                    glossary, 
                    args.translation_memory, 
                    args.evaluate,
                    model,
                    translation_prompt,
                    args.translate_prompt,
                    glossary_prompt,
                    system_prompt,
                    use_cot=args.cot,
                    use_context=args.context,
                    provider_params=provider_params,
                    use_thinking=args.thinking,
                    thinking_tokens=args.thinking_tokens
                )
                logging.info("Translation completed")

                if args.translation_memory and not args.evaluate:
                    save_glossary(session_folder, updated_glossary)
                    logging.info("Updated glossary saved")

                if args.evaluate:
                    logging.info("Starting evaluation of translations")
                    final_blocks, final_glossary = evaluate_translation(
                        translation_blocks, 
                        translated_blocks, 
                        args.source_language, 
                        args.target_language, 
                        args.ant_api,
                        args.openai_api,
                        args.llmapi, 
                        updated_glossary, 
                        args.translation_memory,
                        model,
                        evaluation_prompt,
                        system_prompt,
                        provider_params=provider_params,
                        use_thinking=args.thinking,
                        thinking_tokens=args.thinking_tokens
                    )
                    logging.info("Evaluation completed")
                    evaluation_suffix = "_eval"
                    
                    if args.translation_memory:
                        save_glossary(session_folder, final_glossary)
                        logging.info("Final glossary saved after evaluation")
                else:
                    final_blocks = translated_blocks
                    evaluation_suffix = ""

            final_json_path = os.path.join(session_folder, f"{video_name}_{args.target_language}{evaluation_suffix}_final_blocks.json")
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_blocks, f, ensure_ascii=False, indent=2)
            logging.info(f"Final translation blocks saved as JSON: {final_json_path}")

            if args.llmapi == 'deepl':
                final_srt = parse_deepl_response(final_blocks, srt_content)
            else:
                final_srt = parse_translated_response(final_blocks, srt_content)

            translated_srt_path = os.path.join(session_folder, f"{video_name}_{args.target_language}{evaluation_suffix}.srt")
            with open(translated_srt_path, 'w', encoding='utf-8') as f:
                f.write(final_srt)
            logging.info(f"Translated subtitles saved: {translated_srt_path}")

            if args.task in ['translate', 'translation']:
                logging.info("Translation (and evaluation if requested) completed. Ending process.")
                if args.equalize:
                    output_srt = os.path.join(session_folder, f"{video_name}_{args.target_language}{evaluation_suffix}_final.srt")
                    equalize_srt(translated_srt_path, output_srt, args.max_line_length)
                return

        if args.task in ['full', 'speech_blocks']:
            speech_blocks = create_speech_blocks(final_srt if 'final_srt' in locals() else srt_content, 
                                                 session_folder, video_name, 
                                                 args.target_language if args.task == 'full' else args.source_language, 
                                                 merge_threshold=args.merge_threshold)
            logging.info(f"Created {len(speech_blocks)} speech blocks")

            if args.task == 'speech_blocks':
                logging.info("Speech blocks creation completed. Ending process.")
                return

        if args.task == 'full':
            if not args.tts_voice:
                tts_voices_folder = "tts-voices"
                if os.path.exists(tts_voices_folder):
                    wav_files = [f for f in os.listdir(tts_voices_folder) if f.endswith('.wav')]
                    if wav_files:
                        args.tts_voice = os.path.join(tts_voices_folder, wav_files[0])
                        logging.info(f"Using TTS voice: {args.tts_voice}")
                    else:
                        raise ValueError("No WAV files found in the tts-voices folder")
                else:
                    raise ValueError("No TTS voice specified and tts-voices folder not found")

            try:
                tts_language = get_xtts_language_code(args.target_language)
            except ValueError as e:
                logging.error(str(e))
                return

            audio_files = generate_tts_audio(speech_blocks, args.tts_voice, tts_language, session_folder, video_name)
            logging.info(f"Generated {len(audio_files)} TTS audio files")

            if not audio_files:
                logging.error("No TTS audio files were generated. Cannot proceed with alignment and mixing.")
                return

            alignment_blocks = create_alignment_blocks(session_folder, video_name, args.target_language, args.evaluate)
            logging.info(f"Created {len(alignment_blocks)} alignment blocks")

            aligned_audio_path = align_audio_blocks(alignment_blocks, session_folder)
            logging.info(f"Audio alignment completed: {aligned_audio_path}")

            if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                final_output_path = mix_audio_tracks(video_path, aligned_audio_path, session_folder, video_name, args.target_language, args.evaluate)
                logging.info(f"Final output saved: {final_output_path}")
            else:
                logging.info(f"Aligned audio saved: {aligned_audio_path}")

            if args.equalize:
                input_srt = translated_srt_path
                output_srt = os.path.join(session_folder, f"{video_name}_{args.target_language}{evaluation_suffix}_final.srt")
                equalize_srt(input_srt, output_srt, args.max_line_length)

            cleanup_temp_files(session_folder, args.task)
            logging.info("Temporary files cleaned up")

        logging.info("Process completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
