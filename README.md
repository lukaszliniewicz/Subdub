# Subdub: Advanced Subtitle Translation and Dubbing Tool

Subdub is a command-line tool for creating subtitles from video, translating them, generating dubbed audio and syncing dubbed audio with the original video. It was created to enhance the dubbing functionality of [Pandrator](https://github.com/lukaszliniewicz/pandrator), but can be used on its own, albeit with limited functionality. Pandrator provides a GUI that makes it possible to preview, edit, and regenerate subtitle audio before aligning and synchronising it, as well as manage the entire workflow.

Dubbing sample, including translation from Russian ([video source](https://www.youtube.com/watch?v=_SwUpU0E2Eg&t=61s&pp=ygUn0LLRi9GB0YLRg9C_0LvQtdC90LjQtSDQu9C10LPQsNGB0L7QstCw)):

https://github.com/user-attachments/assets/1ba8068d-986e-4dec-a162-3b7cc49052f4

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Tasks](#tasks)
4. [Arguments](#arguments)
5. [Workflow](#workflow)
6. [Examples](#examples)
7. [Dependencies](#dependencies)

## Installation

1. Clone the repository:
`git clone https://github.com/lukaszliniewicz/Subdub.git`
2. Move to the Subdub directory:
`cd Subdub`
2. Install requirements:
`pip install -r requirements.txt`

3. Make sure [WhisperX](https://github.com/m-bain/whisperX) is available on your system.

4. Ensure the [XTTS API Server](https://daswer123/xtts-api-server) is running.

## Usage

Basic usage:
`python Subdub.py -i [input_file] -sl [source_language] -tl [target_language] -task [task_name] -tts_voice [path to a 6-12s 22050hz mono .wav file]`

If you want to perform translation, you need to have an Anthropic, OpenAI or DeepL API key and provide it as an argument. Using the local LLM API (Text Generation WebUI's) doesn't require an api key. You can also set the keys as environmental variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPL_API_KEY).  

## Tasks

Subdub offers several task modes:

### 1. full
- **Description**: Performs the complete pipeline: transcription, translation, TTS generation, and audio synchronization.
- **Input**: Video file or URL
- **Output**: Translated subtitles, TTS audio, and final dubbed video with mixed audio
- **Usage**: `python Subdub.py -i video.mp4 -sl English -tl Spanish -task full`

### 2. transcribe
- **Description**: Transcribes the audio from a video file using WhisperX.
- **Input**: Video file or URL
- **Output**: SRT subtitle file in the source language
- **Usage**: `python Subdub.py -i video.mp4 -sl English -task transcribe -whisper_model large-v3`

### 3. translate
- **Description**: Translates existing subtitles to the target language using selected AI model.
- **Input**: SRT subtitle file
- **Output**: Translated SRT subtitle file
- **Usage**: `python Subdub.py -i subtitles.srt -sl English -tl French -task translate -llmapi anthropic -llm-model sonnet`

### 4. tts
- **Description**: Generates Text-to-Speech audio from existing subtitles using XTTS.
- **Input**: SRT subtitle file
- **Output**: WAV audio files for each speech block
- **Usage**: `python Subdub.py -i subtitles.srt -tl Spanish -task tts -tts_voice voice.wav`

### 5. speech_blocks
- **Description**: Creates optimized speech blocks from subtitles for advanced TTS processing.
- **Input**: SRT subtitle file
- **Output**: JSON file containing speech blocks with optimal segmentation
- **Usage**: `python Subdub.py -i subtitles.srt -task speech_blocks -merge_threshold 1`

### 6. sync
- **Description**: Synchronizes existing TTS audio with the original video without regenerating audio.
- **Input**: Video file, speech blocks JSON, and TTS audio files
- **Output**: Final dubbed video with mixed audio
- **Usage**: `python Subdub.py -task sync -session existing_session -video original.mp4`

### 7. equalize
- **Description**: Reformats subtitle text for better readability, adjusting line lengths and breaks.
- **Input**: SRT subtitle file
- **Output**: Equalized SRT file with optimal line lengths
- **Usage**: `python Subdub.py -i subtitles.srt -task equalize -characters 42`

### 8. correct
- **Description**: Corrects and improves subtitle text without translation (fixing punctuation, capitalization, etc.)
- **Input**: SRT subtitle file
- **Output**: Corrected SRT subtitle file in the same language
- **Usage**: `python Subdub.py -i subtitles.srt -sl English -task correct -correct_prompt "Fix colloquialisms"`

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-i`, `--input` | Input video, subtitle file, or URL | - |
| `-sl`, `--source_language` | Source language | English |
| `-tl`, `--target_language` | Target language for translation | - |
| `-task` | Task to perform | full |
| `-session` | Session name or path | auto-generated |
| `-llm-char` | Character limit for translation blocks | 4000 |
| `-ant_api` | Anthropic API key | env or prompt |
| `-openai_api` | OpenAI API key | env or prompt |
| `-gemini_api` | Google Gemini API key | env or prompt |
| `-api_deepl` | DeepL API key | env or prompt |
| `-translation_memory` | Enable translation memory/glossary | False |
| `-tts_voice` | Path to TTS voice WAV file | - |
| `-whisper_model` | Whisper model for transcription | large-v2 |
| `-llmapi` | LLM API to use (anthropic, openai, local, deepl, openrouter, gemini) | anthropic |
| `-llm-model` | LLM model (sonnet, haiku, gpt-4o, gpt-4o-mini, etc.) | sonnet |
| `-merge_threshold` | Max time (ms) between subtitles to merge | 1 |
| `-equalize` | Apply SRT equalizer to subtitles | False |
| `-characters` | Max line length for equalization | 60 |
| `-correct` | Enable subtitle correction | False |
| `-correct_prompt` | Additional instructions for correction | - |
| `-translate_prompt` | Additional instructions for translation | - |
| `-cot` | Enable chain-of-thought prompting | False |
| `-context` | Add previous output as context | False |
| `-thinking` | Enable Claude's extended thinking (Sonnet only) | False |
| `-thinking_tokens` | Budget tokens for extended thinking | 4000 |
| `-video` | Input video for sync task | auto-detect |
| `-max_line_length` | Maximum line length for SRT equalization | 42 |

### OpenRouter Parameters

When using `-llmapi openrouter`, these additional parameters are available:

| Argument | Description | Default |
|----------|-------------|---------|
| `-provider` | Providers to prioritize (comma-separated, e.g., 'Anthropic,OpenAI') | - |
| `-sort` | Provider sorting strategy (price, throughput, latency) | - |
| `-fallbacks` | Allow fallbacks to other providers | True |
| `-no-fallbacks` | Disable fallbacks to other providers | - |
| `-ignore` | Providers to ignore (comma-separated list) | - |
| `-data-collection` | Data collection policy (allow, deny) | allow |
| `-require-parameters` | Require providers to support all parameters | False |

### AI Model Selection

Subdub supports multiple AI providers and models:

- **Anthropic Claude**: 
  - `sonnet` (claude-3-7-sonnet-latest, recommended for quality)
  - `haiku` (claude-3-5-haiku-latest, faster)

- **OpenAI**: 
  - `gpt-4o` (highest quality but slower)
  - `gpt-4o-mini` (faster, good quality)

- **Google Gemini**:
  - `gemini-flash` (standard)
  - `gemini-flash-thinking` (with thinking)

- **OpenRouter**:
  - `deepseek-r1` 
  - `qwq-32b`
  - Can add `:nitro` suffix for throughput or `:floor` for lowest cost

- **DeepL**: For pure translation without customization

- **Local**: Uses Text Generation WebUI API

### Glossary

- **Translation Memory/Glossary (-translation_memory)**: This feature maintains a glossary of terms and their translations. It helps ensure consistency across the translation, especially for domain-specific terms or recurring phrases. The glossary is updated throughout the translation process and can be reused in future sessions.

## Workflow

Subdub follows a logical workflow to process videos and generate dubbed audio:

1. **SRT Generation**: If the input is a video file, Subdub uses WhisperX to transcribe the audio and generate an SRT file.

2. **Translation Blocks**: The SRT file is divided into translation blocks, considering character limits and sentence structures.

3. **Translation**: Each block is translated using an API. If translation memory is enabled, a glossary is used and updated during this process.

4. **Translated SRT**: The translated blocks are reassembled into a new SRT file in the target language.

5. **Speech Blocks**: The translated SRT is processed to create speech blocks, which are optimized segments for Text-to-Speech generation.

6. **TTS Generation**: Audio is generated for each speech block using the XTTS API Server.

7. **Alignment Blocks**: Speech blocks are aligned with the original video timing.

8. **Synchronization and Mixing**: The generated audio is synchronized with the video. During this process, the original audio volume is lowered when dubbed audio is playing.

## Examples

1. Full process with translation memory:
python Subdub.py -i video.mp4 -sl English -tl Spanish -task full -translation_memory

2. Translate existing subtitles and evaluate the translation:
python Subdub.py -i subtitles.srt -sl English -tl French -task translate -evaluate

3. Generate TTS for translated subtitles with a custom voice:
python Subdub.py -i translated.srt -tl German -task tts -tts_voice custom_voice.wav

## Dependencies

- FFmpeg
- WhisperX
- Anthropic, OpenAI, DeepL or Text Generation WebUI API (for translation)
- XTTS API Server (for Text-to-Speech)

Ensure all dependencies are installed and properly configured before running Subdub.
