# Subdub: Advanced Subtitle Translation and Dubbing Tool

Subdub is a command-line tool for translating video subtitles and generating dubbed audio. It was created to enhance the dubbing functionality of [Pandrator](https://github.com/lukaszliniewicz/pandrator), but can be used on its own, albeit with limited functionality. Pandrator provides a GUI that makes it possible to preview, edit, and regenerate subtitle audio before aligning and synchronising it, as well as manage the entire workflow.

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
`python Subdub.py -i [input_file] -sl [source_language] -tl [target_language] -task [task_name] -tts_voice [path to 6-12s wav 22050hz mono .wav file]`

If you want to perform translation, you need to have an Anthropic, OpenAI or DeepL API key and provide it as an argument. Using the local LLM API (Text Generation WebUI's) doesn't require an api key. You can also set the keys as environmental values (OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPL_API_KEY).  

## Tasks

Subdub offers several task modes to suit different needs:

### 1. full
- **Description**: Performs all steps: transcription, translation, TTS generation, and audio synchronization.
- **Input**: Video file
- **Output**: Translated subtitles, TTS audio, and final dubbed video
- **Usage**: `python Subdub.py -i video.mp4 -sl English -tl Spanish -task full`

### 2. transcribe
- **Description**: Transcribes the audio from a video file.
- **Input**: Video file
- **Output**: SRT subtitle file in the source language
- **Usage**: `python Subdub.py -i video.mp4 -sl English -task transcribe`. You can also specify the whisper model. 

### 3. translate
- **Description**: Translates existing subtitles to the target language.
- **Input**: SRT subtitle file
- **Output**: Translated SRT subtitle file
- **Usage**: `python Subdub.py -i subtitles.srt -sl English -tl French -task translate`. You can also specify the translation api as well as the model you want to use (for Anthropic and OpenAI). 

### 4. tts
- **Description**: Generates Text-to-Speech audio from existing subtitles.
- **Input**: SRT subtitle file
- **Output**: WAV audio files for each subtitle block
- **Usage**: `python Subdub.py -i subtitles.srt -tl Spanish -task tts -tts_voice voice.wav`

### 5. speech_blocks
- **Description**: Creates speech blocks JSON from subtitles for advanced audio processing.
- **Input**: SRT subtitle file
- **Output**: JSON file containing speech blocks
- **Usage**: `python Subdub.py -i subtitles.srt -task speech_blocks`

### 6. sync
- **Description**: Synchronizes existing TTS audio with the original video.
- **Input**: Video file, speech blocks JSON, and TTS audio files
- **Output**: Final dubbed video
- **Usage**: `python Subdub.py -i video.mp4 -task sync -session existing_session`

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-i`, `--input` | Input video or subtitle file path (required) | - |
| `-sl`, `--source_language` | Source language | English |
| `-tl`, `--target_language` | Target language for translation | - |
| `-task` | Task to perform | full |
| `-session` | Session name or path | - |
| `-llm-char` | Character limit for translation | 4000 |
| `-ant_api` | Anthropic API key | - |
| `-evaluate` | Perform evaluation of translations | False |
| `-translation_memory` | Enable translation memory/glossary feature | False |
| `-tts_voice` | Path to TTS voice WAV file | - |
| `-whisper_model` | Whisper model for transcription | large-v2 |
| `-llm-model` | LLM model for translation (haiku/sonnet) | sonnet |
| `-merge_threshold` | Max time (ms) between subtitles to merge | 1 |

### Evaluation and Glossary

- **Evaluate (-evaluate)**: When enabled, this feature performs an additional pass on the translated subtitles. It uses the AI model to review and improve the initial translations, ensuring better quality and consistency.

- **Translation Memory/Glossary (-translation_memory)**: This feature maintains a glossary of terms and their translations. It helps ensure consistency across the translation, especially for domain-specific terms or recurring phrases. The glossary is updated throughout the translation process and can be reused in future sessions.

## Workflow

Subdub follows a logical workflow to process videos and generate dubbed audio:

1. **SRT Generation**: If the input is a video file, Subdub uses WhisperX to transcribe the audio and generate an SRT file.

2. **Translation Blocks**: The SRT file is divided into translation blocks, considering character limits and sentence structures.

3. **Translation**: Each block is translated using the Anthropic API. If translation memory is enabled, a glossary is used and updated during this process.

4. **Translated SRT**: The translated blocks are reassembled into a new SRT file in the target language.

5. **Speech Blocks**: The translated SRT is processed to create speech blocks, which are optimized segments for Text-to-Speech generation.

6. **TTS Generation**: Audio is generated for each speech block using the XTTS API Server.

7. **Alignment Blocks**: Speech blocks are aligned with the original video timing.

8. **Synchronization and Mixing**: The generated audio is synchronized with the video. During this process, the original audio volume is lowered when dubbed audio is playing, creating a professional-sounding output.

## Examples

1. Full process with translation memory:
python Subdub.py -i video.mp4 -sl English -tl Spanish -task full -translation_memory

2. Translate existing subtitles and evaluate the translation:
python Subdub.py -i subtitles.srt -sl English -tl French -task translate -evaluate

3. Generate TTS for translated subtitles with a custom voice:
python Subdub.py -i translated.srt -tl German -task tts -tts_voice custom_voice.wav

4. Create speech blocks with custom merge threshold:
python Subdub.py -i subtitles.srt -task speech_blocks -merge_threshold 500

## Dependencies

- FFmpeg
- WhisperX
- Anthropic API (for translation)
- XTTS API Server (for Text-to-Speech)

Ensure all dependencies are installed and properly configured before running Subdub.
