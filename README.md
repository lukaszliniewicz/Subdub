# Subdub: Subtitle Translation and Dubbing CLI

Subdub is a command-line tool for:

- transcribing media with WhisperX,
- correcting/translating subtitles with LLMs through LiteLLM (or DeepL),
- generating dubbed speech with XTTS,
- and syncing dubbed audio back into video.

It was created to support the dubbing workflow in [Pandrator](https://github.com/lukaszliniewicz/pandrator), but it also works standalone.

Dubbing sample (Russian -> translated dub):

https://github.com/user-attachments/assets/1ba8068d-986e-4dec-a162-3b7cc49052f4

## Current State After Refactor

The app has been refactored from a monolithic file into modular pipeline stages.

- `subdub.cli` + `subdub.cli_args`: entrypoint and argument parsing.
- `subdub.app`: runtime orchestration and stage ordering.
- `subdub.tasks.*`: stage modules (`input`, `correction`, `translation`, `speech`, `transcribe`).
- `subdub.ai.client`: centralized LiteLLM request layer, callbacks, structured output handling, cost tracking.
- `subdub.media.*` and `subdub.workflows.*`: FFmpeg, TTS, sync, boundary-correction workflows.
- Compatibility re-exports remain in `subdub.ai.translate` and `subdub.models`.

## Installation

### Prerequisites

- Python 3.10+
- FFmpeg available in `PATH`
- WhisperX available either:
  - as a direct `whisperx` command, or
  - via conda fallback (`CONDA_EXE` and `WHISPERX_CONDA_ENV`)
- XTTS API server running at `http://localhost:8020` (for dubbing/TTS)

Required external services depend on your model/task:

- LLM provider key(s): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY`
- `DEEPL_API_KEY` when using `--use-deepl`
- `HF_TOKEN` when using `-diarize`

Environment behavior notes:

- `OPENROUTER_API` is accepted as a legacy alias and copied to `OPENROUTER_API_KEY` if needed.
- For localhost-style `-api_base` endpoints, Subdub auto-fills `OPENAI_API_KEY=lm-studio` when no key is set.

### Setup

```bash
git clone https://github.com/lukaszliniewicz/Subdub.git
cd Subdub
pip install -e .
```

Optional extras:

- `pip install -e .[dev]` for tests/lint/type checks
- `pip install -e .[gui]` for manual correction GUI features

## Quick Start

```bash
# Full pipeline: transcribe -> translate -> speech blocks -> TTS -> sync/mix
subdub -task full -i video.mp4 -sl English -tl Spanish -tts_voice voice.wav

# Transcription only
subdub -task transcribe -i video.mp4 -sl English

# Translate existing subtitles
subdub -task translate -i subtitles.srt -sl English -tl French

# Correct subtitles in-place language (no translation)
subdub -task correct -i subtitles.srt -sl English

# Zoom transcript correction (.vtt input)
subdub -task zoom-transcript -i meeting.vtt
```

Alternative invocation:

```bash
python -m subdub -task full -i video.mp4 -sl English -tl Spanish -tts_voice voice.wav
```

## Task Modes

| Task | Purpose | Typical Input | Main Output |
|---|---|---|---|
| `full` | End-to-end pipeline (STT -> translation -> TTS -> sync/mix) | media file / URL / subtitle file | translated SRT, speech blocks, aligned audio, final dubbed media |
| `transcribe` | WhisperX transcription (+ optional diarization/correction) | media file / URL | source SRT or JSON-derived corrected SRT |
| `translate` | Translate subtitles (LLM or DeepL) | media / `.srt` / WhisperX `.json` | translated SRT + block JSON |
| `correct` | Correct subtitle text in source language | media / `.srt` / WhisperX `.json` | corrected SRT |
| `speech_blocks` | Build speech segmentation JSON for dubbing | media / `.srt` / translated SRT | `*_speech_blocks.json` |
| `sync` | Align existing generated speech with video | existing `-session` (+ optional `-v`) | final mixed dubbed video/audio |
| `equalize` | Standalone subtitle line equalization | `.srt` | `_equalized.srt` |
| `zoom-transcript` | Correct grouped Zoom VTT transcript chunks | `.vtt` | corrected transcript `.txt` |
| `tts` | Legacy parser option (see notes below) | n/a | n/a |

## CLI Reference

Run `subdub -h` for the full list. Most relevant flags are below.

### Core

- `-i`, `--input`: input path or URL (required for most tasks)
- `-task`: `tts|full|transcribe|translate|speech_blocks|sync|equalize|correct|zoom-transcript` (default: `full`)
- `-session`: custom session folder (otherwise auto-generated)
- `-log`: also write detailed logs to `subtitle_app.log`
- `-sl`, `--source_language`: source language (default: `English`)
- `-tl`, `--target_language`: target language for translation/dubbing

### LLM / Provider / Translation

- `-model`: LiteLLM model string (default: `anthropic/claude-3-5-sonnet-20241022`)
- `-ant_api`, `-openai_api`, `-gemini_api`, `-api_deepl`: API keys
- `--use-deepl`: use DeepL instead of LLM translation
- `-api_base`: custom/local OpenAI-compatible endpoint base URL
- `-llm-char`: max characters per translation/correction block (default: `4000`)
- `-max_tokens`: max output tokens for provider call
- `-reasoning_effort`: `minimal|low|medium|high`
- `-evaluate`: second-pass evaluation/improvement stage
- `-translation_memory`: glossary memory file (`translation_glossary.json`)
- `-context`: pass prior response context between blocks
- `--no-remove-subtitles`: prohibit LLM `[REMOVE]` behavior
- `-translate_prompt`: extra translation instructions appended to prompt

OpenRouter-specific:

- `-provider`: prioritized provider(s), comma-separated
- `-sort`: `price|throughput|latency`
- `-fallbacks` / `-no-fallbacks`: enable/disable provider fallback
- `-ignore`: provider(s) to ignore
- `-data-collection`: `allow|deny`
- `-require-parameters`: require provider parameter support

### Transcription / Correction

- `-whisper_model`: Whisper model (default: `large-v3`)
- `-align_model`: custom WhisperX alignment model
- `-whisper_prompt`: custom initial prompt for WhisperX
- `-chunk_size`: WhisperX chunk size (default: `15`)
- `-diarize`: enable speaker diarization
- `--hf_token`: Hugging Face token for diarization
- `--no-boundary-correction`: disable automatic boundary correction
- `-manual_correction`: open manual correction GUI (PyQt6 extra required)
- `--save_txt`: save transcript as `.txt` during WhisperX run
- `-correct`: run correction stage before translation
- `-correct_prompt`: additional correction instructions
- `-resegment`: use word-level re-segmentation pipeline

### Dubbing / Sync / Subtitle Formatting

- `-tts_voice`: 6-12s voice `.wav` for XTTS (for `full`; falls back to first `.wav` in `tts-voices/` if omitted)
- `-merge_threshold`: subtitle merge threshold in ms (default: `250`)
- `--delay_start`: initial per-block delay cap in ms (default: `2000`)
- `--speed_up`: max speed-up percentage during alignment (default: `115`)
- `-v`, `--video`: optional input video override for `sync`
- `-equalize`: also equalize final SRT in task output
- `-max_line_length`: line length for `-equalize` (default: `42`)
- `-characters`: line length for standalone `-task equalize` (default: `60`)

### Prompt Overrides

- `-t_prompt`: full custom translation prompt template
- `-eval_prompt`: full custom evaluation prompt template
- `-gloss_prompt`: full custom glossary prompt template
- `-sys_prompt`: custom system prompt

## Session Outputs

Each run writes artifacts into the session folder (`-session` or auto-generated). Depending on task/settings, typical files include:

- extracted audio (`<video>.wav`)
- transcription output (`<video>.json` and/or corrected `.srt`)
- translated/corrected subtitle outputs (`*.srt`, `*_final_blocks.json`)
- speech segmentation (`*_speech_blocks.json`)
- per-block XTTS wavs (`Sentence_wavs/*.wav`)
- aligned dubbed audio (`aligned_audio.wav`)
- final mixed video (`final_output.mp4`) or dubbed audio only
- optional session log (`subtitle_app.log` when `-log` is enabled)

## LiteLLM Notes

- All LLM operations route through `subdub.ai.client.llm_api_request`.
- LiteLLM callbacks are configured once per process and log input/success/failure details.
- Structured output schemas are used for correction/re-segmentation flows.
- Session-level estimated API cost is accumulated and logged at the end.

## Known Behavior and Caveats

- `-task tts` is still present in parser choices for compatibility, but the current refactored pipeline does not execute a standalone TTS-only stage. Use `full` (or `speech_blocks` + existing audio workflow) instead.
- API key validation happens at startup based on selected `-model` and flags. With the default model, `ANTHROPIC_API_KEY` is expected even if your task does not call translation.
- `-resegment` is designed for media input or WhisperX JSON word timestamps; plain `.srt` input is not a good fit for that mode.

## Dependencies

Python package dependencies are defined in `pyproject.toml`.

External runtime tools/services:

- FFmpeg
- WhisperX
- XTTS API Server
- LLM provider endpoint(s) via LiteLLM (or DeepL when selected)
