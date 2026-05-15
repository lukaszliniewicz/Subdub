import logging
import os

from .errors import ConfigurationError

logger = logging.getLogger(__name__)


def set_api_key(arg_val, env_name):
    if arg_val:
        os.environ[env_name] = arg_val
        return arg_val
    return os.environ.get(env_name)


def validate_api_keys(args, model: str):
    errors = []

    if "gemini" in model.lower():
        if not os.environ.get("GEMINI_API_KEY"):
            errors.append("Gemini model requires GEMINI_API_KEY (pass -gemini_api or set env var)")

    if "anthropic" in model.lower() or "claude" in model.lower():
        if not os.environ.get("ANTHROPIC_API_KEY"):
            errors.append("Claude model requires ANTHROPIC_API_KEY (pass -ant_api or set env var)")

    if "openai" in model.lower() or "gpt" in model.lower() or "o1" in model.lower():
        if not os.environ.get("OPENAI_API_KEY"):
            errors.append("OpenAI model requires OPENAI_API_KEY (pass -openai_api or set env var)")

    if args.use_deepl and not os.environ.get("DEEPL_API_KEY"):
        errors.append("DeepL translation requires DEEPL_API_KEY (pass -api_deepl or set env var)")

    if args.diarize and not os.environ.get("HF_TOKEN"):
        errors.append("Diarization requires HF_TOKEN environment variable")

    if errors:
        logger.error("\n" + "=" * 80)
        logger.error("API KEY CONFIGURATION ERROR")
        logger.error("=" * 80)
        for error in errors:
            logger.error(f"  - {error}")
        logger.error("\nQuick Setup Guide:")
        logger.error("  Windows (PowerShell): $env:GEMINI_API_KEY='your-key-here'")
        logger.error("  macOS/Linux (bash):   export GEMINI_API_KEY='your-key-here'")
        logger.error("  Or use CLI args:      -gemini_api your-key-here")
        logger.error("=" * 80 + "\n")
        raise ConfigurationError("API key validation failed")

    logger.info("All required API keys validated")


def get_xtts_language_code(target_language: str) -> str:
    xtts_language_map = {
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
        "Korean": "ko",
    }

    xtts_language_code = xtts_language_map.get(target_language)
    if not xtts_language_code:
        raise ConfigurationError(
            f"The target language '{target_language}' is not supported by XTTS. "
            f"Supported languages are: {', '.join(xtts_language_map.keys())}"
        )

    return xtts_language_code
