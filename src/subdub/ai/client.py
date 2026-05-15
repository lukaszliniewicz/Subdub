import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import litellm
from litellm import completion

from ..schemas.llm import ResegmentSubtitle, SubtitleList

logger = logging.getLogger(__name__)
MAX_RETRIES = 3
_callbacks_configured = False


def litellm_success_callback(kwargs, completion_response, start_time, end_time):
    duration = (end_time - start_time).total_seconds() if start_time and end_time else 0
    cost = kwargs.get("response_cost", 0.0)

    logger.info("\n=== LiteLLM SUCCESS ===")
    logger.info(f"Model: {kwargs.get('model')}")

    usage = completion_response.get("usage", {}) if isinstance(completion_response, dict) else getattr(completion_response, "usage", None)
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else getattr(usage, "prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else getattr(usage, "completion_tokens", 0)
        logger.info(f"Usage: {prompt_tokens} in / {completion_tokens} out")

    logger.info(f"Duration: {duration:.2f}s | Cost: ${cost:.6f}")
    logger.info("======================\n")

    if logger.isEnabledFor(logging.DEBUG):
        try:
            if hasattr(completion_response, "model_dump_json"):
                logger.debug(f"Full Response:\n{completion_response.model_dump_json(indent=2)}")
            else:
                logger.debug(f"Full Response:\n{json.dumps(completion_response, indent=2, default=str)}")
        except Exception:
            logger.debug(f"Full Response:\n{completion_response}")


def litellm_failure_callback(kwargs, completion_response, start_time, end_time, e):
    logger.error("\n=== LiteLLM FAILURE ===")
    logger.error(f"Model: {kwargs.get('model')}")
    logger.error(f"Error: {str(e)}")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Request (messages): {json.dumps(kwargs.get('messages', []), indent=2, default=str)}")
    logger.error("======================\n")


def litellm_input_callback(kwargs, *args, **kwargs_extra):
    logger.info("\n--- LiteLLM INPUT ---")
    logger.info(f"Model: {kwargs.get('model')}")
    messages = kwargs.get("messages", [])

    if messages:
        last_msg = messages[-1].get("content", "")
        if isinstance(last_msg, str):
            logger.info(f"Prompt Preview: {last_msg[:500]}..." if len(last_msg) > 500 else f"Prompt: {last_msg}")
        else:
            logger.info("Prompt: [Multi-modal content]")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Full Messages: {json.dumps(messages, indent=2, default=str)}")
    logger.info("---------------------\n")


def configure_litellm_callbacks(force: bool = False) -> None:
    global _callbacks_configured
    if _callbacks_configured and not force:
        return

    litellm.drop_params = True
    litellm.modify_params = True
    litellm.success_callback = [litellm_success_callback]
    litellm.failure_callback = [litellm_failure_callback]
    litellm.input_callback = [litellm_input_callback]
    _callbacks_configured = True


def calculate_cost(response, model: Optional[str] = None) -> float:
    try:
        return float(response._hidden_params.get("response_cost", 0.0))
    except Exception as e:
        logger.debug(f"Could not get cost from LiteLLM response: {str(e)}")
        return 0.0


def llm_api_request(
    model: str,
    messages: List[Dict[str, Any]],
    system_prompt: str = "",
    provider_params: Optional[Dict[str, Any]] = None,
    output_schema=None,
    **kwargs,
) -> Tuple[str, object]:
    try:
        configure_litellm_callbacks()
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})

        processed_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if isinstance(msg.get("content"), list):
                new_content = []
                for item in msg["content"]:
                    if hasattr(item, "uri"):
                        new_content.append({"type": "image_url", "image_url": {"url": item.uri}})
                    elif isinstance(item, str):
                        new_content.append({"type": "text", "text": item})
                new_msg["content"] = new_content
            processed_messages.append(new_msg)

        final_messages.extend(processed_messages)

        api_kwargs = provider_params.copy() if provider_params else {}
        temperature = api_kwargs.pop("temperature", None)

        reasoning_config = {}
        if "reasoning_effort" in api_kwargs:
            reasoning_config["reasoning_effort"] = api_kwargs["reasoning_effort"]
            logger.info(f"Reasoning Effort: {reasoning_config['reasoning_effort']}")

        response_format = None
        if output_schema:
            response_format = SubtitleList if output_schema == list[ResegmentSubtitle] else output_schema
            logger.info("Using Structured Output Schema")

        if "reasoning_effort" in api_kwargs:
            del api_kwargs["reasoning_effort"]

        api_kwargs.setdefault("num_retries", 2)
        if temperature is not None:
            api_kwargs["temperature"] = temperature

        response = completion(
            model=model,
            messages=final_messages,
            response_format=response_format,
            **reasoning_config,
            **api_kwargs,
        )

        content = response.choices[0].message.content

        if output_schema == list[ResegmentSubtitle]:
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "subtitles" in data:
                    content = json.dumps(data["subtitles"])
            except Exception:
                pass

        return content, response
    except Exception as e:
        logger.error(f"Error in LiteLLM request: {str(e)}")
        raise
