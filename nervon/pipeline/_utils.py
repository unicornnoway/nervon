from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def format_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role", "unknown")).strip() or "unknown"
        content = str(message.get("content", "")).strip()
        lines.append(f"{index}. {role}: {content}")
    return "\n".join(lines)


def extract_json_object(raw_content: str) -> dict[str, Any] | None:
    text = raw_content.strip()
    if not text:
        return None

    candidates = [text]
    if "```" in text:
        stripped = text.replace("```json", "```")
        parts = [part.strip() for part in stripped.split("```") if part.strip()]
        candidates.extend(parts)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    # Fallback: try json-repair for malformed output (common with small models)
    try:
        from json_repair import repair_json  # type: ignore[import-untyped]
        for candidate in candidates:
            try:
                repaired = repair_json(candidate, return_objects=True)
                if isinstance(repaired, dict):
                    logger.info("JSON recovered via json-repair")
                    return repaired
            except Exception:
                continue
    except ImportError:
        pass  # json-repair not installed, skip fallback

    return None


def llm_completion_with_retry(model: str, messages: list, max_retries: int = 5, **kwargs) -> Any:
    """Call litellm.completion with exponential backoff on rate limits."""
    try:
        import litellm
    except ModuleNotFoundError:
        raise ModuleNotFoundError("litellm is required")

    for attempt in range(max_retries):
        try:
            return litellm.completion(model=model, messages=messages, **kwargs)
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate" in err_str.lower() or "resource_exhausted" in err_str.lower():
                wait = min(2 ** attempt * 3, 60)
                logger.info("LLM rate limited, waiting %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            raise  # Non-rate-limit errors bubble up immediately
    # Final attempt without catching
    return litellm.completion(model=model, messages=messages, **kwargs)


def extract_message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)
