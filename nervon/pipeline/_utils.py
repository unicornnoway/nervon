from __future__ import annotations

import json
import logging
import os
import time
import threading
from typing import Any

logger = logging.getLogger(__name__)


class GeminiLLMKeyRotator:
    """Round-robin key rotation for Gemini LLM calls via litellm.
    
    Mirrors the embedding key rotation pattern in benchmark_locomo.py.
    On 429 rate limit, automatically switches to next key.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._keys: list[str] = []
        self._index = 0
        self._call_count = 0
        self._load_keys()

    def _load_keys(self):
        pool_path = os.path.expanduser("~/.openclaw/secrets/gemini-search-pool.json")
        try:
            with open(pool_path) as f:
                data = json.load(f)
            self._keys = data.get("keys", [])
            if self._keys:
                logger.info("Loaded %d Gemini API keys for LLM rotation", len(self._keys))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Could not load Gemini key pool: %s", e)

    @property
    def available(self) -> bool:
        return len(self._keys) > 0

    def get_key(self) -> str:
        """Get current key, rotating every 50 calls."""
        self._call_count += 1
        if self._call_count % 50 == 0:
            self._index = (self._index + 1) % len(self._keys)
        return self._keys[self._index]

    def rotate(self):
        """Force rotate to next key (e.g., on 429)."""
        self._index = (self._index + 1) % len(self._keys)
        logger.info("Rotated to Gemini key index %d", self._index)


# Singleton instance
_gemini_llm_rotator = GeminiLLMKeyRotator()


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
    """Call litellm.completion with exponential backoff on rate limits.
    
    For Gemini models, automatically applies key rotation from the pool.
    On 429/rate limit, rotates to the next key before retrying.
    """
    try:
        import litellm
    except ModuleNotFoundError:
        raise ModuleNotFoundError("litellm is required")

    is_gemini = model.startswith("gemini/") or model.startswith("google/")
    rotator = _gemini_llm_rotator if is_gemini and _gemini_llm_rotator.available else None

    for attempt in range(max_retries):
        call_kwargs = dict(**kwargs)
        if rotator:
            call_kwargs["api_key"] = rotator.get_key()
        try:
            return litellm.completion(model=model, messages=messages, **call_kwargs)
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate" in err_str.lower() or "resource_exhausted" in err_str.lower():
                if rotator:
                    rotator.rotate()
                wait = min(2 ** attempt * 3, 60)
                logger.info("LLM rate limited, waiting %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            raise  # Non-rate-limit errors bubble up immediately
    # Final attempt without catching
    call_kwargs = dict(**kwargs)
    if rotator:
        call_kwargs["api_key"] = rotator.get_key()
    return litellm.completion(model=model, messages=messages, **call_kwargs)


def extract_message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)
