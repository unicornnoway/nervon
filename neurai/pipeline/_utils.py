from __future__ import annotations

import json
import logging
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

    return None


def extract_message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)
