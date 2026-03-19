from __future__ import annotations

import logging

try:
    import litellm
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    class _LiteLLMStub:
        @staticmethod
        def completion(*args, **kwargs):
            raise ModuleNotFoundError("litellm is required for completion requests")

    litellm = _LiteLLMStub()

from neurai.pipeline._utils import extract_json_object, extract_message_content
from neurai.pipeline.prompts import build_episode_summary_messages

logger = logging.getLogger(__name__)


def summarize_conversation(messages: list[dict], llm_model: str) -> dict[str, str | list[str]]:
    try:
        response = litellm.completion(
            model=llm_model,
            messages=build_episode_summary_messages(messages),
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # pragma: no cover - exercised via tests with mocks
        logger.warning("Episode summary request failed: %s", exc)
        return {"summary": "", "key_topics": []}

    payload = extract_json_object(extract_message_content(response))
    if payload is None:
        logger.warning("Episode summary returned malformed JSON")
        return {"summary": "", "key_topics": []}

    summary = payload.get("summary", "")
    topics = payload.get("key_topics", [])

    normalized_topics: list[str] = []
    if isinstance(topics, list):
        for topic in topics:
            if not isinstance(topic, str):
                continue
            cleaned = topic.strip()
            if cleaned:
                normalized_topics.append(cleaned)

    return {
        "summary": summary.strip() if isinstance(summary, str) else "",
        "key_topics": normalized_topics,
    }
