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

from nervon.pipeline._utils import extract_json_object, extract_message_content, llm_completion_with_retry
from nervon.pipeline.prompts import build_episode_summary_messages
from nervon.pipeline.schemas import EpisodeSummaryResponse

logger = logging.getLogger(__name__)


def summarize_conversation(messages: list[dict], llm_model: str, reference_time: str | None = None) -> dict[str, str | list[str]]:
    try:
        response = llm_completion_with_retry(
            model=llm_model,
            messages=build_episode_summary_messages(messages, reference_time=reference_time),
            response_format=EpisodeSummaryResponse,
        )
    except Exception as exc:  # pragma: no cover - exercised via tests with mocks
        logger.warning("Episode summary request failed: %s", exc)
        return {"summary": "", "key_topics": []}

    raw_content = extract_message_content(response)

    # Try Pydantic parsing first
    parsed = _parse_response(raw_content)
    if parsed is not None:
        return {
            "summary": parsed.summary.strip(),
            "key_topics": _normalize_topics(parsed.key_topics),
        }

    # Fallback: legacy JSON extraction + json-repair
    payload = extract_json_object(raw_content)
    if payload is None:
        logger.warning("Episode summary returned malformed JSON")
        return {"summary": "", "key_topics": []}

    summary = payload.get("summary", "")
    topics = payload.get("key_topics", [])

    normalized_topics = _normalize_topics(topics) if isinstance(topics, list) else []

    return {
        "summary": summary.strip() if isinstance(summary, str) else "",
        "key_topics": normalized_topics,
    }


def _parse_response(content: str) -> EpisodeSummaryResponse | None:
    """Attempt to parse content as an EpisodeSummaryResponse."""
    try:
        return EpisodeSummaryResponse.model_validate_json(content)
    except Exception:
        return None


def _normalize_topics(topics: list) -> list[str]:
    normalized: list[str] = []
    for topic in topics:
        if not isinstance(topic, str):
            continue
        cleaned = topic.strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized
