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
from nervon.pipeline.prompts import build_fact_extraction_messages

logger = logging.getLogger(__name__)


def extract_facts(messages: list[dict], llm_model: str) -> list[str]:
    try:
        response = llm_completion_with_retry(
            model=llm_model,
            messages=build_fact_extraction_messages(messages),
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # pragma: no cover - exercised via tests with mocks
        logger.warning("Fact extraction request failed: %s", exc)
        return []

    payload = extract_json_object(extract_message_content(response))
    if payload is None:
        logger.warning("Fact extraction returned malformed JSON")
        return []

    facts = payload.get("facts", [])
    if not isinstance(facts, list):
        logger.warning("Fact extraction JSON did not contain a facts list")
        return []

    normalized: list[str] = []
    for fact in facts:
        if not isinstance(fact, str):
            continue
        cleaned = fact.strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized
