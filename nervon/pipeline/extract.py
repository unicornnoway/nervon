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
from nervon.pipeline.schemas import FactExtractionResponse

logger = logging.getLogger(__name__)


def extract_facts(messages: list[dict], llm_model: str, reference_time: str | None = None) -> list[str]:
    try:
        response = llm_completion_with_retry(
            model=llm_model,
            messages=build_fact_extraction_messages(messages, reference_time=reference_time),
            response_format=FactExtractionResponse,
        )
    except Exception as exc:  # pragma: no cover - exercised via tests with mocks
        logger.warning("Fact extraction request failed: %s", exc)
        return []

    raw_content = extract_message_content(response)

    # Try Pydantic parsing first (works when provider returns structured output)
    parsed = _parse_response(raw_content)
    if parsed is not None:
        return _normalize_facts(parsed.facts)

    # Fallback: legacy JSON extraction + json-repair
    payload = extract_json_object(raw_content)
    if payload is None:
        logger.warning("Fact extraction returned malformed JSON")
        return []

    facts = payload.get("facts", [])
    if not isinstance(facts, list):
        logger.warning("Fact extraction JSON did not contain a facts list")
        return []

    return _normalize_facts(facts)


def _parse_response(content: str) -> FactExtractionResponse | None:
    """Attempt to parse content as a FactExtractionResponse."""
    try:
        return FactExtractionResponse.model_validate_json(content)
    except Exception:
        return None


def _normalize_facts(facts: list) -> list[str]:
    normalized: list[str] = []
    for fact in facts:
        if not isinstance(fact, str):
            continue
        cleaned = fact.strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized
